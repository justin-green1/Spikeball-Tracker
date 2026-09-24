package com.justin.spikeballtracker.vision;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class NetDetector {
    private static final String TAG = "NetDetector";
    private static final double WORK_HEIGHT = 480.0;    // Image pixel height

    // ---------------------------------------------------------------------
    // candidate parameters.
    private static final double MIN_Y_FRACTION = 0.63;
    private static final int    MIN_BOX_WIDTH  = 69;
    private static final int    MAX_BOX_WIDTH  = 176;
    private static final double lowerThreshold = 50.0;
    private static final double upperThreshold = 100.0;
    private static final double sigmaSpace     = 146.8;
    private static final double sigmaColour    = 184.7;
    private static final int    d              = 7;
    private static final int    MORPH_HEIGHT   = 9;
    private static final int    MORPH_WIDTH    = 19;
    // -----------------------------------------------------------------------
    // ---- selection constants ----
    private static final double MAX_CX_OFFSET   = 0.20;   // centre within middle 40%
    private static final double MIN_CY_FRACTION = 0.50;   // centre below halfway
    private static final double MIN_DARKNESS    = 0.0;    // net is never brighter than its ground
    //-----------------------------------------------------------------------
    // scorer parameters
    /** Where the net usually sits, and what shape it is, in a 480p frame. */
    private static final double PRIOR_CX     = 0.50;
    private static final double PRIOR_CY     = 0.75;
    private static final double PRIOR_ASPECT = 4.2;
    private static final double SIGMA_CX     = 0.25;
    private static final double SIGMA_CY     = 0.20;
    private static final double SIGMA_ASPECT = 1.0;
    //------------------------------------------------------------------------
    // ---- reusable buffers ----
    private Mat integral = new Mat();
    private double[] sum;
    private int integralWidth;

    public static class NetResult{
        public final Rect box;

        NetResult(Rect box){
            this.box = box;
        }
    }

    public enum ColourSpace {
        /** 3-channel YUV; channel 0 is luma. Decoder output. */
        YUV,
        /** 4-channel RGBA, as produced by Utils.bitmapToMat. */
        RGBA,
        /** Already single-channel luma. */
        GRAY
    }

    public NetResult detect(Mat background, ColourSpace space) {
        if (background == null || background.empty()) return null;

        Mat work = new Mat();
        Mat grey = new Mat();
        Mat edges = new Mat();

        try {
            double scale = WORK_HEIGHT / (double) background.rows();
            resizeMat(background, work);
            toGrey(work, grey, space);
            extractEdges(grey, edges);

            List<Rect> candidates = findCandidates(edges, grey, work.cols(), work.rows());
            Rect bestAt480p = pickBest(candidates, work.cols(), work.rows());

            if (bestAt480p == null) {
                return null;
            }

            // Map back to original image coordinate space
            Rect originalBox = scaleToOriginal(bestAt480p, scale);
            return new NetResult(originalBox);

        } finally {
            work.release();
            grey.release();
            edges.release();
        }
    }

    //resize img to standard definition.
    private void resizeMat(Mat src, Mat dst){
        double scale = WORK_HEIGHT/src.rows();
        if (Math.abs(scale - 1.0) < 0.01){
            //image is already in the correct resolution
            src.copyTo(dst);
        }else{
            //resize img to working resolution = 480p
            Imgproc.resize(src, dst, new Size(), scale, scale, Imgproc.INTER_AREA);
        }
    }

    //convert image to greyscale by taking luma plane from YUV because CANNY algorithm expects a greyscale image.
    private void toGrey(Mat src, Mat dst, ColourSpace space) {
        switch (space) {
            case GRAY:
                src.copyTo(dst);
                break;
            case YUV:
                Core.extractChannel(src, dst, 0);          // Y == luma
                break;
            case RGBA:
                Imgproc.cvtColor(src, dst, Imgproc.COLOR_RGBA2GRAY);
                break;
            default:
                throw new IllegalArgumentException("Unhandled colour space: " + space);
        }
    }

    //uses bilateral filter and gaussian blurring openCV algorithms to find the edges in the image.
    private void extractEdges(Mat src, Mat dst){
        Mat filtered = new Mat();
        Mat kernel = null;
        try{
            Imgproc.bilateralFilter(src, filtered, d, sigmaColour, sigmaSpace);
            Imgproc.Canny(filtered, dst, lowerThreshold, upperThreshold);

            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(MORPH_WIDTH,MORPH_HEIGHT));
            Imgproc.morphologyEx(dst,dst, Imgproc.MORPH_CLOSE, kernel);
        }finally {
            filtered.release();
            if (kernel != null) kernel.release();
        }
    }

    private List<Rect> findCandidates(Mat edges, Mat grey, int frameWidth, int frameHeight){
        List<Rect> candidates = new ArrayList<>();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        try{
            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

            // ---- once per frame, before the contour loop ----
            Imgproc.integral(grey, integral, CvType.CV_64F);       // grey = the Y plane, BEFORE bilateral
            integralWidth = integral.cols();
            int need = (int) integral.total();
            if (sum == null || sum.length != need) sum = new double[need];
            integral.get(0, 0, sum);                               // one JNI copy, then plain Java maths

            for (MatOfPoint contour : contours) {
                Rect box = Imgproc.boundingRect(contour);

                // cheapest and most selective first
                if (box.width < MIN_BOX_WIDTH || box.width > MAX_BOX_WIDTH) continue;
                if (box.y + box.height < MIN_Y_FRACTION * frameHeight) continue;

                double cx = (box.x + box.width  / 2.0) / frameWidth;
                double cy = (box.y + box.height / 2.0) / frameHeight;

                if (Math.abs(cx - 0.5) > MAX_CX_OFFSET) continue;
                if (cy < MIN_CY_FRACTION)               continue;

                // darkness contrast: compare the box against a ring around it
                int m   = Math.max(8, (int) (0.35 * box.height));
                int ox0 = Math.max(0, box.x - m);
                int oy0 = Math.max(0, box.y - m);
                int ox1 = Math.min(frameWidth,  box.x + box.width  + m);
                int oy1 = Math.min(frameHeight, box.y + box.height + m);

                double boxSum   = rectSum(box.x, box.y, box.x + box.width, box.y + box.height);
                double outerSum = rectSum(ox0, oy0, ox1, oy1);

                int boxN  = box.width * box.height;
                int ringN = (ox1 - ox0) * (oy1 - oy0) - boxN;
                if (ringN <= 0) continue;

                double boxMean  = boxSum / boxN;
                double ringMean = (outerSum - boxSum) / ringN;

                if (ringMean - boxMean < MIN_DARKNESS) continue;

                candidates.add(box);
            }

            Log.d(TAG, "Raw contours: " + contours.size() + "| Candidates Retained: " + candidates.size());
            return candidates;
        }
        finally{
            for(MatOfPoint c : contours) c.release();
            hierarchy.release();
        }

    }

    private double rectSum(int x0, int y0, int x1, int y1) {
        return sum[y1 * integralWidth + x1] - sum[y0 * integralWidth + x1]
                - sum[y1 * integralWidth + x0] + sum[y0 * integralWidth + x0];
    }

    /** Picks the most net-like candidate, or null if there are none. */
    private Rect pickBest(List<Rect> candidates, int frameWidth, int frameHeight) {
        Rect best = null;
        double bestScore = -1.0;
        for (Rect candidate : candidates) {
            double score = score(candidate, frameWidth, frameHeight);
            if (score > bestScore && score > 0.3) {
                bestScore = score;
                best = candidate;
            }
        }
        return best;
    }

    /**
     * How net-like a candidate is: near the middle horizontally, low in the frame,
     * and clearly wider than it is tall. Each term is 1.0 at the ideal value and
     * falls off smoothly, so nothing is ever excluded outright - a candidate that is
     * off on one term can still win if it is strong on the others.
     */
    private double score(Rect box, int frameWidth, int frameHeight) {
        double cx = (box.x + box.width / 2.0) / frameWidth;
        double cy = (box.y + box.height / 2.0) / frameHeight;
        double aspect = box.width / (double) Math.max(box.height, 1);

        double horizontal = gaussian(cx, PRIOR_CX, SIGMA_CX);
        double vertical = gaussian(cy, PRIOR_CY, SIGMA_CY);
        double shape = gaussian(Math.log(aspect / PRIOR_ASPECT), 0.0, SIGMA_ASPECT);

        return horizontal * vertical * shape;
    }

    /** 1.0 at the mean, falling off with distance. */
    private static double gaussian(double value, double mean, double sigma) {
        double z = (value - mean) / sigma;
        return Math.exp(-z * z);
    }

    private Rect scaleToOriginal(Rect r, double scale) {
        if (r == null) return null;
        double inv = 1.0 / scale;
        return new Rect(
                (int) Math.round(r.x * inv),
                (int) Math.round(r.y * inv),
                (int) Math.round(r.width * inv),
                (int) Math.round(r.height * inv)
        );
    }
}
