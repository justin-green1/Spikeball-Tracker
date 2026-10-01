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

/**
 * Finds the net in a KNN background plate.
 *
 * <p>The only caller is {@link FramePipeline}, and both of its sources resize
 * every frame to 854x480 before the pipeline sees it, so the plate always
 * arrives at the working resolution. Nothing here resizes or maps coordinates
 * back: every pixel constant below is in 480p units, and the returned box is
 * already in plate (and so display) coordinates.
 */
public class NetDetector {
    private static final String TAG = "NetDetector";
    /** The only plate height {@link #detect} accepts; the pixel constants below assume it. */
    private static final int WORK_HEIGHT = 480;

    // ---------------------------------------------------------------------
    // candidate parameters.

    private static final double MIN_Y_FRACTION  = 0.69;     // was 0.67
    private static final int    MIN_BOX_WIDTH   = 75;
    private static final int    MAX_BOX_WIDTH   = 220;
    private static final double lowerThreshold  = 57.1;     // = upper / 2.09
    private static final double upperThreshold  = 119.25;
    private static final double sigmaSpace      = 152.8;    // has no effect at d = 5
    private static final double sigmaColour     = 146.6;
    private static final int    d               = 5;
    private static final int    MORPH_HEIGHT    = 9;
    private static final int    MORPH_WIDTH     = 25;       // was 21
    private static final double MAX_CX_OFFSET   = 0.14;
    private static final double MIN_CY_FRACTION = 0.50;
    private static final double MIN_DARKNESS    = -9.0;
    private static final double MIN_SCORE       = 0.38;
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

    /**
     * @param plate CV_8UC3 Y,U,V background plate from the KNN model, at the
     *              working resolution (854x480)
     * @return the net box in plate coordinates, or null if no candidate scores
     *         high enough or the plate is not {@value #WORK_HEIGHT} rows high
     */
    public NetResult detect(Mat plate) {
        if (plate == null || plate.empty()) return null;

        // Guards the "already at working resolution" contract. A source that
        // skipped its resize would otherwise get boxes judged against 480p
        // constants at the wrong scale, silently.
        if (plate.rows() != WORK_HEIGHT) {
            Log.w(TAG, "plate is " + plate.cols() + "x" + plate.rows()
                    + ", expected " + WORK_HEIGHT + " rows; skipping detect");
            return null;
        }

        Mat grey = new Mat();
        Mat edges = new Mat();

        try {
            // Canny needs one channel, and in YUV that is simply the Y (luma) plane.
            Core.extractChannel(plate, grey, 0);
            extractEdges(grey, edges);

            List<Rect> candidates = findCandidates(edges, grey, grey.cols(), grey.rows());
            Rect best = pickBest(candidates, grey.cols(), grey.rows());

            return best == null ? null : new NetResult(best);

        } finally {
            grey.release();
            edges.release();
        }
    }

    /**
     * Edge map for the candidate search: a bilateral filter flattens texture
     * while keeping strong edges, Canny finds them, and a wide, short
     * elliptical close joins the net rim's broken edge runs into one contour.
     */
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

            //candidates = suppressContained(candidates);
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
            if (score > bestScore && score > MIN_SCORE) {
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
}
