package com.justin.spikeballtracker.vision;

import com.justin.spikeballtracker.tracking.BallCandidate;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Connected components of the widened-box ball mask, each scored by
 * {@link BlobSvm}.
 *
 * <p>Returns every component in the area range with its P(ball); the caller
 * splits them into narrow and wide streams by threshold. Components whose
 * features cannot be computed (fewer than 3 in-box pixels, which the study
 * skipped) get P(ball) = 0.
 *
 * <p>The SVM was trained on 40-400 px blobs only. The floor here is 15, as
 * the narrow stream used, so 15-39 px blobs are scored outside the range the
 * model has seen.
 *
 * <p>Owns reusable Mats, so one instance belongs to one thread.
 */
public final class SvmBlobDetector {

    public static final int MIN_AREA = 15;
    public static final int MAX_AREA = 400;

    private final Mat labels = new Mat();
    private final Mat stats = new Mat();
    private final Mat centroids = new Mat();
    private final BlobFeatures features;
    private final double[] f = new double[BlobSvm.N_FEATURES];

    /**
     * @param boxLo the colour box the mask was cut with, lower Y,U,V bound
     * @param boxHi upper bound, inclusive
     */
    public SvmBlobDetector(Scalar boxLo, Scalar boxHi) {
        features = new BlobFeatures(boxLo, boxHi);
    }

    /**
     * @param mask single-channel 8-bit mask, colour box AND motion (non-zero = foreground)
     * @param yuv  the CV_8UC3 Y,U,V frame the mask came from
     * @return every component of {@link #MIN_AREA}-{@link #MAX_AREA} px, scored
     */
    public List<BallCandidate> findCandidates(Mat mask, Mat yuv, int frameIndex, long timestampUs) {
        int count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids);
        List<BallCandidate> out = new ArrayList<>();
        if (count <= 1) {
            return out;
        }

        // One JNI read each, rather than one per field per component.
        int[] st = new int[count * stats.cols()];
        stats.get(0, 0, st);
        double[] ce = new double[count * 2];
        centroids.get(0, 0, ce);
        int sc = stats.cols();

        for (int i = 1; i < count; i++) {   // label 0 is the background
            int area = st[i * sc + Imgproc.CC_STAT_AREA];
            if (area < MIN_AREA || area > MAX_AREA) {
                continue;
            }
            int left = st[i * sc + Imgproc.CC_STAT_LEFT];
            int top = st[i * sc + Imgproc.CC_STAT_TOP];
            int width = st[i * sc + Imgproc.CC_STAT_WIDTH];
            int height = st[i * sc + Imgproc.CC_STAT_HEIGHT];

            double p = features.compute(yuv, labels, i, left, top, width, height, area, f)
                    ? BlobSvm.probability(f) : 0.0;

            out.add(new BallCandidate(frameIndex, timestampUs, ce[2 * i], ce[2 * i + 1], area,
                    left, top, width, height, p));
        }
        return out;
    }

    public void release() {
        labels.release();
        stats.release();
        centroids.release();
        features.release();
    }
}
