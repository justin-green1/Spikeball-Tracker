package com.justin.spikeballtracker.vision;

import android.util.Log;
import com.justin.spikeballtracker.tracking.BallCandidate;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

public class BlobDetector {

    private static final String TAG = "BlobDetector";
    private static final int WIDE_MIN_AREA = 40;
    private static final int WIDE_MAX_AREA = 300;
    private static final int NARROW_MIN_AREA = 15;
    private static final int NARROW_MAX_AREA = 400;
    private final int minArea;
    private final int maxArea;

    private final Mat labels = new Mat();
    private final Mat stats = new Mat();
    private final Mat centroids = new Mat();

    /**
     * Two presets, area in pixels, inclusive range.
     * @param minArea
     * @param maxArea
     */

    private BlobDetector(int minArea, int maxArea){
        this.minArea = minArea;
        this.maxArea = maxArea;
    }

    public static BlobDetector wide() {
        return new BlobDetector(WIDE_MIN_AREA, WIDE_MAX_AREA);
    }

    public static BlobDetector narrow() {
        return new BlobDetector(NARROW_MIN_AREA, NARROW_MAX_AREA);
    }

    /**
     * @param mask single-channel 8-bit binary mask
     */
    public List<BallCandidate> findCandidates(Mat mask, int frameIndex, long timestampUs) {
        List<BallCandidate> candidates = new ArrayList<>();

        int count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids);

        // Label 0 is always the background, hence i = 1.
        BallCandidate bc;
        for (int i = 1; i < count; i++) {
            // stats is CV_32S, centroids is CV_64F, but Mat.get always hands
            // back a double[] of the channel values at that position.
            int area   = (int) stats.get(i, Imgproc.CC_STAT_AREA)[0];
            if (area < minArea || area > maxArea){
                continue;
            }

            int left   = (int) stats.get(i, Imgproc.CC_STAT_LEFT)[0];
            int top    = (int) stats.get(i, Imgproc.CC_STAT_TOP)[0];
            int width  = (int) stats.get(i, Imgproc.CC_STAT_WIDTH)[0];
            int height = (int) stats.get(i, Imgproc.CC_STAT_HEIGHT)[0];

            double x = centroids.get(i, 0)[0];
            double y = centroids.get(i, 1)[0];


            bc = new BallCandidate(frameIndex, timestampUs, x, y, area, left, top, width, height);
            candidates.add(bc);
        }

        if (!candidates.isEmpty()) {
            Log.d(TAG, "frame " + frameIndex + ": " + candidates);
        }
        return candidates;
    }

    public void release() {
        labels.release();
        stats.release();
        centroids.release();
    }
}