package com.justin.spikeballtracker.vision;

import android.util.Log;

import com.justin.spikeballtracker.tracking.BallCandidate;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class BlobDetector {

    private static final String TAG = "BlobDetector";

    // Ball measured at 20-50 px^2, so these sit either side with headroom.
    // Set MIN_AREA to 1 for the first run if you want to see the noise floor.
    private static final int MIN_AREA = 8;
    private static final int MAX_AREA = 400;

    // A scene cut or camera shake can light up the whole mask. Bail rather
    // than loop over thousands of components.
    private static final int MAX_COMPONENTS = 500;

    private final Mat labels = new Mat();
    private final Mat stats = new Mat();
    private final Mat centroids = new Mat();

    /**
     * @param mask single-channel 8-bit binary mask
     */
    public List<BallCandidate> findCandidates(Mat mask, int frameIndex, long timestampUs) {
        List<BallCandidate> candidates = new ArrayList<>();

        int count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids);

        if (count > MAX_COMPONENTS) {
            Log.w(TAG, "frame " + frameIndex + ": " + count + " components, skipping");
            return candidates;
        }

        // Label 0 is always the background, hence i = 1.
        BallCandidate bc;
        for (int i = 1; i < count; i++) {
            // stats is CV_32S, centroids is CV_64F, but Mat.get always hands
            // back a double[] of the channel values at that position.
            int area = (int) stats.get(i, Imgproc.CC_STAT_AREA)[0];
            if (area < MIN_AREA || area > MAX_AREA) {
                continue;
            }
            double x = centroids.get(i, 0)[0];
            double y = centroids.get(i, 1)[0];

            bc = new BallCandidate(frameIndex, timestampUs, x, y, area);
            Log.v(TAG, bc.toString());
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