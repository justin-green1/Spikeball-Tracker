package com.justin.spikeballtracker.vision;

import android.util.Log;

import com.justin.spikeballtracker.tracking.BallCandidate;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.List;

/**
 * One frame of the detection pipeline, shared by the recorded-video path and
 * the live-camera path.
 *
 * <p>Owns stateful detectors — in particular {@link MotionDetector}, whose KNN
 * model must see every frame of a run in order — so one instance belongs to
 * exactly one run on exactly one thread. Create it when a run starts,
 * {@link #release()} it when the run ends, never share it between paths.
 *
 * <p>Input is always a CV_8UC3 Y,U,V Mat at the working resolution
 * (854x480, landscape). Delivering that is the caller's job.
 */
public final class FramePipeline {

    private static final String TAG = "FramePipeline";

    private static final Scalar MARKER_COLOUR      = new Scalar(255, 0, 0);
    private static final Scalar NET_OVERLAY_COLOUR = new Scalar(0, 255, 255);

    private static final long INITIAL_NET_DELAY_US          = 10_000_000L;
    private static final long NET_RECALIBRATION_INTERVAL_US =  5_000_000L;

    public static final class Result {
        public final int frameIndex;
        public final List<BallCandidate> candidates;
        /** Net box in frame coordinates, or null until the first lock. */
        public final Rect netBox;
        /**
         * The KNN background plate, non-null only on frames where one was
         * fetched. Pipeline-owned and overwritten on the next process() call —
         * consume it before returning.
         */
        public final Mat backgroundPlate;

        Result(int frameIndex, List<BallCandidate> candidates,
               Rect netBox, Mat backgroundPlate) {
            this.frameIndex = frameIndex;
            this.candidates = candidates;
            this.netBox = netBox;
            this.backgroundPlate = backgroundPlate;
        }
    }

    private final MotionDetector motionDetector = new MotionDetector();
    private final ColourDetector colourDetector = new ColourDetector();
    private final BlobDetector   blobDetector   = new BlobDetector();
    private final NetDetector    netDetector    = new NetDetector();

    private final Mat ballMask   = new Mat();
    private final Mat background = new Mat();

    private Rect cachedNetBox    = null;
    private long lastNetUpdateUs = -1L;
    private int  frameIndex      = 0;

    /**
     * @param yuv         CV_8UC3 Y,U,V frame at the working resolution
     * @param timestampUs microseconds since this run's first frame
     * @param displayOut  written with an RGB copy of the frame plus overlays
     * @param plateWanted force a background-plate fetch even when the net does
     *                    not need one (the exporter's ten-second interval)
     */
    public Result process(Mat yuv, long timestampUs, Mat displayOut, boolean plateWanted) {
        int index = frameIndex++;

        Mat motionMask = motionDetector.processFrame(yuv);
        Mat colourMask = colourDetector.processFrame(yuv);
        Core.bitwise_and(colourMask, motionMask, ballMask);

        List<BallCandidate> candidates =
                blobDetector.findCandidates(ballMask, index, timestampUs);

        Imgproc.cvtColor(yuv, displayOut, Imgproc.COLOR_YUV2RGB);
        for (BallCandidate c : candidates) {
            Imgproc.circle(displayOut, new Point(c.x, c.y), 12, MARKER_COLOUR, 2);
        }

        Mat plate = updateNet(timestampUs, plateWanted);

        if (cachedNetBox != null) {
            Imgproc.rectangle(displayOut, cachedNetBox.tl(), cachedNetBox.br(),
                    NET_OVERLAY_COLOUR, 2);
        }

        return new Result(index, candidates, cachedNetBox, plate);
    }

    private Mat updateNet(long timestampUs, boolean plateWanted) {
        boolean needsInitialLock =
                (cachedNetBox == null && timestampUs > INITIAL_NET_DELAY_US);
        boolean needsRecalibration =
                (cachedNetBox != null
                        && timestampUs - lastNetUpdateUs >= NET_RECALIBRATION_INTERVAL_US);

        if (!needsInitialLock && !needsRecalibration && !plateWanted) {
            return null;
        }

        try {
            motionDetector.getBackgroundImage(background);
            if (background.empty()) {
                return null;   // model not warm yet; try again next frame
            }

            if (needsInitialLock || needsRecalibration) {
                NetDetector.NetResult result =
                        netDetector.detect(background, NetDetector.ColourSpace.YUV);
                if (result != null && result.box != null) {
                    cachedNetBox = result.box;
                    lastNetUpdateUs = timestampUs;
                    Log.d(TAG, (needsInitialLock ? "initial net lock: " : "net recalibrated: ")
                            + cachedNetBox);
                }
            }
            return background;
        } catch (Exception e) {
            Log.e(TAG, "net update failed", e);
            return null;
        }
    }

    public void release() {
        motionDetector.release();
        colourDetector.release();
        blobDetector.release();
        ballMask.release();
        background.release();
    }
}