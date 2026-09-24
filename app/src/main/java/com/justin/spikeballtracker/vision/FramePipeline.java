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
import java.util.concurrent.atomic.AtomicBoolean;

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

    private final AtomicBoolean resetRequested = new AtomicBoolean(false);
    private long warmStartUs = 0L;

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

        if (index == 0) {
            warmStartUs = timestampUs;
        }
        if (resetRequested.getAndSet(false)) {
            applyReset(timestampUs);
        }

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

    private static final long NET_WARMUP_US           = 10_000_000L;
    private static final long NET_ATTEMPT_INTERVAL_US =  5_000_000L;

    private Rect cachedNetBox = null;

    /**
     * When a detect was last attempted, successful or not. This gates the cost,
     * so a failure has to move it too — otherwise a scene the detector cannot
     * solve makes it retry on every single frame.
     *
     * <p>Starts one interval in the past, meaning "due as soon as warm".
     */
    private long lastNetAttemptUs = -NET_ATTEMPT_INTERVAL_US;

    private Mat updateNet(long timestampUs, boolean plateWanted) {
        boolean detectDue =
                timestampUs - warmStartUs >= NET_WARMUP_US
                        && timestampUs - lastNetAttemptUs >= NET_ATTEMPT_INTERVAL_US;

        if (!detectDue && !plateWanted) {
            return null;   // nothing needs a plate this frame
        }

        try {
            motionDetector.getBackgroundImage(background);
            if (background.empty()) {
                return null;   // model has nothing yet; don't burn the attempt on it
            }

            if (detectDue) {
                // Stamped *before* the call, so a null result or a throw costs the
                // same interval as a success.
                lastNetAttemptUs = timestampUs;

                NetDetector.NetResult result =
                        netDetector.detect(background, NetDetector.ColourSpace.YUV);

                if (result != null && result.box != null) {
                    cachedNetBox = result.box;
                    Log.d(TAG, "net locked: " + cachedNetBox);
                } else {
                    Log.d(TAG, "no net in this plate; next attempt in "
                            + (NET_ATTEMPT_INTERVAL_US / 1_000_000L) + "s");
                }
            }
            return background;
        } catch (Exception e) {
            Log.e(TAG, "net update failed", e);
            return null;
        }
    }

    public void requestReset() {
        resetRequested.set(true);
    }

    private void applyReset(long timestampUs) {
        motionDetector.reset();
        cachedNetBox = null;              // the old box describes the old framing
        warmStartUs = timestampUs;        // warm-up counts from here, not from run start
        lastNetAttemptUs = timestampUs - NET_ATTEMPT_INTERVAL_US;   // due the moment it is warm
        Log.d(TAG, "background reset; net cleared, warm-up restarts");
    }

    public void release() {
        motionDetector.release();
        colourDetector.release();
        blobDetector.release();
        ballMask.release();
        background.release();
    }
}