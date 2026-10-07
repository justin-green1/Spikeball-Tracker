package com.justin.spikeballtracker.vision;

import android.util.Log;

import com.justin.spikeballtracker.tracking.BallCandidate;
import com.justin.spikeballtracker.tracking.BallTracker;
import com.justin.spikeballtracker.tracking.NetZone;
import com.justin.spikeballtracker.tracking.RallyDetector;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One frame of the detection pipeline, shared by the recorded-video path and
 * the live-camera path.
 *
 * <p>Owns stateful detectors — in particular {@link MotionDetector}, whose KNN
 * model must see every frame of a run in order — so one instance belongs to
 * exactly one run on exactly one thread. Create it when a run starts,
 * {@link #finish()} and {@link #release()} it when the run ends, never share
 * it between paths.
 *
 * <p>The display is the current frame with only the net box drawn on it. The
 * ball tracker answers {@link BallTracker#DELAY} frames late; its answers feed
 * the rally detector and the logs, never the picture.
 *
 * <p>Input is always a CV_8UC3 Y,U,V Mat at the working resolution
 * (854x480, landscape). Delivering that is the caller's job.
 */
public final class FramePipeline {

    private static final String TAG = "FramePipeline";

    /** The display Mat is RGB, so this is (R, G, B). */
    private static final Scalar NET_COLOUR = new Scalar(0, 255, 255);

    // ---------------------------------------------------------------- net timing

    private static final long NET_WARMUP_US           = 10_000_000L;
    private static final long NET_ATTEMPT_INTERVAL_US =  5_000_000L;

    // ---------------------------------------------------------------- result

    /** Everything the pipeline decided about one frame. */
    public static final class Result {
        /** 0-based index of the frame this result describes. */
        public final int frameIndex;

        /** Blobs that passed the narrow colour box and the motion mask. */
        public final List<BallCandidate> narrowCandidates;

        /** Blobs that passed the wide colour box and the motion mask. */
        public final List<BallCandidate> wideCandidates;

        /**
         * The tracker's answer for frame {@code frameIndex - BallTracker.DELAY},
         * or null during the first DELAY frames of a run.
         */
        public final BallTracker.Output ball;

        /** Net box in frame coordinates, or null until the first lock. */
        public final Rect netBox;

        /**
         * Rallies confirmed on this frame, usually empty. A rally is confirmed
         * a few seconds after it ends, so its times are well in the past.
         */
        public final List<RallyDetector.Rally> newRallies;

        Result(int frameIndex,
               List<BallCandidate> narrowCandidates,
               List<BallCandidate> wideCandidates,
               BallTracker.Output ball,
               Rect netBox,
               List<RallyDetector.Rally> newRallies) {
            this.frameIndex = frameIndex;
            this.narrowCandidates = narrowCandidates;
            this.wideCandidates = wideCandidates;
            this.ball = ball;
            this.netBox = netBox;
            this.newRallies = newRallies;
        }
    }

    // ---------------------------------------------------------------- detectors

    private final MotionDetector motionDetector       = new MotionDetector();
    private final ColourDetector narrowColourDetector = ColourDetector.narrow();
    private final ColourDetector wideColourDetector   = ColourDetector.wide();
    private final BlobDetector   narrowBlobDetector   = BlobDetector.narrow();
    private final BlobDetector   wideBlobDetector     = BlobDetector.wide();
    private final NetDetector    netDetector          = new NetDetector();
    private final BallTracker    tracker              = new BallTracker();

    /** Replaced on a background reset, so not final. */
    private RallyDetector rallyDetector = new RallyDetector();

    // ---------------------------------------------------------------- reusable Mats

    private final Mat narrowBallMask = new Mat();
    private final Mat wideBallMask   = new Mat();
    private final Mat background     = new Mat();

    // ---------------------------------------------------------------- run state

    private final AtomicBoolean resetRequested = new AtomicBoolean(false);

    private int frameIndex = 0;
    private long warmStartUs = 0L;
    private Rect cachedNetBox = null;

    /** The rally detector's padded zone around {@link #cachedNetBox}; null with it. */
    private NetZone cachedNetZone = null;

    /** How many of the current rally detector's rallies have been handed out. */
    private int ralliesReported = 0;

    /** Confirmed since the last Result; filled by {@link #collectRallies()}. */
    private final List<RallyDetector.Rally> confirmedRallies = new ArrayList<>();

    /**
     * When a net detect was last attempted, successful or not. This gates the
     * cost, so a failure has to move it too — otherwise a scene the detector
     * cannot solve makes it retry on every single frame.
     *
     * <p>Starts one interval in the past, meaning "due as soon as warm".
     */
    private long lastNetAttemptUs = -NET_ATTEMPT_INTERVAL_US;

    // ---------------------------------------------------------------- lifecycle

    /**
     * Runs one frame through detection, tracking and rally detection, and
     * draws the display.
     *
     * @param yuv         CV_8UC3 Y,U,V frame at the working resolution
     * @param timestampUs microseconds since this run's first frame
     * @param displayOut  written with this frame in RGB, plus the net box
     * @return what the pipeline decided about this frame
     */
    public Result process(Mat yuv, long timestampUs, Mat displayOut) {
        int index = frameIndex++;
        if (index == 0) {
            warmStartUs = timestampUs;
        }
        if (resetRequested.getAndSet(false)) {
            applyReset(timestampUs);
        }

        // Detection: one motion mask, shared by both colour boxes.
        Mat motionMask = motionDetector.processFrame(yuv);
        Core.bitwise_and(narrowColourDetector.processFrame(yuv), motionMask, narrowBallMask);
        Core.bitwise_and(wideColourDetector.processFrame(yuv), motionMask, wideBallMask);

        List<BallCandidate> narrowCandidates =
                narrowBlobDetector.findCandidates(narrowBallMask, index, timestampUs);
        List<BallCandidate> wideCandidates =
                wideBlobDetector.findCandidates(wideBallMask, index, timestampUs);

        // Tracking.
        tracker.process(index, timestampUs, narrowCandidates, wideCandidates);
        BallTracker.Output ball = tracker.latest();

        // Net.
        updateNet(timestampUs);

        // Rallies: the delayed ball answer, judged against the net as it stands now.
        rallyDetector.process(ball, cachedNetZone);
        collectRallies();

        // Display: the live frame and the net, nothing else.
        Imgproc.cvtColor(yuv, displayOut, Imgproc.COLOR_YUV2RGB);
        drawNet(displayOut);

        return new Result(index, narrowCandidates, wideCandidates, ball, cachedNetBox,
                drainRallies());
    }

    /**
     * Ends the run for the rally detector: closes the open flight, scores the
     * frames still waiting, and confirms a rally still in progress. Call once,
     * after the last {@link #process}, before {@link #release()}.
     *
     * @return rallies confirmed by finishing, usually zero or one
     */
    public List<RallyDetector.Rally> finish() {
        rallyDetector.finish();
        collectRallies();
        return drainRallies();
    }

    /** Asks for a background reset; applied at the start of the next frame, on the pipeline thread. */
    public void requestReset() {
        resetRequested.set(true);
    }

    /** Frees every native Mat this pipeline owns. */
    public void release() {
        motionDetector.release();
        narrowColourDetector.release();
        wideColourDetector.release();
        narrowBlobDetector.release();
        wideBlobDetector.release();
        narrowBallMask.release();
        wideBallMask.release();
        background.release();
    }

    // ---------------------------------------------------------------- net

    /**
     * Re-runs net detection on the KNN background plate every
     * {@link #NET_ATTEMPT_INTERVAL_US}, once the model has had
     * {@link #NET_WARMUP_US} to settle.
     */
    private void updateNet(long timestampUs) {
        boolean detectDue =
                timestampUs - warmStartUs >= NET_WARMUP_US
                        && timestampUs - lastNetAttemptUs >= NET_ATTEMPT_INTERVAL_US;
        if (!detectDue) {
            return;
        }

        try {
            motionDetector.getBackgroundImage(background);
            if (background.empty()) {
                return;   // model has nothing yet; don't burn the attempt on it
            }

            // Stamped before the call, so a null result or a throw costs the
            // same interval as a success.
            lastNetAttemptUs = timestampUs;

            NetDetector.NetResult result = netDetector.detect(background);
            if (result != null && result.box != null) {
                cachedNetBox = result.box;
                cachedNetZone = NetZone.around(cachedNetBox.x, cachedNetBox.y,
                        cachedNetBox.width, cachedNetBox.height);
                Log.d(TAG, "net locked: " + cachedNetBox);
            } else {
                Log.d(TAG, "no net in this plate; next attempt in "
                        + (NET_ATTEMPT_INTERVAL_US / 1_000_000L) + "s");
            }
        } catch (Exception e) {
            Log.e(TAG, "net update failed", e);
        }
    }

    /** Draws the cached net box, if there is one. */
    private void drawNet(Mat img) {
        if (cachedNetBox != null) {
            Imgproc.rectangle(img, cachedNetBox.tl(), cachedNetBox.br(), NET_COLOUR, 2);
        }
    }

    // ---------------------------------------------------------------- rallies

    /** Moves rallies the detector has confirmed since the last call into {@link #confirmedRallies}. */
    private void collectRallies() {
        List<RallyDetector.Rally> all = rallyDetector.rallies();
        while (ralliesReported < all.size()) {
            confirmedRallies.add(all.get(ralliesReported++));
        }
    }

    /** Hands out the collected rallies and clears the list. */
    private List<RallyDetector.Rally> drainRallies() {
        if (confirmedRallies.isEmpty()) {
            return Collections.emptyList();
        }
        List<RallyDetector.Rally> out = new ArrayList<>(confirmedRallies);
        confirmedRallies.clear();
        return out;
    }

    // ---------------------------------------------------------------- reset

    /** Clears every model that describes the old framing; called on the pipeline thread. */
    private void applyReset(long timestampUs) {
        // Close the old framing's rallies first; anything confirmed goes out
        // with this frame's Result.
        rallyDetector.finish();
        collectRallies();
        rallyDetector = new RallyDetector();
        ralliesReported = 0;

        motionDetector.reset();
        tracker.reset();
        cachedNetBox = null;                                        // the old box describes the old framing
        cachedNetZone = null;
        warmStartUs = timestampUs;                                  // warm-up counts from here
        lastNetAttemptUs = timestampUs - NET_ATTEMPT_INTERVAL_US;   // due the moment it is warm
        Log.d(TAG, "background reset; net, tracker and rallies cleared, warm-up restarts");
    }
}
