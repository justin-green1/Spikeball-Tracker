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
 * the rally detector, never the picture.
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

    // ---------------------------------------------------------------- detectors

    private final MotionDetector  motionDetector  = new MotionDetector();
    private final ColourDetector  colourDetector  = new ColourDetector();
    private final SvmBlobDetector svmBlobDetector =
            new SvmBlobDetector(ColourDetector.LO, ColourDetector.HI);
    private final NetDetector     netDetector     = new NetDetector();
    private final BallTracker     tracker         = new BallTracker();

    /** Replaced on a background reset, so not final. */
    private RallyDetector rallyDetector = new RallyDetector();

    // ---------------------------------------------------------------- reusable Mats

    private final Mat ballMask   = new Mat();
    private final Mat background = new Mat();

    // ---------------------------------------------------------------- run state

    private final AtomicBoolean resetRequested = new AtomicBoolean(false);

    private int frameIndex = 0;
    private long warmStartUs = 0L;
    private Rect cachedNetBox = null;

    /** The rally detector's padded zone around {@link #cachedNetBox}; null with it. */
    private NetZone cachedNetZone = null;

    /** How many of the current rally detector's rallies have been handed out. */
    private int ralliesReported = 0;

    /** Confirmed since the last {@link #process}; filled by {@link #collectRallies()}. */
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
     * @return rallies confirmed on this frame, usually none. A rally is
     *         confirmed a few seconds after it ends, so its times are well in the past.
     */
    public List<RallyDetector.Rally> process(Mat yuv, long timestampUs, Mat displayOut) {
        int index = frameIndex++;
        if (index == 0) {
            warmStartUs = timestampUs;
        }
        if (resetRequested.getAndSet(false)) {
            applyReset(timestampUs);
        }

        // Detection: the colour box ANDed with the motion mask (shadows kept),
        // then every blob scored by the SVM; the candidates are those at or
        // above BlobSvm.THRESHOLD.
        Core.bitwise_and(colourDetector.processFrame(yuv),
                motionDetector.processFrame(yuv), ballMask);
        List<BallCandidate> candidates = svmBlobDetector.findCandidates(ballMask, yuv);

        // Tracking.
        tracker.process(index, timestampUs, candidates);

        // Net.
        updateNet(timestampUs);

        // Rallies: the delayed ball answer, judged against the net as it stands now.
        rallyDetector.process(tracker.latest(), cachedNetZone);
        collectRallies();

        // Display: the live frame and the net, nothing else.
        Imgproc.cvtColor(yuv, displayOut, Imgproc.COLOR_YUV2RGB);
        drawNet(displayOut);

        return drainRallies();
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
        colourDetector.release();
        svmBlobDetector.release();
        ballMask.release();
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

            Rect box = netDetector.detect(background);
            if (box != null) {
                cachedNetBox = box;
                cachedNetZone = NetZone.around(box.x, box.y, box.width, box.height);
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
        // with this frame's result.
        rallyDetector.finish();
        collectRallies();
        rallyDetector = new RallyDetector();
        ralliesReported = 0;

        motionDetector.reset();
        svmBlobDetector.reset();        // the occupancy grid lives in front of the SVM
        tracker.reset();
        cachedNetBox = null;                                        // the old box describes the old framing
        cachedNetZone = null;
        warmStartUs = timestampUs;                                  // warm-up counts from here
        lastNetAttemptUs = timestampUs - NET_ATTEMPT_INTERVAL_US;   // due the moment it is warm
    }
}
