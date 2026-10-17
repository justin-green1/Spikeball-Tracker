package com.justin.spikeballtracker.vision;

import android.util.Log;

import com.justin.spikeballtracker.log.Json;
import com.justin.spikeballtracker.log.RunLog;
import com.justin.spikeballtracker.tracking.BallCandidate;
import com.justin.spikeballtracker.tracking.BallTracker;
import com.justin.spikeballtracker.tracking.NetZone;
import com.justin.spikeballtracker.tracking.RallyDetector;
import com.justin.spikeballtracker.tracking.Track;
import com.justin.spikeballtracker.video.YuvRange;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
    private RallyDetector rallyDetector;

    // ---------------------------------------------------------------- reusable Mats

    private final Mat ballMask   = new Mat();
    private final Mat background = new Mat();
    private final Mat shadowMask = new Mat();   // run log only

    // ---------------------------------------------------------------- run log

    /** Where every frame is described; null when not logging. */
    private final RunLog log;

    // This frame's events, for the log line.
    private boolean resetThisFrame = false;
    private boolean netTriedThisFrame = false;
    private Rect netFoundThisFrame = null;
    private final List<ScoreRow> scoresThisFrame = new ArrayList<>();

    /** One frame's rally score, as it became final. */
    private static final class ScoreRow {
        final long frameUs;
        final double score;
        final boolean on;

        ScoreRow(long frameUs, double score, boolean on) {
            this.frameUs = frameUs;
            this.score = score;
            this.on = on;
        }
    }
    private final double[] blobRow = new double[8 + BlobSvm.N_FEATURES];

    // ---------------------------------------------------------------- run state

    private boolean resetRequested = false;

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

    /** A pipeline that logs nothing. */
    public FramePipeline() {
        this(null);
    }

    /** @param log where to describe every frame; null for no logging */
    public FramePipeline(RunLog log) {
        this.log = log;
        this.rallyDetector = newRallyDetector();
        svmBlobDetector.setRecording(log != null);
    }

    /**
     * Runs one frame through detection, tracking and rally detection, and
     * draws the display.
     *
     * @param yuv         CV_8UC3 Y,U,V frame at the working resolution
     * @param timestampUs microseconds since this run's first frame
     * @param sourceUs    the frame's time on its source's own clock (camera
     *                    sensor time, or the video's PTS); only logged
     * @param inputRange  the frame's sample range before any range expansion;
     *                    only logged, may be null
     * @param displayOut  written with this frame in RGB, plus the net box
     * @return rallies confirmed on this frame, usually none. A rally is
     *         confirmed a few seconds after it ends, so its times are well in the past.
     */
    public List<RallyDetector.Rally> process(Mat yuv, long timestampUs, long sourceUs,
                                             YuvRange.Stats inputRange, Mat displayOut) {
        long startNs = System.nanoTime();
        int index = frameIndex++;
        if (index == 0) {
            warmStartUs = timestampUs;
        }
        if (resetRequested) {
            resetRequested = false;
            applyReset(timestampUs);
        }

        // Detection: the colour box ANDed with the motion mask (shadows kept),
        // then every blob scored by the SVM; the candidates are those at or
        // above BlobSvm.THRESHOLD.
        Mat colourMask = colourDetector.processFrame(yuv);
        Mat motionMask = motionDetector.processFrame(yuv);
        Core.bitwise_and(colourMask, motionMask, ballMask);
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

        List<RallyDetector.Rally> rallies = drainRallies();
        if (log != null) {
            logFrame(index, timestampUs, sourceUs, (System.nanoTime() - startNs) / 1e6,
                    inputRange, colourMask, motionMask, candidates, rallies);
        }
        return rallies;
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
        List<RallyDetector.Rally> rallies = drainRallies();
        if (log != null) {
            Json j = new Json().obj().field("type", "finish").field("frames", frameIndex);
            j.key("scores");
            appendScores(j);
            appendRallies(j, rallies);
            log.write(j.end().toString());
        }
        return rallies;
    }

    /** Asks for a background reset, applied at the start of the next frame. Pipeline thread only. */
    public void requestReset() {
        resetRequested = true;
    }

    /** Frees every native Mat this pipeline owns. */
    public void release() {
        motionDetector.release();
        colourDetector.release();
        svmBlobDetector.release();
        ballMask.release();
        background.release();
        shadowMask.release();
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
            netTriedThisFrame = true;

            Rect box = netDetector.detect(background);
            netFoundThisFrame = box;
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

    // ---------------------------------------------------------------- run log

    /** A rally detector that, when logging, reports each frame's score into this frame's line. */
    private RallyDetector newRallyDetector() {
        RallyDetector r = new RallyDetector();
        if (log != null) {
            r.setScoreListener((frameUs, score, on) ->
                    scoresThisFrame.add(new ScoreRow(frameUs, score, on)));
        }
        return r;
    }

    /**
     * One {@code frame} line. Column layouts are given in the run header
     * written by the caller; see {@link #describeColumns}.
     */
    private void logFrame(int index, long tUs, long sourceUs, double ms, YuvRange.Stats inputRange,
                          Mat colourMask, Mat motionMask,
                          List<BallCandidate> candidates, List<RallyDetector.Rally> rallies) {
        int moving = Core.countNonZero(motionMask);
        Core.inRange(motionMask, new Scalar(127), new Scalar(127), shadowMask);
        int shadow = Core.countNonZero(shadowMask);

        Json j = new Json().obj()
                .field("type", "frame")
                .field("f", index)
                .field("t", tUs)
                .field("src", sourceUs)
                .field("ms", ms, 2);
        if (inputRange != null) {
            String[] names = {"in_y", "in_u", "in_v"};
            for (int c = 0; c < 3; c++) {
                j.key(names[c]).arr();
                for (int v : inputRange.channel[c]) {
                    j.num(v);
                }
                j.endArr();
            }
        }
        j.field("motion", moving - shadow)
                .field("shadow", shadow)
                .field("colour", Core.countNonZero(colourMask))
                .field("ball_px", Core.countNonZero(ballMask))
                .field("components", svmBlobDetector.lastComponents())
                .field("in_range", svmBlobDetector.lastInRange())
                .field("static", svmBlobDetector.lastStatic());

        j.key("blobs").arr();
        for (int b = 0; b < svmBlobDetector.lastScored(); b++) {
            svmBlobDetector.lastScoredBlob(b, blobRow);
            j.arr();
            j.num(blobRow[0], 2).num(blobRow[1], 2);
            for (int k = 2; k <= 6; k++) {
                j.num((long) blobRow[k]);
            }
            j.num(blobRow[7], 4);
            for (int k = 8; k < blobRow.length; k++) {
                j.num(blobRow[k], 4);
            }
            j.endArr();
        }
        j.endArr();

        j.key("cand").arr();
        for (BallCandidate c : candidates) {
            j.arr().num(c.x, 2).num(c.y, 2).num(c.pBall, 4).endArr();
        }
        j.endArr();

        int seeds = 0, stubs2 = 0, stubs3 = 0;
        j.key("tracks").arr();
        for (Track t : tracker.tracks()) {
            int n = t.size();
            if (n == 1) { seeds++; continue; }
            if (n == 2) { stubs2++; continue; }
            if (n == 3) { stubs3++; continue; }
            Track.Point last = t.points().get(n - 1);
            Track.Shape s = t.shape();
            j.arr().num(n).bool(t.isConfirmed())
                    .num(t.points().get(0).frameIndex).num(last.frameIndex)
                    .num(last.x, 2).num(last.y, 2)
                    .num(s.rms, 3).num(s.curvature, 1).num(s.span, 1).num(s.speed, 1)
                    .endArr();
        }
        j.endArr();
        j.field("seeds", seeds).field("stubs2", stubs2).field("stubs3", stubs3);

        BallTracker.Output out = tracker.latest();
        j.key("out");
        if (out == null) {
            j.nul();
        } else {
            j.arr().num(out.frameIndex).num(out.tUs).bool(out.hasBall)
                    .num(out.x, 2).num(out.y, 2).endArr();
        }

        j.key("scores");
        appendScores(j);

        j.key("net");
        appendRect(j, cachedNetBox);
        if (netTriedThisFrame) {
            j.key("net_try");
            appendRect(j, netFoundThisFrame);
        }
        if (resetThisFrame) {
            j.field("reset", true);
        }
        appendRallies(j, rallies);
        log.write(j.end().toString());

        netTriedThisFrame = false;
        netFoundThisFrame = null;
        resetThisFrame = false;
    }

    /** The scores that became final since the last line, as [frameUs, score, on] rows; then clears them. */
    private void appendScores(Json j) {
        j.arr();
        for (ScoreRow r : scoresThisFrame) {
            j.arr().num(r.frameUs).num(r.score, 4).bool(r.on).endArr();
        }
        j.endArr();
        scoresThisFrame.clear();
    }

    private void appendRallies(Json j, List<RallyDetector.Rally> rallies) {
        if (rallies.isEmpty()) {
            return;
        }
        j.key("rallies").arr();
        for (RallyDetector.Rally r : rallies) {
            j.arr().num(r.startUs).num(r.endUs).endArr();
        }
        j.endArr();
    }

    private static void appendRect(Json j, Rect r) {
        if (r == null) {
            j.nul();
        } else {
            j.arr().num(r.x).num(r.y).num(r.width).num(r.height).endArr();
        }
    }

    /**
     * The start of a run's header line: source, log name, start time and the
     * column layouts. The caller adds its own fields, then {@code end()}s it.
     */
    public static Json header(String source, RunLog log) {
        String started = new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ",
                java.util.Locale.US).format(new java.util.Date());
        Json j = new Json().obj()
                .field("type", "run")
                .field("source", source)
                .field("log", log.name())
                .field("started", started)
                .field("wall_ms", System.currentTimeMillis());
        describeColumns(j);
        return j;
    }

    /**
     * Column names for the arrays in a {@code frame} line, for the run
     * header, so a reader does not have to know this code.
     */
    public static void describeColumns(Json j) {
        j.key("blob_cols").arr().str("x").str("y").str("area").str("left").str("top")
                .str("w").str("h").str("p");
        for (String f : BlobSvm.FEATURES) {
            j.str(f);
        }
        j.endArr();
        j.key("cand_cols").arr().str("x").str("y").str("p").endArr();
        j.key("track_cols").arr().str("points").str("confirmed").str("first_f").str("last_f")
                .str("x").str("y").str("rms").str("curvature").str("span").str("speed").endArr();
        j.key("out_cols").arr().str("f").str("t").str("has_ball").str("x").str("y").endArr();
        j.key("score_cols").arr().str("frame_t").str("score").str("rally_on").endArr();
        j.key("rect_cols").arr().str("x").str("y").str("w").str("h").endArr();
        j.key("range_cols").arr().str("min").str("p1").str("p99").str("max")
                .str("below_nominal").str("above_nominal").endArr();
        j.field("range_note", "in_y/in_u/in_v: the input frame at the working resolution, before any"
                + " range expansion; nominal range Y " + YuvRange.LUMA_LO + "-" + YuvRange.LUMA_HI
                + ", U/V " + YuvRange.CHROMA_LO + "-" + YuvRange.CHROMA_HI);
        j.field("svm_threshold", BlobSvm.THRESHOLD, 3);
        j.key("box_lo").arr().num((long) ColourDetector.LO.val[0]).num((long) ColourDetector.LO.val[1])
                .num((long) ColourDetector.LO.val[2]).endArr();
        j.key("box_hi").arr().num((long) ColourDetector.HI.val[0]).num((long) ColourDetector.HI.val[1])
                .num((long) ColourDetector.HI.val[2]).endArr();
        j.field("tracker_delay_frames", BallTracker.DELAY);
    }

    // ---------------------------------------------------------------- reset

    /** Clears every model that describes the old framing; called on the pipeline thread. */
    private void applyReset(long timestampUs) {
        // Close the old framing's rallies first; anything confirmed goes out
        // with this frame's result.
        rallyDetector.finish();
        collectRallies();
        rallyDetector = newRallyDetector();
        resetThisFrame = true;
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
