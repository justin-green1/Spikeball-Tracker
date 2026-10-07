package com.justin.spikeballtracker.source;

import android.content.Context;
import android.net.Uri;
import android.util.Log;

import com.justin.spikeballtracker.record.DetectionLog;
import com.justin.spikeballtracker.record.FileRally;
import com.justin.spikeballtracker.record.RallyFootage;
import com.justin.spikeballtracker.tracking.BallTracker;
import com.justin.spikeballtracker.tracking.RallyDetector;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.vision.FramePipeline;
import com.justin.spikeballtracker.vision.MotionDetector;

import org.opencv.core.Mat;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Plays a recorded clip through the pipeline at its true frame rate.
 *
 * <p>Every clip writes its {@link DetectionLog}s (narrow and wide candidates,
 * ball, rallies) into one run folder under {@code Documents/SpikeballTracker/}.
 *
 * <p>A confirmed rally's footage is a {@link FileRally}: a time range of the
 * clip itself, cut out only if it is saved.
 */
public final class VideoPlayer {

    private static final String TAG = "VideoPlayer";

    private final Context appContext;
    private final FrameExtractor frameExtractor;

    /** One clip at a time; a task holds this thread for the whole clip. */
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private volatile Future<?> activeTask = null;
    private volatile AtomicBoolean activeCancelFlag = null;
    private volatile RallyListener rallyListener = null;

    public VideoPlayer(Context context) {
        this.appContext = context.getApplicationContext();
        this.frameExtractor = new FrameExtractor(context);
    }

    /** Told about each confirmed rally of a clip that plays to the end; null for none. */
    public void setRallyListener(RallyListener listener) {
        this.rallyListener = listener;
    }

    public void play(Uri videoUri, FrameSink sink) {
        stop();

        final AtomicBoolean cancelled = new AtomicBoolean(false);
        activeCancelFlag = cancelled;

        activeTask = executor.submit(() -> {
            FramePipeline pipeline = new FramePipeline();
            Mat displayMat = new Mat();
            PlaybackClock clock = new PlaybackClock();
            final int[] candidateCounts = new int[2];   // narrow, wide; for the end-of-clip log line

            final AtomicBoolean firstFrame = new AtomicBoolean(true);
            // Set on the first frame, once the clip's colour info is known.
            final AtomicReference<Logs> logs = new AtomicReference<>(null);

            try {
                frameExtractor.streamFrames(videoUri, (frame, timestampUs) -> {
                    if (cancelled.get() || Thread.currentThread().isInterrupted()) {
                        return false;   // stop requested: leave the decode loop
                    }

                    if (firstFrame.getAndSet(false)) {
                        logs.set(new Logs(videoUri, frame));
                    }

                    FramePipeline.Result r = pipeline.process(frame, timestampUs, displayMat);
                    candidateCounts[0] += r.narrowCandidates.size();
                    candidateCounts[1] += r.wideCandidates.size();

                    // For video the pipeline's timestamp is the PTS itself.
                    Logs l = logs.get();
                    l.narrow.write(r.frameIndex, timestampUs, timestampUs,
                            r.narrowCandidates, r.netBox);
                    l.wide.write(r.frameIndex, timestampUs, timestampUs,
                            r.wideCandidates, r.netBox);
                    l.ball.writeBall(r.ball);
                    reportRallies(r.newRallies, r.frameIndex, l.rallies, videoUri, rallyListener);

                    sink.onDisplayFrame(displayMat);
                    return clock.waitFor(timestampUs);
                });
            } finally {
                Logs l = logs.get();

                // The last rally is only confirmed once the clip is over. A
                // stopped clip still logs it, but tells no one.
                try {
                    reportRallies(pipeline.finish(), -1, l == null ? null : l.rallies,
                            videoUri, cancelled.get() ? null : rallyListener);
                } catch (Exception e) {
                    Log.e(TAG, "rally finish failed", e);
                }

                if (l != null) {
                    l.close();
                }
                Log.d(TAG, "narrow candidates: " + candidateCounts[0]
                        + ", wide candidates: " + candidateCounts[1]
                        + ", worst playback lag: " + clock.worstLagMs() + "ms");
                pipeline.release();
                displayMat.release();
                // Clear the interrupt flag so the next task on this single-thread
                // executor doesn't inherit it.
                Thread.interrupted();
            }
        });
    }

    public void stop() {
        AtomicBoolean flag = activeCancelFlag;
        Future<?> task = activeTask;

        if (flag != null) {
            flag.set(true);   // the frame listener checks this and returns false
        }
        if (task != null && !task.isDone()) {
            Log.d(TAG, "stopping active video playback");
            task.cancel(true);   // also interrupt, in case it is asleep between frames
        }

        activeCancelFlag = null;
        activeTask = null;
    }

    public void release() {
        stop();
        executor.shutdownNow();
        frameExtractor.release();
    }

    /** Logs each rally and hands it, with its footage range, to the listener if there is one. */
    private void reportRallies(List<RallyDetector.Rally> rallies, int frameIndex,
                               DetectionLog log, Uri videoUri, RallyListener listener) {
        for (RallyDetector.Rally rally : rallies) {
            Log.i(TAG, String.format(Locale.US, "rally confirmed: %.2f s to %.2f s",
                    rally.startUs / 1e6, rally.endUs / 1e6));
            if (log != null) {
                log.writeRally(rally, frameIndex);
            }
            if (listener != null) {
                RallyFootage footage = new FileRally(appContext, videoUri,
                        Math.max(0L, rally.startUs - RallyFootage.BEFORE_US),
                        rally.endUs + RallyFootage.AFTER_US);
                listener.onRally(rally, footage);
            }
        }
    }

    /** One clip's four logs, sharing one run folder. Playback thread only. */
    private final class Logs {
        final DetectionLog narrow;
        final DetectionLog wide;
        final DetectionLog ball;
        final DetectionLog rallies;

        Logs(Uri videoUri, Mat firstFrame) {
            String clipName = videoUri.getLastPathSegment();
            String runId = DetectionLog.newRunId(clipName);
            FrameExtractor.ColourInfo colour = frameExtractor.colourInfo();

            DetectionLog.Meta m = new DetectionLog.Meta();
            m.source = "video";
            m.clipName = clipName;
            m.frameWidth = firstFrame.cols();
            m.frameHeight = firstFrame.rows();
            m.colourRange = colour.range;
            m.colourStandard = colour.standard;
            m.expandToFullRange = colour.expandToFullRange;
            m.knnHistory = MotionDetector.history();
            m.knnDist2Threshold = MotionDetector.dist2Threshold();
            m.knnDetectShadows = MotionDetector.detectShadows();
            m.knnPreBlurKernel = MotionDetector.preBlurKernel();
            narrow = new DetectionLog(appContext, runId, m, "narrow");
            wide = new DetectionLog(appContext, runId, m, "wide");

            DetectionLog.Meta ballMeta = basicMeta(clipName, firstFrame);
            ballMeta.note = "ball tracker output, emitted " + BallTracker.DELAY
                    + " frames late; blank BallX,BallY = no ball; last " + BallTracker.DELAY
                    + " frames of a clip are never emitted";
            ball = new DetectionLog(appContext, runId, ballMeta, "ball", DetectionLog.BALL_HEADER);

            DetectionLog.Meta rallyMeta = basicMeta(clipName, firstFrame);
            rallyMeta.note = "one row per confirmed rally; times are the clip's presentation time";
            rallies = new DetectionLog(appContext, runId, rallyMeta, "rallies",
                    DetectionLog.RALLY_HEADER);
        }

        private DetectionLog.Meta basicMeta(String clipName, Mat firstFrame) {
            DetectionLog.Meta m = new DetectionLog.Meta();
            m.source = "video";
            m.clipName = clipName;
            m.frameWidth = firstFrame.cols();
            m.frameHeight = firstFrame.rows();
            return m;
        }

        void close() {
            narrow.close();
            wide.close();
            ball.close();
            rallies.close();
        }
    }

    /**
     * Paces playback against each frame's own presentation timestamp rather
     * than a fixed per-frame delay. A fixed sleep is added on top of however
     * long processing took, so the video always runs slower than real time and
     * ignores its true frame rate.
     *
     * <p>Confined to the playback thread, so plain fields are enough.
     */
    private static final class PlaybackClock {
        private long firstTimestampUs = -1L;
        private long startNs = 0L;
        private long worstLagMs = 0L;

        /** @return false if the wait was interrupted and playback should stop. */
        boolean waitFor(long timestampUs) {
            if (firstTimestampUs < 0) {
                firstTimestampUs = timestampUs;
                startNs = System.nanoTime();
            }

            long targetMs = (timestampUs - firstTimestampUs) / 1_000L;
            long actualMs = (System.nanoTime() - startNs) / 1_000_000L;
            long sleepMs = targetMs - actualMs;

            if (sleepMs > 0) {
                try {
                    Thread.sleep(sleepMs);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return false;
                }
            } else if (-sleepMs > worstLagMs) {
                // We cannot speed the video up, only record how far behind it got.
                worstLagMs = -sleepMs;
            }
            return true;
        }

        long worstLagMs() {
            return worstLagMs;
        }
    }
}
