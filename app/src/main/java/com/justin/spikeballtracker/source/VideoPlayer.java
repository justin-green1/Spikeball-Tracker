package com.justin.spikeballtracker.source;

import android.content.Context;
import android.net.Uri;
import android.util.Log;

import com.justin.spikeballtracker.record.DetectionLog;
import com.justin.spikeballtracker.tracking.BallCandidate;
import com.justin.spikeballtracker.video.BackgroundExporter;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.vision.FramePipeline;
import com.justin.spikeballtracker.vision.MotionDetector;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Plays a recorded clip through the pipeline at its true frame rate.
 *
 * <p>Owns the extractor and the background exporter, because it is the only
 * path that produces exportable plates.
 *
 * <p>Every clip also writes a {@link DetectionLog}, the per-frame ball
 * candidates and net box, into the same run folder as its plates.
 */
public final class VideoPlayer {

    private static final String TAG = "VideoPlayer";

    private final Context appContext;
    private final FrameExtractor frameExtractor;
    private final BackgroundExporter backgroundExporter;

    /** One clip at a time; a task holds this thread for the whole clip. */
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    /** Set by the export button, consumed on the processing thread. */
    private final AtomicBoolean exportRequested = new AtomicBoolean(false);

    private volatile Future<?> activeTask = null;
    private volatile AtomicBoolean activeCancelFlag = null;

    public VideoPlayer(Context context) {
        this.appContext = context.getApplicationContext();
        this.frameExtractor = new FrameExtractor(context);
        this.backgroundExporter = new BackgroundExporter(context);
    }

    public void play(Uri videoUri, FrameSink sink) {
        stop();

        final AtomicBoolean cancelled = new AtomicBoolean(false);
        activeCancelFlag = cancelled;

        activeTask = executor.submit(() -> {
            FramePipeline pipeline = new FramePipeline();
            Mat displayMat = new Mat();
            PlaybackClock clock = new PlaybackClock();
            List<BallCandidate> allCandidates = new ArrayList<>();
            final AtomicBoolean firstFrame = new AtomicBoolean(true);
            // Set on the first frame, once the export run (and so the folder) exists.
            final AtomicReference<DetectionLog> log = new AtomicReference<>(null);

            try {
                frameExtractor.streamFrames(videoUri, (frame, timestampUs) -> {
                    if (cancelled.get() || Thread.currentThread().isInterrupted()) {
                        return false;   // stop requested: leave the decode loop
                    }

                    if (firstFrame.getAndSet(false)) {
                        beginExportRun(videoUri, frame);
                        log.set(newDetectionLog(videoUri, frame));
                    }

                    boolean manual = exportRequested.getAndSet(false);
                    boolean plateWanted =
                            manual || backgroundExporter.dueForBackground(timestampUs);

                    FramePipeline.Result r =
                            pipeline.process(frame, timestampUs, displayMat, plateWanted);
                    allCandidates.addAll(r.candidates);

                    // For video the pipeline's timestamp is the PTS itself.
                    log.get().write(r.frameIndex, timestampUs, timestampUs,
                            r.candidates, r.netBox);

                    // offerFrame no longer writes anything, so it can run after
                    // process(); that lets the pipeline own frame numbering.
                    backgroundExporter.offerFrame(frame, r.frameIndex, timestampUs);
                    if (r.backgroundPlate != null) {
                        backgroundExporter.offerBackground(
                                r.backgroundPlate, r.frameIndex, timestampUs);
                        if (manual) {
                            backgroundExporter.exportPlate(
                                    r.backgroundPlate, "manual", r.frameIndex, timestampUs);
                        }
                    }

                    sink.onDisplayFrame(displayMat);
                    return clock.waitFor(timestampUs);
                });
            } finally {
                DetectionLog l = log.get();
                if (l != null) {
                    l.close();
                }
                backgroundExporter.endRun();
                Log.d(TAG, "total candidates: " + allCandidates.size()
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

    /** @return false when no run is active, so there is nothing to export. */
    public boolean requestBackgroundExport() {
        if (!backgroundExporter.isRunActive()) {
            return false;
        }
        exportRequested.set(true);
        return true;
    }

    public void release() {
        stop();
        executor.shutdownNow();
        frameExtractor.release();
    }

    private void beginExportRun(Uri videoUri, Mat firstFrame) {
        // readColourInfo has run by now, so the provenance recorded is the one
        // actually in force for this clip.
        FrameExtractor.ColourInfo colour = frameExtractor.colourInfo();
        BackgroundExporter.Provenance p = new BackgroundExporter.Provenance();
        p.clipName = videoUri.getLastPathSegment();
        p.colourRange = colour.range;
        p.colourStandard = colour.standard;
        p.expandToFullRange = colour.expandToFullRange;
        p.targetWidth = firstFrame.cols();
        p.targetHeight = firstFrame.rows();
        p.knnHistory = MotionDetector.history();
        p.knnDist2Threshold = MotionDetector.dist2Threshold();
        p.knnDetectShadows = MotionDetector.detectShadows();
        p.knnPreBlurKernel = MotionDetector.preBlurKernel();
        backgroundExporter.beginRun(p);
    }

    /** Called after beginExportRun, so it can share the run's folder. */
    private DetectionLog newDetectionLog(Uri videoUri, Mat firstFrame) {
        FrameExtractor.ColourInfo colour = frameExtractor.colourInfo();
        DetectionLog.Meta m = new DetectionLog.Meta();
        m.source = "video";
        m.clipName = videoUri.getLastPathSegment();
        m.frameWidth = firstFrame.cols();
        m.frameHeight = firstFrame.rows();
        m.colourRange = colour.range;
        m.colourStandard = colour.standard;
        m.expandToFullRange = colour.expandToFullRange;
        m.knnHistory = MotionDetector.history();
        m.knnDist2Threshold = MotionDetector.dist2Threshold();
        m.knnDetectShadows = MotionDetector.detectShadows();
        m.knnPreBlurKernel = MotionDetector.preBlurKernel();
        return new DetectionLog(appContext, backgroundExporter.runId(), m);
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