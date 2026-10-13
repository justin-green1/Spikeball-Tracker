package com.justin.spikeballtracker.source;

import android.content.Context;
import android.net.Uri;
import android.util.Log;

import com.justin.spikeballtracker.record.FileRally;
import com.justin.spikeballtracker.record.RallyFootage;
import com.justin.spikeballtracker.tracking.RallyDetector;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.vision.FramePipeline;

import org.opencv.core.Mat;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Plays a recorded clip through the pipeline at its true frame rate.
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

            try {
                frameExtractor.streamFrames(videoUri, (frame, timestampUs) -> {
                    if (cancelled.get() || Thread.currentThread().isInterrupted()) {
                        return false;   // stop requested: leave the decode loop
                    }

                    // For video the pipeline's timestamp is the PTS itself.
                    List<RallyDetector.Rally> rallies =
                            pipeline.process(frame, timestampUs, displayMat);
                    reportRallies(rallies, videoUri, rallyListener);

                    sink.onDisplayFrame(displayMat);
                    return clock.waitFor(timestampUs);
                });
            } finally {
                // The last rally is only confirmed once the clip is over. A
                // stopped clip tells no one about it.
                if (!cancelled.get()) {
                    try {
                        reportRallies(pipeline.finish(), videoUri, rallyListener);
                    } catch (Exception e) {
                        Log.e(TAG, "rally finish failed", e);
                    }
                }
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

    /** Hands each rally, with its footage range, to the listener if there is one. */
    private void reportRallies(List<RallyDetector.Rally> rallies, Uri videoUri,
                               RallyListener listener) {
        if (listener == null) {
            return;
        }
        for (RallyDetector.Rally rally : rallies) {
            RallyFootage footage = new FileRally(appContext, videoUri,
                    Math.max(0L, rally.startUs - RallyFootage.BEFORE_US),
                    rally.endUs + RallyFootage.AFTER_US);
            listener.onRally(rally, footage);
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

        /** @return false if the wait was interrupted and playback should stop. */
        boolean waitFor(long timestampUs) {
            if (firstTimestampUs < 0) {
                firstTimestampUs = timestampUs;
                startNs = System.nanoTime();
            }

            long targetMs = (timestampUs - firstTimestampUs) / 1_000L;
            long actualMs = (System.nanoTime() - startNs) / 1_000_000L;
            long sleepMs = targetMs - actualMs;

            // Behind schedule: no wait. The video cannot be sped up to catch up.
            if (sleepMs > 0) {
                try {
                    Thread.sleep(sleepMs);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return false;
                }
            }
            return true;
        }
    }
}
