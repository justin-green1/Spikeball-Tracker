package com.justin.spikeballtracker.source;

import android.content.Context;
import android.net.Uri;
import android.os.SystemClock;
import android.util.Log;

import com.justin.spikeballtracker.BackgroundThread;
import com.justin.spikeballtracker.record.FileRally;
import com.justin.spikeballtracker.record.RallyFootage;
import com.justin.spikeballtracker.tracking.RallyDetector;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.vision.FramePipeline;

import org.opencv.core.Mat;

import java.util.List;

/**
 * Plays a recorded clip through the pipeline at its true frame rate.
 *
 * <p>Runs on the shared {@link BackgroundThread}, one frame per message:
 * decode a frame, run the pipeline, show it, then schedule the next frame
 * for the moment its presentation timestamp says it is due. Between frames
 * the thread is free for anything else. Stopping just removes the next
 * scheduled frame; since stop runs on the same thread, it can never land in
 * the middle of one.
 *
 * <p>A confirmed rally's footage is a {@link FileRally}: a time range of the
 * clip itself, cut out only if it is saved.
 *
 * <p>The public methods may be called from any thread; they post to the
 * background thread. Everything else runs only there.
 */
public final class VideoPlayer {

    private static final String TAG = "VideoPlayer";

    private final Context appContext;
    private final BackgroundThread background;
    private final FrameExtractor frameExtractor;

    // Background thread only.
    private Playback current = null;
    private RallyListener rallyListener = null;

    public VideoPlayer(Context context, BackgroundThread background) {
        this.appContext = context.getApplicationContext();
        this.background = background;
        this.frameExtractor = new FrameExtractor(context);
    }

    /** Told about each confirmed rally of a clip that plays to the end; null for none. */
    public void setRallyListener(RallyListener listener) {
        background.post(() -> rallyListener = listener);
    }

    /** Stops any clip playing and starts this one. */
    public void play(Uri videoUri, FrameSink sink) {
        background.post(() -> {
            end(false);
            if (!frameExtractor.open(videoUri)) {
                return;
            }
            current = new Playback(videoUri, sink);
            background.post(current);
        });
    }

    /** Stops the clip playing, if any. A stopped clip reports no rally still in progress. */
    public void stop() {
        background.post(() -> end(false));
    }

    /** Stops and frees everything. Posted, so it runs after anything already queued. */
    public void release() {
        background.post(() -> {
            end(false);
            frameExtractor.release();
        });
    }

    /**
     * Ends the current clip. Background thread only.
     *
     * @param finished true when the clip played to the end; only then is the
     *                 last rally, confirmed by finishing, reported
     */
    private void end(boolean finished) {
        Playback p = current;
        if (p == null) {
            return;
        }
        current = null;
        background.handler().removeCallbacks(p);
        if (finished) {
            try {
                reportRallies(p.pipeline.finish(), p.videoUri);
            } catch (Exception e) {
                Log.e(TAG, "rally finish failed", e);
            }
        }
        p.release();
        frameExtractor.close();
    }

    /** Hands each rally, with its footage range, to the listener if there is one. */
    private void reportRallies(List<RallyDetector.Rally> rallies, Uri videoUri) {
        if (rallyListener == null) {
            return;
        }
        for (RallyDetector.Rally rally : rallies) {
            RallyFootage footage = new FileRally(appContext, videoUri,
                    Math.max(0L, rally.startUs - RallyFootage.BEFORE_US),
                    rally.endUs + RallyFootage.AFTER_US);
            rallyListener.onRally(rally, footage);
        }
    }

    /**
     * One clip's pipeline and clock. Each run() handles one frame and
     * schedules the next.
     */
    private final class Playback implements Runnable {
        final Uri videoUri;
        final FrameSink sink;
        final FramePipeline pipeline = new FramePipeline();
        final Mat displayMat = new Mat();

        /**
         * Paces playback against each frame's own presentation timestamp
         * rather than a fixed per-frame delay. A fixed delay is added on top
         * of however long processing took, so the video would always run
         * slower than real time.
         */
        private long firstTimestampUs = -1L;
        private long startUptimeMs = 0L;

        Playback(Uri videoUri, FrameSink sink) {
            this.videoUri = videoUri;
            this.sink = sink;
        }

        @Override
        public void run() {
            if (current != this) {
                return;   // stopped after this frame was scheduled
            }
            try {
                Mat frame = frameExtractor.nextFrame();
                if (frame == null) {
                    end(true);   // end of the clip
                    return;
                }
                long timestampUs = frameExtractor.timestampUs();

                // For video the pipeline's timestamp is the PTS itself.
                reportRallies(pipeline.process(frame, timestampUs, displayMat), videoUri);
                sink.onDisplayFrame(displayMat);

                // The next frame is due once this one's time has come. Behind
                // schedule, it is due now: the video cannot speed up to catch up.
                // Never earlier than now, or a late frame would be queued ahead
                // of work already waiting (a stop, a save) and could starve it.
                long nowMs = SystemClock.uptimeMillis();
                if (firstTimestampUs < 0) {
                    firstTimestampUs = timestampUs;
                    startUptimeMs = nowMs;
                }
                long dueMs = startUptimeMs + (timestampUs - firstTimestampUs) / 1_000L;
                background.handler().postAtTime(this, Math.max(dueMs, nowMs));
            } catch (Exception e) {
                Log.e(TAG, "video frame failed; stopping", e);
                end(false);
            }
        }

        void release() {
            pipeline.release();
            displayMat.release();
        }
    }
}
