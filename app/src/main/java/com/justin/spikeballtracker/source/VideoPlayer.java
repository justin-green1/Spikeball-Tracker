package com.justin.spikeballtracker.source;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.util.Log;

import com.justin.spikeballtracker.WorkerThread;
import com.justin.spikeballtracker.log.Formats;
import com.justin.spikeballtracker.log.Json;
import com.justin.spikeballtracker.log.RunLog;
import com.justin.spikeballtracker.record.FileRally;
import com.justin.spikeballtracker.record.RallyFootage;
import com.justin.spikeballtracker.tracking.RallyDetector;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.vision.FramePipeline;

import org.opencv.core.Mat;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Plays a recorded clip through the pipeline at its true frame rate.
 *
 * <p>A clip runs as one job on the Analysis {@link WorkerThread}: a plain
 * decode-process-wait loop that holds the thread until the clip ends or is
 * stopped. The camera shares that thread, which is fine because the two
 * sources never run at the same time.
 *
 * <p>Stopping is the one thing that has to reach across threads, since a
 * posted "stop" would wait behind the clip it is meant to stop. Each clip
 * gets a number; {@link #play} and {@link #stop} move the current number on,
 * and a clip keeps going only while its number is still current. That one
 * atomic counter is the only shared state here.
 *
 * <p>A confirmed rally's footage is a {@link FileRally}: a time range of the
 * clip itself, cut out only if it is saved.
 */
public final class VideoPlayer {

    private static final String TAG = "VideoPlayer";

    /** Longest single sleep, so a stop is noticed within this even across a gap in the clip. */
    private static final long MAX_SLEEP_MS = 50;

    private final Context appContext;
    private final WorkerThread analysis;
    private final FrameExtractor frameExtractor;

    /** Bumped by every play() and stop(); a clip plays only while its number is current. */
    private final AtomicInteger currentClip = new AtomicInteger();

    private RallyListener rallyListener = null;   // Analysis thread only

    public VideoPlayer(Context context, WorkerThread analysis) {
        this.appContext = context.getApplicationContext();
        this.analysis = analysis;
        this.frameExtractor = new FrameExtractor(context);
    }

    /** Told about each confirmed rally of a clip that plays to the end, on the Analysis thread; null for none. */
    public void setRallyListener(RallyListener listener) {
        analysis.post(() -> rallyListener = listener);
    }

    /** Stops any clip playing and starts this one. */
    public void play(Uri videoUri, FrameSink sink) {
        int clip = currentClip.incrementAndGet();
        analysis.post(() -> playClip(clip, videoUri, sink));
    }

    /** Stops the clip playing, if any, within one frame. */
    public void stop() {
        currentClip.incrementAndGet();
    }

    /** Stops playback and frees the decoder's buffers once the clip has let go of the thread. */
    public void release() {
        stop();
        analysis.post(frameExtractor::release);
    }

    private boolean isCurrent(int clip) {
        return currentClip.get() == clip;
    }

    /** Runs a whole clip. Holds the Analysis thread until it ends or is stopped. */
    private void playClip(int clip, Uri videoUri, FrameSink sink) {
        if (!isCurrent(clip)) {
            return;   // stopped, or replaced by another clip, before it started
        }
        RunLog log = RunLog.open(appContext, "video");
        if (log != null) {
            log.write(FramePipeline.header("video", log)
                    .field("video", displayName(videoUri))
                    .field("uri", videoUri.toString())
                    .field("note", "src = t = the video's presentation time in us")
                    .end().toString());
        }
        FramePipeline pipeline = new FramePipeline(log);
        Mat displayMat = new Mat();
        PlaybackClock clock = new PlaybackClock(clip);
        int[] frames = {0};
        boolean completed = false;

        try {
            frameExtractor.streamFrames(videoUri, (frame, timestampUs) -> {
                if (!isCurrent(clip)) {
                    return false;   // stop requested: leave the decode loop
                }
                if (frames[0]++ == 0 && log != null) {
                    logStream(log);
                }
                // For video the pipeline's timestamp is the PTS itself.
                reportRallies(pipeline.process(frame, timestampUs, timestampUs,
                        frameExtractor.lastRange(), displayMat), videoUri);
                sink.onDisplayFrame(displayMat);
                return clock.waitFor(timestampUs);
            });

            // The last rally is only confirmed once the clip is over. A
            // stopped clip tells no one about it.
            if (isCurrent(clip)) {
                completed = true;
                reportRallies(pipeline.finish(), videoUri);
            }
        } catch (RuntimeException e) {
            Log.e(TAG, "video playback failed", e);
        } finally {
            pipeline.release();
            displayMat.release();
            if (log != null) {
                log.write(new Json().obj().field("type", "end")
                        .field("frames", frames[0])
                        .field("completed", completed).end().toString());
                log.close();
            }
        }
    }

    /** What the decoder actually delivers: size, strides, range handling and declared colour encoding. */
    private void logStream(RunLog log) {
        Json j = new Json().obj()
                .field("type", "stream")
                .field("geometry", frameExtractor.sourceGeometry())
                .field("range_expansion", frameExtractor.isExpandingRange());
        Formats.mediaFormat(j, "track", frameExtractor.trackFormat());
        Formats.mediaFormat(j, "decoder_output", frameExtractor.outputFormat());
        log.write(j.end().toString());
    }

    /** The file's name as the picker showed it, or null. */
    private String displayName(Uri uri) {
        try (Cursor c = appContext.getContentResolver().query(
                uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                return c.getString(0);
            }
        } catch (RuntimeException e) {
            Log.w(TAG, "could not read the video's name", e);
        }
        return null;
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
     * Paces playback against each frame's own presentation timestamp rather
     * than a fixed per-frame delay. A fixed sleep is added on top of however
     * long processing took, so the video always runs slower than real time and
     * ignores its true frame rate.
     */
    private final class PlaybackClock {
        private final int clip;
        private long firstTimestampUs = -1L;
        private long startNs = 0L;

        PlaybackClock(int clip) {
            this.clip = clip;
        }

        /** @return false if the clip was stopped while waiting. */
        boolean waitFor(long timestampUs) {
            if (firstTimestampUs < 0) {
                firstTimestampUs = timestampUs;
                startNs = System.nanoTime();
            }
            long targetMs = (timestampUs - firstTimestampUs) / 1_000L;

            // Behind schedule: no wait. The video cannot be sped up to catch up.
            while (isCurrent(clip)) {
                long sleepMs = targetMs - (System.nanoTime() - startNs) / 1_000_000L;
                if (sleepMs <= 0) {
                    return true;
                }
                try {
                    Thread.sleep(Math.min(sleepMs, MAX_SLEEP_MS));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return false;
                }
            }
            return false;
        }
    }
}
