package com.justin.spikeballtracker.record;

import android.content.Context;
import android.media.MediaMuxer;
import android.util.Log;

import com.justin.spikeballtracker.BackgroundThread;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Holds the last rally's footage, from either source, and saves it to
 * {@code Movies/SpikeballTracker/} on request.
 *
 * <p>The footage is replaced only when a newer rally is kept; saving does not
 * clear it, so it can be saved more than once. It lives in RAM, so it is gone
 * when the app closes.
 *
 * <p>{@link #keep} and {@link #hasRally} are called on the background thread
 * (sources report rallies there). {@link #saveLast} may be called from any
 * thread; the save itself runs on the background thread, against the footage
 * held when it starts. While a save is writing (a fraction of a second for a
 * typical rally) the thread does nothing else, so the camera drops a few
 * analysis frames or the video pauses briefly.
 */
public final class RallySaver {

    private static final String TAG = "RallySaver";

    public interface SaveCallback {
        /** Called on the background thread. */
        void onSaved(String displayName, double seconds);

        /** Called on the background thread. */
        void onFailed(String reason);
    }

    private final Context appContext;
    private final BackgroundThread background;

    private RallyFootage last = null;   // background thread only

    public RallySaver(Context context, BackgroundThread background) {
        this.appContext = context.getApplicationContext();
        this.background = background;
    }

    /** Makes {@code footage} the last rally. A null (footage not captured) keeps the previous one. */
    public void keep(RallyFootage footage) {
        if (footage != null) {
            last = footage;
        }
    }

    /** True once any rally's footage has been kept. */
    public boolean hasRally() {
        return last != null;
    }

    /** Saves the last rally as {@code rally_<date-time>.mp4}. Returns at once. */
    public void saveLast(SaveCallback callback) {
        background.post(() -> {
            RallyFootage footage = last;
            if (footage == null) {
                callback.onFailed("no rally yet");
                return;
            }
            String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
            String name = "rally_" + stamp + ".mp4";
            try {
                long durationUs = write(footage, name);
                callback.onSaved(name, durationUs / 1_000_000.0);
            } catch (Exception e) {
                Log.e(TAG, "rally save failed", e);
                callback.onFailed(e.getMessage() != null ? e.getMessage() : e.toString());
            }
        });
    }

    private long write(RallyFootage footage, String name) throws Exception {
        Mp4File out = Mp4File.open(appContext, name);
        MediaMuxer muxer = null;
        boolean ok = false;
        try {
            muxer = out.newMuxer();
            long durationUs = footage.writeTo(muxer);
            ok = true;
            return durationUs;
        } finally {
            if (muxer != null) {
                try {
                    muxer.release();
                } catch (IllegalStateException e) {
                    Log.w(TAG, "muxer release failed", e);
                }
            }
            out.finish(ok);
        }
    }
}
