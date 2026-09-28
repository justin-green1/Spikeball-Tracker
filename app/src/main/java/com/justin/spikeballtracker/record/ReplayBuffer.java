package com.justin.spikeballtracker.record;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

import androidx.annotation.NonNull;
import androidx.camera.core.Preview;
import androidx.camera.core.SurfaceRequest;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Keeps the last minute of camera footage in RAM and saves the last N seconds
 * of it as an MP4 on request.
 *
 * <p>How frames get here: a CameraX {@link Preview} use case is bound next to
 * the analysis stream, and its surface is the hardware H.264 encoder's input
 * surface. The camera writes straight into the encoder, so no pixels touch
 * Java or OpenCV. That keeps the recording at the full camera frame rate
 * and native 1280x720 however slow the detectors are, at the price of
 * the recorded frames being the camera's parallel stream rather than the exact
 * frames the analyser saw. Both streams come off the same sensor, on the same
 * clock.
 *
 * <p>The encoder's output goes into an {@link EncodedRing}. Saving remuxes a
 * keyframe-aligned slice of that ring into an MP4 with {@link MediaMuxer}. No
 * re-encode, so a save takes well under a second and costs no quality.
 *
 * <p>Threads:
 * <ul>
 *   <li><b>encoder thread</b> — surface requests, all MediaCodec callbacks,
 *       session teardown. Owns {@link #current} and {@link #live}.</li>
 *   <li><b>save thread</b> — one save at a time; only reads the ring.</li>
 * </ul>
 *
 * <p>Lifetime: the ring is kept when the camera stops, so the last footage
 * can still be saved afterwards. It is cleared when the next camera session
 * starts, because a new encoder's packets cannot be muxed with the old one's
 * stream headers.
 */
public final class ReplayBuffer {

    private static final String TAG = "ReplayBuffer";

    /** The longest clip a save can ask for. */
    public static final int MAX_SAVE_SECONDS = 60;

    /**
     * How much is actually held. The margin over MAX_SAVE_SECONDS covers
     * rounding the start back to a keyframe (up to one GOP) plus the time the
     * save itself takes, during which the oldest packets keep ageing out.
     */
    private static final long RETAIN_US = (MAX_SAVE_SECONDS + 6) * 1_000_000L;

    private static final String MIME = MediaFormat.MIMETYPE_VIDEO_AVC;
    private static final int BIT_RATE = 5_000_000;          // ~38 MB per minute at 720p
    private static final int FRAME_RATE = 30;               // a hint to the encoder; timestamps rule
    private static final int I_FRAME_INTERVAL_S = 1;        // save start granularity
    private static final int MAX_FPS_FOR_SLOTS = 60;        // sizes the packet metadata ring
    private static final double BYTE_HEADROOM = 1.25;       // VBR overshoots on busy scenes

    private static final String FOLDER = "SpikeballTracker";

    public interface SaveCallback {
        /** Called on the save thread. */
        void onSaved(String displayName, double seconds);

        /** Called on the save thread. */
        void onFailed(String reason);
    }

    private final Context appContext;
    private final HandlerThread encoderThread;
    private final Handler encoderHandler;
    private final Executor encoderExecutor;
    private final ExecutorService saveExecutor = Executors.newSingleThreadExecutor();

    /** Allocated on first use, so an app that never opens the camera never pays ~50 MB. */
    private volatile EncodedRing ring = null;

    /** Stream headers (csd-0/csd-1) for what is in the ring. Guarded by the ring's lock. */
    private MediaFormat outputFormat = null;
    private volatile int rotationDegrees = 0;

    private Session current = null;                        // encoder thread only
    private final List<Session> live = new ArrayList<>();  // encoder thread only

    public ReplayBuffer(Context context) {
        this.appContext = context.getApplicationContext();
        this.encoderThread = new HandlerThread("ReplayEncoder");
        this.encoderThread.start();
        this.encoderHandler = new Handler(encoderThread.getLooper());
        this.encoderExecutor = encoderHandler::post;
    }

    /** Routes the preview use case's frames into the encoder. Call before binding. */
    public void attachTo(Preview preview) {
        preview.setSurfaceProvider(encoderExecutor, this::onSurfaceRequested);
    }

    /** Seconds of footage currently held; 0 before the first camera session. */
    public double bufferedSeconds() {
        EncodedRing r = ring;
        return r == null ? 0.0 : r.bufferedUs() / 1_000_000.0;
    }

    /**
     * Saves the last {@code seconds} (capped at what is held) to
     * {@code Movies/SpikeballTracker/}. Returns at once; the callback reports
     * the outcome from the save thread.
     */
    public void save(int seconds, SaveCallback callback) {
        int clamped = Math.max(1, Math.min(seconds, MAX_SAVE_SECONDS));
        saveExecutor.execute(() -> {
            try {
                Saved s = saveBlocking(clamped * 1_000_000L);
                Log.i(TAG, "saved " + s.name + " (" + s.seconds + " s, "
                        + s.packets + " packets)");
                callback.onSaved(s.name, s.seconds);
            } catch (Exception e) {
                Log.e(TAG, "replay save failed", e);
                callback.onFailed(e.getMessage() != null ? e.getMessage() : e.toString());
            }
        });
    }

    /**
     * Stops any encoder and the worker threads. A save already running is
     * allowed to finish: {@code shutdown}, not {@code shutdownNow}.
     */
    public void release() {
        encoderHandler.post(() -> {
            for (Session s : new ArrayList<>(live)) {
                s.end();
            }
            current = null;
        });
        encoderThread.quitSafely();
        saveExecutor.shutdown();
    }

    // --- encoder side (encoder thread) ---------------------------------------

    private void onSurfaceRequested(@NonNull SurfaceRequest request) {
        EncodedRing r = ensureRing();
        if (r == null) {
            request.willNotProvideSurface();
            return;
        }

        Size res = request.getResolution();
        MediaFormat format = MediaFormat.createVideoFormat(MIME, res.getWidth(), res.getHeight());
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
        format.setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE);
        format.setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL_S);

        Session session;
        try {
            MediaCodec codec = MediaCodec.createEncoderByType(MIME);
            session = new Session(codec);
            codec.setCallback(session, encoderHandler);    // must precede configure
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            session.surface = codec.createInputSurface();  // after configure, before start
            codec.start();
        } catch (Exception e) {
            Log.e(TAG, "encoder setup failed for " + res + "; replay disabled this session", e);
            request.willNotProvideSurface();
            return;
        }

        // A new encoder means new stream headers, so the old packets cannot
        // follow them into one file.
        synchronized (r) {
            r.clear();
            outputFormat = null;
        }
        rotationDegrees = 0;
        current = session;
        live.add(session);

        request.setTransformationInfoListener(encoderExecutor, info -> {
            if (current == session) {
                rotationDegrees = info.getRotationDegrees();
            }
        });

        // The result arrives once the camera has stopped writing to the surface,
        // which is the only safe moment to tear the encoder down.
        request.provideSurface(session.surface, encoderExecutor, result -> {
            Log.d(TAG, "preview surface finished, code " + result.getResultCode());
            session.end();
            if (current == session) {
                current = null;
            }
        });

        Log.d(TAG, "encoder session started at " + res);
    }

    private EncodedRing ensureRing() {
        if (ring != null) {
            return ring;
        }
        double retainS = RETAIN_US / 1_000_000.0;
        int capacity = (int) (BIT_RATE / 8.0 * retainS * BYTE_HEADROOM);
        int slots = (int) Math.ceil(retainS) * MAX_FPS_FOR_SLOTS;
        try {
            ring = new EncodedRing(capacity, slots, RETAIN_US);
            Log.i(TAG, "ring allocated: " + (capacity / (1024 * 1024)) + " MB, "
                    + slots + " packet slots");
        } catch (OutOfMemoryError e) {
            Log.e(TAG, "not enough heap for a " + (capacity / (1024 * 1024))
                    + " MB replay ring; replay disabled");
        }
        return ring;
    }

    /** One encoder, living exactly as long as one preview surface. */
    private final class Session extends MediaCodec.Callback {
        final MediaCodec codec;
        Surface surface;
        private boolean ended = false;

        Session(MediaCodec codec) {
            this.codec = codec;
        }

        @Override
        public void onInputBufferAvailable(@NonNull MediaCodec codec, int index) {
            // Surface input: the camera fills the encoder, never us.
        }

        @Override
        public void onOutputBufferAvailable(@NonNull MediaCodec codec, int index,
                                            @NonNull MediaCodec.BufferInfo info) {
            if (ended) {
                return;   // a callback already queued when the codec was stopped
            }
            try {
                boolean config = (info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0;
                if (current == this && info.size > 0 && !config) {
                    ByteBuffer buf = codec.getOutputBuffer(index);
                    if (buf != null) {
                        buf.position(info.offset);
                        buf.limit(info.offset + info.size);
                        boolean key = (info.flags & MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0;
                        if (!ring.add(buf, info.presentationTimeUs, key)) {
                            Log.w(TAG, "packet of " + info.size + " B larger than the ring; dropped");
                        }
                    }
                }
                codec.releaseOutputBuffer(index, false);
            } catch (IllegalStateException e) {
                Log.w(TAG, "output buffer after codec stopped", e);
            }
        }

        @Override
        public void onOutputFormatChanged(@NonNull MediaCodec codec, @NonNull MediaFormat format) {
            // Arrives before the first packet and carries the SPS/PPS the muxer needs.
            if (current == this) {
                EncodedRing r = ring;
                synchronized (r) {
                    outputFormat = format;
                }
                Log.d(TAG, "encoder output format: " + format);
            }
        }

        @Override
        public void onError(@NonNull MediaCodec codec, @NonNull MediaCodec.CodecException e) {
            Log.e(TAG, "encoder error; ending this session", e);
            end();
            if (current == this) {
                current = null;
            }
        }

        void end() {
            if (ended) {
                return;
            }
            ended = true;
            live.remove(this);
            try {
                codec.stop();
            } catch (IllegalStateException e) {
                Log.w(TAG, "codec stop failed", e);
            }
            codec.release();
            if (surface != null) {
                surface.release();
            }
        }
    }

    // --- save side (save thread) ---------------------------------------------

    private static final class Saved {
        String name;
        double seconds;
        int packets;
    }

    private Saved saveBlocking(long durationUs) throws IOException {
        EncodedRing r = ring;
        if (r == null) {
            throw new IOException("nothing recorded yet");
        }

        // Format and plan must describe the same session, so take both under the
        // same lock that a new session's clear() takes.
        MediaFormat format;
        EncodedRing.Plan plan;
        synchronized (r) {
            format = outputFormat;
            plan = r.plan(durationUs);
        }
        if (format == null || plan == null) {
            throw new IOException("nothing buffered yet");
        }
        int rotation = rotationDegrees;

        String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
        long seconds = Math.round(plan.durationUs() / 1_000_000.0);
        String name = "replay_" + stamp + "_" + seconds + "s.mp4";

        ByteBuffer scratch = ByteBuffer.allocateDirect(Math.max(r.maxPacketSize(), 64 * 1024));
        EncodedRing.Packet packet = new EncodedRing.Packet();
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();

        Output out = Output.open(appContext, name);
        MediaMuxer muxer = null;
        boolean ok = false;
        int written = 0;
        try {
            muxer = out.newMuxer();
            muxer.setOrientationHint(rotation);
            int track = muxer.addTrack(format);
            muxer.start();

            for (long seq = plan.startSeq; seq < plan.endSeq; seq++) {
                if (!r.copy(seq, scratch, packet)) {
                    // Evicted underneath us: the camera restarted mid-save, or
                    // the save ran slower than the retention margin allows.
                    throw new IOException("buffer was overwritten during the save");
                }
                // Rebased to zero: camera timestamps count from device boot.
                info.set(0, packet.size, packet.ptsUs - plan.firstPtsUs,
                        packet.keyFrame ? MediaCodec.BUFFER_FLAG_KEY_FRAME : 0);
                muxer.writeSampleData(track, scratch, info);
                written++;
            }

            muxer.stop();
            ok = true;
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

        Saved s = new Saved();
        s.name = name;
        s.seconds = plan.durationUs() / 1_000_000.0;
        s.packets = written;
        return s;
    }

    /**
     * Where one MP4 goes: a pending MediaStore entry on API 29+, a plain file in
     * the public Movies folder below that. Same split as BackgroundExporter.
     */
    private static final class Output {
        private final Context ctx;
        private Uri uri;
        private ParcelFileDescriptor pfd;
        private File file;

        private Output(Context ctx) {
            this.ctx = ctx;
        }

        static Output open(Context ctx, String name) throws IOException {
            Output o = new Output(ctx);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ContentResolver resolver = ctx.getContentResolver();
                ContentValues cv = new ContentValues();
                cv.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
                cv.put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4");
                cv.put(MediaStore.MediaColumns.RELATIVE_PATH,
                        Environment.DIRECTORY_MOVIES + File.separator + FOLDER);
                cv.put(MediaStore.MediaColumns.IS_PENDING, 1);
                o.uri = resolver.insert(
                        MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), cv);
                if (o.uri == null) {
                    throw new IOException("MediaStore insert returned null for " + name);
                }
                // MP4 writing seeks back to patch the header, so the fd must be read-write.
                o.pfd = resolver.openFileDescriptor(o.uri, "rw");
                if (o.pfd == null) {
                    resolver.delete(o.uri, null, null);
                    throw new IOException("could not open " + o.uri);
                }
            } else {
                File dir = new File(
                        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
                        FOLDER);
                //noinspection ResultOfMethodCallIgnored
                dir.mkdirs();
                o.file = new File(dir, name);
            }
            return o;
        }

        MediaMuxer newMuxer() throws IOException {
            int fmt = MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4;
            return pfd != null
                    ? new MediaMuxer(pfd.getFileDescriptor(), fmt)
                    : new MediaMuxer(file.getAbsolutePath(), fmt);
        }

        /** Publishes the file on success; deletes the partial one on failure. */
        void finish(boolean ok) {
            if (pfd != null) {
                try {
                    pfd.close();
                } catch (IOException e) {
                    Log.w(TAG, "fd close failed", e);
                }
            }
            if (uri != null) {
                ContentResolver resolver = ctx.getContentResolver();
                if (ok) {
                    ContentValues cv = new ContentValues();
                    cv.put(MediaStore.MediaColumns.IS_PENDING, 0);
                    resolver.update(uri, cv, null, null);
                } else {
                    resolver.delete(uri, null, null);
                }
            } else if (file != null) {
                if (ok) {
                    MediaScannerConnection.scanFile(ctx,
                            new String[]{file.getAbsolutePath()}, new String[]{"video/mp4"}, null);
                } else {
                    //noinspection ResultOfMethodCallIgnored
                    file.delete();
                }
            }
        }
    }
}
