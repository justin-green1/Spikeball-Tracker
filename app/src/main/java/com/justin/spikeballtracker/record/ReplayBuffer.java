package com.justin.spikeballtracker.record;

import android.content.Context;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

import androidx.annotation.NonNull;
import androidx.camera.core.Preview;
import androidx.camera.core.SurfaceRequest;

import com.justin.spikeballtracker.WorkerThread;
import com.justin.spikeballtracker.log.Formats;
import com.justin.spikeballtracker.log.Json;

import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * Keeps the last minute of camera footage in RAM, and copies a confirmed
 * rally's footage out of it before it is overwritten.
 *
 * <p>How frames get here: a CameraX {@link Preview} use case is bound next to
 * the analysis stream, and its surface is the hardware H.264 encoder's input
 * surface. The camera writes straight into the encoder, so no pixels touch
 * Java or OpenCV. That keeps the recording at the full camera frame rate
 * and native 1920x1080 however slow the detectors are, at the price of
 * the recorded frames being the camera's parallel stream rather than the exact
 * frames the analyser saw. Both streams come off the same sensor, on the same
 * clock.
 *
 * <p>The encoder's output goes into an {@link EncodedRing}, which keeps
 * overwriting its oldest minute. {@link #copyRally} copies one rally's packets
 * into a {@link BufferedRally} of their own, which {@link RallySaver} remuxes
 * into an MP4 with no re-encode.
 *
 * <p>Everything here runs on this buffer's own Recorder {@link WorkerThread}:
 * surface requests, all MediaCodec callbacks, session teardown, and the copy
 * in {@link #copyRally}. Nothing else touches the ring, so none of the state
 * needs a lock, and neither detection nor saving can hold up the encoder.
 * Callers ask for a copy and get the finished, unchanging
 * {@link BufferedRally} back through a callback.
 *
 * <p>Lifetime: the ring is kept when the camera stops, so the rally confirmed
 * at the stop can still be copied. It is cleared when the next camera session
 * starts, because a new encoder's packets cannot be muxed with the old one's
 * stream headers. A rally already copied is unaffected.
 */
public final class ReplayBuffer {

    private static final String TAG = "ReplayBuffer";

    /**
     * How much is held: a minute, plus the ~5 s a rally takes to be confirmed
     * after it ends and the 1 s before it that a clip also keeps. A rally of
     * up to about a minute can therefore still be copied whole.
     */
    private static final long RETAIN_US = 66 * 1_000_000L;

    /**
     * The recording's newest packet should be within a frame or two of the
     * newest analysis frame. Further apart than this means the two streams'
     * timestamps are on different clocks, and the rally is shifted to match.
     */
    private static final long CLOCK_MISMATCH_US = 1_000_000L;

    private static final String MIME = MediaFormat.MIMETYPE_VIDEO_AVC;
    private static final int BIT_RATE = 10_000_000;         // ~75 MB per minute at 1080p
    private static final int FRAME_RATE = 30;               // a hint to the encoder; timestamps rule
    private static final int I_FRAME_INTERVAL_S = 1;        // clip start granularity
    private static final int MAX_FPS_FOR_SLOTS = 60;        // sizes the packet metadata ring
    private static final double BYTE_HEADROOM = 1.25;       // VBR overshoots on busy scenes

    /**
     * Also write every encoder session, start to finish, to the gallery as
     * {@code Movies/SpikeballTracker/session_<stamp>.mp4}: the same packets
     * that go into the ring, remuxed as they arrive. About 75 MB a minute.
     * For playing a live session back through the video path later.
     */
    private static final boolean SAVE_SESSION_VIDEO = true;

    private final Context appContext;
    private final WorkerThread recorder = new WorkerThread("Recorder");

    /** Where recording events go as finished JSON lines; Recorder thread only. */
    private Consumer<String> eventSink = null;

    /** Allocated on first use, so an app that never opens the camera never pays ~100 MB. */
    private EncodedRing ring = null;

    /** Stream headers (csd-0/csd-1) for what is in the ring. */
    private MediaFormat outputFormat = null;
    private int rotationDegrees = 0;

    private Session current = null;
    private final List<Session> live = new ArrayList<>();

    public ReplayBuffer(Context context) {
        this.appContext = context.getApplicationContext();
    }

    /**
     * Receives a {@code recording} line when a session video starts and a
     * {@code recording_end} line when it is saved, called on the Recorder
     * thread. Null for none.
     */
    public void setEventSink(Consumer<String> sink) {
        recorder.post(() -> eventSink = sink);
    }

    /** Routes the preview use case's frames into the encoder. Call before binding. */
    public void attachTo(Preview preview) {
        preview.setSurfaceProvider(recorder.executor(), this::onSurfaceRequested);
    }

    /**
     * Copies the footage from {@code fromUs} to {@code toUs} out of the ring,
     * on the Recorder thread, and hands it to {@code done} there. The clip
     * starts on the keyframe at or before {@code fromUs} (up to one second
     * earlier), and is cut short at either end by what the ring holds.
     * Returns at once; may be called from any thread.
     *
     * @param fromUs start, on the camera's sensor clock (ImageProxy timestamp / 1000)
     * @param toUs   end, same clock
     * @param nowUs  the newest analysis frame's time, same clock; used to check
     *               the recording really is on that clock
     * @param done   receives the copy, or null when the footage is not (or no
     *               longer) held; called on the Recorder thread
     */
    public void copyRally(long fromUs, long toUs, long nowUs, Consumer<RallyFootage> done) {
        recorder.post(() -> done.accept(copy(fromUs, toUs, nowUs)));
    }

    /** {@link #copyRally}'s work. Recorder thread only. A few ms per 10 s of footage. */
    private BufferedRally copy(long fromUs, long toUs, long nowUs) {
        EncodedRing r = ring;
        if (r == null) {
            Log.w(TAG, "nothing recorded; rally footage not kept");
            return null;
        }
        if (outputFormat == null || r.count() == 0) {
            Log.w(TAG, "no encoded footage yet; rally footage not kept");
            return null;
        }

        long skewUs = r.newestPtsUs() - nowUs;
        if (Math.abs(skewUs) > CLOCK_MISMATCH_US) {
            Log.w(TAG, "recording and analysis timestamps are " + skewUs / 1000
                    + " ms apart, more than a clock's worth of latency; "
                    + "shifting the rally onto the recording's clock");
            fromUs += skewUs;
            toUs += skewUs;
        }

        EncodedRing.Plan plan = r.planRange(fromUs, toUs);
        EncodedRing.Snapshot packets = plan == null ? null : r.snapshot(plan);
        if (packets == null) {
            Log.w(TAG, "rally footage is no longer in the ring; not kept");
            return null;
        }
        if (plan.firstPtsUs > fromUs) {
            Log.w(TAG, "rally began before the oldest footage held; the clip starts "
                    + (plan.firstPtsUs - fromUs) / 1000 + " ms late");
        }
        return new BufferedRally(packets, outputFormat, rotationDegrees);
    }

    /** Stops any encoder, then the Recorder thread, after the copies already asked for. */
    public void release() {
        recorder.post(() -> {
            for (Session s : new ArrayList<>(live)) {
                s.end();
            }
            current = null;
        });
        recorder.quit();
    }

    // --- encoder side ---------------------------------------------------------

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
            codec.setCallback(session, recorder.handler());    // must precede configure
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
        r.clear();
        outputFormat = null;
        rotationDegrees = 0;
        current = session;
        live.add(session);

        request.setTransformationInfoListener(recorder.executor(), info -> {
            if (current == session) {
                rotationDegrees = info.getRotationDegrees();
            }
        });

        // The result arrives once the camera has stopped writing to the surface,
        // which is the only safe moment to tear the encoder down.
        request.provideSurface(session.surface, recorder.executor(), result -> {
            session.end();
            if (current == session) {
                current = null;
            }
        });
    }

    private EncodedRing ensureRing() {
        if (ring != null) {
            return ring;
        }
        double retainS = RETAIN_US / 1_000_000.0;
        int capacity = (int) (BIT_RATE / 8.0 * retainS * BYTE_HEADROOM);
        int slots = (int) Math.ceil(retainS) * MAX_FPS_FOR_SLOTS;
        // At 1080p the full ring is ~100 MB of Java heap. If the heap cannot
        // spare that, a half-size ring still works: the byte limit evicts
        // first, so it simply holds about half as many seconds.
        for (int attempt = capacity; attempt >= capacity / 2; attempt /= 2) {
            try {
                ring = new EncodedRing(attempt, slots, RETAIN_US);
                if (attempt < capacity) {
                    Log.w(TAG, "heap too small for a " + (capacity / (1024 * 1024))
                            + " MB replay ring; using " + (attempt / (1024 * 1024))
                            + " MB, about " + Math.round(attempt / (BIT_RATE / 8.0 * BYTE_HEADROOM))
                            + " s of 1080p");
                }
                return ring;
            } catch (OutOfMemoryError e) {
                Log.w(TAG, "could not allocate a " + (attempt / (1024 * 1024)) + " MB replay ring"
                        + " (heap limit " + (Runtime.getRuntime().maxMemory() / (1024 * 1024)) + " MB)");
            }
        }
        Log.e(TAG, "not enough heap for a replay ring; replay disabled");
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
                        writeToFile(buf, info, key);
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
                outputFormat = format;
                if (SAVE_SESSION_VIDEO && muxer == null && !fileFailed) {
                    startFile(format);
                }
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

        // ------------------------------------------------ session video file

        private Mp4File file = null;
        private MediaMuxer muxer = null;
        private String fileName = null;
        private boolean fileFailed = false;
        private int track = -1;
        private long firstPtsUs = -1L;
        private long lastPtsUs = -1L;
        private int samples = 0;
        private final MediaCodec.BufferInfo fileInfo = new MediaCodec.BufferInfo();

        private void startFile(MediaFormat format) {
            String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
            fileName = "session_" + stamp + ".mp4";
            try {
                file = Mp4File.open(appContext, fileName);
                muxer = file.newMuxer();
                muxer.setOrientationHint(rotationDegrees);
                track = muxer.addTrack(format);
                muxer.start();
                Log.i(TAG, "saving this camera session to Movies/SpikeballTracker/" + fileName);
            } catch (Exception e) {
                Log.e(TAG, "could not start " + fileName + "; this session is not saved", e);
                abandonFile();
            }
        }

        /** Appends one packet, timestamps rebased so the file starts at 0. */
        private void writeToFile(ByteBuffer buf, MediaCodec.BufferInfo info, boolean key) {
            if (muxer == null) {
                return;
            }
            if (firstPtsUs < 0) {
                if (!key) {
                    return;   // the file must start on a keyframe
                }
                firstPtsUs = info.presentationTimeUs;
                Json j = new Json().obj().field("type", "recording")
                        .field("file", "Movies/SpikeballTracker/" + fileName)
                        .field("first_src_us", firstPtsUs)
                        .field("note", "video PTS = camera src - first_src_us");
                Formats.mediaFormat(j, "format", outputFormat);
                emit(j.end().toString());
            }
            try {
                fileInfo.set(info.offset, info.size, info.presentationTimeUs - firstPtsUs,
                        key ? MediaCodec.BUFFER_FLAG_KEY_FRAME : 0);
                muxer.writeSampleData(track, buf, fileInfo);
                lastPtsUs = info.presentationTimeUs;
                samples++;
            } catch (RuntimeException e) {
                Log.e(TAG, fileName + " write failed; closing the file", e);
                finishFile();
            }
        }

        /** Closes the session video, publishing it if it holds anything. */
        private void finishFile() {
            if (muxer == null) {
                return;
            }
            boolean ok = samples > 0;
            try {
                if (ok) {
                    muxer.stop();
                }
            } catch (RuntimeException e) {
                Log.e(TAG, fileName + " could not be finalised", e);
                ok = false;
            }
            muxer.release();
            muxer = null;
            file.finish(ok);
            file = null;
            Log.i(TAG, (ok ? "saved " : "discarded ") + fileName + ", " + samples + " frames");
            emit(new Json().obj().field("type", "recording_end")
                    .field("file", "Movies/SpikeballTracker/" + fileName)
                    .field("saved", ok)
                    .field("frames", samples)
                    .field("last_src_us", lastPtsUs)
                    .end().toString());
        }

        private void abandonFile() {
            fileFailed = true;
            if (muxer != null) {
                muxer.release();
                muxer = null;
            }
            if (file != null) {
                file.finish(false);
                file = null;
            }
        }

        private void emit(String line) {
            if (eventSink != null) {
                eventSink.accept(line);
            }
        }

        void end() {
            if (ended) {
                return;
            }
            ended = true;
            live.remove(this);
            finishFile();
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
}
