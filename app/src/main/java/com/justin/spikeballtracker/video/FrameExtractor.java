package com.justin.spikeballtracker.video;

import android.content.Context;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Decodes a video file one frame at a time into the pipeline's CV_8UC3 Y,U,V
 * format at {@link #TARGET_SIZE}.
 *
 * <p>{@link #open}, then {@link #nextFrame} until it returns null, then
 * {@link #close}. Each call does a bounded amount of work, so the caller can
 * interleave frames with other jobs on the same thread. Not thread safe: use
 * it from one thread only (the app's background thread).
 */
public class FrameExtractor {
    private final Context context;
    private static final String TAG = "FrameExtractor";
    private static final long BUFFER_TIMEOUT_US = 10_000;
    private final YuvFrameBuilder yuvBuilder = new YuvFrameBuilder("video");

    /** The pipeline's working resolution, shared with the camera path. */
    public static final Size TARGET_SIZE = new Size(854, 480);

    /** The decode size the pipeline is designed around; other sizes are handled, and logged. */
    private static final int PREFERRED_WIDTH = 1920;
    private static final int PREFERRED_HEIGHT = 1080;

    /**
     * True when the decoder's samples are limited range and must be expanded.
     * Undeclared footage is treated as limited, which is the convention for
     * H.264 and what phone cameras produce.
     */
    private boolean expandToFullRange = true;

    // The open stream; all null/false when nothing is open.
    private MediaExtractor extractor = null;
    private MediaCodec codec = null;
    private final MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
    private boolean inputDone = false;
    private boolean outputDone = false;
    private long timestampUs = 0L;

    /** Channel-wise 16-235 / 16-240 to 0-255 map, built once. */
    private final Mat fullRangeLut = buildFullRangeLut();


    public FrameExtractor(Context context) {
        this.context = context.getApplicationContext();
    }

    /**
     * Opens a video file and starts its decoder. Closes anything already open.
     *
     * @return false if the file cannot be opened or has no video track
     */
    public boolean open(Uri videoUri) {
        close();
        try {
            extractor = new MediaExtractor();                 // demuxes the file into encoded samples
            extractor.setDataSource(context, videoUri, null);

            int trackIndex = findVideoTrack(extractor);
            if (trackIndex < 0) {
                Log.e(TAG, "No video track found in " + videoUri);
                close();
                return false;
            }

            extractor.selectTrack(trackIndex);
            MediaFormat format = extractor.getTrackFormat(trackIndex);
            expandToFullRange = !isFullRange(format);
            logStreamSize(format);

            codec = createDecoder(format);                    // low-level decoder, raw pixel output
            inputDone = false;
            outputDone = false;
            return true;
        } catch (IOException | RuntimeException e) {
            Log.e(TAG, "Could not open video " + videoUri, e);
            close();
            return false;
        }
    }

    /**
     * Decodes until the next frame is ready.
     *
     * <p>The returned Mat is owned by this FrameExtractor and overwritten by
     * the next call. The pixels are a copy, not a view into the decoder's
     * buffer. Do not keep it; take a {@code clone()} to keep the pixels.
     *
     * @return the frame as CV_8UC3 Y,U,V at {@link #TARGET_SIZE}, or null at
     *         the end of the stream or when nothing is open
     */
    public Mat nextFrame() {
        while (codec != null && !outputDone) {
            if (!inputDone) {
                inputDone = feedInput(extractor, codec);
            }

            int outIndex = codec.dequeueOutputBuffer(bufferInfo, BUFFER_TIMEOUT_US);
            if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                // Carries the real output size, crop and strides; YuvFrameBuilder
                // reads the same from each Image, this is for the log.
                Log.i(TAG, "decoder output format " + codec.getOutputFormat());
            } else if (outIndex >= 0) {
                Mat frame = null;
                if (bufferInfo.size > 0) {
                    Image image = codec.getOutputImage(outIndex);
                    if (image != null) {
                        frame = buildFrame(image);
                        image.close();
                    }
                }
                codec.releaseOutputBuffer(outIndex, false);

                if ((bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    outputDone = true;
                }
                if (frame != null) {
                    timestampUs = bufferInfo.presentationTimeUs;
                    return frame;
                }
            }
        }
        return null;
    }

    /** Presentation timestamp of the frame last returned by {@link #nextFrame}, in microseconds. */
    public long timestampUs() {
        return timestampUs;
    }

    /** Stops and frees the decoder and the per-stream buffers. Safe to call when nothing is open. */
    public void close() {
        release(codec, extractor);
        codec = null;
        extractor = null;
        yuvBuilder.release();
    }

    /**
     * Whether the container declares full-range (0-255) samples. The key is
     * optional and frequently absent on consumer footage; an undeclared range
     * conventionally means limited (16-235).
     */
    private static boolean isFullRange(MediaFormat format) {
        return format.containsKey(MediaFormat.KEY_COLOR_RANGE)
                && format.getInteger(MediaFormat.KEY_COLOR_RANGE) == MediaFormat.COLOR_RANGE_FULL;
    }

    /**
     * Builds the lookup table that maps limited-range samples to full range.
     *
     * <p>Luma and chroma use different conventions: luma occupies 16-235 and
     * expands about its floor, chroma occupies 16-240 and expands about its
     * neutral point of 128. A single 3-channel LUT applies both in one pass,
     * which is far cheaper per frame than splitting and merging channels.
     *
     * @return a 1x256 CV_8UC3 table, channels in Y, U, V order
     */
    private static Mat buildFullRangeLut() {
        Mat lut = new Mat(1, 256, CvType.CV_8UC3);
        byte[] table = new byte[256 * 3];

        for (int i = 0; i < 256; i++) {
            int y = clampByte((int) Math.round((i - 16) * 255.0 / 219.0));
            int c = clampByte((int) Math.round((i - 128) * 255.0 / 224.0 + 128));
            table[i * 3] = (byte) y;
            table[i * 3 + 1] = (byte) c;
            table[i * 3 + 2] = (byte) c;
        }

        lut.put(0, 0, table);
        return lut;
    }

    private static int clampByte(int v) {
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }

    /** Returns the index of the first video track, or -1 if there is none. */
    private int findVideoTrack(MediaExtractor extractor) {
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            String mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("video/")) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Logs the stream's coded size against the 1920x1080 the pipeline is
     * designed around.
     *
     * <p>MediaCodec cannot be asked for a different output size: in buffer
     * mode a decoder always delivers the stream's own resolution. So a
     * 1920x1080 clip decodes at 1920x1080, and anything else decodes at its
     * own size and is resized from there by {@link YuvFrameBuilder}.
     */
    private static void logStreamSize(MediaFormat format) {
        int w = format.getInteger(MediaFormat.KEY_WIDTH);
        int h = format.getInteger(MediaFormat.KEY_HEIGHT);
        int rotation = format.containsKey(MediaFormat.KEY_ROTATION)
                ? format.getInteger(MediaFormat.KEY_ROTATION) : 0;
        Log.i(TAG, "video track " + w + "x" + h + ", rotation " + rotation
                + ", " + (isFullRange(format) ? "full" : "limited") + " range");
        if (w != PREFERRED_WIDTH || h != PREFERRED_HEIGHT) {
            Log.w(TAG, "track is " + w + "x" + h + ", not " + PREFERRED_WIDTH + "x"
                    + PREFERRED_HEIGHT + "; the decoder cannot rescale, so frames are"
                    + " resized from " + w + "x" + h);
        }
        if (rotation != 0) {
            Log.w(TAG, "track asks for " + rotation + " degree rotation on display; frames are"
                    + " processed as stored, unrotated");
        }
    }

    /**
     * Creates and starts a decoder for the given track format, choosing one
     * that declares support for the track's size and asking it for flexible
     * YUV 4:2:0 output, which is what {@link MediaCodec#getOutputImage} needs.
     */
    private MediaCodec createDecoder(MediaFormat format) throws IOException {
        String mime = format.getString(MediaFormat.KEY_MIME);
        String name = new MediaCodecList(MediaCodecList.REGULAR_CODECS).findDecoderForFormat(format);

        MediaCodec codec;
        if (name != null) {
            codec = MediaCodec.createByCodecName(name);
        } else {
            Log.w(TAG, "no decoder declares support for " + format.getInteger(MediaFormat.KEY_WIDTH)
                    + "x" + format.getInteger(MediaFormat.KEY_HEIGHT) + " " + mime
                    + "; trying the default, which may fail or drop frames");
            codec = MediaCodec.createDecoderByType(mime);
        }

        format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
        codec.configure(format, null, null, 0);   // null Surface = raw pixel output
        codec.start();
        Log.i(TAG, "decoder " + codec.getName());
        return codec;
    }

    /**
     * Queues one encoded sample into the decoder.
     * @return true once the end of the stream has been signaled
     */
    private boolean feedInput(MediaExtractor extractor, MediaCodec codec) {
        int inIndex = codec.dequeueInputBuffer(BUFFER_TIMEOUT_US);
        if (inIndex < 0) {
            return false;   // no buffer free right now; try again next pass
        }

        ByteBuffer buffer = codec.getInputBuffer(inIndex);
        int size = extractor.readSampleData(buffer, 0);

        if (size < 0) {
            codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
            return true;
        }

        codec.queueInputBuffer(inIndex, 0, size, extractor.getSampleTime(), 0);
        extractor.advance();
        return false;
    }

    private void release(MediaCodec codec, MediaExtractor extractor) {
        if (codec != null) {
            try {
                codec.stop();
                codec.release();
            } catch (IllegalStateException e) {
                Log.w(TAG, "codec release failed", e);
            }
        }
        if (extractor != null) {
            extractor.release();
        }
    }

    /**
     * The decoded Image as the pipeline's CV_8UC3 Y,U,V frame at
     * {@link #TARGET_SIZE}: each plane resized directly from its native size,
     * then the limited-to-full range expansion if the stream needs it.
     */
    private Mat buildFrame(Image image) {
        Mat yuv = yuvBuilder.build(image, TARGET_SIZE);
        if (expandToFullRange) {
            Core.LUT(yuv, fullRangeLut, yuv);
        }
        return yuv;
    }

    /** Closes any open stream and frees everything. Call once, when the owner is destroyed. */
    public void release() {
        close();
        fullRangeLut.release();
    }
}
