package com.justin.spikeballtracker.video;

import android.content.Context;
import android.graphics.Rect;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.Build;
import android.util.Log;


import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class FrameExtractor {
    private final Context context;
    private static final String TAG = "FrameExtractor";
    private static final long BUFFER_TIMEOUT_US = 10_000;
    private final YuvFrameBuilder yuvBuilder = new YuvFrameBuilder();

    private volatile String mime = "unknown";
    private volatile String colourRange = "undeclared";
    private volatile String colourStandard = "unknown";
    public static final Size TARGET_SIZE = new Size(854, 480);

    /**
     * True when the decoder's samples are limited range and must be expanded.
     * Undeclared footage is treated as limited, which is the convention for
     * H.264 and what phone cameras produce.
     */
    private volatile boolean expandToFullRange = true;

    /** Channel-wise 16-235 / 16-240 to 0-255 map, built once. */
    private final Mat fullRangeLut = buildFullRangeLut();


    public FrameExtractor(Context context) {
        this.context = context.getApplicationContext();
    }

    /**
     * An immutable snapshot of what the current stream declared about colour.
     *
     * <p>The fields it copies live on this long-lived extractor and are reset
     * per stream; taking a snapshot means a caller cannot accidentally read a
     * previous video's metadata, which is the failure mode that made an
     * untagged file report as limited bt709.
     */
    public static final class ColourInfo {
        public final String mime;
        public final String range;
        public final String standard;
        public final boolean expandToFullRange;

        ColourInfo(String mime, String range, String standard, boolean expandToFullRange) {
            this.mime = mime;
            this.range = range;
            this.standard = standard;
            this.expandToFullRange = expandToFullRange;
        }

        @Override
        public String toString() {
            return "mime=" + mime + " range=" + range + " standard=" + standard
                    + " expandToFullRange=" + expandToFullRange;
        }
    }

    /**
     * What the current stream declared. Valid once decoding has begun; the
     * fields are set by readColourInfo before the first frame is delivered.
     */
    public ColourInfo colourInfo() {
        return new ColourInfo(mime, colourRange, colourStandard, expandToFullRange);
    }

    public interface FrameListener {
        /**
         * Called once per decoded frame, on the decoding thread.
         *
         * <p>{@code frame} is a 3-channel YUV Mat (CV_8UC3) owned by this
         * FrameExtractor. The pixels are a copy, not a view into the decoder's
         * buffer — but the extractor overwrites this same Mat on the next decoded
         * frame. Do not store it, do not derive submats from it, and do not hand it
         * to another thread. To keep the pixels beyond this call, take a
         * {@code frame.clone()} and release the copy yourself.
         *
         * @param frame       current frame as CV_8UC3 YUV; valid only for this call
         * @param timestampUs presentation timestamp, in microseconds
         * @return true to continue decoding, false to stop
         */
        boolean onFrame(Mat frame, long timestampUs);
    }

    /**
     * Decodes a video file and delivers every frame to the listener in order.
     * Blocks until the video ends or the listener requests a stop.
     */
    public void streamFrames(Uri videoUri, FrameListener listener) {
        MediaExtractor extractor = null;                                //MediaExtractor facilitates extraction of demuxed, typically encoded, media data from a data source.
        MediaCodec codec = null;                                        //used to access low level media codecs

        try {
            extractor = new MediaExtractor();
            extractor.setDataSource(context, videoUri, null);

            int trackIndex = findVideoTrack(extractor);
            if (trackIndex < 0) {
                Log.e(TAG, "No video track found in " + videoUri);
                return;
            }

            extractor.selectTrack(trackIndex);
            MediaFormat format = extractor.getTrackFormat(trackIndex);
            Log.d(TAG, "video format: " + format);

            readColourInfo(format);
            Log.d(TAG, "colour: mime=" + mime
                    + " range=" + colourRange
                    + " standard=" + colourStandard
                    + " -> " + (expandToFullRange
                            ? "expanding to full range"
                            : "already full range, passing through"));

            codec = createDecoder(format);
            decodeLoop(extractor, codec, listener);

        } catch (IOException e) {
            Log.e(TAG, "Could not open video " + videoUri, e);
        } finally {
            release(codec, extractor);
            yuvBuilder.release();
        }
    }

    /**
     * Records what the container claims about colour range and primaries.
     *
     * <p>Both keys are optional and frequently absent on consumer footage; an
     * undeclared range conventionally means limited (16-235), but this method
     * reports "undeclared" rather than assuming, so callers can decide.
     */
    private void readColourInfo(MediaFormat format) {
        // Reset first. These fields outlive a single stream, and the keys below
        // are optional, so without this an untagged video silently inherits the
        // colour metadata of whichever video was opened before it.
        mime = "unknown";
        colourRange = "undeclared";
        colourStandard = "unknown";
        expandToFullRange = true;

        mime = format.getString(MediaFormat.KEY_MIME);

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
            return;   // the colour keys did not exist before API 24
        }

        if (format.containsKey(MediaFormat.KEY_COLOR_RANGE)) {
            int range = format.getInteger(MediaFormat.KEY_COLOR_RANGE);
            if (range == MediaFormat.COLOR_RANGE_FULL) {
                colourRange = "full";
                expandToFullRange = false;   // already 0-255, expanding would clip
            } else if (range == MediaFormat.COLOR_RANGE_LIMITED) {
                colourRange = "limited";
            } else {
                colourRange = "unknown(" + range + ")";
            }
        }

        if (format.containsKey(MediaFormat.KEY_COLOR_STANDARD)) {
            int std = format.getInteger(MediaFormat.KEY_COLOR_STANDARD);
            if (std == MediaFormat.COLOR_STANDARD_BT709) {
                colourStandard = "bt709";
            } else if (std == MediaFormat.COLOR_STANDARD_BT601_PAL
                    || std == MediaFormat.COLOR_STANDARD_BT601_NTSC) {
                colourStandard = "bt601";
            } else if (std == MediaFormat.COLOR_STANDARD_BT2020) {
                colourStandard = "bt2020";
            } else {
                colourStandard = "unknown(" + std + ")";
            }
        }
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

    /** Creates and starts a decoder for the given track format. */
    private MediaCodec createDecoder(MediaFormat format) throws IOException {
        String mime = format.getString(MediaFormat.KEY_MIME);
        MediaCodec codec = MediaCodec.createDecoderByType(mime);
        codec.configure(format, null, null, 0);   // null Surface = raw pixel output
        codec.start();
        return codec;
    }

    /**
     * Pumps encoded samples into the decoder and pulls decoded frames out until
     * the stream ends or the listener returns false.
     */
    private void decodeLoop(MediaExtractor extractor, MediaCodec codec, FrameListener listener) {
        MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
        boolean inputDone = false;
        boolean outputDone = false;

        while (!outputDone) {
            if (!inputDone) {
                inputDone = feedInput(extractor, codec);
            }

            int outIndex = codec.dequeueOutputBuffer(bufferInfo, BUFFER_TIMEOUT_US);
            if (outIndex >= 0) {
                Mat frame = null;
                if (bufferInfo.size > 0) {
                    Image image = codec.getOutputImage(outIndex);
                    if (image != null) {
                        frame = buildFrame(image);
                        image.close();
                    }
                }
                codec.releaseOutputBuffer(outIndex, false);

                if (frame != null
                        && !listener.onFrame(frame, bufferInfo.presentationTimeUs)) {
                    return;
                }

                if ((bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    outputDone = true;
                }
            }
        }
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

    private Mat buildFrame(Image image) {
        Mat yuv = yuvBuilder.build(image);
        Imgproc.resize(yuv, yuv, TARGET_SIZE, 0, 0, Imgproc.INTER_AREA);
        if (expandToFullRange) {
            Core.LUT(yuv, fullRangeLut, yuv);
        }
        return yuv;
    }

    /**
     * Frees resources held for the lifetime of this extractor, as opposed to
     * the per-stream scratch buffers. Call from the owner's onDestroy.
     */
    public void release() {
        fullRangeLut.release();
    }
}
