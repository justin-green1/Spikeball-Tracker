package com.justin.spikeballtracker.video;

import android.content.Context;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
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
    private final Mat resizedUMat;
    private final Mat resizedVMat;
    private final Mat uPacked = new Mat();
    private final Mat vPacked = new Mat();
    private final Mat yuvMat;

    public FrameExtractor(Context context) {
        this.context = context.getApplicationContext();
        this.resizedUMat = new Mat();
        this.resizedVMat = new Mat();
        this.yuvMat = new Mat();
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

            codec = createDecoder(format);
            decodeLoop(extractor, codec, listener);

        } catch (IOException e) {
            Log.e(TAG, "Could not open video " + videoUri, e);
        } finally {
            release(codec, extractor);
            releaseScratch();
        }
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
                        frame = buildYuvMat(image);
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

    /**
     * Wraps the luma (Y) plane of a decoded frame as a single-channel Mat.
     *
     * <p>No pixel data is copied — the Mat views the codec's output buffer directly,
     * so it is only valid until the Image is closed. Callers must consume or clone
     * it before returning.
     *
     * @param image decoded frame in YUV_420_888 format
     * @return grayscale Mat sharing the image's underlying buffer
     */
    /**
     * Builds a 3-channel YUV Mat from a decoded frame.
     *
     * <p>Chroma planes may be semi-planar (NV12/NV21, pixel stride 2) or truly
     * planar (I420, pixel stride 1); both are handled. The U and V planes are
     * upsampled to full resolution and merged with Y into {@link #yuvMat}.
     *
     * <p>The returned Mat is owned by this extractor and is overwritten on the
     * next call. Callers must not retain it.
     *
     * @param image decoded frame in YUV_420_888 format
     * @return CV_8UC3 Mat with channels in Y, U, V order
     */
    private Mat buildYuvMat(Image image) {
        Image.Plane yPlane = image.getPlanes()[0];
        Image.Plane uPlane = image.getPlanes()[1];
        Image.Plane vPlane = image.getPlanes()[2];

        int width = image.getWidth();
        int height = image.getHeight();
        int chromaWidth = width / 2;
        int chromaHeight = height / 2;

        Mat yMat = new Mat(height, width, CvType.CV_8UC1,
                yPlane.getBuffer(), yPlane.getRowStride());

        Mat uSrc = null;
        Mat vSrc = null;
        try {
            if (uPlane.getPixelStride() == 2) {
                // Semi-planar. Each plane's buffer starts on its own component,
                // so reading pairs and keeping channel 0 is correct for both
                // NV12 and NV21 without needing to know which one this is.
                uSrc = new Mat(chromaHeight, chromaWidth, CvType.CV_8UC2,
                        uPlane.getBuffer(), uPlane.getRowStride());
                vSrc = new Mat(chromaHeight, chromaWidth, CvType.CV_8UC2,
                        vPlane.getBuffer(), vPlane.getRowStride());

                Core.extractChannel(uSrc, uPacked, 0);
                Core.extractChannel(vSrc, vPacked, 0);
            } else {
                // Truly planar (I420) — one byte per sample, no de-interleaving.
                uSrc = new Mat(chromaHeight, chromaWidth, CvType.CV_8UC1,
                        uPlane.getBuffer(), uPlane.getRowStride());
                vSrc = new Mat(chromaHeight, chromaWidth, CvType.CV_8UC1,
                        vPlane.getBuffer(), vPlane.getRowStride());

                uSrc.copyTo(uPacked);
                vSrc.copyTo(vPacked);
            }

            Size fullSize = yMat.size();
            Imgproc.resize(uPacked, resizedUMat, fullSize, 0, 0, Imgproc.INTER_NEAREST);
            Imgproc.resize(vPacked, resizedVMat, fullSize, 0, 0, Imgproc.INTER_NEAREST);

            Core.merge(Arrays.asList(yMat, resizedUMat, resizedVMat), yuvMat);
        } finally {
            if (uSrc != null) uSrc.release();
            if (vSrc != null) vSrc.release();
            yMat.release();
        }

        Imgproc.resize(yuvMat, yuvMat, new Size(854, 480), 0, 0, Imgproc.INTER_AREA);

        return yuvMat;
    }

    /** Frees the per-stream scratch buffers. */
    private void releaseScratch() {
        uPacked.release();
        vPacked.release();
        resizedUMat.release();
        resizedVMat.release();
        yuvMat.release();
    }
}
