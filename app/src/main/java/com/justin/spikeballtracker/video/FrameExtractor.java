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
         * <p>{@code transientLuma} is a view into the decoder's buffer and is
         * invalid once this method returns. Do not store it, do not derive
         * submats from it, and do not hand it to another thread. If you need
         * to keep the pixels, call {@code transientLuma.clone()} and release
         * the copy yourself.
         *
         * @return true to continue decoding, false to stop
         */
        boolean onFrame(Mat transientLuma, long timestampUs);
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
                if (bufferInfo.size > 0) {
                    Image image = codec.getOutputImage(outIndex);
                    if (image != null) {
                        Mat lumaMat = buildYuvMat(image);
                        boolean keepGoing = listener.onFrame(lumaMat, bufferInfo.presentationTimeUs);
                        lumaMat.release();
                        image.close();

                        if (!keepGoing) {
                            return;
                        }
                    }
                }
                codec.releaseOutputBuffer(outIndex, false);

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
    private Mat buildYuvMat(Image image) {
        Image.Plane yPlane = image.getPlanes()[0];
        ByteBuffer yBuffer = yPlane.getBuffer();
        Image.Plane uPlane = image.getPlanes()[1];
        ByteBuffer uBuffer = uPlane.getBuffer();
        Image.Plane vPlane = image.getPlanes()[2];
        ByteBuffer vBuffer = vPlane.getBuffer();

        Mat yMat = new Mat(
            image.getHeight(),
            image.getWidth(),
            CvType.CV_8UC1,
            yBuffer,
            yPlane.getRowStride());

        Mat uMat = new Mat(
                image.getHeight()/2,
                image.getWidth()/2,
                CvType.CV_8UC1,
                uBuffer,
                uPlane.getRowStride());

        Mat vMat = new Mat(
                image.getHeight()/2,
                image.getWidth()/2,
                CvType.CV_8UC1,
                vBuffer,
                vPlane.getRowStride());

        Size newSize = yMat.size();

        Imgproc.resize(uMat, resizedUMat, newSize);
        Imgproc.resize(vMat, resizedVMat, newSize);

        Core.merge(Arrays.asList(yMat, resizedUMat, resizedVMat), yuvMat);

        vMat.release();
        uMat.release();
        yMat.release();
        return yuvMat;
    }

    public void release(){
        resizedUMat.release();
        resizedVMat.release();
        yuvMat.release();
    }
}
