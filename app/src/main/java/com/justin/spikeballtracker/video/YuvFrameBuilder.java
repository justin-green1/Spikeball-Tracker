package com.justin.spikeballtracker.video;

import android.media.Image;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;

/**
 * Converts a decoded YUV_420_888 Image into a 3-channel CV_8UC3 Y,U,V Mat.
 *
 * <p>Deliberately does no resizing and no range expansion. Both are policy
 * belonging to the caller: MediaCodec output and CameraX ImageAnalysis output
 * share this plane layout but not those decisions.
 *
 * <p>Owns its scratch buffers, so every source needs its own instance.
 * Not thread safe. The returned Mat is overwritten on the next call.
 */
public final class YuvFrameBuilder {

    private final Mat uPacked   = new Mat();
    private final Mat vPacked   = new Mat();
    private final Mat resizedU  = new Mat();
    private final Mat resizedV  = new Mat();
    private final Mat yuvMat    = new Mat();

    public Mat build(Image image) {
        Image.Plane yPlane = image.getPlanes()[0];
        Image.Plane uPlane = image.getPlanes()[1];
        Image.Plane vPlane = image.getPlanes()[2];

        int width  = image.getWidth();
        int height = image.getHeight();
        int chromaWidth  = width  / 2;
        int chromaHeight = height / 2;

        Mat yMat = new Mat(height, width, CvType.CV_8UC1,
                yPlane.getBuffer(), yPlane.getRowStride());

        Mat uSrc = null;
        Mat vSrc = null;

        try {
            if (uPlane.getPixelStride() == 2) {
                // Semi-planar (NV12/NV21) — this is the branch your phone takes.
                uSrc = new Mat(chromaHeight, chromaWidth, CvType.CV_8UC2,
                        uPlane.getBuffer(), uPlane.getRowStride());
                vSrc = new Mat(chromaHeight, chromaWidth, CvType.CV_8UC2,
                        vPlane.getBuffer(), vPlane.getRowStride());

                Core.extractChannel(uSrc, uPacked, 0);
                Core.extractChannel(vSrc, vPacked, 0);
            } else {
                // Truly planar (I420).
                uSrc = new Mat(chromaHeight, chromaWidth, CvType.CV_8UC1,
                        uPlane.getBuffer(), uPlane.getRowStride());
                vSrc = new Mat(chromaHeight, chromaWidth, CvType.CV_8UC1,
                        vPlane.getBuffer(), vPlane.getRowStride());

                uSrc.copyTo(uPacked);
                vSrc.copyTo(vPacked);
            }

            Size fullSize = yMat.size();
            Imgproc.resize(uPacked, resizedU, fullSize, 0, 0, Imgproc.INTER_NEAREST);
            Imgproc.resize(vPacked, resizedV, fullSize, 0, 0, Imgproc.INTER_NEAREST);

            Core.merge(Arrays.asList(yMat, resizedU, resizedV), yuvMat);
        } finally {
            if (uSrc != null) uSrc.release();
            if (vSrc != null) vSrc.release();
            yMat.release();
        }

        return yuvMat;
    }

    public void release() {
        uPacked.release();
        vPacked.release();
        resizedU.release();
        resizedV.release();
        yuvMat.release();
    }
}