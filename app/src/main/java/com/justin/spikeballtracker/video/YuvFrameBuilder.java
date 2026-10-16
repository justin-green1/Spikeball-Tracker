package com.justin.spikeballtracker.video;

import android.graphics.Rect;
import android.media.Image;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

/**
 * Converts a YUV_420_888 {@link Image} straight into a CV_8UC3 Y,U,V Mat at the
 * caller's working resolution.
 *
 * <p>Each plane is resized on its own, directly from its native size to the
 * target with {@code INTER_AREA}, and only then merged. A 1920x1080 frame has a
 * 1920x1080 Y plane and 960x540 U and V planes; all three go straight to
 * 854x480. There is no intermediate full-resolution chroma: the chroma the
 * encoder or ISP already subsampled is not expanded and then shrunk again.
 *
 * <p>For a piecewise-constant chroma plane, the old route (nearest-neighbour
 * upsample to the luma size, then area-average the merged image) and the new
 * route (area-resize the chroma plane directly) integrate the same area of the
 * same samples, so the two agree up to rounding in OpenCV's fixed and
 * floating point arithmetic. The new route reads a quarter of the chroma
 * pixels and allocates no full-size Mats.
 *
 * <p>Planes are read through their row and pixel strides, and only the
 * image's crop rectangle is used: decoders commonly pad 1080 rows to 1088 and
 * report the valid area through the crop. Pixel stride 1 (planar) is read as a
 * zero-copy view; pixel stride 2 (NV12/NV21) and wider are viewed as
 * multi-channel elements and channel 0 is extracted, which reads only the
 * plane's own bytes.
 *
 * <p>Source resolutions:
 * <ul>
 *   <li>16:9 at any size: every plane is resized to the target. A source
 *       whose chroma plane is smaller than the target (below 1708x960) has
 *       its chroma enlarged by INTER_AREA, which then interpolates rather than
 *       averages; this is logged.</li>
 *   <li>Not 16:9 (within 1 %): the largest centred region with the target's
 *       aspect ratio is used, so the detectors' geometry is not stretched.
 *       This is logged.</li>
 * </ul>
 *
 * <p>Does no range expansion: that is the caller's policy, because MediaCodec
 * and CameraX output share this plane layout but not their colour range.
 *
 * <p>Owns its scratch buffers, so every source needs its own instance.
 * Not thread safe. The returned Mat is overwritten on the next call.
 */
public final class YuvFrameBuilder {

    private static final String TAG = "YuvFrameBuilder";

    /** Relative aspect-ratio difference treated as "the same shape". */
    private static final double ASPECT_TOLERANCE = 0.01;

    private final String name;

    private final Mat yResized = new Mat();
    private final Mat uResized = new Mat();
    private final Mat vResized = new Mat();
    private final Mat packed   = new Mat();   // a strided plane gathered to one byte per sample
    private final Mat yuvMat   = new Mat();
    private final List<Mat> resized = Arrays.asList(yResized, uResized, vResized);

    /** What was last logged, so a change of source is reported once, not every frame. */
    private String lastGeometry = null;

    /** @param name which source this builder serves, for the log */
    public YuvFrameBuilder(String name) {
        this.name = name;
    }

    /**
     * @param image  a YUV_420_888 image; not closed or otherwise modified
     * @param target working resolution, e.g. 854x480
     * @return CV_8UC3 Y,U,V Mat of exactly {@code target}, owned by this builder
     */
    public Mat build(Image image, Size target) {
        if (image.getFormat() != android.graphics.ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException("expected YUV_420_888, got format " + image.getFormat());
        }
        Image.Plane[] planes = image.getPlanes();

        Rect valid = validRegion(image);
        Rect used = centredToAspect(valid, target.width / target.height);
        reportGeometry(image, valid, used, target);

        // Chroma covering the luma region; for odd edges this rounds outward.
        int cLeft   = used.left >> 1;
        int cTop    = used.top >> 1;
        int cWidth  = ((used.right + 1) >> 1) - cLeft;
        int cHeight = ((used.bottom + 1) >> 1) - cTop;

        resizePlane(planes[0], used.left, used.top, used.width(), used.height(), target, yResized);
        resizePlane(planes[1], cLeft, cTop, cWidth, cHeight, target, uResized);
        resizePlane(planes[2], cLeft, cTop, cWidth, cHeight, target, vResized);

        Core.merge(resized, yuvMat);
        return yuvMat;
    }

    public void release() {
        yResized.release();
        uResized.release();
        vResized.release();
        packed.release();
        yuvMat.release();
    }

    // ------------------------------------------------------------------ planes

    /**
     * Resizes one plane's region to {@code target} with INTER_AREA, reading it
     * in place through the plane's strides wherever it can.
     *
     * @param x, y, w, h region in this plane's own sample coordinates
     */
    private void resizePlane(Image.Plane plane, int x, int y, int w, int h,
                             Size target, Mat out) {
        int pixelStride = plane.getPixelStride();
        int rowStride   = plane.getRowStride();
        ByteBuffer region = regionOf(plane.getBuffer(), x, y, w, h, pixelStride, rowStride);

        if (pixelStride == 1) {
            Mat view = new Mat(h, w, CvType.CV_8UC1, region, rowStride);
            try {
                Imgproc.resize(view, out, target, 0, 0, Imgproc.INTER_AREA);
            } finally {
                view.release();
            }
            return;
        }

        if (pixelStride <= 4 && rowStride >= w * pixelStride) {
            // One element per sample, pixelStride bytes wide; channel 0 is the
            // sample. extractChannel touches only channel 0, so the last
            // element's trailing bytes, which may lie past the plane, are never read.
            Mat view = new Mat(h, w, CvType.makeType(CvType.CV_8U, pixelStride), region, rowStride);
            try {
                Core.extractChannel(view, packed, 0);
            } finally {
                view.release();
            }
        } else {
            gatherStrided(region, w, h, rowStride, pixelStride, packed);
        }
        Imgproc.resize(packed, out, target, 0, 0, Imgproc.INTER_AREA);
    }

    /**
     * A direct buffer starting at sample (x, y), after checking the plane holds
     * every sample of the region.
     */
    private static ByteBuffer regionOf(ByteBuffer plane, int x, int y, int w, int h,
                                       int pixelStride, int rowStride) {
        long start = (long) y * rowStride + (long) x * pixelStride;
        long span  = (long) (h - 1) * rowStride + (long) (w - 1) * pixelStride + 1;
        if (start + span > plane.remaining()) {
            throw new IllegalStateException("plane of " + plane.remaining() + " B cannot hold a "
                    + w + "x" + h + " region at (" + x + "," + y + ") with row stride "
                    + rowStride + " and pixel stride " + pixelStride);
        }
        ByteBuffer dup = plane.duplicate();
        dup.position(plane.position() + (int) start);
        return dup.slice();
    }

    /** Fallback for layouts the views above cannot describe; not expected on real devices. */
    private static void gatherStrided(ByteBuffer src, int w, int h, int rowStride,
                                      int pixelStride, Mat dst) {
        dst.create(h, w, CvType.CV_8UC1);
        byte[] row = new byte[(w - 1) * pixelStride + 1];
        byte[] samples = new byte[w];
        for (int r = 0; r < h; r++) {
            src.position(r * rowStride);
            src.get(row);
            for (int c = 0; c < w; c++) {
                samples[c] = row[c * pixelStride];
            }
            dst.put(r, 0, samples);
        }
    }

    // ------------------------------------------------------------------ geometry

    /** The crop rectangle clipped to the image, or the whole image if the crop is unusable. */
    private static Rect validRegion(Image image) {
        Rect full = new Rect(0, 0, image.getWidth(), image.getHeight());
        Rect crop = image.getCropRect();
        if (crop == null || crop.isEmpty() || !crop.intersect(full)) {
            return full;
        }
        return crop;
    }

    /**
     * The largest region inside {@code r} with the given aspect ratio, centred,
     * with even offsets so the chroma samples stay aligned with the luma.
     * Returns {@code r} itself when it is already that shape.
     */
    private static Rect centredToAspect(Rect r, double aspect) {
        int w = r.width();
        int h = r.height();
        double ratio = ((double) w / h) / aspect;
        if (Math.abs(ratio - 1.0) <= ASPECT_TOLERANCE) {
            return r;
        }
        if (ratio > 1.0) {
            int newW = Math.min(w, (int) Math.round(h * aspect) & ~1);
            int left = r.left + (((w - newW) / 2) & ~1);
            return new Rect(left, r.top, left + newW, r.bottom);
        }
        int newH = Math.min(h, (int) Math.round(w / aspect) & ~1);
        int top = r.top + (((h - newH) / 2) & ~1);
        return new Rect(r.left, top, r.right, top + newH);
    }

    private void reportGeometry(Image image, Rect valid, Rect used, Size target) {
        Image.Plane[] p = image.getPlanes();
        String geometry = image.getWidth() + "x" + image.getHeight()
                + " crop " + valid.toShortString()
                + " Y(row " + p[0].getRowStride() + ", px " + p[0].getPixelStride() + ")"
                + " U(row " + p[1].getRowStride() + ", px " + p[1].getPixelStride() + ")"
                + " V(row " + p[2].getRowStride() + ", px " + p[2].getPixelStride() + ")";
        if (geometry.equals(lastGeometry)) {
            return;
        }
        lastGeometry = geometry;

        int tw = (int) target.width;
        int th = (int) target.height;
        Log.i(TAG, name + ": source " + geometry + " -> " + tw + "x" + th);

        if (valid.width() != 1920 || valid.height() != 1080) {
            Log.w(TAG, name + ": source is " + valid.width() + "x" + valid.height()
                    + ", not the requested 1920x1080; resizing from the actual size");
        }
        if (!used.equals(valid)) {
            Log.w(TAG, name + ": source aspect " + valid.width() + ":" + valid.height()
                    + " differs from the target's; using the centred region "
                    + used.toShortString() + " instead of stretching");
        }
        if ((used.width() + 1) / 2 < tw || (used.height() + 1) / 2 < th) {
            Log.w(TAG, name + ": chroma plane " + (used.width() + 1) / 2 + "x" + (used.height() + 1) / 2
                    + " is smaller than " + tw + "x" + th + "; INTER_AREA enlarges it by interpolation");
        }
    }
}
