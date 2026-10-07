package com.justin.spikeballtracker.vision;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * The seven features {@link BlobSvm} was trained on, computed for one
 * connected component exactly as the offline study computed them
 * ({@code studies/classifier/blob_features.py} and {@code shape_filters.py}).
 *
 * <p>The study did not see components directly. It drew a 1 px ring just
 * outside each blob's filled outline on a crop, then recovered the blob as
 * "everything the ring encloses". This class takes the same steps on the
 * component itself, so the features match:
 *
 * <ol>
 *   <li><b>solid</b>: the component's outer contour, filled (holes closed);</li>
 *   <li><b>ring</b>: solid dilated by a 3x3 square, minus solid;</li>
 *   <li><b>silhouette</b>: every pixel not 4-connected to the outside without
 *       crossing the ring, minus the ring itself. Usually equal to solid; it
 *       can also hold a pocket the ring closes off;</li>
 *   <li><b>blob pixels</b>: silhouette AND the widened colour box, read from
 *       the Y,U,V frame;</li>
 *   <li><b>surround band</b>: silhouette dilated by 13x13, minus silhouette
 *       dilated by 5x5 (pixels 3-6 px out), clipped to the frame.</li>
 * </ol>
 *
 * <p>Features, in {@link BlobSvm#FEATURES} order:
 * <pre>
 *   aspect      (long + 1) / (short + 1) of the silhouette contour's min-area rectangle
 *   fill        component area / ((long + 1) * (short + 1))
 *   Y_mean      mean luma of the blob pixels
 *   Y_std       population std of that luma
 *   U_std       population std of U
 *   hue_deg     atan2(mean V - 128, mean U - 128) in degrees, 0-360
 *   Y_contrast  Y_mean minus the band's mean luma (NaN if the band is empty)
 * </pre>
 *
 * <p>"Component area" is the component's pixel count, as BlobDetector filters
 * on; the colour statistics use the silhouette pixels inside the box, which can
 * include closed holes. Both are as trained.
 *
 * <p>Owns reusable Mats, so one instance belongs to one thread.
 */
public final class BlobFeatures {

    /** Local mask padding. The ring sits 1 px out, and the flood fill needs a clear border beyond it. */
    private static final int PAD = 2;

    /** Band reach: the 13x13 dilation extends 6 px; one more for safety. */
    private static final int BAND_REACH = 7;

    private final int yLo, yHi, uLo, uHi, vLo, vHi;

    private final Mat kernel3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
    private final Mat kernel5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
    private final Mat kernel13 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(13, 13));

    private final Mat comp = new Mat();
    private final Mat solid = new Mat();
    private final Mat dilated = new Mat();
    private final Mat ring = new Mat();
    private final Mat floodMask = new Mat();
    private final Mat inside = new Mat();
    private final Mat hierarchy = new Mat();
    private final Mat bandSil = new Mat();
    private final Mat bandOuter = new Mat();
    private final Mat bandInner = new Mat();

    /**
     * @param boxLo lower Y,U,V bound of the colour box the blobs were cut with
     * @param boxHi upper bound, inclusive
     */
    public BlobFeatures(Scalar boxLo, Scalar boxHi) {
        yLo = (int) boxLo.val[0];
        uLo = (int) boxLo.val[1];
        vLo = (int) boxLo.val[2];
        yHi = (int) boxHi.val[0];
        uHi = (int) boxHi.val[1];
        vHi = (int) boxHi.val[2];
    }

    /**
     * Computes the features of one component.
     *
     * @param yuv    CV_8UC3 Y,U,V frame the mask came from
     * @param labels CV_32S label image from connectedComponentsWithStats
     * @param label  the component's label
     * @param left   bounding box, from the component's stats
     * @param top    "
     * @param width  "
     * @param height "
     * @param area   the component's pixel count
     * @param out    written with {@link BlobSvm#N_FEATURES} values
     * @return false if the blob has fewer than 3 in-box pixels or no contour,
     *         which the study skipped; {@code out} is then undefined
     */
    public boolean compute(Mat yuv, Mat labels, int label,
                           int left, int top, int width, int height, int area,
                           double[] out) {
        final int frameW = yuv.cols();
        final int frameH = yuv.rows();

        // ---- 1. the component, in a padded local mask
        int lw = width + 2 * PAD;
        int lh = height + 2 * PAD;
        int[] lab = new int[width * height];
        Mat labRoi = labels.submat(top, top + height, left, left + width);
        labRoi.get(0, 0, lab);
        labRoi.release();
        byte[] local = new byte[lw * lh];
        for (int r = 0; r < height; r++) {
            int src = r * width;
            int dst = (r + PAD) * lw + PAD;
            for (int c = 0; c < width; c++) {
                if (lab[src + c] == label) {
                    local[dst + c] = 1;
                }
            }
        }
        comp.create(lh, lw, CvType.CV_8UC1);
        comp.put(0, 0, local);

        // ---- 2. solid: outer contour filled
        List<MatOfPoint> outer = new ArrayList<>();
        Imgproc.findContours(comp, outer, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        solid.create(lh, lw, CvType.CV_8UC1);
        solid.setTo(Scalar.all(0));
        Imgproc.drawContours(solid, outer, -1, new Scalar(1), Imgproc.FILLED);
        releaseAll(outer);

        // ---- 3. ring = dilate3x3(solid) - solid
        Imgproc.dilate(solid, dilated, kernel3);
        Core.subtract(dilated, solid, ring);

        // ---- 4. silhouette: not reachable from the border without crossing the ring
        floodMask.create(lh + 2, lw + 2, CvType.CV_8UC1);
        floodMask.setTo(Scalar.all(0));
        Imgproc.floodFill(ring, floodMask, new Point(0, 0), new Scalar(2),
                null, new Scalar(0), new Scalar(0), 4);
        Core.compare(ring, new Scalar(0), inside, Core.CMP_EQ);   // 255 = silhouette

        // ---- 5. shape: the silhouette's longest outer contour
        List<MatOfPoint> sil = new ArrayList<>();
        Imgproc.findContours(inside, sil, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        MatOfPoint longest = null;
        for (MatOfPoint c : sil) {
            if (longest == null || c.rows() > longest.rows()) {
                longest = c;
            }
        }
        if (longest == null) {
            releaseAll(sil);
            return false;
        }
        MatOfPoint2f pts = new MatOfPoint2f(longest.toArray());
        RotatedRect rect = Imgproc.minAreaRect(pts);
        pts.release();
        releaseAll(sil);
        double a = Math.max(rect.size.width, rect.size.height) + 1;
        double b = Math.min(rect.size.width, rect.size.height) + 1;
        double aspect = a / b;
        double fill = area / (a * b);

        // ---- 6. pixels: one read of the frame around the blob, band included
        int bx0 = Math.max(0, left - BAND_REACH);
        int by0 = Math.max(0, top - BAND_REACH);
        int bx1 = Math.min(frameW, left + width + BAND_REACH);
        int by1 = Math.min(frameH, top + height + BAND_REACH);
        int bw = bx1 - bx0;
        int bh = by1 - by0;
        byte[] px = new byte[bw * bh * 3];
        Mat yuvRoi = yuv.submat(by0, by1, bx0, bx1);
        yuvRoi.get(0, 0, px);
        yuvRoi.release();

        byte[] insideBytes = new byte[lw * lh];
        inside.get(0, 0, insideBytes);
        byte[] silBytes = new byte[bw * bh];

        // Silhouette pixels: blob statistics, and the band's seed.
        long n = 0;
        double sumY = 0, sumU = 0, sumV = 0;
        for (int r = 0; r < lh; r++) {
            int fy = top + r - PAD;
            if (fy < 0 || fy >= frameH) continue;
            for (int c = 0; c < lw; c++) {
                if (insideBytes[r * lw + c] == 0) continue;
                int fx = left + c - PAD;
                if (fx < 0 || fx >= frameW) continue;
                int bi = (fy - by0) * bw + (fx - bx0);
                silBytes[bi] = 1;
                int y = px[3 * bi] & 0xFF;
                int u = px[3 * bi + 1] & 0xFF;
                int v = px[3 * bi + 2] & 0xFF;
                if (y < yLo || y > yHi || u < uLo || u > uHi || v < vLo || v > vHi) continue;
                n++;
                sumY += y;
                sumU += u;
                sumV += v;
            }
        }
        if (n < 3) {
            return false;
        }
        double meanY = sumY / n, meanU = sumU / n, meanV = sumV / n;

        // Second pass for the spreads (two-pass, as numpy does).
        double ssY = 0, ssU = 0;
        for (int bi = 0; bi < bw * bh; bi++) {
            if (silBytes[bi] == 0) continue;
            int y = px[3 * bi] & 0xFF;
            int u = px[3 * bi + 1] & 0xFF;
            int v = px[3 * bi + 2] & 0xFF;
            if (y < yLo || y > yHi || u < uLo || u > uHi || v < vLo || v > vHi) continue;
            ssY += (y - meanY) * (y - meanY);
            ssU += (u - meanU) * (u - meanU);
        }

        // ---- 7. surround band: 13x13 dilation minus 5x5 dilation
        bandSil.create(bh, bw, CvType.CV_8UC1);
        bandSil.put(0, 0, silBytes);
        Imgproc.dilate(bandSil, bandOuter, kernel13);
        Imgproc.dilate(bandSil, bandInner, kernel5);
        byte[] outerBytes = new byte[bw * bh];
        byte[] innerBytes = new byte[bw * bh];
        bandOuter.get(0, 0, outerBytes);
        bandInner.get(0, 0, innerBytes);
        long nb = 0;
        double sumBandY = 0;
        for (int bi = 0; bi < bw * bh; bi++) {
            if (outerBytes[bi] != 0 && innerBytes[bi] == 0) {
                nb++;
                sumBandY += px[3 * bi] & 0xFF;
            }
        }

        double hue = Math.toDegrees(Math.atan2(meanV - 128, meanU - 128));
        hue = ((hue % 360) + 360) % 360;

        out[0] = aspect;
        out[1] = fill;
        out[2] = meanY;
        out[3] = Math.sqrt(ssY / n);
        out[4] = Math.sqrt(ssU / n);
        out[5] = hue;
        out[6] = nb > 0 ? meanY - sumBandY / nb : Double.NaN;
        return true;
    }

    private static void releaseAll(List<MatOfPoint> contours) {
        for (MatOfPoint c : contours) {
            c.release();
        }
    }

    public void release() {
        kernel3.release();
        kernel5.release();
        kernel13.release();
        comp.release();
        solid.release();
        dilated.release();
        ring.release();
        floodMask.release();
        inside.release();
        hierarchy.release();
        bandSil.release();
        bandOuter.release();
        bandInner.release();
    }
}
