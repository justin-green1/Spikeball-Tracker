package com.justin.spikeballtracker.vision;

/**
 * The seven features {@link BlobSvm} was trained on, computed for one
 * connected component exactly as the offline study computed them
 * ({@code studies/classifier/blob_features.py} and {@code shape_filters.py}).
 *
 * <p>Plain Java on arrays the caller has read out of the frame once, so a
 * busy frame (S3 clips give ~140 components a frame, up to ~370) costs no
 * per-blob OpenCV calls or native allocations.
 *
 * <p>The study did not see components directly. It drew a 1 px ring just
 * outside each blob's filled outline on a crop, then recovered the blob as
 * "everything the ring encloses". This class takes the same steps on the
 * component itself, so the features match:
 *
 * <ol>
 *   <li><b>solid</b>: the component with its holes filled (background not
 *       4-connected to the outside, which for an 8-connected component is
 *       what filling its outer contour gives);</li>
 *   <li><b>ring</b>: solid dilated by a 3x3 square, minus solid;</li>
 *   <li><b>silhouette</b>: every pixel not 4-connected to the outside without
 *       crossing the ring, minus the ring itself. Usually equal to solid; it
 *       can also hold a pocket the ring closes off;</li>
 *   <li><b>blob pixels</b>: silhouette AND the widened colour box;</li>
 *   <li><b>surround band</b>: silhouette dilated by 13x13, minus silhouette
 *       dilated by 5x5 (pixels 3-6 px out, Chebyshev), clipped to the frame.</li>
 * </ol>
 *
 * <p>Features, in {@link BlobSvm#FEATURES} order:
 * <pre>
 *   aspect      (long + 1) / (short + 1) of the min-area rectangle around the
 *               silhouette's main piece (pixel centres)
 *   fill        component area / ((long + 1) * (short + 1))
 *   Y_mean      mean luma of the blob pixels
 *   Y_std       population std of that luma
 *   U_std       population std of U
 *   hue_deg     atan2(mean V - 128, mean U - 128) in degrees, 0-360
 *   Y_contrast  Y_mean minus the band's mean luma (NaN if the band is empty)
 * </pre>
 *
 * <p>"Component area" is the component's pixel count, as the detector filters
 * on; the colour statistics use the silhouette pixels inside the box, which can
 * include closed holes. Both are as trained.
 *
 * <p>The study measured the rectangle on the silhouette's longest outer
 * contour. When a ring-closed pocket makes the silhouette two pieces, this
 * takes the piece with the most pixels, the same piece in every case tested.
 * Where two rectangle orientations tie exactly on area, OpenCV's minAreaRect
 * can pick the other one, and since the features use the sides + 1 that changes
 * aspect and fill: 22 of 3 250 blobs tested, one decision flip at the wide
 * threshold. Everything else matches to rounding; BlobFeaturesParityTest
 * checks both.
 *
 * <p>Holds reusable scratch arrays, so one instance belongs to one thread.
 */
public final class BlobFeatures {

    /** Local grid padding. The ring sits 1 px out, and the flood fill needs a clear border beyond it. */
    private static final int PAD = 2;

    /** Band reach: the 13x13 dilation extends 6 px. */
    private static final int BAND_OUT = 6;
    /** The 5x5 dilation extends 2 px; band pixels are further out than this. */
    private static final int BAND_IN = 2;

    private final int yLo, yHi, uLo, uHi, vLo, vHi;

    // Scratch, grown on demand.
    private byte[] grid = new byte[0];      // local grid states, see constants below
    private int[] stack = new int[0];
    private byte[] sil = new byte[0];       // band grid: 1 = silhouette
    private byte[] tmp = new byte[0];       // band grid: row distance to the silhouette, capped
    private byte[] near = new byte[0];      // band grid: bit 0 within BAND_IN above, bit 1 within BAND_OUT above
    private int[] pieceOf = new int[0];
    private double[] hx = new double[0];
    private double[] hy = new double[0];
    private double[] cx = new double[0];
    private double[] cy = new double[0];
    private final double[] side = new double[2];
    private byte[] rowSolid = new byte[0];

    private static final byte COMP = 1, HOLE_OR_OUT = 0, OUT = 2, RING = 3, INSIDE = 4;

    /**
     * @param boxLo lower Y,U,V bound of the colour box the blobs were cut with
     * @param boxHi upper bound, inclusive
     */
    public BlobFeatures(int[] boxLo, int[] boxHi) {
        yLo = boxLo[0];
        uLo = boxLo[1];
        vLo = boxLo[2];
        yHi = boxHi[0];
        uHi = boxHi[1];
        vHi = boxHi[2];
    }

    /**
     * Computes the features of one component.
     *
     * @param yuv    the whole frame, interleaved Y,U,V, row-major, {@code frameW * frameH * 3}
     * @param labels the whole label image, row-major, {@code frameW * frameH}
     * @param label  the component's label
     * @param left   bounding box, from the component's stats
     * @param area   the component's pixel count
     * @param out    written with {@link BlobSvm#N_FEATURES} values
     * @return false if the blob has fewer than 3 in-box pixels, which the study
     *         skipped; {@code out} is then undefined
     */
    public boolean compute(byte[] yuv, int[] labels, int frameW, int frameH, int label,
                           int left, int top, int width, int height, int area,
                           double[] out) {
        // ---- local grid: the bbox plus PAD on every side (virtual, may leave the frame)
        final int lw = width + 2 * PAD;
        final int lh = height + 2 * PAD;
        final int ln = lw * lh;
        if (grid.length < ln) {
            grid = new byte[ln];
            stack = new int[ln];
            pieceOf = new int[ln];
        }
        java.util.Arrays.fill(grid, 0, ln, HOLE_OR_OUT);
        for (int r = 0; r < height; r++) {
            int src = (top + r) * frameW + left;
            int dst = (r + PAD) * lw + PAD;
            for (int c = 0; c < width; c++) {
                if (labels[src + c] == label) grid[dst + c] = COMP;
            }
        }

        // ---- 1. solid: mark background 4-connected to the border as OUT; the rest is solid
        flood4(lw, lh, HOLE_OR_OUT, OUT);
        // grid: OUT = outside, anything else (COMP or a hole, still 0) = solid
        // ---- 2. ring: non-solid pixels with a solid 8-neighbour
        //         Encode solid as COMP; leftover 0s were holes.
        int solidCount = 0;
        for (int i = 0; i < ln; i++) {
            if (grid[i] != OUT) {
                grid[i] = COMP;
                solidCount++;
            }
        }
        // rowSolid[i] = solid at c-1, c or c+1 (interior columns, every row)
        if (rowSolid.length < ln) rowSolid = new byte[ln];
        for (int r = 0; r < lh; r++) {
            int row = r * lw;
            for (int c = 1; c < lw - 1; c++) {
                int i = row + c;
                rowSolid[i] = (byte) (grid[i - 1] == COMP || grid[i] == COMP || grid[i + 1] == COMP ? 1 : 0);
            }
        }
        for (int r = 1; r < lh - 1; r++) {
            int row = r * lw;
            for (int c = 1; c < lw - 1; c++) {
                int i = row + c;
                if (grid[i] == OUT && (rowSolid[i - lw] | rowSolid[i] | rowSolid[i + lw]) != 0) {
                    grid[i] = RING;
                }
            }
        }
        // ---- 3. silhouette: everything except the ring and what the border reaches around it
        for (int i = 0; i < ln; i++) {
            grid[i] = grid[i] == RING ? RING : HOLE_OR_OUT;     // ring, or not-yet-known
        }
        flood4(lw, lh, HOLE_OR_OUT, OUT);
        int insideCount = 0;
        for (int i = 0; i < ln; i++) {
            if (grid[i] == HOLE_OR_OUT) {
                grid[i] = INSIDE;
                insideCount++;
            }
        }
        if (insideCount == 0) {
            return false;
        }

        // ---- 4. shape: min-area rectangle of the silhouette's largest 8-connected piece.
        // The silhouette is solid plus any pockets the ring closed off. Solid is one
        // 8-connected piece, so with no pockets (almost always) there is nothing to label.
        int bestPiece = -1, bestCount = insideCount;
        if (insideCount != solidCount) {
            int pieces = 0;
            bestCount = 0;
            java.util.Arrays.fill(pieceOf, 0, ln, -1);
            for (int i = 0; i < ln; i++) {
                if (grid[i] != INSIDE || pieceOf[i] >= 0) continue;
                int count = fillPiece8(lw, lh, i, pieces);
                if (count > bestCount) {
                    bestCount = count;
                    bestPiece = pieces;
                }
                pieces++;
            }
        }
        double[] side = minAreaRect(lw, lh, bestPiece, bestCount);
        double a = Math.max(side[0], side[1]) + 1;
        double b = Math.min(side[0], side[1]) + 1;

        // ---- 5. colour statistics on silhouette AND box, and the band seed
        final int bx0 = Math.max(0, left - BAND_OUT);
        final int by0 = Math.max(0, top - BAND_OUT);
        final int bx1 = Math.min(frameW, left + width + BAND_OUT);
        final int by1 = Math.min(frameH, top + height + BAND_OUT);
        final int bw = bx1 - bx0, bh = by1 - by0, bn = bw * bh;
        if (sil.length < bn) {
            sil = new byte[bn];
            tmp = new byte[bn];
            near = new byte[bn];
        }
        java.util.Arrays.fill(sil, 0, bn, (byte) 0);

        long n = 0;
        double sumY = 0, sumU = 0, sumV = 0;
        for (int r = 0; r < lh; r++) {
            int fy = top + r - PAD;
            if (fy < 0 || fy >= frameH) continue;
            for (int c = 0; c < lw; c++) {
                if (grid[r * lw + c] != INSIDE) continue;
                int fx = left + c - PAD;
                if (fx < 0 || fx >= frameW) continue;
                sil[(fy - by0) * bw + (fx - bx0)] = 1;
                int p = 3 * (fy * frameW + fx);
                int y = yuv[p] & 0xFF, u = yuv[p + 1] & 0xFF, v = yuv[p + 2] & 0xFF;
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
        final double meanY = sumY / n, meanU = sumU / n, meanV = sumV / n;
        double ssY = 0, ssU = 0;
        for (int r = 0; r < bh; r++) {
            for (int c = 0; c < bw; c++) {
                if (sil[r * bw + c] == 0) continue;
                int p = 3 * ((by0 + r) * frameW + bx0 + c);
                int y = yuv[p] & 0xFF, u = yuv[p + 1] & 0xFF, v = yuv[p + 2] & 0xFF;
                if (y < yLo || y > yHi || u < uLo || u > uHi || v < vLo || v > vHi) continue;
                ssY += (y - meanY) * (y - meanY);
                ssU += (u - meanU) * (u - meanU);
            }
        }

        // ---- 6. band: Chebyshev distance to the silhouette in (BAND_IN, BAND_OUT].
        // Row sweeps give each cell its distance to the nearest silhouette cell in
        // its own row (capped); a cell is within k of the silhouette iff some row
        // within k of it has a row distance <= k at that column.
        for (int r = 0; r < bh; r++) {
            int row = r * bw, last = -1000;
            for (int c = 0; c < bw; c++) {
                if (sil[row + c] != 0) last = c;
                int d = c - last;
                tmp[row + c] = (byte) (d > BAND_OUT ? BAND_OUT + 1 : d);
            }
            last = 1000000;
            for (int c = bw - 1; c >= 0; c--) {
                if (sil[row + c] != 0) last = c;
                int d = last - c;
                if (d < tmp[row + c]) tmp[row + c] = (byte) d;
            }
        }
        long nb = 0;
        double sumBandY = 0;
        for (int c = 0; c < bw; c++) {
            int lastNear = -1000, lastFar = -1000;
            for (int r = 0; r < bh; r++) {
                int d = tmp[r * bw + c];
                if (d <= BAND_IN) lastNear = r;
                if (d <= BAND_OUT) lastFar = r;
                near[r * bw + c] = (byte) ((r - lastNear <= BAND_IN ? 1 : 0) | (r - lastFar <= BAND_OUT ? 2 : 0));
            }
            lastNear = 1000000;
            lastFar = 1000000;
            for (int r = bh - 1; r >= 0; r--) {
                int i = r * bw + c;
                int d = tmp[i];
                if (d <= BAND_IN) lastNear = r;
                if (d <= BAND_OUT) lastFar = r;
                boolean isNear = (near[i] & 1) != 0 || lastNear - r <= BAND_IN;
                boolean isFar = (near[i] & 2) != 0 || lastFar - r <= BAND_OUT;
                if (isFar && !isNear) {
                    nb++;
                    sumBandY += yuv[3 * ((by0 + r) * frameW + bx0 + c)] & 0xFF;
                }
            }
        }

        double hue = Math.toDegrees(Math.atan2(meanV - 128, meanU - 128));
        hue = ((hue % 360) + 360) % 360;

        out[0] = a / b;
        out[1] = area / (a * b);
        out[2] = meanY;
        out[3] = Math.sqrt(ssY / n);
        out[4] = Math.sqrt(ssU / n);
        out[5] = hue;
        out[6] = nb > 0 ? meanY - sumBandY / nb : Double.NaN;
        return true;
    }

    // ---------------------------------------------------------------- helpers

    /** 4-connected flood from every border cell holding {@code from}, relabelling to {@code to}. */
    private void flood4(int w, int h, byte from, byte to) {
        int sp = 0;
        for (int c = 0; c < w; c++) {
            sp = seed(c, from, to, sp);
            sp = seed((h - 1) * w + c, from, to, sp);
        }
        for (int r = 1; r < h - 1; r++) {
            sp = seed(r * w, from, to, sp);
            sp = seed(r * w + w - 1, from, to, sp);
        }
        while (sp > 0) {
            int i = stack[--sp];
            int r = i / w, c = i - r * w;
            if (c > 0 && grid[i - 1] == from) { grid[i - 1] = to; stack[sp++] = i - 1; }
            if (c < w - 1 && grid[i + 1] == from) { grid[i + 1] = to; stack[sp++] = i + 1; }
            if (r > 0 && grid[i - w] == from) { grid[i - w] = to; stack[sp++] = i - w; }
            if (r < h - 1 && grid[i + w] == from) { grid[i + w] = to; stack[sp++] = i + w; }
        }
    }

    private int seed(int i, byte from, byte to, int sp) {
        if (grid[i] == from) {
            grid[i] = to;
            stack[sp++] = i;
        }
        return sp;
    }

    /** Labels the 8-connected INSIDE piece containing {@code start}; returns its size. */
    private int fillPiece8(int w, int h, int start, int piece) {
        int sp = 0, count = 0;
        pieceOf[start] = piece;
        stack[sp++] = start;
        while (sp > 0) {
            int i = stack[--sp];
            count++;
            int r = i / w, c = i - r * w;
            for (int dr = -1; dr <= 1; dr++) {
                int rr = r + dr;
                if (rr < 0 || rr >= h) continue;
                for (int dc = -1; dc <= 1; dc++) {
                    int cc = c + dc;
                    if (cc < 0 || cc >= w) continue;
                    int j = rr * w + cc;
                    if (grid[j] == INSIDE && pieceOf[j] < 0) {
                        pieceOf[j] = piece;
                        stack[sp++] = j;
                    }
                }
            }
        }
        return count;
    }

    /**
     * Side lengths of the minimum-area rectangle around one piece's pixel
     * centres: convex hull (monotone chain), then each hull edge as a
     * candidate orientation.
     */
    private double[] minAreaRect(int w, int h, int piece, int count) {
        final boolean all = piece < 0;
        // Boundary pixels are enough for the hull; collect pixels with a non-piece 4-neighbour,
        // in row-major order, which is already sorted by (y, x).
        if (hx.length < count) {
            hx = new double[count];
            hy = new double[count];
        }
        int m = 0;
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                int i = r * w + c;
                if (!in(i, all, piece)) continue;
                boolean edge = c == 0 || c == w - 1 || r == 0 || r == h - 1
                        || !in(i - 1, all, piece) || !in(i + 1, all, piece)
                        || !in(i - w, all, piece) || !in(i + w, all, piece);
                if (edge) {
                    hx[m] = c;
                    hy[m] = r;
                    m++;
                }
            }
        }
        if (m == 1) {
            side[0] = 0;
            side[1] = 0;
            return side;
        }
        // Monotone chain needs points sorted by x then y; ours are sorted by y then x,
        // which works the same with the axes swapped.
        if (cx.length < 2 * m) {
            cx = new double[2 * m];
            cy = new double[2 * m];
        }
        int k = 0;
        for (int i = 0; i < m; i++) {
            while (k >= 2 && cross(cy[k - 2], cx[k - 2], cy[k - 1], cx[k - 1], hy[i], hx[i]) <= 0) k--;
            cx[k] = hx[i];
            cy[k] = hy[i];
            k++;
        }
        for (int i = m - 2, lower = k + 1; i >= 0; i--) {
            while (k >= lower && cross(cy[k - 2], cx[k - 2], cy[k - 1], cx[k - 1], hy[i], hx[i]) <= 0) k--;
            cx[k] = hx[i];
            cy[k] = hy[i];
            k++;
        }
        int hn = k - 1;   // last point repeats the first
        if (hn < 2) {     // all points on one line: the "rectangle" is that segment
            double dx = cx[1] - cx[0], dy = cy[1] - cy[0];
            double len = Math.sqrt(dx * dx + dy * dy);
            for (int i = 0; i < m; i++) {
                double ex = hx[i] - cx[0], ey = hy[i] - cy[0];
                len = Math.max(len, Math.sqrt(ex * ex + ey * ey));
            }
            side[0] = len;
            side[1] = 0;
            return side;
        }
        double bestArea = Double.MAX_VALUE, bw = 0, bh = 0;
        for (int e = 0; e < hn; e++) {
            double ex = cx[e + 1] - cx[e], ey = cy[e + 1] - cy[e];
            double len = Math.sqrt(ex * ex + ey * ey);
            if (len == 0) continue;
            ex /= len;
            ey /= len;
            double minU = Double.MAX_VALUE, maxU = -Double.MAX_VALUE;
            double minV = Double.MAX_VALUE, maxV = -Double.MAX_VALUE;
            for (int j = 0; j < hn; j++) {
                double px = cx[j] - cx[e], py = cy[j] - cy[e];
                double u = px * ex + py * ey;
                double v = -px * ey + py * ex;
                if (u < minU) minU = u;
                if (u > maxU) maxU = u;
                if (v < minV) minV = v;
                if (v > maxV) maxV = v;
            }
            double rw = maxU - minU, rh = maxV - minV, ar = rw * rh;
            if (ar < bestArea - 1e-9) {
                bestArea = ar;
                bw = rw;
                bh = rh;
            }
        }
        side[0] = bw;
        side[1] = bh;
        return side;
    }

    private boolean in(int i, boolean all, int piece) {
        return all ? grid[i] == INSIDE : pieceOf[i] == piece;
    }

    private static double cross(double ox, double oy, double ax, double ay, double bx, double by) {
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
    }
}
