package com.justin.spikeballtracker.vision;

import com.justin.spikeballtracker.tracking.BallCandidate;
import com.justin.spikeballtracker.tracking.OccupancyGrid;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Connected components of the ball mask (blobs), each scored by
 * {@link BlobSvm}; those scoring at least {@link BlobSvm#THRESHOLD} are the
 * candidates, the one stream the tracker follows.
 *
 * <p>Blobs whose features cannot be computed (fewer than 3 in-box pixels,
 * which the study skipped) get P(ball) = 0, so they are never candidates.
 *
 * <p>{@link #MIN_AREA} is 30 px (16 Oct 2026; 40 px from 13 Oct, 15 before).
 * The SVM was trained on 40-400 px blobs, so 30-39 px blobs are scored
 * outside the sizes it has seen; it has not been retrained for them. The
 * lower floor recovers small, distant balls: on S1C03's labelled rally
 * frames, balls cut by the floor fell from 14.5 % to 10.9 %, and in the
 * offline replay (five clips, old-SVM scores) per-frame tracking F1 rose by
 * about 0.01 with no more false rallies. Below about 40 px most blobs are
 * clutter (15-39 px blobs were 45-61 % of all blobs on frames 901-5400 of
 * S1C01, S3C01 and s5c04, box B), which the tracker's shape test must reject.
 *
 * <p>Static clutter is dropped before scoring (13 Oct 2026): an
 * {@link OccupancyGrid} in front of the SVM learns from every in-range blob,
 * and a blob whose cell is static is neither scored nor returned. The grid is
 * read before the frame's update, as the offline study that chose this
 * position ran it (data/results/grid_position/REPORT.md).
 *
 * <p>Cost: the frame and the label image are read out of OpenCV once per
 * frame, and everything per blob is plain Java ({@link BlobFeatures}, then
 * {@link BlobSvm}), all on the calling thread.
 *
 * <p>Owns reusable buffers: one instance belongs to one pipeline thread.
 */
public final class SvmBlobDetector {

    private static final int MIN_AREA = 30;
    private static final int MAX_AREA = 400;

    private final Mat labels = new Mat();
    private final Mat stats = new Mat();
    private final Mat centroids = new Mat();

    private final BlobFeatures features;
    private final double[] f = new double[BlobSvm.N_FEATURES];

    private byte[] yuvBytes = new byte[0];
    private int[] labelInts = new int[0];
    private int[] st = new int[0];
    private double[] ce = new double[0];
    private int[] todo = new int[0];
    private int[] live = new int[0];

    /** Static-clutter memory; replaced by {@link #reset()}. */
    private OccupancyGrid grid = new OccupancyGrid();
    private double[] prob = new double[0];

    /**
     * @param boxLo the colour box the mask was cut with, lower Y,U,V bound
     * @param boxHi upper bound, inclusive
     */
    public SvmBlobDetector(Scalar boxLo, Scalar boxHi) {
        int[] lo = {(int) boxLo.val[0], (int) boxLo.val[1], (int) boxLo.val[2]};
        int[] hi = {(int) boxHi.val[0], (int) boxHi.val[1], (int) boxHi.val[2]};
        features = new BlobFeatures(lo, hi);
    }

    /**
     * @param mask single-channel 8-bit mask, colour box AND motion (non-zero = foreground)
     * @param yuv  the CV_8UC3 Y,U,V frame the mask came from
     * @return the candidates: components of {@link #MIN_AREA}-{@link #MAX_AREA} px
     *         outside the grid's static cells with P(ball) at least {@link BlobSvm#THRESHOLD}
     */
    public List<BallCandidate> findCandidates(Mat mask, Mat yuv) {
        int count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids);
        List<BallCandidate> out = new ArrayList<>();

        final int sc = stats.cols();
        if (st.length < count * sc) st = new int[count * sc];
        if (ce.length < count * 2) ce = new double[count * 2];
        if (todo.length < count) {
            todo = new int[count];
            live = new int[count];
            prob = new double[count];
        }
        stats.get(0, 0, st);
        centroids.get(0, 0, ce);

        int n = 0;
        for (int i = 1; i < count; i++) {   // label 0 is the background
            int a = st[i * sc + Imgproc.CC_STAT_AREA];
            if (a >= MIN_AREA && a <= MAX_AREA) todo[n++] = i;
        }

        // Occupancy grid: read every blob's cell first, then learn from all of them.
        int nLive = 0;
        for (int j = 0; j < n; j++) {
            int i = todo[j];
            if (!grid.isStatic(ce[2 * i], ce[2 * i + 1])) live[nLive++] = i;
        }
        grid.nextFrame();
        for (int j = 0; j < n; j++) {
            int i = todo[j];
            grid.hit(ce[2 * i], ce[2 * i + 1]);
        }
        int[] swap = todo;
        todo = live;
        live = swap;
        n = nLive;

        if (n > 0) {
            final int w = yuv.cols(), h = yuv.rows();
            if (yuvBytes.length != w * h * 3) yuvBytes = new byte[w * h * 3];
            if (labelInts.length != w * h) labelInts = new int[w * h];
            yuv.get(0, 0, yuvBytes);
            labels.get(0, 0, labelInts);

            score(n, w, h, sc);

            for (int j = 0; j < n; j++) {
                if (prob[j] < BlobSvm.THRESHOLD) continue;
                int i = todo[j];
                out.add(new BallCandidate(ce[2 * i], ce[2 * i + 1], prob[j]));
            }
        }
        return out;
    }

    /** Scores the first n blobs of the todo list into prob. */
    private void score(int n, int w, int h, int sc) {
        for (int j = 0; j < n; j++) {
            int i = todo[j];
            int b = i * sc;
            prob[j] = features.compute(yuvBytes, labelInts, w, h, i,
                    st[b + Imgproc.CC_STAT_LEFT], st[b + Imgproc.CC_STAT_TOP],
                    st[b + Imgproc.CC_STAT_WIDTH], st[b + Imgproc.CC_STAT_HEIGHT],
                    st[b + Imgproc.CC_STAT_AREA], f)
                    ? BlobSvm.probability(f) : 0.0;
        }
    }

    /** Forgets the static clutter; call when the camera's view changes (background reset). */
    public void reset() {
        grid = new OccupancyGrid();
    }

    public void release() {
        labels.release();
        stats.release();
        centroids.release();
    }
}
