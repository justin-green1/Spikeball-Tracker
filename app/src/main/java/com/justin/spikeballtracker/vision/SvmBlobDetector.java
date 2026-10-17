package com.justin.spikeballtracker.vision;

import com.justin.spikeballtracker.tracking.BallCandidate;
import com.justin.spikeballtracker.tracking.OccupancyGrid;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
 * {@link BlobSvm}). Busy frames are split across worker threads plus the
 * calling one; results are the same as single-threaded and in the same order.
 * It uses {@link #WORKERS} workers (up to 3, leaving two cores for the
 * decoder and UI).
 *
 * <p>Owns reusable buffers and a thread pool: one instance belongs to one
 * pipeline thread, and {@link #release()} stops the pool.
 */
public final class SvmBlobDetector {

    private static final int MIN_AREA = 30;
    private static final int MAX_AREA = 400;

    /** Extra threads for busy frames; the calling thread takes a share too. */
    private static final int WORKERS =
            Math.max(0, Math.min(3, Runtime.getRuntime().availableProcessors() - 2));

    /** Below this many blobs a frame is scored on the calling thread alone. */
    private static final int PARALLEL_MIN_BLOBS = 24;

    private final Mat labels = new Mat();
    private final Mat stats = new Mat();
    private final Mat centroids = new Mat();

    /** One per share: [0] is the calling thread's, [1..WORKERS] the workers'. */
    private final BlobFeatures[] features;
    private final double[][] f;
    private final ExecutorService pool;
    private final List<Future<?>> pending = new ArrayList<>();

    private byte[] yuvBytes = new byte[0];
    private int[] labelInts = new int[0];
    private int[] st = new int[0];
    private double[] ce = new double[0];
    private int[] todo = new int[0];
    private int[] live = new int[0];

    // Last frame's report, for the run log; filled only while recording.
    private boolean recording = false;
    private int lastComponents = 0, lastInRange = 0, lastStatic = 0, lastScored = 0;
    private double[] featLog = new double[0];   // lastScored x N_FEATURES; NaN where not computed

    /** Static-clutter memory; replaced by {@link #reset()}. */
    private OccupancyGrid grid = new OccupancyGrid();
    private double[] prob = new double[0];

    /**
     * @param boxLo the colour box the mask was cut with, lower Y,U,V bound
     * @param boxHi upper bound, inclusive
     */
    public SvmBlobDetector(Scalar boxLo, Scalar boxHi) {
        features = new BlobFeatures[WORKERS + 1];
        f = new double[WORKERS + 1][BlobSvm.N_FEATURES];
        int[] lo = {(int) boxLo.val[0], (int) boxLo.val[1], (int) boxLo.val[2]};
        int[] hi = {(int) boxHi.val[0], (int) boxHi.val[1], (int) boxHi.val[2]};
        for (int k = 0; k <= WORKERS; k++) {
            features[k] = new BlobFeatures(lo, hi);
        }
        pool = WORKERS == 0 ? null : Executors.newFixedThreadPool(WORKERS, r -> {
            Thread t = new Thread(r, "svm-blob");
            t.setDaemon(true);
            return t;
        });
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
        lastComponents = count - 1;
        lastInRange = n;
        lastStatic = n - nLive;
        lastScored = nLive;
        n = nLive;
        if (recording && featLog.length < n * BlobSvm.N_FEATURES) {
            featLog = new double[n * BlobSvm.N_FEATURES];
        }

        if (n > 0) {
            final int w = yuv.cols(), h = yuv.rows();
            if (yuvBytes.length != w * h * 3) yuvBytes = new byte[w * h * 3];
            if (labelInts.length != w * h) labelInts = new int[w * h];
            yuv.get(0, 0, yuvBytes);
            labels.get(0, 0, labelInts);

            final int shares = (pool == null || n < PARALLEL_MIN_BLOBS) ? 1 : WORKERS + 1;
            final int total = n;
            pending.clear();
            for (int k = 1; k < shares; k++) {
                final int share = k;
                pending.add(pool.submit(() -> score(share, shares, total, w, h, sc)));
            }
            score(0, shares, total, w, h, sc);
            for (Future<?> fu : pending) {
                try {
                    fu.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("blob scoring interrupted", e);
                } catch (ExecutionException e) {
                    throw new IllegalStateException("blob scoring failed", e.getCause());
                }
            }

            for (int j = 0; j < n; j++) {
                if (prob[j] < BlobSvm.THRESHOLD) continue;
                int i = todo[j];
                out.add(new BallCandidate(ce[2 * i], ce[2 * i + 1], prob[j]));
            }
        }
        return out;
    }

    /** Scores blobs share, share + shares, ... of the to-do list. Shares touch disjoint slots of prob. */
    private void score(int share, int shares, int n, int w, int h, int sc) {
        BlobFeatures bf = features[share];
        double[] x = f[share];
        for (int j = share; j < n; j += shares) {
            int i = todo[j];
            int b = i * sc;
            boolean ok = bf.compute(yuvBytes, labelInts, w, h, i,
                    st[b + Imgproc.CC_STAT_LEFT], st[b + Imgproc.CC_STAT_TOP],
                    st[b + Imgproc.CC_STAT_WIDTH], st[b + Imgproc.CC_STAT_HEIGHT],
                    st[b + Imgproc.CC_STAT_AREA], x);
            prob[j] = ok ? BlobSvm.probability(x) : 0.0;
            if (recording) {
                int o = j * BlobSvm.N_FEATURES;
                for (int k = 0; k < BlobSvm.N_FEATURES; k++) {
                    featLog[o + k] = ok ? x[k] : Double.NaN;
                }
            }
        }
    }

    // ---------------------------------------------------------------- run log

    /** Keep each scored blob's features for {@link #lastScoredBlob}. Costs a copy per blob. */
    public void setRecording(boolean on) {
        recording = on;
    }

    /** Connected components in the last mask, background excluded. */
    public int lastComponents() { return lastComponents; }

    /** Of those, how many were within the area limits. */
    public int lastInRange() { return lastInRange; }

    /** Of those, how many the occupancy grid dropped as static clutter. */
    public int lastStatic() { return lastStatic; }

    /** How many blobs the SVM scored on the last frame. */
    public int lastScored() { return lastScored; }

    /**
     * Scored blob {@code j} of the last frame, into {@code out}: centroid x, y,
     * area, box left, top, width, height, P(ball), then the features in
     * {@link BlobSvm#FEATURES} order (NaN when not computed or not recording).
     * Valid until the next {@link #findCandidates}.
     *
     * @param out at least {@code 8 + BlobSvm.N_FEATURES} long
     */
    public void lastScoredBlob(int j, double[] out) {
        int i = todo[j];
        int b = i * stats.cols();
        out[0] = ce[2 * i];
        out[1] = ce[2 * i + 1];
        out[2] = st[b + Imgproc.CC_STAT_AREA];
        out[3] = st[b + Imgproc.CC_STAT_LEFT];
        out[4] = st[b + Imgproc.CC_STAT_TOP];
        out[5] = st[b + Imgproc.CC_STAT_WIDTH];
        out[6] = st[b + Imgproc.CC_STAT_HEIGHT];
        out[7] = prob[j];
        for (int k = 0; k < BlobSvm.N_FEATURES; k++) {
            out[8 + k] = recording ? featLog[j * BlobSvm.N_FEATURES + k] : Double.NaN;
        }
    }

    /** Forgets the static clutter; call when the camera's view changes (background reset). */
    public void reset() {
        grid = new OccupancyGrid();
    }

    public void release() {
        if (pool != null) {
            pool.shutdownNow();
        }
        labels.release();
        stats.release();
        centroids.release();
    }
}
