package com.justin.spikeballtracker.vision;

import com.justin.spikeballtracker.tracking.BallCandidate;

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
 * Connected components of the widened-box ball mask, each scored by
 * {@link BlobSvm}.
 *
 * <p>Returns every component in the area range with its P(ball); the caller
 * splits them into narrow and wide streams by threshold. Components whose
 * features cannot be computed (fewer than 3 in-box pixels, which the study
 * skipped) get P(ball) = 0.
 *
 * <p>The SVM was trained on 40-400 px blobs only. The floor here is
 * {@link #MIN_AREA}, so smaller blobs are scored outside the range the model
 * has seen. Raising it is also the simplest way to cut cost: on S3C01 there are
 * ~140 blobs a frame at 15 px, ~110 at 20, ~60 at 40.
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

    private static final int MIN_AREA = 15;
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
     * @return every component of {@link #MIN_AREA}-{@link #MAX_AREA} px, scored
     */
    public List<BallCandidate> findCandidates(Mat mask, Mat yuv) {
        int count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids);
        List<BallCandidate> out = new ArrayList<>();

        final int sc = stats.cols();
        if (st.length < count * sc) st = new int[count * sc];
        if (ce.length < count * 2) ce = new double[count * 2];
        if (todo.length < count) {
            todo = new int[count];
            prob = new double[count];
        }
        stats.get(0, 0, st);
        centroids.get(0, 0, ce);

        int n = 0;
        for (int i = 1; i < count; i++) {   // label 0 is the background
            int a = st[i * sc + Imgproc.CC_STAT_AREA];
            if (a >= MIN_AREA && a <= MAX_AREA) todo[n++] = i;
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
                int i = todo[j];
                out.add(new BallCandidate(ce[2 * i], ce[2 * i + 1], prob[j]));
            }
        }
        return out;
    }

    /** Scores blobs share, share + shares, ... of the todo list. Shares touch disjoint slots of prob. */
    private void score(int share, int shares, int n, int w, int h, int sc) {
        BlobFeatures bf = features[share];
        double[] x = f[share];
        for (int j = share; j < n; j += shares) {
            int i = todo[j];
            int b = i * sc;
            prob[j] = bf.compute(yuvBytes, labelInts, w, h, i,
                    st[b + Imgproc.CC_STAT_LEFT], st[b + Imgproc.CC_STAT_TOP],
                    st[b + Imgproc.CC_STAT_WIDTH], st[b + Imgproc.CC_STAT_HEIGHT],
                    st[b + Imgproc.CC_STAT_AREA], x)
                    ? BlobSvm.probability(x) : 0.0;
        }
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
