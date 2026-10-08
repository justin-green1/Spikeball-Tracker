package com.justin.spikeballtracker.vision;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.DataInputStream;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;

/**
 * BlobFeatures against the study's own feature code (blob_features.py via the
 * crop-ring silhouette, run on the same pixels) for 222 blobs: 150 from
 * S3C01 frames as the phone decodes them, 50 synthetic odd shapes (rings, C
 * shapes, speckle, frame edges), plus every blob where the two disagree.
 *
 * <p>Each blob is stored as the patch of frame around it (its box plus the
 * 6 px the surround band can reach, clipped where the real frame ends), so the
 * features are the same as on the whole frame.
 *
 * <p>Colour, luma and contrast must match exactly. Aspect and fill must match
 * too, except on "tie" blobs: two rectangle orientations have exactly the same
 * area and OpenCV's minAreaRect picks the other one (22 of 3 250 blobs
 * tested). Those are counted rather than compared; at most one of them may
 * land on the other side of a threshold.
 */
public class BlobFeaturesParityTest {

    @Test
    public void matchesStudyFeatures() throws Exception {
        InputStream raw = getClass().getClassLoader().getResourceAsStream("blob_svm/blob_features_fixture.bin.gz");
        assertNotNull("missing test resource blob_svm/blob_features_fixture.bin.gz", raw);
        BlobFeatures bf = new BlobFeatures(new int[]{0, 35, 126}, new int[]{255, 128, 171});
        double[] f = new double[BlobSvm.N_FEATURES];
        int checked = 0, ties = 0, decisionFlips = 0;
        try (DataInputStream in = new DataInputStream(new GZIPInputStream(raw))) {
            int n = in.readInt();
            for (int k = 0; k < n; k++) {
                int w = in.readInt(), h = in.readInt();
                int left = in.readInt(), top = in.readInt(), bw = in.readInt(), bh = in.readInt();
                int area = in.readInt();
                byte[] yuv = new byte[w * h * 3];
                in.readFully(yuv);
                byte[] mask = new byte[w * h];
                in.readFully(mask);
                int[] labels = new int[w * h];
                for (int i = 0; i < labels.length; i++) labels[i] = mask[i];
                double[] want = new double[BlobSvm.N_FEATURES];
                for (int j = 0; j < want.length; j++) want[j] = in.readDouble();
                boolean tie = in.readBoolean();
                double pWant = in.readDouble();

                assertTrue("blob " + k + " skipped", bf.compute(yuv, labels, w, h, 1,
                        left, top, bw, bh, area, f));
                for (int j = 2; j < BlobSvm.N_FEATURES; j++) {
                    assertEquals(BlobSvm.FEATURES[j] + ", blob " + k, want[j], f[j], 1e-9);
                }
                if (tie) {
                    ties++;
                } else {
                    assertEquals("aspect, blob " + k, want[0], f[0], 1e-5);
                    assertEquals("fill, blob " + k, want[1], f[1], 1e-5);
                    assertEquals("p, blob " + k, pWant, BlobSvm.probability(f), 1e-5);
                }
                double p = BlobSvm.probability(f);
                if ((p >= BlobSvm.WIDE_THRESHOLD) != (pWant >= BlobSvm.WIDE_THRESHOLD)
                        || (p >= BlobSvm.NARROW_THRESHOLD) != (pWant >= BlobSvm.NARROW_THRESHOLD)) {
                    decisionFlips++;
                }
                checked++;
            }
        }
        assertEquals(222, checked);
        assertEquals(22, ties);
        assertTrue("decision flips " + decisionFlips, decisionFlips <= 1);
    }
}
