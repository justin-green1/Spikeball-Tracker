package com.justin.spikeballtracker.vision;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * BlobSvm against scikit-learn on 420 feature rows: 400 real blobs (labelled
 * and unlabelled, from blob_features.csv and all_blob_features.csv) and 20
 * with one feature set to NaN to exercise the imputer. Expected values are
 * svm_final.pkl's decision_function and predict_proba(X)[:, 1].
 */
public class BlobSvmParityTest {

    @Test
    public void matchesSklearn() throws Exception {
        InputStream in = getClass().getClassLoader().getResourceAsStream("blob_svm/parity.csv");
        assertNotNull("missing test resource blob_svm/parity.csv", in);
        int rows = 0;
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            r.readLine();
            for (String l; (l = r.readLine()) != null; ) {
                if (l.isEmpty()) continue;
                String[] s = l.split(",", -1);
                double[] x = new double[BlobSvm.N_FEATURES];
                for (int j = 0; j < x.length; j++) x[j] = Double.parseDouble(s[j]);
                double dec = Double.parseDouble(s[BlobSvm.N_FEATURES]);
                double p = Double.parseDouble(s[BlobSvm.N_FEATURES + 1]);
                assertEquals("decision, row " + rows, dec, BlobSvm.decision(x), 1e-9);
                assertEquals("probability, row " + rows, p, BlobSvm.probability(x), 1e-9);
                rows++;
            }
        }
        assertEquals(420, rows);
    }
}
