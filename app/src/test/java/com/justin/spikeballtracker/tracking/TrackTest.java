package com.justin.spikeballtracker.tracking;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class TrackTest {

    private static final long T0 = 1_000_000_000L;
    private static final long FRAME = 33_333L;

    /** Points taken exactly from a known curve must give that curve back. */
    @Test
    public void recoversAnExactParabola() {
        Track t = null;
        for (int k = 0; k < 10; k++) {
            long ts = T0 + k * FRAME;
            double u = (ts - T0) / 1e6;
            double x = 200 + 90 * u;
            double y = 300 - 360 * u + 540 * u * u;
            if (t == null) t = new Track(k, ts, x, y); else t.add(k, ts, x, y);
        }
        long future = T0 + 12 * FRAME;
        double u = (future - T0) / 1e6;
        assertEquals(200 + 90 * u,                   t.predictX(future), 1e-6);
        assertEquals(300 - 360 * u + 540 * u * u,    t.predictY(future), 1e-6);

        Track.Shape s = t.shape();
        assertEquals(540.0, s.curvature, 1e-6);
        assertEquals(0.0,   s.rms,       1e-6);
        assertEquals(10,    s.n);
    }

    /** The noisy arc with a dropped frame; numbers computed independently in Python. */
    @Test
    public void matchesPythonOnNoisyArc() {
        int[]    f = {0, 1, 2, 3, 4, 6, 7, 8};
        double[] x = {200.2, 203.2, 206.5, 208.7, 212.2, 218.0, 220.6, 223.7};
        double[] y = {300.5, 287.8, 278.7, 269.7, 261.8, 249.9, 245.3, 242.8};

        Track t = new Track(f[0], T0, x[0], y[0]);
        for (int i = 1; i < f.length; i++) t.add(f[i], T0 + f[i] * FRAME, x[i], y[i]);

        long q = T0 + 10 * FRAME;
        assertEquals(229.546, t.predictX(q), 1e-3);
        assertEquals(240.372, t.predictY(q), 1e-3);

        Track.Shape s = t.shape();
        assertEquals(542.4037, s.curvature, 1e-3);
        assertEquals(0.4392,   s.rms,       1e-3);
        assertEquals(62.302,   s.span,      1e-3);
        assertEquals(236.776,  s.speed,     1e-3);
        assertEquals(8,        s.n);
    }

    /** Under 4 points y must be a straight line. */
    @Test
    public void threePointsFallBackToALine() {
        Track t = new Track(0, T0, 200.2, 300.5);
        t.add(1, T0 + FRAME,     203.2, 287.8);
        t.add(2, T0 + 2 * FRAME, 206.5, 278.7);

        long q = T0 + 3 * FRAME;
        assertEquals(209.600, t.predictX(q), 1e-3);
        assertEquals(267.200, t.predictY(q), 1e-3);
        assertEquals(0.0, t.shape().curvature, 0.0);
    }
}