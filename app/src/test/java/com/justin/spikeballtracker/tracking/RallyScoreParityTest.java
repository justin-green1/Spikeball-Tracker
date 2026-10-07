package com.justin.spikeballtracker.tracking;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Confirmed rallies on S2C05, against rally_mirror.py. All 21 reference
 * rallies must be confirmed during the run (4.1-5.8 s after each ends), not
 * only by finish().
 *
 * <p>The per-frame score check went with the detector's test-only score hook;
 * the rallies depend on every score, so a score error still shows up here.
 */
public class RallyScoreParityTest {

    private static List<String[]> rows(String resource) throws Exception {
        InputStream in = RallyScoreParityTest.class.getClassLoader().getResourceAsStream(resource);
        assertNotNull("missing test resource " + resource, in);
        List<String[]> out = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            r.readLine();
            for (String l; (l = r.readLine()) != null; ) if (!l.isEmpty()) out.add(l.split(",", -1));
        }
        return out;
    }

    @Test
    public void ralliesMatchPython() throws Exception {
        List<String[]> det  = rows("s2c05/detections_narrow.csv");
        List<String[]> ball = rows("s2c05/reference_velocity_seed.csv");
        List<String[]> refR = rows("s2c05/reference_rallies.csv");

        RallyDetector rd = new RallyDetector();

        for (int i = 0; i < det.size(); i++) {
            String[] d = det.get(i);
            int n = d.length;
            NetZone zone = d[n - 4].isEmpty() ? null : NetZone.around(
                    Double.parseDouble(d[n - 4]), Double.parseDouble(d[n - 3]),
                    Double.parseDouble(d[n - 2]), Double.parseDouble(d[n - 1]));
            int g = i - BallTracker.DELAY;
            BallTracker.Output o = null;
            if (g >= 0) {
                String[] b = ball.get(g);
                boolean has = !b[3].isEmpty();
                o = new BallTracker.Output(g, Long.parseLong(det.get(g)[1]), has,
                        has ? Double.parseDouble(b[3]) : 0, has ? Double.parseDouble(b[4]) : 0);
            }
            rd.process(o, zone);
        }
        // Live, a rally has to be confirmed while frames are still arriving,
        // not only when the run ends. Counted before finish() for that reason.
        int confirmedLive = rd.rallies().size();
        rd.finish();

        // Rallies: same count, same edges.
        List<RallyDetector.Rally> rallies = rd.rallies();
        for (int k = 0; k < Math.min(rallies.size(), refR.size()); k++) {
            assertEquals("rally " + k + " start", Long.parseLong(refR.get(k)[0]), rallies.get(k).startUs);
            assertEquals("rally " + k + " end",   Long.parseLong(refR.get(k)[1]), rallies.get(k).endUs);
        }
        assertEquals("number of rallies", refR.size(), rallies.size());
        assertEquals("rallies confirmed before finish()", refR.size(), confirmedLive);
    }
}