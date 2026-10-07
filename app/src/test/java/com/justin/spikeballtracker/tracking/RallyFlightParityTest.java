package com.justin.spikeballtracker.tracking;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Flights and net links on S2C05, compared with rally_mirror.py, which agrees
 * flight for flight with the session's fused_features.arc_net_links.
 */
public class RallyFlightParityTest {

    private static final int DELAY = BallTracker.DELAY;

    private static List<String[]> rows(String resource) throws Exception {
        InputStream in = RallyFlightParityTest.class.getClassLoader().getResourceAsStream(resource);
        assertNotNull("missing test resource " + resource, in);
        List<String[]> out = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            r.readLine();                                         // header
            for (String l; (l = r.readLine()) != null; ) if (!l.isEmpty()) out.add(l.split(",", -1));
        }
        return out;
    }

    @Test
    public void flightsAndNetLinksMatchPython() throws Exception {
        List<String[]> det  = rows("s2c05/detections_narrow.csv");          // timestamps + the app's net box
        List<String[]> ball = rows("s2c05/reference_velocity_seed.csv");    // the tracker's answers
        List<String[]> ref  = rows("s2c05/reference_flights.csv");

        RallyDetector rd = new RallyDetector();
        for (int i = 0; i < det.size(); i++) {
            String[] d = det.get(i);
            int n = d.length;                                     // net columns are the last four
            NetZone zone = d[n - 4].isEmpty() ? null : NetZone.around(
                    Double.parseDouble(d[n - 4]), Double.parseDouble(d[n - 3]),
                    Double.parseDouble(d[n - 2]), Double.parseDouble(d[n - 1]));

            int g = i - DELAY;                                    // the frame the tracker answers for now
            BallTracker.Output o = null;
            if (g >= 0) {
                String[] b = ball.get(g);
                boolean has = !b[3].isEmpty();
                o = new BallTracker.Output(g, Long.parseLong(det.get(g)[1]), has,
                        has ? Double.parseDouble(b[3]) : 0, has ? Double.parseDouble(b[4]) : 0);
            }
            rd.process(o, zone);
        }

        List<RallyDetector.NetLink> got = rd.flights();
        for (int k = 0; k < Math.min(got.size(), ref.size()); k++) {
            RallyDetector.NetLink f = got.get(k);
            String[] e = ref.get(k);
            String where = "flight " + k + " (starts " + e[0] + " us)";
            assertEquals(where + " start", Long.parseLong(e[0]), f.startUs);
            assertEquals(where + " end", Long.parseLong(e[1]), f.endUs);
            assertEquals(where + " points", Integer.parseInt(e[2]), f.points);
            assertEquals(where + " toNet", e[3].equals("1"), f.toNet);
            assertEquals(where + " fromNet", e[4].equals("1"), f.fromNet);
            if (f.toNet)   assertEquals(where + " contactTo", Long.parseLong(e[5]), f.contactToUs);
            if (f.fromNet) assertEquals(where + " contactFrom", Long.parseLong(e[6]), f.contactFromUs);
        }
        assertEquals("number of finished flights", ref.size(), got.size());
    }
}