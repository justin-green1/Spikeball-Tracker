package com.justin.spikeballtracker.tracking;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Feeds the app's own S2C05 detections through BallTracker and compares every
 * frame with the Python reference (java_mirror.py, cross-checked against the
 * session's trackers.fused).
 */
public class BallTrackerParityTest {

    /** Pick the file that matches your plausibleThird. */
    //private static final String REFERENCE = "s2c05/reference_python_seed.csv";
    private static final String REFERENCE = "s2c05/reference_velocity_seed.csv";

    /** "(463.0, 362.5)" -> x, y. Areas and net columns contain no parentheses. */
    private static final Pattern POINT = Pattern.compile("\\(([-0-9.]+), ([-0-9.]+)\\)");

    private static final class Row {
        int frame; long tUs; List<BallCandidate> cands = new ArrayList<>();
    }

    private static List<String> lines(String resource) throws Exception {
        InputStream in = BallTrackerParityTest.class.getClassLoader().getResourceAsStream(resource);
        assertNotNull("missing test resource " + resource, in);
        List<String> out = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            r.readLine();                                    // header
            for (String l; (l = r.readLine()) != null; ) if (!l.isEmpty()) out.add(l);
        }
        return out;
    }

    private static List<Row> detections(String resource) throws Exception {
        List<Row> rows = new ArrayList<>();
        for (String l : lines(resource)) {
            String[] head = l.split(",", 3);
            Row row = new Row();
            row.frame = Integer.parseInt(head[0]);
            row.tUs = Long.parseLong(head[1]);
            Matcher m = POINT.matcher(l);
            while (m.find()) {
                double x = Double.parseDouble(m.group(1)), y = Double.parseDouble(m.group(2));
                row.cands.add(new BallCandidate(x, y, 1.0));
            }
            rows.add(row);
        }
        return rows;
    }

    @Test
    public void ballOutputMatchesPython() throws Exception {
        List<Row> narrow = detections("s2c05/detections_narrow.csv");
        List<Row> wide   = detections("s2c05/detections_wide.csv");
        List<String> ref = lines(REFERENCE);

        BallTracker tracker = new BallTracker();
        for (int i = 0; i < narrow.size(); i++) {
            Row n = narrow.get(i), w = wide.get(i);
            tracker.process(n.frame - 1, n.tUs, n.cands, w.cands);   // pipeline frameIndex is 0-based

            BallTracker.Output out = tracker.latest();
            if (out != null) {
                String[] er = ref.get(out.frameIndex).split(",", -1);
                boolean expHas = !er[3].isEmpty();
                boolean bad = out.hasBall != expHas
                        || (expHas && (Math.abs(out.x - Double.parseDouble(er[3])) > 1e-3
                        || Math.abs(out.y - Double.parseDouble(er[4])) > 1e-3));
                if (bad) {
                    fail("ball mismatch at Frame " + (out.frameIndex + 1) + ": got "
                            + (out.hasBall ? "(" + out.x + ", " + out.y + ")" : "none")
                            + ", expected " + (expHas ? "(" + er[3] + ", " + er[4] + ")" : "none"));
                }
            }
        }
    }
}