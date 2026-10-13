package com.justin.spikeballtracker.tracking;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import org.junit.Test;

import java.util.Collections;
import java.util.List;

public class OccupancyGridTest {

    /** Only x and y matter to the grid; everything else is filler. */
    private static List<BallCandidate> at(double x, double y) {
        return Collections.singletonList(new BallCandidate(x, y, 1.0));
    }

    private static final List<BallCandidate> NONE = Collections.emptyList();

    /**
     * A speck in one place every frame. Its cell passes 32.5 after 52 updates,
     * but the 60-frame warm-up keeps it non-static until frame 61.
     */
    @Test
    public void warmUpHoldsThenSpeckIsStatic() {
        OccupancyGrid g = new OccupancyGrid();
        int firstStatic = -1;
        for (int frame = 1; frame <= 80; frame++) {
            if (firstStatic < 0 && g.isStatic(100, 100)) firstStatic = frame;
            g.update(at(100, 100));                  // ask first, then update
        }
        assertEquals(61, firstStatic);
    }

    /** Present on every other frame: the cell settles near 25, below 32.5. */
    @Test
    public void flickerIsNeverStatic() {
        OccupancyGrid g = new OccupancyGrid();
        for (int i = 0; i < 400; i++) {
            boolean present = (i % 2 == 0);
            if (present) assertFalse("static at frame " + i, g.isStatic(100, 100));
            g.update(present ? at(100, 100) : NONE);
        }
    }

    /** Moving 20 px a frame (wrapping inside the frame): no cell is hit often enough. */
    @Test
    public void moverIsNeverStatic() {
        OccupancyGrid g = new OccupancyGrid();
        for (int i = 0; i < 400; i++) {
            double x = 20 + (20 * i) % 800;
            assertFalse("static at frame " + i, g.isStatic(x, 240));
            g.update(at(x, 240));
        }
    }

    /** Static for 100 frames, then gone: 43.37 decays below 32.5 after 15 empty frames. */
    @Test
    public void staticFadesAfter15EmptyFrames() {
        OccupancyGrid g = new OccupancyGrid();
        for (int i = 0; i < 100; i++) g.update(at(100, 100));

        int stillStatic = 0;
        while (g.isStatic(100, 100)) {
            g.update(NONE);
            stillStatic++;
            if (stillStatic > 1000) break;           // guard against an infinite loop if decay is broken
        }
        assertEquals(15, stillStatic);
    }

    /** A cell's count must not leak into its neighbour. */
    @Test
    public void neighbouringCellIsUnaffected() {
        OccupancyGrid g = new OccupancyGrid();
        for (int i = 0; i < 100; i++) g.update(at(100, 100));   // cell (6, 6)
        assertFalse(g.isStatic(100 + 16, 100));                 // cell (6, 7)
    }

    /** Out-of-frame coordinates are clamped, not a crash. */
    @Test
    public void outOfFrameDoesNotCrash() {
        OccupancyGrid g = new OccupancyGrid();
        g.update(at(-5, -5));
        g.update(at(900, 500));
        g.isStatic(-5, 900);
    }
}