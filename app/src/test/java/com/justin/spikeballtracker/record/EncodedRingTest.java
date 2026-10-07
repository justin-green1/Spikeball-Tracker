package com.justin.spikeballtracker.record;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import org.junit.Test;

import java.nio.ByteBuffer;

/**
 * The two methods a rally copy relies on: {@link EncodedRing#planRange} picks
 * the right packets, and {@link EncodedRing#snapshot} copies them intact,
 * including a packet split across the end of the byte array.
 *
 * <p>Test stream: one packet every 100 ms, a keyframe every 10th (every
 * second), packet k is k+1 bytes all equal to (byte) k.
 */
public class EncodedRingTest {

    private static final long STEP_US = 100_000L;

    private static EncodedRing filled(int capacityBytes, int packets) {
        EncodedRing ring = new EncodedRing(capacityBytes, 1000, Long.MAX_VALUE);
        for (int k = 0; k < packets; k++) {
            byte[] b = new byte[k + 1];
            java.util.Arrays.fill(b, (byte) k);
            ring.add(ByteBuffer.wrap(b), k * STEP_US, k % 10 == 0);
        }
        return ring;
    }

    @Test
    public void startsOnTheKeyframeAtOrBeforeTheStart() {
        EncodedRing ring = filled(1 << 20, 50);
        // 2.35 s to 3.2 s: keyframe at 2.0 s (packet 20) to packet 32.
        EncodedRing.Plan p = ring.planRange(2_350_000L, 3_200_000L);
        assertNotNull(p);
        assertEquals(20, p.startSeq);
        assertEquals(33, p.endSeq);
        assertEquals(2_000_000L, p.firstPtsUs);
        assertEquals(3_200_000L, p.lastPtsUs);
    }

    @Test
    public void aStartOnAKeyframeStartsThere() {
        EncodedRing ring = filled(1 << 20, 50);
        EncodedRing.Plan p = ring.planRange(3_000_000L, 3_500_000L);
        assertEquals(30, p.startSeq);
    }

    @Test
    public void anEndPastTheNewestPacketStopsAtTheNewest() {
        EncodedRing ring = filled(1 << 20, 50);
        EncodedRing.Plan p = ring.planRange(4_000_000L, 9_000_000L);
        assertEquals(40, p.startSeq);
        assertEquals(50, p.endSeq);
        assertEquals(4_900_000L, p.lastPtsUs);
    }

    @Test
    public void aStartBeforeTheOldestPacketUsesTheOldestKeyframe() {
        // 2000 bytes holds only the newest packets, so the oldest have been evicted.
        EncodedRing ring = filled(2000, 70);
        EncodedRing.Plan p = ring.planRange(0L, 6_900_000L);
        assertNotNull(p);
        assertEquals(0, p.startSeq % 10);              // a keyframe
        assertEquals(70, p.endSeq);
        // ...and the first keyframe still held, not a later one.
        EncodedRing.Plan whole = ring.planRange(Long.MIN_VALUE, Long.MAX_VALUE);
        assertEquals(whole.startSeq, p.startSeq);
    }

    @Test
    public void nothingBeforeTheFirstKeyframeIsNull() {
        EncodedRing ring = filled(1 << 20, 50);
        assertNull(ring.planRange(-5_000_000L, -1L));
    }

    @Test
    public void snapshotCopiesEveryPacketIntact() {
        EncodedRing ring = filled(1 << 20, 50);
        EncodedRing.Snapshot s = ring.snapshot(ring.planRange(2_350_000L, 3_200_000L));
        assertNotNull(s);
        assertEquals(13, s.count());
        for (int i = 0; i < s.count(); i++) {
            int k = 20 + i;
            assertEquals(k + 1, s.size[i]);
            assertEquals(k * STEP_US, s.ptsUs[i]);
            assertEquals(k % 10 == 0, s.keyFrame[i]);
            byte[] expected = new byte[k + 1];
            java.util.Arrays.fill(expected, (byte) k);
            byte[] got = new byte[s.size[i]];
            System.arraycopy(s.data, s.offset[i], got, 0, s.size[i]);
            assertArrayEquals("packet " + k, expected, got);
        }
        assertEquals(1_200_000L, s.durationUs());
    }

    @Test
    public void snapshotSurvivesAPacketSplitAcrossTheEnd() {
        // 7-byte packets in a 100-byte ring: packet k is written at 7k mod 100,
        // so packet 28 sits at 96..102 and is split across the end. The ring
        // then holds packets 16..29; keyframes every third packet.
        EncodedRing ring = new EncodedRing(100, 1000, Long.MAX_VALUE);
        for (int k = 0; k < 30; k++) {
            byte[] b = new byte[7];
            java.util.Arrays.fill(b, (byte) k);
            ring.add(ByteBuffer.wrap(b), k * STEP_US, k % 3 == 0);
        }
        EncodedRing.Plan p = ring.planRange(2_500_000L, Long.MAX_VALUE);
        assertNotNull(p);
        assertEquals(24, p.startSeq);
        assertEquals(30, p.endSeq);

        EncodedRing.Snapshot s = ring.snapshot(p);
        assertNotNull(s);
        for (int i = 0; i < s.count(); i++) {
            int k = 24 + i;
            for (int j = 0; j < 7; j++) {
                assertEquals("packet " + k + " byte " + j, (byte) k, s.data[s.offset[i] + j]);
            }
        }
    }

    @Test
    public void snapshotOfEvictedPacketsIsNull() {
        EncodedRing ring = filled(2000, 40);
        EncodedRing.Plan p = ring.planRange(Long.MIN_VALUE, Long.MAX_VALUE);
        // Push enough new packets through to evict the planned ones.
        for (int k = 40; k < 120; k++) {
            ring.add(ByteBuffer.wrap(new byte[k + 1]), k * STEP_US, k % 10 == 0);
        }
        assertNull(ring.snapshot(p));
    }
}
