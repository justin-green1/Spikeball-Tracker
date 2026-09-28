package com.justin.spikeballtracker.record;

import java.nio.ByteBuffer;

/**
 * The last few tens of seconds of encoded H.264 access units, held in RAM.
 *
 * <p>Why encoded rather than raw: a raw 1280x720 YUV420 frame is ~1.4 MB, so a
 * minute at 30 fps would be ~2.5 GB. The same minute encoded at 5 Mbit/s is
 * ~38 MB. The ring stores the encoder's output verbatim, so saving is a remux,
 * never a re-encode, and costs no quality.
 *
 * <p>Storage is one preallocated byte array used as a circular buffer, plus a
 * parallel ring of per-packet metadata. A packet may straddle the end of the
 * array; it is simply split in two. Nothing is allocated per frame.
 *
 * <p>Packets are addressed by a monotonically increasing sequence number, so a
 * reader can hold a sequence across calls and find out later whether it has
 * since been evicted, rather than silently reading overwritten bytes.
 *
 * <p>Deliberately free of Android types so it can be unit-tested on a plain JVM.
 * Every public method is synchronized; the writer (encoder thread) and a reader
 * (save thread) only ever hold the lock for one packet's worth of copying.
 */
public final class EncodedRing {

    private final byte[] data;
    private final long retainUs;

    // Per-packet metadata, indexed by seq % maxPackets.
    private final int maxPackets;
    private final int[] offset;
    private final int[] size;
    private final long[] ptsUs;
    private final boolean[] keyFrame;

    private long headSeq = 0;   // oldest packet still held
    private long tailSeq = 0;   // next sequence to be written
    private int headPos = 0;    // byte offset of the oldest packet
    private int usedBytes = 0;

    /**
     * @param capacityBytes size of the byte store; the hard ceiling on memory
     * @param maxPackets    metadata slots; must cover retainUs at the highest frame rate
     * @param retainUs      packets older than this, relative to the newest, are dropped
     */
    public EncodedRing(int capacityBytes, int maxPackets, long retainUs) {
        this.data = new byte[capacityBytes];
        this.maxPackets = maxPackets;
        this.retainUs = retainUs;
        this.offset = new int[maxPackets];
        this.size = new int[maxPackets];
        this.ptsUs = new long[maxPackets];
        this.keyFrame = new boolean[maxPackets];
    }

    /** Drops everything. Sequence numbers keep counting, so stale readers still fail safely. */
    public synchronized void clear() {
        headSeq = tailSeq;
        headPos = 0;
        usedBytes = 0;
    }

    /**
     * Appends one access unit, evicting from the old end as needed.
     *
     * @param src bytes from {@code src.position()} to {@code src.limit()}; the
     *            buffer's position is left unchanged
     * @return false if the packet is larger than the whole ring and was dropped
     */
    public synchronized boolean add(ByteBuffer src, long pts, boolean isKeyFrame) {
        int n = src.remaining();
        if (n > data.length) {
            return false;
        }

        // Evict on time first, then on space and slots.
        while (count() > 0 && pts - ptsUs[slot(headSeq)] > retainUs) {
            evictOldest();
        }
        while (count() > 0 && (usedBytes + n > data.length || count() >= maxPackets)) {
            evictOldest();
        }

        int writePos = (headPos + usedBytes) % data.length;
        ByteBuffer view = src.duplicate();
        int first = Math.min(n, data.length - writePos);
        view.get(data, writePos, first);
        if (first < n) {
            view.get(data, 0, n - first);   // wrapped past the end of the array
        }

        int s = slot(tailSeq);
        offset[s] = writePos;
        size[s] = n;
        ptsUs[s] = pts;
        keyFrame[s] = isKeyFrame;
        if (count() == 0) {
            headPos = writePos;
        }
        usedBytes += n;
        tailSeq++;
        return true;
    }

    /** What a save should mux: packets [startSeq, endSeq). */
    public static final class Plan {
        public final long startSeq;
        public final long endSeq;
        public final long firstPtsUs;
        public final long lastPtsUs;

        Plan(long startSeq, long endSeq, long firstPtsUs, long lastPtsUs) {
            this.startSeq = startSeq;
            this.endSeq = endSeq;
            this.firstPtsUs = firstPtsUs;
            this.lastPtsUs = lastPtsUs;
        }

        public long durationUs() {
            return lastPtsUs - firstPtsUs;
        }
    }

    /**
     * Chooses the packets covering the last {@code durationUs}, starting on a
     * keyframe so the clip decodes from its first frame.
     *
     * <p>Starts at the latest keyframe at or before {@code newest - durationUs},
     * so the clip is at least as long as asked whenever enough is buffered. With
     * less buffered than asked, it starts at the oldest keyframe held.
     *
     * @return null when there is nothing decodable in the ring
     */
    public synchronized Plan plan(long durationUs) {
        if (count() == 0) {
            return null;
        }
        long newestPts = ptsUs[slot(tailSeq - 1)];
        long bound = newestPts - durationUs;

        long start = -1;
        for (long seq = headSeq; seq < tailSeq; seq++) {
            int s = slot(seq);
            if (!keyFrame[s]) {
                continue;
            }
            if (ptsUs[s] <= bound || start < 0) {
                start = seq;          // latest keyframe not after the bound, or the first one
            }
            if (ptsUs[s] > bound) {
                break;
            }
        }
        if (start < 0) {
            return null;
        }
        return new Plan(start, tailSeq, ptsUs[slot(start)], newestPts);
    }

    /** Metadata of one copied packet. */
    public static final class Packet {
        public int size;
        public long ptsUs;
        public boolean keyFrame;
    }

    /**
     * Copies one packet into {@code dst} (cleared first, left flipped for reading).
     *
     * @return false if the packet has been evicted, or was never written
     */
    public synchronized boolean copy(long seq, ByteBuffer dst, Packet out) {
        if (seq < headSeq || seq >= tailSeq) {
            return false;
        }
        int s = slot(seq);
        int n = size[s];
        int pos = offset[s];

        dst.clear();
        int first = Math.min(n, data.length - pos);
        dst.put(data, pos, first);
        if (first < n) {
            dst.put(data, 0, n - first);
        }
        dst.flip();

        out.size = n;
        out.ptsUs = ptsUs[s];
        out.keyFrame = keyFrame[s];
        return true;
    }

    /** Largest single packet held; a save sizes its scratch buffer from this. */
    public synchronized int maxPacketSize() {
        int max = 0;
        for (long seq = headSeq; seq < tailSeq; seq++) {
            max = Math.max(max, size[slot(seq)]);
        }
        return max;
    }

    /** Span between the oldest and newest packet held, in microseconds. */
    public synchronized long bufferedUs() {
        if (count() == 0) {
            return 0;
        }
        return ptsUs[slot(tailSeq - 1)] - ptsUs[slot(headSeq)];
    }

    public synchronized int count() {
        return (int) (tailSeq - headSeq);
    }

    public synchronized int usedBytes() {
        return usedBytes;
    }

    public int capacityBytes() {
        return data.length;
    }

    private void evictOldest() {
        int s = slot(headSeq);
        usedBytes -= size[s];
        headPos = (offset[s] + size[s]) % data.length;
        headSeq++;
        if (count() == 0) {
            headPos = 0;
            usedBytes = 0;
        }
    }

    private int slot(long seq) {
        return (int) (seq % maxPackets);
    }
}
