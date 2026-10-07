package com.justin.spikeballtracker.record;

import android.media.MediaCodec;
import android.media.MediaFormat;
import android.media.MediaMuxer;

import java.nio.ByteBuffer;

/**
 * A camera rally: its encoded packets copied out of the replay ring, with the
 * stream headers and rotation they need. Independent of the ring, so the
 * ring can keep overwriting itself, and even be cleared by a new camera
 * session, without touching it.
 *
 * <p>Immutable, so a save can read it on one thread while a newer rally
 * replaces it on another.
 */
public final class BufferedRally implements RallyFootage {

    private final EncodedRing.Snapshot packets;
    private final MediaFormat format;
    private final int rotationDegrees;

    BufferedRally(EncodedRing.Snapshot packets, MediaFormat format, int rotationDegrees) {
        this.packets = packets;
        this.format = format;
        this.rotationDegrees = rotationDegrees;
    }

    /** Length held, first to last frame. */
    public long durationUs() {
        return packets.durationUs();
    }

    /** Bytes held, for the log. */
    public int sizeBytes() {
        return packets.data.length;
    }

    @Override
    public long writeTo(MediaMuxer muxer) {
        muxer.setOrientationHint(rotationDegrees);
        int track = muxer.addTrack(format);
        muxer.start();

        // A direct buffer, as the old replay save used; one packet at a time.
        ByteBuffer scratch = ByteBuffer.allocateDirect(Math.max(packets.maxPacketSize(), 64 * 1024));
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        long firstPts = packets.ptsUs[0];

        for (int i = 0; i < packets.count(); i++) {
            scratch.clear();
            scratch.put(packets.data, packets.offset[i], packets.size[i]);
            scratch.flip();
            // Rebased to zero: camera timestamps count from device boot.
            info.set(0, packets.size[i], packets.ptsUs[i] - firstPts,
                    packets.keyFrame[i] ? MediaCodec.BUFFER_FLAG_KEY_FRAME : 0);
            muxer.writeSampleData(track, scratch, info);
        }

        muxer.stop();
        return packets.durationUs();
    }
}
