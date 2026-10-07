package com.justin.spikeballtracker.record;

import android.content.Context;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.net.Uri;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * A rally from a played video: a time range of the source file, which is
 * already permanent, so nothing is copied until it is saved. Saving remuxes
 * the video track from the keyframe at or before {@code fromUs} up to
 * {@code toUs}; no re-encode, no audio.
 */
public final class FileRally implements RallyFootage {

    /** Read buffer when the file does not say how big its samples get. */
    private static final int DEFAULT_SAMPLE_BYTES = 4 * 1024 * 1024;

    private final Context appContext;
    private final Uri videoUri;
    private final long fromUs;
    private final long toUs;

    /**
     * @param fromUs start of the clip, presentation time in the source video
     * @param toUs   end of the clip, same clock; past the end of the file is fine
     */
    public FileRally(Context context, Uri videoUri, long fromUs, long toUs) {
        this.appContext = context.getApplicationContext();
        this.videoUri = videoUri;
        this.fromUs = fromUs;
        this.toUs = toUs;
    }

    @Override
    public long writeTo(MediaMuxer muxer) throws IOException {
        MediaExtractor extractor = new MediaExtractor();
        try {
            extractor.setDataSource(appContext, videoUri, null);
            int source = findVideoTrack(extractor);
            if (source < 0) {
                throw new IOException("no video track in " + videoUri);
            }
            extractor.selectTrack(source);
            MediaFormat format = extractor.getTrackFormat(source);

            if (format.containsKey(MediaFormat.KEY_ROTATION)) {
                muxer.setOrientationHint(format.getInteger(MediaFormat.KEY_ROTATION));
            }
            int track = muxer.addTrack(format);

            int capacity = format.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)
                    ? Math.max(format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE), 64 * 1024)
                    : DEFAULT_SAMPLE_BYTES;
            ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();

            // Start on a keyframe so the clip decodes from its first frame.
            extractor.seekTo(fromUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC);

            long firstUs = -1;
            long lastUs = -1;
            for (long t = extractor.getSampleTime(); t >= 0 && t <= toUs; t = extractor.getSampleTime()) {
                int n = extractor.readSampleData(buffer, 0);
                if (n < 0) {
                    break;
                }
                if (firstUs < 0) {
                    firstUs = t;
                    muxer.start();   // only once there is something to write
                }
                boolean key = (extractor.getSampleFlags() & MediaExtractor.SAMPLE_FLAG_SYNC) != 0;
                info.set(0, n, t - firstUs, key ? MediaCodec.BUFFER_FLAG_KEY_FRAME : 0);
                muxer.writeSampleData(track, buffer, info);
                lastUs = t;
                extractor.advance();
            }
            if (firstUs < 0) {
                throw new IOException("no frames between " + fromUs + " and " + toUs + " us");
            }

            muxer.stop();
            return lastUs - firstUs;
        } finally {
            extractor.release();
        }
    }

    private static int findVideoTrack(MediaExtractor extractor) {
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            String mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("video/")) {
                return i;
            }
        }
        return -1;
    }
}
