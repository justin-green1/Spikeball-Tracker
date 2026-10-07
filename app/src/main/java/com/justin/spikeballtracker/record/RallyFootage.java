package com.justin.spikeballtracker.record;

import android.media.MediaMuxer;

import java.io.IOException;

/**
 * One confirmed rally's video, held until it is saved or replaced by the
 * next rally. {@link BufferedRally} is a copy out of the camera's replay ring;
 * {@link FileRally} points at a range of the video file being played.
 */
public interface RallyFootage {

    /** Footage kept before the detected rally start. */
    long BEFORE_US = 1_000_000L;

    /** Footage kept after the detected rally end. */
    long AFTER_US = 3_000_000L;

    /**
     * Adds a video track to {@code muxer}, starts it, writes every sample and
     * stops it. The caller creates and releases the muxer.
     *
     * @return the length written, first to last frame, in microseconds
     */
    long writeTo(MediaMuxer muxer) throws IOException;
}
