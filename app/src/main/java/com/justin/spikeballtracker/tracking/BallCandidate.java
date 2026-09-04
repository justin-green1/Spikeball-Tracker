package com.justin.spikeballtracker.tracking;

import java.util.Locale;

public final class BallCandidate {
    public final int frameIndex;
    public final long timestampUs;
    public final double x;
    public final double y;
    public final int area;

    public BallCandidate(int frameIndex, long timestampUs, double x, double y, int area) {
        this.frameIndex = frameIndex;
        this.timestampUs = timestampUs;
        this.x = x;
        this.y = y;
        this.area = area;
    }

    @Override
    public String toString() {
        return String.format(Locale.US, "f%d (%.1f, %.1f) area=%d", frameIndex, x, y, area);
    }
}