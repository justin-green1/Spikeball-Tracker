package com.justin.spikeballtracker.tracking;

import java.util.Locale;

public final class BallCandidate {
    public final int frameIndex;
    public final long timestampUs;
    public final double x;
    public final double y;
    public final int area;
    double left;
    double top;
    double width;
    double height;

    /** The blob SVM's P(ball), or NaN when the candidate came from a plain colour box. */
    public final double pBall;

    public BallCandidate(int frameIndex, long timestampUs, double x, double y, int area, double left, double top, double width, double height) {
        this(frameIndex, timestampUs, x, y, area, left, top, width, height, Double.NaN);
    }

    public BallCandidate(int frameIndex, long timestampUs, double x, double y, int area, double left, double top, double width, double height, double pBall) {
        this.frameIndex = frameIndex;
        this.timestampUs = timestampUs;
        this.x = x;
        this.y = y;
        this.area = area;
        this.left = left;
        this.top = top;
        this.width = width;
        this.height = height;
        this.pBall = pBall;
    }

    @Override
    public String toString() {
        if (Double.isNaN(pBall)) {
            return String.format(Locale.US, "f%d (%.1f, %.1f) area=%d", frameIndex, x, y, area);
        }
        return String.format(Locale.US, "f%d (%.1f, %.1f) area=%d p=%.3f", frameIndex, x, y, area, pBall);
    }
}
