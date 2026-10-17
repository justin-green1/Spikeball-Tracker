package com.justin.spikeballtracker.source;

import org.opencv.core.Mat;

/**
 * Where a source hands over a frame that is ready to show.
 *
 * <p>Called on the Analysis thread, never the UI thread. The Mat is
 * CV_8UC3 RGB, is owned by the source, and may be overwritten as soon as the
 * call returns — copy anything you need to keep.
 */
public interface FrameSink {
    void onDisplayFrame(Mat displayMat);
}