package com.justin.spikeballtracker.vision;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class ColourDetector {
    private static final String TAG = "ColourDetector";
    private final Mat colourMask;
    private final Scalar lowerBound;
    private final Scalar upperBound;

    public ColourDetector(){
        this.colourMask = new Mat();
        this.lowerBound = new Scalar(0, 30, 148);
        this.upperBound = new Scalar(255, 94, 185);
    }

    public Mat processFrame(Mat Frame) {
        Core.inRange(Frame, lowerBound, upperBound, colourMask);
        return colourMask;
    }

    public void release(){
        colourMask.release();
    }
}
