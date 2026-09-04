package com.justin.spikeballtracker.vision;

import org.opencv.core.Mat;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

public class NetDetector {
    RotatedRect ellipsePos = new RotatedRect();
    public NetDetector(Mat background){
         ellipsePos = Imgproc.fitEllipseDirect(background);
    }

    public RotatedRect getEllipsePos() {
        return ellipsePos;
    }
}
