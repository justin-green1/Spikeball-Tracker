package com.justin.spikeballtracker.vision;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

public class MotionDetector {

    private final BackgroundSubtractorMOG2 subtractor;
    private final Mat frameMat;
    private final Mat foregroundMask;
    private final Mat displayMat;

    public MotionDetector(){
        this.subtractor = Video.createBackgroundSubtractorMOG2();
        this.frameMat = new Mat();
        this.foregroundMask = new Mat();
        this.displayMat = new Mat();
    }

    public Bitmap processFrame(Bitmap frame){

        Utils.bitmapToMat(frame, frameMat);
        subtractor.apply(frameMat, foregroundMask);
        Imgproc.cvtColor(foregroundMask, displayMat, Imgproc.COLOR_GRAY2RGBA);

        Bitmap outputBitmap = Bitmap.createBitmap(frame.getWidth(), frame.getHeight(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(displayMat, outputBitmap);

        return outputBitmap;
    }

    public void release() {
        this.frameMat.release();
        this.foregroundMask.release();
        this.displayMat.release();
        this.subtractor.clear();
    }

}
