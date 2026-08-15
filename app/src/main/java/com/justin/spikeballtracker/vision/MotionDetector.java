package com.justin.spikeballtracker.vision;

import org.opencv.core.Mat;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

public class MotionDetector {

    private static final String TAG = "MotionDetector";
    private final BackgroundSubtractorMOG2 subtractor;
    private final Mat foregroundMask;

    /**
     * Performs MOG2 background subtraction on a sequence of video frames.
     *
     * <p>Maintains a statistical background model across calls to
     * {@link #processFrame(Mat)}, so a single instance must be reused for
     * all frames of a video. Callers must invoke {@link #release()} when
     * finished to free native OpenCV resources.
     */
    public MotionDetector(){
        this.subtractor = Video.createBackgroundSubtractorMOG2();
        this.foregroundMask = new Mat();
    }

    /**
     * Applies MOG2 background subtraction to the frame
     *
     * @param frame video frame which background subtraction is applied to
     * @return output Mat frame which has had background subtraction performed on it as Bitmap.
     */
    public Mat processFrame(Mat frame){
        subtractor.apply(frame, foregroundMask);
        return foregroundMask;
    }

    /**
     * releases all object fields
     */
    public void release() {
        this.foregroundMask.release();
        this.subtractor.clear();
    }

}
