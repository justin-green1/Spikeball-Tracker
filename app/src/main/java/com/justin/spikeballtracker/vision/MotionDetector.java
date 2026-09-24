package com.justin.spikeballtracker.vision;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

public class MotionDetector {

    private static final String TAG = "MotionDetector";
    private BackgroundSubtractorKNN subtractor;
    private static final int HISTORY = 3600;
    private static final double DIST_2_THRESHOLD = 100;
    private static final boolean DETECT_SHADOWS = true;
    private static final Size BLUR_KERNEL = new Size(9, 9);
    private final Mat blurredFrame;
    private final Mat foregroundMask;

    /**
     * Performs KNN background subtraction on a sequence of video frames.
     *
     * <p>Maintains a statistical background model across calls to
     * {@link #processFrame(Mat)}, so a single instance must be reused for
     * all frames of a video. Callers must invoke {@link #release()} when
     * finished to free native OpenCV resources.
     */
    public MotionDetector(){
        this.subtractor = newSubtractor();
        this.foregroundMask = new Mat();
        this.blurredFrame = new Mat();
    }

    private static BackgroundSubtractorKNN newSubtractor() {
        return Video.createBackgroundSubtractorKNN(HISTORY, DIST_2_THRESHOLD, DETECT_SHADOWS);
    }

    /**
     * Applies MOG2 background subtraction to the frame
     *
     * @param frame video frame which background subtraction is applied to
     * @return output Mat frame which has had background subtraction performed on it as Bitmap.
     */
    public Mat processFrame(Mat frame) {
        Imgproc.GaussianBlur(frame, blurredFrame, BLUR_KERNEL, 0);
        subtractor.apply(blurredFrame, foregroundMask);
        return foregroundMask;
    }

    /**
     * The KNN settings in force. Exposed so a run's provenance can be recorded
     * alongside exported frames, and so offline work can rebuild the same model.
     */
    public static int history() {
        return HISTORY;
    }

    public static double dist2Threshold() {
        return DIST_2_THRESHOLD;
    }

    public static boolean detectShadows() {
        return DETECT_SHADOWS;
    }

    /** Width of the Gaussian kernel applied before every subtractor.apply. */
    public static int preBlurKernel() {
        return (int) BLUR_KERNEL.width;
    }

    public void getBackgroundImage(Mat background) {
        subtractor.getBackgroundImage(background);
    }

    /**
     * Throws the learned background away and starts a fresh model.
     *
     * <p>Needed when the camera moves: every pixel's history then describes a
     * scene that is no longer in front of the lens, and with HISTORY frames of
     * evidence behind it the model would take that long to forget on its own.
     *
     * <p>The subtractor is replaced rather than cleared, because OpenCV's
     * Algorithm.clear() does not reset KNN's sample history — only a new
     * instance does. The old one is freed once nothing references it.
     */
    public void reset() {
        this.subtractor = newSubtractor();
    }

    /**
     * releases all object fields
     */
    public void release() {
        this.foregroundMask.release();
        this.subtractor.clear();
        this.blurredFrame.release();
    }
}
