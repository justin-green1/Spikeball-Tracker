package com.justin.spikeballtracker.vision;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.Video;

public class MotionDetector {

    private BackgroundSubtractorKNN subtractor;
    private static final int HISTORY = 3600;
    private static final double DIST_2_THRESHOLD = 100;
    private static final boolean DETECT_SHADOWS = true;
    private static final Size BLUR_KERNEL = new Size(9, 9);
    private final Mat blurredFrame;

    /** This frame's mask: 255 motion, 127 shadow, 0 still. */
    private final Mat maskWithShadows;

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
        this.maskWithShadows = new Mat();
        this.blurredFrame = new Mat();
    }

    private static BackgroundSubtractorKNN newSubtractor() {
        return Video.createBackgroundSubtractorKNN(HISTORY, DIST_2_THRESHOLD, DETECT_SHADOWS);
    }

    /**
     * Blurs the frame with the 9x9 pre-blur, then feeds it to the KNN
     * background subtractor, updating the model.
     *
     * <p>The mask keeps shadow pixels (127) rather than thresholding them
     * away: the blob SVM's training blobs were cut from this unthresholded
     * mask, so shadow pixels count as foreground here too.
     *
     * @param frame CV_8UC3 Y,U,V frame at the working resolution
     * @return foreground mask (255 motion, 127 shadow, 0 still), owned by this
     *         detector and overwritten on the next call
     */
    public Mat processFrame(Mat frame) {
        Imgproc.GaussianBlur(frame, blurredFrame, BLUR_KERNEL, 0);
        subtractor.apply(blurredFrame, maskWithShadows);
        return maskWithShadows;
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
        this.maskWithShadows.release();
        this.subtractor.clear();
        this.blurredFrame.release();
    }
}
