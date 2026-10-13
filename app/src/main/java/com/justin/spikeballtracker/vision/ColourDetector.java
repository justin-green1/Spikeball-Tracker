package com.justin.spikeballtracker.vision;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

/**
 * Masks the pixels inside the ball's colour box.
 *
 * <p>Matches on chroma alone; the luma bounds stay wide open so detection
 * survives shade and direct sun.
 */
public class ColourDetector {

    /**
     * The colour box the blob SVM's training crops were cut with
     * (studies/classifier/extract_blob_candidates.py). The SVM decides what is
     * a ball inside it, so it is deliberately loose.
     */
    public static final Scalar LO = new Scalar(0, 35, 126);
    public static final Scalar HI = new Scalar(255, 128, 171);

    private final Mat colourMask = new Mat();

    /**
     * @param frame CV_8UC3 Y,U,V frame
     * @return 255 inside the box, 0 outside; owned by this detector and
     *         overwritten on the next call
     */
    public Mat processFrame(Mat frame) {
        Core.inRange(frame, LO, HI, colourMask);
        return colourMask;
    }

    public void release(){
        colourMask.release();
    }
}
