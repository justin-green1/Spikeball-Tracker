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
     * The box in front of the blob SVM ("box B", 15 Oct 2026). Deliberately
     * loose: the SVM decides what is a ball inside it.
     *
     * <p>It started as the wide box widened by 10 % of its span on each side,
     * (0, 35, 126)-(255, 128, 171), the box the SVM's first training crops were
     * cut with. That box passed ~36 % of every frame: its V floor and U ceiling
     * reached the grass (U ~87, V 126-131) and the near-neutral greys
     * (U, V ~128). Box B keeps the widened U floor and V ceiling, which no
     * background reaches, and tightens only those two edges:
     *
     * <pre>
     *   U 35-128 -> 35-124,  V 126-171 -> 129-171
     * </pre>
     *
     * <p>Measured with the 40 px area floor on 197 phone frames (S1-S5) and the
     * 4 651 hand-labelled study blobs: frame pass 35.5 % -> 24.3 %, labelled
     * not-ball blobs removed 40 %, labelled ball blobs kept 98.1 % (worst
     * session S5, 95.3 %). Box A, U 35-126 and V 128-171, kept more small
     * balls (99.2 %, S5 97.6 %) but passed 28.2 % of every frame and removed
     * only 26 % of not-balls.
     *
     * <p>The SVM was trained on blobs cut with box A, which contains box B, so
     * in the app it sees a subset of the colours it was trained on. The tighter
     * box trims some ball edges, which can shift the shape and colour features
     * slightly from training.
     */
    public static final Scalar LO = new Scalar(0, 35, 129);
    public static final Scalar HI = new Scalar(255, 124, 171);
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
