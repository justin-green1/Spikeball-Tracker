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
     * The box in front of the blob SVM ("box A", 13 Oct 2026). Deliberately
     * loose: the SVM decides what is a ball inside it.
     *
     * <p>It started as the wide box widened by 10 % of its span on each side,
     * (0, 35, 126)-(255, 128, 171), the box the SVM's first training crops were
     * cut with. That box passed ~36 % of every frame: its V floor and U ceiling
     * reached the grass (U ~87, V 126-131) and the near-neutral greys
     * (U, V ~128). Box A keeps the widened U floor and V ceiling, which no
     * background reaches, and tightens only those two edges:
     *
     * <pre>
     *   U 35-128 -> 35-126,  V 126-171 -> 128-171
     * </pre>
     *
     * <p>Measured with the 40 px area floor on 197 phone frames (S1-S5) and the
     * 4 651 hand-labelled study blobs: frame pass 35.5 % -> 28.2 %, labelled
     * not-ball blobs removed 26 %, labelled ball blobs kept 99.2 % (worst
     * session S5, 97.6 %). Tighter boxes removed more background but lost more
     * small balls, e.g. U 35-124, V 129-171: 24.3 % frame pass, 98.1 % of balls
     * kept, S5 95.3 %.
     *
     * <p>The SVM must be trained on blobs cut with this box: a tighter box
     * trims ball edges, which shifts the shape and colour features.
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
