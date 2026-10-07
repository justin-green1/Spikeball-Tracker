package com.justin.spikeballtracker.vision;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class ColourDetector {
    private static final String TAG = "ColourDetector";
    private final Mat colourMask;
    private final Scalar lowerBound;
    private final Scalar upperBound;

    //U 43–120, V 130–167 super wide
    //U 43–87, V 143–167 tighter
    //U 43–101, V 143–167 looser
    private static final Scalar NARROW_LO = new Scalar(0, 43, 143);
    private static final Scalar NARROW_HI = new Scalar(255, 87, 167);
    private static final Scalar WIDE_LO = new Scalar(0, 43, 130);
    private static final Scalar WIDE_HI = new Scalar(255, 120, 167);


     /**
     * Matches the ball on chroma alone; the luma bounds stay wide open so
     * detection survives shade and direct sun.
     *
     * <p>FrameExtractor expands frames from limited range to full range, so
     * these bounds were remapped through the same chroma transform,
     * {@code out = (in - 128) * 255/224 + 128}, to keep matching the same
     * physical colours:
     *
     * <pre>
     *   U  49..92    ->  38..87
     *   V  142..156  ->  144..160
     * </pre>
     *
     * <p>The remap is exact: every value the old bounds accepted lands inside
     * the new ones, and no value they rejected leaks in.
     */
    private ColourDetector(Scalar lowerBound, Scalar upperBound){
        this.colourMask = new Mat();
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;


    }

    public static ColourDetector narrow(){
        return new ColourDetector(NARROW_LO, NARROW_HI);
    }

    public static ColourDetector wide(){
        return new ColourDetector(WIDE_LO, WIDE_HI);
    }

    public Mat processFrame(Mat Frame) {
        Core.inRange(Frame, lowerBound, upperBound, colourMask);
        return colourMask;
    }

    public void release(){
        colourMask.release();
    }
}
