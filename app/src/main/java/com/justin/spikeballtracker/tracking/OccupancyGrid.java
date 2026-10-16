package com.justin.spikeballtracker.tracking;

import java.util.List;

/**
 * Remembers where blobs keep appearing, so static clutter can be ignored.
 *
 * <p>The frame is cut into {@value #CELL} px cells. Every frame each cell
 * decays by {@value #DECAY} and gains 1 per blob centred in it, so a cell
 * settles at (blobs per frame) / (1 - DECAY), a memory of about 50 frames. A
 * cell is static once it holds {@value #THRESH} of a blob per frame on average
 * (a blob there on ~65 % of recent frames). Nothing is static for the first
 * {@value #WARMUP_FRAMES} frames.
 *
 * <p>Position (13 Oct 2026): the grid runs in front of the blob SVM, in
 * {@code SvmBlobDetector}. It learns from every blob in its area range (30-400 px), scored or not,
 * and blobs in static cells are not scored at all. It used to sit in
 * {@link BallTracker}, one grid per stream, learning only from candidates the
 * SVM had accepted. An offline replay against ground truth on S1-S3
 * (data/results/grid_position/REPORT.md) found tracker F1 within 0.004 of the
 * old position on every session, the same rallies found, and 13-22 % of blobs
 * not scored in typical footage (about half on busy S3 footage). Cells larger
 * than 16 px cost real ball frames there, so keep these constants.
 */
public class OccupancyGrid {

    private static final int    FRAME_W         = 854;
    private static final int    FRAME_H         = 480;
    private static final int    CELL            = 16;
    private static final int    ROWS            = FRAME_H / CELL + 1;
    private static final int    COLS            = FRAME_W / CELL + 1;

    private static final float  DECAY           = 0.98f;
    private static final float  THRESH          = 0.65f;
    private static final int    WARMUP_FRAMES   = 60;
    //what a cell settles to;
    private static final float  FULL            = 1f / (1f - DECAY);
    private static final float  STATIC          = THRESH * FULL;

    private final float[][] grid = new float[ROWS][COLS];
    private int frames = 0;

    /** Starts a frame: every cell decays. Call once per frame, before {@link #hit}. */
    public void nextFrame(){
        for (float[] row : grid){
            for (int c = 0; c < COLS; c++){
                row[c] *= DECAY;
            }
        }
        frames ++;
    }

    /** Counts one blob centred at (x, y) in the current frame. */
    public void hit(double x, double y){
        grid[row(y)][col(x)] += 1f;
    }

    /** {@link #nextFrame()}, then one {@link #hit} per detection. */
    public void update(List<BallCandidate> detections){
        nextFrame();
        for (BallCandidate d : detections){
            hit(d.x, d.y);
        }
    }

    /** True when (x, y)'s cell is static. Read before this frame's update. */
    public boolean isStatic(double x, double y){
        if (frames < WARMUP_FRAMES) return false;
        return grid[row(y)][col(x)] >= STATIC;
    }

    private static int row (double y) { return clamp((int) (y / CELL) , ROWS); }
    private static int col (double x) { return clamp((int) (x / CELL) , COLS); }

    private static int clamp(int i, int n){
        return i < 0 ? 0 : (i >= n ? n - 1: i);
    }
}
