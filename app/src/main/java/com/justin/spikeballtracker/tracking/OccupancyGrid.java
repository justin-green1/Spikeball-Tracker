package com.justin.spikeballtracker.tracking;

import java.util.List;

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

    public void update(List<BallCandidate> detections){
        for (float[] row : grid){
            for (int c = 0; c < COLS; c++){
                row[c] *= DECAY;
            }
        }
        frames ++;

        for (BallCandidate d : detections){
            grid[row(d.y)][col(d.x)] += 1f;
        }
    }

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
