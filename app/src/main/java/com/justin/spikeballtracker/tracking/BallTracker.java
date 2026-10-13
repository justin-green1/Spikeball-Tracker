package com.justin.spikeballtracker.tracking;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class BallTracker {

    // gates --
    private static final double NARROW_GATE_PX      = 14.0;
    private static final double WIDE_GATE_PX        = 8.0;
    private static final double YOUNG_SPEED_PX_S    = 1200.0;
    private static final double SEED_DV_PX_S       = 750.0;
    private static final long   DROP_AFTER_US       = 116_667L;

    // confirmation --
    private static final int    MIN_POINTS          = 10;
    private static final int    MIN_POINTS_NARROW   = 6;
    private static final double MIN_CURV            = 315.0;
    private static final double MAX_CURV            = 1350.0;
    private static final double MIN_SPAN            = 60.0;
    private static final double MAX_RMS             = 6.0;
    private static final double MIN_SPEED           = 90.0;

    private static final int    MAX_TRACKS          = 400;

    public static final int    DELAY               = 15;
    private static final int    CAP                 = DELAY + 1;

    private final int[]         slotFrame           = new int[CAP];
    private final long[]        slotUs              = new long[CAP];
    private final boolean[]     slotHas             = new boolean[CAP];
    private final double[]      slotX               = new double[CAP];
    private final double[]      slotY               = new double[CAP];
    private final int[]         slotNh              = new int[CAP];
    private final int[]         slotN               = new int[CAP];

    private OccupancyGrid narrowGrid = new OccupancyGrid();
    private OccupancyGrid wideGrid = new OccupancyGrid();
    private final List<Track> tracks = new ArrayList<>();

    private final List<BallCandidate> liveN = new ArrayList<>();
    private final List<BallCandidate> liveW = new ArrayList<>();

    private Output latest = null;

    public BallTracker(){
        java.util.Arrays.fill(slotFrame, -1);
    }

    public void process(int frameIndex, long tUs, List <BallCandidate> narrow, List<BallCandidate> wide){
        int s = frameIndex % CAP;
        slotFrame[s] = frameIndex;
        slotUs[s] = tUs;
        slotHas[s] = false;

        liveN.clear();
        for (BallCandidate c : narrow){
            if (!narrowGrid.isStatic(c.x, c.y)) liveN.add(c);
        }
        narrowGrid.update(narrow);
        liveW.clear();
        for (BallCandidate c : wide){
            if (!wideGrid.isStatic(c.x, c.y)) liveW.add(c);
        }
        wideGrid.update(wide);

        boolean[] takenN = new boolean[liveN.size()];
        boolean[] takenW = new boolean[liveW.size()];

        tracks.sort((a,b) -> a.narrowHits != b.narrowHits
                ? Integer.compare(b.narrowHits, a.narrowHits)
                : Integer.compare(b.size(), a.size()));

        for (Track t : tracks){
            long dtUs = tUs - t.lastUs();
            if (dtUs <= 0) continue;
            double px = t.predictX(tUs), py = t.predictY(tUs);
            boolean young = t.size() < 3;
            double youngGate = YOUNG_SPEED_PX_S * dtUs / 1e6;

            if (tryExtend(t, liveN, takenN, young ? youngGate : NARROW_GATE_PX, px, py, frameIndex, tUs)){
                t.narrowHits++;
            }else{
                tryExtend(t, liveW, takenW, young ? youngGate : WIDE_GATE_PX, px, py, frameIndex, tUs);
            }
        }

        for(int i = 0; i < liveN.size(); i ++){
            if(takenN[i] || tracks.size() >= MAX_TRACKS) continue;
            BallCandidate c = liveN.get(i);
            Track nt = new Track(frameIndex, tUs, c.x, c.y);
            nt.narrowHits = 1;
            tracks.add(nt);
        }
        for(int i = 0; i < liveW.size(); i ++){
            if(takenW[i] || tracks.size() >= MAX_TRACKS) continue;
            BallCandidate c = liveW.get(i);
            tracks.add(new Track(frameIndex, tUs, c.x, c.y));
        }

        Iterator<Track> it = tracks.iterator();
        while (it.hasNext()){
            Track t = it.next();
            if(tUs - t.lastUs() > DROP_AFTER_US) {it.remove(); continue; }
            int need = t.narrowHits >= 1 ? MIN_POINTS_NARROW : MIN_POINTS;

            if(t.size() >= need){
                Track.Shape shape = t.shape();
                t.confirmed = shape.rms <= MAX_RMS
                        && shape.curvature >= MIN_CURV && shape.curvature <= MAX_CURV
                        && shape.span >= MIN_SPAN && shape.speed >= MIN_SPEED;
            }
        }

        Track best = null;
        for(Track t : tracks){
            if(!t.confirmed) continue;
            if(best == null || t.narrowHits > best.narrowHits
                    || (t.narrowHits == best.narrowHits && t.size() > best.size())){
                best = t;
            }
        }
        if (best != null)claim(best, frameIndex);
        emit(frameIndex - DELAY);
    }

    private static boolean tryExtend (Track t, List<BallCandidate> live, boolean[] taken,
                                      double gate, double px, double py, int frameIndex, long tUs){
        int best = -1;
        double bestD = gate;
        for (int i = 0; i < live.size(); i++){
            if(taken[i]) continue;
            BallCandidate c = live.get(i);
            double dx = c.x - px, dy = c.y - py;
            double d = Math.sqrt(dx * dx + dy * dy);
            if (d < bestD) {best = i; bestD = d;}
        }
        if (best < 0) {return false;}

        BallCandidate c = live.get(best);
        if(t.size() == 2 && !plausibleThird(t, c, tUs)) return false;

        taken[best] = true;
        t.add(frameIndex, tUs, c.x, c.y);
        return true;
    }

    private static boolean plausibleThird(Track t, BallCandidate c, long tUs){
        Track.Point p1 = t.points().get(0), p2 = t.points().get(1);
        double dt1 = (p2.tUs - p1.tUs) / 1e6,   dt2 = (tUs - p2.tUs) / 1e6;
        double vx1 = (p2.x - p1.x) / dt1,       vx2 = (c.x - p2.x) / dt2;
        double vy1 = (p2.y - p1.y) / dt1,       vy2 = (c.y - p2.y) / dt2;
        double dvx = vx1 - vx2,                 dvy = vy1 - vy2;
        return Math.sqrt(dvx * dvx + dvy * dvy) <= SEED_DV_PX_S;
    }

    private void claim(Track t, int frameIndex){
        List<Track.Point> pts = t.points();
        int first = Math.max(pts.get(0).frameIndex, frameIndex - DELAY);
        int last = pts.get(pts.size() - 1).frameIndex;
        int k = 0;

        for (int f = first; f <= last; f++){
            int s = f % CAP;
            if(slotFrame[s] != f) continue;

            while (k < pts.size() && pts.get(k).frameIndex < f) k++;
            double x, y;
            if (k < pts.size() && pts.get(k).frameIndex == f){
                x = pts.get(k).x;
                y = pts.get(k).y;
            }else{
                x = t.predictX(slotUs[s]);
                y = t.predictY(slotUs[s]);
            }

            boolean outranks = !slotHas[s]
                    || t.narrowHits > slotNh[s]
                    || (t.narrowHits == slotNh[s] && t.size() >= slotN[s]);

            if (outranks){
                slotHas[s] = true;
                slotX[s] = x;
                slotY[s] = y;
                slotNh[s] = t.narrowHits;
                slotN[s] = t.size();
            }
        }
    }

    public static final class Output{
        public final int frameIndex;
        public final long tUs;
        public final boolean hasBall;
        public final double  x, y;

        public Output(int frameIndex,long tUs, boolean hasBall, double x, double y) {
            this.frameIndex = frameIndex;
            this.tUs = tUs;
            this.hasBall = hasBall;
            this.x = x;
            this.y = y;
        }
    }

    private void emit(int f){
        if (f < 0){latest = null; return;}
        int s = f % CAP;
        boolean has = slotFrame[s] == f && slotHas[s];
        latest = new Output(f, slotUs[s], has, slotX[s], slotY[s]);
    }

    public Output latest() {
        return latest;
    }

    public void reset(){
        tracks.clear();
        narrowGrid = new OccupancyGrid();
        wideGrid = new OccupancyGrid();
        java.util.Arrays.fill(slotFrame, -1);
        latest = null;
    }
}
