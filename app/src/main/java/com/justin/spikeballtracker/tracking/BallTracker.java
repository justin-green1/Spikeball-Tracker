package com.justin.spikeballtracker.tracking;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Follows the ball through one stream of candidates (blobs the SVM scored at
 * or above {@code BlobSvm.THRESHOLD}), building tracks in stages.
 *
 * <ol>
 *   <li><b>Seed</b> (1 point): any candidate no track used. On a later frame,
 *       every candidate within {@link #SEED_REACH_PX_PER_FRAME} px per elapsed
 *       frame becomes its own 2-point stub, and the seed retires. No second
 *       point within 3 frames: the seed is dropped.</li>
 *   <li><b>Stub, 2 points</b>: the third point is the candidate whose implied
 *       change in velocity is smallest, if under {@link #SEED_DV_PX_S}. None
 *       within 3 frames: dropped.</li>
 *   <li><b>Stub, 3 points</b>: the fourth point is the nearest candidate to an
 *       exact parabola through the three (y; x a straight line), within the
 *       gate. None within 3 frames: dropped.</li>
 *   <li><b>Track, 4+ points</b>: extended from its own fit within the gate. It
 *       survives {@link #DROP_AFTER_US} (15 frames) without a point, so it can
 *       bridge an occlusion. It is reported as the ball while it passes the
 *       shape test (fit error, curvature, span, speed), re-run every frame.</li>
 * </ol>
 *
 * <p>Candidates: tracks of 4+ points claim them exclusively, longest first;
 * seeds and stubs share whatever is left. A candidate nobody used seeds a new
 * track. The confirmed track with the most points is the ball, and its
 * answer for each frame is emitted {@link #DELAY} frames late, so a track
 * confirmed or re-found within that window back-fills the frames before it.
 *
 * <p>Replay on S1C03, S2C01, S2C02, S2C04 and S2C05 (16 Oct 2026, old-SVM
 * blob scores, 30 px blob floor), against the previous design (young gate
 * 1200 px/s, 6 points to confirm, curvature from 315 px/s^2): labelled rallies
 * 55/82 with 3 false (was 52, 5 false); per-frame tracking F1 0.506 (0.488);
 * simulated 4-15 frame occlusions bridged 85-59 % (87-61 %).
 */
public class BallTracker {

    // seeds and stubs --
    /**
     * A seed's second point may be this far from it per frame elapsed: up to
     * 150 px if it comes 3 frames later, so a fast ball (20-40 px per frame)
     * that skips a frame can still be picked up.
     */
    private static final double SEED_REACH_PX_PER_FRAME = 50.0;
    /** Largest change in velocity a third point may imply. */
    private static final double SEED_DV_PX_S            = 750.0;

    // gates --
    private static final double GATE_PX                 = 14.0;
    /**
     * The gate widens by this much for every frame a track (3+ points) has
     * gone without a point, beyond the usual one: 4 px per missed frame at
     * 30 fps, so 14 px after one frame, 50 px after ten, about 72 px at the
     * 15-frame limit. Bridges occlusions whose re-sighting has drifted from
     * the extrapolation.
     */
    private static final double GATE_GROW_PX_PER_S      = 120.0;
    private static final long   FRAME_US                = 33_333L;

    // dropping --
    /**
     * A track of 4+ points survives 15 frames without a point (dropped after
     * 15.5), so a ball hidden for up to half a second can be picked up again by
     * the same track. DELAY (15 frames) limits how much of a gap can still be
     * filled in the output once the ball reappears. It applies from 4 points,
     * not only while confirmed: most flights pass the span test only after
     * 6-10 points, and an unconfirmed flight that dies at an occlusion cannot
     * be bridged (simulated 4-15 frame occlusions bridged ~40 % that way,
     * ~85-59 % this way).
     */
    private static final long   DROP_AFTER_US           = 516_667L;
    /** Seeds and stubs (under 4 points) are dropped after 3 frames without a point. */
    private static final long   STUB_DROP_AFTER_US      = 116_667L;

    // confirmation --
    private static final int    MIN_POINTS              = 4;
    /**
     * The curvature floor is 0: any downward-curving or straight path passes
     * on curvature (a fast shot can look straight over a few frames). The
     * ceiling still rejects paths bending harder than gravity allows.
     */
    private static final double MIN_CURV                = 0.0;
    private static final double MAX_CURV                = 1350.0;
    private static final double MIN_SPAN                = 60.0;
    private static final double MAX_RMS                 = 6.0;
    private static final double MIN_SPEED               = 90.0;

    private static final int    MAX_TRACKS              = 400;

    public static final int     DELAY                   = 15;
    private static final int    CAP                     = DELAY + 1;

    private final int[]         slotFrame               = new int[CAP];
    private final long[]        slotUs                  = new long[CAP];
    private final boolean[]     slotHas                 = new boolean[CAP];
    private final double[]      slotX                   = new double[CAP];
    private final double[]      slotY                   = new double[CAP];
    private final int[]         slotN                   = new int[CAP];

    private final List<Track> tracks = new ArrayList<>();
    private final List<Track> born = new ArrayList<>();

    private Output latest = null;

    public BallTracker(){
        java.util.Arrays.fill(slotFrame, -1);
    }

    /**
     * One frame's candidates: blobs the SVM scored at or above
     * {@code BlobSvm.THRESHOLD}, static clutter already removed.
     */
    public void process(int frameIndex, long tUs, List<BallCandidate> candidates){
        int s = frameIndex % CAP;
        slotFrame[s] = frameIndex;
        slotUs[s] = tUs;
        slotHas[s] = false;

        int n = candidates.size();
        boolean[] claimed = new boolean[n];   // by a 4+ point track: off limits to everyone else
        boolean[] used = new boolean[n];      // by any track: does not seed a new one

        // Longest track first: it gets first pick of the candidates. The sort
        // is stable, so equal lengths keep their order from the last frame.
        tracks.sort((a,b) -> Integer.compare(b.size(), a.size()));

        born.clear();
        Iterator<Track> it = tracks.iterator();
        while (it.hasNext()){
            Track t = it.next();
            long dtUs = tUs - t.lastUs();
            if (dtUs <= 0) continue;

            int size = t.size();
            if (size >= 4){
                int c = nearest(candidates, claimed, t.predictX(tUs), t.predictY(tUs), gate(dtUs));
                if (c >= 0){
                    add(t, candidates.get(c), frameIndex, tUs);
                    claimed[c] = true;
                    used[c] = true;
                }
            }else if (size == 3){
                double[] p = parabolaThrough3(t, tUs);
                int c = nearest(candidates, claimed, p[0], p[1], gate(dtUs));
                if (c >= 0){
                    add(t, candidates.get(c), frameIndex, tUs);
                    used[c] = true;
                }
            }else if (size == 2){
                int c = smallestVelocityChange(t, candidates, claimed, tUs);
                if (c >= 0){
                    add(t, candidates.get(c), frameIndex, tUs);
                    used[c] = true;
                }
            }else if (branch(t, candidates, claimed, used, dtUs, frameIndex, tUs)){
                it.remove();   // the seed lives on in its stubs
            }
        }
        tracks.addAll(born);

        for(int i = 0; i < n; i ++){
            if(used[i] || tracks.size() >= MAX_TRACKS) continue;
            BallCandidate c = candidates.get(i);
            tracks.add(new Track(frameIndex, tUs, c.x, c.y));
        }

        it = tracks.iterator();
        while (it.hasNext()){
            Track t = it.next();
            long limit = t.size() >= 4 ? DROP_AFTER_US : STUB_DROP_AFTER_US;
            if(tUs - t.lastUs() > limit) {it.remove(); continue; }
            if(t.size() >= MIN_POINTS){
                Track.Shape shape = t.shape();
                t.confirmed = shape.rms <= MAX_RMS
                        && shape.curvature >= MIN_CURV && shape.curvature <= MAX_CURV
                        && shape.span >= MIN_SPAN && shape.speed >= MIN_SPEED;
            }
        }

        Track best = null;
        for(Track t : tracks){
            if(!t.confirmed) continue;
            if(best == null || t.size() > best.size()){
                best = t;
            }
        }
        if (best != null)claim(best, frameIndex);
        emit(frameIndex - DELAY);
    }

    /** 14 px, plus 4 px for every frame beyond the usual one since the track's last point. */
    private static double gate(long dtUs){
        double missedUs = Math.max(0L, dtUs - FRAME_US);
        return GATE_PX + GATE_GROW_PX_PER_S * missedUs / 1e6;
    }

    /** @return the index of the nearest unclaimed candidate within the gate, or -1 */
    private static int nearest(List<BallCandidate> candidates, boolean[] claimed,
                               double px, double py, double gate){
        int best = -1;
        double bestD = gate;
        for (int i = 0; i < candidates.size(); i++){
            if(claimed[i]) continue;
            BallCandidate c = candidates.get(i);
            double dx = c.x - px, dy = c.y - py;
            double d = Math.sqrt(dx * dx + dy * dy);
            if (d < bestD) {best = i; bestD = d;}
        }
        return best;
    }

    /**
     * Turns a seed into one 2-point stub for every unclaimed candidate within
     * reach (SEED_REACH_PX_PER_FRAME per elapsed frame).
     *
     * @return true if at least one stub was made, so the seed can retire
     */
    private boolean branch(Track seed, List<BallCandidate> candidates, boolean[] claimed,
                           boolean[] used, long dtUs, int frameIndex, long tUs){
        Track.Point p0 = seed.points().get(0);
        double reach = SEED_REACH_PX_PER_FRAME * dtUs / (double) FRAME_US;
        boolean branched = false;
        for (int i = 0; i < candidates.size(); i++){
            if(claimed[i]) continue;
            if(tracks.size() + born.size() >= MAX_TRACKS) break;
            BallCandidate c = candidates.get(i);
            double dx = c.x - p0.x, dy = c.y - p0.y;
            if (Math.sqrt(dx * dx + dy * dy) > reach) continue;

            Track stub = new Track(p0.frameIndex, p0.tUs, p0.x, p0.y);
            stub.add(frameIndex, tUs, c.x, c.y);
            born.add(stub);
            used[i] = true;
            branched = true;
        }
        return branched;
    }

    /** @return the unclaimed candidate implying the smallest velocity change under SEED_DV_PX_S, or -1 */
    private static int smallestVelocityChange(Track t, List<BallCandidate> candidates,
                                              boolean[] claimed, long tUs){
        Track.Point p1 = t.points().get(0), p2 = t.points().get(1);
        double dt1 = (p2.tUs - p1.tUs) / 1e6,   dt2 = (tUs - p2.tUs) / 1e6;
        double vx1 = (p2.x - p1.x) / dt1,       vy1 = (p2.y - p1.y) / dt1;
        int best = -1;
        double bestDv = SEED_DV_PX_S;
        for (int i = 0; i < candidates.size(); i++){
            if(claimed[i]) continue;
            BallCandidate c = candidates.get(i);
            double dvx = vx1 - (c.x - p2.x) / dt2,  dvy = vy1 - (c.y - p2.y) / dt2;
            double dv = Math.sqrt(dvx * dvx + dvy * dvy);
            if (dv <= bestDv) {best = i; bestDv = dv;}
        }
        return best;
    }

    /**
     * Where a 3-point stub expects its fourth point: y on the exact parabola
     * through its three points (Lagrange form), x on its straight-line fit.
     */
    private static double[] parabolaThrough3(Track t, long tUs){
        List<Track.Point> p = t.points();
        double t0 = p.get(0).tUs / 1e6, t1 = p.get(1).tUs / 1e6, t2 = p.get(2).tUs / 1e6;
        double u = tUs / 1e6;
        double y = p.get(0).y * (u - t1) * (u - t2) / ((t0 - t1) * (t0 - t2))
                 + p.get(1).y * (u - t0) * (u - t2) / ((t1 - t0) * (t1 - t2))
                 + p.get(2).y * (u - t0) * (u - t1) / ((t2 - t0) * (t2 - t1));
        return new double[]{t.predictX(tUs), y};
    }

    private static void add(Track t, BallCandidate c, int frameIndex, long tUs){
        t.add(frameIndex, tUs, c.x, c.y);
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

            if (!slotHas[s] || t.size() >= slotN[s]){
                slotHas[s] = true;
                slotX[s] = x;
                slotY[s] = y;
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

    /** The seeds, stubs and tracks alive after the last frame; read-only, valid until the next process(). */
    public List<Track> tracks() {
        return java.util.Collections.unmodifiableList(tracks);
    }

    public void reset(){
        tracks.clear();
        java.util.Arrays.fill(slotFrame, -1);
        latest = null;
    }
}
