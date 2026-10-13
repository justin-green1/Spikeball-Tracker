package com.justin.spikeballtracker.tracking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RallyDetector {
    //---- R1 constants: Extrapolating and determining net intersections.//
    private static final long   GAP_US          = 150_000L;
    private static final double JUMP_PX_PER_S   = 1200.0;
    //
    private static final int    MIN_FIT         = 4;
    private static final double MIN_CURV        = 315.0;
    private static final double MAX_CURV        = 1350.0;
    private static final double FIXED_CURV      = 495.0;
    private static final int    EXTRAP_STEPS    = 15;
    private static final long   STEP_US         = 33_333L;
    private static final double FRAME_W         = 854.0;
    private static final double BELOW_STOP_PX   = 40.0;

    private static final long   NO_CONTACT      = Long.MIN_VALUE;

    //---- R2 constants: Score and rallies.
    private static final long   HALF_FRAME_US   = 16_667L;
    private static final long   COVER_REACH_US  = EXTRAP_STEPS * STEP_US + HALF_FRAME_US;
    private static final long   WINDOW_BACK_US  = 1_516_667L;
    private static final long   WINDOW_AHEAD_US = 1_483_333L;
    private static final double SCALE           = 0.8888888888888888;
    private static final double HI              = 0.4;
    private static final double LO              = 0.4 * 0.75;
    private static final long   MERGE_GAP_US    = 2_050_000L;
    private static final long   MIN_LEN_US      = 450_000L;
    private static final long   SNAP_PRE_US     = 266_667L;
    private static final long   SNAP_POST_US    = 500_000L;

    //R1
    private Track flight = null;
    private NetZone zone = null;

    //R2
    private final List<Frame> frames = new ArrayList<>();
    private int nextToScore = 0;
    private long coverUntilUs = Long.MIN_VALUE;

    private boolean on = false;
    private int openStart, lastOn;
    private int pendingStart = -1, pendingEnd;
    private final List<Rally> rallies = new ArrayList<>();

    private static final class NetLink{
        final long startUs, endUs;
        final boolean toNet,fromNet;
        final long contactToUs, contactFromUs;

        private NetLink(long startUs, long endUs, boolean toNet, long contactToUs,boolean fromNet, long contactFromUs) {
            this.startUs = startUs;
            this.endUs = endUs;
            this.toNet = toNet;
            this.fromNet = fromNet;
            this.contactToUs = contactToUs;
            this.contactFromUs = contactFromUs;
        }
    }

    private static final class Frame{
        final long tUs;
        final boolean ball;
        boolean cover;

        private Frame(long tUs, boolean ball) {
            this.tUs = tUs;
            this.ball = ball;
        }
    }

    public static class Rally{
        public final long startUs, endUs;

        public Rally(long startUs, long endUs) {
            this.startUs = startUs;
            this.endUs = endUs;
        }
    }

    public void process(BallTracker.Output o, NetZone currentZone){
        zone = currentZone;
        if(o == null) return;

        Frame fr = new Frame(o.tUs, o.hasBall);
        fr.cover = o.tUs <= coverUntilUs;
        frames.add(fr);

        updateFlights(o);

        scoreReady(finalUs(), false);
    }

    private void updateFlights(BallTracker.Output o){
        if(o.hasBall){
            if (flight != null){
                if(continues(o)){
                    flight.add(o.frameIndex, o.tUs, o.x, o.y);
                    return;
                }
                closeFlight();
            }
            flight = new Track(o.frameIndex, o.tUs, o.x, o.y);
        }else if(flight != null && o.tUs - flight.lastUs() > GAP_US){
            closeFlight();
        }
    }

    private boolean continues(BallTracker.Output o){
        Track.Point last = flight.points().getLast();
        long dtUs = o.tUs - last.tUs;
        double dx = o.x - last.x, dy = o.y - last.y;
        double dist = Math.sqrt(dx * dx + dy * dy);
        return dtUs <= GAP_US && dist <= JUMP_PX_PER_S * dtUs / 1e6;
    }

    private static NetLink link(Track f, NetZone z){
        List<Track.Point> pts = f.points();
        Track.Point first = pts.getFirst(), last = pts.getLast();

        boolean toNet = false, fromNet = false;
        long contactTo = NO_CONTACT, contactFrom = NO_CONTACT;

        if (z != null){
            if(z.contains(last.x, last.y)) {toNet = true; contactTo = last.tUs;}
            if(z.contains(first.x, first.y)) {fromNet = true; contactFrom = first.tUs;}

            if (pts.size() >= 3){
                double c = f.curvature();
                if (pts.size() < MIN_FIT || c < MIN_CURV || c > MAX_CURV) c = FIXED_CURV;
                double[] ab = f.fitWithCurvature(c);

                if(!toNet){
                    contactTo = extrapolate(f, ab[0], ab[1], c, z, last.tUs, first.tUs, +1);
                    toNet = contactTo != NO_CONTACT;
                }

                if(!fromNet){
                    contactFrom = extrapolate(f, ab[0], ab[1], c, z, first.tUs, first.tUs, -1);
                    fromNet = contactFrom != NO_CONTACT;
                }
            }
        }
        return new NetLink(first.tUs, last.tUs, toNet, contactTo, fromNet, contactFrom);
    }

    /**
     * Follows the flights path for up to 0.5 seconds away from each end.
     * looks for the path intersecting with the NetZone.
     *
     * @return the contact time or NO_CONTACT.
     */
    private static long extrapolate(Track f, double a, double b, double c, NetZone z,
                                    long fromUs, long firstUs, int dir){
        for (int k = 1; k <= EXTRAP_STEPS; k++){
            long t = fromUs + dir * k * STEP_US;
            double u = (t - firstUs) / 1e6;
            double x = f.predictX(t);
            double y = a + b * u + c * u * u;

            if(x < 0 || x > FRAME_W || y > z.bottom + BELOW_STOP_PX){
                return NO_CONTACT;
            }
            double vy = b + 2 * c * u;
            if (z.contains(x, y) && (dir > 0 ? vy > 0 : vy < 0)){
                return t;
            }
        }
        return NO_CONTACT;
    }

    private void closeFlight(){
        NetLink link = link(flight, zone);
        flight = null;

        if(link.toNet || link.fromNet){
            long s0 = link.fromNet ? link.contactFromUs : link.startUs;
            long s1 = link.toNet ? link.contactToUs : link.endUs;
            markCover(s0, s1);
        }
    }

    //frames from s0 to s1 are on a net-linked flight. including upcoming frames.
    private void markCover(long s0, long s1){
        for(int i = frames.size() - 1; i >= 0; i--){
            Frame f = frames.get(i);
            if(f.tUs < s0 - HALF_FRAME_US) break;
            if(f.tUs <= s1 + HALF_FRAME_US) f.cover = true;
        }
        coverUntilUs = Math.max(coverUntilUs, s1 + HALF_FRAME_US);
    }

    //Frames before this cannot gain cover
    private long finalUs(){
        long start = (flight != null) ? flight.firstUs() : frames.getLast().tUs;
        return start - COVER_REACH_US;
    }

    //score every frame whose whole window is final
    private void scoreReady(long finalUs, boolean all){
        while (nextToScore < frames.size()){
            Frame f = frames.get(nextToScore);
            if (!all && f.tUs + WINDOW_AHEAD_US > finalUs) return;
            onScore(nextToScore, score(nextToScore));
            nextToScore ++;
        }
    }

    //half the ball share + half the net-cover share over the 90 frame window
    private double score(int i){
        long t = frames.get(i).tUs;
        int j = i;

        while(j >= 0 && frames.get(j).tUs >= t - WINDOW_BACK_US) j--;
        int n = 0, balls = 0, cover = 0;

        for(j = j + 1; j < frames.size() && frames.get(j).tUs < t + WINDOW_AHEAD_US; j++){
            n++;
            if (frames.get(j).ball) balls ++;
            if (frames.get(j).cover) cover ++;
        }
        return 0.5 * ((double) balls / n + (double) cover / n);
    }

    private void onScore(int i, double s){
        Frame f = frames.get(i);

        if(pendingStart >= 0 && f.tUs - frames.get(pendingEnd).tUs > MERGE_GAP_US){
            finalisePending();
        }

        if(!on && s / SCALE >= HI){
            on = true;
            if (pendingStart >= 0){
                openStart = pendingStart;
                pendingStart = -1;
            }else{
                openStart = i;
            }
        }else if(on && s / SCALE < LO){
            on = false;
            pendingStart = openStart;
            pendingEnd = lastOn;
        }
        if(on) lastOn = i;
    }

    private void finalisePending(){
        int s = pendingStart, e = pendingEnd;
        pendingStart = -1;
        if (frames.get(e).tUs - frames.get(s).tUs < MIN_LEN_US) return;

        int firstBall = -1, lastBall = -1;
        for (int k = s; k <= e; k++){
            if (frames.get(k).ball){
                if(firstBall < 0) firstBall = k;
                lastBall = k;
            }
        }
        if(firstBall >= 0){
            rallies.add(new Rally(frames.get(firstBall).tUs - SNAP_PRE_US,
                                    frames.get(lastBall).tUs + SNAP_POST_US));
        }else{
            rallies.add(new Rally(frames.get(s).tUs, frames.get(e).tUs));
        }
    }

    public void finish(){
        if(flight != null) closeFlight();
        scoreReady(Long.MAX_VALUE, true);
        if(on){
            on = false;
            pendingStart = openStart;
            pendingEnd = lastOn;
        }
        if(pendingStart >= 0) finalisePending();
    }

    public List<Rally> rallies(){ return Collections.unmodifiableList(rallies);}
}
