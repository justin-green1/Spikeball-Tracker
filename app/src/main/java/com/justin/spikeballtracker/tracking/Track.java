package com.justin.spikeballtracker.tracking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class Track {
    private static final double EPS = 1e-15;

    private final long t0Us;
    private long lastUs;
    private final List<Point> points = new ArrayList<>();

    private double n, su, suu, suuu, suuuu;
    private double sx, sux;
    private double sy, suy, suuy;

    private double p, q;
    private double a, b, c;
    private boolean dirty = true;

    protected boolean confirmed = false;

    public Track (int frameIndex, long tUs, double x, double y){
        this.t0Us = tUs;
        add(frameIndex, tUs, x, y);
    }

    private double toU(long tUs){
        return (tUs - t0Us)/1e6;
    }

    public void add(int frameIndex, long tUs, double x, double y){
        double u = toU(tUs);
        double u2 = u*u;

        n       += 1;
        su      += u;
        suu     += u2;
        suuu    += u2 * u;
        suuuu   += u2 * u2;
        sx      += x;
        sux     += u * x;
        sy      += y;
        suy     += u * y;
        suuy    += u2 * y;

        points.add(new Point(frameIndex, tUs, x, y));
        lastUs = tUs;
        dirty = true;
    }

    private void solve(){
        if(!dirty) return;

        // ---  x = p + q * u : 2x2 least squares   ---
        double det2 = n * suu - su * su;
        if(Math.abs(det2) < EPS){
            p = sx / n;
            q = 0;
        }else{
            p = (suu * sx - su * sux) / det2;
            q = (n * sux - su * sx) / det2;
        }

        // --- y ---
        if ( n < 4){ // assume a straight line with no curve.
            if(Math.abs(det2) < EPS){
                a = sy / n;
                b = 0;
            }else{
                a = (suu * sy - su * suy) / det2;
                b = (n * suy - su * sy) / det2;
            }
            c = 0;
        }else{
            double d = det3(n, su, suu,
                            su, suu, suuu,
                            suu, suuu, suuuu);
            if (Math.abs(d) < EPS){
                a = sy / n;
                b = 0;
                c = 0;
            }
            else {
                //Cramer's rule;
                a = det3(sy, su, suu,
                        suy, suu, suuu,
                        suuy, suuu, suuuu) / d;
                b = det3(n, sy, suu,
                        su, suy, suuu,
                        suu, suuy, suuuu) / d;
                c = det3(n, su, sy,
                        su, suu, suy,
                        suu, suuu, suuy) / d;
            }
        }
        dirty = false;
    }

    private static double det3(double a11, double a12, double a13,
                       double a21, double a22, double a23,
                       double a31, double a32, double a33){
            return a11 * (a22 * a33 - a23 * a32)
                -  a12 * (a21 * a33 - a23 * a31)
                +  a13 * (a21 * a32 - a22 * a31);
    }

    public double predictX(long tUs){
        solve();
        double u = toU(tUs);
        return p + q * u;
    }

    public double predictY(long tUs){
        solve();
        double u = toU(tUs);
        return a + b * u + c * u * u;
    }

    public int size() {return points.size();}
    public long firstUs() {return t0Us;}
    public long lastUs() {return lastUs;}
    public List<Point> points() {return Collections.unmodifiableList(points);}
    /** Whether the last shape test passed (tracks of 4+ points only). */
    public boolean isConfirmed() {return confirmed;}

    public static final class Point{
        public final int frameIndex;
        public final long tUs;
        public final double x,y;

        Point(int frameIndex, long tUs, double x, double y){
            this.frameIndex = frameIndex;
            this.tUs = tUs;
            this.x = x;
            this.y = y;
        }
    }

    public static final class Shape{
        public final int n;
        public final double rms, curvature, span, speed;

        Shape(int n, double rms, double curvature, double span, double speed){
            this.n = n;
            this.rms = rms;
            this.curvature = curvature;
            this.span = span;
            this.speed = speed;
        }
    }

    public Shape shape(){
        solve();
        double sumSq = 0, path = 0;
        double minX = Double.MAX_VALUE, maxX = -Double.MAX_VALUE;
        double minY = Double.MAX_VALUE, maxY = -Double.MAX_VALUE;

        Point prev = null;
        for(Point pt : points) {
            double u = toU(pt.tUs);
            double rx = pt.x - (p + q * u);
            double ry = pt.y - (a + b * u + c * u * u);
            sumSq += rx * rx + ry * ry;

            minX = Math.min(minX, pt.x);
            maxX = Math.max(maxX, pt.x);
            minY = Math.min(minY, pt.y);
            maxY = Math.max(maxY, pt.y);

            if (prev != null) {
                double dx = pt.x - prev.x, dy = pt.y - prev.y;
                path += Math.sqrt(dx * dx + dy * dy);
            }
            prev = pt;
        }
        int count = points.size();
        double rms = Math.sqrt(sumSq / count);
        double dxS = maxX - minX, dyS = maxY - minY;
        double span = Math.sqrt(dxS * dxS + dyS * dyS);
        double elapsed = (lastUs - t0Us) / 1e6;
        double speed = elapsed > 0 ? path / elapsed : 0;

        return new Shape(count, rms, c, span, speed);
    }

    /**
     * the y fitted curvature c(px/s^2); zero for fewer than 4 points
     */
    public double curvature(){
        solve();
        return c;
    }

    public double[] fitWithCurvature(double cFixed){
        double ry = sy - cFixed * suu;
        double ruy = suy - cFixed * suuu;
        double det2 = n * suu - su * su;
        if (Math.abs(det2) < EPS) return new double[] {ry / n , 0};
        return new double[] {(suu * ry - su * ruy) / det2 , (n * ruy - su * ry) / det2};
    }
}
