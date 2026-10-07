package com.justin.spikeballtracker.tracking;

public class NetZone {

    private static final double PAD_SIDE = 30;
    private static final double PAD_ABOVE = 60;
    private static final double PAD_BELOW = 15;

    public final double left, top, right, bottom;

    public NetZone(double left, double top, double right, double bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    //the zone around a net box given as x,y width and height. in px.
    public static NetZone around(double x, double y, double w, double h){
        return new NetZone(x - PAD_SIDE, y-PAD_ABOVE, x + w + PAD_SIDE, y + h + PAD_BELOW);
    }

    //
    public boolean contains(double px, double py){
        return px >= left && px <= right && py >= top && py <= bottom;
    }
}
