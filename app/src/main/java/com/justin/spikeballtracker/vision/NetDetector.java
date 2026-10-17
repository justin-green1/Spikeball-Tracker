package com.justin.spikeballtracker.vision;

import android.content.res.AssetManager;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;

/**
 * Finds the net in a KNN background plate with a YOLO26-nano detector.
 *
 * <p>Replaces the edge-and-contour detector. Same contract: the only caller
 * is {@link FramePipeline}, which hands over a CV_8UC3 Y,U,V plate at the
 * working resolution (854x480), and gets back the net box in plate (and so
 * display) coordinates, or null when there is no confident net.
 *
 * <p>The model ({@value #MODEL_ASSET} in the assets) was trained on exported
 * background plates from sessions S1-S4 and S6, converted to colour the way
 * this class does it (YUV to RGB), with one class, "net". It was exported
 * from Ultralytics at a fixed 384x640 input without the built-in NMS: OpenCV
 * 4.12's DNN module cannot load the end-to-end head (it has no TopK layer).
 * The raw output is one column per anchor; the net is a single object, so the
 * best-scoring anchor is the answer and no NMS is needed.
 *
 * <p>Preprocessing matches Ultralytics' letterbox exactly: scale the longest
 * side to fit, bilinear resize, pad to 640x384 with grey 114, divide by 255.
 * For an 854x480 plate that is a 640x360 image with 12 rows of padding above
 * and below.
 */
public class NetDetector {
    private static final String TAG = "NetDetector";

    /** File name in app/src/main/assets. */
    public static final String MODEL_ASSET = "yolo_net.onnx";

    /** The network's fixed input size, as exported (imgsz = 384, 640). */
    private static final int IN_W = 640;
    private static final int IN_H = 384;
    private static final double PAD_GREY = 114.0;

    /** Output values per anchor: centre x, centre y, width, height, net confidence. */
    private static final int ROWS = 5;

    /**
     * Below this the plate is treated as "no net", and the pipeline keeps the
     * box it already has, so a missed attempt costs little. On 46 held-out
     * session 5 images, raising the floor from 0.25 to 0.5 dropped the one
     * clearly wrong box (confidence 0.38) and one correct, low-confidence
     * box (40 hits at 0.25, 39 at 0.5).
     */
    private static final float MIN_CONFIDENCE = 0.5f;

    /**
     * The ONNX file, read once by {@link #loadModel} on the UI thread before
     * any source exists. Each pipeline is built later, on the Analysis thread,
     * from a job posted after that, so the post orders the write before every
     * read and no lock is needed.
     */
    private static byte[] modelBytes;

    /** Built on first use, on the pipeline's own thread; one per pipeline. */
    private Net net;
    private boolean netUnavailable;

    // ---- reusable buffers ----
    private final Mat rgb     = new Mat();
    private final Mat resized = new Mat();
    private final Mat padded  = new Mat();

    /**
     * Reads the model from the app's assets. Call once, from
     * {@code MainActivity.onCreate}, after OpenCV has loaded and before any
     * frame source starts.
     */
    public static void loadModel(AssetManager assets) {
        try (InputStream in = assets.open(MODEL_ASSET)) {
            modelBytes = in.readAllBytes();
            Log.i(TAG, "loaded " + MODEL_ASSET + " (" + modelBytes.length + " bytes)");
        } catch (IOException e) {
            Log.e(TAG, "could not read " + MODEL_ASSET + "; net detection is off", e);
        }
    }

    /**
     * @param plate CV_8UC3 Y,U,V background plate from the KNN model, at the
     *              working resolution (854x480)
     * @return the net box in plate coordinates, or null if the model is
     *         missing or its best box is less than {@link #MIN_CONFIDENCE}
     */
    public Rect detect(Mat plate) {
        if (plate == null || plate.empty()) return null;
        Net n = net();
        if (n == null) return null;

        long startNs = System.nanoTime();
        Mat blob = null;
        Mat out = null;
        try {
            int w = plate.cols();
            int h = plate.rows();

            // Letterbox, as Ultralytics does it.
            double r  = Math.min(IN_W / (double) w, IN_H / (double) h);
            int nw    = (int) Math.round(w * r);
            int nh    = (int) Math.round(h * r);
            int padX  = (IN_W - nw) / 2;
            int padY  = (IN_H - nh) / 2;

            Imgproc.cvtColor(plate, rgb, Imgproc.COLOR_YUV2RGB);
            Imgproc.resize(rgb, resized, new Size(nw, nh), 0, 0, Imgproc.INTER_LINEAR);
            Core.copyMakeBorder(resized, padded,
                    padY, IN_H - nh - padY, padX, IN_W - nw - padX,
                    Core.BORDER_CONSTANT, new Scalar(PAD_GREY, PAD_GREY, PAD_GREY));

            // Already RGB, so no channel swap.
            blob = Dnn.blobFromImage(padded, 1.0 / 255.0, new Size(IN_W, IN_H),
                    new Scalar(0, 0, 0), false, false);
            n.setInput(blob);
            out = n.forward();                      // 1 x 5 x anchors, float

            int anchors = (int) (out.total() / ROWS);
            float[] data = new float[ROWS * anchors];
            out.get(new int[] {0, 0, 0}, data);     // row-major: all cx, all cy, ... all conf

            int best = -1;
            float bestConf = -1f;
            int confRow = 4 * anchors;
            for (int i = 0; i < anchors; i++) {
                float c = data[confRow + i];
                if (c > bestConf) {
                    bestConf = c;
                    best = i;
                }
            }

            long ms = (System.nanoTime() - startNs) / 1_000_000L;
            if (best < 0 || bestConf < MIN_CONFIDENCE) {
                Log.i(TAG, String.format(Locale.US,
                        "no net (best confidence %.2f), %d ms", bestConf, ms));
                return null;
            }

            // Back from the 640x384 input to plate coordinates.
            double cx = data[best];
            double cy = data[anchors + best];
            double bw = data[2 * anchors + best];
            double bh = data[3 * anchors + best];
            double x0 = clamp((cx - bw / 2 - padX) / r, 0, w);
            double y0 = clamp((cy - bh / 2 - padY) / r, 0, h);
            double x1 = clamp((cx + bw / 2 - padX) / r, 0, w);
            double y1 = clamp((cy + bh / 2 - padY) / r, 0, h);

            Rect box = new Rect((int) Math.round(x0), (int) Math.round(y0),
                    (int) Math.round(x1 - x0), (int) Math.round(y1 - y0));
            Log.i(TAG, String.format(Locale.US,
                    "net %dx%d at (%d, %d), confidence %.2f, %d ms",
                    box.width, box.height, box.x, box.y, bestConf, ms));
            return box.width > 0 && box.height > 0 ? box : null;

        } finally {
            if (blob != null) blob.release();
            if (out != null) out.release();
        }
    }

    /** Frees the buffers. The network itself is freed by the garbage collector. */
    public void release() {
        rgb.release();
        resized.release();
        padded.release();
        net = null;
    }

    private Net net() {
        if (net != null || netUnavailable) return net;
        if (modelBytes == null) {
            Log.e(TAG, "model not loaded (NetDetector.loadModel was not called)");
            netUnavailable = true;
            return null;
        }
        MatOfByte buffer = new MatOfByte(modelBytes);
        try {
            long startNs = System.nanoTime();
            net = Dnn.readNetFromONNX(buffer);
            net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
            net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
            Log.i(TAG, "network ready in " + (System.nanoTime() - startNs) / 1_000_000L + " ms");
        } catch (Exception e) {
            Log.e(TAG, "could not build the network; net detection is off", e);
            net = null;
            netUnavailable = true;
        } finally {
            buffer.release();
        }
        return net;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
