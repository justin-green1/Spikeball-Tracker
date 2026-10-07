package com.justin.spikeballtracker.video;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Exports the exact YUV background data NetDetector consumes, to permanent device storage,
 * so parameter tuning can run offline against the real thing rather than a proxy.
 *
 * <p>Artefacts exported:
 * <ul>
 *   <li><b>The plate</b> — the KNN background model from {@code subtractor.getBackgroundImage},
 *       exported every 10 seconds of presentation time and on the Export BG button.
 *       The first interval plate lands at t = 10s, which is where the initial net
 *       lock happens, so it is the same plate NetDetector locks on. Source frames
 *       are deliberately not written: only the plate is what NetDetector sees.</li>
 * </ul>
 *
 * <p>Each plate is written twice, losslessly:
 * <ul>
 *   <li>{@code .bin} — raw {@code height * width * 3} bytes, channel interleaved (Y, U, V).</li>
 *   <li>{@code .png} — OpenCV BGR-mapped lossless PNG (where B=Y, G=U, R=V).</li>
 * </ul>
 *
 * <p>Files land in {@code Documents/SpikeballTracker/<runId>/}.
 *
 * <p>Not thread safe. Call it only from the processing thread.
 */
public final class BackgroundExporter {

    private static final String TAG = "BackgroundExporter";

    /** Schema version of manifest.json. Bumped to 2 for background interval export. */
    private static final int MANIFEST_SCHEMA = 2;

    private static final String ROOT_FOLDER = "SpikeballTracker";

    /** 10 seconds expressed in presentation timestamp microseconds. */
    public static final long BACKGROUND_INTERVAL_US = 10_000_000L;

    /** Level 1: quick to encode, still lossless. */
    private static final MatOfInt PNG_PARAMS =
            new MatOfInt(Imgcodecs.IMWRITE_PNG_COMPRESSION, 1);

    public static final class Provenance {
        public String clipName = "unknown";
        public String colourRange = "undeclared";
        public String colourStandard = "unknown";
        public boolean expandToFullRange = true;
        public int targetWidth = 854;
        public int targetHeight = 480;
        public int knnHistory = -1;
        public double knnDist2Threshold = -1;
        public boolean knnDetectShadows = true;
        public int knnPreBlurKernel = 9;
    }

    private final Context appContext;

    private volatile boolean runActive;
    private String runId;
    private Provenance provenance;
    private int framesSeen;
    private int platesSaved;
    private long lastBackgroundTimestampUs;
    private JSONArray artefacts;
    private Uri manifestUri;
    private File legacyManifestFile;

    public BackgroundExporter(Context context) {
        this.appContext = context.getApplicationContext();
    }

    // --- run lifecycle -------------------------------------------------------

    public void beginRun(Provenance p) {
        if (runActive) {
            endRun();
        }
        this.provenance = p;
        this.runId = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date())
                + "_" + sanitise(p.clipName);
        this.framesSeen = 0;
        this.platesSaved = 0;
        this.lastBackgroundTimestampUs = -1L;
        this.artefacts = new JSONArray();
        this.manifestUri = null;
        this.legacyManifestFile = null;
        this.runActive = true;
        Log.i(TAG, "run started: " + runId);
    }

    public boolean isRunActive() {
        return runActive;
    }

    /**
     * The current run's folder name under Documents/SpikeballTracker/, so other
     * per-run outputs (the detection log) can land beside the plates. Null
     * before the first run.
     */
    public String runId() {
        return runId;
    }

    public void endRun() {
        if (!runActive) {
            return;
        }
        writeManifest();
        runActive = false;
        Log.i(TAG, "run ended: " + runId
                + " (" + platesSaved + " plates across " + framesSeen + " frames seen)");
    }

    // --- capture -------------------------------------------------------------

    /**
     * Retained for frame tracking. No longer writes source frames to storage.
     */
    public void offerFrame(Mat yuv, int frameIndex, long timestampUs) {
        if (!runActive) {
            return;
        }
        framesSeen++;
    }

    /**
     * Whether a plate is due at this timestamp. Callers use it to decide whether
     * to pay for a {@code getBackgroundImage} at all, so the interval lives in
     * one place rather than being duplicated at the call site.
     *
     * <p>The first plate is due at t = 10s rather than at t = 0, both because the
     * KNN model has nothing useful in it before then and so that the cadence
     * lands on round ten-second boundaries.
     */
    public boolean dueForBackground(long timestampUs) {
        if (!runActive) {
            return false;
        }
        if (lastBackgroundTimestampUs < 0) {
            return timestampUs >= BACKGROUND_INTERVAL_US;
        }
        return (timestampUs - lastBackgroundTimestampUs) >= BACKGROUND_INTERVAL_US;
    }

    /**
     * Offers the current background plate from {@code subtractor.getBackgroundImage}.
     * Saves it at most once every 10 seconds of video presentation time.
     *
     * <p>The interval only advances on a successful write, so an empty or
     * unwritable plate is retried on the next frame instead of silently costing
     * ten seconds of coverage.
     */
    public void offerBackground(Mat background, int frameIndex, long timestampUs) {
        if (!dueForBackground(timestampUs)) {
            return;
        }
        if (background == null || background.empty()) {
            return;   // model not ready yet; try again next frame
        }
        if (exportPlate(background, "interval", frameIndex, timestampUs)) {
            lastBackgroundTimestampUs = timestampUs;
        }
    }

    /**
     * Exports a KNN background plate immediately with an explicit trigger tag.
     *
     * @param trigger "interval" for 10s saves, "manual" for the UI button
     * @return true when both files were written
     */
    public boolean exportPlate(Mat background, String trigger, int frameIndex, long timestampUs) {
        if (!runActive) {
            Log.w(TAG, "exportPlate with no active run; ignoring");
            return false;
        }
        String base = String.format(Locale.US, "background_%05d_%s", frameIndex, trigger);
        boolean ok = writePair(background, base, "plate", frameIndex, timestampUs, trigger);
        if (ok) {
            platesSaved++;
        }
        writeManifest();
        return ok;
    }

    // --- writing -------------------------------------------------------------

    private boolean writePair(Mat mat, String base, String kind,
                              int frameIndex, long timestampUs, String trigger) {
        if (mat == null || mat.empty()) {
            Log.w(TAG, "refusing to write empty Mat: " + base);
            return false;
        }
        if (mat.channels() != 3 || mat.depth() != CvType.CV_8U) {
            Log.w(TAG, "unexpected Mat layout for " + base
                    + ": channels=" + mat.channels() + " depth=" + mat.depth());
            return false;
        }

        byte[] raw = new byte[(int) (mat.total() * mat.channels())];
        Mat source = mat.isContinuous() ? mat : mat.clone();
        source.get(0, 0, raw);
        if (source != mat) {
            source.release();
        }

        byte[] png = null;
        MatOfByte encoded = new MatOfByte();
        try {
            if (Imgcodecs.imencode(".png", mat, encoded, PNG_PARAMS)) {
                png = encoded.toArray();
            } else {
                Log.w(TAG, "PNG encode failed for " + base);
            }
        } finally {
            encoded.release();
        }

        boolean ok = writeBytes(base + ".bin", "application/octet-stream", raw);
        if (png != null) {
            ok &= writeBytes(base + ".png", "image/png", png);
        }

        if (ok) {
            try {
                JSONObject entry = new JSONObject();
                entry.put("kind", kind);
                entry.put("bin", base + ".bin");
                entry.put("png", png != null ? base + ".png" : JSONObject.NULL);
                entry.put("frameIndex", frameIndex);
                entry.put("timestampUs", timestampUs);
                entry.put("trigger", trigger);
                entry.put("width", mat.cols());
                entry.put("height", mat.rows());
                entry.put("bytes", raw.length);
                artefacts.put(entry);
            } catch (Exception e) {
                Log.w(TAG, "manifest entry failed for " + base, e);
            }
        } else {
            Log.e(TAG, "write failed for " + base + "; not recording it in the manifest");
        }
        return ok;
    }

    private void writeManifest() {
        try {
            JSONObject root = new JSONObject();
            root.put("schema", MANIFEST_SCHEMA);
            root.put("runId", runId);
            root.put("writtenAt",
                    new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));

            JSONObject layout = new JSONObject();
            layout.put("channelOrder", "Y,U,V");
            layout.put("dtype", "uint8");
            layout.put("binShape", "height, width, 3 (interleaved)");
            layout.put("pngNote", "channels written as BGR by OpenCV; "
                    + "cv2.imread gives [:,:,0]=Y, [:,:,1]=U, [:,:,2]=V");
            root.put("layout", layout);

            JSONObject pipe = new JSONObject();
            pipe.put("clipName", provenance.clipName);
            pipe.put("colourRange", provenance.colourRange);
            pipe.put("colourStandard", provenance.colourStandard);
            pipe.put("expandToFullRange", provenance.expandToFullRange);
            pipe.put("targetWidth", provenance.targetWidth);
            pipe.put("targetHeight", provenance.targetHeight);
            pipe.put("chromaUpsample", "INTER_NEAREST");
            pipe.put("resizeInterpolation", "INTER_AREA");
            pipe.put("lutLuma", "clampByte(Math.round((i - 16) * 255.0 / 219.0))");
            pipe.put("lutChroma", "clampByte(Math.round((i - 128) * 255.0 / 224.0 + 128))");
            pipe.put("lutAppliedAfterResize", true);
            root.put("pipeline", pipe);

            JSONObject knn = new JSONObject();
            knn.put("history", provenance.knnHistory);
            knn.put("dist2Threshold", provenance.knnDist2Threshold);
            knn.put("detectShadows", provenance.knnDetectShadows);
            knn.put("preBlurKernel", provenance.knnPreBlurKernel);
            root.put("knn", knn);

            JSONObject env = new JSONObject();
            env.put("opencvVersion", Core.VERSION);
            env.put("androidSdk", Build.VERSION.SDK_INT);
            env.put("device", Build.MANUFACTURER + " " + Build.MODEL);
            root.put("environment", env);

            root.put("framesSeen", framesSeen);
            root.put("platesSaved", platesSaved);
            root.put("backgroundIntervalUs", BACKGROUND_INTERVAL_US);
            root.put("artefacts", artefacts);

            rewriteManifest(root.toString(2).getBytes("UTF-8"));
        } catch (Exception e) {
            Log.e(TAG, "manifest write failed", e);
        }
    }

    // --- storage -------------------------------------------------------------

    private String relativeDir() {
        return Environment.DIRECTORY_DOCUMENTS + File.separator + ROOT_FOLDER
                + File.separator + runId;
    }

    private File legacyDir() {
        File dir = new File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
                ROOT_FOLDER + File.separator + runId);
        //noinspection ResultOfMethodCallIgnored
        dir.mkdirs();
        return dir;
    }

    private boolean writeBytes(String name, String mime, byte[] data) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                Uri uri = insertPending(name, mime);
                if (uri == null) {
                    return false;
                }
                try (OutputStream os = appContext.getContentResolver().openOutputStream(uri)) {
                    if (os == null) {
                        return false;
                    }
                    os.write(data);
                }
                clearPending(uri);
            } else {
                try (FileOutputStream fos = new FileOutputStream(new File(legacyDir(), name))) {
                    fos.write(data);
                }
            }
            return true;
        } catch (Exception e) {
            Log.e(TAG, "write failed: " + name, e);
            return false;
        }
    }

    private void rewriteManifest(byte[] data) throws Exception {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            if (manifestUri == null) {
                manifestUri = insertPending("manifest.json", "application/json");
                if (manifestUri == null) {
                    return;
                }
            }
            try (OutputStream os =
                         appContext.getContentResolver().openOutputStream(manifestUri, "wt")) {
                if (os != null) {
                    os.write(data);
                }
            }
            clearPending(manifestUri);
        } else {
            if (legacyManifestFile == null) {
                legacyManifestFile = new File(legacyDir(), "manifest.json");
            }
            try (FileOutputStream fos = new FileOutputStream(legacyManifestFile, false)) {
                fos.write(data);
            }
        }
    }

    private Uri insertPending(String name, String mime) {
        ContentValues cv = new ContentValues();
        cv.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
        cv.put(MediaStore.MediaColumns.MIME_TYPE, mime);
        cv.put(MediaStore.MediaColumns.RELATIVE_PATH, relativeDir());
        cv.put(MediaStore.MediaColumns.IS_PENDING, 1);
        Uri collection = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY);
        Uri uri = appContext.getContentResolver().insert(collection, cv);
        if (uri == null) {
            Log.e(TAG, "MediaStore insert returned null for " + name);
        }
        return uri;
    }

    private void clearPending(Uri uri) {
        ContentValues cv = new ContentValues();
        cv.put(MediaStore.MediaColumns.IS_PENDING, 0);
        appContext.getContentResolver().update(uri, cv, null, null);
    }

    private static String sanitise(String s) {
        return s == null ? "clip" : s.replaceAll("[^A-Za-z0-9._-]", "_");
    }
}