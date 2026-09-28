package com.justin.spikeballtracker.record;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import com.justin.spikeballtracker.tracking.BallCandidate;

import org.json.JSONObject;
import org.opencv.core.Core;
import org.opencv.core.Rect;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Writes what the pipeline decided on every frame (ball candidates and net
 * box) to a CSV, so a run can be scored and replayed on a computer.
 *
 * <p>Files land beside the background plates, in
 * {@code Documents/SpikeballTracker/<runId>/}:
 * <ul>
 *   <li>{@code detections.csv}, one row per processed frame:
 *   <pre>Frame,TimestampUs,SourceTimestampUs,Candidates,Areas,NetX,NetY,NetW,NetH</pre>
 *   {@code Candidates} is a stringified Python list of {@code (x, y)} tuples,
 *   the format every script in the Python repo already reads with
 *   {@code ast.literal_eval}; {@code Areas} is the matching list of pixel
 *   areas. The net columns are empty until the first lock.</li>
 *   <li>{@code detections_meta.json}: coordinate space, indexing and source,
 *   everything needed to line the rows up with the footage.</li>
 * </ul>
 *
 * <p>Conventions, chosen to match the Python repo rather than the Java:
 * {@code Frame} is 1-indexed (pipeline frameIndex + 1). Coordinates are in
 * the pipeline's 854x480 working frame, not the video's native size. For
 * video, {@code SourceTimestampUs} is the presentation timestamp, which is the
 * robust key for matching rows to decoded frames. For the camera it is the
 * sensor timestamp, the same clock the replay buffer's encoder sees.
 *
 * <p>Opening storage is deferred to the first {@link #write}, so constructing
 * one on the UI thread costs nothing. A storage failure is logged once and
 * disables the log; it never interrupts detection.
 *
 * <p>Not thread safe. Write and close from the processing thread only.
 */
public final class DetectionLog {

    private static final String TAG = "DetectionLog";
    private static final String ROOT_FOLDER = "SpikeballTracker";
    private static final int SCHEMA = 1;

    /** Rows between flushes: about 10 s at 30 fps, the most a crash can lose. */
    private static final int FLUSH_EVERY = 300;

    private static final String HEADER =
            "Frame,TimestampUs,SourceTimestampUs,Candidates,Areas,NetX,NetY,NetW,NetH\n";

    /** Describes the run; written once to detections_meta.json. */
    public static final class Meta {
        public String source = "unknown";        // "video" or "camera"
        public String clipName = "unknown";
        public int frameWidth = 854;
        public int frameHeight = 480;
        public String colourRange = "n/a";
        public String colourStandard = "n/a";
        public boolean expandToFullRange = false;
        public int knnHistory = -1;
        public double knnDist2Threshold = -1;
        public boolean knnDetectShadows = true;
        public int knnPreBlurKernel = -1;
        public String note = "";
    }

    private final Context appContext;
    private final String runId;
    private final Meta meta;

    private Writer csv = null;
    private Uri csvUri = null;          // API 29+ only
    private boolean failed = false;
    private boolean closed = false;
    private int rows = 0;

    private final StringBuilder line = new StringBuilder(256);

    /**
     * @param runId folder name under Documents/SpikeballTracker/; pass the
     *              background exporter's id so plates and detections of one
     *              run share a folder
     */
    public DetectionLog(Context context, String runId, Meta meta) {
        this.appContext = context.getApplicationContext();
        this.runId = runId;
        this.meta = meta;
    }

    /** A folder name for runs with no exporter to share one with. */
    public static String newRunId(String label) {
        String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
        return stamp + "_" + (label == null ? "run" : label.replaceAll("[^A-Za-z0-9._-]", "_"));
    }

    /**
     * Appends one frame's result.
     *
     * @param frameIndex        pipeline frame index, 0-based (written +1)
     * @param timestampUs       the timestamp the pipeline was given
     * @param sourceTimestampUs PTS for video, sensor time for the camera
     * @param netBox            null until the net is first locked
     */
    public void write(int frameIndex, long timestampUs, long sourceTimestampUs,
                      List<BallCandidate> candidates, Rect netBox) {
        if (failed || closed) {
            return;
        }
        if (csv == null && !open()) {
            return;
        }

        line.setLength(0);
        line.append(frameIndex + 1).append(',')
                .append(timestampUs).append(',')
                .append(sourceTimestampUs).append(",\"[");
        for (int i = 0; i < candidates.size(); i++) {
            BallCandidate c = candidates.get(i);
            if (i > 0) line.append(", ");
            line.append('(').append(fmt(c.x)).append(", ").append(fmt(c.y)).append(')');
        }
        line.append("]\",\"[");
        for (int i = 0; i < candidates.size(); i++) {
            if (i > 0) line.append(", ");
            line.append(candidates.get(i).area);
        }
        line.append("]\",");
        if (netBox != null) {
            line.append(netBox.x).append(',').append(netBox.y).append(',')
                    .append(netBox.width).append(',').append(netBox.height);
        } else {
            line.append(",,,");
        }
        line.append('\n');

        try {
            csv.write(line.toString());
            if (++rows % FLUSH_EVERY == 0) {
                csv.flush();
            }
        } catch (IOException e) {
            fail("CSV write failed", e);
        }
    }

    /** Flushes and publishes the file. Safe to call more than once. */
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (csv == null) {
            return;   // nothing was ever written, so no empty file is created
        }
        try {
            csv.close();
        } catch (IOException e) {
            Log.w(TAG, "CSV close failed", e);
        }
        if (csvUri != null) {
            clearPending(csvUri);
        }
        Log.i(TAG, "detections saved: " + rows + " frames in Documents/"
                + ROOT_FOLDER + "/" + runId + "/detections.csv");
    }

    // --- storage -------------------------------------------------------------

    private boolean open() {
        try {
            writeMeta();
            OutputStream os;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                csvUri = insert("detections.csv", "text/csv", true);
                os = appContext.getContentResolver().openOutputStream(csvUri, "wt");
                if (os == null) {
                    throw new IOException("no output stream for " + csvUri);
                }
            } else {
                os = new FileOutputStream(new File(legacyDir(), "detections.csv"), false);
            }
            csv = new BufferedWriter(new OutputStreamWriter(os, StandardCharsets.UTF_8), 64 * 1024);
            csv.write(HEADER);
            Log.i(TAG, "logging detections to Documents/" + ROOT_FOLDER + "/" + runId + "/");
            return true;
        } catch (Exception e) {
            fail("could not open detection log", e);
            return false;
        }
    }

    private void writeMeta() throws Exception {
        JSONObject root = new JSONObject();
        root.put("schema", SCHEMA);
        root.put("runId", runId);
        root.put("startedAt",
                new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date()));
        root.put("source", meta.source);
        root.put("clipName", meta.clipName);

        JSONObject coords = new JSONObject();
        coords.put("frameWidth", meta.frameWidth);
        coords.put("frameHeight", meta.frameHeight);
        coords.put("space", "pipeline working frame; scale x by nativeWidth/frameWidth "
                + "and y by nativeHeight/frameHeight to compare with native-pixel labels");
        coords.put("frameIndexing", "1-based: Frame = pipeline frameIndex + 1");
        coords.put("timestampUs", "timestamp passed to FramePipeline.process");
        coords.put("sourceTimestampUs", "video: presentation timestamp (match decoded "
                + "frames on this); camera: sensor timestamp in microseconds");
        coords.put("candidates", "stringified list of (x, y) blob centroids; "
                + "Areas is the matching list of pixel areas");
        coords.put("net", "NetX,NetY,NetW,NetH bounding box; empty until first lock");
        root.put("layout", coords);

        JSONObject colour = new JSONObject();
        colour.put("colourRange", meta.colourRange);
        colour.put("colourStandard", meta.colourStandard);
        colour.put("expandToFullRange", meta.expandToFullRange);
        root.put("colour", colour);

        JSONObject knn = new JSONObject();
        knn.put("history", meta.knnHistory);
        knn.put("dist2Threshold", meta.knnDist2Threshold);
        knn.put("detectShadows", meta.knnDetectShadows);
        knn.put("preBlurKernel", meta.knnPreBlurKernel);
        root.put("knn", knn);

        JSONObject env = new JSONObject();
        env.put("opencvVersion", Core.VERSION);
        env.put("androidSdk", Build.VERSION.SDK_INT);
        env.put("device", Build.MANUFACTURER + " " + Build.MODEL);
        root.put("environment", env);

        if (!meta.note.isEmpty()) {
            root.put("note", meta.note);
        }

        byte[] bytes = root.toString(2).getBytes(StandardCharsets.UTF_8);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            Uri uri = insert("detections_meta.json", "application/json", false);
            try (OutputStream os = appContext.getContentResolver().openOutputStream(uri, "wt")) {
                if (os == null) {
                    throw new IOException("no output stream for " + uri);
                }
                os.write(bytes);
            }
        } else {
            try (FileOutputStream fos =
                         new FileOutputStream(new File(legacyDir(), "detections_meta.json"))) {
                fos.write(bytes);
            }
        }
    }

    /**
     * @param pending true for a file still being written; it stays hidden from
     *                other apps until {@link #clearPending}. Same MediaStore
     *                pattern as BackgroundExporter.
     */
    private Uri insert(String name, String mime, boolean pending) throws IOException {
        ContentValues cv = new ContentValues();
        cv.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
        cv.put(MediaStore.MediaColumns.MIME_TYPE, mime);
        cv.put(MediaStore.MediaColumns.RELATIVE_PATH,
                Environment.DIRECTORY_DOCUMENTS + File.separator + ROOT_FOLDER
                        + File.separator + runId);
        if (pending) {
            cv.put(MediaStore.MediaColumns.IS_PENDING, 1);
        }
        ContentResolver resolver = appContext.getContentResolver();
        Uri uri = resolver.insert(
                MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), cv);
        if (uri == null) {
            throw new IOException("MediaStore insert returned null for " + name);
        }
        return uri;
    }

    private void clearPending(Uri uri) {
        try {
            ContentValues cv = new ContentValues();
            cv.put(MediaStore.MediaColumns.IS_PENDING, 0);
            appContext.getContentResolver().update(uri, cv, null, null);
        } catch (Exception e) {
            Log.w(TAG, "could not publish " + uri, e);
        }
    }

    private File legacyDir() {
        File dir = new File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
                ROOT_FOLDER + File.separator + runId);
        //noinspection ResultOfMethodCallIgnored
        dir.mkdirs();
        return dir;
    }

    private void fail(String what, Exception e) {
        failed = true;
        Log.e(TAG, what + "; detection logging disabled for this run", e);
        if (csv != null) {
            try {
                csv.close();
            } catch (IOException ignored) {
                // already failing
            }
        }
        if (csvUri != null) {
            clearPending(csvUri);   // keep what was written rather than hide it
        }
    }

    private static String fmt(double v) {
        return String.format(Locale.US, "%.1f", v);
    }
}