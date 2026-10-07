package com.justin.spikeballtracker.record;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.media.MediaMuxer;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.util.Log;

import java.io.File;
import java.io.IOException;

/**
 * Where one saved MP4 goes, {@code Movies/SpikeballTracker/<name>}: a pending
 * MediaStore entry on API 29+, a plain file in the public Movies folder below
 * that. Same split as DetectionLog.
 *
 * <p>Open, write through {@link #newMuxer()}, then always {@link #finish}.
 */
final class Mp4File {

    private static final String TAG = "Mp4File";
    private static final String FOLDER = "SpikeballTracker";

    private final Context ctx;
    private Uri uri;
    private ParcelFileDescriptor pfd;
    private File file;

    private Mp4File(Context ctx) {
        this.ctx = ctx;
    }

    static Mp4File open(Context ctx, String name) throws IOException {
        Mp4File o = new Mp4File(ctx);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentResolver resolver = ctx.getContentResolver();
            ContentValues cv = new ContentValues();
            cv.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
            cv.put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4");
            cv.put(MediaStore.MediaColumns.RELATIVE_PATH,
                    Environment.DIRECTORY_MOVIES + File.separator + FOLDER);
            cv.put(MediaStore.MediaColumns.IS_PENDING, 1);
            o.uri = resolver.insert(
                    MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), cv);
            if (o.uri == null) {
                throw new IOException("MediaStore insert returned null for " + name);
            }
            // MP4 writing seeks back to patch the header, so the fd must be read-write.
            o.pfd = resolver.openFileDescriptor(o.uri, "rw");
            if (o.pfd == null) {
                resolver.delete(o.uri, null, null);
                throw new IOException("could not open " + o.uri);
            }
        } else {
            File dir = new File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
                    FOLDER);
            //noinspection ResultOfMethodCallIgnored
            dir.mkdirs();
            o.file = new File(dir, name);
        }
        return o;
    }

    MediaMuxer newMuxer() throws IOException {
        int fmt = MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4;
        return pfd != null
                ? new MediaMuxer(pfd.getFileDescriptor(), fmt)
                : new MediaMuxer(file.getAbsolutePath(), fmt);
    }

    /** Publishes the file on success; deletes the partial one on failure. */
    void finish(boolean ok) {
        if (pfd != null) {
            try {
                pfd.close();
            } catch (IOException e) {
                Log.w(TAG, "fd close failed", e);
            }
        }
        if (uri != null) {
            ContentResolver resolver = ctx.getContentResolver();
            if (ok) {
                ContentValues cv = new ContentValues();
                cv.put(MediaStore.MediaColumns.IS_PENDING, 0);
                resolver.update(uri, cv, null, null);
            } else {
                resolver.delete(uri, null, null);
            }
        } else if (file != null) {
            if (ok) {
                MediaScannerConnection.scanFile(ctx,
                        new String[]{file.getAbsolutePath()}, new String[]{"video/mp4"}, null);
            } else {
                //noinspection ResultOfMethodCallIgnored
                file.delete();
            }
        }
    }
}
