package com.justin.spikeballtracker.record;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.media.MediaMuxer;
import android.net.Uri;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.util.Log;

import java.io.File;
import java.io.IOException;

/**
 * Where one saved MP4 goes, {@code Movies/SpikeballTracker/<name>}: a pending
 * MediaStore entry, published once the file is complete.
 *
 * <p>Open, write through {@link #newMuxer()}, then always {@link #finish}.
 */
final class Mp4File {

    private static final String TAG = "Mp4File";
    private static final String FOLDER = "SpikeballTracker";

    private final ContentResolver resolver;
    private final Uri uri;
    private final ParcelFileDescriptor pfd;

    private Mp4File(ContentResolver resolver, Uri uri, ParcelFileDescriptor pfd) {
        this.resolver = resolver;
        this.uri = uri;
        this.pfd = pfd;
    }

    static Mp4File open(Context ctx, String name) throws IOException {
        ContentResolver resolver = ctx.getContentResolver();
        ContentValues cv = new ContentValues();
        cv.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
        cv.put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4");
        cv.put(MediaStore.MediaColumns.RELATIVE_PATH,
                Environment.DIRECTORY_MOVIES + File.separator + FOLDER);
        cv.put(MediaStore.MediaColumns.IS_PENDING, 1);
        Uri uri = resolver.insert(
                MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), cv);
        if (uri == null) {
            throw new IOException("MediaStore insert returned null for " + name);
        }
        // MP4 writing seeks back to patch the header, so the fd must be read-write.
        ParcelFileDescriptor pfd = resolver.openFileDescriptor(uri, "rw");
        if (pfd == null) {
            resolver.delete(uri, null, null);
            throw new IOException("could not open " + uri);
        }
        return new Mp4File(resolver, uri, pfd);
    }

    MediaMuxer newMuxer() throws IOException {
        return new MediaMuxer(pfd.getFileDescriptor(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
    }

    /** Publishes the file on success; deletes the partial one on failure. */
    void finish(boolean ok) {
        try {
            pfd.close();
        } catch (IOException e) {
            Log.w(TAG, "fd close failed", e);
        }
        if (ok) {
            ContentValues cv = new ContentValues();
            cv.put(MediaStore.MediaColumns.IS_PENDING, 0);
            resolver.update(uri, cv, null, null);
        } else {
            resolver.delete(uri, null, null);
        }
    }
}
