package com.justin.spikeballtracker;

import android.os.Handler;
import android.os.HandlerThread;

import java.util.concurrent.Executor;

/**
 * The one background thread shared by both frame sources, the replay encoder
 * and the saver. Jobs run one at a time, in the order posted.
 *
 * <p>Reconstructed from its call sites: single-stream-candidates uses this
 * class but the file was never committed. Same shape as the later
 * {@code WorkerThread}, with a fixed name.
 */
public final class BackgroundThread {

    private final HandlerThread thread;
    private final Handler handler;
    private final Executor executor;

    public BackgroundThread() {
        thread = new HandlerThread("Background");
        thread.start();
        handler = new Handler(thread.getLooper());
        executor = handler::post;
    }

    /** Runs the job on this thread, after every job already posted. */
    public void post(Runnable job) {
        handler.post(job);
    }

    /** For APIs that take a Handler (MediaCodec callbacks, delayed posts). */
    public Handler handler() {
        return handler;
    }

    /** For APIs that take an Executor (CameraX). */
    public Executor executor() {
        return executor;
    }

    /** Runs the jobs already posted, then stops the thread. */
    public void quit() {
        thread.quitSafely();
    }
}
