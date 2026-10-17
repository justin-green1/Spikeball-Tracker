package com.justin.spikeballtracker.source;

import android.content.Context;
import android.media.Image;
import android.util.Log;

import androidx.annotation.OptIn;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.ResolutionInfo;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;
import com.justin.spikeballtracker.WorkerThread;
import com.justin.spikeballtracker.log.Formats;
import com.justin.spikeballtracker.log.Json;
import com.justin.spikeballtracker.log.RunLog;
import com.justin.spikeballtracker.record.RallyFootage;
import com.justin.spikeballtracker.record.ReplayBuffer;
import com.justin.spikeballtracker.tracking.RallyDetector;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.video.YuvFrameBuilder;
import com.justin.spikeballtracker.video.YuvRange;
import com.justin.spikeballtracker.vision.FramePipeline;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.List;

/**
 * Live camera frames, converted into the pipeline's working format and run
 * through the same {@link FramePipeline} the recorded-video path uses.
 *
 * <p>Conversion, detection and rendering all happen on the Analysis
 * {@link WorkerThread}, shared with the video path (the two never run at
 * the same time). The public methods may be called from the UI thread: they
 * post to the Analysis thread, or, for binding the camera, to the main
 * thread as CameraX requires. All other state is touched only on the
 * Analysis thread, so it needs no locks.
 *
 * <p>The analysis buffer is always landscape regardless of how the phone is
 * held, and the detectors' geometry priors assume the world is upright in that
 * buffer — which is only true with the activity locked to landscape.
 *
 * <p>A second use case, {@link Preview}, runs alongside the analysis stream and
 * feeds {@link ReplayBuffer}'s encoder, which holds the last minute of 1080p
 * footage. Each confirmed rally's footage is copied out of it, so the rally
 * survives the minute rolling on. The recording runs on its own Recorder
 * thread inside ReplayBuffer, so a slow pipeline drops analysis frames
 * (only the latest is kept) but never recorded ones.
 *
 * <p>Both streams ask for 1920x1080. That is a request: CameraX picks the
 * nearest size the device can run for this combination of streams, and the
 * sizes it settled on are logged once bound. The analysis frames are resized
 * plane by plane from whatever size arrives, so a different size changes the
 * detail the detectors see but never breaks them.
 */
public final class CameraSource {

    private static final String TAG = "CameraSource";

    /**
     * Expand the analysis frames from limited to full range, as the video
     * path does for limited-range files, so both feed the detectors the same
     * numbers.
     *
     * <p>Applied regardless of the frame's dataspace label. On the test phone
     * the analysis stream is labelled full range (JFIF), but its content is
     * limited: the session MP4, which holds the camera's numbers unchanged,
     * clips its highlights at Y 234-235 (live-vs-video test 1, 17 Oct 2026).
     * The label and the measured range are both logged every run.
     */
    private static final boolean EXPAND_CAMERA_RANGE = true;

    /** What the analysis stream asks for; the device supplies the nearest it has. */
    private static final android.util.Size ANALYSIS_TARGET =
            new android.util.Size(1920, 1080);

    /** What the replay recording asks for: footage is kept at full resolution. */
    private static final android.util.Size RECORD_TARGET =
            new android.util.Size(1920, 1080);

    private final Context appContext;
    private final WorkerThread analysisThread;
    private final ReplayBuffer replayBuffer;

    private final YuvFrameBuilder yuvBuilder = new YuvFrameBuilder("camera");
    private final YuvRange range = new YuvRange();
    private final Mat rgb = new Mat();
    private final Mat rotated = new Mat();

    private ProcessCameraProvider cameraProvider = null;

    /** One camera session's pipeline, log and clock. */
    private static final class Run {
        final RunLog log;
        final FramePipeline pipeline;
        int frames = 0;

        Run(Context appContext) {
            log = RunLog.open(appContext, "camera");
            pipeline = new FramePipeline(log);
            if (log != null) {
                log.write(FramePipeline.header("camera", log)
                        .field("note", "src = camera sensor time in us; t = src minus the first frame's")
                        .end().toString());
            }
        }

        /**
         * Sensor time of this session's first and newest frames, in ns.
         * The pipeline's timestamps count from the first; the replay ring is
         * stamped with raw sensor time.
         */
        long firstTimestampNs = -1L;
        long lastTimestampNs = -1L;
    }

    // Analysis thread only.
    private Run run = null;
    private FrameSink sink = null;
    private RallyListener rallyListener = null;

    public CameraSource(Context context, WorkerThread analysis) {
        this.appContext = context.getApplicationContext();
        this.analysisThread = analysis;
        this.replayBuffer = new ReplayBuffer(appContext);
        // The session video's name and start time go into the current run's log.
        replayBuffer.setEventSink(line -> analysisThread.post(() -> {
            if (run != null && run.log != null) {
                run.log.write(line);
            } else {
                Log.i(TAG, "recording event outside a run: " + line);
            }
        }));
    }

    /**
     * Told about each confirmed rally once its footage has been copied out of
     * the replay ring; called on ReplayBuffer's Recorder thread. Null for none.
     */
    public void setRallyListener(RallyListener listener) {
        analysisThread.post(() -> rallyListener = listener);
    }

    /** Caller must already hold CAMERA permission. */
    public void start(LifecycleOwner owner, FrameSink frameSink) {
        // A fresh session, set up before the camera is bound and so before any frame.
        analysisThread.post(() -> {
            endRun();
            run = new Run(appContext);
            sink = frameSink;
        });

        ListenableFuture<ProcessCameraProvider> providerFuture =
                ProcessCameraProvider.getInstance(appContext);

        providerFuture.addListener(() -> {
            try {
                cameraProvider = providerFuture.get();

                ResolutionSelector resolution = new ResolutionSelector.Builder()
                        .setAspectRatioStrategy(
                                AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(new ResolutionStrategy(
                                ANALYSIS_TARGET,
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                        .build();

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setResolutionSelector(resolution)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build();

                analysis.setAnalyzer(analysisThread.executor(), this::onFrame);

                Preview recording = new Preview.Builder()
                        .setResolutionSelector(new ResolutionSelector.Builder()
                                .setAspectRatioStrategy(
                                        AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                                .setResolutionStrategy(new ResolutionStrategy(
                                        RECORD_TARGET,
                                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                                .build())
                        .build();
                replayBuffer.attachTo(recording);

                cameraProvider.unbindAll();

                try {
                    cameraProvider.bindToLifecycle(
                            owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis, recording);
                } catch (IllegalArgumentException e) {
                    // The device cannot run both streams at these sizes. Detection
                    // matters more than the replay, so keep analysis alone.
                    Log.w(TAG, "analysis + recording not supported; binding analysis only", e);
                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(
                            owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis);
                }
                logBoundSize("analysis", analysis.getResolutionInfo(), ANALYSIS_TARGET);
                logBoundSize("recording", recording.getResolutionInfo(), RECORD_TARGET);
            } catch (Exception e) {
                Log.e(TAG, "camera bind failed", e);
            }
        }, ContextCompat.getMainExecutor(appContext));   // bindToLifecycle must be on main
    }

    /** Logs the size CameraX settled on for one stream, warning if it is not the one asked for. */
    private static void logBoundSize(String stream, ResolutionInfo info,
                                     android.util.Size requested) {
        if (info == null) {
            Log.i(TAG, stream + " stream not bound");
            return;
        }
        android.util.Size actual = info.getResolution();
        if (actual.equals(requested)) {
            Log.i(TAG, stream + " stream bound at " + actual);
        } else {
            Log.w(TAG, stream + " stream bound at " + actual + ", not the requested " + requested);
        }
    }

    public void stop() {
        if (cameraProvider != null) {
            ContextCompat.getMainExecutor(appContext)
                    .execute(() -> cameraProvider.unbindAll());
        }
        analysisThread.post(() -> {
            endRun();
            sink = null;
        });
    }

    /** Stops and frees everything. Posted, so it runs after any frame already queued. */
    public void release() {
        stop();
        analysisThread.post(() -> {
            yuvBuilder.release();
            range.release();
            rgb.release();
            rotated.release();
        });
        replayBuffer.release();
    }

    /**
     * Ends the current session: a rally still in progress is confirmed now,
     * and the pipeline is freed. The ring is kept after the camera stops, so
     * the rally's footage can still be copied. Analysis thread only.
     */
    private void endRun() {
        Run previous = run;
        run = null;
        if (previous == null) {
            return;
        }
        try {
            reportRallies(previous, previous.pipeline.finish(), rallyListener);
        } catch (Exception e) {
            Log.e(TAG, "rally finish failed", e);
        }
        previous.pipeline.release();
        if (previous.log != null) {
            previous.log.write(new Json().obj().field("type", "end")
                    .field("frames", previous.frames).end().toString());
            previous.log.close();
        }
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private void onFrame(ImageProxy proxy) {
        try {
            Run current = run;
            FrameSink target = sink;
            if (current == null || target == null) {
                return;   // a frame queued before stop() ran
            }

            Image image = proxy.getImage();
            if (image == null) {
                return;
            }

            // Each plane resized directly to the working size, its range
            // measured as delivered, then expanded to full range.
            Mat yuv = yuvBuilder.build(image, FrameExtractor.TARGET_SIZE);
            YuvRange.Stats inputRange = RunLog.ENABLED ? range.measure(yuv) : null;
            if (EXPAND_CAMERA_RANGE) {
                range.expandToFull(yuv);
            }

            // getTimestamp() is nanoseconds from an arbitrary epoch. Rebasing on
            // the first frame is what makes the pipeline's ten-second initial
            // net delay mean "ten seconds after the camera opened".
            long tsNs = proxy.getImageInfo().getTimestamp();
            if (current.firstTimestampNs < 0) {
                current.firstTimestampNs = tsNs;
            }
            current.lastTimestampNs = tsNs;
            long timestampUs = (tsNs - current.firstTimestampNs) / 1_000L;

            if (current.frames++ == 0 && current.log != null) {
                logStream(current.log, image, proxy.getImageInfo().getRotationDegrees());
            }
            List<RallyDetector.Rally> rallies =
                    current.pipeline.process(yuv, timestampUs, tsNs / 1_000L, inputRange, rgb);
            reportRallies(current, rallies, rallyListener);

            target.onDisplayFrame(orientForDisplay(proxy.getImageInfo().getRotationDegrees()));

        } catch (Exception e) {
            Log.e(TAG, "camera frame failed", e);
        } finally {
            proxy.close();   // without this the buffer pool drains and the stream stops
        }
    }

    /**
     * If anyone is listening, asks the replay buffer to copy each rally's
     * footage, the rally plus its margins, and hand both to the listener.
     * The copy happens on the Recorder thread, so the analyser never waits
     * for it. Analysis thread only.
     */
    private void reportRallies(Run run, List<RallyDetector.Rally> rallies,
                               RallyListener listener) {
        if (listener == null) {
            return;
        }
        // Pipeline time counts from the session's first frame; the ring is on raw sensor time.
        long baseUs = run.firstTimestampNs / 1_000L;
        long nowUs = run.lastTimestampNs / 1_000L;
        for (RallyDetector.Rally rally : rallies) {
            replayBuffer.copyRally(
                    baseUs + rally.startUs - RallyFootage.BEFORE_US,
                    baseUs + rally.endUs + RallyFootage.AFTER_US,
                    nowUs,
                    footage -> listener.onRally(rally, footage));
        }
    }

    /** What the camera actually delivers: size, strides and declared colour encoding. */
    private void logStream(RunLog log, Image image, int rotationDegrees) {
        Json j = new Json().obj()
                .field("type", "stream")
                .field("geometry", yuvBuilder.geometry())
                .field("rotation", rotationDegrees)
                .field("range_expansion", EXPAND_CAMERA_RANGE);
        Formats.dataSpace(j, "dataspace", image.getDataSpace());
        log.write(j.end().toString());
    }

    /**
     * Locked to landscape this is always 0 and the frame passes through
     * untouched. Kept as a guard, not as a feature.
     */
    private Mat orientForDisplay(int rotationDegrees) {
        switch (rotationDegrees) {
            case 90:
                Core.rotate(rgb, rotated, Core.ROTATE_90_CLOCKWISE);
                return rotated;
            case 180:
                Core.rotate(rgb, rotated, Core.ROTATE_180);
                return rotated;
            case 270:
                Core.rotate(rgb, rotated, Core.ROTATE_90_COUNTERCLOCKWISE);
                return rotated;
            default:
                return rgb;
        }
    }

    /** Clears the learned background, net and tracks; does nothing when the camera is not running. */
    public void requestBackgroundReset() {
        analysisThread.post(() -> {
            if (run != null) {
                run.pipeline.requestReset();
            }
        });
    }
}
