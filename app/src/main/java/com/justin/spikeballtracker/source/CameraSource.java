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
import com.justin.spikeballtracker.BackgroundThread;
import com.justin.spikeballtracker.record.RallyFootage;
import com.justin.spikeballtracker.record.ReplayBuffer;
import com.justin.spikeballtracker.tracking.RallyDetector;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.video.YuvFrameBuilder;
import com.justin.spikeballtracker.vision.FramePipeline;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.List;

/**
 * Live camera frames, converted into the pipeline's working format and run
 * through the same {@link FramePipeline} the recorded-video path uses.
 *
 * <p>Conversion, detection and rendering all happen on the shared
 * {@link BackgroundThread}, as do the replay encoder's callbacks. The public
 * methods may be called from the UI thread: they post to the background
 * thread, or, for binding the camera, to the main thread as CameraX
 * requires. All other state is touched only on the background thread, so
 * it needs no locks.
 *
 * <p>The analysis buffer is always landscape regardless of how the phone is
 * held, and the detectors' geometry priors assume the world is upright in that
 * buffer — which is only true with the activity locked to landscape.
 *
 * <p>A second use case, {@link Preview}, runs alongside the analysis stream and
 * feeds {@link ReplayBuffer}'s encoder, which holds the last minute of 1080p
 * footage. Each confirmed rally's footage is copied out of it, so the rally
 * survives the minute rolling on. A slow pipeline drops analysis frames
 * (only the latest is kept). The recording comes from the camera straight
 * into the encoder, but its output is collected on the same background
 * thread, so it only stays smooth while no single job holds that thread
 * for more than about a tenth of a second.
 *
 * <p>Both streams ask for 1920x1080. That is a request: CameraX picks the
 * nearest size the device can run for this combination of streams, and the
 * sizes it settled on are logged once bound. The analysis frames are resized
 * plane by plane from whatever size arrives, so a different size changes the
 * detail the detectors see but never breaks them.
 */
public final class CameraSource {

    private static final String TAG = "CameraSource";

    /** What the analysis stream asks for; the device supplies the nearest it has. */
    private static final android.util.Size ANALYSIS_TARGET =
            new android.util.Size(1920, 1080);

    /** What the replay recording asks for: footage is kept at full resolution. */
    private static final android.util.Size RECORD_TARGET =
            new android.util.Size(1920, 1080);

    private final Context appContext;
    private final BackgroundThread background;
    private final ReplayBuffer replayBuffer;

    private final YuvFrameBuilder yuvBuilder = new YuvFrameBuilder("camera");
    private final Mat rgb = new Mat();
    private final Mat rotated = new Mat();

    private ProcessCameraProvider cameraProvider = null;

    /** One camera session's pipeline and clock. */
    private static final class Run {
        final FramePipeline pipeline = new FramePipeline();

        /**
         * Sensor time of this session's first and newest frames, in ns.
         * The pipeline's timestamps count from the first; the replay ring is
         * stamped with raw sensor time.
         */
        long firstTimestampNs = -1L;
        long lastTimestampNs = -1L;
    }

    // Background thread only.
    private Run run = null;
    private FrameSink sink = null;
    private RallyListener rallyListener = null;

    public CameraSource(Context context, BackgroundThread background) {
        this.appContext = context.getApplicationContext();
        this.background = background;
        this.replayBuffer = new ReplayBuffer(background);
    }

    /** Told about each confirmed rally, with its footage, on the background thread; null for none. */
    public void setRallyListener(RallyListener listener) {
        background.post(() -> rallyListener = listener);
    }

    /** Caller must already hold CAMERA permission. */
    public void start(LifecycleOwner owner, FrameSink frameSink) {
        // A fresh session, set up before the camera is bound and so before any frame.
        background.post(() -> {
            endRun();
            run = new Run();
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

                analysis.setAnalyzer(background.executor(), this::onFrame);

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
        background.post(() -> {
            endRun();
            sink = null;
        });
    }

    /** Stops and frees everything. Posted, so it runs after any frame already queued. */
    public void release() {
        stop();
        background.post(() -> {
            yuvBuilder.release();
            rgb.release();
            rotated.release();
        });
        replayBuffer.release();
    }

    /**
     * Ends the current session: a rally still in progress is confirmed now,
     * and the pipeline is freed. The ring is kept after the camera stops, so
     * the rally's footage can still be copied. Background thread only.
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

            // Each plane resized directly to the working size. No range
            // expansion: CameraX YUV is already full range.
            Mat yuv = yuvBuilder.build(image, FrameExtractor.TARGET_SIZE);

            // getTimestamp() is nanoseconds from an arbitrary epoch. Rebasing on
            // the first frame is what makes the pipeline's ten-second initial
            // net delay mean "ten seconds after the camera opened".
            long tsNs = proxy.getImageInfo().getTimestamp();
            if (current.firstTimestampNs < 0) {
                current.firstTimestampNs = tsNs;
            }
            current.lastTimestampNs = tsNs;
            long timestampUs = (tsNs - current.firstTimestampNs) / 1_000L;

            List<RallyDetector.Rally> rallies = current.pipeline.process(yuv, timestampUs, rgb);
            reportRallies(current, rallies, rallyListener);

            target.onDisplayFrame(orientForDisplay(proxy.getImageInfo().getRotationDegrees()));

        } catch (Exception e) {
            Log.e(TAG, "camera frame failed", e);
        } finally {
            proxy.close();   // without this the buffer pool drains and the stream stops
        }
    }

    /**
     * If anyone is listening, copies each rally's footage out of the replay
     * ring and hands both over. Background thread only.
     *
     * <p>The copy is a few ms per 10 s of rally, taken once per rally, seconds
     * after it ended; at worst the analyser drops a frame for it.
     */
    private void reportRallies(Run run, List<RallyDetector.Rally> rallies,
                               RallyListener listener) {
        if (listener == null) {
            return;
        }
        for (RallyDetector.Rally rally : rallies) {
            listener.onRally(rally, copyFootage(run, rally));
        }
    }

    /** The rally plus its margins, from the ring; null if the ring no longer has it. */
    private RallyFootage copyFootage(Run run, RallyDetector.Rally rally) {
        // Pipeline time counts from the session's first frame; the ring is on raw sensor time.
        long baseUs = run.firstTimestampNs / 1_000L;
        return replayBuffer.copyRally(
                baseUs + rally.startUs - RallyFootage.BEFORE_US,
                baseUs + rally.endUs + RallyFootage.AFTER_US,
                run.lastTimestampNs / 1_000L);
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
        background.post(() -> {
            if (run != null) {
                run.pipeline.requestReset();
            }
        });
    }
}
