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
import com.justin.spikeballtracker.record.RallyFootage;
import com.justin.spikeballtracker.record.ReplayBuffer;
import com.justin.spikeballtracker.tracking.RallyDetector;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.video.YuvFrameBuilder;
import com.justin.spikeballtracker.vision.FramePipeline;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Live camera frames, converted into the pipeline's working format and run
 * through the same {@link FramePipeline} the recorded-video path uses.
 *
 * <p>Conversion, detection and rendering all happen on one private thread.
 * That thread is deliberately not shared with the video path, whose task holds
 * its executor for the length of a whole clip.
 *
 * <p>The analysis buffer is always landscape regardless of how the phone is
 * held, and the detectors' geometry priors assume the world is upright in that
 * buffer — which is only true with the activity locked to landscape.
 *
 * <p>A second use case, {@link Preview}, runs alongside the analysis stream and
 * feeds {@link ReplayBuffer}'s encoder, which holds the last minute of 1080p
 * footage. Each confirmed rally's footage is copied out of it, so the rally
 * survives the minute rolling on. The two streams are independent: a slow
 * pipeline drops analysis frames but never recorded ones.
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
    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    private final ReplayBuffer replayBuffer;

    private final YuvFrameBuilder yuvBuilder = new YuvFrameBuilder("camera");
    private final Mat rgb = new Mat();
    private final Mat rotated = new Mat();

    private ProcessCameraProvider cameraProvider = null;

    /**
     * One camera session's pipeline and clock. Swapped as a unit through a
     * single volatile field, so a frame can never be processed by one
     * session's pipeline against another session's clock.
     */
    private static final class Run {
        final FramePipeline pipeline = new FramePipeline();

        /**
         * Sensor time of this session's first and newest frames, in ns.
         * Analyser thread only. The pipeline's timestamps count from the
         * first; the replay ring is stamped with raw sensor time.
         */
        long firstTimestampNs = -1L;
        long lastTimestampNs = -1L;
    }

    private volatile Run run = null;
    private volatile FrameSink sink = null;
    private volatile RallyListener rallyListener = null;

    public CameraSource(Context context) {
        this.appContext = context.getApplicationContext();
        this.replayBuffer = new ReplayBuffer();
    }

    /** Told about each confirmed rally, with its footage, on the camera thread; null for none. */
    public void setRallyListener(RallyListener listener) {
        this.rallyListener = listener;
    }

    /** Caller must already hold CAMERA permission. */
    public void start(LifecycleOwner owner, FrameSink frameSink) {
        this.sink = frameSink;

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

                analysis.setAnalyzer(cameraExecutor, this::onFrame);

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
                replaceRun(new Run());

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
        replaceRun(null);
        sink = null;
    }

    /**
     * Frees everything, in an order that survives frames still in flight.
     *
     * <p>{@code shutdown}, not {@code shutdownNow}: the queued releases are the
     * whole point, and cancelling them would leak the native buffers.
     */
    public void release() {
        stop();
        cameraExecutor.execute(() -> {
            yuvBuilder.release();
            rgb.release();
            rotated.release();
        });
        cameraExecutor.shutdown();
        try {
            if (!cameraExecutor.awaitTermination(1, TimeUnit.SECONDS)) {
                Log.w(TAG, "camera executor did not drain in time");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        replayBuffer.release();
    }

    /**
     * Swaps the session, finishing the old one's rallies and releasing its
     * pipeline on the thread that used it. The release is queued behind any
     * frame still in flight.
     */
    private void replaceRun(Run next) {
        Run previous = run;
        run = next;
        if (previous != null) {
            RallyListener listener = rallyListener;
            cameraExecutor.execute(() -> {
                // A rally still in progress is confirmed now. The ring is kept
                // after the camera stops, so its footage can still be copied.
                try {
                    reportRallies(previous, previous.pipeline.finish(), listener);
                } catch (Exception e) {
                    Log.e(TAG, "rally finish failed", e);
                }
                previous.pipeline.release();
            });
        }
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private void onFrame(ImageProxy proxy) {
        try {
            Run current = run;
            FrameSink target = sink;
            if (current == null || target == null) {
                return;   // a frame still in flight after stop()
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
     * ring and hands both over. Analyser thread only.
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
        Run current = run;
        if (current != null) {
            current.pipeline.requestReset();
        }
    }
}
