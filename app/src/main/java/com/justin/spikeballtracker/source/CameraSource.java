package com.justin.spikeballtracker.source;

import android.content.Context;
import android.media.Image;
import android.util.Log;

import androidx.annotation.OptIn;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.video.YuvFrameBuilder;
import com.justin.spikeballtracker.vision.FramePipeline;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

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
 */
public final class CameraSource {

    private static final String TAG = "CameraSource";

    /** What the analysis stream asks for; the device supplies the nearest it has. */
    private static final android.util.Size ANALYSIS_TARGET =
            new android.util.Size(1280, 720);

    private final Context appContext;
    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();

    private final YuvFrameBuilder yuvBuilder = new YuvFrameBuilder();
    private final Mat rgb = new Mat();
    private final Mat rotated = new Mat();
    private final AtomicInteger frameCounter = new AtomicInteger();

    private ProcessCameraProvider cameraProvider = null;
    private volatile FramePipeline pipeline = null;
    private volatile FrameSink sink = null;
    private long firstTimestampNs = 0L;   // analyser thread only

    public CameraSource(Context context) {
        this.appContext = context.getApplicationContext();
    }

    public boolean isRunning() {
        return pipeline != null;
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
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER))
                        .build();

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setResolutionSelector(resolution)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build();

                analysis.setAnalyzer(cameraExecutor, this::onFrame);

                cameraProvider.unbindAll();
                frameCounter.set(0);
                replacePipeline(new FramePipeline());

                cameraProvider.bindToLifecycle(
                        owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis);

                Log.d(TAG, "camera bound");
            } catch (Exception e) {
                Log.e(TAG, "camera bind failed", e);
            }
        }, ContextCompat.getMainExecutor(appContext));   // bindToLifecycle must be on main
    }

    public void stop() {
        if (cameraProvider != null) {
            ContextCompat.getMainExecutor(appContext)
                    .execute(() -> cameraProvider.unbindAll());
        }
        replacePipeline(null);
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
    }

    /** Swaps the pipeline, releasing the old one on the thread that used it. */
    private void replacePipeline(FramePipeline next) {
        FramePipeline previous = pipeline;
        pipeline = next;
        if (previous != null) {
            cameraExecutor.execute(previous::release);
        }
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private void onFrame(ImageProxy proxy) {
        try {
            FramePipeline current = pipeline;
            FrameSink target = sink;
            if (current == null || target == null) {
                return;   // a frame still in flight after stop()
            }

            Image image = proxy.getImage();
            if (image == null) {
                return;
            }

            Mat yuv = yuvBuilder.build(image);
            Imgproc.resize(yuv, yuv, FrameExtractor.TARGET_SIZE, 0, 0, Imgproc.INTER_AREA);

            // getTimestamp() is nanoseconds from an arbitrary epoch. Rebasing on
            // the first frame is what makes the pipeline's ten-second initial
            // net delay mean "ten seconds after the camera opened".
            int index = frameCounter.getAndIncrement();
            long tsNs = proxy.getImageInfo().getTimestamp();
            if (index == 0) {
                firstTimestampNs = tsNs;
            }
            long timestampUs = (tsNs - firstTimestampNs) / 1_000L;

            long t0 = System.nanoTime();
            FramePipeline.Result r = current.process(yuv, timestampUs, rgb, false);
            long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;

            if (index % 30 == 0) {
                Log.d(TAG, "frame " + index + " " + elapsedMs + "ms"
                        + " cands=" + r.candidates.size()
                        + " net=" + r.netBox);
            }

            target.onDisplayFrame(orientForDisplay(proxy.getImageInfo().getRotationDegrees()));

        } catch (Exception e) {
            Log.e(TAG, "camera frame failed", e);
        } finally {
            proxy.close();   // without this the buffer pool drains and the stream stops
        }
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

    /** @return false when the camera is not running, so there is nothing to reset. */
    public boolean requestBackgroundReset() {
        FramePipeline current = pipeline;
        if (current == null) {
            return false;
        }
        current.requestReset();
        return true;
    }
}