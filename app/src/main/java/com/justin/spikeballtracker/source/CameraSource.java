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
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;
import com.justin.spikeballtracker.record.ReplayBuffer;
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
import com.justin.spikeballtracker.record.DetectionLog;
import com.justin.spikeballtracker.vision.MotionDetector;

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
 * feeds {@link ReplayBuffer}'s encoder, which holds the last minute of 720p
 * footage for {@link #saveReplay}. The two streams are independent: a slow
 * pipeline drops analysis frames but never recorded ones.
 */
public final class CameraSource {

    private static final String TAG = "CameraSource";

    /** What the analysis stream asks for; the device supplies the nearest it has. */
    private static final android.util.Size ANALYSIS_TARGET =
            new android.util.Size(1280, 720);

    /** What the replay recording asks for. */
    private static final android.util.Size RECORD_TARGET =
            new android.util.Size(1280, 720);

    private final Context appContext;
    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    private final ReplayBuffer replayBuffer;

    private final YuvFrameBuilder yuvBuilder = new YuvFrameBuilder();
    private final Mat rgb = new Mat();
    private final Mat rotated = new Mat();
    private final AtomicInteger frameCounter = new AtomicInteger();

    private ProcessCameraProvider cameraProvider = null;
    /**
     * One camera session's pipeline and its detection log. Swapped as a unit
     * through a single volatile field, so a frame can never be processed by one
     * session's pipeline and logged into another session's file.
     */
    private static final class Run {
        final FramePipeline pipeline;
        final DetectionLog log;

        Run(FramePipeline pipeline, DetectionLog log) {
            this.pipeline = pipeline;
            this.log = log;
        }

        /** Analyser thread only: both hold state that thread was using. */
        void release() {
            pipeline.release();
            log.close();
        }
    }

    private volatile Run run = null;
    private volatile FrameSink sink = null;
    private long firstTimestampNs = 0L;   // analyser thread only

    public CameraSource(Context context) {
        this.appContext = context.getApplicationContext();
        this.replayBuffer = new ReplayBuffer(appContext);
    }

    public boolean isRunning() {
        return run != null;
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

                Preview recording = new Preview.Builder()
                        .setResolutionSelector(new ResolutionSelector.Builder()
                                .setAspectRatioStrategy(
                                        AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                                .setResolutionStrategy(new ResolutionStrategy(
                                        RECORD_TARGET,
                                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER))
                                .build())
                        .build();
                replayBuffer.attachTo(recording);

                cameraProvider.unbindAll();
                frameCounter.set(0);
                replaceRun(newRun());

                try {
                    cameraProvider.bindToLifecycle(
                            owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis, recording);
                    Log.d(TAG, "camera bound (analysis + replay recording)");
                } catch (IllegalArgumentException e) {
                    // The device cannot run both streams at these sizes. Detection
                    // matters more than the replay, so keep analysis alone.
                    Log.w(TAG, "analysis + recording not supported; binding analysis only", e);
                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(
                            owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis);
                    Log.d(TAG, "camera bound (analysis only, no replay)");
                }
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
     * Swaps the session, releasing the old pipeline and closing the old log on
     * the thread that used them. The close is queued behind any frame still in
     * flight, so that frame's row is written before the file is published.
     */
    private void replaceRun(Run next) {
        Run previous = run;
        run = next;
        if (previous != null) {
            cameraExecutor.execute(previous::release);
        }
    }

    /** Cheap: the log opens its file on the first frame, on the analyser thread. */
    private Run newRun() {
        DetectionLog.Meta m = new DetectionLog.Meta();
        m.source = "camera";
        m.clipName = "camera";
        m.frameWidth = (int) FrameExtractor.TARGET_SIZE.width;
        m.frameHeight = (int) FrameExtractor.TARGET_SIZE.height;
        m.colourRange = "camera (assumed full)";
        m.colourStandard = "camera";
        m.expandToFullRange = false;   // the camera path applies no LUT
        m.knnHistory = MotionDetector.history();
        m.knnDist2Threshold = MotionDetector.dist2Threshold();
        m.knnDetectShadows = MotionDetector.detectShadows();
        m.knnPreBlurKernel = MotionDetector.preBlurKernel();
        m.note = "analysis frames are resized to the working frame regardless of the "
                + "camera's aspect ratio; frames dropped by KEEP_ONLY_LATEST have no row";
        return new Run(new FramePipeline(),
                new DetectionLog(appContext, DetectionLog.newRunId("camera"), m));
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
            FramePipeline.Result r = current.pipeline.process(yuv, timestampUs, rgb, false);
            long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;

            // Sensor time is the clock the replay encoder also stamps its frames
            // with, so it is what lines these rows up with a saved replay later.
            current.log.write(r.frameIndex, timestampUs, tsNs / 1_000L,
                    r.candidates, r.netBox);

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
        Run current = run;
        if (current == null) {
            return false;
        }
        current.pipeline.requestReset();
        return true;
    }

    /**
     * Seconds of footage available to {@link #saveReplay}. Still valid after
     * {@link #stop()}; the buffer is only cleared when the camera next starts.
     */
    public double replaySeconds() {
        return replayBuffer.bufferedSeconds();
    }

    /** Saves the last {@code seconds} of camera footage as an MP4. Returns at once. */
    public void saveReplay(int seconds, ReplayBuffer.SaveCallback callback) {
        replayBuffer.save(seconds, callback);
    }
}