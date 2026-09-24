package com.justin.spikeballtracker.ui;

import android.graphics.Bitmap;

import android.graphics.ImageDecoder;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.Manifest;
import android.content.pm.PackageManager;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.content.ContextCompat;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;


import com.justin.spikeballtracker.R;
import com.justin.spikeballtracker.tracking.BallCandidate;
import com.justin.spikeballtracker.video.YuvFrameBuilder;
import com.justin.spikeballtracker.vision.BlobDetector;
import com.justin.spikeballtracker.video.BackgroundExporter;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.vision.ColourDetector;
import com.justin.spikeballtracker.vision.FramePipeline;
import com.justin.spikeballtracker.vision.MotionDetector;
import com.justin.spikeballtracker.vision.NetDetector;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import com.google.common.util.concurrent.ListenableFuture;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final Scalar MARKER_COLOUR = new Scalar(255, 0, 0);
    private FrameExtractor frameExtractor;
    private BackgroundExporter backgroundExporter;
    // -----
    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    private ProcessCameraProvider cameraProvider = null;
    private final AtomicInteger cameraFrameCounter = new AtomicInteger();
    //----
    private ImageView frameImageView;
    //----
    private final YuvFrameBuilder cameraYuvBuilder = new YuvFrameBuilder();
    private final Mat cameraRgb = new Mat();
    private final Mat cameraRotated = new Mat();
    //---
    private static final Size PIPELINE_SIZE = new Size(854, 480);

    /** Set by the export button, consumed on the processing thread. */
    private final AtomicBoolean backgroundExportRequested = new AtomicBoolean(false);

    private final ExecutorService processingExecutor = Executors.newSingleThreadExecutor();

    private final ActivityResultLauncher<String[]> videoPickerLauncher =
            registerForActivityResult(new ActivityResultContracts.OpenDocument(), uri -> {
                Log.d(TAG, "picker returned: " + uri);
                if (uri == null) {
                    Log.w(TAG, "No video selected");
                    return;
                }
                displayProcessedVideo(uri);
            });

    private final ActivityResultLauncher<PickVisualMediaRequest> photoPickerLauncher =
            registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
                Log.d(TAG, "photo picker returned: " + uri);
                if (uri == null) {
                    Log.w(TAG, "No photo selected");
                    return;
                }
                displayProcessedPhoto(uri);
            });

    private volatile Future<?> activeVideoTask = null;
    private volatile AtomicBoolean activeVideoCancelFlag = null;

    static {
        boolean openCvLoaded = OpenCVLoader.initLocal();
        if (openCvLoaded) {
            Log.d(TAG, "OpenCV library correctly loaded");
        } else {
            Log.e(TAG, "OpenCV library failed to load");
        }
    }

    private void displayProcessedVideo(Uri videoUri) {
        stopVideo();
        stopCamera();

        final AtomicBoolean cancelled = new AtomicBoolean(false);
        activeVideoCancelFlag = cancelled;

        activeVideoTask = processingExecutor.submit(() -> {
            FramePipeline pipeline = new FramePipeline();

            Mat displayMat = new Mat();   // RGB copy of the frame, drawn on
            List<BallCandidate> allCandidates = new ArrayList<>();

            // The provenance block has to run before the first offerFrame, and the
            // pipeline's own frame index is not available until after process().
            final AtomicBoolean firstFrame = new AtomicBoolean(true);

            // Wall-clock anchor for playback pacing, set on the first frame.
            AtomicLong firstTimestampUs = new AtomicLong(-1);
            AtomicLong playbackStartNs = new AtomicLong(0);
            AtomicLong worstLagMs = new AtomicLong(0);

            try {
                frameExtractor.streamFrames(videoUri, (frame, timestampUs) -> {
                    if (cancelled.get() || Thread.currentThread().isInterrupted()) {
                        return false;   // stop requested: leave the decode loop
                    }

                    if (firstFrame.getAndSet(false)) {
                        // readColourInfo has run by now, so the provenance recorded
                        // is the one actually in force for this clip.
                        FrameExtractor.ColourInfo colour = frameExtractor.colourInfo();
                        BackgroundExporter.Provenance p = new BackgroundExporter.Provenance();
                        p.clipName = videoUri.getLastPathSegment();
                        p.colourRange = colour.range;
                        p.colourStandard = colour.standard;
                        p.expandToFullRange = colour.expandToFullRange;
                        p.targetWidth = frame.cols();
                        p.targetHeight = frame.rows();
                        p.knnHistory = MotionDetector.history();
                        p.knnDist2Threshold = MotionDetector.dist2Threshold();
                        p.knnDetectShadows = MotionDetector.detectShadows();
                        p.knnPreBlurKernel = MotionDetector.preBlurKernel();
                        backgroundExporter.beginRun(p);
                    }

                    // 1. detection, overlays, and the periodic net update
                    boolean manual = backgroundExportRequested.getAndSet(false);
                    boolean plateWanted =
                            manual || backgroundExporter.dueForBackground(timestampUs);

                    FramePipeline.Result r =
                            pipeline.process(frame, timestampUs, displayMat, plateWanted);

                    allCandidates.addAll(r.candidates);

                    // 2. export. offerFrame no longer writes anything, so it can run
                    //    after process(); that lets the pipeline own frame numbering.
                    backgroundExporter.offerFrame(frame, r.frameIndex, timestampUs);

                    if (r.backgroundPlate != null) {
                        backgroundExporter.offerBackground(
                                r.backgroundPlate, r.frameIndex, timestampUs);
                        if (manual) {
                            backgroundExporter.exportPlate(
                                    r.backgroundPlate, "manual", r.frameIndex, timestampUs);
                        }
                    }

                    // 3. render to screen
                    Bitmap bitmap = Bitmap.createBitmap(
                            displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(displayMat, bitmap);
                    runOnUiThread(() -> frameImageView.setImageBitmap(bitmap));

                    // 4. pace playback against the frame's own presentation
                    //    timestamp, not a fixed per-frame delay. A fixed sleep is
                    //    added on top of however long processing took, so the video
                    //    always ran slower than real time (and ignored its real fps).
                    if (firstTimestampUs.get() < 0) {
                        firstTimestampUs.set(timestampUs);
                        playbackStartNs.set(System.nanoTime());
                    }

                    long targetElapsedMs = (timestampUs - firstTimestampUs.get()) / 1_000L;
                    long actualElapsedMs = (System.nanoTime() - playbackStartNs.get()) / 1_000_000L;
                    long sleepMs = targetElapsedMs - actualElapsedMs;

                    if (sleepMs > 0) {
                        try {
                            Thread.sleep(sleepMs);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return false;
                        }
                    } else if (-sleepMs > worstLagMs.get()) {
                        // Processing is slower than real time; we cannot speed the
                        // video up, only record how far behind it got.
                        worstLagMs.set(-sleepMs);
                    }
                    return true;
                });
            } finally {
                backgroundExporter.endRun();
                Log.d(TAG, "total candidates: " + allCandidates.size()
                        + ", worst playback lag: " + worstLagMs.get() + "ms");
                pipeline.release();
                displayMat.release();
                // Clear the interrupt flag so the next task on this single-thread
                // executor doesn't inherit it.
                Thread.interrupted();
            }
        });
    }

    private static final int BLUR_SIZE = 5;
    private void displayProcessedPhoto(Uri photoUri) {
        stopVideo();
        stopCamera();

        NetDetector netDetector = new NetDetector();

        processingExecutor.execute(() -> {
            Bitmap bmp = null;
            Mat background = null;

            try {
                // 1. Decode photo to a software-backed Bitmap
                ImageDecoder.Source src = ImageDecoder.createSource(getContentResolver(), photoUri);
                bmp = ImageDecoder.decodeBitmap(src, (decoder, info, s) -> {
                    decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                });

                // 2. Convert Bitmap to OpenCV Mat (RGBA)
                background = new Mat();
                Utils.bitmapToMat(bmp, background);
                Imgproc.GaussianBlur(background, background, new Size(BLUR_SIZE,BLUR_SIZE), 0);

                // 3. Run detection pipeline (returns NetResult with full-frame coordinates)
                NetDetector.NetResult result = netDetector.detect(background, NetDetector.ColourSpace.RGBA);

                if (result != null && result.box != null) {
                    // Line thickness scaled to image resolution so it is clearly visible on high-res photos
                    int thickness = Math.max(3, (int) Math.round(background.rows() * 0.005));

                    // Draw bounding box directly onto the full-resolution RGBA background
                    Imgproc.rectangle(
                            background,
                            result.box.tl(),
                            result.box.br(),
                            new Scalar(0, 255, 0, 255), // Bright green
                            thickness
                    );

                    Log.d(TAG, String.format("Net detected: %dx%d at (%d, %d)",
                            result.box.width, result.box.height, result.box.x, result.box.y));
                } else {
                    Log.w(TAG, "No net detected in photo");
                }

                // 4. Render the annotated frame to Bitmap and update UI
                Bitmap displayBitmap = Bitmap.createBitmap(
                        background.cols(),
                        background.rows(),
                        Bitmap.Config.ARGB_8888
                );
                Utils.matToBitmap(background, displayBitmap);
                runOnUiThread(() -> frameImageView.setImageBitmap(displayBitmap));

            } catch (Exception e) {
                Log.e(TAG, "Pipeline failed", e);
            } finally {
                // 5. Clean up memory
                if (background != null) background.release();
                if (bmp != null) bmp.recycle();
            }
        });
    }

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    Log.d(TAG, "camera permission granted");
                    startCamera();
                } else {
                    Log.w(TAG, "camera permission denied");
                }
            });

    private void stopVideo() {
        AtomicBoolean flag = activeVideoCancelFlag;
        Future<?> task = activeVideoTask;

        if (flag != null) {
            flag.set(true);   // the frame listener checks this and returns false
        }
        if (task != null && !task.isDone()) {
            Log.d(TAG, "Stopping active video playback...");
            task.cancel(true); // also interrupt, in case it is asleep between frames
        }

        activeVideoCancelFlag = null;
        activeVideoTask = null;
    }

    private void startCamera() {


        ListenableFuture<ProcessCameraProvider> providerFuture =
                ProcessCameraProvider.getInstance(this);

        providerFuture.addListener(() -> {
            try {
                cameraProvider = providerFuture.get();

                ResolutionSelector resolution = new ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(new ResolutionStrategy(
                                new android.util.Size(1280, 720),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER))
                        .build();

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setResolutionSelector(resolution)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build();

                analysis.setAnalyzer(cameraExecutor, this::onCameraFrame);

                cameraFrameCounter.set(0);
                cameraProvider.unbindAll();

                if (cameraPipeline != null) cameraPipeline.release();
                cameraPipeline = new FramePipeline();
                cameraFirstTimestampNs = -1L;

                cameraProvider.bindToLifecycle(
                        this, CameraSelector.DEFAULT_BACK_CAMERA, analysis);

                Log.d(TAG, "camera bound");
            } catch (Exception e) {
                Log.e(TAG, "camera bind failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private volatile FramePipeline cameraPipeline = null;
    private long cameraFirstTimestampNs = -1L;

    @OptIn(markerClass = ExperimentalGetImage.class)
    private void onCameraFrame(ImageProxy proxy) {
        try {
            FramePipeline pipeline = cameraPipeline;
            if (pipeline == null) {
                return;   // a frame still in flight after stopCamera()
            }

            Image image = proxy.getImage();
            if (image == null) {
                return;   // nothing to convert
            }

            Mat yuv = cameraYuvBuilder.build(image);
            Imgproc.resize(yuv, yuv, PIPELINE_SIZE, 0, 0, Imgproc.INTER_AREA);

            // getTimestamp() is nanoseconds from an arbitrary epoch, so rebase it:
            // INITIAL_NET_DELAY_US only means "ten seconds in" if zero is the start.
            long tsNs = proxy.getImageInfo().getTimestamp();
            if (cameraFirstTimestampNs < 0) {
                cameraFirstTimestampNs = tsNs;
            }
            long timestampUs = (tsNs - cameraFirstTimestampNs) / 1_000L;

            long t0 = System.nanoTime();
            FramePipeline.Result r = pipeline.process(yuv, timestampUs, cameraRgb, false);
            long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;

            int i = cameraFrameCounter.getAndIncrement();
            if (i % 30 == 0) {
                Log.d(TAG, "frame " + i + " " + elapsedMs + "ms"
                        + " cands=" + r.candidates.size()
                        + " net=" + r.netBox);
            }

            // With the activity locked to landscape this should always be 0 and the
            // switch should fall straight through. Kept as a guard, not as a feature.
            Mat display;
            switch (proxy.getImageInfo().getRotationDegrees()) {
                case 90:
                    Core.rotate(cameraRgb, cameraRotated, Core.ROTATE_90_CLOCKWISE);
                    display = cameraRotated;
                    break;
                case 180:
                    Core.rotate(cameraRgb, cameraRotated, Core.ROTATE_180);
                    display = cameraRotated;
                    break;
                case 270:
                    Core.rotate(cameraRgb, cameraRotated, Core.ROTATE_90_COUNTERCLOCKWISE);
                    display = cameraRotated;
                    break;
                default:
                    display = cameraRgb;
                    break;
            }

            // Dimensions come from the rotated Mat: 90 and 270 swap them.
            Bitmap bitmap = Bitmap.createBitmap(
                    display.cols(), display.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(display, bitmap);
            runOnUiThread(() -> frameImageView.setImageBitmap(bitmap));

        } catch (Exception e) {
            Log.e(TAG, "camera frame failed", e);
        } finally {
            proxy.close();
        }
    }

    private void stopCamera() {
        if (cameraProvider != null) {
            runOnUiThread(() -> cameraProvider.unbindAll());
        }
        FramePipeline p = cameraPipeline;
        cameraPipeline = null;
        if (p != null) {
            cameraExecutor.execute(p::release);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopVideo();
        stopCamera();
        processingExecutor.shutdownNow();
        frameExtractor.release();
        cameraYuvBuilder.release();
        cameraRgb.release();
        cameraRotated.release();
        cameraExecutor.shutdownNow();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        frameExtractor = new FrameExtractor(this);
        backgroundExporter = new BackgroundExporter(this);
        frameImageView = findViewById(R.id.frame_image_view);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Button selectPhotoButton = findViewById(R.id.select_photo_button);
        selectPhotoButton.setOnClickListener(v -> {
            stopVideo();
            stopCamera();
            photoPickerLauncher.launch(
                    new PickVisualMediaRequest.Builder()
                            .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                            .build());
        });

        Button selectVideoButton = findViewById(R.id.select_video_button);
        selectVideoButton.setOnClickListener(v -> {
            stopVideo();
            stopCamera();
            videoPickerLauncher.launch(new String[]{"video/*"});
        });

        Button stopButton = findViewById(R.id.stop_video_button);
        stopButton.setOnClickListener(v ->{
            stopVideo();
            stopCamera();
        });

        Button exportButton = findViewById(R.id.export_background_button);
        exportButton.setOnClickListener(v -> {
            if (backgroundExporter.isRunActive()) {
                // Consumed on the processing thread at the next net update, so
                // the plate is read on the thread that owns the subtractor.
                backgroundExportRequested.set(true);
                Log.d(TAG, "background export requested");
            } else {
                Log.w(TAG, "no video running, nothing to export");
            }
        });

        Button openCameraButton = findViewById(R.id.open_camera_button);
        openCameraButton.setOnClickListener(v -> {
            stopVideo();
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
            }
        });
    }
}