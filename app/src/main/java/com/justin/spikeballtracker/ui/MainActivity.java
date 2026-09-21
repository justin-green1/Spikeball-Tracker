package com.justin.spikeballtracker.ui;

import android.graphics.Bitmap;

import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.justin.spikeballtracker.R;
import com.justin.spikeballtracker.tracking.BallCandidate;
import com.justin.spikeballtracker.vision.BlobDetector;
import com.justin.spikeballtracker.video.BackgroundExporter;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.vision.ColourDetector;
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

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final Scalar MARKER_COLOUR = new Scalar(255, 0, 0);
    private FrameExtractor frameExtractor;
    private BackgroundExporter backgroundExporter;

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

        ImageView frameImageView = findViewById(R.id.frame_image_view);

        final AtomicBoolean cancelled = new AtomicBoolean(false);
        activeVideoCancelFlag = cancelled;

        activeVideoTask = processingExecutor.submit(() -> {
            MotionDetector motionDetector = new MotionDetector();
            ColourDetector colourDetector = new ColourDetector();
            BlobDetector blobDetector = new BlobDetector();

            Mat ballMask = new Mat();     // single channel, input to blob detection
            Mat displayMat = new Mat();   // RGB copy of the frame, drawn on

            AtomicInteger frameCounter = new AtomicInteger();
            List<BallCandidate> allCandidates = new ArrayList<>();

            // Wall-clock anchor for playback pacing, set on the first frame.
            AtomicLong firstTimestampUs = new AtomicLong(-1);
            AtomicLong playbackStartNs = new AtomicLong(0);
            AtomicLong worstLagMs = new AtomicLong(0);

            cachedNetBox = null;
            lastNetUpdateUs = -1L;

            try {
                frameExtractor.streamFrames(videoUri, (frame, timestampUs) -> {
                    if (cancelled.get() || Thread.currentThread().isInterrupted()) {
                        return false;   // stop requested: leave the decode loop
                    }

                    int frameIndex = frameCounter.getAndIncrement();

                    if (frameIndex == 0) {
                        // readColourInfo has run by now, so the provenance
                        // recorded is the one actually in force for this clip.
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

                    // Export before anything touches the frame, so what lands on
                    // disk is byte-identical to what the pipeline consumes.
                    backgroundExporter.offerFrame(frame, frameIndex, timestampUs);

                    // 1. feature processing on raw frames
                    Mat motionMask = motionDetector.processFrame(frame);
                    Mat colourMask = colourDetector.processFrame(frame);

                    Core.bitwise_and(colourMask, motionMask, ballMask);

                    List<BallCandidate> candidates =
                            blobDetector.findCandidates(ballMask, frameIndex, timestampUs);
                    allCandidates.addAll(candidates);


                    //2. visual for UI and draw ball candidates
                    Imgproc.cvtColor(frame, displayMat, Imgproc.COLOR_YUV2RGB);
                    for (BallCandidate c : candidates) {
                        Imgproc.circle(displayMat, new Point(c.x, c.y), 12, MARKER_COLOUR, 2);
                    }

                    //3. draw the net
                    updateAndDrawNet(motionDetector, displayMat, timestampUs, frameIndex);

                    //4. render to screen
                    Bitmap bitmap = Bitmap.createBitmap(
                            displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(displayMat, bitmap);
                    runOnUiThread(() -> frameImageView.setImageBitmap(bitmap));

                    // 5. pace playback against the frame's own presentation
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
                motionDetector.release();
                colourDetector.release();
                blobDetector.release();
                ballMask.release();
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

        ImageView frameImageView = findViewById(R.id.frame_image_view);
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

    private NetDetector netDetector = new NetDetector();
    private Rect cachedNetBox = null;
    private long lastNetUpdateUs = -1;
    private static final long INITIAL_NET_DELAY_US = 10_000_000L;
    private static final long NET_RECALIBRATION_INTERVAL_US = 5_000_000L;
    private static final Scalar NET_OVERLAY_COLOUR = new Scalar(0, 255, 255);

    /**
     * Periodically updates the nets bounding box using the background from MotionDetector
     *
     * @param motionDetector motion detector object
     * @param displayMat
     * @param timestampUs frames current time stamp.
     */

    private void updateAndDrawNet(MotionDetector motionDetector, Mat displayMat, long timestampUs, int frameIndex){
        boolean needsInitialLock = (cachedNetBox == null && timestampUs > INITIAL_NET_DELAY_US);
        boolean needsRecalibration = (cachedNetBox != null && timestampUs-lastNetUpdateUs >= NET_RECALIBRATION_INTERVAL_US);
        boolean exportRequested = backgroundExportRequested.getAndSet(false);
        boolean plateDue = backgroundExporter.dueForBackground(timestampUs);

        //Log.v(TAG, "Needs recalibration: " + needsRecalibration + ". Needs initial lock: " + needsInitialLock);

        if(needsRecalibration || needsInitialLock || exportRequested || plateDue){
            Mat backgroundMat = new Mat();
            try{
                motionDetector.getBackgroundImage(backgroundMat);
                if(!backgroundMat.empty()){
                    // Export the plate before detect touches it, so the file is
                    // exactly the Mat NetDetector is about to be handed. At
                    // 854x480 its resizeMat is a no-op copy, so this is also
                    // pixel-for-pixel the image its Canny runs on.
                    // The exporter owns the ten-second interval; this just offers
                    // it the plate whenever one has been fetched.
                    backgroundExporter.offerBackground(backgroundMat, frameIndex, timestampUs);

                    if(exportRequested){
                        backgroundExporter.exportPlate(
                                backgroundMat, "manual", frameIndex, timestampUs);
                    }

                    if(needsRecalibration || needsInitialLock){
                        NetDetector.NetResult result = netDetector.detect(backgroundMat, NetDetector.ColourSpace.YUV);
                        if(result != null && result.box != null){
                            cachedNetBox = result.box;
                            lastNetUpdateUs = timestampUs;
                            Log.d(TAG, (needsInitialLock ? "Initial net locked: " : "Net recalibrated: ") + cachedNetBox);
                        }
                    }
                }
            }catch (Exception e){
                Log.e(TAG, "Failed to update net from background model", e);
            }finally {
                backgroundMat.release();
            }
        }

        if(cachedNetBox != null){
            Imgproc.rectangle(displayMat, cachedNetBox.tl(), cachedNetBox.br(), NET_OVERLAY_COLOUR, 2);
        }
    }

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

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopVideo();
        processingExecutor.shutdownNow();
        frameExtractor.release();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        frameExtractor = new FrameExtractor(this);
        backgroundExporter = new BackgroundExporter(this);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Button selectPhotoButton = findViewById(R.id.select_photo_button);
        selectPhotoButton.setOnClickListener(v -> {
            stopVideo();
            photoPickerLauncher.launch(
                    new PickVisualMediaRequest.Builder()
                            .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                            .build());
        });

        Button selectVideoButton = findViewById(R.id.select_video_button);
        selectVideoButton.setOnClickListener(v -> {
            stopVideo();
            videoPickerLauncher.launch(new String[]{"video/*"});
        });

        Button stopButton = findViewById(R.id.stop_video_button);
        stopButton.setOnClickListener(v -> stopVideo());

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
    }
}