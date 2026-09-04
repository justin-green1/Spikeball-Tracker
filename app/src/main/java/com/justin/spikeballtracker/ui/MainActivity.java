package com.justin.spikeballtracker.ui;

import android.graphics.Bitmap;

import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.justin.spikeballtracker.R;
import com.justin.spikeballtracker.tracking.BallCandidate;
import com.justin.spikeballtracker.vision.BlobDetector;
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.vision.ColourDetector;
import com.justin.spikeballtracker.vision.MotionDetector;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final Scalar MARKER_COLOUR = new Scalar(255, 0, 0);
    private static final int FRAME_DELAY_MS = 33;
    private FrameExtractor frameExtractor;
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

    static {
        boolean openCvLoaded = OpenCVLoader.initLocal();
        if (openCvLoaded) {
            Log.d(TAG, "OpenCV library correctly loaded");
        } else {
            Log.e(TAG, "OpenCV library failed to load");
        }
    }

    private void displayProcessedVideo(Uri videoUri) {
        ImageView frameImageView = findViewById(R.id.frame_image_view);

        processingExecutor.execute(() -> {
            MotionDetector motionDetector = new MotionDetector();
            ColourDetector colourDetector = new ColourDetector();
            BlobDetector blobDetector = new BlobDetector();

            Mat ballMask = new Mat();     // single channel, input to blob detection
            Mat displayMat = new Mat();   // RGB copy of the frame, drawn on

            // A lambda can't mutate a captured local, but it can mutate the object
            // a captured reference points at. Hence AtomicInteger rather than int.
            AtomicInteger frameCounter = new AtomicInteger();
            List<BallCandidate> allCandidates = new ArrayList<>();

            try {
                frameExtractor.streamFrames(videoUri, (frame, timestampUs) -> {
                    int frameIndex = frameCounter.getAndIncrement();

                    Mat motionMask = motionDetector.processFrame(frame);
                    Mat colourMask = colourDetector.processFrame(frame);

                    Core.bitwise_and(colourMask, motionMask, ballMask);

                    List<BallCandidate> candidates =
                            blobDetector.findCandidates(ballMask, frameIndex, timestampUs);
                    allCandidates.addAll(candidates);

                    // Debug view: original footage with a ring round each candidate.
                    // Ring rather than filled dot so you can see whether it's
                    // actually centred on the ball.
                    Imgproc.cvtColor(frame, displayMat, Imgproc.COLOR_YUV2RGB);
                    for (BallCandidate c : candidates) {
                        Imgproc.circle(displayMat, new Point(c.x, c.y), 12, MARKER_COLOUR, 2);
                    }

                    Bitmap bitmap = Bitmap.createBitmap(
                            displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(displayMat, bitmap);
                    runOnUiThread(() -> frameImageView.setImageBitmap(bitmap));

                    try {
                        Thread.sleep(FRAME_DELAY_MS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return false;
                    }
                    return true;
                });
            } finally {
                Log.d(TAG, "total candidates: " + allCandidates.size());
                motionDetector.release();
                colourDetector.release();
                blobDetector.release();
                ballMask.release();
                displayMat.release();
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        processingExecutor.shutdownNow();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        frameExtractor = new FrameExtractor(this);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Button selectVideoButton = findViewById(R.id.select_video_button);
        selectVideoButton.setOnClickListener(v -> videoPickerLauncher.launch(new String[]{"video/*"}));
    }
}