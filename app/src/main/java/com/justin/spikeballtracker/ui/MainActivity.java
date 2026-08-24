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
import com.justin.spikeballtracker.video.FrameExtractor;
import com.justin.spikeballtracker.vision.ColourDetector;
import com.justin.spikeballtracker.vision.MotionDetector;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
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
            Mat foregroundMask = new Mat();
            Mat zeros = new Mat();


            try {
                frameExtractor.streamFrames(videoUri, (frame, timestampMs) -> {
                    // Owned by motionDetector — do not release, and do not hold beyond
                    // this callback; the next processFrame call overwrites it.
                    Mat motionMask = motionDetector.processFrame(frame);
                    Mat colourMask = colourDetector.processFrame(frame);
                    if (zeros.empty()) {
                        zeros.create(motionMask.size(), CvType.CV_8UC1);
                        zeros.setTo(new Scalar(0));
                    }

                    // Allocated per frame rather than reused: the UI thread may still be
                    // drawing the previous bitmap when the next frame is written.
                    Core.merge(Arrays.asList(colourMask, motionMask, zeros), foregroundMask);
                    //Scalar lowerBound = new Scalar(255,  255, 0);
                    //Scalar upperBound = new Scalar(255, 255, 255);
                    //Core.inRange(foregroundMask, lowerBound, upperBound, colourMask);


                    Bitmap maskBitmap = Bitmap.createBitmap(
                            colourMask.cols(),
                            colourMask.rows(),
                            Bitmap.Config.ARGB_8888);
                    //Utils.matToBitmap(colourMask, maskBitmap);
                    Utils.matToBitmap(foregroundMask, maskBitmap);
                    runOnUiThread(() -> frameImageView.setImageBitmap(maskBitmap));

                    try {
                        Thread.sleep(FRAME_DELAY_MS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return false;   // stop streaming
                    }
                    return true;        // keep going
                });
            } finally {
                motionDetector.release();
                colourDetector.release();
                frameExtractor.release();
                foregroundMask.release();
                zeros.release();
            }
        });
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        processingExecutor.shutdown();
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