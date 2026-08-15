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
import com.justin.spikeballtracker.vision.MotionDetector;

import org.opencv.android.OpenCVLoader;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int FRAME_COUNT = 90;
    private FrameExtractor frameExtractor;
    private final ExecutorService processingExecutor = Executors.newSingleThreadExecutor();

    static {
        boolean openCvLoaded = OpenCVLoader.initLocal();
        if (openCvLoaded) {
            Log.d(TAG, "OpenCV library correctly loaded");
        } else {
            Log.e(TAG, "OpenCV library failed to load");
        }
    }

    private final ActivityResultLauncher<String[]> videoPickerLauncher =
            registerForActivityResult(new ActivityResultContracts.OpenDocument(), uri -> {
                Log.d(TAG, "picker returned: " + uri);
                if (uri == null) {
                    Log.w(TAG, "No video selected");
                    return;
                }
                displayProcessedVideo(uri);
            });

    private void displayProcessedVideo(Uri videoUri) {
        ImageView frameImageView = findViewById(R.id.frame_image_view);
        processingExecutor.execute(() -> {
            List<Bitmap> frames;
            MotionDetector motionDetector = new MotionDetector();

            frames = frameExtractor.extractFrames(videoUri, FRAME_COUNT);

            if (frames == null) {
                return;
            }
            try {
                for (Bitmap frame : frames) {
                    Bitmap processed = motionDetector.processFrame(frame);
                    runOnUiThread(() -> frameImageView.setImageBitmap(processed));
                    try {
                        Thread.sleep(33);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
            } finally {
                motionDetector.release();
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