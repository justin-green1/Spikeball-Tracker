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

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    static{
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
                if(uri == null){
                    Log.w(TAG, "No video selected");
                    return;
                }
                displayProcessesVideo(uri);
            });

    private void displayProcessesVideo(Uri videoUri){
        List<Bitmap> frames;
        int frameCount = 30;
        MotionDetector motionDetector = new MotionDetector();
        FrameExtractor frameExtractor = new FrameExtractor(this);

        frames = frameExtractor.extractFrames(videoUri, frameCount);
        ImageView frameImageView = findViewById(R.id.frame_image_view);

        if(frames == null){
            return;
        }
        for(Bitmap frame : frames){
            frameImageView.setImageBitmap(motionDetector.processFrame(frame));
        }
        motionDetector.release();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Button selectVideoButton = findViewById(R.id.select_video_button);
        videoPickerLauncher.launch(new String[]{"video/*"});
    }
}