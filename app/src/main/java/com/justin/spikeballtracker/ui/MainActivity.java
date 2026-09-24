package com.justin.spikeballtracker.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.justin.spikeballtracker.R;
import com.justin.spikeballtracker.source.CameraSource;
import com.justin.spikeballtracker.source.PhotoAnalyser;
import com.justin.spikeballtracker.source.VideoPlayer;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

/**
 * Wiring only: three frame sources, one ImageView, and the buttons that switch
 * between them. Anything that touches pixels lives in a source.
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    static {
        if (OpenCVLoader.initLocal()) {
            Log.d(TAG, "OpenCV library correctly loaded");
        } else {
            Log.e(TAG, "OpenCV library failed to load");
        }
    }

    private ImageView frameImageView;

    private VideoPlayer videoPlayer;
    private PhotoAnalyser photoAnalyser;
    private CameraSource cameraSource;

    // Activity result contracts must be registered as field initialisers;
    // registering after the activity is STARTED throws.

    private final ActivityResultLauncher<String[]> videoPickerLauncher =
            registerForActivityResult(new ActivityResultContracts.OpenDocument(), uri -> {
                if (uri == null) {
                    Log.w(TAG, "no video selected");
                    return;
                }
                videoPlayer.play(uri, this::showFrame);
            });

    private final ActivityResultLauncher<PickVisualMediaRequest> photoPickerLauncher =
            registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
                if (uri == null) {
                    Log.w(TAG, "no photo selected");
                    return;
                }
                photoAnalyser.analyse(uri, this::showFrame);
            });

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    cameraSource.start(this, this::showFrame);
                } else {
                    Log.w(TAG, "camera permission denied");
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        frameImageView = findViewById(R.id.frame_image_view);

        videoPlayer = new VideoPlayer(this);
        photoAnalyser = new PhotoAnalyser(this);
        cameraSource = new CameraSource(this);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Button selectVideoButton = findViewById(R.id.select_video_button);
        selectVideoButton.setOnClickListener(v -> {
            stopAllSources();
            videoPickerLauncher.launch(new String[]{"video/*"});
        });

        Button selectPhotoButton = findViewById(R.id.select_photo_button);
        selectPhotoButton.setOnClickListener(v -> {
            stopAllSources();
            photoPickerLauncher.launch(
                    new PickVisualMediaRequest.Builder()
                            .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                            .build());
        });

        Button openCameraButton = findViewById(R.id.open_camera_button);
        openCameraButton.setOnClickListener(v -> {
            stopAllSources();
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
                cameraSource.start(this, this::showFrame);
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
            }
        });

        Button stopButton = findViewById(R.id.stop_video_button);
        stopButton.setOnClickListener(v -> stopAllSources());

        Button exportButton = findViewById(R.id.export_background_button);
        exportButton.setOnClickListener(v -> {
            if (videoPlayer.requestBackgroundExport()) {
                Log.d(TAG, "background export requested");
            } else {
                Log.w(TAG, "no video running, nothing to export");
            }
        });

        Button clearButton = findViewById(R.id.clear_background_button);
        clearButton.setOnClickListener(v -> {
            if (cameraSource.requestBackgroundReset()) {
                Log.d(TAG, "background reset requested");
            } else {
                Log.w(TAG, "camera not running, nothing to reset");
            }
        });
    }

    /**
     * The one sink. Called on a source's worker thread, so the Bitmap is built
     * there and only the assignment is posted to the UI thread.
     *
     * <p>A fresh Bitmap per frame is deliberate: it crosses threads, and
     * reusing one while the ImageView is mid-draw tears.
     */
    private void showFrame(Mat displayMat) {
        Bitmap bitmap = Bitmap.createBitmap(
                displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(displayMat, bitmap);
        runOnUiThread(() -> frameImageView.setImageBitmap(bitmap));
    }

    private void stopAllSources() {
        videoPlayer.stop();
        cameraSource.stop();
        // PhotoAnalyser is one-shot; there is nothing to interrupt.
    }

    @Override
    protected void onDestroy() {
        stopAllSources();
        videoPlayer.release();
        photoAnalyser.release();
        cameraSource.release();
        super.onDestroy();
    }
}