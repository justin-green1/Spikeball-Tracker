package com.justin.spikeballtracker.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.justin.spikeballtracker.R;
import com.justin.spikeballtracker.WorkerThread;
import com.justin.spikeballtracker.record.RallyFootage;
import com.justin.spikeballtracker.record.RallySaver;
import com.justin.spikeballtracker.source.CameraSource;
import com.justin.spikeballtracker.source.VideoPlayer;
import com.justin.spikeballtracker.tracking.RallyDetector;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.Locale;

/**
 * Wiring only: two frame sources, one ImageView, the buttons that switch
 * between them, and the last-rally save. Anything that touches pixels lives
 * in a source.
 *
 * <p>Threads: this UI thread, plus three {@link WorkerThread}s. This activity
 * owns Analysis, which both sources share (only one runs at a time);
 * ReplayBuffer owns Recorder and RallySaver owns Saver. Callbacks (frames,
 * rallies, saves) arrive on those threads and post UI changes back here.
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    static {
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV library failed to load");
        }
    }

    private ImageView frameImageView;
    private Button saveRallyButton;
    private WorkerThread analysis;
    private VideoPlayer videoPlayer;
    private CameraSource cameraSource;
    private RallySaver rallySaver;

    // Activity result contracts must be registered as field initialisers;
    // registering after the activity is STARTED throws.

    private final ActivityResultLauncher<String[]> videoPickerLauncher =
            registerForActivityResult(new ActivityResultContracts.OpenDocument(), uri -> {
                if (uri != null) {
                    videoPlayer.play(uri, this::showFrame);
                }
            });

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    cameraSource.start(this, this::showFrame);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        frameImageView = findViewById(R.id.frame_image_view);

        analysis = new WorkerThread("Analysis");
        videoPlayer = new VideoPlayer(this, analysis);
        cameraSource = new CameraSource(this, analysis);
        rallySaver = new RallySaver(this);
        videoPlayer.setRallyListener(this::onRally);
        cameraSource.setRallyListener(this::onRally);

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

        Button clearButton = findViewById(R.id.clear_background_button);
        clearButton.setOnClickListener(v -> cameraSource.requestBackgroundReset());

        saveRallyButton = findViewById(R.id.save_rally_button);
        saveRallyButton.setEnabled(false);   // greyed out until the first rally
        saveRallyButton.setOnClickListener(v -> saveLastRally());
    }

    /**
     * A source confirmed a rally: keep its footage as the last rally, and
     * say so briefly. Called on Analysis (video) or Recorder (camera, once
     * the footage is copied); everything here just posts onwards.
     */
    private void onRally(RallyDetector.Rally rally, RallyFootage footage) {
        rallySaver.keep(footage);
        double lengthS = (rally.endUs - Math.max(0, rally.startUs)) / 1e6;
        toast(String.format(Locale.US, "Rally detected · %.1f s", lengthS));
        if (footage != null) {
            runOnUiThread(() -> saveRallyButton.setEnabled(true));
        }
    }

    private void saveLastRally() {
        rallySaver.saveLast(new RallySaver.SaveCallback() {
            @Override
            public void onSaved(String displayName, double seconds) {
                toast(String.format(Locale.US,
                        "Saved %.1f s to Movies/SpikeballTracker/%s", seconds, displayName));
            }

            @Override
            public void onFailed(String reason) {
                toast("Rally save failed: " + reason);
            }
        });
    }

    /** Safe from any thread. */
    private void toast(String message) {
        runOnUiThread(() -> {
            if (!isFinishing() && !isDestroyed()) {
                Toast.makeText(this, message, Toast.LENGTH_LONG).show();
            }
        });
    }

    /**
     * The one sink. Called on the Analysis thread, so the Bitmap is built
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
    }

    @Override
    protected void onDestroy() {
        stopAllSources();
        videoPlayer.release();
        cameraSource.release();
        rallySaver.release();
        analysis.quit();   // runs the releases just posted, then stops
        super.onDestroy();
    }
}
