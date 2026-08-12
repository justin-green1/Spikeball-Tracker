package com.justin.spikeballtracker;

import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.FileUriExposedException;
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

import org.opencv.android.OpenCVLoader;

import java.io.IOException;

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
                if(uri == null){
                    Log.w(TAG, "No video selected");
                    return;
                }
                Log.d(TAG, "Video selected: " + uri);
                Bitmap firstFrame = extractFirstFrame(uri);
                if(firstFrame == null){
                    Log.w(TAG, "Image might be unreadable");
                    return;
                }
                ImageView frameImageView = findViewById(R.id.frame_image_view);
                frameImageView.setImageBitmap(firstFrame);
            });

    private Bitmap extractFirstFrame(Uri videoUri){
        try(MediaMetadataRetriever retriever = new MediaMetadataRetriever()){
            retriever.setDataSource(this, videoUri);
            return retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
        }
        catch(IllegalArgumentException | IOException | SecurityException e) {
            Log.e(TAG, "Image could not be extracted", e);
        }
        return null;
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
        selectVideoButton.setOnClickListener(v ->
                videoPickerLauncher.launch(new String[]{"video/*"}));
    }
}