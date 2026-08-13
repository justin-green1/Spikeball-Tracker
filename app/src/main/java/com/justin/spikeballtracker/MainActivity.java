package com.justin.spikeballtracker;

import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

import java.io.IOException;
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
                if(uri == null){
                    Log.w(TAG, "No video selected");
                    return;
                }
                processVideo(uri);
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

    private Bitmap toGrayscale(Bitmap source){
        Mat sourceMat = new Mat();
        Mat grayMat = new Mat();
        try{
            Utils.bitmapToMat(source, sourceMat);
            Imgproc.cvtColor(sourceMat, grayMat, Imgproc.COLOR_RGBA2GRAY);
            Bitmap outputBitmap = Bitmap.createBitmap(source.getWidth(), source.getHeight(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(grayMat, outputBitmap);
            return outputBitmap;
        }finally{
            sourceMat.release();
            grayMat.release();
        }
    }

    private void processVideo(Uri videoUri){
        List<Bitmap> frames;
        int frameCount = 30;
        Mat frameMat = new Mat();
        Mat foregroundMask = new Mat();
        Mat displayMat = new Mat();

        try(MediaMetadataRetriever retriever = new MediaMetadataRetriever()){
            retriever.setDataSource(this, videoUri);
            frames = retriever.getFramesAtIndex(0, frameCount);

            BackgroundSubtractorMOG2 subtractor = Video.createBackgroundSubtractorMOG2();
            ImageView frameImageView = findViewById(R.id.frame_image_view);
            for(Bitmap frame : frames){
                Utils.bitmapToMat(frame, frameMat);
                subtractor.apply(frameMat, foregroundMask);
                Imgproc.cvtColor(foregroundMask, displayMat, Imgproc.COLOR_GRAY2RGBA);

                Bitmap outputBitmap = Bitmap.createBitmap(frame.getWidth(), frame.getHeight(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(displayMat, outputBitmap);


                frameImageView.setImageBitmap(outputBitmap);
            }
        } catch(IllegalArgumentException | IOException | SecurityException e) {
            Log.e(TAG, "Video file not extracted correctly", e);
        } finally{
            frameMat.release();
            foregroundMask.release();
            displayMat.release();
        }
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