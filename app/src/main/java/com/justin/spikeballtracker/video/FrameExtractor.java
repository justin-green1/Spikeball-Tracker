package com.justin.spikeballtracker.video;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.util.List;

public class FrameExtractor {
    private final Context context;
    private static final String TAG = "FrameExtractor";
    public interface FrameListener {
        boolean onFrame(Mat frame, long timestampMs);
    }


    public void streamFrames(Uri videoUri, FrameListener listener){
        Mat matFrame = new Mat();

        try(MediaMetadataRetriever retriever = new MediaMetadataRetriever()){
            retriever.setDataSource(context, videoUri);

            String countString = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT);
            String durationString = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            if(countString == null || durationString == null){
                Log.d(TAG, "File selected does not have frames");
                return;
            }
            long durationMs = Integer.parseInt(durationString);
            int frameCount = Integer.parseInt(countString);

            for(int i = 0; i < frameCount; i++) {
                Bitmap bitmapFrame = retriever.getFrameAtIndex(i);
                if(bitmapFrame != null){
                    Utils.bitmapToMat(bitmapFrame, matFrame);
                    long timestampMs = (long) (i * (durationMs / (double) frameCount));
                    if(!listener.onFrame(matFrame, timestampMs)){
                        break;
                    }
                }
            }
        }
        catch (IOException e) {
            Log.e(TAG, "There was an error", e);
            return;
        }
        finally {
            matFrame.release();
        }
    }

    public FrameExtractor(Context context) {
        this.context = context.getApplicationContext();
    }

    /**
     * Extracts a sequence of consecutive frames from a video file.
     *
     * @param videoUri   URI of the video, obtained from the system file picker
     * @param frameCount number of consecutive frames to extract, starting at index 0
     * @return list of decoded frames, or null if the video could not be read
     */
    public List<Bitmap> extractFrames(Uri videoUri, int frameCount){
        try(MediaMetadataRetriever retriever = new MediaMetadataRetriever()){
            retriever.setDataSource(context, videoUri);
            return retriever.getFramesAtIndex(0, frameCount);
        }
        catch (IOException e) {
            Log.e(TAG, "There was an error", e);
            return null;
        }
    }
}
