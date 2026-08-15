package com.justin.spikeballtracker.video;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;

import java.io.IOException;
import java.util.List;

public class FrameExtractor {
    private final Context context;
    private static final String TAG = "FrameExtractor";

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
            throw new RuntimeException(e);
        }
    }
}
