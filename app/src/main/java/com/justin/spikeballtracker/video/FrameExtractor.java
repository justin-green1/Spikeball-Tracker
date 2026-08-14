package com.justin.spikeballtracker.video;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;

import java.io.IOException;
import java.util.List;

public class FrameExtractor {
    private final Context context;

    public FrameExtractor(Context context) {
        this.context = context.getApplicationContext();
    }

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
