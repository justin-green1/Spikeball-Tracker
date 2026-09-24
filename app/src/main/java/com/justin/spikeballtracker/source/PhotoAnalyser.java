package com.justin.spikeballtracker.source;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.util.Log;

import com.justin.spikeballtracker.vision.NetDetector;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * One-shot net detection on a picked photo.
 *
 * <p>The odd one out: it runs {@link NetDetector} directly on full-resolution
 * RGBA rather than going through the pipeline's 854x480 YUV, and it pre-blurs
 * where the video path does not. Both asymmetries are deliberate and recorded
 * in netdetector-blur-asymmetry.md — keeping this path in its own class is
 * what stops them being mistaken for the shared behaviour.
 */
public final class PhotoAnalyser {

    private static final String TAG = "PhotoAnalyser";
    private static final int BLUR_SIZE = 5;
    private static final Scalar BOX_COLOUR = new Scalar(0, 255, 0, 255);

    private final Context appContext;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final NetDetector netDetector = new NetDetector();

    public PhotoAnalyser(Context context) {
        this.appContext = context.getApplicationContext();
    }

    public void analyse(Uri photoUri, FrameSink sink) {
        executor.execute(() -> {
            Bitmap bmp = null;
            Mat background = new Mat();

            try {
                // OpenCV cannot read hardware bitmaps, hence the software allocator.
                ImageDecoder.Source src =
                        ImageDecoder.createSource(appContext.getContentResolver(), photoUri);
                bmp = ImageDecoder.decodeBitmap(src, (decoder, info, s) ->
                        decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE));

                Utils.bitmapToMat(bmp, background);
                Imgproc.GaussianBlur(background, background,
                        new Size(BLUR_SIZE, BLUR_SIZE), 0);

                NetDetector.NetResult result =
                        netDetector.detect(background, NetDetector.ColourSpace.RGBA);

                if (result != null && result.box != null) {
                    // Thickness scaled to the photo's resolution so it stays visible.
                    int thickness = Math.max(3, (int) Math.round(background.rows() * 0.005));
                    Imgproc.rectangle(background, result.box.tl(), result.box.br(),
                            BOX_COLOUR, thickness);
                    Log.d(TAG, String.format(Locale.US, "net detected: %dx%d at (%d, %d)",
                            result.box.width, result.box.height,
                            result.box.x, result.box.y));
                } else {
                    Log.w(TAG, "no net detected in photo");
                }

                sink.onDisplayFrame(background);

            } catch (Exception e) {
                Log.e(TAG, "photo analysis failed", e);
            } finally {
                background.release();
                if (bmp != null) bmp.recycle();
            }
        });
    }

    public void release() {
        executor.shutdownNow();
    }
}