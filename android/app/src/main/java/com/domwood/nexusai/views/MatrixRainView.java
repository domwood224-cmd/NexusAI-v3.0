package com.domwood.nexusai.views;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.Random;

public class MatrixRainView extends View {
    private final Paint paint = new Paint();
    private final Random random = new Random();
    private int cols = 0;
    private float[] drops;
    private final String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*";
    private boolean running = false;
    private boolean sizeReady = false;
    private final Runnable invalidator = new Runnable() {
        @Override
        public void run() {
            if (running && isAttachedToWindow()) {
                try {
                    invalidate();
                } catch (Exception ignored) {}
                postDelayed(this, 80);
            }
        }
    };

    public MatrixRainView(Context context) {
        super(context);
        init();
    }

    public MatrixRainView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public MatrixRainView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        try {
            paint.setTextSize(16);
            paint.setColor(Color.parseColor("#00FF41"));
            paint.setAntiAlias(false);
            setLayerType(LAYER_TYPE_SOFTWARE, null);
        } catch (Exception e) {
            paint.setTextSize(16);
            paint.setColor(Color.GREEN);
        }
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        try {
            if (w > 0 && h > 0) {
                cols = Math.max(1, w / 16);
                drops = new float[cols];
                for (int i = 0; i < cols; i++) {
                    drops[i] = random.nextFloat() * -80;
                }
                sizeReady = true;
            }
        } catch (Exception e) {
            sizeReady = false;
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        try {
            if (!sizeReady || drops == null || drops.length == 0) return;
            int height = getHeight();
            if (height <= 0) return;

            for (int i = 0; i < drops.length; i++) {
                char c = chars.charAt(random.nextInt(chars.length()));
                float x = i * 16;
                float y = drops[i] * 16;
                paint.setAlpha(random.nextInt(70) + 20);
                canvas.drawText(String.valueOf(c), x, y, paint);
                if (y > height && random.nextFloat() > 0.97f) {
                    drops[i] = 0;
                }
                drops[i] += 0.6f;
            }
        } catch (Exception e) {
            // Silently skip this frame
        }
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        running = true;
        // Delay first post to let layout finish
        postDelayed(invalidator, 200);
    }

    @Override
    protected void onDetachedFromWindow() {
        running = false;
        try { removeCallbacks(invalidator); } catch (Exception ignored) {}
        super.onDetachedFromWindow();
    }
}
