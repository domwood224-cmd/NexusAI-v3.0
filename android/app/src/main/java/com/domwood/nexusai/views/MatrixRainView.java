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
    private float[] drops;
    private final String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*";
    private boolean running = true;
    private final Runnable invalidator = new Runnable() {
        @Override
        public void run() {
            if (running && isAttachedToWindow()) {
                invalidate();
                postDelayed(this, 60);
            }
        }
    };

    public MatrixRainView(Context context) { super(context); init(); }
    public MatrixRainView(Context context, AttributeSet attrs) { super(context, attrs); init(); }
    public MatrixRainView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr); init();
    }

    private void init() {
        paint.setTextSize(16);
        paint.setColor(Color.parseColor("#00FF41"));
        paint.setAntiAlias(false);
        setLayerType(LAYER_TYPE_SOFTWARE, null);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        if (w > 0) {
            int cols = Math.max(1, w / 16);
            drops = new float[cols];
            for (int i = 0; i < cols; i++) drops[i] = random.nextFloat() * -80;
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (drops == null || drops.length == 0) return;

        for (int i = 0; i < drops.length; i++) {
            char c = chars.charAt(random.nextInt(chars.length()));
            float x = i * 16;
            float y = drops[i] * 16;
            paint.setAlpha(random.nextInt(70) + 20);
            canvas.drawText(String.valueOf(c), x, y, paint);
            if (y > getHeight() && random.nextFloat() > 0.97f) {
                drops[i] = 0;
            }
            drops[i] += 0.6f;
        }
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        running = true;
        post(invalidator);
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        running = false;
        removeCallbacks(invalidator);
    }
}
