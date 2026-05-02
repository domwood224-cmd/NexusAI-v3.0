package com.domwood.nexusai.views;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import java.util.Random;

public class MatrixRainView extends View {
    private Paint paint = new Paint();
    private Random random = new Random();
    private float[] drops;
    private String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*";

    public MatrixRainView(Context context) { super(context); init(); }
    public MatrixRainView(Context context, AttributeSet attrs) { super(context, attrs); init(); }
    public MatrixRainView(Context context, AttributeSet attrs, int defStyleAttr) { super(context, attrs, defStyleAttr); init(); }

    private void init() {
        paint.setTextSize(14);
        paint.setColor(Color.parseColor("#00FF41"));
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        int cols = Math.max(1, w / 14);
        drops = new float[cols];
        for (int i = 0; i < cols; i++) drops[i] = random.nextFloat() * -100;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        for (int i = 0; i < drops.length; i++) {
            char c = chars.charAt(random.nextInt(chars.length()));
            float x = i * 14;
            float y = drops[i] * 14;
            paint.setAlpha(random.nextInt(60) + 10);
            canvas.drawText(String.valueOf(c), x, y, paint);
            if (y > getHeight() && random.nextFloat() > 0.98f) drops[i] = 0;
            drops[i] += 0.5f;
        }
        postInvalidateDelayed(50);
    }
}