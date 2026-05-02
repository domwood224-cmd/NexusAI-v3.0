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
    private int width, height;
    private float[] drops;
    private String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*<>{}[]|/\~`";
    private int fontSize = 14;
    private int numColumns;

    public MatrixRainView(Context context) { super(context); init(); }
    public MatrixRainView(Context context, AttributeSet attrs) { super(context, attrs); init(); }
    public MatrixRainView(Context context, AttributeSet attrs, int defStyleAttr) { super(context, attrs, defStyleAttr); init(); }

    private void init() {
        paint.setAntiAlias(false);
        paint.setTextSize(fontSize);
        paint.setColor(Color.parseColor("#00FF41"));
        paint.setAlpha(40);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        width = w;
        height = h;
        numColumns = Math.max(1, width / fontSize);
        drops = new float[numColumns];
        for (int i = 0; i < numColumns; i++) {
            drops[i] = random.nextFloat() * -100;
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawColor(Color.TRANSPARENT);
        for (int i = 0; i < numColumns; i++) {
            char c = chars.charAt(random.nextInt(chars.length()));
            float x = i * fontSize;
            float y = drops[i] * fontSize;
            paint.setAlpha(random.nextInt(60) + 10);
            canvas.drawText(String.valueOf(c), x, y, paint);
            if (y > height && random.nextFloat() > 0.98f) {
                drops[i] = 0;
            }
            drops[i] += 0.5f;
        }
        postInvalidateDelayed(50);
    }
}