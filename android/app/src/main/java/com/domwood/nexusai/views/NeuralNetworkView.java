package com.domwood.nexusai.views;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNetworkView extends View {
    private final Paint nodePaint = new Paint();
    private final Paint activeNodePaint = new Paint();
    private final Paint linePaint = new Paint();
    private final Paint activeLinePaint = new Paint();
    private final Paint pulsePaint = new Paint();
    private final Random random = new Random();
    private final ArrayList<float[]> nodes = new ArrayList<>();
    private final ArrayList<int[]> connections = new ArrayList<>();
    private final int[] layerSizes = {4, 6, 8, 6, 3};
    private float activity = 0.5f;
    private float pulsePhase = 0;
    private boolean running = true;
    private final Runnable invalidator = new Runnable() {
        @Override
        public void run() {
            if (running && isAttachedToWindow()) {
                invalidate();
                postDelayed(this, 50);
            }
        }
    };

    public NeuralNetworkView(Context context) { super(context); init(); }
    public NeuralNetworkView(Context context, AttributeSet attrs) { super(context, attrs); init(); }
    public NeuralNetworkView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr); init();
    }

    private void init() {
        nodePaint.setColor(Color.parseColor("#003300"));
        nodePaint.setAntiAlias(true);
        activeNodePaint.setColor(Color.parseColor("#00FF41"));
        activeNodePaint.setAntiAlias(true);
        linePaint.setColor(Color.parseColor("#002200"));
        linePaint.setAntiAlias(true);
        linePaint.setStrokeWidth(1f);
        activeLinePaint.setColor(Color.parseColor("#00FF41"));
        activeLinePaint.setAntiAlias(true);
        activeLinePaint.setStrokeWidth(1.5f);
        activeLinePaint.setAlpha(150);
        pulsePaint.setColor(Color.parseColor("#00FF41"));
        pulsePaint.setAntiAlias(true);
        pulsePaint.setStyle(Paint.Style.STROKE);
        setLayerType(LAYER_TYPE_SOFTWARE, null);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        if (w > 0 && h > 0) generateNetwork();
    }

    private void generateNetwork() {
        nodes.clear();
        connections.clear();
        int padding = 40;
        int w = getWidth();
        int h = getHeight();
        if (w == 0 || h == 0) return;

        float layerSpacing = (w - padding * 2) / (float) Math.max(1, layerSizes.length - 1);

        for (int l = 0; l < layerSizes.length; l++) {
            int count = layerSizes[l];
            float nodeSpacing = (h - padding * 2) / (float) Math.max(1, count + 1);
            for (int n = 0; n < count; n++) {
                float x = padding + l * layerSpacing;
                float y = padding + nodeSpacing * (n + 1);
                nodes.add(new float[]{x, y, l, n, random.nextFloat()});
            }
        }

        for (int l = 0; l < layerSizes.length - 1; l++) {
            int startIdx = 0;
            for (int p = 0; p < l; p++) startIdx += layerSizes[p];
            int endIdx = startIdx + layerSizes[l];

            for (int i = startIdx; i < endIdx; i++) {
                int numConns = 1 + random.nextInt(3);
                for (int c = 0; c < numConns; c++) {
                    int j = endIdx + random.nextInt(layerSizes[l + 1]);
                    connections.add(new int[]{i, j, random.nextFloat() > 0.5f ? 1 : 0});
                }
            }
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (nodes.isEmpty()) return;

        pulsePhase += 0.02f;
        activity = 0.4f + 0.4f * (float) Math.sin(pulsePhase * 2);

        // Draw connections
        for (int[] conn : connections) {
            try {
                if (conn[2] == 1 || random.nextFloat() < activity) {
                    float[] from = nodes.get(conn[0]);
                    float[] to = nodes.get(conn[1]);
                    Paint p = conn[2] == 1 ? activeLinePaint : linePaint;
                    if (conn[2] != 1) p.setAlpha((int) (30 + 80 * activity));
                    canvas.drawLine(from[0], from[1], to[0], to[1], p);
                }
            } catch (IndexOutOfBoundsException ignored) {}
        }

        // Draw pulses
        if (!connections.isEmpty() && random.nextFloat() < activity * 0.3f) {
            try {
                int[] conn = connections.get(random.nextInt(connections.size()));
                float[] from = nodes.get(conn[0]);
                float[] to = nodes.get(conn[1]);
                float progress = (float) (Math.sin(pulsePhase * 5 + conn[0]) * 0.5 + 0.5);
                float px = from[0] + (to[0] - from[0]) * progress;
                float py = from[1] + (to[1] - from[1]) * progress;
                pulsePaint.setAlpha(200);
                canvas.drawCircle(px, py, 3, pulsePaint);
            } catch (IndexOutOfBoundsException ignored) {}
        }

        // Draw nodes
        for (float[] node : nodes) {
            float activation = (float) (Math.sin(pulsePhase * 3 + node[0] * 0.1 + node[1] * 0.1) * 0.5 + 0.5);
            float radius = 6 + activation * 4;
            boolean active = activation > activity || random.nextFloat() < 0.15f;

            if (active) {
                activeNodePaint.setAlpha((int) (100 + 155 * activation));
                canvas.drawCircle(node[0], node[1], radius + 4, nodePaint);
                canvas.drawCircle(node[0], node[1], radius, activeNodePaint);
            } else {
                nodePaint.setAlpha(60);
                canvas.drawCircle(node[0], node[1], radius, nodePaint);
            }
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
