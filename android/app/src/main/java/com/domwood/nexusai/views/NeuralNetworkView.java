package com.domwood.nexusai.views;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNetworkView extends View {
    private Paint nodePaint = new Paint();
    private Paint activeNodePaint = new Paint();
    private Paint linePaint = new Paint();
    private Paint activeLinePaint = new Paint();
    private Paint pulsePaint = new Paint();
    private Random random = new Random();
    private ArrayList<float[]> nodes = new ArrayList<>();
    private ArrayList<int[]> connections = new ArrayList<>();
    private int[] layerSizes = {4, 6, 8, 6, 3};
    private float activity = 0.5f;
    private float pulsePhase = 0;

    public NeuralNetworkView(Context context) { super(context); init(); }
    public NeuralNetworkView(Context context, AttributeSet attrs) { super(context, attrs); init(); }

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
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        generateNetwork();
    }

    private void generateNetwork() {
        nodes.clear();
        connections.clear();
        int padding = 40;
        float layerSpacing = (getWidth() - padding * 2) / (float) Math.max(1, layerSizes.length - 1);

        for (int l = 0; l < layerSizes.length; l++) {
            int count = layerSizes[l];
            float nodeSpacing = (getHeight() - padding * 2) / (float) Math.max(1, count + 1);
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
            int nextEndIdx = endIdx + layerSizes[l + 1];

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
        canvas.drawColor(Color.TRANSPARENT);

        pulsePhase += 0.02f;
        activity = 0.4f + 0.4f * (float) Math.sin(pulsePhase * 2);

        // Draw connections
        for (int[] conn : connections) {
            if (conn[2] == 1 || random.nextFloat() < activity) {
                float[] from = nodes.get(conn[0]);
                float[] to = nodes.get(conn[1]);
                Paint p = conn[2] == 1 ? activeLinePaint : linePaint;
                if (conn[2] != 1) {
                    p.setAlpha((int) (30 + 80 * activity));
                }
                canvas.drawLine(from[0], from[1], to[0], to[1], p);
            }
        }

        // Draw pulse along random connections
        if (random.nextFloat() < activity * 0.3f && !connections.isEmpty()) {
            int[] conn = connections.get(random.nextInt(connections.size()));
            float[] from = nodes.get(conn[0]);
            float[] to = nodes.get(conn[1]);
            float progress = (float) (Math.sin(pulsePhase * 5 + conn[0]) * 0.5 + 0.5);
            float px = from[0] + (to[0] - from[0]) * progress;
            float py = from[1] + (to[1] - from[1]) * progress;
            pulsePaint.setAlpha(200);
            canvas.drawCircle(px, py, 3, pulsePaint);
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

        postInvalidateDelayed(40);
    }
}