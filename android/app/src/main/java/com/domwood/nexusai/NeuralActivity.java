package com.domwood.nexusai;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.LinearLayout;

import androidx.appcompat.app.AppCompatActivity;

import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class NeuralActivity extends AppCompatActivity {
    private static final String TAG = "NexusAI.Neural";
    private Random random = new Random();
    private Timer statsTimer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            setContentView(R.layout.activity_neural);
        } catch (Exception e) {
            Log.e(TAG, "Failed to inflate neural layout", e);
            finish();
            return;
        }

        // Back button
        View backArea = findViewById(R.id.neuralBackBtn);
        if (backArea != null) {
            backArea.setOnClickListener(v -> finish());
        }

        startStatsUpdater();
    }

    private void startStatsUpdater() {
        if (statsTimer != null) {
            try { statsTimer.cancel(); } catch (Exception ignored) {}
        }
        statsTimer = new Timer();
        statsTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                runOnUiThread(() -> {
                    try {
                        int nodes = 8 + random.nextInt(20);
                        int links = nodes * 2 + random.nextInt(nodes);
                        int layers = 3 + random.nextInt(5);
                        int activity = 30 + random.nextInt(70);

                        TextView nodeView = findViewById(R.id.neuralNodeCount);
                        TextView linkView = findViewById(R.id.neuralLinkCount);
                        TextView layerView = findViewById(R.id.neuralLayerCount);
                        ProgressBar bar = findViewById(R.id.activityBar);

                        if (nodeView != null) nodeView.setText(String.valueOf(nodes));
                        if (linkView != null) linkView.setText(String.valueOf(links));
                        if (layerView != null) layerView.setText(String.valueOf(layers));
                        if (bar != null) bar.setProgress(activity);
                    } catch (Exception ignored) {}
                });
            }
        }, 0, 2000);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (statsTimer != null) {
            try { statsTimer.cancel(); } catch (Exception ignored) {}
            statsTimer = null;
        }
    }
}
