package com.domwood.nexusai;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class NeuralActivity extends AppCompatActivity {
    private Random random = new Random();
    private Timer statsTimer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_neural);
        getSupportActionBar().hide();

        statsTimer = new Timer();
        Handler handler = new Handler(Looper.getMainLooper());
        statsTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                handler.post(() -> {
                    int nodes = 8 + random.nextInt(20);
                    int links = nodes * 2 + random.nextInt(nodes);
                    int layers = 3 + random.nextInt(5);
                    int activity = 30 + random.nextInt(70);

                    ((TextView) findViewById(R.id.neuralNodeCount)).setText(String.valueOf(nodes));
                    ((TextView) findViewById(R.id.neuralLinkCount)).setText(String.valueOf(links));
                    ((TextView) findViewById(R.id.neuralLayerCount)).setText(String.valueOf(layers));
                    ((ProgressBar) findViewById(R.id.activityBar)).setProgress(activity);
                });
            }
        }, 0, 2000);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (statsTimer != null) statsTimer.cancel();
    }
}