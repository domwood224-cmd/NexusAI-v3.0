package com.domwood.nexusai;

import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {
    private TextView systemInfo;
    private Timer clockTimer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        File storageDir = ExternalStorageManager.getAppExternalDirectory(this);
        if (storageDir != null) {
            ExternalStorageManager.saveTextFile(storageDir, "nexusai_app_data.txt",
                "NexusAI v3.1 initialized on " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date()) + "\n");
        }

        setupCards();
        systemInfo = findViewById(R.id.systemInfo);
        updateSystemInfo();
        startClock();
    }

    private void setupCards() {
        findViewById(R.id.cardChat).setOnClickListener(v -> startActivity(new Intent(this, ChatActivity.class)));
        findViewById(R.id.cardNeural).setOnClickListener(v -> startActivity(new Intent(this, NeuralActivity.class)));
        findViewById(R.id.cardNotes).setOnClickListener(v -> startActivity(new Intent(this, NotesActivity.class)));
        findViewById(R.id.cardSettings).setOnClickListener(v -> startActivity(new Intent(this, SettingsActivity.class)));
    }

    private void updateSystemInfo() {
        systemInfo.setText(
            "> Android: " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + ")\n" +
            "> Device: " + Build.MODEL + "\n" +
            "> Kernel: " + System.getProperty("os.version", "N/A") + "\n" +
            "> Build: v3.1.0 [2026-05-02]\n" +
            "> Status: ALL SYSTEMS NOMINAL"
        );
    }

    private void startClock() {
        clockTimer = new Timer();
        clockTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                runOnUiThread(() -> {
                    String time = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
                    TextView status = findViewById(R.id.systemStatus);
                    if (status != null) status.setText("> SYSTEM ONLINE " + time);
                });
            }
        }, 0, 1000);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (clockTimer != null) clockTimer.cancel();
    }
}