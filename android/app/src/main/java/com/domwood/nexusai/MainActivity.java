package com.domwood.nexusai;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "NexusAI.Main";
    private TextView systemInfo;
    private Timer clockTimer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            setContentView(R.layout.activity_main);
        } catch (Exception e) {
            Log.e(TAG, "Failed to inflate main layout", e);
            Toast.makeText(this, "Layout error: " + e.getMessage(), Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        // Check for previous crash
        try {
            SharedPreferences crashPrefs = getSharedPreferences("nexusai_crash", MODE_PRIVATE);
            String lastCrash = crashPrefs.getString("last_crash", "");
            if (lastCrash != null && !lastCrash.isEmpty()) {
                Log.w(TAG, "Previous crash detected:\n" + lastCrash);
                // Show first line of crash info
                String firstLine = lastCrash.split("\n")[0];
                Toast.makeText(this, "Previous crash: " + firstLine, Toast.LENGTH_LONG).show();
                crashPrefs.edit().remove("last_crash").apply();
            }
        } catch (Exception ignored) {}

        // Initialize storage (non-critical)
        try {
            File storageDir = ExternalStorageManager.getAppExternalDirectory(this);
            if (storageDir != null) {
                ExternalStorageManager.saveTextFile(storageDir, "nexusai_app_data.txt",
                    "NexusAI v6.5 initialized on "
                    + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date()) + "\n");
            }
        } catch (Exception e) {
            Log.w(TAG, "Storage init failed (non-critical)", e);
        }

        setupCards();
        updateSystemInfo();
        startClock();
    }

    private void setupCards() {
        try {
            android.view.View chatCard = findViewById(R.id.cardChat);
            if (chatCard != null) {
                chatCard.setOnClickListener(v ->
                    startActivity(new Intent(this, ChatActivity.class)));
            }
            android.view.View neuralCard = findViewById(R.id.cardNeural);
            if (neuralCard != null) {
                neuralCard.setOnClickListener(v ->
                    startActivity(new Intent(this, NeuralActivity.class)));
            }
            android.view.View notesCard = findViewById(R.id.cardNotes);
            if (notesCard != null) {
                notesCard.setOnClickListener(v ->
                    startActivity(new Intent(this, NotesActivity.class)));
            }
            android.view.View settingsCard = findViewById(R.id.cardSettings);
            if (settingsCard != null) {
                settingsCard.setOnClickListener(v ->
                    startActivity(new Intent(this, SettingsActivity.class)));
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to setup card listeners", e);
        }
    }

    private void updateSystemInfo() {
        try {
            systemInfo = findViewById(R.id.systemInfo);
            if (systemInfo != null) {
                systemInfo.setText(
                    "> Android: " + Build.VERSION.RELEASE
                        + " (API " + Build.VERSION.SDK_INT + ")\n"
                        + "> Device: " + Build.MODEL + "\n"
                        + "> Kernel: " + System.getProperty("os.version", "N/A") + "\n"
                        + "> Build: v6.5.0 [2026-05-02]\n"
                        + "> Status: ALL SYSTEMS NOMINAL"
                );
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to update system info", e);
        }
    }

    private void startClock() {
        try {
            clockTimer = new Timer();
            clockTimer.scheduleAtFixedRate(new TimerTask() {
                @Override
                public void run() {
                    runOnUiThread(() -> {
                        try {
                            String time = new SimpleDateFormat("HH:mm:ss", Locale.US)
                                .format(new Date());
                            TextView status = findViewById(R.id.systemStatus);
                            if (status != null) {
                                status.setText("> SYSTEM ONLINE " + time);
                            }
                        } catch (Exception ignored) {}
                    });
                }
            }, 0, 1000);
        } catch (Exception e) {
            Log.e(TAG, "Failed to start clock", e);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (clockTimer != null) {
            try { clockTimer.cancel(); } catch (Exception ignored) {}
            clockTimer = null;
        }
    }
}
