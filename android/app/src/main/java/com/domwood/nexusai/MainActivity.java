package com.domwood.nexusai;

import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "NexusAI";
    private static final String SAVED_FILE_NAME = "nexusai_app_data.txt";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(48, 48, 48, 48);
        layout.setBackgroundColor(Color.parseColor("#121212"));

        ScrollView scroll = new ScrollView(this);
        scroll.addView(layout);
        setContentView(scroll);

        // Title
        TextView title = new TextView(this);
        title.setText("NexusAI v3.0");
        title.setTextSize(28f);
        title.setTextColor(Color.parseColor("#BB86FC"));
        title.setPadding(0, 0, 0, 32);
        layout.addView(title);

        // Subtitle
        TextView subtitle = new TextView(this);
        subtitle.setText("AI-Powered Assistant");
        subtitle.setTextSize(14f);
        subtitle.setTextColor(Color.parseColor("#B0B0B0"));
        subtitle.setPadding(0, 0, 0, 48);
        layout.addView(subtitle);

        // Status message
        TextView status = new TextView(this);
        status.setTextSize(16f);
        status.setTextColor(Color.parseColor("#E0E0E0"));
        layout.addView(status);

        try {
            File storageDir = ExternalStorageManager.getAppExternalDirectory(this);

            if (storageDir == null) {
                status.setText("Storage initialization failed.");
                status.setTextColor(Color.parseColor("#CF6679"));
                Log.e(TAG, "getAppExternalDirectory returned null");
                return;
            }

            String currentTime = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date());
            String content = "NexusAI v3.0 - App data initialized\n"
                    + "Timestamp: " + currentTime + "\n"
                    + "Storage: " + storageDir.getAbsolutePath() + "\n";

            boolean saved = ExternalStorageManager.saveTextFile(storageDir, SAVED_FILE_NAME, content);

            if (saved) {
                StringBuilder sb = new StringBuilder();
                sb.append("✓ App initialized successfully\n\n");
                sb.append("Data saved to:\n");
                sb.append(new File(storageDir, SAVED_FILE_NAME).getAbsolutePath()).append("\n\n");

                // Show stored data
                File dataFile = new File(storageDir, SAVED_FILE_NAME);
                if (dataFile.exists()) {
                    sb.append("─── Data Preview ───\n");
                    sb.append("File size: ").append(dataFile.length()).append(" bytes\n");
                    sb.append("Last modified: ").append(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date(dataFile.lastModified()))).append("\n");
                }

                sb.append("\n─── System Info ───\n");
                sb.append("Android: ").append(android.os.Build.VERSION.RELEASE).append("\n");
                sb.append("Device: ").append(android.os.Build.MODEL).append("\n");
                sb.append("API Level: ").append(android.os.Build.VERSION.SDK_INT);

                status.setText(sb.toString());
                status.setTextColor(Color.parseColor("#4CAF50"));
                Log.i(TAG, "App initialized: " + storageDir.getAbsolutePath());
            } else {
                status.setText("Failed to save app data.\nInternal storage will be used on next launch.");
                status.setTextColor(Color.parseColor("#FFB74D"));
                Log.e(TAG, "saveTextFile returned false");
            }
        } catch (Exception e) {
            status.setText("Error: " + e.getMessage());
            status.setTextColor(Color.parseColor("#CF6679"));
            Log.e(TAG, "onCreate failed", e);
        }
    }
}
