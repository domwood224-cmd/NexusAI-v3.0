package com.domwood.nexusai;

import android.os.Bundle;
import android.util.Log;
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

        TextView textView = new TextView(this);
        textView.setTextSize(18f);
        textView.setPadding(32, 32, 32, 32);

        File storageDir = ExternalStorageManager.getAppExternalDirectory(this);
        String message;

        if (storageDir == null) {
            message = "External storage unavailable. App cannot save data to SD card.";
            Log.e(TAG, message);
        } else {
            String currentTime = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date());
            String content = "NexusAI app data saved on " + currentTime + "\n";
            boolean saved = ExternalStorageManager.saveTextFile(storageDir, SAVED_FILE_NAME, content);
            if (saved) {
                message = "Saved app data to SD card path:\n" + new File(storageDir, SAVED_FILE_NAME).getAbsolutePath();
                Log.i(TAG, message);
            } else {
                message = "Failed to save app data to SD card.";
                Log.e(TAG, message);
            }
        }

        textView.setText(message);
        setContentView(textView);
    }
}
