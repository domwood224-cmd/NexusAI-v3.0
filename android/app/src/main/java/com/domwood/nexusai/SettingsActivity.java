package com.domwood.nexusai;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class SettingsActivity extends AppCompatActivity {
    private static final String TAG = "NexusAI.Settings";
    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            setContentView(R.layout.activity_settings);
        } catch (Exception e) {
            Log.e(TAG, "Failed to inflate settings layout", e);
            finish();
            return;
        }

        // Back button
        android.view.View backBtn = findViewById(R.id.settingsBackBtn);
        if (backBtn != null) backBtn.setOnClickListener(v -> finish());

        prefs = getSharedPreferences("nexusai_settings", Context.MODE_PRIVATE);

        EditText apiUrl = findViewById(R.id.settingsApiUrl);
        EditText apiKey = findViewById(R.id.settingsApiKey);
        EditText model = findViewById(R.id.settingsModel);
        EditText sysPrompt = findViewById(R.id.settingsSystemPrompt);

        if (apiUrl != null) apiUrl.setText(prefs.getString("api_url", ""));
        if (apiKey != null) apiKey.setText(prefs.getString("api_key", ""));
        if (model != null) model.setText(prefs.getString("model", "gpt-3.5-turbo"));
        if (sysPrompt != null) sysPrompt.setText(prefs.getString("system_prompt",
            "You are NexusAI, a biohazard analysis neural assistant. Respond in a technical, clinical style."));

        android.view.View saveBtn = findViewById(R.id.settingsSaveBtn);
        if (saveBtn != null) {
            saveBtn.setOnClickListener(v -> {
                String url = apiUrl != null ? apiUrl.getText().toString().trim() : "";
                String key = apiKey != null ? apiKey.getText().toString().trim() : "";
                String mdl = model != null ? model.getText().toString().trim() : "";
                String prompt = sysPrompt != null ? sysPrompt.getText().toString().trim() : "";

                prefs.edit()
                    .putString("api_url", url)
                    .putString("api_key", key)
                    .putString("model", mdl)
                    .putString("system_prompt", prompt)
                    .apply();
                Toast.makeText(this, "[CONFIG SAVED]", Toast.LENGTH_SHORT).show();
            });
        }

        android.view.View clearChatBtn = findViewById(R.id.settingsClearChatBtn);
        if (clearChatBtn != null) {
            clearChatBtn.setOnClickListener(v -> {
                getSharedPreferences("nexusai_chat", Context.MODE_PRIVATE).edit().clear().apply();
                Toast.makeText(this, "[CHAT HISTORY PURGED]", Toast.LENGTH_SHORT).show();
            });
        }

        android.view.View clearNotesBtn = findViewById(R.id.settingsClearNotesBtn);
        if (clearNotesBtn != null) {
            clearNotesBtn.setOnClickListener(v -> {
                getSharedPreferences("nexusai_notes", Context.MODE_PRIVATE).edit().clear().apply();
                Toast.makeText(this, "[ALL LOGS PURGED]", Toast.LENGTH_SHORT).show();
            });
        }

        String info = "> NexusAI v6.5.0 [BUILD-2026]\n" +
            "> Android: " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + ")\n" +
            "> Device: " + Build.MANUFACTURER + " " + Build.MODEL + "\n" +
            "> Kernel: " + System.getProperty("os.version", "N/A") + "\n" +
            "> Processor: " + Build.HARDWARE + "\n" +
            "> Package: com.domwood.nexusai";
        TextView infoView = findViewById(R.id.settingsInfo);
        if (infoView != null) infoView.setText(info);
    }
}
