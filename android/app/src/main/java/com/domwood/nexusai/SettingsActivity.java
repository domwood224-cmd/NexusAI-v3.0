package com.domwood.nexusai;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.widget.EditText;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class SettingsActivity extends AppCompatActivity {
    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
        getSupportActionBar().hide();

        prefs = getSharedPreferences("nexusai_settings", Context.MODE_PRIVATE);

        EditText apiUrl = findViewById(R.id.settingsApiUrl);
        EditText apiKey = findViewById(R.id.settingsApiKey);
        EditText model = findViewById(R.id.settingsModel);
        EditText sysPrompt = findViewById(R.id.settingsSystemPrompt);

        apiUrl.setText(prefs.getString("api_url", ""));
        apiKey.setText(prefs.getString("api_key", ""));
        model.setText(prefs.getString("model", "gpt-3.5-turbo"));
        sysPrompt.setText(prefs.getString("system_prompt", "You are NexusAI, a biohazard analysis neural assistant. Respond in a technical, clinical style."));

        findViewById(R.id.settingsSaveBtn).setOnClickListener(v -> {
            prefs.edit()
                .putString("api_url", apiUrl.getText().toString().trim())
                .putString("api_key", apiKey.getText().toString().trim())
                .putString("model", model.getText().toString().trim())
                .putString("system_prompt", sysPrompt.getText().toString().trim())
                .apply();
            Toast.makeText(this, "[CONFIG SAVED]", Toast.LENGTH_SHORT).show();
        });

        findViewById(R.id.settingsClearChatBtn).setOnClickListener(v -> {
            getSharedPreferences("nexusai_chat", Context.MODE_PRIVATE).edit().clear().apply();
            Toast.makeText(this, "[CHAT HISTORY PURGED]", Toast.LENGTH_SHORT).show();
        });

        findViewById(R.id.settingsClearNotesBtn).setOnClickListener(v -> {
            getSharedPreferences("nexusai_notes", Context.MODE_PRIVATE).edit().clear().apply();
            Toast.makeText(this, "[ALL LOGS PURGED]", Toast.LENGTH_SHORT).show();
        });

        String info = "> NexusAI v3.1.0 [BUILD-2026]\n" +
            "> Android: " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + ")\n" +
            "> Device: " + Build.MANUFACTURER + " " + Build.MODEL + "\n" +
            "> Kernel: " + System.getProperty("os.version") + "\n" +
            "> Processor: " + Build.HARDWARE + "\n" +
            "> Package: com.domwood.nexusai";
        ((TextView) findViewById(R.id.settingsInfo)).setText(info);
    }
}