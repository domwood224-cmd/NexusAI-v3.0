package com.domwood.nexusai;

import android.app.Application;
import android.content.Context;
import android.os.Build;
import android.util.Log;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatDelegate;

public class NexusApp extends Application {
    private static final String TAG = "NexusAI";
    private Thread.UncaughtExceptionHandler defaultHandler;

    @Override
    public void onCreate() {
        super.onCreate();
        installCrashHandler();
        forceDarkMode();
    }

    private void forceDarkMode() {
        try {
            AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES);
        } catch (Exception e) {
            Log.w(TAG, "Dark mode setup failed (non-critical)", e);
        }
    }

    private void installCrashHandler() {
        final Context ctx = this;
        defaultHandler = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
            Log.e(TAG, "=== UNCAUGHT EXCEPTION ===", throwable);
            StringBuilder sb = new StringBuilder();
            sb.append("CRASH: ").append(throwable.getClass().getName()).append("\n");
            sb.append("MSG: ").append(throwable.getMessage()).append("\n");
            for (StackTraceElement ste : throwable.getStackTrace()) {
                sb.append("  at ").append(ste.toString()).append("\n");
            }
            Throwable cause = throwable.getCause();
            while (cause != null) {
                sb.append("CAUSE: ").append(cause.getClass().getName())
                  .append(": ").append(cause.getMessage()).append("\n");
                cause = cause.getCause();
            }
            sb.append("DEVICE: ").append(Build.MANUFACTURER).append(" ")
              .append(Build.MODEL).append("\n");
            sb.append("SDK: ").append(Build.VERSION.SDK_INT).append("\n");
            Log.e(TAG, sb.toString());

            try {
                getSharedPreferences("nexusai_crash", MODE_PRIVATE)
                    .edit().putString("last_crash", sb.toString()).apply();
            } catch (Exception ignored) {}

            try {
                Toast.makeText(ctx,
                    "Error: " + throwable.getClass().getSimpleName()
                    + " - " + throwable.getMessage(),
                    Toast.LENGTH_LONG).show();
            } catch (Exception ignored) {}

            if (defaultHandler != null) {
                defaultHandler.uncaughtException(thread, throwable);
            } else {
                thread.stop();
            }
        });
    }
}
