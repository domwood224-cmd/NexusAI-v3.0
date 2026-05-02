package com.domwood.nexusai;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class ExternalStorageManager {
    private static final String TAG = "NexusAI.Storage";
    private static final String ROOT_DIR_NAME = "NexusAI";

    /**
     * Returns the app-scoped external directory.
     * Uses getExternalFilesDir which requires NO permissions on Android 10+.
     * Falls back to internal storage if external is unavailable.
     */
    public static File getAppExternalDirectory(Context context) {
        File externalDir = context.getExternalFilesDir(null);
        if (externalDir != null) {
            File root = new File(externalDir, ROOT_DIR_NAME);
            if (!root.exists() && !root.mkdirs()) {
                Log.w(TAG, "Could not create directory: " + root.getAbsolutePath());
                // Fall through to internal storage
            } else {
                return root;
            }
        }

        // Fallback to internal app storage (always available, no permissions needed)
        File internalDir = new File(context.getFilesDir(), ROOT_DIR_NAME);
        if (!internalDir.exists() && !internalDir.mkdirs()) {
            Log.e(TAG, "Could not create internal directory: " + internalDir.getAbsolutePath());
            return context.getFilesDir();
        }
        return internalDir;
    }

    public static boolean saveTextFile(File directory, String filename, String content) {
        if (directory == null) {
            Log.w(TAG, "Directory is null.");
            return false;
        }

        if (!directory.exists()) {
            if (!directory.mkdirs()) {
                Log.e(TAG, "Failed to create directory: " + directory.getAbsolutePath());
                return false;
            }
        }

        File file = new File(directory, filename);
        try (FileOutputStream output = new FileOutputStream(file)) {
            output.write(content.getBytes());
            output.flush();
            Log.i(TAG, "Saved file to " + file.getAbsolutePath());
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Failed to save file: " + file.getAbsolutePath(), e);
            return false;
        }
    }
}
