package com.domwood.nexusai;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class ExternalStorageManager {
    private static final String TAG = "NexusAI.Storage";
    private static final String ROOT_DIR_NAME = "NexusAI";

    public static File getAppExternalDirectory(Context context) {
        try {
            File externalDir = context.getExternalFilesDir(null);
            if (externalDir != null) {
                File root = new File(externalDir, ROOT_DIR_NAME);
                if (!root.exists() && !root.mkdirs()) {
                    Log.w(TAG, "Could not create external directory: " + root.getAbsolutePath());
                } else {
                    return root;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "External storage unavailable", e);
        }

        try {
            File internalDir = new File(context.getFilesDir(), ROOT_DIR_NAME);
            if (!internalDir.exists() && !internalDir.mkdirs()) {
                Log.e(TAG, "Could not create internal directory: " + internalDir.getAbsolutePath());
                return context.getFilesDir();
            }
            return internalDir;
        } catch (Exception e) {
            Log.e(TAG, "Internal storage failed", e);
            return context.getFilesDir();
        }
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
            output.write(content.getBytes(StandardCharsets.UTF_8));
            output.flush();
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Failed to save file: " + file.getAbsolutePath(), e);
            return false;
        }
    }
}
