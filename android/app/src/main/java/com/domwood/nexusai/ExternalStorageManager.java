package com.domwood.nexusai;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import androidx.annotation.Nullable;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class ExternalStorageManager {
    private static final String TAG = "NexusAI.Storage";
    private static final String ROOT_DIR_NAME = "NexusAI";

    @Nullable
    public static File getAppExternalDirectory(Context context) {
        File[] externalDirs = context.getExternalFilesDirs(null);
        File fallback = null;

        if (externalDirs != null) {
            for (File dir : externalDirs) {
                if (dir == null) {
                    continue;
                }

                String state = Environment.getExternalStorageState(dir);
                if (!Environment.MEDIA_MOUNTED.equals(state)) {
                    continue;
                }

                if (Environment.isExternalStorageRemovable(dir)) {
                    return getAppRootDirectory(dir);
                }

                if (fallback == null) {
                    fallback = dir;
                }
            }
        }

        if (fallback != null) {
            return getAppRootDirectory(fallback);
        }

        return null;
    }

    private static File getAppRootDirectory(File externalFilesDir) {
        File root = new File(externalFilesDir, ROOT_DIR_NAME);
        if (!root.exists() && !root.mkdirs()) {
            Log.w(TAG, "Could not create external app storage directory: " + root.getAbsolutePath());
        }
        return root;
    }

    public static boolean saveTextFile(File directory, String filename, String content) {
        if (directory == null || !directory.exists()) {
            Log.w(TAG, "External directory is not available for saving.");
            return false;
        }

        File file = new File(directory, filename);
        try (FileOutputStream output = new FileOutputStream(file)) {
            output.write(content.getBytes());
            output.flush();
            Log.i(TAG, "Saved file to " + file.getAbsolutePath());
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Failed to save file to external storage.", e);
            return false;
        }
    }
}
