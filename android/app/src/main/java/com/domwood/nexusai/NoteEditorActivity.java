package com.domwood.nexusai;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONArray;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class NoteEditorActivity extends AppCompatActivity {
    private static final String TAG = "NexusAI.NoteEditor";
    private int noteId = -1;
    private SharedPreferences notePrefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            setContentView(R.layout.activity_note_editor);
        } catch (Exception e) {
            Log.e(TAG, "Failed to inflate editor layout", e);
            finish();
            return;
        }

        notePrefs = getSharedPreferences("nexusai_notes", Context.MODE_PRIVATE);
        noteId = getIntent().getIntExtra("id", -1);

        EditText titleEdit = findViewById(R.id.editorTitle);
        EditText contentEdit = findViewById(R.id.editorContent);

        if (noteId >= 0) loadNote(noteId, titleEdit, contentEdit);

        android.view.View backBtn = findViewById(R.id.editorBackBtn);
        if (backBtn != null) backBtn.setOnClickListener(v -> finish());

        android.view.View saveBtn = findViewById(R.id.editorSaveBtn);
        if (saveBtn != null) saveBtn.setOnClickListener(v -> saveNote(titleEdit, contentEdit));

        android.view.View deleteBtn = findViewById(R.id.editorDeleteBtn);
        if (deleteBtn != null) {
            deleteBtn.setOnClickListener(v -> {
                if (noteId < 0) return;
                new AlertDialog.Builder(this)
                    .setTitle("PURGE RECORD")
                    .setMessage("This action cannot be reversed. Proceed?")
                    .setPositiveButton("PURGE", (d, w) -> {
                        deleteNote();
                        Toast.makeText(this, "[RECORD PURGED]", Toast.LENGTH_SHORT).show();
                        finish();
                    })
                    .setNegativeButton("ABORT", null)
                    .show();
            });
        }
    }

    private void loadNote(int id, EditText titleEdit, EditText contentEdit) {
        try {
            JSONArray arr = new JSONArray(notePrefs.getString("notes", "[]"));
            for (int i = 0; i < arr.length(); i++) {
                JSONObject obj = arr.getJSONObject(i);
                if (obj.optInt("id", -1) == id) {
                    titleEdit.setText(obj.optString("title", ""));
                    contentEdit.setText(obj.optString("content", ""));
                    break;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to load note " + id, e);
        }
    }

    private void saveNote(EditText titleEdit, EditText contentEdit) {
        try {
            JSONArray arr = new JSONArray(notePrefs.getString("notes", "[]"));
            String title = titleEdit.getText().toString().trim();
            String content = contentEdit.getText().toString();
            String date = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(new Date());

            if (title.isEmpty() && content.isEmpty()) {
                Toast.makeText(this, "[EMPTY RECORD - NOT SAVED]", Toast.LENGTH_SHORT).show();
                return;
            }

            if (noteId >= 0) {
                for (int i = 0; i < arr.length(); i++) {
                    JSONObject obj = arr.getJSONObject(i);
                    if (obj.optInt("id", -1) == noteId) {
                        obj.put("title", title);
                        obj.put("content", content);
                        obj.put("date", date);
                        break;
                    }
                }
            } else {
                int maxId = 0;
                for (int i = 0; i < arr.length(); i++) {
                    maxId = Math.max(maxId, arr.getJSONObject(i).optInt("id", 0));
                }
                JSONObject obj = new JSONObject();
                obj.put("id", maxId + 1);
                obj.put("title", title);
                obj.put("content", content);
                obj.put("date", date);
                arr.put(obj);
            }
            notePrefs.edit().putString("notes", arr.toString()).apply();
            Toast.makeText(this, "[RECORD COMMITTED]", Toast.LENGTH_SHORT).show();
            finish();
        } catch (Exception e) {
            Log.e(TAG, "Failed to save note", e);
            Toast.makeText(this, "[SAVE FAILED]", Toast.LENGTH_SHORT).show();
        }
    }

    private void deleteNote() {
        try {
            JSONArray arr = new JSONArray(notePrefs.getString("notes", "[]"));
            JSONArray newArr = new JSONArray();
            for (int i = 0; i < arr.length(); i++) {
                if (arr.getJSONObject(i).optInt("id", -1) != noteId) {
                    newArr.put(arr.getJSONObject(i));
                }
            }
            notePrefs.edit().putString("notes", newArr.toString()).apply();
        } catch (Exception e) {
            Log.w(TAG, "Failed to delete note", e);
        }
    }
}
