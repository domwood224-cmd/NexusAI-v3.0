package com.domwood.nexusai;

import android.app.AlertDialog;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.EditText;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONArray;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class NoteEditorActivity extends AppCompatActivity {
    private int noteId = -1;
    private SharedPreferences notePrefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_note_editor);
        getSupportActionBar().hide();

        notePrefs = getSharedPreferences("nexusai_notes", Context.MODE_PRIVATE);
        noteId = getIntent().getIntExtra("id", -1);

        EditText titleEdit = findViewById(R.id.editorTitle);
        EditText contentEdit = findViewById(R.id.editorContent);

        if (noteId >= 0) loadNote(noteId, titleEdit, contentEdit);

        findViewById(R.id.editorBackBtn).setOnClickListener(v -> finish());
        findViewById(R.id.editorSaveBtn).setOnClickListener(v -> saveNote(titleEdit, contentEdit));
        findViewById(R.id.editorDeleteBtn).setOnClickListener(v -> {
            new AlertDialog.Builder(this)
                .setTitle("PURGE RECORD")
                .setMessage("This action cannot be reversed. Proceed?")
                .setPositiveButton("PURGE", (d, w) -> { deleteNote(); finish(); })
                .setNegativeButton("ABORT", null)
                .show();
        });
    }

    private void loadNote(int id, EditText titleEdit, EditText contentEdit) {
        try {
            JSONArray arr = new JSONArray(notePrefs.getString("notes", "[]"));
            for (int i = 0; i < arr.length(); i++) {
                JSONObject obj = arr.getJSONObject(i);
                if (obj.getInt("id") == id) {
                    titleEdit.setText(obj.getString("title"));
                    contentEdit.setText(obj.getString("content"));
                    break;
                }
            }
        } catch (Exception ignored) {}
    }

    private void saveNote(EditText titleEdit, EditText contentEdit) {
        try {
            JSONArray arr = new JSONArray(notePrefs.getString("notes", "[]"));
            String title = titleEdit.getText().toString();
            String content = contentEdit.getText().toString();
            String date = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(new Date());

            if (noteId >= 0) {
                for (int i = 0; i < arr.length(); i++) {
                    JSONObject obj = arr.getJSONObject(i);
                    if (obj.getInt("id") == noteId) {
                        obj.put("title", title);
                        obj.put("content", content);
                        obj.put("date", date);
                        break;
                    }
                }
            } else {
                int maxId = 0;
                for (int i = 0; i < arr.length(); i++) maxId = Math.max(maxId, arr.getJSONObject(i).getInt("id"));
                JSONObject obj = new JSONObject();
                obj.put("id", maxId + 1);
                obj.put("title", title);
                obj.put("content", content);
                obj.put("date", date);
                arr.put(obj);
            }
            notePrefs.edit().putString("notes", arr.toString()).apply();
            finish();
        } catch (Exception e) { e.printStackTrace(); }
    }

    private void deleteNote() {
        try {
            JSONArray arr = new JSONArray(notePrefs.getString("notes", "[]"));
            JSONArray newArr = new JSONArray();
            for (int i = 0; i < arr.length(); i++) {
                if (arr.getJSONObject(i).getInt("id") != noteId) newArr.put(arr.getJSONObject(i));
            }
            notePrefs.edit().putString("notes", newArr.toString()).apply();
        } catch (Exception ignored) {}
    }
}