package com.domwood.nexusai;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class NotesActivity extends AppCompatActivity {
    private RecyclerView recyclerView;
    private NoteAdapter adapter;
    private SharedPreferences notePrefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_notes);
        if (getSupportActionBar() != null) getSupportActionBar().hide();

        notePrefs = getSharedPreferences("nexusai_notes", Context.MODE_PRIVATE);
        recyclerView = findViewById(R.id.notesRecyclerView);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        adapter = new NoteAdapter(loadNotes());
        recyclerView.setAdapter(adapter);
        updateCount();

        findViewById(R.id.notesNewBtn).setOnClickListener(v -> {
            Intent i = new Intent(this, NoteEditorActivity.class);
            i.putExtra("id", -1);
            startActivity(i);
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        adapter.setNotes(loadNotes());
        updateCount();
    }

    private void updateCount() {
        TextView count = findViewById(R.id.notesCount);
        count.setText("[" + adapter.getItemCount() + " RECORDS]");
    }

    private List<Note> loadNotes() {
        List<Note> notes = new ArrayList<>();
        try {
            JSONArray arr = new JSONArray(notePrefs.getString("notes", "[]"));
            for (int i = 0; i < arr.length(); i++) {
                JSONObject obj = arr.getJSONObject(i);
                notes.add(new Note(obj.getInt("id"), obj.getString("title"), obj.getString("content"), obj.getString("date")));
            }
        } catch (Exception ignored) {}
        Collections.sort(notes, (a, b) -> b.id - a.id);
        return notes;
    }

    static class Note {
        int id; String title, content, date;
        Note(int id, String title, String content, String date) {
            this.id = id; this.title = title; this.content = content; this.date = date;
        }
    }

    class NoteAdapter extends RecyclerView.Adapter<NoteAdapter.ViewHolder> {
        private List<Note> notes = new ArrayList<>();
        NoteAdapter(List<Note> n) { notes.addAll(n); }
        void setNotes(List<Note> n) { notes.clear(); notes.addAll(n); notifyDataSetChanged(); }

        @Override
        public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            return new ViewHolder(LayoutInflater.from(parent.getContext()).inflate(R.layout.item_note, parent, false));
        }

        @Override
        public void onBindViewHolder(ViewHolder h, int pos) {
            Note note = notes.get(pos);
            h.title.setText(note.title.isEmpty() ? "[UNTITLED]" : note.title);
            h.date.setText(note.date);
            h.preview.setText(note.content.isEmpty() ? "[EMPTY]" : note.content.substring(0, Math.min(note.content.length(), 100)));
            h.itemView.setOnClickListener(v -> {
                Intent i = new Intent(NotesActivity.this, NoteEditorActivity.class);
                i.putExtra("id", note.id);
                startActivity(i);
            });
        }

        @Override
        public int getItemCount() { return notes.size(); }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView title, date, preview;
            ViewHolder(View v) { super(v); title = v.findViewById(R.id.noteTitle); date = v.findViewById(R.id.noteDate); preview = v.findViewById(R.id.notePreview); }
        }
    }
}