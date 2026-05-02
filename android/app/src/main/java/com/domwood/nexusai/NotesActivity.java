package com.domwood.nexusai;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
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
    private static final String TAG = "NexusAI.Notes";
    private RecyclerView recyclerView;
    private NoteAdapter adapter;
    private SharedPreferences notePrefs;
    private TextView emptyState;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            setContentView(R.layout.activity_notes);
        } catch (Exception e) {
            Log.e(TAG, "Failed to inflate notes layout", e);
            finish();
            return;
        }

        notePrefs = getSharedPreferences("nexusai_notes", Context.MODE_PRIVATE);

        // Back button
        View backArea = findViewById(R.id.notesBackBtn);
        if (backArea != null) {
            backArea.setOnClickListener(v -> finish());
        }

        recyclerView = findViewById(R.id.notesRecyclerView);
        emptyState = findViewById(R.id.notesEmptyState);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        adapter = new NoteAdapter(loadNotes());
        recyclerView.setAdapter(adapter);
        updateCount();
        updateEmptyState();

        View newBtn = findViewById(R.id.notesNewBtn);
        if (newBtn != null) {
            newBtn.setOnClickListener(v -> {
                Intent i = new Intent(this, NoteEditorActivity.class);
                i.putExtra("id", -1);
                startActivity(i);
            });
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        adapter.setNotes(loadNotes());
        updateCount();
        updateEmptyState();
    }

    private void updateCount() {
        TextView count = findViewById(R.id.notesCount);
        if (count != null) {
            count.setText("[" + adapter.getItemCount() + " RECORDS]");
        }
    }

    private void updateEmptyState() {
        if (emptyState != null) {
            emptyState.setVisibility(adapter.getItemCount() == 0 ? View.VISIBLE : View.GONE);
        }
    }

    private List<Note> loadNotes() {
        List<Note> notes = new ArrayList<>();
        try {
            JSONArray arr = new JSONArray(notePrefs.getString("notes", "[]"));
            for (int i = 0; i < arr.length(); i++) {
                JSONObject obj = arr.getJSONObject(i);
                notes.add(new Note(
                    obj.optInt("id", 0),
                    obj.optString("title", ""),
                    obj.optString("content", ""),
                    obj.optString("date", "")));
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to load notes", e);
        }
        Collections.sort(notes, (a, b) -> b.id - a.id);
        return notes;
    }

    static class Note {
        int id;
        String title, content, date;
        Note(int id, String title, String content, String date) {
            this.id = id; this.title = title; this.content = content; this.date = date;
        }
    }

    class NoteAdapter extends RecyclerView.Adapter<NoteAdapter.ViewHolder> {
        private final List<Note> notes = new ArrayList<>();

        NoteAdapter(List<Note> n) { notes.addAll(n); }
        void setNotes(List<Note> n) { notes.clear(); notes.addAll(n); notifyDataSetChanged(); }

        @Override
        public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            return new ViewHolder(LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_note, parent, false));
        }

        @Override
        public void onBindViewHolder(ViewHolder h, int pos) {
            try {
                Note note = notes.get(pos);
                h.title.setText(note.title.isEmpty() ? "[UNTITLED]" : note.title);
                h.date.setText(note.date);
                if (note.content.isEmpty()) {
                    h.preview.setText("[EMPTY]");
                } else {
                    h.preview.setText(note.content.substring(0, Math.min(note.content.length(), 100)));
                }
                h.itemView.setOnClickListener(v -> {
                    Intent i = new Intent(NotesActivity.this, NoteEditorActivity.class);
                    i.putExtra("id", note.id);
                    startActivity(i);
                });
            } catch (Exception e) {
                Log.w(TAG, "Failed to bind note at " + pos, e);
            }
        }

        @Override
        public int getItemCount() { return notes.size(); }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView title, date, preview;
            ViewHolder(View v) {
                super(v);
                title = v.findViewById(R.id.noteTitle);
                date = v.findViewById(R.id.noteDate);
                preview = v.findViewById(R.id.notePreview);
            }
        }
    }
}
