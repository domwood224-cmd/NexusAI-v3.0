package com.domwood.nexusai;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ChatActivity extends AppCompatActivity {
    private RecyclerView recyclerView;
    private MessageAdapter adapter;
    private EditText chatInput;
    private Button chatSendBtn;
    private SharedPreferences prefs;
    private ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chat);
        getSupportActionBar().hide();

        prefs = getSharedPreferences("nexusai_settings", Context.MODE_PRIVATE);
        recyclerView = findViewById(R.id.chatRecyclerView);
        chatInput = findViewById(R.id.chatInput);
        chatSendBtn = findViewById(R.id.chatSendBtn);

        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        adapter = new MessageAdapter(loadMessages());
        recyclerView.setAdapter(adapter);
        recyclerView.scrollToPosition(adapter.getItemCount() - 1);

        chatSendBtn.setOnClickListener(v -> sendMessage());

        if (adapter.getItemCount() == 0) {
            adapter.addMessage(new ChatMessage("SYSTEM",
                "NEXUS AI NEURAL INTERFACE v3.1\n[SYSTEM ONLINE]\n\nConfigure API endpoint in System Config\nto initialize neural link.",
                "ai"));
        }
    }

    private void sendMessage() {
        String text = chatInput.getText().toString().trim();
        if (text.isEmpty()) return;
        chatInput.setText("");

        String time = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
        adapter.addMessage(new ChatMessage("USER", text, "user", time));
        recyclerView.scrollToPosition(adapter.getItemCount() - 1);

        String apiUrl = prefs.getString("api_url", "");
        String apiKey = prefs.getString("api_key", "");

        if (apiUrl.isEmpty() || apiKey.isEmpty()) {
            String t = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
            adapter.addMessage(new ChatMessage("SYSTEM",
                "[ERROR] API endpoint not configured.\nNavigate to System Config.", "ai", t));
            recyclerView.scrollToPosition(adapter.getItemCount() - 1);
            saveMessages();
            return;
        }

        executor.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("model", prefs.getString("model", "gpt-3.5-turbo"));

                JSONArray messages = new JSONArray();
                String sysPrompt = prefs.getString("system_prompt", "");
                if (!sysPrompt.isEmpty()) {
                    messages.put(new JSONObject().put("role", "system").put("content", sysPrompt));
                }

                for (ChatMessage msg : adapter.getMessages()) {
                    String role = msg.type.equals("user") ? "user" : "assistant";
                    messages.put(new JSONObject().put("role", role).put("content", msg.text));
                }
                body.put("messages", messages);

                URL url = new URL(apiUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setRequestProperty("Authorization", "Bearer " + apiKey);
                conn.setConnectTimeout(30000);
                conn.setReadTimeout(60000);
                conn.setDoOutput(true);

                try (OutputStream os = conn.getOutputStream()) {
                    os.write(body.toString().getBytes());
                    os.flush();
                }

                int code = conn.getResponseCode();
                BufferedReader reader;
                if (code >= 200 && code < 300) {
                    reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                } else {
                    reader = new BufferedReader(new InputStreamReader(conn.getErrorStream()));
                }
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) response.append(line);
                reader.close();

                String respStr = response.toString();
                String aiText;
                if (code >= 200 && code < 300) {
                    JSONObject respJson = new JSONObject(respStr);
                    aiText = respJson.getJSONArray("choices").getJSONObject(0).getJSONObject("message").getString("content");
                } else {
                    aiText = "[ERROR " + code + "] " + respStr.substring(0, Math.min(respStr.length(), 300));
                }

                String t = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
                runOnUiThread(() -> {
                    adapter.addMessage(new ChatMessage("NEXUS", aiText, "ai", t));
                    recyclerView.scrollToPosition(adapter.getItemCount() - 1);
                    saveMessages();
                });
            } catch (Exception e) {
                String t = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
                String errText = "[CONNECTION FAILED]\n" + e.getMessage();
                runOnUiThread(() -> {
                    adapter.addMessage(new ChatMessage("SYSTEM", errText, "ai", t));
                    recyclerView.scrollToPosition(adapter.getItemCount() - 1);
                    saveMessages();
                });
            }
        });
    }

    private List<ChatMessage> loadMessages() {
        List<ChatMessage> msgs = new ArrayList<>();
        try {
            SharedPreferences chatPrefs = getSharedPreferences("nexusai_chat", Context.MODE_PRIVATE);
            String json = chatPrefs.getString("messages", "[]");
            JSONArray arr = new JSONArray(json);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject obj = arr.getJSONObject(i);
                msgs.add(new ChatMessage(obj.getString("sender"), obj.getString("text"), obj.getString("type"), obj.optString("time", "")));
            }
        } catch (Exception ignored) {}
        return msgs;
    }

    private void saveMessages() {
        try {
            SharedPreferences chatPrefs = getSharedPreferences("nexusai_chat", Context.MODE_PRIVATE);
            JSONArray arr = new JSONArray();
            for (ChatMessage m : adapter.getMessages()) {
                JSONObject obj = new JSONObject();
                obj.put("sender", m.sender);
                obj.put("text", m.text);
                obj.put("type", m.type);
                obj.put("time", m.time != null ? m.time : "");
                arr.put(obj);
            }
            chatPrefs.edit().putString("messages", arr.toString()).apply();
        } catch (Exception ignored) {}
    }

    static class ChatMessage {
        String sender, text, type, time;
        ChatMessage(String sender, String text, String type) { this(sender, text, type, ""); }
        ChatMessage(String sender, String text, String type, String time) { this.sender = sender; this.text = text; this.type = type; this.time = time; }
    }

    class MessageAdapter extends RecyclerView.Adapter<MessageAdapter.ViewHolder> {
        private List<ChatMessage> messages = new ArrayList<>();

        MessageAdapter(List<ChatMessage> msgs) { messages.addAll(msgs); }
        List<ChatMessage> getMessages() { return messages; }
        void addMessage(ChatMessage m) { messages.add(m); notifyItemInserted(messages.size() - 1); }

        @Override
        public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_chat_message, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(ViewHolder holder, int position) {
            ChatMessage msg = messages.get(position);
            LinearLayout.LayoutParams lp = (LinearLayout.LayoutParams) holder.bubble.getLayoutParams();

            if (msg.type.equals("user")) {
                lp.gravity = android.view.Gravity.END;
                holder.bubble.setBackgroundResource(R.drawable.bg_user_bubble);
                holder.sender.setTextColor(0xFF00FF41);
                holder.sender.setText(msg.sender);
                holder.text.setTextColor(0xFF00FF41);
            } else {
                lp.gravity = android.view.Gravity.START;
                holder.bubble.setBackgroundResource(R.drawable.bg_ai_bubble);
                holder.sender.setTextColor(0xFFFF6D00);
                holder.sender.setText(msg.sender);
                holder.text.setTextColor(0xFFFF6D00);
            }
            holder.bubble.setLayoutParams(lp);
            holder.text.setText(msg.text);
            if (msg.time != null && !msg.time.isEmpty()) {
                holder.time.setTextColor(0xFF006600);
                holder.time.setText(msg.time);
            }
        }

        @Override
        public int getItemCount() { return messages.size(); }

        class ViewHolder extends RecyclerView.ViewHolder {
            LinearLayout bubble;
            TextView sender, text, time;
            ViewHolder(View v) {
                super(v);
                bubble = v.findViewById(R.id.messageBubble);
                sender = v.findViewById(R.id.messageSender);
                text = v.findViewById(R.id.messageText);
                time = v.findViewById(R.id.messageTime);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
    }
}