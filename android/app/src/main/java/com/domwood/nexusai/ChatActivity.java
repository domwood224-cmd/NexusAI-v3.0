package com.domwood.nexusai;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ChatActivity extends AppCompatActivity {
    private static final String TAG = "NexusAI.Chat";
    private static final int MAX_HISTORY_MESSAGES = 30;

    private RecyclerView recyclerView;
    private MessageAdapter adapter;
    private EditText chatInput;
    private Button chatSendBtn;
    private SharedPreferences prefs;
    private ExecutorService executor;
    private volatile boolean isSending = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            setContentView(R.layout.activity_chat);
        } catch (Exception e) {
            Log.e(TAG, "Failed to inflate chat layout", e);
            finish();
            return;
        }

        try {
            prefs = getSharedPreferences("nexusai_settings", Context.MODE_PRIVATE);

            recyclerView = findViewById(R.id.chatRecyclerView);
            chatInput = findViewById(R.id.chatInput);
            chatSendBtn = findViewById(R.id.chatSendBtn);

            if (recyclerView != null) {
                recyclerView.setLayoutManager(new LinearLayoutManager(this));
                adapter = new MessageAdapter(loadMessages());
                recyclerView.setAdapter(adapter);
                if (adapter.getItemCount() > 0) {
                    recyclerView.scrollToPosition(adapter.getItemCount() - 1);
                }
            }

            if (chatSendBtn != null) {
                chatSendBtn.setOnClickListener(v -> sendMessage());
            }

            if (chatInput != null) {
                chatInput.setOnEditorActionListener((v, actionId, event) -> {
                    if (actionId == EditorInfo.IME_ACTION_SEND ||
                        (event != null && event.getAction() == KeyEvent.ACTION_DOWN
                            && event.getKeyCode() == KeyEvent.KEYCODE_ENTER)) {
                        sendMessage();
                        return true;
                    }
                    return false;
                });
            }

            View backBtn = findViewById(R.id.chatBackBtn);
            if (backBtn != null) {
                backBtn.setOnClickListener(v -> finish());
            }

            if (adapter != null && adapter.getItemCount() == 0) {
                adapter.addMessage(new ChatMessage("SYSTEM",
                    "NEXUS AI NEURAL INTERFACE v6.2\n[SYSTEM ONLINE]\n\nConfigure API endpoint in System Config\nto initialize neural link.",
                    "ai"));
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to setup chat", e);
        }
    }

    private void sendMessage() {
        if (isSending) return;
        if (chatInput == null || chatSendBtn == null || adapter == null) return;

        String text = chatInput.getText().toString().trim();
        if (text.isEmpty()) return;
        chatInput.setText("");

        String time = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
        adapter.addMessage(new ChatMessage("USER", text, "user", time));
        if (recyclerView != null) {
            recyclerView.scrollToPosition(adapter.getItemCount() - 1);
        }

        if (prefs == null) {
            showSystemError("Settings not loaded. Restart the app.");
            return;
        }

        String apiUrl = prefs.getString("api_url", "");
        String apiKey = prefs.getString("api_key", "");

        if (apiUrl.isEmpty() || apiKey.isEmpty()) {
            String t = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
            adapter.addMessage(new ChatMessage("SYSTEM",
                "[ERROR] API endpoint not configured.\nNavigate to System Config to set your API URL and Key.",
                "ai", t));
            recyclerView.scrollToPosition(adapter.getItemCount() - 1);
            saveMessages();
            return;
        }

        isSending = true;
        chatSendBtn.setEnabled(false);
        chatSendBtn.setText("...");
        executor = Executors.newSingleThreadExecutor();
        executor.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("model", prefs.getString("model", "gpt-3.5-turbo"));

                JSONArray messages = new JSONArray();
                String sysPrompt = prefs.getString("system_prompt", "");
                if (!sysPrompt.isEmpty()) {
                    messages.put(new JSONObject().put("role", "system").put("content", sysPrompt));
                }

                List<ChatMessage> allMsgs = adapter.getMessages();
                int startIdx = Math.max(0, allMsgs.size() - MAX_HISTORY_MESSAGES);
                for (int i = startIdx; i < allMsgs.size(); i++) {
                    ChatMessage msg = allMsgs.get(i);
                    if ("SYSTEM".equals(msg.sender)) continue;
                    String role = "user".equals(msg.type) ? "user" : "assistant";
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
                    os.write(body.toString().getBytes(StandardCharsets.UTF_8));
                    os.flush();
                }

                int code = conn.getResponseCode();
                BufferedReader reader;
                if (code >= 200 && code < 300) {
                    reader = new BufferedReader(new InputStreamReader(
                        conn.getInputStream(), StandardCharsets.UTF_8));
                } else {
                    InputStream errStream = conn.getErrorStream();
                    InputStream fallback = errStream != null ? errStream : conn.getInputStream();
                    reader = new BufferedReader(new InputStreamReader(fallback, StandardCharsets.UTF_8));
                }
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) response.append(line);
                reader.close();

                String respStr = response.toString();
                String aiText;
                if (code >= 200 && code < 300) {
                    try {
                        JSONObject respJson = new JSONObject(respStr);
                        JSONArray choices = respJson.getJSONArray("choices");
                        if (choices.length() > 0) {
                            aiText = choices.getJSONObject(0).getJSONObject("message")
                                .getString("content");
                        } else {
                            aiText = "[WARNING] Empty response from API.";
                        }
                    } catch (Exception parseErr) {
                        aiText = "[PARSE ERROR] Could not read API response.";
                        Log.w(TAG, "Parse error", parseErr);
                    }
                } else {
                    String errBody = respStr.length() > 300
                        ? respStr.substring(0, 300) : respStr;
                    aiText = "[ERROR " + code + "] " + errBody;
                }

                final String responseText = aiText;
                String t = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
                runOnUiThread(() -> {
                    if (isFinishing() || isDestroyed()) return;
                    if (adapter != null) {
                        adapter.addMessage(new ChatMessage("NEXUS", responseText, "ai", t));
                        if (recyclerView != null) {
                            recyclerView.scrollToPosition(adapter.getItemCount() - 1);
                        }
                        saveMessages();
                    }
                });
            } catch (final Exception e) {
                Log.e(TAG, "Chat request failed", e);
                String t = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
                final String errText = "[CONNECTION FAILED]\n"
                    + (e.getMessage() != null ? e.getMessage() : "Unknown error");
                runOnUiThread(() -> {
                    if (isFinishing() || isDestroyed()) return;
                    if (adapter != null) {
                        adapter.addMessage(new ChatMessage("SYSTEM", errText, "ai", t));
                        if (recyclerView != null) {
                            recyclerView.scrollToPosition(adapter.getItemCount() - 1);
                        }
                        saveMessages();
                    }
                });
            } finally {
                runOnUiThread(() -> {
                    if (isFinishing() || isDestroyed()) return;
                    isSending = false;
                    if (chatSendBtn != null) {
                        chatSendBtn.setEnabled(true);
                        chatSendBtn.setText(">");
                    }
                    if (executor != null) {
                        executor.shutdownNow();
                        executor = null;
                    }
                });
            }
        });
    }

    private void showSystemError(String msg) {
        String t = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
        if (adapter != null) {
            adapter.addMessage(new ChatMessage("SYSTEM", msg, "ai", t));
            if (recyclerView != null) recyclerView.scrollToPosition(adapter.getItemCount() - 1);
        }
    }

    private List<ChatMessage> loadMessages() {
        List<ChatMessage> msgs = new ArrayList<>();
        try {
            SharedPreferences chatPrefs = getSharedPreferences("nexusai_chat", Context.MODE_PRIVATE);
            String json = chatPrefs.getString("messages", "[]");
            JSONArray arr = new JSONArray(json);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject obj = arr.getJSONObject(i);
                msgs.add(new ChatMessage(
                    obj.optString("sender", "SYSTEM"),
                    obj.optString("text", ""),
                    obj.optString("type", "ai"),
                    obj.optString("time", "")));
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to load messages", e);
        }
        return msgs;
    }

    private void saveMessages() {
        try {
            if (adapter == null) return;
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
        } catch (Exception e) {
            Log.w(TAG, "Failed to save messages", e);
        }
    }

    static class ChatMessage {
        String sender, text, type, time;
        ChatMessage(String sender, String text, String type) {
            this(sender, text, type, "");
        }
        ChatMessage(String sender, String text, String type, String time) {
            this.sender = sender; this.text = text;
            this.type = type; this.time = time;
        }
    }

    class MessageAdapter extends RecyclerView.Adapter<MessageAdapter.ViewHolder> {
        private final List<ChatMessage> messages = new ArrayList<>();

        MessageAdapter(List<ChatMessage> msgs) { messages.addAll(msgs); }
        List<ChatMessage> getMessages() { return new ArrayList<>(messages); }
        void addMessage(ChatMessage m) {
            messages.add(m);
            notifyItemInserted(messages.size() - 1);
        }

        @Override
        public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_chat_message, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(ViewHolder holder, int position) {
            try {
                ChatMessage msg = messages.get(position);
                if (holder.bubble == null) return;

                LinearLayout.LayoutParams lp;
                ViewGroup.LayoutParams baseLp = holder.bubble.getLayoutParams();
                if (baseLp instanceof LinearLayout.LayoutParams) {
                    lp = (LinearLayout.LayoutParams) baseLp;
                } else {
                    lp = new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT);
                }

                if ("user".equals(msg.type)) {
                    lp.gravity = Gravity.END;
                    holder.bubble.setBackgroundResource(R.drawable.bg_user_bubble);
                } else {
                    lp.gravity = Gravity.START;
                    holder.bubble.setBackgroundResource(R.drawable.bg_ai_bubble);
                }
                holder.bubble.setLayoutParams(lp);

                if (holder.sender != null) {
                    holder.sender.setTextColor(
                        "user".equals(msg.type) ? 0xFF00FF41 : 0xFFFF6D00);
                    holder.sender.setText(msg.sender);
                }
                if (holder.text != null) {
                    holder.text.setTextColor(
                        "user".equals(msg.type) ? 0xFF00FF41 : 0xFFFF6D00);
                    holder.text.setText(msg.text);
                }
                if (holder.time != null) {
                    if (msg.time != null && !msg.time.isEmpty()) {
                        holder.time.setVisibility(View.VISIBLE);
                        holder.time.setTextColor(0xFF006600);
                        holder.time.setText(msg.time);
                    } else {
                        holder.time.setVisibility(View.GONE);
                    }
                }
            } catch (Exception e) {
                Log.w(TAG, "Failed to bind message at " + position, e);
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
        if (executor != null) {
            try { executor.shutdownNow(); } catch (Exception ignored) {}
            executor = null;
        }
    }
}
