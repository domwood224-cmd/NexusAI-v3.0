package com.nexusai.assistant.memory;

import java.util.HashMap;
import java.util.Map;

public class MemoryManager {
    private final Map<String, String> storage = new HashMap<>();

    public void store(String key, String value) {
        storage.put(key, value);
    }

    public String retrieve(String key) {
        return storage.get(key);
    }
}
