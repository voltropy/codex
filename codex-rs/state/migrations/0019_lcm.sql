CREATE TABLE IF NOT EXISTS lcm_messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    raw_item_json TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(thread_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_lcm_messages_thread_created_at
    ON lcm_messages(thread_id, created_at DESC, message_id DESC);

CREATE TABLE IF NOT EXISTS lcm_summaries (
    summary_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    depth INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    file_ids TEXT NOT NULL,
    earliest_at INTEGER,
    latest_at INTEGER,
    descendant_count INTEGER NOT NULL DEFAULT 0,
    descendant_token_count INTEGER NOT NULL DEFAULT 0,
    source_message_token_count INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lcm_summaries_thread_created_at
    ON lcm_summaries(thread_id, created_at DESC, summary_id DESC);

CREATE TABLE IF NOT EXISTS lcm_context_items (
    thread_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    message_id INTEGER,
    summary_id TEXT,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (thread_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_lcm_context_items_thread_type
    ON lcm_context_items(thread_id, item_type, ordinal);

CREATE TABLE IF NOT EXISTS lcm_summary_messages (
    summary_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (summary_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_lcm_summary_messages_summary
    ON lcm_summary_messages(summary_id, ordinal);

CREATE TABLE IF NOT EXISTS lcm_summary_parents (
    summary_id TEXT NOT NULL,
    parent_summary_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (summary_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_lcm_summary_parents_parent
    ON lcm_summary_parents(parent_summary_id, ordinal);

CREATE TABLE IF NOT EXISTS lcm_large_files (
    file_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    file_name TEXT,
    mime_type TEXT,
    byte_size INTEGER,
    storage_uri TEXT NOT NULL,
    exploration_summary TEXT,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lcm_large_files_thread_created_at
    ON lcm_large_files(thread_id, created_at DESC, file_id DESC);
