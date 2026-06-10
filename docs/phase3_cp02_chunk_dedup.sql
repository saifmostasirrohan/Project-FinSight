-- Remove duplicate chunks created by repeated uploads before enforcing uniqueness.
DELETE FROM chunks AS duplicate
USING chunks AS original
WHERE duplicate.session_id = original.session_id
  AND md5(duplicate.content) = md5(original.content)
  AND duplicate.id > original.id;

-- PostgREST upserts need named columns that match a unique index.
ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS content_hash text
GENERATED ALWAYS AS (md5(content)) STORED;

CREATE UNIQUE INDEX IF NOT EXISTS unique_chunk_session_content_hash_idx
ON chunks (session_id, content_hash);
