CREATE TABLE questions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  topic TEXT NOT NULL,
  track TEXT NOT NULL,
  level TEXT NOT NULL,
  zone TEXT NOT NULL,
  status TEXT NOT NULL,
  scenario TEXT NOT NULL,
  common_mistake TEXT,
  realistic_solution TEXT NOT NULL,
  napkin_math TEXT,
  deep_dive_title TEXT,
  deep_dive_url TEXT,
  provenance TEXT NOT NULL,
  created_at TEXT,
  last_modified TEXT,
  file_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  authors_json TEXT
);

CREATE INDEX idx_questions_topic ON questions(topic);
CREATE INDEX idx_questions_track_level ON questions(track, level);
CREATE INDEX idx_questions_zone ON questions(zone);
CREATE INDEX idx_questions_status ON questions(status);

CREATE TABLE chains (
  id TEXT PRIMARY KEY,
  name TEXT,
  topic TEXT
);

CREATE TABLE chain_questions (
  chain_id TEXT NOT NULL,
  question_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  PRIMARY KEY (chain_id, position),
  FOREIGN KEY (chain_id) REFERENCES chains(id),
  FOREIGN KEY (question_id) REFERENCES questions(id)
);

CREATE TABLE tags (
  question_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  FOREIGN KEY (question_id) REFERENCES questions(id)
);

CREATE TABLE release_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Full-text search (B.5). Content-table pattern keeps the FTS index in sync
-- with INSERTs/UPDATEs/DELETEs via triggers. Cold-start fingerprint check
-- includes triggers — see staffml-vault-worker/src/index.ts.
CREATE VIRTUAL TABLE questions_fts USING fts5(
  title, scenario, realistic_solution,
  content='questions', content_rowid='rowid'
);

CREATE TRIGGER questions_ai AFTER INSERT ON questions BEGIN
  INSERT INTO questions_fts(rowid, title, scenario, realistic_solution)
  VALUES (new.rowid, new.title, new.scenario, new.realistic_solution);
END;

CREATE TRIGGER questions_ad AFTER DELETE ON questions BEGIN
  INSERT INTO questions_fts(questions_fts, rowid, title, scenario, realistic_solution)
  VALUES('delete', old.rowid, old.title, old.scenario, old.realistic_solution);
END;

CREATE TRIGGER questions_au AFTER UPDATE ON questions BEGIN
  INSERT INTO questions_fts(questions_fts, rowid, title, scenario, realistic_solution)
  VALUES('delete', old.rowid, old.title, old.scenario, old.realistic_solution);
  INSERT INTO questions_fts(rowid, title, scenario, realistic_solution)
  VALUES (new.rowid, new.title, new.scenario, new.realistic_solution);
END;
