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
