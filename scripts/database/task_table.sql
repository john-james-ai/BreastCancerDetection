use bcd;
DROP TABLE IF EXISTS task;
CREATE TABLE task (
    uid VARCHAR(64),
    mode VARCHAR(8),
    stage_id INTEGER,
    stage VARCHAR(32),
    method VARCHAR(64),
    params VARCHAR(128),
    images_processed INTEGER,
    created DATETIME
);