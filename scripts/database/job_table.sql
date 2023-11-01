use bcd;
DROP TABLE IF EXISTS job;
CREATE TABLE job (
    uid VARCHAR(64),
    name VARCHAR(64),
    mode VARCHAR(8),
    tasks_processed INTEGER,
    task_processing_time FLOAT,
    started DATETIME,
    ended DATETIME,
    duration FLOAT,
    status VARCHAR(16)
);