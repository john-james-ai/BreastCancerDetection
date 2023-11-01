use bcd;
DROP TABLE IF EXISTS task;
CREATE TABLE task (
    uid VARCHAR(64),
    name VARCHAR(64),
    mode VARCHAR(8),
    stage_id INTEGER,
    stage VARCHAR(32),
    method_name VARCHAR(64),
    method_module VARCHAR(128),
    method_params_string VARCHAR(128),
    method_params_name VARCHAR(64),
    method_params_module VARCHAR(128),
    task_params_string VARCHAR(128),
    task_params_name VARCHAR(64),
    task_params_module VARCHAR(128),
    images_processed INTEGER,
    image_processing_time FLOAT,
    started DATETIME,
    ended DATETIME,
    duration FLOAT,
    status VARCHAR(16),
    job_id VARCHAR(64)
);