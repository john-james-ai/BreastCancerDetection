use bcd;
DROP TABLE IF EXISTS eval;
CREATE TABLE eval (
    orig_uid VARCHAR(64),
    test_uid VARCHAR(64),
    mode VARCHAR(8),
    stage_id INTEGER,
    stage VARCHAR(32),
    method VARCHAR(64),
    params VARCHAR(128),
    image_view VARCHAR(4),
    abnormality_type VARCHAR(24),
    assessment INTEGER,
    cancer TINYINT,
    build_time FLOAT,
    task_id VARCHAR(64),
    mse FLOAT,
    psnr FLOAT,
    ssim FLOAT,
    evaluated DATETIME
);