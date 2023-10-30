use bcd;
DROP TABLE IF EXISTS eval;
CREATE TABLE eval (
    image_uid VARCHAR(64),
    mode VARCHAR(8),
    stage_id INTEGER,
    stage VARCHAR(32),
    step VARCHAR(64),
    method VARCHAR(64),
    mse FLOAT,
    psnr FLOAT,
    ssim FLOAT,
    image_view VARCHAR(4),
    abnormality_type VARCHAR(24),
    assessment INTEGER,
    cancer TINYINT,
    evaluated DATETIME
);