use bcd;
DROP TABLE IF EXISTS eval;
CREATE TABLE eval (
    test_no INTEGER,
    source_image_uid VARCHAR(64),
    source_image_filepath VARCHAR(128),
    test_image_uid VARCHAR(64),
    test_image_filepath VARCHAR(128),
    mode VARCHAR(8),
    stage_id INTEGER,
    stage VARCHAR(32),
    image_view VARCHAR(4),
    abnormality_type VARCHAR(24),
    assessment INTEGER,
    cancer TINYINT,
    method VARCHAR(64),
    params VARCHAR(128),
    comp_time FLOAT,
    mse FLOAT,
    psnr FLOAT,
    ssim FLOAT,
    evaluated DATETIME
);