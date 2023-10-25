use bcd;
DROP TABLE IF EXISTS image;
CREATE TABLE image (
    id VARCHAR(64) PRIMARY KEY,
    case_id VARCHAR(64),
    stage_id INT,
    stage VARCHAR(64),
    left_or_right_breast VARCHAR(8),
    image_view VARCHAR(4),
    abnormality_type VARCHAR(24),
    assessment INT,
    breast_density INT,
    bit_depth INT,
    height INT,
    width INT,
    size BIGINT,
    aspect_ratio FLOAT,
    min_pixel_value INT,
    max_pixel_value INT,
    range_pixel_values INT,
    mean_pixel_value FLOAT,
    median_pixel_value INT,
    std_pixel_value FLOAT,
    filepath VARCHAR(256),
    fileset VARCHAR(8),
    cancer TINYINT,
    preprocessor VARCHAR(64),
    task_id VARCHAR(64),
    created DATETIME
);

use bcd_test;
DROP TABLE IF EXISTS image;
CREATE TABLE image (
    id VARCHAR(64) PRIMARY KEY,
    case_id VARCHAR(64),
    stage_id INT,
    stage VARCHAR(64),
    left_or_right_breast VARCHAR(8),
    image_view VARCHAR(4),
    abnormality_type VARCHAR(24),
    assessment INT,
    breast_density INT,
    bit_depth INT,
    height INT,
    width INT,
    size BIGINT,
    aspect_ratio FLOAT,
    min_pixel_value INT,
    max_pixel_value INT,
    range_pixel_values INT,
    mean_pixel_value FLOAT,
    median_pixel_value INT,
    std_pixel_value FLOAT,
    filepath VARCHAR(256),
    fileset VARCHAR(8),
    cancer TINYINT,
    preprocessor VARCHAR(64),
    task_id VARCHAR(64),
    created DATETIME
);


use bcd_dev;
DROP TABLE IF EXISTS image;
CREATE TABLE image (
    id VARCHAR(64) PRIMARY KEY,
    case_id VARCHAR(64),
    stage_id INT,
    stage VARCHAR(64),
    left_or_right_breast VARCHAR(8),
    image_view VARCHAR(4),
    abnormality_type VARCHAR(24),
    assessment INT,
    breast_density INT,
    bit_depth INT,
    height INT,
    width INT,
    size BIGINT,
    aspect_ratio FLOAT,
    min_pixel_value INT,
    max_pixel_value INT,
    range_pixel_values INT,
    mean_pixel_value FLOAT,
    median_pixel_value INT,
    std_pixel_value FLOAT,
    filepath VARCHAR(256),
    fileset VARCHAR(8),
    cancer TINYINT,
    preprocessor VARCHAR(64),
    task_id VARCHAR(64),
    created DATETIME
);