CREATE TABLE politiken_ia_prv.recommendation_editorials (
    editorial_id VARCHAR(36) PRIMARY KEY,
    editorial_name VARCHAR(255) NOT NULL,
    editorial_description VARCHAR(255),
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    modified_at TIMESTAMPTZ,
    expire_at TIMESTAMPTZ NOT NULL,
    user_type VARCHAR(20) NOT NULL,
    editorial_status VARCHAR(20) NOT NULL,
    articles SUPER NOT NULL,
    users SUPER,
    user_filter SUPER
);

CREATE TABLE politiken_ia_prv.recommendation_runs (
  run_id VARCHAR PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT GETDATE(),
  ended_at TIMESTAMPTZ,
  run_status VARCHAR,
  training_start  DATE,
  training_end    DATE,
  num_interactions INT,
  config          SUPER
);

CREATE TABLE politiken_ia_prv.recommendation_models (
    model_id VARCHAR PRIMARY KEY,
    model_name VARCHAR,
    model_category VARCHAR, -- editorial, non personal, personal
    model_status VARCHAR,  -- pending, active, ended, cancelled
    model_hash VARCHAR,
    created_at TIMESTAMPTZ DEFAULT GETDATE(),
    modified_at TIMESTAMPTZ,
    expire_at TIMESTAMPTZ,
    editorial_id VARCHAR REFERENCES politiken_ia_prv.recommendation_editorials(editorial_id),
    run_id VARCHAR REFERENCES politiken_ia_prv.recommendation_runs(run_id),
    metadata SUPER
);

CREATE TABLE politiken_ia_prv.recommendation_lists (
    list_id VARCHAR PRIMARY KEY,
    sso_id VARCHAR,
    recommendations SUPER,
    num_recommendations INT,
    is_control BOOLEAN,
    candidate_models SUPER,
    candidate_model_names SUPER,
    run_id VARCHAR REFERENCES politiken_ia_prv.recommendation_runs(run_id),
    model_id VARCHAR REFERENCES politiken_ia_prv.recommendation_models(model_id)
);

-- views
CREATE VIEW politiken_ia_prv.recommendations AS
SELECT
    rl.list_id || '_' || (rec.rank::int)::VARCHAR AS recommendation_id,
    rl.sso_id,
    rec.article_id::int as article_id,
    rec.rank::int as rank,
    rec.score::float as score,
    rl.list_id,
    rl.model_id,
    rl.run_id
FROM
    politiken_ia_prv.recommendation_lists rl
    LEFT JOIN rl.recommendations rec ON TRUE
WHERE
    rl.recommendations IS NOT NULL
    AND NOT rl.is_control;

CREATE VIEW politiken_ia_prv.recommendation_editorial_articles AS
SELECT
    rec.editorial_id,
    article_id_val::int as article_id
FROM politiken_ia_prv.recommendation_editorials rec
JOIN rec.articles AS article_id_val ON TRUE;    -- unnest SUPER array

CREATE VIEW politiken_ia_prv.recommendation_editorial_users AS
SELECT
    rec.editorial_id,
    sso_id_val::varchar as sso_id
FROM politiken_ia_prv.recommendation_editorials rec
JOIN rec.users AS sso_id_val ON TRUE;    -- unnest SUPER array

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE politiken_ia_prv.recommendation_editorials TO web_politiken;


-- Internal users table
DROP TABLE IF EXISTS politiken_ia_prv.internal_recommendation_users;
CREATE TABLE politiken_ia_prv.internal_recommendation_users AS
SELECT distinct email, sso_id, sso_id_hash
FROM jppol_dsa_shrd_external.userservice_latest_v1
WHERE email in (
    'marcus.friis@pol.dk',
    'johannes.s.andersen@pol.dk',
    'mikkel.stampe.davidsen@pol.dk',
    'david.m.houser@pol.dk',
    'thomas.berndt@pol.dk',
    'jens.lenler@pol.dk',
    'anders.g.nielsen@pol.dk'
);
