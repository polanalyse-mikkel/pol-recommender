# Recommender

## Secrets

To setup secrets, create a copy of [`.env.template`](.env.template) as `.env` from recommender root.

```sh
cp -r .env.template .env
```

Fill out `.env` with secrets

```sh
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
AWS_S3_BUCKET=RECOMMENDATION_DATA_BUCKET_NAME
POL_CONNECTOR_DSN=PSYCOPG_DSN_TO_REDSHIFT
```

## Data

Recommendation data is stored in a Redshift database. All except `layer_one_re`The schema is as follows:

```mermaid
erDiagram
    recommendation_editorials {
        varchar editorial_id PK
        varchar editorial_name
        varchar editorial_description
        varchar created_by
        timestamptz created_at
        timestamptz modified_at
        timestamptz expire_at
        varchar user_type
        varchar editorial_status
        super articles
        super users
        super user_filter
    }

    recommendation_runs {
        varchar run_id PK
        timestamptz created_at
        timestamptz ended_at
        varchar run_status
        date training_start
        date training_end
        int num_interactions
        super config
    }

    recommendation_models {
        varchar model_id PK
        varchar model_name
        varchar model_category
        varchar model_status
        varchar model_hash
        timestamptz created_at
        timestamptz modified_at
        timestamptz expire_at
        varchar editorial_id FK
        varchar run_id FK
        super metadata
    }

    recommendation_lists {
        varchar list_id PK
        varchar sso_id
        super recommendations
        int num_recommendations
        boolean is_control
        super candidate_models
        super candidate_model_names
        varchar run_id FK
        varchar model_id FK
    }

    layer_one_recommendations {
        date event_date
        timestamp event_datetime
        timestamptz event_timestamptz
        varchar sso_id
        varchar hashed_session_id
        varchar recommendation_model
        boolean is_recommendation
        varchar article_id
        varchar event_type
        boolean is_impression
        boolean is_view
        boolean is_click
        varchar recommendation_spot
        varchar article_pins
        varchar parent_article
        boolean is_pin_click
        varchar visitor_type
        varchar device_type
        varchar pageview_id
    }

    recommendation_runs ||--o{ recommendation_models : has
    recommendation_editorials ||--o{ recommendation_models : backs_editorial_model
    recommendation_runs ||--o{ recommendation_lists : produces
    recommendation_models ||--o{ recommendation_lists : selected_for

    recommendation_lists }o..o{ layer_one_recommendations : "logical via sso_id"
    recommendation_models }o..o{ layer_one_recommendations : "logical via model_name/recommendation_model"
```
