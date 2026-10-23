# Recommender

Run as

```sh
echo "y" | python main.py
```

Run this on a schedule using cron as

```sh
* * * * * source /path/to/venv/bin/activate && \
    source /path/to/secrets/s3.sh && \
    echo "y" | python /path/to/recommender/main.py
```

or use the setup helper script [`setup_crontab.sh`](/scripts/setup_crontab.sh).

## Setups

### Secrets

To setup secrets, create a copy of [`secrets_template/`](secrets_template) as `secrets/` from recommender root.

```sh
cp -r secrets_template secrets
```

Fill out `s3.sh` with secrets

```sh
export AWS_ACCESS_KEY_ID=YOUR_ID
export AWS_SECRET_ACCESS_KEY=YOUR_KEY
```

Then put them in env variables with

```sh
source secrets/s3.sh
```

Afterwards, check they're configured correctly using

```sh
echo $AWS_SECRET_ACCESS_KEY
```

Now you can put stuff into s3 buckets. This functionality should be updated in the future, but it works for now :)

## Notes

This project uses Polars for DataFrame handling. If you are unfamiliar with it, but familiar with pandas check out [this cheat sheet](https://franzdiebold.github.io/polars-cheat-sheet/Polars_cheat_sheet.pdf).

## TODO

* Rename column `run_timestamp` -> `run_timestamptz`.
