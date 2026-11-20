from pathlib import PurePosixPath
from typing import Any

import orjson

from recommender.models.nonpersonal import PoolRecommender
from recommender.utils.s3 import download_from_s3


class S3Recommender(PoolRecommender):
    base_directory = "s3-recommender"

    def __init__(
        self, key: str, bucket: str | None = None, *, randomize_order: bool = False
    ):
        self._validate_key(key)
        payload = download_from_s3(key, bucket)
        pool = self._validate_s3_payload(payload)
        super().__init__(pool, randomize_order=randomize_order)

    @classmethod
    def _validate_key(cls, key: str) -> None:
        path = PurePosixPath(key)
        if not path.parts or path.parts[0] != cls.base_directory:
            raise ValueError(
                f"S3Recommender can only load keys from '{cls.base_directory}/'"
            )

    @staticmethod
    def _validate_s3_payload(payload: Any) -> list[int]:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        elif not isinstance(payload, bytes):
            raise TypeError("S3 payload must be JSON bytes or string")

        try:
            data = orjson.loads(payload)
        except orjson.JSONDecodeError as e:
            raise ValueError("S3 payload is not valid JSON") from e

        if not isinstance(data, list):
            raise TypeError("S3 payload must be a JSON list")

        if not all(isinstance(item, int) for item in data):
            raise ValueError("S3 payload must be a JSON list of integers")

        return data

    @property
    def name(self) -> str:
        return "S3Pool"

    @property
    def category(self) -> str:
        return "analytical"


if __name__ == "__main__":
    base_path = "pool.json"
    key = f"s3-recommender/{base_path}"
    model = S3Recommender(key)
    print(model.recommend())
