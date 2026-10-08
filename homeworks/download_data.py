"""Download and normalize Amazon Reviews 2014 / Beauty 5-core.

The script creates one shared data directory for HW2 and HW3:

    homeworks/data/interactions.parquet
    homeworks/data/items.parquet
    homeworks/data/users.parquet

Source: https://mcauleylab.ucsd.edu/public_datasets/data/amazon/index_2014.html
"""

import argparse
import ast
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# SNAP currently serves these legacy 2014 files reliably over HTTP; its HTTPS
# endpoint may reset the TLS connection.
BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
REVIEWS_FILENAME = "reviews_Beauty_5.json.gz"
METADATA_FILENAME = "meta_Beauty.json.gz"


def download_file(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}", flush=True)
    with tqdm(
        desc=destination.name,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        file=sys.stdout,
    ) as progress:

        def update_progress(
            block_number: int,
            block_size: int,
            total_size: int,
        ) -> None:
            if total_size > 0 and progress.total != total_size:
                progress.total = total_size
            downloaded = block_number * block_size
            progress.update(max(0, downloaded - progress.n))

        urllib.request.urlretrieve(url, destination, reporthook=update_progress)


def read_gzip_records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # The original 2014 files contain Python-style literals.
                yield ast.literal_eval(line)


def prepare_dataset(
    data_dir: Path,
    *,
    force: bool = False,
    max_reviews: int | None = None,
) -> None:
    raw_dir = data_dir / "raw"
    reviews_path = raw_dir / REVIEWS_FILENAME
    metadata_path = raw_dir / METADATA_FILENAME

    download_file(f"{BASE_URL}/{REVIEWS_FILENAME}", reviews_path, force)
    download_file(f"{BASE_URL}/{METADATA_FILENAME}", metadata_path, force)

    interaction_rows = []
    for row_number, record in enumerate(read_gzip_records(reviews_path)):
        if max_reviews is not None and row_number >= max_reviews:
            break

        interaction_rows.append(
            {
                "user_id": str(record["reviewerID"]),
                "item_id": str(record["asin"]),
                "rating": float(record["overall"]),
                "timestamp": pd.to_datetime(
                    int(record["unixReviewTime"]),
                    unit="s",
                    utc=True,
                ),
            }
        )

    interactions = pd.DataFrame(interaction_rows).drop_duplicates(
        ["user_id", "item_id", "timestamp"]
    )
    selected_item_ids = set(interactions["item_id"])

    item_rows = []
    for record in read_gzip_records(metadata_path):
        item_id = str(record.get("asin", ""))
        if item_id not in selected_item_ids:
            continue

        category_groups = record.get("categories") or []
        categories = sorted(
            {str(category) for group in category_groups for category in group}
        )
        item_rows.append(
            {
                "item_id": item_id,
                "title": str(record.get("title") or "unknown title"),
                "brand": str(record.get("brand") or "unknown"),
                "price": pd.to_numeric(record.get("price"), errors="coerce"),
                "categories": categories,
            }
        )

    items = pd.DataFrame(item_rows).drop_duplicates("item_id")
    missing_item_ids = selected_item_ids.difference(items["item_id"])
    if missing_item_ids:
        missing_items = pd.DataFrame(
            {
                "item_id": sorted(missing_item_ids),
                "title": "unknown title",
                "brand": "unknown",
                "price": float("nan"),
                "categories": [[] for _ in missing_item_ids],
            }
        )
        items = pd.concat([items, missing_items], ignore_index=True)

    users = pd.DataFrame({"user_id": sorted(interactions["user_id"].unique())})

    data_dir.mkdir(parents=True, exist_ok=True)
    interactions.sort_values("timestamp").to_parquet(
        data_dir / "interactions.parquet",
        index=False,
    )
    items.to_parquet(data_dir / "items.parquet", index=False)
    users.to_parquet(data_dir / "users.parquet", index=False)

    print(
        f"Saved {len(interactions):,} interactions, "
        f"{len(users):,} users and {len(items):,} items to {data_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-reviews", type=int)
    arguments = parser.parse_args()

    prepare_dataset(
        arguments.data_dir,
        force=arguments.force,
        max_reviews=arguments.max_reviews,
    )
