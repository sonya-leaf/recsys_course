import ast
from collections import Counter
from pathlib import Path

import truststore

truststore.inject_into_ssl()

import pandas as pd
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLEAN_PATH = DATA_DIR / "recipes.parquet"

MIN_INGREDIENT_COUNT = 5

REPO_ID = "rogozinushka/povarenok-recipes"
PARQUET_FILENAME = "default/train/0000.parquet"
PARQUET_REVISION = "refs/convert/parquet"


def download_raw() -> Path:
    print("Скачиваем датасет рецептов...")
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=PARQUET_FILENAME,
            repo_type="dataset",
            revision=PARQUET_REVISION,
        )
    )


def parse_ingredients(raw: str | None) -> tuple[str, ...]:
    # Кортеж, а не список: колонка "ingredients" должна быть хэшируемой —
    # её хэширует Streamlit при кэшировании (@st.cache_data/cache_resource).
    if not raw or not isinstance(raw, str):
        return ()
    try:
        ingredients = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return ()
    if not isinstance(ingredients, dict):
        return ()
    return tuple(name.strip().lower() for name in ingredients if name and name.strip())


def filter_rare_ingredients(recipes: pd.DataFrame, min_count: int = MIN_INGREDIENT_COUNT) -> pd.DataFrame:
    frequency = Counter(
        ingredient for ingredients in recipes["ingredients"] for ingredient in ingredients
    )
    rare_ingredients = {ingredient for ingredient, count in frequency.items() if count < min_count}
    has_rare_ingredient = recipes["ingredients"].apply(lambda ingredients: not rare_ingredients.isdisjoint(ingredients))
    return recipes[~has_rare_ingredient]


def build_clean_dataset(force: bool = False) -> pd.DataFrame:
    if CLEAN_PATH.exists() and not force:
        recipes = pd.read_parquet(CLEAN_PATH)
        recipes["ingredients"] = recipes["ingredients"].apply(tuple)
        return recipes

    raw_path = download_raw()
    recipes = pd.read_parquet(raw_path)
    recipes["ingredients"] = recipes["ingredients"].apply(parse_ingredients)
    recipes = recipes[recipes["name"].str.strip().astype(bool)]
    recipes = recipes[recipes["ingredients"].apply(len) > 0]
    recipes = filter_rare_ingredients(recipes)
    recipes = recipes[["name", "url", "ingredients"]].reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    recipes.to_parquet(CLEAN_PATH)
    return recipes


if __name__ == "__main__":
    df = build_clean_dataset(force=True)
    print(f"Рецептов после очистки: {len(df)}")
    print(df.head(3))
