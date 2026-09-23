import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_ingredient_matrix(recipes: pd.DataFrame):
    """Multi-hot представление рецептов: строки — рецепты, столбцы — ингредиенты."""
    vectorizer = CountVectorizer(analyzer=lambda tokens: tokens, binary=True)
    matrix = vectorizer.fit_transform(recipes["ingredients"])
    return vectorizer, matrix


class BaseIngredientRecommender:
    """Общая часть обоих рекомендателей: подготовка multi-hot матрицы, фильтр
    по исключённым ингредиентам и случайная подборка, если пользователь ещё
    ничего не выбрал. Конкретный способ ранжирования рецептов задают наследники
    в методе `score`."""

    def __init__(self, recipes: pd.DataFrame, vectorizer: CountVectorizer | None = None, matrix=None):
        self.recipes = recipes.reset_index(drop=True)
        if vectorizer is None or matrix is None:
            vectorizer, matrix = build_ingredient_matrix(self.recipes)
        self.vectorizer = vectorizer
        self.matrix = matrix

    @property
    def vocabulary(self) -> list[str]:
        return sorted(self.vectorizer.get_feature_names_out())

    def score(self, include: list[str]) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, include: list[str], exclude: list[str], top_k: int = 20) -> pd.DataFrame:
        if not include:
            return self.random_recipes(exclude, top_k)

        similarity = self.score(include)
        allowed = self.allowed_mask(exclude)
        candidates = self.recipes.assign(score=similarity)[allowed]
        candidates = candidates[candidates["score"] > 0]
        return candidates.sort_values("score", ascending=False).head(top_k)

    def random_recipes(self, exclude: list[str], top_k: int = 20) -> pd.DataFrame:
        candidates = self.recipes[self.allowed_mask(exclude)]
        sample_size = min(top_k, len(candidates))
        return candidates.sample(n=sample_size).assign(score=np.nan)

    def allowed_mask(self, exclude: list[str]) -> pd.Series:
        if not exclude:
            return pd.Series(True, index=self.recipes.index)
        exclude_vector = self.vectorizer.transform([exclude])
        overlap = self.matrix.dot(exclude_vector.T).toarray().ravel()
        return overlap == 0


class BagOfWordsRecommender(BaseIngredientRecommender):
    """Представляет рецепт как мешок ингредиентов (multi-hot вектор) и ищет
    ближайшие рецепты к запросу пользователя по косинусной близости."""

    def score(self, include: list[str]) -> np.ndarray:
        query_vector = self.vectorizer.transform([include])
        return cosine_similarity(query_vector, self.matrix)[0]


class EaseRecommender(BaseIngredientRecommender):
    """EASE (Embarrassingly Shallow AutoEncoder, Steck, 2019) поверх матрицы
    «рецепт — ингредиент»: рецепты играют роль пользователей, ингредиенты —
    роль items. Модель обучает линейную матрицу близости ингредиентов B по
    тому, как они реально сочетаются в рецептах датасета — в отличие от BoW,
    который просто считает пересечение множеств без какого-либо обучения."""

    def __init__(
        self,
        recipes: pd.DataFrame,
        vectorizer: CountVectorizer | None = None,
        matrix=None,
        reg: float = 300.0,
    ):
        super().__init__(recipes, vectorizer, matrix)
        self.item_similarity = self._fit(reg)

    def _fit(self, reg: float) -> np.ndarray:
        gram = (self.matrix.T @ self.matrix).toarray().astype(np.float64)
        gram[np.diag_indices_from(gram)] += reg
        inverse = np.linalg.inv(gram)
        item_similarity = inverse / (-np.diag(inverse))
        np.fill_diagonal(item_similarity, 0.0)
        return item_similarity

    def score(self, include: list[str]) -> np.ndarray:
        query_vector = self.vectorizer.transform([include]).toarray()[0]
        ingredient_scores = query_vector @ self.item_similarity
        return self.matrix.dot(ingredient_scores)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max нормализация в [0, 1]. BoW (косинусная близость) и EASE (произвольная
    билинейная форма) дают оценки в разных, несравнимых шкалах — без этого шага
    вес a в гибридном рекомендателе сводился бы к перемножению на разные
    константы, а не к содержательному смешиванию двух ранжирований."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class BlendedRecommender:
    """Взвешивает нормализованные оценки BoW и EASE: a * bow_score + (5 - a) * ease_score,
    где a выбирает пользователь в диапазоне [0, 5]. a=5 — чистый BoW, a=0 — чистый
    EASE. Переиспользует уже обученные BagOfWordsRecommender и EaseRecommender, ничего
    заново не обучает."""

    def __init__(self, bow: BagOfWordsRecommender, ease: EaseRecommender):
        self.bow = bow
        self.ease = ease
        self.recipes = bow.recipes

    def recommend(self, include: list[str], exclude: list[str], alpha: float, top_k: int = 20) -> pd.DataFrame:
        if not include:
            return self.bow.random_recipes(exclude, top_k)

        bow_score = normalize_scores(self.bow.score(include))
        ease_score = normalize_scores(self.ease.score(include))
        blended = alpha * bow_score + (5 - alpha) * ease_score

        allowed = self.bow.allowed_mask(exclude)
        candidates = self.recipes.assign(score=blended)[allowed]
        candidates = candidates[candidates["score"] > 0]
        return candidates.sort_values("score", ascending=False).head(top_k)
