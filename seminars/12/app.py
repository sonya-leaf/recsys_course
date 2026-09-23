import pandas as pd
import streamlit as st

from src.prepare_data import build_clean_dataset
from src.recommender import BagOfWordsRecommender, BlendedRecommender, EaseRecommender, build_ingredient_matrix

st.set_page_config(page_title="Рецепты «Поварёнок»", layout="wide")


@st.cache_data
def load_recipes():
    return build_clean_dataset()


@st.cache_resource
def load_recommenders(recipes):
    vectorizer, matrix = build_ingredient_matrix(recipes)
    bow = BagOfWordsRecommender(recipes, vectorizer, matrix)
    ease = EaseRecommender(recipes, vectorizer, matrix)
    blended = BlendedRecommender(bow, ease)
    return bow, ease, blended


def render_results(results: pd.DataFrame) -> None:
    for _, recipe in results.iterrows():
        with st.container(border=True):
            title = f"**[{recipe['name']}]({recipe['url']})**"
            if pd.notna(recipe["score"]):
                title += f" — оценка {recipe['score']:.2f}"
            st.markdown(title)
            st.caption(", ".join(recipe["ingredients"]))


st.title("🍲 Рекомендации рецептов «Поварёнок»")
st.write(
    "Каждый рецепт представлен как совокупность ингредиентов. Выберите, какие "
    "ингредиенты должны быть в рецепте, а какие — исключить. Сравниваем три "
    "подхода: косинусную близость на основе Bag of Words, EASE — линейную "
    "модель, которая учитывает, какие ингредиенты сочетаются в "
    "рецептах датасета, — и их взвешенную комбинацию. "
    "Пока ничего не выбрано — показываем случайную подборку."
)

recipes = load_recipes()
bow_recommender, ease_recommender, blended_recommender = load_recommenders(recipes)

col1, col2 = st.columns(2)
with col1:
    include = st.multiselect("Включить ингредиенты", options=bow_recommender.vocabulary)
with col2:
    exclude = st.multiselect("Исключить ингредиенты", options=bow_recommender.vocabulary)

slider_col1, slider_col2 = st.columns([2, 1])
with slider_col1:
    top_k = st.slider("Сколько рецептов показать", min_value=5, max_value=20, value=10)
with slider_col2:
    alpha = st.slider(
        "Вес BoW (a)",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=1.0,
        help="Гибрид: a · BoW + (5 − a) · EASE",
    )

if not include:
    bow_results = ease_results = blended_results = bow_recommender.random_recipes(exclude, top_k)
else:
    bow_results = bow_recommender.recommend(include, exclude, top_k=top_k)
    ease_results = ease_recommender.recommend(include, exclude, top_k=top_k)
    blended_results = blended_recommender.recommend(include, exclude, alpha=alpha, top_k=top_k)

left, middle, right = st.columns(3)
with left:
    st.subheader("Bag of Words")
    st.caption(f"Найдено рецептов: {len(bow_results)} из {len(recipes)}")
    render_results(bow_results)
with middle:
    st.subheader("EASE")
    st.caption(f"Найдено рецептов: {len(ease_results)} из {len(recipes)}")
    render_results(ease_results)
with right:
    st.subheader(f"Гибрид (a = {alpha:g})")
    st.caption(f"Найдено рецептов: {len(blended_results)} из {len(recipes)}")
    render_results(blended_results)