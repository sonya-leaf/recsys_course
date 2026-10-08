import inspect
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse


def _ok() -> None:
    print("All good! :)")


_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _timestamps(days: Sequence[int]) -> list[datetime]:
    return [_EPOCH + timedelta(days=day) for day in days]


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "item_id": ["a", "b", "a", "c", "a", "d"],
            "rating": [5.0, 4.0, 5.0, 2.0, 3.0, 1.0],
            "timestamp": _timestamps([0, 1, 2, 3, 4, 5]),
        }
    )


def _assert_callable_typed(function: Callable[..., Any]) -> None:
    signature = inspect.signature(function)
    assert signature.return_annotation is not inspect.Signature.empty, (
        f"У {function.__qualname__} отсутствует тип возвращаемого значения"
    )
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        assert parameter.annotation is not inspect.Signature.empty, (
            f"У параметра {function.__qualname__}.{parameter.name} отсутствует тип"
        )


def _assert_typed(obj: Any) -> None:
    if inspect.isfunction(obj):
        _assert_callable_typed(obj)
        return
    if inspect.isclass(obj):
        assert getattr(obj, "__annotations__", {}), (
            f"У класса {obj.__qualname__} отсутствуют аннотации атрибутов"
        )
        for name, value in vars(obj).items():
            if inspect.isfunction(value) and (
                not name.startswith("_") or name == "__init__"
            ):
                _assert_callable_typed(value)


def _assert_recommendations(
    recommendations: Any,
    n_users: int,
    topn: int,
) -> None:
    """Проверяет общий контракт выдачи: list[list[str]] нужной длины без дублей."""
    assert isinstance(recommendations, list), (
        f"recommend должен вернуть list, а не {type(recommendations).__name__}"
    )
    assert len(recommendations) == n_users, (
        "Число списков рекомендаций должно совпадать с числом пользователей"
    )
    for user_recommendations in recommendations:
        assert isinstance(user_recommendations, list), (
            "Каждый элемент выдачи должен быть list[str]"
        )
        assert len(user_recommendations) <= topn, "Выдача длиннее topn"
        assert all(isinstance(item, str) for item in user_recommendations), (
            "Идентификаторы товаров должны быть строками"
        )
        assert len(set(user_recommendations)) == len(user_recommendations), (
            "В выдаче одного пользователя есть повторы"
        )


# --------------------------------------------------------------------------
# Раздел 1. Разведочный анализ данных
# --------------------------------------------------------------------------


def check_rating_distribution(function: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
    _assert_typed(function)
    result = function(_events())
    assert result.to_dict(as_series=False) == {
        "rating": [1, 2, 3, 4, 5],
        "n_events": [1, 1, 1, 1, 2],
        "event_share": [1 / 6, 1 / 6, 1 / 6, 1 / 6, 2 / 6],
    }
    _ok()


def check_user_activity_distribution(
    function: Callable[[pl.DataFrame], pl.DataFrame],
) -> None:
    _assert_typed(function)
    result = function(_events())
    assert result.to_dict(as_series=False) == {
        "activity": [2],
        "n_users": [3],
        "user_share": [1.0],
    }
    _ok()


def check_item_popularity_curve(
    function: Callable[[pl.DataFrame], pl.DataFrame],
) -> None:
    _assert_typed(function)
    result = function(_events())
    assert result.columns == [
        "item_id",
        "n_interactions",
        "percentile_rank",
        "cumulative_event_share",
    ]
    assert result["item_id"].to_list() == ["a", "b", "c", "d"]
    assert result["n_interactions"].to_list() == [3, 1, 1, 1]
    assert np.allclose(result["percentile_rank"].to_numpy(), [0.0, 1 / 3, 2 / 3, 1.0])
    assert np.allclose(
        result["cumulative_event_share"].to_numpy(), [0.5, 2 / 3, 5 / 6, 1.0]
    )
    _ok()


def check_events_by_month(function: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
    _assert_typed(function)
    frame = _events().with_columns(
        pl.Series(
            "timestamp",
            _timestamps([0, 1, 2, 3])
            + [
                datetime(2020, 3, 5, tzinfo=timezone.utc),
                datetime(2020, 3, 6, tzinfo=timezone.utc),
            ],
        )
    )
    result = function(frame)
    assert result.columns == ["month", "n_events"]
    assert result["n_events"].to_list() == [4, 0, 2]
    _ok()


# --------------------------------------------------------------------------
# Раздел 2. Временное разбиение
# --------------------------------------------------------------------------


def check_temporal_ratio_split(function: Callable[..., Any]) -> None:
    _assert_typed(function)
    frame = pl.DataFrame(
        {
            "row_id": list(range(100, 110)),
            "timestamp": _timestamps([8, 1, 4, 2, 9, 0, 3, 7, 6, 5]),
            "value": list(range(10)),
        }
    )
    train, validation, test = function(frame, 0.6, 0.2, 0.2)
    assert [len(train), len(validation), len(test)] == [6, 2, 2]
    assert (
        set(train["row_id"].to_list())
        | set(validation["row_id"].to_list())
        | set(test["row_id"].to_list())
    ) == set(frame["row_id"].to_list())
    assert train["timestamp"].max() <= validation["timestamp"].min()
    assert validation["timestamp"].max() <= test["timestamp"].min()
    invalid = [(0.8, 0.1, 0.2), (0.8, 0.2, 0.0), (-0.1, 0.2, 0.9)]
    for ratios in invalid:
        try:
            function(frame, *ratios)
            raise AssertionError(f"Ожидался ValueError для ratios={ratios}")
        except ValueError:
            pass
    _ok()


def check_cold_start_report(function: Callable[..., dict[str, float]]) -> None:
    _assert_typed(function)
    train = pl.DataFrame({"user_id": ["u1", "u2"], "item_id": ["a", "b"]})
    evaluation = pl.DataFrame({"user_id": ["u1", "u3"], "item_id": ["a", "c"]})
    assert function(train, evaluation) == {
        "cold_user_share": 0.5,
        "cold_item_share": 0.5,
    }
    _ok()


# --------------------------------------------------------------------------
# Раздел 3. Метрики
# --------------------------------------------------------------------------


def check_metric_data_helpers(
    validate_metric_inputs: Callable[..., None],
    build_target_lists: Callable[..., list[list[str]]],
) -> None:
    _assert_typed(validate_metric_inputs)
    _assert_typed(build_target_lists)

    validate_metric_inputs([["a"], []], [["a", "b"], ["c"]], 2)
    for recommendations, targets, k in [
        ([["a"]], [["a"], ["b"]], 2),
        ([["a"]], [["a"]], 0),
    ]:
        try:
            validate_metric_inputs(recommendations, targets, k)
            raise AssertionError(f"Ожидался ValueError для k={k}")
        except ValueError:
            pass

    frame = pl.concat([_events(), _events()[:1]])
    targets = build_target_lists(frame, ["u2", "u1", "u404"])
    assert targets == [["a", "c"], ["a", "b"], []], (
        "Таргеты должны идти в порядке user_ids, без дублей"
    )
    _ok()


def _metric_lists() -> tuple[list[list[str]], list[list[str]]]:
    recommendations = [
        ["item1", "item2", "item3", "item4", "item5"],
        ["item2", "item1", "item4", "item5", "item3"],
        ["item5", "item4", "item3", "item2", "item1"],
        ["item6", "item7", "item8", "item9", "item10"],
    ]
    targets = [
        ["item1", "item3"],
        ["item1", "item2", "item4"],
        ["item5"],
        [],
    ]
    return recommendations, targets


def check_recall_at_k(function: Callable[..., float]) -> None:
    _assert_typed(function)
    recommendations, targets = _metric_lists()
    assert np.isclose(function(recommendations, targets, 3), 0.75)
    _ok()


def check_hit_rate_at_k(function: Callable[..., float]) -> None:
    _assert_typed(function)
    recommendations, targets = _metric_lists()
    assert np.isclose(function(recommendations, targets, 3), 0.75)
    _ok()


def check_map_at_k(function: Callable[..., float]) -> None:
    _assert_typed(function)
    recommendations, targets = _metric_lists()
    assert np.isclose(function(recommendations, targets, 3), 17.0 / 24.0)
    _ok()


def check_mrr_at_k(function: Callable[..., float]) -> None:
    _assert_typed(function)
    recommendations, targets = _metric_lists()
    assert np.isclose(function(recommendations, targets, 3), 0.75)
    _ok()


def check_ndcg_at_k(function: Callable[..., float]) -> None:
    _assert_typed(function)
    recommendations, targets = _metric_lists()
    assert np.isclose(function(recommendations, targets, 3), 0.72993, atol=1e-5)
    _ok()


def check_coverage_at_k(function: Callable[..., float]) -> None:
    _assert_typed(function)
    recommendations, targets = _metric_lists()
    assert np.isclose(function(recommendations, targets, 10, n_items=20), 0.5)
    assert np.isclose(function(recommendations, targets, 3), 1.0)
    for invalid in [0, -1]:
        try:
            function(recommendations, targets, 3, n_items=invalid)
            raise AssertionError("Ожидался ValueError для неположительного n_items")
        except ValueError:
            pass
    _ok()


def check_evaluate(function: Callable[..., dict[str, float]]) -> None:
    _assert_typed(function)
    recommendations, targets = _metric_lists()
    result = function(recommendations, targets, (3, 5), 20)
    expected_keys = {
        f"{metric}@{cutoff}"
        for cutoff in (3, 5)
        for metric in ("recall", "hit_rate", "map", "mrr", "ndcg", "coverage")
    }
    assert set(result) == expected_keys
    assert np.isclose(result["recall@3"], 0.75)
    assert all(0.0 <= value <= 1.0 for value in result.values())
    try:
        function(recommendations, targets, (), 20)
        raise AssertionError("Ожидался ValueError для пустого списка cutoffs")
    except ValueError:
        pass
    _ok()


# --------------------------------------------------------------------------
# Раздел 4. Общий интерфейс и экспериментальный контур
# --------------------------------------------------------------------------


def check_top_n_indices(function: Callable[..., np.ndarray]) -> None:
    _assert_typed(function)
    scores = np.asarray(
        [
            [1.0, 4.0, -np.inf, 3.0],
            [5.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert function(scores, 2).tolist() == [[1, 3], [0, 1]], (
        "Индексы должны быть отсортированы по убыванию оценки"
    )
    # n больше числа столбцов — берём всё, что есть, и не падаем.
    assert function(scores, 10).shape == (2, 4)
    _ok()


def check_base_recommender(cls: type[Any]) -> None:
    _assert_typed(cls)

    class Concrete(cls):
        def _fit(self, interactions: pl.DataFrame, **kwargs: Any) -> None:
            pass

        def _recommend(self, user_ids: list[str], topn: int) -> list[list[str]]:
            return [self._remove_seen(user, self.popular_items, topn) for user in user_ids]

    model = Concrete("Concrete", show_progress=False)
    try:
        model.recommend(["u1"], 2)
        raise AssertionError("До fit метод recommend обязан бросать ValueError")
    except ValueError:
        pass

    model.fit(_events())
    assert sorted(model.items) == ["a", "b", "c", "d"]
    assert model.popular_items[0] == "a", "popular_items сортируются по числу событий"
    assert model.seen_items["u1"] == {"a", "b"}
    assert model.item_to_index[model.items[0]] == 0

    result = model.recommend(["u1", "cold"], 2)
    _assert_recommendations(result, 2, 2)
    assert not set(result[0]) & {"a", "b"}, "История пользователя должна быть исключена"
    assert result[1][0] == "a", "Холодный пользователь получает самый популярный товар"

    assert sorted(model._seen_indices("u1").tolist()) == sorted(
        [model.item_to_index["a"], model.item_to_index["b"]]
    )
    assert model._seen_indices("cold").tolist() == []

    try:
        model.recommend(["u1"], 0)
        raise AssertionError("Ожидался ValueError для неположительного topn")
    except ValueError:
        pass
    _ok()


def check_matrix_recommender(cls: type[Any]) -> None:
    _assert_typed(cls)

    class Concrete(cls):
        def _fit(self, interactions: pl.DataFrame, **kwargs: Any) -> None:
            self._create_interaction_matrix(interactions)

        def _score_batch(self, user_indices: np.ndarray) -> np.ndarray:
            # Оценка товара = его популярность, одинаковая для всех пользователей.
            popularity = np.asarray(self.interaction_matrix.sum(axis=0)).ravel()
            return np.tile(popularity, (len(user_indices), 1)).astype(np.float32)

    model = Concrete("Concrete", show_progress=False).fit(_events())
    matrix = model.interaction_matrix
    assert matrix.shape == (3, 4) and matrix.nnz == 6
    assert np.all(matrix.data == 1.0), "По умолчанию матрица должна быть бинарной"
    assert model.id_to_item[model.item_to_id["a"]] == "a"

    result = model.recommend(["u1", "cold"], 2)
    _assert_recommendations(result, 2, 2)
    assert not set(result[0]) & {"a", "b"}, "История должна быть замаскирована"
    assert result[1][0] == "a", "Холодный пользователь получает популярное"

    class Weighted(Concrete):
        def _fit(self, interactions: pl.DataFrame, **kwargs: Any) -> None:
            self._create_interaction_matrix(interactions, weight_column="rating")

    weighted = Weighted("Weighted", show_progress=False).fit(_events())
    row = weighted.interaction_matrix[weighted.user_to_id["u1"]].toarray().ravel()
    assert row[weighted.item_to_id["a"]] == 5.0, "weight_column должен задавать веса"
    assert row[weighted.item_to_id["b"]] == 4.0
    _ok()


# --------------------------------------------------------------------------
# Разделы 5-11. Модели
# --------------------------------------------------------------------------


def check_random_retriever(cls: type[Any]) -> None:
    _assert_typed(cls)
    model = cls(seed=17, show_progress=False).fit(_events())
    first = model.recommend(["u1", "cold"], 2)
    second = model.recommend(["u1", "cold"], 2)
    _assert_recommendations(first, 2, 2)
    assert first == second, "При фиксированном seed выдача должна быть воспроизводимой"
    assert not set(first[0]) & {"a", "b"}, "История пользователя должна быть исключена"
    assert cls(seed=1, show_progress=False).fit(_events()).recommend(
        ["u1"], 2
    ) != cls(seed=2, show_progress=False).fit(_events()).recommend(["u1"], 2), (
        "Разные seed должны давать разную выдачу"
    )
    _ok()


def check_top_popular_retriever(cls: type[Any]) -> None:
    _assert_typed(cls)
    model = cls(weighting="count", show_progress=False).fit(_events())
    result = model.recommend(["cold", "u2"], 2)
    _assert_recommendations(result, 2, 2)
    assert result[0][0] == "a", "Самый частый товар должен быть первым"
    assert not set(result[1]) & {"a", "c"}, "История пользователя должна быть исключена"

    # weighting="rating": b (4.0) обгоняет c (2.0) и d (1.0), но не a (5+5+3=13).
    weighted = cls(weighting="rating", show_progress=False).fit(_events())
    assert weighted.recommend(["cold"], 2)[0] == ["a", "b"]
    _ok()


def check_decay_popularity_retriever(cls: type[Any]) -> None:
    _assert_typed(cls)
    frame = pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item_id": ["old", "old", "new"],
            "rating": [5.0, 5.0, 5.0],
            "timestamp": _timestamps([0, 0, 10]),
        }
    )
    fresh = cls(half_life_days=1.0, show_progress=False).fit(frame).recommend(["cold"], 2)
    assert fresh[0][0] == "new", (
        "При коротком периоде полураспада свежий товар должен обгонять старый"
    )
    stale = (
        cls(half_life_days=10_000.0, show_progress=False)
        .fit(frame)
        .recommend(["cold"], 2)
    )
    assert stale[0][0] == "old", (
        "При очень длинном периоде полураспада побеждает более частый товар"
    )
    _ok()


def check_ease_weights(function: Callable[..., np.ndarray]) -> None:
    _assert_typed(function)
    matrix = sparse.csr_matrix(
        np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    )
    weights = function(matrix, 10.0)
    assert weights.shape == (3, 3)
    assert np.allclose(np.diag(weights), 0.0), "Диагональ B обязана быть нулевой"
    assert np.isfinite(weights).all()
    # Товары 0 и 1 встречаются вместе чаще, чем 1 и 2, значит связь сильнее.
    assert weights[0, 1] > weights[2, 1]
    _ok()


def check_ease_retriever(cls: type[Any]) -> None:
    _assert_typed(cls)
    model = cls(regularization=10.0, show_progress=False).fit(_events())
    result = model.recommend(["u1", "cold"], 2)
    _assert_recommendations(result, 2, 2)
    assert not set(result[0]) & {"a", "b"}, "История пользователя должна быть исключена"
    assert result[1][0] == "a", "Холодный пользователь получает популярное"
    _ok()


class _FakeImplicitModel:
    """Заглушка implicit-модели: запоминает аргументы и отдаёт детерминированные факторы."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, matrix: sparse.csr_matrix, **kwargs: Any) -> None:
        self.fit_matrix = matrix.copy()
        n_users, n_items = matrix.shape
        factors = int(self.kwargs.get("factors", 2))
        self.user_factors = np.ones((n_users, factors), dtype=np.float32)
        # Первый товар получает наибольшую оценку, последний — наименьшую.
        self.item_factors = np.tile(
            np.linspace(1.0, 0.0, n_items, dtype=np.float32)[:, None] / factors,
            (1, factors),
        )


def _check_implicit_subclass(
    cls: type[Any],
    *,
    has_alpha: bool,
) -> None:
    fake_module = types.SimpleNamespace(
        als=types.SimpleNamespace(
            AlternatingLeastSquares=_FakeImplicitModel
        ),
        bpr=types.SimpleNamespace(
            BayesianPersonalizedRanking=_FakeImplicitModel
        ),
    )

    previous_module = sys.modules.get("implicit")
    function_globals = cls._fit.__globals__
    previous_global = function_globals.get("implicit")

    sys.modules["implicit"] = fake_module
    function_globals["implicit"] = fake_module

    try:
        kwargs = dict(
            factors=4,
            iterations=2,
            show_progress=False,
        )

        if has_alpha:
            kwargs["alpha"] = 3.7

        model = cls(**kwargs).fit(_events())

        assert np.all(model.model.fit_matrix.data == 1.0), (
            "В implicit должна передаваться бинарная матрица взаимодействий."
        )

        if has_alpha:
            assert np.isclose(model.model.kwargs.get("alpha"), 4.7), (
                "Для бинарной матрицы нужно передать в implicit "
                "alpha_library = 1 + alpha."
            )
        else:
            assert "alpha" not in model.model.kwargs, (
                "BPR не использует alpha."
            )

        result = model.recommend(["u1", "cold"], 2)
        _assert_recommendations(result, 2, 2)
        assert not set(result[0]) & {"a", "b"}, (
            "История должна быть исключена"
        )

    finally:
        if previous_module is None:
            sys.modules.pop("implicit", None)
        else:
            sys.modules["implicit"] = previous_module

        if previous_global is None:
            function_globals.pop("implicit", None)
        else:
            function_globals["implicit"] = previous_global

    _ok()


def check_als_retriever(cls: type[Any]) -> None:
    _assert_typed(cls)
    _check_implicit_subclass(cls, has_alpha=True)


def check_bpr_retriever(cls: type[Any]) -> None:
    _assert_typed(cls)
    _check_implicit_subclass(cls, has_alpha=False)


def check_encode_titles(function: Callable[..., np.ndarray]) -> None:
    _assert_typed(function)

    class Encoder:
        def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
            return np.asarray([[3.0, 4.0]] * len(texts), dtype=np.float32)

    embeddings = function(["a", "b"], "unused", Encoder(), False)
    assert embeddings.shape == (2, 2)
    assert embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0), (
        "Векторы должны быть нормированы: тогда косинус — это скалярное произведение"
    )
    _ok()


def check_text_itemknn(cls: type[Any]) -> None:
    _assert_typed(cls)

    class Encoder:
        def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
            return np.asarray(
                [[1.0, 0.0] if "rock" in text else [0.0, 1.0] for text in texts],
                dtype=np.float32,
            )

    item_frame = pl.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "title": ["rock one", "rock two", "jazz one", "jazz two"],
        }
    )
    model = cls(weighting="uniform", encoder=Encoder(), show_progress=False).fit(
        _events(), items=item_frame
    )
    result = model.recommend(["u1", "cold"], 2)
    _assert_recommendations(result, 2, 2)
    assert not set(result[0]) & {"a", "b"}, "История пользователя должна быть исключена"
    # u2 слушал a (rock) и c (jazz); u3 слушал a (rock) и d (jazz).
    # У u1 профиль чисто "rock", поэтому из оставшихся c и d порядок не важен,
    # но профиль обязан быть нормированным и конечным.
    assert np.isfinite(model.user_profiles).all()
    assert np.allclose(
        np.linalg.norm(model.user_profiles[model.user_to_id["u1"]]), 1.0, atol=1e-5
    )
    _ok()


# --------------------------------------------------------------------------
# Раздел 12. Transformer
# --------------------------------------------------------------------------


def check_sequence_data(
    build_sequence_examples: Callable[..., Any],
    dataset_cls: type[Any],
    collate_fn: Callable[..., Any],
) -> None:
    import torch

    _assert_typed(build_sequence_examples)
    _assert_typed(dataset_cls)
    _assert_typed(collate_fn)

    examples = build_sequence_examples([[1, 2, 3, 4], [5]], 2, False)
    assert examples == [([1], 2), ([1, 2], 3), ([2, 3], 4)], (
        "Префикс обрезается слева до max_len, история длины 1 примеров не даёт"
    )

    dataset = dataset_cls(examples)
    assert len(dataset) == 3 and dataset[0] == ([1], 2)

    batch = collate_fn([([1, 2], 3), ([4], 5)])
    assert set(batch) == {"input_ids", "attention_mask", "lengths", "targets"}
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2], [4, 0]])), (
        "Короткие последовательности дополняются PAD=0 справа"
    )
    assert torch.equal(
        batch["attention_mask"], torch.tensor([[True, True], [True, False]])
    )
    assert torch.equal(batch["lengths"], torch.tensor([2, 1]))
    assert torch.equal(batch["targets"], torch.tensor([3, 5]))
    _ok()


def check_causal_self_attention(cls: type[Any]) -> None:
    import torch

    _assert_typed(cls)
    torch.manual_seed(0)
    attention = cls(d_model=8, n_heads=2, dropout=0.0).eval()

    x = torch.randn(2, 4, 8)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]], dtype=torch.bool
    )
    with torch.no_grad():
        output = attention(x, mask)
    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all(), "В выходе не должно быть NaN/inf"

    # Каузальность: изменение будущей позиции не влияет на прошлые.
    full_mask = torch.ones(1, 4, dtype=torch.bool)
    base = torch.randn(1, 4, 8)
    changed = base.clone()
    changed[:, -1, :] += 100.0
    with torch.no_grad():
        assert torch.allclose(
            attention(base, full_mask)[:, :-1], attention(changed, full_mask)[:, :-1],
            atol=1e-5,
        ), "Изменение будущего токена изменило прошлые позиции — каузальность нарушена"

    # Padding: изменение замаскированной позиции не влияет на валидные.
    padded_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    changed = base.clone()
    changed[:, -1, :] += 100.0
    with torch.no_grad():
        assert torch.allclose(
            attention(base, padded_mask)[:, :3],
            attention(changed, padded_mask)[:, :3],
            atol=1e-5,
        ), "Padding-позиция повлияла на валидные — маска применена неверно"
    _ok()


def check_feed_forward(cls: type[Any]) -> None:
    import torch

    _assert_typed(cls)
    module = cls(d_model=8, dropout=0.0).eval()
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        output = module(x)
    assert output.shape == (2, 3, 8)
    hidden = [
        layer.out_features
        for layer in module.modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    assert hidden == [32, 8], "Ожидается два линейных слоя: d -> 4d и 4d -> d"

    # Position-wise: каждая позиция обрабатывается независимо от соседей.
    changed = x.clone()
    changed[:, -1] += 10.0
    with torch.no_grad():
        assert torch.allclose(module(x)[:, :-1], module(changed)[:, :-1], atol=1e-6), (
            "MLP не должен смешивать информацию между позициями"
        )
    _ok()


def check_transformer_block(cls: type[Any]) -> None:
    import torch

    _assert_typed(cls)
    torch.manual_seed(0)
    block = cls(d_model=8, n_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        output = block(x, mask)
    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all()

    # Pre-LayerNorm с residual: занулив обе ветви, получаем тождественное отображение.
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
        assert torch.allclose(block(x, mask), x, atol=1e-6), (
            "Ожидается residual-связь: x + Attn(LN(x)), затем x + FF(LN(x))"
        )
    _ok()


def check_next_item_transformer(cls: type[Any]) -> None:
    import torch

    _assert_typed(cls)
    torch.manual_seed(0)
    model = cls(
        n_items=7, max_len=4, d_model=8, n_heads=2, n_layers=1, dropout=0.0
    ).eval()

    input_ids = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]])
    mask = input_ids.ne(0)
    lengths = torch.tensor([3, 1])

    with torch.no_grad():
        logits = model(input_ids, mask, lengths)

    assert logits.shape == (2, 7), (
        "Prediction head должен выдавать по одному логиту "
        "на каждый реальный товар; PAD=0 не является целевым классом"
    )
    assert torch.isfinite(logits).all()

    padded = torch.tensor([[4, 0, 0, 0], [4, 0, 0, 0]])
    with torch.no_grad():
        padded_logits = model(
            padded,
            padded.ne(0),
            torch.tensor([1, 1]),
        )

    assert torch.allclose(logits[1], padded_logits[0], atol=1e-5)
    _ok()


def check_next_item_loss(function: Callable[..., Any]) -> None:
    import torch

    _assert_typed(function)

    logits = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 0.0, 10.0],
    ])

    correct = function(logits, torch.tensor([1, 3]))
    wrong = function(logits, torch.tensor([3, 1]))

    assert correct.ndim == 0 and torch.isfinite(correct)
    assert wrong.ndim == 0 and torch.isfinite(wrong)

    assert correct < wrong, (
        "Лосс должен быть меньше, когда максимальный логит "
        "соответствует целевому товару"
    )

    assert correct < 1e-2, (
        "Уверенные правильные предсказания должны давать малый лосс"
    )

    _ok()


def check_train_epoch(function: Callable[..., float]) -> None:
    import torch

    _assert_typed(function)

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 4)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            rows = torch.arange(len(input_ids))
            return self.embedding(input_ids[rows, lengths - 1])

    model = Tiny()
    before = model.embedding.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {
        "input_ids": torch.tensor([[1, 2], [2, 0]]),
        "attention_mask": torch.tensor([[True, True], [True, False]]),
        "lengths": torch.tensor([2, 1]),
        "targets": torch.tensor([3, 1]),
    }
    loss = function(model, [batch], optimizer, torch.device("cpu"), "test", False)
    assert np.isfinite(loss), "Функция должна возвращать среднее значение лосса"
    assert not torch.allclose(before, model.embedding.weight), (
        "После эпохи веса модели обязаны измениться"
    )
    _ok()


def check_transformer_retriever(cls: type[Any]) -> None:
    _assert_typed(cls)
    model = cls(
        max_len=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=2,
        device="cpu",
        show_progress=False,
    ).fit(_events())
    result = model.recommend(["u1", "cold"], 2)
    _assert_recommendations(result, 2, 2)
    assert not set(result[0]) & {"a", "b"}, "История пользователя должна быть исключена"
    assert result[1][0] == "a", "Холодный пользователь получает популярное"
    assert model.histories["u1"] == [
        model.item_to_token["a"],
        model.item_to_token["b"],
    ], "История должна быть упорядочена по времени и переведена в токены"
    assert 0 not in model.item_to_token.values(), "Токен 0 зарезервирован под PAD"
    _ok()
