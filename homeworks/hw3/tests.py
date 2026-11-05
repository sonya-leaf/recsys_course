import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import numpy as np
import polars as pl

UTC = timezone.utc
W0 = datetime(2023, 1, 2, tzinfo=UTC)


def _week(n: int) -> datetime:
    return W0 + timedelta(weeks=n)


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u1", "u4"],
            "item_id": ["a", "a", "b", "a", "c", "b", "d", "e"],
            "rating": [5.0, 1.0, 2.0, 4.0, 3.0, 5.0, 4.0, 3.0],
            "timestamp": [
                W0 + timedelta(days=1),
                W0 + timedelta(days=2),
                _week(1) + timedelta(days=1),
                _week(1),
                _week(2),
                _week(2) + timedelta(days=1),
                _week(5),
                _week(6),
            ],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _ok() -> None:
    print("All good! :)")


def _assert_raises(exception: type[Exception], callback: Callable[[], Any]) -> None:
    try:
        callback()
    except exception:
        return
    raise AssertionError(f"Ожидалось исключение {exception.__name__}")


def _assert_callable_typed(function: Callable[..., Any]) -> None:
    signature = inspect.signature(function)
    assert (
        signature.return_annotation is not inspect.Signature.empty
    ), f"У {function.__qualname__} отсутствует тип возвращаемого значения"
    for parameter in signature.parameters.values():
        assert (
            parameter.annotation is not inspect.Signature.empty
        ), f"У параметра {function.__qualname__}.{parameter.name} отсутствует тип"


def _assert_typed(*functions: Callable[..., Any]) -> None:
    for function in functions:
        _assert_callable_typed(function)


def _materialize_aggs(
    session_fn: Callable[..., pl.DataFrame],
    raw_fn: Callable[..., None],
    aggs_fn: Callable[..., None],
    root: Path,
    events: pl.DataFrame,
) -> None:
    raw_fn(session_fn(events), root)
    aggs_fn(root)


# --------------------------------------------------------------------------
# Разделы 1-3. Сессии, временное разбиение и обучающие примеры
# --------------------------------------------------------------------------


def check_weekly_sessions(function: Callable[..., pl.DataFrame]) -> None:
    _assert_typed(function)
    data = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u1"],
            "item_id": ["a", "b", "c", "a", "d"],
            "rating": [1.0, 2.0, 3.0, 4.0, 5.0],
            "timestamp": [
                W0,
                W0 + timedelta(days=6, hours=23),
                _week(1),
                _week(1),
                datetime(2024, 1, 1, tzinfo=UTC),
            ],
        }
    )
    out = function(data)
    assert out["week_start"].to_list()[:4] == [W0, W0, _week(1), _week(1)]
    assert out["session_id"][0] == out["session_id"][1]
    assert out["session_id"][2] != out["session_id"][0]
    assert out["session_id"][2] != out["session_id"][3]
    assert out["week_start"][-1] == datetime(2024, 1, 1, tzinfo=UTC)
    _ok()


def check_temporal_week_split(
    session_fn: Callable[..., pl.DataFrame], function: Callable[..., tuple]
) -> None:
    _assert_typed(session_fn, function)
    data = pl.DataFrame(
        {
            "user_id": ["u"] * 20,
            "item_id": [str(i) for i in range(20)],
            "rating": [5.0] * 20,
            "timestamp": [_week(i // 2) for i in range(20)],
        }
    )
    data = session_fn(data)
    train, val, test = function(data)
    assert len(train) > len(val) and len(train) > len(test)
    assert len(train) + len(val) + len(test) == len(data)
    assert (
        train["week_start"].max() < val["week_start"].min() < test["week_start"].min()
    )
    assert not set(train["session_id"]) & set(val["session_id"])
    _assert_raises(ValueError, lambda: function(data, 0.8, 0.3, 0.1))
    _ok()


def check_positive_labels(
    session_fn: Callable[..., pl.DataFrame], function: Callable[..., pl.DataFrame]
) -> None:
    _assert_typed(session_fn, function)
    positive = function(session_fn(_events()))
    u1_a = positive.filter((pl.col("user_id") == "u1") & (pl.col("item_id") == "a"))
    assert len(u1_a) == 1 and u1_a["relevance"][0] == 5
    assert (
        positive.select(
            pl.struct(["session_id", "item_id"]).is_duplicated().any()
        ).item()
        is False
    )
    assert set(positive["binary_target"].to_list()) == {1}
    _ok()


# --------------------------------------------------------------------------
# Раздел 4. Datamart: Raw и Aggs
# --------------------------------------------------------------------------


def check_raw_layer(
    session_fn: Callable[..., pl.DataFrame], function: Callable[..., None]
) -> None:
    _assert_typed(session_fn, function)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        function(session_fn(_events()), root)
        paths = list((root / "raw" / "events").glob("week=*/events.parquet"))
        assert paths
        rows = [pl.read_parquet(path) for path in paths]
        assert sum(len(frame) for frame in rows) == len(_events())
        for path, df in zip(paths, rows):
            assert all(
                w.date().isoformat() == path.parent.name[5:] for w in df["week_start"]
            )
    _ok()


def check_weekly_counters(
    session_fn: Callable[..., pl.DataFrame], function: Callable[..., pl.DataFrame]
) -> None:
    _assert_typed(session_fn, function)
    weekly = session_fn(_events()).filter(pl.col("timestamp") < _week(1))
    counters = function(weekly)
    assert len(counters) == 1
    row = counters.row(0, named=True)
    assert row["n_events"] == 2 and row["rating_sum"] == 6
    assert row["rating_count"] == 2 and row["n_high_ratings"] == 1
    assert row["first_event_ts"] < row["last_event_ts"]
    _ok()


def check_aggs_layer(
    session_fn: Callable[..., pl.DataFrame],
    raw_fn: Callable[..., None],
    aggs_fn: Callable[..., None],
) -> None:
    _assert_typed(session_fn, raw_fn, aggs_fn)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_fn(session_fn(_events()), root)
        aggs_fn(root)
        aggs = [
            pl.read_parquet(p)
            for p in (root / "aggs" / "user_item").glob("week=*/counters.parquet")
        ]
        assert aggs and sum(int(x["n_events"].sum()) for x in aggs) == len(_events())
    _ok()


def check_weekly_counters_reader(
    session_fn: Callable[..., pl.DataFrame],
    raw_fn: Callable[..., None],
    aggs_fn: Callable[..., None],
    function: Callable[..., pl.DataFrame],
) -> None:
    _assert_typed(session_fn, raw_fn, aggs_fn, function)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _materialize_aggs(session_fn, raw_fn, aggs_fn, root, _events())
        assert sorted(function(root, _week(3))["week_start"].unique().to_list()) == [
            _week(0),
            _week(1),
            _week(2),
        ]
        assert sorted(
            function(root, _week(3), lookback_weeks=2)["week_start"]
            .unique()
            .to_list()
        ) == [_week(1), _week(2)]
        assert function(root, _week(0))["week_start"].to_list() == []
        _assert_raises(ValueError, lambda: function(root, _week(3), lookback_weeks=0))
        with TemporaryDirectory() as empty:
            _assert_raises(FileNotFoundError, lambda: function(Path(empty), _week(3)))
    _ok()


def check_item_categories(function: Callable[..., Any]) -> None:
    _assert_typed(function)
    items = pl.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4", "i5", "i6"],
            "title": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "categories": [
                ["Beauty", "Nail Polish"],
                ["Beauty", "Shampoos"],
                ["Beauty", "Shampoos"],
                ["   "],
                [],
                pl.Series(["Nail Polish"]),
            ],
        }
    )
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        mapping = function(items, root)
        assert isinstance(mapping, pl.DataFrame)
        assert mapping.schema == {
            "item_id": pl.String,
            "category": pl.String,
        }, "Ожидались колонки item_id/category типа String"
        assert sorted(mapping["item_id"].to_list()) == sorted(
            items["item_id"].to_list()
        ), "Нужна ровно одна строка на каждый товар"
        tag = dict(zip(mapping["item_id"], mapping["category"]))
        assert tag["i1"] == "Nail Polish" and tag["i6"] == "Nail Polish"
        assert tag["i4"] == "unknown" and tag["i5"] == "unknown"

        saved = pl.read_parquet(root / "common" / "item_categories.parquet")
        assert saved.sort("item_id").to_dicts() == mapping.sort("item_id").to_dicts()

        shuffled = function(items.sample(fraction=1.0, shuffle=True, seed=7), root)
        assert shuffled.sort("item_id").to_dicts() == mapping.sort("item_id").to_dicts()

        plain = pl.DataFrame({"item_id": ["i1"], "title": ["t1"]})
        assert function(plain, root) is None
    _ok()


# --------------------------------------------------------------------------
# Разделы 4-5. Point-in-time признаки и их присоединение
# --------------------------------------------------------------------------


def check_lifetime_features(
    session_fn: Callable[..., pl.DataFrame],
    raw_fn: Callable[..., None],
    aggs_fn: Callable[..., None],
    feature_fn: Callable[..., dict],
) -> None:
    _assert_typed(session_fn, raw_fn, aggs_fn, feature_fn)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _materialize_aggs(session_fn, raw_fn, aggs_fn, root, _events())
        features = feature_fn(root, _week(3))
        user = features["user"].filter(pl.col("user_id") == "u1").row(0, named=True)
        item = features["item"].filter(pl.col("item_id") == "a").row(0, named=True)
        item_b = features["item"].filter(pl.col("item_id") == "b").row(0, named=True)
        assert user["user_events_total"] == 3 and user["user_items_unique"] == 2
        assert np.isclose(user["user_mean_rating"], 8 / 3)
        assert item["item_events_total"] == 3 and item["item_users_unique"] == 2
        assert np.isclose(item["item_mean_rating"], 10 / 3)
        assert np.isclose(item_b["item_bayesian_mean_rating"], 3.3484848484848486)
    _ok()


def check_rolling_features(
    session_fn: Callable[..., pl.DataFrame],
    raw_fn: Callable[..., None],
    aggs_fn: Callable[..., None],
    feature_fn: Callable[..., dict],
) -> None:
    _assert_typed(session_fn, raw_fn, aggs_fn, feature_fn)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _materialize_aggs(session_fn, raw_fn, aggs_fn, root, _events())
        result = feature_fn(root, _week(6), lookback_weeks=4)
        users = result["user"]
        u1 = users.filter(pl.col("user_id") == "u1").row(0, named=True)
        assert u1["user_events_total"] == 4
        assert u1["user_events_4w"] == 1
        a = result["item"].filter(pl.col("item_id") == "a").row(0, named=True)
        assert a["item_events_4w"] == 0
    _ok()


def check_category_features(
    session_fn: Callable[..., pl.DataFrame],
    raw_fn: Callable[..., None],
    aggs_fn: Callable[..., None],
    feature_fn: Callable[..., dict],
) -> None:
    _assert_typed(session_fn, raw_fn, aggs_fn, feature_fn)
    categories = pl.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e"],
            "category": ["x", "x", "y", "y", "z"],
        }
    )
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _materialize_aggs(session_fn, raw_fn, aggs_fn, root, _events())
        metadata = root / "common" / "item_categories.parquet"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        categories.write_parquet(metadata)
        result = feature_fn(root, _week(6))
        assert {"user_category", "category"} <= set(result)
        category = result["category"].sort("category")
        x = category.filter(pl.col("category") == "x").row(0, named=True)
        y = category.filter(pl.col("category") == "y").row(0, named=True)
        assert x["category_events_total"] == 5 and x["category_events_4w"] == 1
        assert y["category_events_total"] == 2 and y["category_events_4w"] == 2
        u1_y = (
            result["user_category"]
            .filter((pl.col("user_id") == "u1") & (pl.col("category") == "y"))
            .row(0, named=True)
        )
        assert u1_y["user_category_events_total"] == 1
        assert u1_y["user_category_events_4w"] == 1
    _ok()


def check_feature_storage(
    session_fn: Callable[..., pl.DataFrame],
    raw_fn: Callable[..., None],
    aggs_fn: Callable[..., None],
    build_fn: Callable[..., None],
    feature_fn: Callable[..., dict],
) -> None:
    _assert_typed(session_fn, raw_fn, aggs_fn, build_fn, feature_fn)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _materialize_aggs(session_fn, raw_fn, aggs_fn, root, _events())
        snap = feature_fn(root, _week(2))
        build_fn(root, [_week(2)])
        for entity in ("user", "item", "global"):
            restored = pl.read_parquet(
                root / "features" / entity / "week=2023-01-16.parquet"
            )
            assert restored.shape == snap[entity].shape
    _ok()


def check_feature_join(
    session_fn: Callable[..., pl.DataFrame],
    raw_fn: Callable[..., None],
    aggs_fn: Callable[..., None],
    build_fn: Callable[..., None],
    join_fn: Callable[..., pl.DataFrame],
) -> None:
    _assert_typed(session_fn, raw_fn, aggs_fn, build_fn, join_fn)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _materialize_aggs(session_fn, raw_fn, aggs_fn, root, _events())
        build_fn(root, [_week(1), _week(3)])
        basis = pl.DataFrame(
            {
                "session_id": ["2023-01-09:u1", "2023-01-23:u1"],
                "week_start": [_week(1), _week(3)],
                "user_id": ["u1", "u1"],
                "item_id": ["a", "a"],
                "binary_target": [1, 0],
                "relevance": [5, 0],
            }
        )
        result = join_fn(basis, root).sort("week_start")
        assert len(result) == 2 and result["user_events_total"].to_list() == [2, 3]
        assert result["relevance"].to_list() == [5, 0]
        assert "user_item_rating_diff" in result.columns
        assert "item_popularity_share" in result.columns
    _ok()


# --------------------------------------------------------------------------
# Раздел 6. Метрики ранжирования
# --------------------------------------------------------------------------


def check_exp_ndcg_at_k(function: Callable[..., float]) -> None:
    _assert_typed(function)
    gt = {"a": 5, "b": 1, "c": 3}
    assert np.isclose(function(["a", "c", "b"], gt, 3), 1)
    assert np.isclose(function(["c", "a"], gt, 2), 0.7498995145002061)
    assert function(["z"], gt, 2) == 0
    assert function([], gt, 3) == 0
    _assert_raises(ValueError, lambda: function(["a", "a"], gt, 3))
    _ok()


def check_hit_rate_at_k(function: Callable[..., float]) -> None:
    _assert_typed(function)
    assert function(["b", "a"], {"a": 1}, 1) == 0
    assert function(["b", "a"], {"a": 1}, 2) == 1
    assert function([], {"a": 1}, 5) == 0
    _ok()


def check_mrr_at_k(function: Callable[..., float]) -> None:
    _assert_typed(function)
    assert function(["c", "b", "a"], {"a": 4}, 3) == 1 / 3
    assert function(["c", "b", "a"], {"a": 4}, 2) == 0
    assert function(["a", "b"], {"a": 2}, 3) == 1
    _ok()


def check_evaluate_ranking(function: Callable[..., dict[str, float]]) -> None:
    _assert_typed(function)
    truth = pl.DataFrame(
        {
            "session_id": ["s1", "s1", "s2"],
            "item_id": ["a", "b", "c"],
            "relevance": [5, 1, 4],
        }
    )
    recs = pl.DataFrame(
        {"session_id": ["s1", "s1"], "item_id": ["b", "a"], "position": [1, 2]}
    )
    result = function(recs, truth, 2)
    assert set(result) == {"ExpNDCG@2", "HitRate@2", "MRR@2"}
    assert result["HitRate@2"] == 0.5 and result["MRR@2"] == 0.5
    assert result["ExpNDCG@2"] < 0.5
    _ok()


# --------------------------------------------------------------------------
# Разделы 7-13. Группы, кандидаты, scoring и reranking
# --------------------------------------------------------------------------


def check_informative_ranking_groups(
    function: Callable[..., pl.DataFrame],
) -> None:
    _assert_typed(function)
    frame = pl.DataFrame(
        {
            "session_id": [
                "keep_a",
                "keep_a",
                "flat",
                "flat",
                "single",
                "keep_b",
                "keep_b",
                "keep_b",
            ],
            "item_id": ["a", "b", "a", "b", "a", "a", "b", "c"],
            "relevance": [0, 5, 0, 0, 3, 1, 1, 2],
        }
    )
    result = function(frame).sort(["session_id", "item_id"])
    expected = frame.filter(pl.col("session_id").is_in(["keep_a", "keep_b"])).sort(
        ["session_id", "item_id"]
    )
    assert result.columns == frame.columns
    assert result.to_dicts() == expected.to_dicts()
    _ok()


def check_ranking_groups(function: Callable[..., Any]) -> None:
    _assert_typed(function)
    ordered_wrong = pl.DataFrame(
        {
            "session_id": ["B", "A", "B", "A"],
            "item_id": ["x", "x", "y", "y"],
            "f": [1.0, 2.0, 3.0, 4.0],
            "relevance": [0, 5, 1, 0],
        }
    )
    pool = function(ordered_wrong, ["f"])
    assert pool.num_row() == 4 and pool.num_col() == 1
    hashes = np.asarray(pool.get_group_id_hash())
    assert hashes[0] == hashes[1] and hashes[2] == hashes[3] and hashes[0] != hashes[2]
    assert np.asarray(pool.get_label()).tolist() == [5, 0, 0, 1]
    _ok()


def check_candidate_deduplication(function: Callable[..., pl.DataFrame]) -> None:
    _assert_typed(function)
    first = pl.DataFrame(
        {
            "session_id": ["s", "s"],
            "week_start": [W0, W0],
            "user_id": ["u", "u"],
            "item_id": ["a", "b"],
            "position": [1, 2],
        }
    )
    second = pl.DataFrame(
        {
            "session_id": ["s", "s"],
            "week_start": [W0, W0],
            "user_id": ["u", "u"],
            "item_id": ["b", "c"],
            "position": [1, 2],
        }
    )
    merged = function({"Pop": first, "ALS": second})
    assert len(merged) == 3 and set(merged["item_id"]) == {"a", "b", "c"}
    assert {"session_id", "week_start", "user_id"} <= set(merged.columns)
    b = merged.filter(pl.col("item_id") == "b")
    assert len(b) == 1
    assert b["rank_Pop"][0] == 2 and b["rank_ALS"][0] == 1
    assert b["from_Pop"][0] == 1 and b["from_ALS"][0] == 1
    assert b["source_count"][0] == 2
    a = merged.filter(pl.col("item_id") == "a")
    assert a["rank_ALS"][0] == 101 and a["from_ALS"][0] == 0
    _ok()


def check_candidate_labels(function: Callable[..., pl.DataFrame]) -> None:
    _assert_typed(function)
    candidates = pl.DataFrame(
        {
            "session_id": ["s", "s", "s"],
            "week_start": [W0, W0, W0],
            "user_id": ["u", "u", "u"],
            "item_id": ["a", "b", "c"],
        }
    )
    truth = pl.DataFrame(
        {
            "session_id": ["s", "s"],
            "item_id": ["a", "c"],
            "relevance": [5, 2],
        }
    )
    result = function(candidates, truth).sort("item_id")
    assert result["relevance"].to_list() == [5, 0, 2]
    assert result["binary_target"].to_list() == [1, 0, 1]
    assert len(result) == len(candidates)
    _ok()


def check_informative_candidate_groups(
    function: Callable[..., pl.DataFrame],
) -> None:
    _assert_typed(function)
    frame = pl.DataFrame(
        {
            "session_id": ["keep", "keep", "no_positive", "no_positive", "single"],
            "item_id": ["a", "b", "a", "b", "a"],
            "binary_target": [1, 0, 0, 0, 1],
            "relevance": [5, 0, 0, 0, 4],
        }
    )
    result = function(frame).sort("item_id")
    expected = frame.filter(pl.col("session_id") == "keep").sort("item_id")
    assert result.columns == frame.columns
    assert result.to_dicts() == expected.to_dicts()
    _ok()


class _FakeClassifier:
    def __init__(self) -> None:
        self.seen_rows = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.seen_rows += len(X)
        score = np.clip(X[:, 0], 0, 1)
        return np.column_stack([1 - score, score])

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise AssertionError("Для классификатора требуется predict_proba")


class _FakeRanker:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0] * 10 - 1


def check_candidate_scoring(function: Callable[..., np.ndarray]) -> None:
    _assert_typed(function)
    frame = pl.DataFrame({"feature": [0.1, 0.9, 0.5]})
    classifier = _FakeClassifier()
    result = function(classifier, frame, "classifier", ["feature"])
    assert np.allclose(result, [0.1, 0.9, 0.5]) and classifier.seen_rows == 3
    assert np.allclose(function(_FakeRanker(), frame, "ranker", ["feature"]), [0, 8, 4])
    _assert_raises(
        ValueError, lambda: function(classifier, frame, "invalid", ["feature"])
    )
    _ok()


def check_rerank_candidates(function: Callable[..., pl.DataFrame]) -> None:
    _assert_typed(function)
    frame = pl.DataFrame(
        {
            "session_id": ["s", "s", "s"],
            "item_id": ["a", "c", "b"],
            "feature": [0.1, 0.8, 0.8],
        }
    )
    model = _FakeClassifier()
    result = function(model, frame, "classifier", ["feature"], k=2)
    assert result["item_id"].to_list() == ["b", "c"]
    assert result["position"].to_list() == [1, 2]
    assert model.seen_rows == 3
    duplicate = pl.concat([frame, frame.head(1)])
    _assert_raises(
        ValueError, lambda: function(model, duplicate, "classifier", ["feature"], k=2)
    )
    _ok()


def check_point_in_time_leakage(
    session_fn: Callable[..., pl.DataFrame],
    raw_fn: Callable[..., None],
    aggs_fn: Callable[..., None],
    feature_fn: Callable[..., dict],
) -> None:
    _assert_typed(session_fn, raw_fn, aggs_fn, feature_fn)
    base = _events()
    week = _week(2)
    extra = pl.DataFrame(
        {
            "user_id": ["u1"] * 5,
            "item_id": ["future"] * 5,
            "rating": [5.0] * 5,
            "timestamp": [_week(4)] * 5,
        }
    )
    with TemporaryDirectory() as tmp:
        original_root = Path(tmp) / "original"
        changed_root = Path(tmp) / "changed"
        _materialize_aggs(session_fn, raw_fn, aggs_fn, original_root, base)
        _materialize_aggs(
            session_fn, raw_fn, aggs_fn, changed_root, pl.concat([base, extra])
        )
        original = feature_fn(original_root, week)
        changed = feature_fn(changed_root, week)
    for entity in ("user", "item", "global"):
        assert (
            original[entity].sort(original[entity].columns[0]).to_dicts()
            == changed[entity].sort(changed[entity].columns[0]).to_dicts()
        )
    assert "future" not in changed["item"]["item_id"]
    _ok()
