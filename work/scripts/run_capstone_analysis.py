from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFolder, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO = "FlyRank/internship-warehouse"
MONTHS = ("2025-08", "2025-09", "2025-10")
DAILY_COLUMNS = [
    "report_date",
    "client_hash_id",
    "content_hash_id",
    "gsc_data_available",
    "ga4_data_available",
    "gsc_impressions",
    "gsc_clicks",
    "gsc_sum_position",
    "ga4_pageviews",
    "ga4_sessions",
    "ga4_engaged_sessions",
    "sessions_ai",
    "scroll_events",
]


def repo_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "raw" / "content_refresh_anonymized.csv").exists():
            return candidate
    raise FileNotFoundError("Run this from inside the starter repo.")


ROOT = repo_root()
OUT_DIR = ROOT / "work" / "outputs"
FIG_DIR = ROOT / "work" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def get_token() -> str:
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("No cached Hugging Face token found. Run `huggingface-cli login` first.")
    return token


def precision_at_k(scores, labels, k: int) -> float:
    labels = np.asarray(labels)
    order = np.argsort(-np.asarray(scores))[: min(k, len(labels))]
    return float(labels[order].mean())


def aggregate_month(month: str, token: str) -> pd.DataFrame:
    cache_path = OUT_DIR / f"warehouse_month_{month}.csv"
    if cache_path.exists():
        print(f"Loading cached aggregate: {cache_path.relative_to(ROOT)}", flush=True)
        return pd.read_csv(cache_path)

    filename = f"fact_content_daily_performance/month={month}/data_0.parquet"
    print(f"Aggregating warehouse month {month}", flush=True)

    local_path = hf_hub_download(REPO, repo_type="dataset", filename=filename, token=token)
    parquet_file = pq.ParquetFile(local_path)
    print(f"{month}: downloaded {parquet_file.metadata.num_rows:,} rows", flush=True)
    frame = pd.read_parquet(local_path, columns=DAILY_COLUMNS)
    frame["days_with_impressions"] = (frame["gsc_impressions"] > 0).astype("int16")
    frame["days_with_sessions"] = (frame["ga4_sessions"] > 0).astype("int16")
    frame["gsc_days_available"] = frame["gsc_data_available"].fillna(False).astype("int16")
    frame["ga4_days_available"] = frame["ga4_data_available"].fillna(False).astype("int16")
    month_df = frame.groupby(["client_hash_id", "content_hash_id"], as_index=False).agg(
        days=("report_date", "nunique"),
        gsc_days_available=("gsc_days_available", "sum"),
        ga4_days_available=("ga4_days_available", "sum"),
        impressions=("gsc_impressions", "sum"),
        clicks=("gsc_clicks", "sum"),
        sum_position=("gsc_sum_position", "sum"),
        pageviews=("ga4_pageviews", "sum"),
        sessions=("ga4_sessions", "sum"),
        engaged_sessions=("ga4_engaged_sessions", "sum"),
        sessions_ai=("sessions_ai", "sum"),
        scroll_events=("scroll_events", "sum"),
        days_with_impressions=("days_with_impressions", "sum"),
        days_with_sessions=("days_with_sessions", "sum"),
    )
    month_df["avg_position"] = np.where(
        month_df["impressions"] > 0,
        month_df["sum_position"] / month_df["impressions"],
        np.nan,
    )
    month_df["ctr"] = np.where(
        month_df["impressions"] > 0,
        month_df["clicks"] / month_df["impressions"] * 100,
        0.0,
    )
    month_df["engagement_rate"] = np.where(
        month_df["sessions"] > 0,
        month_df["engaged_sessions"] / month_df["sessions"] * 100,
        0.0,
    )
    month_df["scroll_rate"] = np.where(
        month_df["pageviews"] > 0,
        month_df["scroll_events"] / month_df["pageviews"] * 100,
        0.0,
    )
    month_df["ai_traffic_pct"] = np.where(
        month_df["sessions"] > 0,
        month_df["sessions_ai"] / month_df["sessions"] * 100,
        0.0,
    )
    month_df.to_csv(cache_path, index=False)
    print(f"Wrote {cache_path.relative_to(ROOT)} with {len(month_df):,} rows", flush=True)
    return month_df


def build_feature_table(token: str) -> pd.DataFrame:
    month_data = {month: aggregate_month(month, token) for month in MONTHS}
    apr = month_data["2025-08"]
    may = month_data["2025-09"]
    jun = month_data["2025-10"]

    dim_content_path = hf_hub_download(
        REPO, repo_type="dataset", filename="dim_content.parquet", token=token
    )
    dim_clients_path = hf_hub_download(
        REPO, repo_type="dataset", filename="dim_clients.parquet", token=token
    )
    dim_cols = [
        "client_hash_id",
        "content_hash_id",
        "content_created_date",
        "content_updated_date",
        "content_type",
        "search_volume",
        "competition",
        "competition_level",
        "cpc",
        "main_intent",
        "backlinks",
        "category_count",
        "char_count",
        "word_count",
        "is_published",
        "is_deleted",
    ]
    dim = pd.read_parquet(dim_content_path, columns=dim_cols)
    clients = pd.read_parquet(dim_clients_path)

    feature = may.add_suffix("_may").rename(
        columns={"client_hash_id_may": "client_hash_id", "content_hash_id_may": "content_hash_id"}
    )
    apr_features = apr.add_suffix("_apr").rename(
        columns={"client_hash_id_apr": "client_hash_id", "content_hash_id_apr": "content_hash_id"}
    )
    jun_target = (
        jun[["client_hash_id", "content_hash_id", "impressions", "clicks", "sessions"]]
        .add_suffix("_jun")
        .rename(
            columns={"client_hash_id_jun": "client_hash_id", "content_hash_id_jun": "content_hash_id"}
        )
    )

    feature = feature.merge(apr_features, on=["client_hash_id", "content_hash_id"], how="left")
    feature = feature.merge(jun_target, on=["client_hash_id", "content_hash_id"], how="left")
    feature = feature.merge(dim, on=["client_hash_id", "content_hash_id"], how="left")
    feature = feature.merge(
        clients[["client_hash_id", "access_profile", "gsc_data_start", "ga4_data_start"]],
        on="client_hash_id",
        how="left",
    )

    fill_zero = [
        "impressions_apr",
        "clicks_apr",
        "sessions_apr",
        "ctr_apr",
        "avg_position_apr",
        "impressions_jun",
        "clicks_jun",
        "sessions_jun",
    ]
    for col in fill_zero:
        if col in feature.columns:
            feature[col] = feature[col].fillna(0)

    feature["feature_month"] = "2025-09"
    feature["target_month"] = "2025-10"
    feature["future_decline_label"] = (
        feature["impressions_jun"] < 0.8 * feature["impressions_may"]
    ).astype(int)
    feature["imp_change_apr_to_may"] = np.where(
        feature["impressions_apr"] > 0,
        (feature["impressions_may"] - feature["impressions_apr"]) / feature["impressions_apr"],
        0.0,
    )
    feature["may_to_jun_change"] = np.where(
        feature["impressions_may"] > 0,
        (feature["impressions_jun"] - feature["impressions_may"]) / feature["impressions_may"],
        0.0,
    )

    ref_date = pd.Timestamp("2025-09-30")
    feature["content_created_date"] = pd.to_datetime(feature["content_created_date"])
    feature["content_updated_date"] = pd.to_datetime(feature["content_updated_date"])
    feature["content_age_days"] = (ref_date - feature["content_created_date"]).dt.days
    feature["days_since_update"] = (
        ref_date - feature["content_updated_date"].fillna(feature["content_created_date"])
    ).dt.days

    eligible = feature[
        (feature["impressions_may"] >= 100)
        & (feature["gsc_days_available_may"] > 0)
        & (feature["is_published"].fillna(True))
        & (~feature["is_deleted"].fillna(False))
    ].copy()
    eligible.to_csv(OUT_DIR / "capstone_feature_table.csv", index=False)
    return eligible


def train_and_score(eligible: pd.DataFrame) -> dict:
    numeric_features = [
        "impressions_may",
        "clicks_may",
        "ctr_may",
        "avg_position_may",
        "days_with_impressions_may",
        "sessions_may",
        "engagement_rate_may",
        "scroll_rate_may",
        "sessions_ai_may",
        "impressions_apr",
        "clicks_apr",
        "ctr_apr",
        "avg_position_apr",
        "imp_change_apr_to_may",
        "search_volume",
        "competition",
        "cpc",
        "backlinks",
        "category_count",
        "char_count",
        "word_count",
        "content_age_days",
        "days_since_update",
    ]
    categorical_features = ["content_type", "competition_level", "main_intent", "access_profile"]

    eligible[numeric_features] = eligible[numeric_features].replace([np.inf, -np.inf], np.nan)
    y = eligible["future_decline_label"].astype(int)
    groups = eligible["client_hash_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(eligible, y, groups))
    train = eligible.iloc[train_idx].copy()
    test = eligible.iloc[test_idx].copy()
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []
    fitted = {}
    x_train = train[numeric_features + categorical_features]
    x_test = test[numeric_features + categorical_features]
    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        pipe.fit(x_train, y_train)
        proba = pipe.predict_proba(x_test)[:, 1]
        fitted[name] = pipe
        results.append(
            {
                "method": name,
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "avg_precision": float(average_precision_score(y_test, proba)),
                "precision_at_50": precision_at_k(proba, y_test, 50),
                "precision_at_100": precision_at_k(proba, y_test, 100),
            }
        )

    valid = test["avg_position_may"].fillna(0).between(1, 20, inclusive="both")
    visible = test["impressions_may"] >= 500
    low_ctr = test["ctr_may"] < 0.5
    rule_match = valid & visible & low_ctr
    position_factor = (21 - test["avg_position_may"].fillna(99)).clip(lower=1, upper=20) / 20
    ctr_gap = (0.5 - test["ctr_may"].fillna(0)).clip(lower=0) / 0.5
    volume_factor = np.log1p(test["impressions_may"]) / np.log1p(eligible["impressions_may"].max())
    baseline_score = (rule_match.astype(int) * 100 * position_factor * ctr_gap * volume_factor).to_numpy()
    results.append(
        {
            "method": "baseline_visible_low_ctr",
            "roc_auc": float(roc_auc_score(y_test, baseline_score)),
            "avg_precision": float(average_precision_score(y_test, baseline_score)),
            "precision_at_50": precision_at_k(baseline_score, y_test, 50),
            "precision_at_100": precision_at_k(baseline_score, y_test, 100),
        }
    )

    results_df = pd.DataFrame(results).sort_values("precision_at_50", ascending=False)
    best_name = "random_forest"
    best_pipe = fitted[best_name]
    preprocessor_fitted = best_pipe.named_steps["pre"]
    feature_names = preprocessor_fitted.get_feature_names_out()
    importances = best_pipe.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    all_proba = best_pipe.predict_proba(eligible[numeric_features + categorical_features])[:, 1]
    eligible["model_probability"] = all_proba
    valid_all = eligible["avg_position_may"].fillna(0).between(1, 20, inclusive="both")
    visible_all = eligible["impressions_may"] >= 500
    low_ctr_all = eligible["ctr_may"] < 0.5
    baseline_all = (
        (valid_all & visible_all & low_ctr_all).astype(int)
        * 100
        * ((21 - eligible["avg_position_may"].fillna(99)).clip(lower=1, upper=20) / 20)
        * ((0.5 - eligible["ctr_may"].fillna(0)).clip(lower=0) / 0.5)
        * (np.log1p(eligible["impressions_may"]) / np.log1p(eligible["impressions_may"].max()))
    )
    eligible["baseline_action_score"] = baseline_all.round(2)
    eligible["final_action_score"] = (
        100 * (0.7 * eligible["model_probability"] + 0.3 * (eligible["baseline_action_score"] / 100))
    ).round(2)

    def reason_codes(row: pd.Series) -> str:
        reasons = []
        if row["model_probability"] >= 0.65:
            reasons.append("future_decline_risk")
        if row["model_probability"] >= 0.65 and row["avg_position_may"] > 50 and row["clicks_may"] == 0:
            reasons.append("deep_zero_click_risk")
        if (
            row["impressions_may"] >= 500
            and 1 <= (row["avg_position_may"] if pd.notna(row["avg_position_may"]) else 999) <= 20
            and row["ctr_may"] < 0.5
        ):
            reasons.append("visible_low_ctr_page")
        if row["sessions_may"] >= 30 and row["engagement_rate_may"] < 20:
            reasons.append("weak_engagement_context")
        if row["impressions_may"] >= 3000:
            reasons.append("high_volume_page")
        return ";".join(reasons) or "model_ranked_monitor"

    def action_label(reason_text: str) -> str:
        if "deep_zero_click_risk" in reason_text:
            return "monitor_or_prune"
        if "visible_low_ctr_page" in reason_text:
            return "review_title_meta"
        if "weak_engagement_context" in reason_text:
            return "review_on_page_engagement"
        if "future_decline_risk" in reason_text:
            return "refresh_or_monitor_decline"
        return "monitor"

    eligible["reason_codes"] = eligible.apply(reason_codes, axis=1)
    eligible["action_label"] = eligible["reason_codes"].map(action_label)
    ranked = eligible.sort_values(
        ["final_action_score", "impressions_may"], ascending=[False, False]
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    recommendation_columns = [
        "rank",
        "content_hash_id",
        "client_hash_id",
        "final_action_score",
        "model_probability",
        "baseline_action_score",
        "action_label",
        "reason_codes",
        "impressions_may",
        "clicks_may",
        "ctr_may",
        "avg_position_may",
        "sessions_may",
        "engagement_rate_may",
        "impressions_jun",
        "future_decline_label",
    ]
    ranked[recommendation_columns].to_csv(OUT_DIR / "capstone_ranked_recommendations.csv", index=False)
    ranked[recommendation_columns].head(20).to_csv(
        OUT_DIR / "capstone_top20_recommendations.csv", index=False
    )

    write_charts(results_df, importance_df, ranked)

    return {
        "release": "FlyRank/internship-warehouse v20260703",
        "feature_window": "2025-08-01 through 2025-09-30; final feature month 2025-09",
        "target_window": "2025-10-01 through 2025-10-31",
        "label": "future_decline_label = October 2025 impressions < 80% of September 2025 impressions",
        "selected_model": "random_forest",
        "eligible_rows": int(len(eligible)),
        "eligible_clients": int(eligible["client_hash_id"].nunique()),
        "base_rate": round(float(y.mean()), 4),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_clients": int(train["client_hash_id"].nunique()),
        "test_clients": int(test["client_hash_id"].nunique()),
        "results": results_df.to_dict(orient="records"),
        "top_features": importance_df.to_dict(orient="records"),
        "top20_preview": ranked[recommendation_columns].head(20).to_dict(orient="records"),
        "outputs": {
            "feature_table_csv": "work/outputs/capstone_feature_table.csv",
            "ranked_recommendations_csv": "work/outputs/capstone_ranked_recommendations.csv",
            "top20_csv": "work/outputs/capstone_top20_recommendations.csv",
            "model_vs_baseline_chart": "work/figures/capstone_model_vs_baseline.png",
            "feature_importance_chart": "work/figures/capstone_feature_importance.png",
            "action_mix_chart": "work/figures/capstone_action_mix.png",
        },
        "leakage_exclusions": [
            "June metrics are label/output only, not model features",
            "trend_direction and trend_pct are not warehouse fields and are not used",
            "IDs are used for grouping/splitting only",
            "validation holds out whole clients",
        ],
    }


def write_charts(results_df: pd.DataFrame, importance_df: pd.DataFrame, ranked: pd.DataFrame) -> None:
    ax = results_df.set_index("method")[["precision_at_50", "precision_at_100"]].plot(
        kind="bar", figsize=(7, 4)
    )
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.set_title("Model vs baseline on held-out clients")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "capstone_model_vs_baseline.png", dpi=180)
    plt.close()

    show = importance_df.head(12).iloc[::-1].copy()
    show["feature"] = show["feature"].str.replace("num__", "", regex=False).str.replace(
        "cat__", "", regex=False
    )
    plt.figure(figsize=(7, 5))
    plt.barh(show["feature"], show["importance"])
    plt.title("Top random forest feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "capstone_feature_importance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    ranked.head(100)["action_label"].value_counts().plot(kind="bar")
    plt.title("Action mix in top 100 recommendations")
    plt.ylabel("Pages")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "capstone_action_mix.png", dpi=180)
    plt.close()


def main() -> None:
    token = get_token()
    eligible = build_feature_table(token)
    summary = train_and_score(eligible)
    metrics_path = OUT_DIR / "capstone_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in [
        "eligible_rows",
        "eligible_clients",
        "base_rate",
        "train_rows",
        "test_rows",
        "train_clients",
        "test_clients",
    ]}, indent=2), flush=True)
    print(pd.DataFrame(summary["results"]).to_string(index=False), flush=True)
    print(f"Wrote {metrics_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
