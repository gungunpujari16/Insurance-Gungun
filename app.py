"""
Insurance Claim Settlement Bias & Classification Dashboard
---------------------------------------------------------
Run locally:
    streamlit run app.py

This app is designed for the supplied Insurance.csv structure, but it is
written defensively so it can work with similarly structured claim datasets.
"""

from __future__ import annotations

import io
import re
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Insurance Claim Settlement Bias Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
    .metric-card {
        background: linear-gradient(135deg, #f7f9fc 0%, #ffffff 100%);
        border: 1px solid #e7edf5;
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 2px 10px rgba(20, 30, 50, 0.04);
    }
    .small-note {
        color: #657080;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .risk-box {
        border-left: 5px solid #cf3e53;
        background: #fff6f7;
        padding: 14px 16px;
        border-radius: 8px;
        margin: 8px 0px;
    }
    .good-box {
        border-left: 5px solid #2d8a5f;
        background: #f3fbf7;
        padding: 14px 16px;
        border-radius: 8px;
        margin: 8px 0px;
    }
    .info-box {
        border-left: 5px solid #3867d6;
        background: #f4f7ff;
        padding: 14px 16px;
        border-radius: 8px;
        margin: 8px 0px;
    }
    div[data-testid="stDataFrame"] {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

DEFAULT_TARGET = "POLICY_STATUS"
DEFAULT_POSITIVE_KEYWORD = "Repudiate"
IDENTIFIER_COLS = ["POLICY_NO", "PI_NAME"]


def normalize_colname(col: str) -> str:
    """Normalize raw CSV column names into uppercase snake-like labels."""
    return str(col).strip()


def parse_money_value(value) -> float:
    """Convert values like '1,000,000' into numeric values."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", ".", "-"}:
        return np.nan
    return pd.to_numeric(text, errors="coerce")


def make_ohe() -> OneHotEncoder:
    """Create OneHotEncoder compatible with old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # for older scikit-learn versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def pct(x: float, decimals: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{100 * x:.{decimals}f}%"


def safe_rate(num: float, den: float) -> float:
    return np.nan if den == 0 else num / den


def reason_category(reason) -> str:
    """Bucket claim reason into interpretable categories."""
    s = str(reason).lower().strip()
    if s in {"", "nan", "none", "null"}:
        return "Missing/Not Recorded"
    if any(k in s for k in ["heart", "cardiac", "cardio", "chest", "coronary", "pulmonary"]):
        return "Cardiac/Heart"
    if any(
        k in s
        for k in [
            "accident",
            "road",
            "vehicle",
            "bike",
            "car",
            "train",
            "fall",
            "drown",
            "electrocution",
            "electric",
            "murder",
            "hit",
        ]
    ):
        return "Accident/External"
    if "cancer" in s:
        return "Cancer"
    if any(k in s for k in ["covid", "corona"]):
        return "COVID"
    if "suicide" in s:
        return "Suicide"
    if any(k in s for k in ["kidney", "renal"]):
        return "Kidney/Renal"
    if "liver" in s:
        return "Liver"
    if "natural" in s:
        return "Natural Death"
    if any(k in s for k in ["fever", "infection", "sepsis", "tb", "tuberculosis", "malaria", "dengue"]):
        return "Fever/Infection"
    return "Other/Unclear"


def safe_qcut(series: pd.Series, q: int = 4, labels: Optional[List[str]] = None) -> pd.Series:
    """Quantile bins that do not crash when there are duplicate edges."""
    s = pd.to_numeric(series, errors="coerce")
    if labels is None:
        labels = [f"Q{i}" for i in range(1, q + 1)]
    try:
        ranked = s.rank(method="first")
        return pd.qcut(ranked, q=q, labels=labels, duplicates="drop").astype("object")
    except Exception:
        return pd.Series(["Not enough variation"] * len(series), index=series.index, dtype="object")


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def prepare_data(raw_df: pd.DataFrame, target_col: str, positive_value: str) -> pd.DataFrame:
    """Clean raw data and create feature engineering columns."""
    df = raw_df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    # Basic cleaning for object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
        df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)

    # Numeric conversions for common insurance fields
    if "SUM_ASSURED" in df.columns:
        df["SUM_ASSURED_NUM"] = df["SUM_ASSURED"].apply(parse_money_value)
    if "PI_ANNUAL_INCOME" in df.columns:
        df["PI_ANNUAL_INCOME_NUM"] = df["PI_ANNUAL_INCOME"].apply(parse_money_value)

    # Target flag: 1 = selected positive class, default Repudiate Death
    df[target_col] = df[target_col].astype(str).str.strip()
    df["TARGET_POSITIVE"] = (df[target_col] == str(positive_value).strip()).astype(int)

    # Age bands
    if "PI_AGE" in df.columns:
        df["PI_AGE"] = pd.to_numeric(df["PI_AGE"], errors="coerce")
        df["AGE_BAND"] = pd.cut(
            df["PI_AGE"],
            bins=[0, 30, 40, 50, 60, 70, 120],
            labels=["<=30", "31-40", "41-50", "51-60", "61-70", "70+"],
            include_lowest=True,
        ).astype("object")

    # Sum assured quantile band
    if "SUM_ASSURED_NUM" in df.columns:
        df["SUM_ASSURED_BAND"] = safe_qcut(
            df["SUM_ASSURED_NUM"],
            q=4,
            labels=["Low", "Mid-Low", "Mid-High", "High"],
        )

    # Income bands + declared flag
    if "PI_ANNUAL_INCOME_NUM" in df.columns:
        income = df["PI_ANNUAL_INCOME_NUM"]
        df["INCOME_DECLARED_FLAG"] = np.where(income.fillna(0) > 0, "Income Declared", "0 / Not Declared")
        df["INCOME_BAND"] = pd.cut(
            income,
            bins=[-1, 0, 100000, 200000, 500000, 10**12],
            labels=["0 / Not Declared", "1-100k", "100k-200k", "200k-500k", "500k+"],
            include_lowest=True,
        ).astype("object")

    # Sum assured to income ratio: important underwriting check; undefined if income is zero/missing.
    if "SUM_ASSURED_NUM" in df.columns and "PI_ANNUAL_INCOME_NUM" in df.columns:
        df["SA_TO_INCOME_RATIO"] = np.where(
            df["PI_ANNUAL_INCOME_NUM"] > 0,
            df["SUM_ASSURED_NUM"] / df["PI_ANNUAL_INCOME_NUM"],
            np.nan,
        )
        df["SA_TO_INCOME_BAND"] = pd.cut(
            df["SA_TO_INCOME_RATIO"],
            bins=[-np.inf, 2, 5, 10, 20, np.inf],
            labels=["<=2x", "2x-5x", "5x-10x", "10x-20x", ">20x"],
        ).astype("object")
        df["SA_TO_INCOME_BAND"] = df["SA_TO_INCOME_BAND"].where(
            df["SA_TO_INCOME_RATIO"].notna(), "Income Missing/Zero"
        )

    # Reason category
    if "REASON_FOR_CLAIM" in df.columns:
        df["REASON_CATEGORY"] = df["REASON_FOR_CLAIM"].apply(reason_category)

    # Useful clean flags
    if "EARLY_NON" in df.columns:
        df["EARLY_FLAG"] = np.where(df["EARLY_NON"].astype(str).str.upper().eq("EARLY"), "Early", "Non-Early")
    if "MEDICAL_NONMED" in df.columns:
        df["MEDICAL_FLAG"] = np.where(
            df["MEDICAL_NONMED"].astype(str).str.upper().eq("MEDICAL"), "Medical", "Non-Medical"
        )

    # Simple diagnostic score for claim investigation risk. This is not a decision rule; it is an audit aid.
    risk_score = pd.Series(0, index=df.index, dtype="float")
    if "EARLY_FLAG" in df.columns:
        risk_score += (df["EARLY_FLAG"] == "Early").astype(int)
    if "MEDICAL_FLAG" in df.columns:
        risk_score += (df["MEDICAL_FLAG"] == "Non-Medical").astype(int)
    if "INCOME_DECLARED_FLAG" in df.columns:
        risk_score += (df["INCOME_DECLARED_FLAG"] == "0 / Not Declared").astype(int)
    if "SA_TO_INCOME_RATIO" in df.columns:
        risk_score += (df["SA_TO_INCOME_RATIO"] > 10).fillna(False).astype(int)
    if "SUM_ASSURED_BAND" in df.columns:
        risk_score += (df["SUM_ASSURED_BAND"] == "High").astype(int)
    df["CLAIM_INVESTIGATION_RISK_SCORE"] = risk_score
    df["CLAIM_INVESTIGATION_RISK_BAND"] = pd.cut(
        risk_score,
        bins=[-0.1, 1, 2, 3, 10],
        labels=["Low", "Moderate", "High", "Very High"],
    ).astype("object")

    return df


def segment_summary(df: pd.DataFrame, group_cols: Sequence[str], min_n: int = 20) -> pd.DataFrame:
    """Return grouped approval/repudiation table."""
    if not group_cols:
        return pd.DataFrame()
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    g = (
        df.groupby(group_cols, dropna=False, observed=False)
        .agg(
            Claims=("TARGET_POSITIVE", "size"),
            Positive_Cases=("TARGET_POSITIVE", "sum"),
            Positive_Rate=("TARGET_POSITIVE", "mean"),
        )
        .reset_index()
    )
    g["Negative_Cases"] = g["Claims"] - g["Positive_Cases"]
    overall = df["TARGET_POSITIVE"].mean()
    g["Rate_vs_Overall_pp"] = (g["Positive_Rate"] - overall) * 100
    g["Disparity_Ratio_vs_Overall"] = g["Positive_Rate"] / overall if overall > 0 else np.nan
    g = g[g["Claims"] >= min_n].sort_values(["Positive_Rate", "Claims"], ascending=[False, False])
    return g


def row_percentage_crosstab(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    ct = pd.crosstab(df[group_col].fillna("Missing"), df[target_col].fillna("Missing"))
    row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    combined = pd.concat({"Count": ct, "Row %": row_pct.round(1)}, axis=1)
    return combined


def cramers_v(df: pd.DataFrame, col: str, target_col: str) -> Tuple[float, float]:
    """Cramer's V and p-value for categorical association."""
    tab = pd.crosstab(df[col].fillna("Missing"), df[target_col].fillna("Missing"))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    r, k = tab.shape
    denom = n * min(k - 1, r - 1)
    v = np.sqrt(chi2 / denom) if denom > 0 else np.nan
    return float(v), float(p)


def fig_rate_bar(table: pd.DataFrame, category_col: str, title: str, top_n: int = 15) -> go.Figure:
    data = table.head(top_n).copy()
    data["Positive Rate %"] = data["Positive_Rate"] * 100
    fig = px.bar(
        data,
        x="Positive Rate %",
        y=category_col,
        orientation="h",
        text=data["Positive Rate %"].round(1).astype(str) + "%",
        hover_data=["Claims", "Positive_Cases", "Negative_Cases", "Rate_vs_Overall_pp"],
        title=title,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(450, 30 * len(data)))
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def format_segment_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "Positive_Rate" in out.columns:
        out["Positive_Rate"] = (out["Positive_Rate"] * 100).round(1).astype(str) + "%"
    if "Rate_vs_Overall_pp" in out.columns:
        out["Rate_vs_Overall_pp"] = out["Rate_vs_Overall_pp"].round(1)
    if "Disparity_Ratio_vs_Overall" in out.columns:
        out["Disparity_Ratio_vs_Overall"] = out["Disparity_Ratio_vs_Overall"].round(2)
    return out


def get_default_audit_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "PI_GENDER",
        "AGE_BAND",
        "INCOME_BAND",
        "INCOME_DECLARED_FLAG",
        "SUM_ASSURED_BAND",
        "SA_TO_INCOME_BAND",
        "ZONE",
        "PI_STATE",
        "PI_OCCUPATION",
        "PAYMENT_MODE",
        "EARLY_FLAG",
        "MEDICAL_FLAG",
        "REASON_CATEGORY",
        "CLAIM_INVESTIGATION_RISK_BAND",
    ]
    return [c for c in candidates if c in df.columns]


def build_model_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Create X/y and identify numeric/categorical columns for ML."""
    exclude = set(IDENTIFIER_COLS + [target_col, "TARGET_POSITIVE"])
    # Avoid using raw high-cardinality text reason if engineered reason category exists.
    if "REASON_CATEGORY" in df.columns:
        exclude.add("REASON_FOR_CLAIM")
    # Avoid duplicate raw strings when clean numeric versions exist.
    if "SUM_ASSURED_NUM" in df.columns:
        exclude.add("SUM_ASSURED")
    if "PI_ANNUAL_INCOME_NUM" in df.columns:
        exclude.add("PI_ANNUAL_INCOME")

    usable_cols = [c for c in df.columns if c not in exclude]
    X = df[usable_cols].copy()
    y = df["TARGET_POSITIVE"].astype(int)

    # Keep only columns that are not all missing and have at least two unique values.
    usable_cols = [c for c in X.columns if X[c].notna().sum() > 0 and X[c].nunique(dropna=True) > 1]
    X = X[usable_cols]

    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_models(random_state: int, n_neighbors: int, tree_depth: int) -> Dict[str, object]:
    return {
        "KNN": KNeighborsClassifier(n_neighbors=n_neighbors),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=tree_depth,
            min_samples_leaf=20,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=max(tree_depth + 2, 5),
            min_samples_leaf=10,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def metric_dict(y_true, y_pred, y_proba) -> Dict[str, float]:
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    except Exception:
        out["ROC-AUC"] = np.nan
    return out


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    test_size: float,
    random_state: int,
    n_neighbors: int,
    tree_depth: int,
) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    models = get_models(random_state, n_neighbors, tree_depth)
    results = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)
        try:
            train_proba = pipe.predict_proba(X_train)[:, 1]
            test_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            train_proba = train_pred
            test_proba = test_pred

        results[name] = {
            "pipeline": pipe,
            "train_metrics": metric_dict(y_train, train_pred, train_proba),
            "test_metrics": metric_dict(y_test, test_pred, test_proba),
            "y_train": y_train,
            "y_test": y_test,
            "train_pred": train_pred,
            "test_pred": test_pred,
            "train_proba": train_proba,
            "test_proba": test_proba,
            "X_train": X_train,
            "X_test": X_test,
        }
    return results


def metrics_to_frame(results: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for model_name, data in results.items():
        for split in ["train", "test"]:
            m = data[f"{split}_metrics"]
            row = {"Model": model_name, "Split": split.title()}
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_metrics(metrics_df: pd.DataFrame, split: str = "Test") -> go.Figure:
    long_df = metrics_df[metrics_df["Split"] == split].melt(
        id_vars=["Model", "Split"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        long_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text=long_df["Score"].round(3),
        title=f"{split} Metrics by Algorithm",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=480)
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def plot_train_test_gap(metrics_df: pd.DataFrame) -> go.Figure:
    pivot = metrics_df.pivot(index="Model", columns="Split", values="Accuracy").reset_index()
    if "Train" in pivot.columns and "Test" in pivot.columns:
        pivot["Accuracy Gap"] = pivot["Train"] - pivot["Test"]
    else:
        pivot["Accuracy Gap"] = np.nan
    fig = px.bar(
        pivot,
        x="Model",
        y="Accuracy Gap",
        text=pivot["Accuracy Gap"].round(3),
        title="Training vs Testing Accuracy Gap — Overfitting Check",
    )
    fig.update_layout(height=420)
    return fig


def plot_roc_curves(results: Dict[str, object]) -> go.Figure:
    fig = go.Figure()
    for model_name, data in results.items():
        y_test = data["y_test"]
        y_score = data["test_proba"]
        try:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{model_name} AUC={roc_auc:.3f}",
                )
            )
        except Exception:
            continue
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash"))
    )
    fig.update_layout(
        title="ROC Curve — Model Stability / Discrimination Power",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate / Recall",
        height=550,
    )
    return fig


def plot_confusion(y_true, y_pred, model_name: str) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{labels[i][j]}<br>{cm[i, j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted: Negative", "Predicted: Positive"],
            y=["Actual: Negative", "Actual: Positive"],
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(title=f"Confusion Matrix — {model_name}", height=420)
    return fig


def get_feature_names(pipe: Pipeline, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    preprocessor = pipe.named_steps["preprocess"]
    names = []
    if numeric_cols:
        names.extend(numeric_cols)
    if categorical_cols:
        try:
            cat_pipeline = preprocessor.named_transformers_["cat"]
            ohe = cat_pipeline.named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
            names.extend(cat_names)
        except Exception:
            names.extend(categorical_cols)
    return names


def feature_importance_table(
    pipe: Pipeline, numeric_cols: List[str], categorical_cols: List[str], top_n: int = 25
) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    names = get_feature_names(pipe, numeric_cols, categorical_cols)
    importances = model.feature_importances_
    n = min(len(names), len(importances))
    table = pd.DataFrame({"Feature": names[:n], "Importance": importances[:n]})
    table = table.sort_values("Importance", ascending=False).head(top_n)
    return table


def make_download_button(df: pd.DataFrame, filename: str, label: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# -----------------------------------------------------------------------------
# App header and data upload
# -----------------------------------------------------------------------------

st.title("🛡️ Insurance Claim Settlement Bias & Classification Dashboard")
st.caption(
    "Descriptive analytics, diagnostic bias audit, feature engineering, supervised classification, model metrics, ROC curves, and findings."
)

with st.sidebar:
    st.header("1) Upload Data")
    uploaded = st.file_uploader("Upload Insurance.csv", type=["csv"])
    st.markdown(
        "<div class='small-note'>The expected target column is <b>POLICY_STATUS</b>. "
        "The dashboard treats the selected positive class as the risk/outcome to detect.</div>",
        unsafe_allow_html=True,
    )

if uploaded is None:
    st.info("Upload your `Insurance.csv` file from the sidebar to begin the analysis.")
    st.markdown(
        """
        ### What this dashboard will do
        1. **Descriptive analytics**: cross-tabulations against policy status.  
        2. **Diagnostic analysis**: detect unusual approval/repudiation rate differences by age, income, zone/team, occupation, medical status, early/non-early status, etc.  
        3. **Feature engineering**: create age bands, income bands, sum assured bands, sum assured-to-income ratio, reason categories, and investigation risk scores.  
        4. **Supervised learning**: KNN, Decision Tree, Random Forest, and Gradient Boosting.  
        5. **Model evaluation**: train/test accuracy, precision, recall, F1-score, ROC-AUC, ROC curve, and confusion matrix.  
        6. **Findings**: practical audit conclusions and recommendations.
        """
    )
    st.stop()

raw_df = load_csv(uploaded.getvalue())
raw_df.columns = [normalize_colname(c) for c in raw_df.columns]

with st.sidebar:
    st.header("2) Target Setup")
    target_default_ix = raw_df.columns.tolist().index(DEFAULT_TARGET) if DEFAULT_TARGET in raw_df.columns else 0
    target_col = st.selectbox("Policy status / target column", raw_df.columns.tolist(), index=target_default_ix)
    unique_targets = raw_df[target_col].astype(str).str.strip().dropna().unique().tolist()
    positive_default_ix = 0
    for i, val in enumerate(unique_targets):
        if DEFAULT_POSITIVE_KEYWORD.lower() in val.lower():
            positive_default_ix = i
            break
    positive_value = st.selectbox("Positive / risk class to detect", unique_targets, index=positive_default_ix)

    st.header("3) Analysis Controls")
    min_group_size = st.slider("Minimum segment size", min_value=5, max_value=100, value=30, step=5)
    test_size = st.slider("Test size for ML", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    n_neighbors = st.slider("KNN neighbors", min_value=3, max_value=25, value=7, step=2)
    tree_depth = st.slider("Decision tree max depth", min_value=2, max_value=12, value=5, step=1)

try:
    df = prepare_data(raw_df, target_col, positive_value)
except Exception as exc:
    st.error(f"Data preparation failed: {exc}")
    st.stop()

# Helpful labels
positive_label = positive_value
negative_label = "Other policy status"
if len(unique_targets) == 2:
    negative_label = [v for v in unique_targets if v != positive_value][0]

overall_positive_rate = df["TARGET_POSITIVE"].mean()

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tabs = st.tabs(
    [
        "🏠 Overview",
        "📊 Descriptive Cross Tabs",
        "🔎 Diagnostic Bias Audit",
        "🧬 Feature Engineering",
        "🤖 Supervised Learning",
        "✅ Findings",
    ]
)

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Company Claim Settlement Overview")
    c1, c2, c3, c4 = st.columns(4)
    total_claims = len(df)
    positive_cases = int(df["TARGET_POSITIVE"].sum())
    negative_cases = total_claims - positive_cases
    with c1:
        st.metric("Total Claims", f"{total_claims:,}")
    with c2:
        st.metric(f"Positive Class: {positive_label}", f"{positive_cases:,}")
    with c3:
        st.metric("Other Status", f"{negative_cases:,}")
    with c4:
        st.metric("Positive Rate", pct(overall_positive_rate))

    st.markdown(
        f"""
        <div class='info-box'>
        <b>Interpretation:</b> In this dashboard, <b>{positive_label}</b> is treated as the positive/risk class. 
        So precision, recall, F1-score, and ROC-AUC are evaluated for detecting <b>{positive_label}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        status_counts = df[target_col].value_counts().reset_index()
        status_counts.columns = [target_col, "Claims"]
        fig = px.bar(status_counts, x=target_col, y="Claims", text="Claims", title="Policy Status Distribution")
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "EARLY_FLAG" in df.columns or "MEDICAL_FLAG" in df.columns:
            overview_col = st.selectbox(
                "Quick distribution view",
                [c for c in ["EARLY_FLAG", "MEDICAL_FLAG", "PAYMENT_MODE", "AGE_BAND", "INCOME_BAND", "SUM_ASSURED_BAND"] if c in df.columns],
            )
            quick = segment_summary(df, [overview_col], min_n=1)
            st.plotly_chart(fig_rate_bar(quick, overview_col, f"{positive_label} Rate by {overview_col}"), use_container_width=True)

    st.write("### Raw data preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

# -----------------------------------------------------------------------------
# Descriptive Cross Tabs
# -----------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Descriptive Analytics: Cross Tabulation Against Policy Status")
    st.markdown(
        """
        Cross-tabulation shows how claim status is distributed inside each category.  
        Use this to answer: **which groups have a higher approval/repudiation rate, and is the group size large enough to trust?**
        """
    )

    candidate_cols = [c for c in get_default_audit_columns(df) if c != target_col]
    if not candidate_cols:
        candidate_cols = [c for c in df.columns if c not in [target_col, "TARGET_POSITIVE"]]

    selected_col = st.selectbox("Select column for cross-tabulation", candidate_cols)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.write("#### Count and row-percentage crosstab")
        st.dataframe(row_percentage_crosstab(df, selected_col, target_col), use_container_width=True)
    with col_b:
        seg = segment_summary(df, [selected_col], min_n=min_group_size)
        st.write(f"#### {positive_label} rate by {selected_col}")
        st.plotly_chart(fig_rate_bar(seg, selected_col, f"{positive_label} Rate by {selected_col}"), use_container_width=True)

    st.write("#### Segment table")
    formatted_seg = format_segment_table(seg)
    st.dataframe(formatted_seg, use_container_width=True)
    make_download_button(seg, f"segment_{selected_col}.csv", "Download this segment table")

    st.divider()
    st.write("### Multi-feature cross-tab / segment finder")
    st.markdown(
        "This helps avoid shallow single-column conclusions. For example, income should be read with age, medical/non-medical status, early/non-early status, zone/team, occupation, and sum assured."
    )
    multi_cols = st.multiselect(
        "Choose 2-3 columns for deeper cross-feature segmentation",
        candidate_cols,
        default=[c for c in ["EARLY_FLAG", "MEDICAL_FLAG"] if c in candidate_cols],
        max_selections=3,
    )
    if multi_cols:
        multi_seg = segment_summary(df, multi_cols, min_n=min_group_size)
        st.dataframe(format_segment_table(multi_seg.head(50)), use_container_width=True)
        make_download_button(multi_seg, "multi_feature_segments.csv", "Download multi-feature segment table")
    else:
        st.warning("Select at least one column.")

# -----------------------------------------------------------------------------
# Diagnostic Bias Audit
# -----------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Diagnostic Analysis: Bias and Unusual Settlement Pattern Audit")
    st.markdown(
        f"""
        <div class='risk-box'>
        <b>Important:</b> This dashboard can detect <b>possible settlement disparity</b>, not prove illegal bias by itself. 
        A high repudiation rate for a group may be caused by valid underwriting or claim-risk reasons. 
        Bias suspicion becomes stronger when the disparity remains high after comparing similar groups, such as same early/non-early status, similar sum assured, similar income band, same zone/team, or similar reason category.
        </div>
        """,
        unsafe_allow_html=True,
    )

    audit_cols = st.multiselect(
        "Columns to audit for settlement disparity",
        get_default_audit_columns(df),
        default=[c for c in ["PI_GENDER", "AGE_BAND", "INCOME_BAND", "ZONE", "PI_OCCUPATION", "PAYMENT_MODE", "EARLY_FLAG", "MEDICAL_FLAG"] if c in get_default_audit_columns(df)],
    )

    audit_rows = []
    for col in audit_cols:
        seg = segment_summary(df, [col], min_n=min_group_size)
        if seg.empty:
            continue
        high = seg.iloc[0]
        low = seg.sort_values("Positive_Rate", ascending=True).iloc[0]
        v, pval = cramers_v(df, col, target_col)
        audit_rows.append(
            {
                "Column": col,
                "Highest-Risk Group": high[col],
                "Highest Group Claims": int(high["Claims"]),
                "Highest Positive Rate": high["Positive_Rate"],
                "Lowest-Risk Group": low[col],
                "Lowest Group Claims": int(low["Claims"]),
                "Lowest Positive Rate": low["Positive_Rate"],
                "Rate Gap pp": (high["Positive_Rate"] - low["Positive_Rate"]) * 100,
                "Disparity Ratio High/Low": safe_rate(high["Positive_Rate"], low["Positive_Rate"]),
                "Cramers V": v,
                "Chi-square p-value": pval,
            }
        )

    audit_df = pd.DataFrame(audit_rows).sort_values("Rate Gap pp", ascending=False) if audit_rows else pd.DataFrame()
    if not audit_df.empty:
        display_audit = audit_df.copy()
        for c in ["Highest Positive Rate", "Lowest Positive Rate"]:
            display_audit[c] = (display_audit[c] * 100).round(1).astype(str) + "%"
        display_audit["Rate Gap pp"] = display_audit["Rate Gap pp"].round(1)
        display_audit["Disparity Ratio High/Low"] = display_audit["Disparity Ratio High/Low"].round(2)
        display_audit["Cramers V"] = display_audit["Cramers V"].round(3)
        display_audit["Chi-square p-value"] = display_audit["Chi-square p-value"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "-")
        st.write("### Disparity summary by column")
        st.dataframe(display_audit, use_container_width=True)
        make_download_button(audit_df, "bias_disparity_audit.csv", "Download audit summary")

        fig = px.bar(
            audit_df,
            x="Column",
            y="Rate Gap pp",
            text=audit_df["Rate Gap pp"].round(1),
            title="Largest Positive-Rate Gaps by Column",
            hover_data=["Highest-Risk Group", "Lowest-Risk Group", "Cramers V", "Chi-square p-value"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No audit table generated. Lower the minimum segment size or select more columns.")

    st.divider()
    st.write("### Deep diagnostic segment finder")
    default_diag = [c for c in ["ZONE", "EARLY_FLAG", "MEDICAL_FLAG"] if c in get_default_audit_columns(df)]
    diag_cols = st.multiselect(
        "Choose columns to combine for diagnostic bias probing",
        get_default_audit_columns(df),
        default=default_diag,
        max_selections=4,
    )
    if diag_cols:
        diag = segment_summary(df, diag_cols, min_n=min_group_size)
        st.dataframe(format_segment_table(diag.head(100)), use_container_width=True)
        make_download_button(diag, "diagnostic_deep_segments.csv", "Download diagnostic segments")

        if not diag.empty:
            st.markdown(
                f"""
                <div class='info-box'>
                <b>How to read this:</b> segments at the top have the highest <b>{positive_label}</b> rate after applying your minimum group size filter. 
                These are not automatically biased segments; they are priority groups for audit, claim-file review, and process comparison.
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Feature Engineering for Claim Classification")
    st.markdown(
        """
        Feature engineering converts raw policy/claim information into columns that models and managers can understand.  
        This matters because a model trained only on raw text/categorical values may miss important ratios and risk patterns.
        """
    )

    engineering_notes = []
    if "SUM_ASSURED_NUM" in df.columns:
        engineering_notes.append(["SUM_ASSURED_NUM", "Converted comma-formatted SUM_ASSURED into a numeric column."])
    if "PI_ANNUAL_INCOME_NUM" in df.columns:
        engineering_notes.append(["PI_ANNUAL_INCOME_NUM", "Converted comma-formatted PI_ANNUAL_INCOME into a numeric column."])
    if "AGE_BAND" in df.columns:
        engineering_notes.append(["AGE_BAND", "Grouped policyholder age into interpretable age bands."])
    if "SUM_ASSURED_BAND" in df.columns:
        engineering_notes.append(["SUM_ASSURED_BAND", "Divided sum assured into Low, Mid-Low, Mid-High, and High quantile bands."])
    if "INCOME_BAND" in df.columns:
        engineering_notes.append(["INCOME_BAND", "Grouped income into bands including 0 / Not Declared."])
    if "SA_TO_INCOME_RATIO" in df.columns:
        engineering_notes.append(["SA_TO_INCOME_RATIO", "Calculated Sum Assured divided by Annual Income where income is available."])
    if "REASON_CATEGORY" in df.columns:
        engineering_notes.append(["REASON_CATEGORY", "Grouped raw claim reasons into broader medical/accident/natural categories."])
    if "CLAIM_INVESTIGATION_RISK_SCORE" in df.columns:
        engineering_notes.append(["CLAIM_INVESTIGATION_RISK_SCORE", "Created an audit score using early claim, non-medical, income missing, high sum assured, and high SA-to-income ratio indicators."])

    st.write("### Engineered columns created")
    st.dataframe(pd.DataFrame(engineering_notes, columns=["Engineered Feature", "Purpose"]), use_container_width=True)

    st.write("### Cleaned and engineered data preview")
    important_preview_cols = [
        c
        for c in [
            target_col,
            "TARGET_POSITIVE",
            "PI_GENDER",
            "PI_AGE",
            "AGE_BAND",
            "SUM_ASSURED_NUM",
            "SUM_ASSURED_BAND",
            "PI_ANNUAL_INCOME_NUM",
            "INCOME_BAND",
            "SA_TO_INCOME_RATIO",
            "SA_TO_INCOME_BAND",
            "EARLY_FLAG",
            "MEDICAL_FLAG",
            "ZONE",
            "PI_OCCUPATION",
            "REASON_CATEGORY",
            "CLAIM_INVESTIGATION_RISK_SCORE",
            "CLAIM_INVESTIGATION_RISK_BAND",
        ]
        if c in df.columns
    ]
    st.dataframe(df[important_preview_cols].head(50), use_container_width=True)

    st.write("### Missing value check after engineering")
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    missing["Missing %"] = missing["Missing Values"] / len(df) * 100
    st.dataframe(missing.sort_values("Missing Values", ascending=False), use_container_width=True)

# -----------------------------------------------------------------------------
# Supervised Learning
# -----------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Supervised Learning: Claim Status Classification")
    st.markdown(
        f"""
        The models below try to predict whether a claim will be **{positive_label}**.  
        In this context, **recall** is especially important because it tells us how many actual {positive_label} cases the model catches.
        """
    )

    X, y, numeric_cols, categorical_cols = build_model_features(df, target_col)
    st.write("#### Model input summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Rows used", f"{len(X):,}")
    with m2:
        st.metric("Model features", f"{X.shape[1]:,}")
    with m3:
        st.metric("Numeric features", f"{len(numeric_cols):,}")
    with m4:
        st.metric("Categorical features", f"{len(categorical_cols):,}")

    with st.expander("See model feature list"):
        st.write("Numeric features", numeric_cols)
        st.write("Categorical features", categorical_cols)

    if y.nunique() < 2:
        st.error("The target has only one class after setup. Choose a different positive class or target column.")
        st.stop()

    run_model = st.button("Train / Refresh Models", type="primary")
    if "model_results" not in st.session_state or run_model:
        with st.spinner("Training KNN, Decision Tree, Random Forest, and Gradient Boosting..."):
            st.session_state["model_results"] = train_all_models(
                X,
                y,
                numeric_cols,
                categorical_cols,
                test_size=float(test_size),
                random_state=int(random_state),
                n_neighbors=int(n_neighbors),
                tree_depth=int(tree_depth),
            )
            st.session_state["model_X"] = X
            st.session_state["model_y"] = y
            st.session_state["numeric_cols"] = numeric_cols
            st.session_state["categorical_cols"] = categorical_cols

    results = st.session_state["model_results"]
    metrics_df = metrics_to_frame(results)
    display_metrics = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        display_metrics[col] = display_metrics[col].round(3)

    st.write("### Training and testing metrics")
    st.dataframe(display_metrics, use_container_width=True)
    make_download_button(metrics_df, "model_metrics.csv", "Download model metrics")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_metrics(metrics_df, split="Test"), use_container_width=True)
    with right:
        st.plotly_chart(plot_train_test_gap(metrics_df), use_container_width=True)

    st.write("### ROC curves")
    st.plotly_chart(plot_roc_curves(results), use_container_width=True)

    st.write("### Confusion matrices")
    selected_model_cm = st.selectbox("Select model for confusion matrix", list(results.keys()))
    st.plotly_chart(
        plot_confusion(
            results[selected_model_cm]["y_test"],
            results[selected_model_cm]["test_pred"],
            selected_model_cm,
        ),
        use_container_width=True,
    )

    cm = confusion_matrix(results[selected_model_cm]["y_test"], results[selected_model_cm]["test_pred"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    st.markdown(
        f"""
        **Confusion matrix reading for {selected_model_cm}:**  
        - **TP:** {tp} claims correctly predicted as `{positive_label}`  
        - **FN:** {fn} actual `{positive_label}` claims missed by the model  
        - **FP:** {fp} claims wrongly flagged as `{positive_label}`  
        - **TN:** {tn} claims correctly predicted as `{negative_label}`
        """
    )

    st.write("### Cross-validation stability check")
    do_cv = st.checkbox("Run 5-fold cross-validation", value=False)
    if do_cv:
        cv_rows = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_state))
        preprocessor = make_preprocessor(numeric_cols, categorical_cols)
        for model_name, model in get_models(int(random_state), int(n_neighbors), int(tree_depth)).items():
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
            scores = cross_validate(
                pipe,
                X,
                y,
                cv=cv,
                scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                n_jobs=-1,
                error_score="raise",
            )
            cv_rows.append(
                {
                    "Model": model_name,
                    "CV Accuracy Mean": scores["test_accuracy"].mean(),
                    "CV Accuracy Std": scores["test_accuracy"].std(),
                    "CV Precision Mean": scores["test_precision"].mean(),
                    "CV Recall Mean": scores["test_recall"].mean(),
                    "CV F1 Mean": scores["test_f1"].mean(),
                    "CV ROC-AUC Mean": scores["test_roc_auc"].mean(),
                    "CV ROC-AUC Std": scores["test_roc_auc"].std(),
                }
            )
        cv_df = pd.DataFrame(cv_rows)
        st.dataframe(cv_df.round(3), use_container_width=True)
        make_download_button(cv_df, "cross_validation_scores.csv", "Download CV scores")

    st.write("### Feature importance for tree-based models")
    importance_model = st.selectbox("Select model for feature importance", ["Decision Tree", "Random Forest", "Gradient Boosting"])
    imp = feature_importance_table(results[importance_model]["pipeline"], numeric_cols, categorical_cols, top_n=25)
    if imp.empty:
        st.info("Feature importance is not available for this model.")
    else:
        st.dataframe(imp, use_container_width=True)
        fig = px.bar(
            imp.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top Feature Importances — {importance_model}",
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Findings
# -----------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Findings and Practical Recommendations")

    st.write("### 1) Overall settlement pattern")
    st.markdown(
        f"""
        - Total claims analyzed: **{len(df):,}**  
        - `{positive_label}` cases: **{int(df['TARGET_POSITIVE'].sum()):,}**  
        - Positive/risk rate: **{pct(overall_positive_rate)}**
        """
    )

    st.write("### 2) Strongest single-column signals")
    signal_rows = []
    for col in get_default_audit_columns(df):
        seg = segment_summary(df, [col], min_n=min_group_size)
        if seg.empty:
            continue
        high = seg.iloc[0]
        signal_rows.append(
            {
                "Column": col,
                "Highest segment": high[col],
                "Claims": int(high["Claims"]),
                "Positive cases": int(high["Positive_Cases"]),
                "Positive rate": high["Positive_Rate"],
                "Rate vs overall pp": high["Rate_vs_Overall_pp"],
            }
        )
    signals = pd.DataFrame(signal_rows).sort_values("Positive rate", ascending=False).head(10) if signal_rows else pd.DataFrame()
    if not signals.empty:
        display_signals = signals.copy()
        display_signals["Positive rate"] = (display_signals["Positive rate"] * 100).round(1).astype(str) + "%"
        display_signals["Rate vs overall pp"] = display_signals["Rate vs overall pp"].round(1)
        st.dataframe(display_signals, use_container_width=True)

    st.write("### 3) Strongest cross-feature audit segments")
    default_find_cols = [c for c in ["ZONE", "EARLY_FLAG", "MEDICAL_FLAG"] if c in df.columns]
    if len(default_find_cols) >= 2:
        final_segments = segment_summary(df, default_find_cols, min_n=min_group_size).head(10)
        st.dataframe(format_segment_table(final_segments), use_container_width=True)

    st.write("### 4) Model summary")
    if "model_results" in st.session_state:
        metrics_df = metrics_to_frame(st.session_state["model_results"])
        test_only = metrics_df[metrics_df["Split"] == "Test"].copy()
        best_auc = test_only.sort_values("ROC-AUC", ascending=False).iloc[0]
        best_recall = test_only.sort_values("Recall", ascending=False).iloc[0]
        st.markdown(
            f"""
            - Best model by **ROC-AUC**: **{best_auc['Model']}** with ROC-AUC **{best_auc['ROC-AUC']:.3f}**.  
            - Best model by **Recall** for `{positive_label}`: **{best_recall['Model']}** with recall **{best_recall['Recall']:.3f}**.  
            - If your audit priority is to catch as many potentially repudiated claims as possible, prioritize **recall**.  
            - If your priority is to avoid wrongly flagging approved claims, prioritize **precision**.
            """
        )
    else:
        st.info("Train models in the Supervised Learning tab to populate model findings.")

    st.write("### 5) Recommended actions")
    st.markdown(
        f"""
        <div class='good-box'>
        <b>Recommended audit approach:</b><br>
        1. Review high-rate segments manually, especially where group size is meaningful.<br>
        2. Compare similar claim profiles before concluding bias: same early/non-early status, medical status, sum assured band, income band, reason category, and zone/team.<br>
        3. Use model feature importance as an audit clue, not as final proof.<br>
        4. Investigate process bias by checking whether similar claims receive different outcomes across zone/team, age band, gender, income band, or occupation.<br>
        5. Document claim-file evidence for each high-risk segment: missing documents, misrepresentation, waiting period issues, medical disclosures, and investigation notes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("### 6) Governance warning")
    st.markdown(
        """
        A machine learning model should **not** be used as an automatic claim rejection tool.  
        It should be used as an audit and review-support tool. Final claim decisions should remain explainable, documented, and compliant with internal policy and insurance regulations.
        """
    )

    st.write("### Download engineered dataset")
    make_download_button(df, "engineered_insurance_claims.csv", "Download engineered dataset")
