
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

# Load machine learning dependencies
joblib = importlib.import_module("joblib")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Initialize Flask
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "Online Retail.xlsx"
RESULT_FILE = OUTPUT_DIR / "customer_segments.csv"
MODEL_FILE = OUTPUT_DIR / "kmeans_pipeline.joblib"

REQUIRED_COLUMNS = {
    "InvoiceNo",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
}


def read_retail_file(file):
    """Read an uploaded CSV/Excel file or the default dataset."""

    if file is not None and file.filename:
        filename = secure_filename(file.filename)
        extension = Path(filename).suffix.lower()

        if extension == ".csv":
            return pd.read_csv(file)

        if extension in (".xlsx", ".xls"):
            return pd.read_excel(file)

        raise ValueError("Please upload a CSV, XLSX, or XLS file.")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Upload a dataset or place "
            "'Online Retail.xlsx' inside the data folder."
        )

    return pd.read_excel(DATA_FILE)


def build_customer_segments(raw, n_clusters=4):
    """Clean transactions, build RFM features, and cluster customers."""

    if raw is None or raw.empty:
        raise ValueError("The uploaded dataset is empty.")

    # Normalize column names
    raw = raw.copy()
    raw.columns = raw.columns.astype(str).str.strip()

    missing = REQUIRED_COLUMNS - set(raw.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Clean data types
    df = raw.copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    for col in ["Quantity", "UnitPrice", "CustomerID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "InvoiceNo",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
            "CustomerID",
        ]
    )

    # Remove cancellation invoices and invalid sales
    df = df[
        ~df["InvoiceNo"].str.upper().str.startswith("C", na=False)
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (df["CustomerID"] > 0)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid sales transactions remain after cleaning."
        )

    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # RFM analysis
    reference_date = (
        df["InvoiceDate"].max() + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda dates: (reference_date - dates.max()).days,
            ),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    if len(rfm) < 3:
        raise ValueError(
            "At least three identifiable customers are required."
        )

    features = ["Recency", "Frequency", "Monetary"]

    # Transform skewed features and standardize them
    model_data = rfm[features].clip(lower=0).copy()
    model_data = np.log1p(model_data)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(model_data)

    # Ensure the cluster count is valid
    n_clusters = int(n_clusters)
    cluster_count = min(max(2, n_clusters), len(rfm) - 1)

    model = KMeans(
        n_clusters=cluster_count,
        n_init=10,
        random_state=42,
    )

    rfm["Cluster"] = model.fit_predict(scaled_data)

    # Evaluate clustering quality
    labels = rfm["Cluster"].to_numpy()
    unique_labels = np.unique(labels)

    score = None

    if 1 < len(unique_labels) < len(rfm):
        score = float(silhouette_score(scaled_data, labels))

    # Build cluster profiles
    cluster_profiles = (
        rfm.groupby("Cluster")
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            CustomerCount=("CustomerID", "count"),
        )
    )

    cluster_profiles["SharePercent"] = (
        cluster_profiles["CustomerCount"] / len(rfm) * 100
    )

    # Assign a consistent business label to each cluster.
    median_recency = cluster_profiles["Recency"].median()
    median_frequency = cluster_profiles["Frequency"].median()
    median_monetary = cluster_profiles["Monetary"].median()

    def describe_cluster(row):
        high_frequency = row["Frequency"] >= median_frequency
        high_monetary = row["Monetary"] >= median_monetary
        recent = row["Recency"] <= median_recency

        if high_frequency and high_monetary and recent:
            return "High-value loyal"

        if high_frequency and high_monetary:
            return "Previously valuable"

        if high_frequency and not recent:
            return "Needs re-engagement"

        if recent and not high_frequency:
            return "Recent / developing"

        return "Low-frequency / low-spend"

    cluster_profiles["Segment"] = cluster_profiles.apply(
        describe_cluster, axis=1
    )

    segment_mapping = cluster_profiles["Segment"].to_dict()
    rfm["Segment"] = rfm["Cluster"].map(segment_mapping)

    # Save customer-level results and trained objects
    rfm.to_csv(RESULT_FILE, index=False)

    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "features": features,
            "reference_date": reference_date,
            "segment_mapping": segment_mapping,
        },
        MODEL_FILE,
    )

    # Create interactive Plotly visualization
    fig = px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color=rfm["Cluster"].astype(str),
        hover_data=[
            "CustomerID",
            "Recency",
            "Segment",
        ],
        title="Customer Segments: Purchase Frequency vs Spend",
        labels={
            "Frequency": "Purchase frequency",
            "Monetary": "Total spend",
            "color": "Cluster",
        },
    )

    fig.update_layout(template="plotly_white")

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )

    # Dashboard summary
    summary = {
        "transactions": int(len(df)),
        "customers": int(rfm["CustomerID"].nunique()),
        "revenue": round(float(df["Revenue"].sum()), 2),
        "clusters": int(rfm["Cluster"].nunique()),
        "silhouette": (
            round(score, 4) if score is not None else None
        ),
        "reference_date": reference_date.strftime("%Y-%m-%d"),
    }

    # Human-readable cluster summary table
    profile_table = (
        cluster_profiles.reset_index()
        .rename(
            columns={
                "Recency": "AvgRecency",
                "Frequency": "AvgFrequency",
                "Monetary": "AvgMonetary",
                "SharePercent": "SharePercent",
            }
        )
        .round(2)
        .to_dict(orient="records")
    )

    return summary, plot_html, profile_table, rfm


@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    plot_html = None
    profiles = None
    preview = None
    error = None
    selected_clusters = 4

    if request.method == "POST":
        try:
            selected_clusters = int(
                request.form.get("clusters", 4)
            )
            selected_clusters = min(
                max(selected_clusters, 2), 10
            )

            upload = request.files.get("dataset")
            raw = read_retail_file(upload)

            summary, plot_html, profiles, results = (
                build_customer_segments(
                    raw,
                    selected_clusters,
                )
            )

            preview = results.head(10).to_dict(
                orient="records"
            )

        except (
            ValueError,
            FileNotFoundError,
            ImportError,
            OSError,
        ) as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Customer segmentation failed.")
            error = (
                "An unexpected error occurred. "
                "Check the terminal logs for details."
            )

    return render_template(
        "index.html",
        summary=summary,
        plot_html=plot_html,
        profiles=profiles,
        preview=preview,
        error=error,
        selected_clusters=selected_clusters,
    )


@app.route("/download")
def download_results():
    if not RESULT_FILE.exists():
        return (
            "Run customer segmentation before downloading results.",
            404,
        )

    return send_file(
        RESULT_FILE,
        as_attachment=True,
        download_name="customer_segments.csv",
        mimetype="text/csv",
    )


if __name__ == "__main__":
    app.run(debug=True)
