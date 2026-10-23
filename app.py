import hashlib
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Telecom CDR Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Telecom CDR Big Data Analytics")
st.caption(
    "HyperLogLog • Count-Min Sketch • K-Means Clustering"
)
st.write(
    "Upload a CDR CSV to begin. Open each analysis section below when you are ready."
)

REQUIRED_COLUMNS = [
    "caller_id",
    "receiver_id",
    "call_duration_sec",
    "timestamp",
    "call_type",
    "cell_tower",
]


# ----------------------------
# HyperLogLog (educational)
# ----------------------------
class HyperLogLog:
    def __init__(self, precision=6):
        if not 4 <= precision <= 16:
            raise ValueError("precision must be between 4 and 16")
        self.p = precision
        self.m = 1 << precision
        self.registers = [0] * self.m

    def add(self, item):
        digest = hashlib.sha256(str(item).encode("utf-8")).digest()
        x = int.from_bytes(digest[:8], "big")
        index = x >> (64 - self.p)
        remaining_bits = 64 - self.p
        remaining = x & ((1 << remaining_bits) - 1)
        if remaining == 0:
            rank = remaining_bits + 1
        else:
            rank = remaining_bits - remaining.bit_length() + 1
        self.registers[index] = max(self.registers[index], rank)

    def estimate(self):
        m = self.m
        alpha = 0.7213 / (1.0 + 1.079 / m)
        denominator = sum(2.0 ** (-r) for r in self.registers)
        raw = alpha * m * m / denominator
        zeros = self.registers.count(0)
        if raw <= 2.5 * m and zeros:
            return m * math.log(m / zeros)
        return raw


# ----------------------------
# Count-Min Sketch (educational)
# ----------------------------
class CountMinSketch:
    def __init__(self, width=200, depth=5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)

    def _index(self, item, row):
        key = f"{row}:{item}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:8], "big") % self.width

    def add(self, item, count=1):
        for row in range(self.depth):
            self.table[row, self._index(item, row)] += count

    def estimate(self, item):
        return int(
            min(
                self.table[row, self._index(item, row)]
                for row in range(self.depth)
            )
        )


# ----------------------------
# Cleaning and validation
# ----------------------------
def prepare_data(dataframe):
    missing = [c for c in REQUIRED_COLUMNS if c not in dataframe.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    data = dataframe[REQUIRED_COLUMNS].copy()
    data["caller_id"] = data["caller_id"].astype("string").str.strip()
    data["receiver_id"] = data["receiver_id"].astype("string").str.strip()
    data["call_duration_sec"] = pd.to_numeric(
        data["call_duration_sec"], errors="coerce"
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["call_type"] = data["call_type"].astype("string").str.strip()
    data["cell_tower"] = data["cell_tower"].astype("string").str.strip()

    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[
        (data["caller_id"] != "")
        & (data["receiver_id"] != "")
        & (data["call_type"] != "")
        & (data["cell_tower"] != "")
        & (data["call_duration_sec"] >= 0)
    ]
    return data.drop_duplicates().reset_index(drop=True)


# ----------------------------
# Upload first: no analysis until valid file uploaded
# ----------------------------
st.header("Upload CDR Dataset")
uploaded_file = st.file_uploader(
    "Choose a telecom CDR CSV file",
    type=["csv"],
    help=(
        "Required columns: caller_id, receiver_id, call_duration_sec, "
        "timestamp, call_type, cell_tower."
    ),
)

if uploaded_file is None:
    st.info("Upload a CSV file above to unlock the analysis sections.")
    st.markdown(
        """
        **Required CSV columns**
        - `caller_id`
        - `receiver_id`
        - `call_duration_sec`
        - `timestamp`
        - `call_type`
        - `cell_tower`
        """
    )
    st.stop()

try:
    raw_data = pd.read_csv(uploaded_file)
    cdr = prepare_data(raw_data)
except Exception as error:
    st.error(f"Could not load or clean the CSV: {error}")
    st.stop()

if cdr.empty:
    st.error("No valid records remain after cleaning. Please check your CSV.")
    st.stop()

st.success(
    f"Dataset uploaded successfully: {len(cdr):,} valid records from "
    f"{uploaded_file.name}"
)
st.caption(
    f"Rows read: {len(raw_data):,} • Rows removed during cleaning: "
    f"{len(raw_data) - len(cdr):,}"
)

# Compute reusable outputs once per Streamlit run
exact_unique = int(cdr["caller_id"].nunique())
hll = HyperLogLog(precision=6)
cms = CountMinSketch(width=200, depth=5)
for caller in cdr["caller_id"]:
    hll.add(caller)
    cms.add(caller)

hll_estimate = int(round(hll.estimate()))
hll_abs_error = abs(hll_estimate - exact_unique)
hll_pct_error = hll_abs_error / exact_unique * 100 if exact_unique else 0.0

exact_counts = cdr["caller_id"].value_counts()
top_callers = exact_counts.head(10)
frequency = pd.DataFrame({
    "caller_id": top_callers.index.astype(str),
    "exact_calls": top_callers.values,
})
frequency["cms_estimate"] = frequency["caller_id"].map(cms.estimate)
frequency["absolute_error"] = (
    frequency["cms_estimate"] - frequency["exact_calls"]
).abs()

caller_features = (
    cdr.groupby("caller_id")
    .agg(
        total_calls=("receiver_id", "count"),
        avg_call_duration=("call_duration_sec", "mean"),
        total_call_duration=("call_duration_sec", "sum"),
        unique_receivers=("receiver_id", "nunique"),
    )
    .reset_index()
)

if len(caller_features) >= 3:
    feature_columns = [
        "total_calls",
        "avg_call_duration",
        "total_call_duration",
        "unique_receivers",
    ]
    scaled = StandardScaler().fit_transform(caller_features[feature_columns])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    caller_features["cluster"] = kmeans.fit_predict(scaled).astype(str)
    cluster_summary = (
        caller_features.groupby("cluster")
        .agg(
            caller_count=("caller_id", "count"),
            avg_calls=("total_calls", "mean"),
            avg_call_duration_sec=("avg_call_duration", "mean"),
            avg_unique_receivers=("unique_receivers", "mean"),
        )
        .reset_index()
        .sort_values("cluster")
    )
else:
    caller_features["cluster"] = "Not available"
    cluster_summary = pd.DataFrame()


# ----------------------------
# Expandable analysis sections
# ----------------------------
with st.expander("1. Data Cleaning & Validation", expanded=False):
    st.subheader("Cleaning summary")
    cleaning_metrics = st.columns(3)
    cleaning_metrics[0].metric("Rows uploaded", f"{len(raw_data):,}")
    cleaning_metrics[1].metric("Rows after cleaning", f"{len(cdr):,}")
    cleaning_metrics[2].metric(
        "Rows removed", f"{len(raw_data) - len(cdr):,}"
    )
    st.write("Missing values in the uploaded file:")
    st.dataframe(
        raw_data[REQUIRED_COLUMNS].isna().sum().rename("missing_values").to_frame(),
        use_container_width=True,
    )
    st.write("Preview of cleaned data:")
    st.dataframe(cdr.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        "Download cleaned CDR CSV",
        cdr.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_telecom_cdr.csv",
        mime="text/csv",
        key="download_cleaned",
    )

with st.expander("2. Dataset Exploration", expanded=False):
    metric_cols = st.columns(2)
    metric_cols[0].metric("Total CDR Records", f"{len(cdr):,}")
    metric_cols[1].metric("Exact Unique Callers", f"{exact_unique:,}")
    metric_cols = st.columns(2)
    metric_cols[0].metric(
        "Total Call Duration",
        f"{cdr['call_duration_sec'].sum() / 60:,.1f} min",
    )
    metric_cols[1].metric(
        "Average Call Duration",
        f"{cdr['call_duration_sec'].mean():,.1f} sec",
    )

    left, right = st.columns(2)
    with left:
        st.write("Calls by type")
        st.bar_chart(cdr["call_type"].value_counts())
    with right:
        st.write("Calls by cell tower")
        st.bar_chart(cdr["cell_tower"].value_counts())

    st.write("Caller frequency distribution (top 10)")
    st.dataframe(
        exact_counts.head(10).rename("exact_calls").to_frame(),
        use_container_width=True,
    )

with st.expander("3. HyperLogLog Analysis", expanded=False):
    hll_cols = st.columns(2)
    hll_cols[0].metric("Exact unique callers", f"{exact_unique:,}")
    hll_cols[1].metric(
        "HLL estimate",
        f"{hll_estimate:,}",
        delta=f"{hll_estimate - exact_unique:+,} vs exact",
    )
    st.dataframe(
        pd.DataFrame({
            "Metric": [
                "Exact unique callers",
                "HLL estimate (rounded)",
                "Absolute error",
                "Absolute percentage error",
                "Register count",
            ],
            "Value": [
                exact_unique,
                hll_estimate,
                hll_abs_error,
                f"{hll_pct_error:.2f}%",
                hll.m,
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "This is an educational HLL implementation. The observed error is "
        "specific to this dataset and run, not a universal accuracy guarantee."
    )

with st.expander("4. Count-Min Sketch Analysis", expanded=False):
    st.write("Top 10 callers: exact frequency versus CMS estimate")
    plot_data = frequency.sort_values("exact_calls").reset_index(drop=True)
    y_positions = np.arange(len(plot_data))
    bar_height = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        y_positions - bar_height / 2,
        plot_data["exact_calls"],
        bar_height,
        label="Exact calls",
    )
    ax.barh(
        y_positions + bar_height / 2,
        plot_data["cms_estimate"],
        bar_height,
        label="CMS estimate",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["caller_id"])
    ax.set_xlabel("Number of calls")
    ax.set_title("Exact Frequency vs Count-Min Sketch")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(frequency, use_container_width=True, hide_index=True)
    st.caption(
        "CMS can overestimate counts because of hash collisions. With "
        "non-negative updates, the standard estimate should not be below "
        "the exact frequency."
    )

with st.expander("5. K-Means Clustering", expanded=False):
    if len(caller_features) >= 3:
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(7, 5))
            for cluster_name in sorted(caller_features["cluster"].unique()):
                group = caller_features[
                    caller_features["cluster"] == cluster_name
                ]
                ax.scatter(
                    group["total_calls"],
                    group["avg_call_duration"],
                    label=f"Cluster {cluster_name}",
                    alpha=0.8,
                )
            ax.set_xlabel("Total calls per caller")
            ax.set_ylabel("Average call duration (seconds)")
            ax.set_title("Caller Behavior Clusters")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.write("Cluster summary")
            st.dataframe(
                cluster_summary.round(2),
                use_container_width=True,
                hide_index=True,
            )
        st.download_button(
            "Download caller clustering results CSV",
            caller_features.to_csv(index=False).encode("utf-8"),
            file_name="caller_clustering_results.csv",
            mime="text/csv",
            key="download_clusters",
        )
        st.caption(
            "Cluster IDs are arbitrary and do not automatically indicate "
            "fraud or suspicious behavior."
        )
    else:
        st.warning("At least 3 distinct callers are required for K-Means.")

with st.expander("6. Final Results & Downloads", expanded=False):
    st.subheader("Summary")
    summary = pd.DataFrame({
        "Metric": [
            "Valid CDR records",
            "Exact unique callers",
            "HLL estimate (rounded)",
            "HLL absolute percentage error",
            "Total call duration (minutes)",
        ],
        "Value": [
            len(cdr),
            exact_unique,
            hll_estimate,
            f"{hll_pct_error:.2f}%",
            round(cdr["call_duration_sec"].sum() / 60, 2),
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download cleaned CDR CSV",
        cdr.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_telecom_cdr.csv",
        mime="text/csv",
        key="download_final_cleaned",
    )
    if len(caller_features) >= 3:
        st.download_button(
            "Download caller clustering results CSV",
            caller_features.to_csv(index=False).encode("utf-8"),
            file_name="caller_clustering_results.csv",
            mime="text/csv",
            key="download_final_clusters",
        )

st.divider()
st.caption(
    "Educational prototype using synthetic/sample data. It is not a production "
    "telecom system and does not currently use distributed processing such as Apache Spark."
)
