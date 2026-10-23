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
st.caption("HyperLogLog | Count-Min Sketch | K-Means Clustering")
st.write(
    "Analyze synthetic or anonymized telecom Call Detail Records (CDRs) "
    "using approximate counting and caller-behavior clustering."
)


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
        raw_estimate = alpha * m * m / denominator
        zero_registers = self.registers.count(0)

        # Small-cardinality correction.
        if raw_estimate <= 2.5 * m and zero_registers:
            return m * math.log(m / zero_registers)

        return raw_estimate


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
# Data validation and cleaning
# ----------------------------
REQUIRED_COLUMNS = [
    "caller_id",
    "receiver_id",
    "call_duration_sec",
    "timestamp",
    "call_type",
    "cell_tower",
]


def prepare_data(dataframe):
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "The CSV is missing these required columns: " + ", ".join(missing)
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

    data = data.dropna(
        subset=[
            "caller_id",
            "receiver_id",
            "call_duration_sec",
            "timestamp",
            "call_type",
            "cell_tower",
        ]
    )
    data = data[
        (data["caller_id"] != "")
        & (data["receiver_id"] != "")
        & (data["call_type"] != "")
        & (data["cell_tower"] != "")
        & (data["call_duration_sec"] >= 0)
    ]
    return data.drop_duplicates().reset_index(drop=True)


# ----------------------------
# Sidebar: upload dataset
# ----------------------------
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CDR CSV (optional)",
    type=["csv"],
    help="CSV must contain caller_id, receiver_id, call_duration_sec, timestamp, call_type, and cell_tower.",
)

try:
    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name
    else:
        raw_data = pd.read_csv("cleaned_telecom_cdr.csv")
        source_label = "cleaned_telecom_cdr.csv"

    cdr = prepare_data(raw_data)
except FileNotFoundError:
    st.error(
        "The default dataset 'cleaned_telecom_cdr.csv' was not found. "
        "Upload a CSV using the sidebar."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if cdr.empty:
    st.error("No valid records remain after cleaning. Please check the CSV.")
    st.stop()

st.sidebar.success(f"Loaded {len(cdr):,} valid records")
st.sidebar.caption(f"Source: {source_label}")
st.sidebar.caption(f"Rows removed during cleaning: {len(raw_data) - len(cdr):,}")


# ----------------------------
# Summary and HLL
# ----------------------------
hll = HyperLogLog(precision=6)
cms = CountMinSketch(width=200, depth=5)

for caller in cdr["caller_id"]:
    hll.add(caller)
    cms.add(caller)

exact_unique = int(cdr["caller_id"].nunique())
estimated_unique = int(round(hll.estimate()))
hll_absolute_error = abs(estimated_unique - exact_unique)
hll_percentage_error = (
    hll_absolute_error / exact_unique * 100 if exact_unique else 0.0
)
total_duration_minutes = cdr["call_duration_sec"].sum() / 60

st.subheader("Dataset overview")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Total CDR Records", f"{len(cdr):,}")
metric2.metric("Exact Unique Callers", f"{exact_unique:,}")
metric3.metric(
    "HLL Estimate",
    f"{estimated_unique:,}",
    delta=f"{estimated_unique - exact_unique:+,} vs exact",
)
metric4.metric("Total Call Duration", f"{total_duration_minutes:,.1f} min")

with st.expander("HyperLogLog accuracy details", expanded=True):
    hll_table = pd.DataFrame(
        {
            "Metric": [
                "Exact unique callers",
                "HyperLogLog estimate (rounded)",
                "Absolute error",
                "Absolute percentage error",
            ],
            "Value": [
                exact_unique,
                estimated_unique,
                hll_absolute_error,
                f"{hll_percentage_error:.2f}%",
            ],
        }
    )
    st.dataframe(hll_table, use_container_width=True, hide_index=True)
    st.caption(
        "HLL is approximate. This error describes this dataset and this run only; "
        "it is not a general accuracy guarantee."
    )


# ----------------------------
# Count-Min Sketch analysis
# ----------------------------
st.subheader("1. Caller Frequency — Count-Min Sketch")
exact_counts = cdr["caller_id"].value_counts()
top_callers = exact_counts.head(10)

frequency = pd.DataFrame(
    {
        "caller_id": top_callers.index.astype(str),
        "exact_calls": top_callers.values,
    }
)
frequency["cms_estimate"] = frequency["caller_id"].map(cms.estimate)
frequency["absolute_error"] = (
    frequency["cms_estimate"] - frequency["exact_calls"]
).abs()

# Grouped horizontal bars: do not stack estimates of the same quantity.
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

st.write("Top 10 callers: exact counts and CMS estimates")
st.dataframe(frequency, use_container_width=True, hide_index=True)
st.caption(
    "Count-Min Sketch estimates are upper-bound estimates in the standard "
    "algorithm; collisions can cause overestimation."
)


# ----------------------------
# K-Means caller behavior
# ----------------------------
st.subheader("2. K-Means Caller Behavior Clusters")

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

feature_columns = [
    "total_calls",
    "avg_call_duration",
    "total_call_duration",
    "unique_receivers",
]

if len(caller_features) >= 3:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(caller_features[feature_columns])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    caller_features["cluster"] = kmeans.fit_predict(scaled_features).astype(str)

    cluster_counts = (
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

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cluster_name in sorted(caller_features["cluster"].unique()):
            group = caller_features[caller_features["cluster"] == cluster_name]
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

    with chart_right:
        st.write("Cluster summary")
        st.dataframe(
            cluster_counts.round(2),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Cluster labels are arbitrary. Interpret groups using their feature "
            "averages; clusters do not automatically indicate fraud."
        )

    st.download_button(
        "Download caller clustering results CSV",
        caller_features.to_csv(index=False).encode("utf-8"),
        file_name="caller_clustering_results.csv",
        mime="text/csv",
    )
else:
    st.warning("At least 3 distinct callers are required for K-Means clustering.")


# ----------------------------
# Call type and tower analysis
# ----------------------------
st.subheader("3. Call Type and Cell Tower Analysis")
left_column, right_column = st.columns(2)

with left_column:
    st.write("Calls by type")
    call_type_counts = cdr["call_type"].value_counts()
    st.bar_chart(call_type_counts)

with right_column:
    st.write("Calls by cell tower")
    tower_counts = cdr["cell_tower"].value_counts()
    st.bar_chart(tower_counts)


# ----------------------------
# Data preview and downloads
# ----------------------------
st.subheader("4. CDR Data Preview")
st.dataframe(cdr.head(100), use_container_width=True)

st.download_button(
    "Download cleaned CDR CSV",
    data=cdr.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_telecom_cdr.csv",
    mime="text/csv",
)

st.info(
    "Educational prototype using synthetic/sample data. It is not a production "
    "telecom system and does not currently use distributed processing such as Spark."
)
