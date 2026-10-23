import hashlib
import math
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Telecom CDR Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telecom CDR Big Data Analytics")
st.caption(
    "HyperLogLog | Count-Min Sketch | K-Means Clustering"
)

class HyperLogLog:
    def __init__(self, precision=6):
        self.p = precision
        self.m = 1 << precision
        self.registers = [0] * self.m

    def add(self, item):
        digest = hashlib.sha256(
            str(item).encode()
        ).digest()
        x = int.from_bytes(digest[:8], "big")
        index = x >> (64 - self.p)
        remaining = x & ((1 << (64 - self.p)) - 1)
        width = 64 - self.p
        rank = (
            width + 1 if remaining == 0
            else width - remaining.bit_length() + 1
        )
        self.registers[index] = max(
            self.registers[index], rank
        )

    def estimate(self):
        m = self.m
        alpha = 0.7213 / (1 + 1.079 / m)
        denominator = sum(
            2.0 ** (-r) for r in self.registers
        )
        raw = alpha * m * m / denominator
        zeros = self.registers.count(0)
        if raw <= 2.5 * m and zeros:
            return m * math.log(m / zeros)
        return raw


class CountMinSketch:
    def __init__(self, width=200, depth=5):
        self.width = width
        self.depth = depth
        self.table = np.zeros(
            (depth, width), dtype=np.int64
        )

    def index(self, item, row):
        key = f"{row}:{item}".encode()
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:8], "big") % self.width

    def add(self, item):
        for row in range(self.depth):
            self.table[row, self.index(item, row)] += 1

    def estimate(self, item):
        return int(min(
            self.table[row, self.index(item, row)]
            for row in range(self.depth)
        ))


def prepare_data(df):
    required = [
        "caller_id", "receiver_id",
        "call_duration_sec", "timestamp",
        "call_type", "cell_tower"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    df = df[required].copy()
    df["caller_id"] = df["caller_id"].astype("string").str.strip()
    df["receiver_id"] = df["receiver_id"].astype("string").str.strip()
    df["call_duration_sec"] = pd.to_numeric(
        df["call_duration_sec"], errors="coerce"
    )
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna(subset=[
        "caller_id", "receiver_id",
        "call_duration_sec", "timestamp"
    ])
    df = df[
        (df["caller_id"] != "") &
        (df["receiver_id"] != "") &
        (df["call_duration_sec"] >= 0)
    ]
    return df.drop_duplicates().reset_index(drop=True)


uploaded = st.sidebar.file_uploader(
    "Upload CDR CSV (optional)", type=["csv"]
)

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_csv("cleaned_telecom_cdr.csv")

    cdr = prepare_data(raw)
except FileNotFoundError:
    st.error(
        "Dataset not found. Upload a CDR CSV using the sidebar."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if cdr.empty:
    st.error("No valid CDR records remain after cleaning.")
    st.stop()

st.sidebar.success(f"{len(cdr):,} valid records loaded")

# Summary metrics
hll = HyperLogLog(precision=6)
cms = CountMinSketch(width=200, depth=5)

for caller in cdr["caller_id"]:
    hll.add(caller)
    cms.add(caller)

exact_unique = cdr["caller_id"].nunique()
estimated_unique = round(hll.estimate())

a, b, c, d = st.columns(4)
a.metric("Total CDR Records", f"{len(cdr):,}")
b.metric("Exact Unique Callers", f"{exact_unique:,}")
c.metric("HLL Estimate", f"{estimated_unique:,}")
d.metric(
    "Total Call Duration",
    f"{cdr['call_duration_sec'].sum() / 60:,.1f} min"
)

st.subheader("1. Caller Frequency — Count-Min Sketch")

exact_counts = cdr["caller_id"].value_counts()
top_callers = exact_counts.head(10)

freq = pd.DataFrame({
    "caller_id": top_callers.index,
    "exact_calls": top_callers.values
})
freq["cms_estimate"] = freq["caller_id"].map(cms.estimate)

st.bar_chart(
    freq.set_index("caller_id")[["exact_calls", "cms_estimate"]]
)
st.dataframe(freq, use_container_width=True, hide_index=True)

st.subheader("2. K-Means Caller Behavior Clusters")

features = cdr.groupby("caller_id").agg(
    total_calls=("receiver_id", "count"),
    avg_call_duration=("call_duration_sec", "mean"),
    total_call_duration=("call_duration_sec", "sum"),
    unique_receivers=("receiver_id", "nunique")
).reset_index()

if len(features) >= 3:
    columns = [
        "total_calls", "avg_call_duration",
        "total_call_duration", "unique_receivers"
    ]
    scaled = StandardScaler().fit_transform(features[columns])
    model = KMeans(
        n_clusters=3, random_state=42, n_init=10
    )
    features["cluster"] = model.fit_predict(scaled).astype(str)

    fig, ax = plt.subplots(figsize=(8, 4))
    for cluster in sorted(features["cluster"].unique()):
        group = features[features["cluster"] == cluster]
        ax.scatter(
            group["total_calls"],
            group["avg_call_duration"],
            label=f"Cluster {cluster}"
        )
    ax.set_xlabel("Total Calls")
    ax.set_ylabel("Average Duration (seconds)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.write("Callers in each cluster")
    st.dataframe(
        features["cluster"].value_counts()
        .sort_index().rename("caller_count").to_frame(),
        use_container_width=True
    )
    st.download_button(
        "Download clustering results CSV",
        features.to_csv(index=False).encode("utf-8"),
        file_name="caller_clustering_results.csv",
        mime="text/csv"
    )
else:
    st.warning("At least 3 distinct callers are needed for clustering.")

st.subheader("3. Call Type and Cell Tower Analysis")
left, right = st.columns(2)
with left:
    st.write("Calls by type")
    st.bar_chart(cdr["call_type"].value_counts())
with right:
    st.write("Calls by cell tower")
    st.bar_chart(cdr["cell_tower"].value_counts())

st.subheader("4. CDR Data Preview")
st.dataframe(cdr.head(100), use_container_width=True)

st.download_button(
    "Download cleaned CDR CSV",
    cdr.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_telecom_cdr.csv",
    mime="text/csv"
)

st.info(
    "Educational prototype using approximate sketches and "
    "synthetic/sample data. It is not a production telecom system."
)
