"""Lab 5 — Predict EC2 On-Demand pricing from memory and vCPU specifications."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# A small NumPy-based implementation of the course's linear regression approach.
# Avoids scikit-learn and SciPy native DLLs, which some Windows security policies block.

class LinearRegression:
    """Ordinary least-squares regression with two predictors and an intercept."""

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)
        x1 = values[:, 0]
        x2 = values[:, 1]
        x1_mean, x2_mean, y_mean = x1.mean(), x2.mean(), targets.mean()
        a = x1 - x1_mean
        b = x2 - x2_mean
        z = targets - y_mean
        s11 = float(np.sum(a * a))
        s22 = float(np.sum(b * b))
        s12 = float(np.sum(a * b))
        s1y = float(np.sum(a * z))
        s2y = float(np.sum(b * z))
        determinant = s11 * s22 - s12 * s12
        if determinant <= 0:
            raise ValueError("Cannot fit regression because both predictors are collinear.")
        weight1 = (s1y * s22 - s2y * s12) / determinant
        weight2 = (s2y * s11 - s1y * s12) / determinant
        self.coef_ = np.array([weight1, weight2])
        self.intercept_ = float(y_mean - weight1 * x1_mean - weight2 * x2_mean)
        return self

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        return self.intercept_ + values[:, 0] * self.coef_[0] + values[:, 1] * self.coef_[1]


def train_test_split(X, y, test_size=0.2, random_state=42):
    """Reproducible shuffled 80/20 split compatible with the activity's example."""
    shuffled = np.random.RandomState(random_state).permutation(len(X))
    n_test = int(np.ceil(test_size * len(X)))
    test_indices, train_indices = shuffled[:n_test], shuffled[n_test:]
    return (X.iloc[train_indices], X.iloc[test_indices],
            y.iloc[train_indices], y.iloc[test_indices])


def mean_absolute_error(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def mean_squared_error(actual, predicted):
    residual = np.asarray(actual) - np.asarray(predicted)
    return float(np.mean(residual ** 2))


def r2_score(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    numerator = float(np.sum((actual - predicted) ** 2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return 1 - numerator / denominator if denominator > 0 else float("nan")

st.set_page_config(page_title="EC2 Cost Prediction | Lab 5", page_icon="☁️", layout="wide")

DATA_FILE = Path(__file__).with_name("ec2dataset.csv")
MODEL_NAMES = ["Positive log-linear regression", "Activity baseline: linear regression"]


@st.cache_data(show_spinner=False)
def load_ec2_data():
    """Use the same 861-row EC2 dataset as the previous activity."""
    df = pd.read_csv(DATA_FILE)
    df["Memory_GiB"] = pd.to_numeric(
        df["Instance Memory"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
    )
    df["vCPU_Count"] = pd.to_numeric(
        df["vCPUs"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    # Strings such as '$0.096 hourly' and 'unavailable' appear in the dataset.
    cost_columns = [
        "On Demand", "Linux Reserved cost", "Linux Spot Minimum cost",
        "Windows On Demand cost", "Windows Reserved cost"
    ]
    for col in cost_columns:
        df[col + "_USD"] = pd.to_numeric(
            df[col].astype(str).str.extract(r"\$\s*([\d,]+(?:\.\d+)?)")[0]
            .str.replace(",", "", regex=False), errors="coerce"
        )
    df["Instance_Family"] = df["API Name"].astype(str).str.split(".").str[0]
    valid = df.dropna(subset=["Memory_GiB", "vCPU_Count", "On Demand_USD"]).copy()
    valid = valid.loc[
        (valid["Memory_GiB"] > 0)
        & (valid["vCPU_Count"] > 0)
        & (valid["On Demand_USD"] > 0)
    ].copy()
    valid["Monthly_On_Demand"] = valid["On Demand_USD"] * 730
    return df, valid


@st.cache_resource(show_spinner=False)
def train_models(cleaned_data):
    """Build the activity's exact baseline plus a positive-estimate alternative.

    The two models share the same reproducible 80/20 split for fair comparison.
    """
    X = cleaned_data[["Memory_GiB", "vCPU_Count"]]
    y = cleaned_data["On Demand_USD"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    baseline = LinearRegression().fit(X_train, y_train)
    # Log-linear regression makes mathematically positive price predictions.
    # It is still linear regression, fitted in log-log space.
    positive = LinearRegression().fit(np.log(X_train), np.log(y_train))
    predictions = {
        MODEL_NAMES[0]: np.exp(positive.predict(np.log(X_test))),
        MODEL_NAMES[1]: baseline.predict(X_test),
    }
    metrics = {}
    for name, values in predictions.items():
        metrics[name] = {
            "MAE": float(mean_absolute_error(y_test, values)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, values))),
            "R2": float(r2_score(y_test, values)),
            "Negative Test Predictions": int(np.sum(values < 0)),
        }
    return baseline, positive, X_train, X_test, y_train, y_test, predictions, metrics


try:
    raw_df, data = load_ec2_data()
except FileNotFoundError:
    st.error("ec2dataset.csv is missing. Put it next to app.py in the GitHub repository.")
    st.stop()
except Exception as exc:
    st.error(f"The EC2 dataset could not be loaded: {exc}")
    st.stop()

if len(data) < 10:
    st.error("Not enough rows have usable memory, vCPU and positive On-Demand pricing.")
    st.stop()

baseline, positive, X_train, X_test, y_train, y_test, test_preds, test_metrics = train_models(data)

st.title("☁️ EC2 Instance Cost Prediction")
st.caption("Cloud Economics · Lab 5 · Streamlit regression dashboard")
st.write(
    "Predict Linux **On-Demand hourly EC2 cost** from **memory (GiB)** and **vCPUs**, "
    "evaluate regression performance, and explore cost-aware EC2 alternatives."
)

header_cols = st.columns(4)
header_cols[0].metric("Dataset rows", f"{len(raw_df):,}")
header_cols[1].metric("Usable priced rows", f"{len(data):,}")
header_cols[2].metric("Training rows", f"{len(X_train):,}")
header_cols[3].metric("Testing rows", f"{len(X_test):,}")
st.caption(
    f"{len(raw_df) - len(data)} rows are excluded from model training because of "
    "missing, unavailable, or invalid values in the required fields. "
    "An 80/20 train/test split (random_state=42) is used."
)

st.sidebar.title("Prediction settings")
method = st.sidebar.selectbox(
    "Regression method", MODEL_NAMES,
    help="The activity baseline uses untransformed features and costs. The optional "
         "log-linear model prevents negative predictions by predicting log(price).",
)
hours = st.sidebar.number_input(
    "Hours per month", min_value=1, max_value=744, value=730, step=1
)
st.sidebar.info("The CSV is a teaching dataset, not a live AWS price feed. All values are USD.")

predict_tab, eval_tab, insights_tab, data_tab = st.tabs(
    ["🎯 Cost prediction", "📊 Model evaluation", "💡 Cost optimization", "📄 Dataset"]
)

with predict_tab:
    st.subheader("Predict the hourly price of a new EC2 configuration")
    st.caption("Enter a hypothetical EC2 configuration. The activity's example uses 4 GiB and 2 vCPUs.")
    left, right = st.columns(2)
    with left:
        memory = st.number_input(
            "Memory (GiB)", min_value=0.5, max_value=32768.0,
            value=4.0, step=0.5, format="%.1f"
        )
    with right:
        cpu = st.number_input(
            "Virtual CPUs (vCPUs)", min_value=1, max_value=896,
            value=2, step=1
        )

    candidate = pd.DataFrame({"Memory_GiB": [float(memory)], "vCPU_Count": [float(cpu)]})
    if method == MODEL_NAMES[0]:
        predicted = float(np.exp(positive.predict(np.log(candidate)))[0])
    else:
        predicted = float(baseline.predict(candidate)[0])

    if predicted < 0:
        st.error(
            f"This basic linear model predicts ${predicted:.4f}/hour, an impossible negative "
            "price. This is a limitation of the simple baseline. Switch to **Positive log-linear "
            "regression** in the sidebar to get a positive estimate."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted hourly cost", f"${predicted:,.4f}")
        c2.metric(f"Estimated cost · {hours} h", f"${predicted * hours:,.2f}")
        c3.metric("Annualized estimate", f"${predicted * hours * 12:,.2f}")

    st.caption(
        "These are statistical estimates, NOT AWS quotes. Memory and vCPUs alone do not capture "
        "instance family, CPU generation, operating system, architecture, network, region, or "
        "discount commitments. Never use predictions alone for purchasing decisions."
    )
    mem_min, mem_max = float(data["Memory_GiB"].min()), float(data["Memory_GiB"].max())
    cpu_min, cpu_max = float(data["vCPU_Count"].min()), float(data["vCPU_Count"].max())
    if not (mem_min <= memory <= mem_max and cpu_min <= cpu <= cpu_max):
        st.warning("You are extrapolating beyond the training-data feature range.")

    st.markdown("#### Similar real configurations in the uploaded dataset")
    matching = data.loc[
        (data["vCPU_Count"] == cpu) & (data["Memory_GiB"] == memory)
    ].sort_values("On Demand_USD")
    comparison_cols = [
        "API Name", "Memory_GiB", "vCPU_Count", "Network Performance",
        "On Demand_USD", "Linux Reserved cost_USD", "Linux Spot Minimum cost_USD"
    ]
    if matching.empty:
        st.info("No row matches both specifications exactly. Try another configuration.")
    else:
        st.write(
            f"{len(matching)} instance type(s) have exactly {memory:g} GiB "
            f"and {cpu} vCPUs in this snapshot. Their actual listed prices vary by instance type."
        )
        st.dataframe(matching[comparison_cols].head(12), use_container_width=True, hide_index=True)

with eval_tab:
    st.subheader("How accurate are the predictions?")
    m = test_metrics[method]
    mcols = st.columns(4)
    mcols[0].metric("MAE", f"${m['MAE']:.4f}/h")
    mcols[1].metric("RMSE", f"${m['RMSE']:.4f}/h")
    mcols[2].metric("R²", f"{m['R2']:.3f}")
    mcols[3].metric("Negative test predictions", m["Negative Test Predictions"])
    st.caption(
        "MAE and RMSE measure typical and larger errors (lower is better). "
        "R² measures overall variance explained, but can be heavily influenced by extremely "
        "expensive instances. Negative estimates are invalid prices."
    )

    viz = pd.DataFrame(
        {"Actual cost ($/h)": y_test.to_numpy(),
         "Predicted cost ($/h)": test_preds[method]}
    )
    fig = px.scatter(
        viz, x="Actual cost ($/h)", y="Predicted cost ($/h)",
        title="Actual vs Predicted On-Demand Hourly Cost", opacity=0.75
    )
    axis_max = float(max(viz["Actual cost ($/h)"].max(), viz["Predicted cost ($/h)"].max()))
    fig.add_shape(
        type="line", x0=0, y0=0, x1=axis_max, y1=axis_max,
        line={"color": "#D04D48", "dash": "dash", "width": 2}
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The dashed line represents a perfect prediction (actual price = predicted price).")

    st.markdown("#### Compare both regression approaches")
    model_metrics = pd.DataFrame(test_metrics).T.rename_axis("Model").reset_index()
    st.dataframe(
        model_metrics.style.format({"MAE": "${:.4f}", "RMSE": "${:.4f}", "R2": "{:.3f}"}),
        use_container_width=True, hide_index=True
    )
    st.info(
        "The course activity uses ordinary linear regression directly on memory and vCPU values. "
        "That baseline is shown for fidelity. The optional log-linear regression also uses "
        "ordinary least-squares regression but models the logarithm of price and specs to ensure "
        "positive price predictions. Its metrics may be worse on very large outlier instances."
    )

with insights_tab:
    st.subheader("Pricing trends and potential cost optimization")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            data, x="Memory_GiB", y="On Demand_USD", color="Instance_Family",
            hover_name="API Name", title="Memory vs On-Demand hourly price"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            data, x="On Demand_USD", nbins=40,
            title="Hourly On-Demand price distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Shortlist instances meeting your minimum memory and CPU needs")
    eligible = data.loc[
        (data["Memory_GiB"] >= memory) & (data["vCPU_Count"] >= cpu)
    ].copy()
    eligible = eligible.sort_values("On Demand_USD")
    st.write(f"**{len(eligible)}** dataset instances meet or exceed {memory:g} GiB and {cpu} vCPUs.")
    if not eligible.empty:
        st.dataframe(
            eligible[["API Name", "Memory_GiB", "vCPU_Count", "Network Performance",
                      "On Demand_USD", "Linux Reserved cost_USD", "Linux Spot Minimum cost_USD"]].head(10),
            use_container_width=True, hide_index=True
        )

    st.markdown("#### Recommendations")
    st.markdown(
        "1. **Right-size compute and memory.** Use the shortlist to find lower-price instances "
        "that still meet the workload's actual CPU and RAM requirements. Confirm observed "
        "utilization and test performance before downsizing.\n"
        "2. **Compare purchasing models.** On-Demand provides flexibility; Reserved pricing "
        "can help with stable, predictable usage; Spot minimum prices may be attractive for "
        "interruptible workloads. Compare comparable terms before claiming savings.\n"
        "3. **Validate predictions against real instance prices.** Predicted cost is a starting "
        "point, not a substitute for actual region-specific AWS quotes or network/storage needs."
    )

with data_tab:
    st.subheader("Dataset and cleaning summary")
    st.write(f"Original CSV: **{len(raw_df)}** rows and **11** source columns")
    st.write(f"Rows usable for regression: **{len(data)}**")
    st.write(f"Rows excluded: **{len(raw_df) - len(data)}**")
    st.dataframe(
        raw_df[["API Name", "Memory_GiB", "vCPU_Count", "On Demand_USD",
                "Linux Reserved cost_USD", "Linux Spot Minimum cost_USD"]].head(25),
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "Download cleaned EC2 dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="ec2_cleaned_for_prediction.csv", mime="text/csv"
    )

st.divider()
st.caption(
    "Lab 5 · EC2 cost prediction · Models are fitted to the provided historical/example "
    "CSV snapshot. Monthly extrapolation uses the selected hours; default 730."
)
