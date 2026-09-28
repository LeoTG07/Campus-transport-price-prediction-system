"""
train_model.py
================
CAMPUS TRANSPORT PRICE PREDICTION SYSTEM
Model training script.

What this script does (in order):
1. Loads the historical dataset (data/campus_transport_dataset.csv)
2. Cleans and preprocesses it (duplicates, missing values, encoding)
3. Selects the two predictors used by the proposed system: Fuel Price and
   Festive Period (exactly as specified in Chapter 3, Figure 3.8)
4. Splits the data into training and testing sets
5. Trains TWO supervised regression models:
      - Linear Regression   (baseline)
      - Random Forest Regressor (ensemble)
6. Evaluates both models using MAE, MSE, RMSE and R²
7. Selects the best-performing model automatically
8. Saves the best model, both individual models, and the evaluation
   metrics to disk so the Streamlit app (app.py) can load them
9. Generates comparison charts into the outputs/ folder — these are the
   images you can use as figures/screenshots in Chapter 4 of your project.

Run it from the project folder with:
    python train_model.py
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # allows chart generation without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", palette="deep")

DATA_PATH = os.path.join("data", "campus_transport_dataset.csv")
MODEL_DIR = "model"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_data(path: str) -> pd.DataFrame:
    """Load the raw dataset and clean it (Data Preparation phase of CRISP-DM)."""
    df = pd.read_csv(path)

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 2. Drop rows with missing values in the columns we actually need
    required_cols = [
        "Year",
        "Fuel_Price_NGN_per_Litre",
        "Transport_Price_NGN",
        "Festive_Period",
    ]
    df = df.dropna(subset=required_cols)

    # 3. Parse "Year" (e.g. "Jan-2016") into a proper date for chronological
    #    sorting and for the fare-trend chart. This column is used for
    #    display/trend purposes only — it is NOT fed to the model.
    df["Period"] = pd.to_datetime(df["Year"], format="%b-%Y")
    df = df.sort_values("Period").reset_index(drop=True)

    # 4. Ensure correct data types (encoding step)
    df["Fuel_Price_NGN_per_Litre"] = df["Fuel_Price_NGN_per_Litre"].astype(float)
    df["Transport_Price_NGN"] = df["Transport_Price_NGN"].astype(float)
    df["Festive_Period"] = df["Festive_Period"].astype(int)  # already 0/1

    return df


def train_and_evaluate(df: pd.DataFrame):
    """Modelling + Evaluation phases of CRISP-DM."""

    feature_cols = ["Fuel_Price_NGN_per_Litre", "Festive_Period"]
    target_col = "Transport_Price_NGN"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Each model is wrapped in a Pipeline that (a) scales the fuel price
    # feature and (b) then fits the estimator. Bundling scaler + model
    # together means a single saved file can preprocess AND predict.
    candidate_models = {
        "Linear Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=8,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    results = {}
    predictions = {}

    for name, pipeline in candidate_models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "MAE": round(float(mae), 2),
            "MSE": round(float(mse), 2),
            "RMSE": round(rmse, 2),
            "R2": round(float(r2), 4),
        }
        predictions[name] = y_pred

        # save each individual trained pipeline
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{safe_name}.pkl"))

    # Best model = lowest MAE (ties broken by highest R2)
    best_name = min(
        results, key=lambda k: (results[k]["MAE"], -results[k]["R2"])
    )
    best_pipeline = candidate_models[best_name]
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.pkl"))

    metrics_out = {
        "results": results,
        "best_model": best_name,
        "feature_columns": feature_cols,
        "target_column": target_col,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(df)),
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    for name, m in results.items():
        marker = "  <-- SELECTED (best)" if name == best_name else ""
        print(f"{name}:{marker}")
        print(f"    MAE  = NGN {m['MAE']}")
        print(f"    MSE  = {m['MSE']}")
        print(f"    RMSE = NGN {m['RMSE']}")
        print(f"    R2   = {m['R2']}")
    print("=" * 60)
    print(f"Best model saved to: {MODEL_DIR}/best_model.pkl")
    print(f"Metrics saved to:    {MODEL_DIR}/metrics.json")

    return results, predictions, X_test, y_test, best_name


def generate_charts(df, results, predictions, X_test, y_test, best_name):
    """Generate figures for Chapter 4 screenshots / report use."""

    # --- Chart 1: Model comparison (MAE & RMSE) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = list(results.keys())
    mae_vals = [results[n]["MAE"] for n in names]
    rmse_vals = [results[n]["RMSE"] for n in names]
    r2_vals = [results[n]["R2"] for n in names]

    colors = ["#4C6EF5" if n != best_name else "#2F9E44" for n in names]

    axes[0].bar(names, mae_vals, color=colors)
    axes[0].set_title("Mean Absolute Error (lower is better)")
    axes[0].set_ylabel("MAE (NGN)")
    for i, v in enumerate(mae_vals):
        axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(names, r2_vals, color=colors)
    axes[1].set_title("R\u00b2 Score (higher is better)")
    axes[1].set_ylabel("R\u00b2")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(r2_vals):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", rotation=10)

    fig.suptitle("Model Comparison: Linear Regression vs Random Forest Regressor")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=200)
    plt.close(fig)

    # --- Chart 2: Actual vs Predicted (best model) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    y_pred_best = predictions[best_name]
    ax.scatter(y_test, y_pred_best, alpha=0.7, color="#2F9E44", edgecolor="white")
    lims = [
        min(y_test.min(), y_pred_best.min()) - 10,
        max(y_test.max(), y_pred_best.max()) + 10,
    ]
    ax.plot(lims, lims, "--", color="gray", label="Perfect Prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Transport Price (NGN)")
    ax.set_ylabel("Predicted Transport Price (NGN)")
    ax.set_title(f"Actual vs Predicted Fare — {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=200)
    plt.close(fig)

    # --- Chart 3: Historical fare trend over time ---
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(df["Period"], df["Transport_Price_NGN"], color="#4C6EF5", label="Transport Price (NGN)")
    festive_pts = df[df["Festive_Period"] == 1]
    ax1.scatter(
        festive_pts["Period"],
        festive_pts["Transport_Price_NGN"],
        color="#E8590C",
        zorder=5,
        label="Festive Period",
        s=25,
    )
    ax1.set_ylabel("Transport Price (NGN)", color="#4C6EF5")
    ax1.set_xlabel("Period")

    ax2 = ax1.twinx()
    ax2.plot(df["Period"], df["Fuel_Price_NGN_per_Litre"], color="#AE3EC9", alpha=0.5, label="Fuel Price (NGN/L)")
    ax2.set_ylabel("Fuel Price (NGN/Litre)", color="#AE3EC9")

    fig.suptitle("Historical Campus Transport Price vs Fuel Price (2016 - 2026)")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fare_trend_history.png"), dpi=200)
    plt.close(fig)

    # --- Chart 4: Fuel Price vs Transport Price relationship ---
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="Fuel_Price_NGN_per_Litre",
        y="Transport_Price_NGN",
        hue=df["Festive_Period"].map({0: "Non-Festive", 1: "Festive"}),
        palette={"Non-Festive": "#4C6EF5", "Festive": "#E8590C"},
        ax=ax,
    )
    ax.set_title("Fuel Price vs Transport Price by Festive Status")
    ax.set_xlabel("Fuel Price (NGN/Litre)")
    ax.set_ylabel("Transport Price (NGN)")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_relationship.png"), dpi=200)
    plt.close(fig)

    print(f"Charts saved to: {OUTPUT_DIR}/")


def build_reference_fee_table(df: pd.DataFrame):
    """Save the most recent flat fee per vehicle type as a small reference
    table. These fees are NOT model-predicted (they don't vary with
    distance) — they are shown in the app purely for context."""
    latest = df.sort_values("Period").iloc[-1]
    fee_table = {
        "as_of": latest["Year"],
        "fuel_price": float(latest["Fuel_Price_NGN_per_Litre"]),
        "fees": {
            "17-Seater Bus": int(latest["Fee_Bus_17Seater_NGN"]),
            "7-Seater Shuttle": int(latest["Fee_ShuttleBus_7Seater_NGN"]),
            "4-Seater Tricycle (Keke)": int(latest["Fee_Tricycle_4Seater_NGN"]),
        },
    }
    with open(os.path.join(MODEL_DIR, "reference_fees.json"), "w") as f:
        json.dump(fee_table, f, indent=2)


def main():
    print("Loading and preprocessing dataset...")
    df = load_and_preprocess_data(DATA_PATH)
    print(f"Dataset ready: {len(df)} rows after cleaning.\n")

    results, predictions, X_test, y_test, best_name = train_and_evaluate(df)

    print("\nGenerating charts...")
    generate_charts(df, results, predictions, X_test, y_test, best_name)

    build_reference_fee_table(df)

    # also save the cleaned dataset for the app to reuse (with Period column)
    df.to_csv(os.path.join(MODEL_DIR, "clean_dataset.csv"), index=False)

    print("\nAll done. You can now run the Streamlit app with:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
