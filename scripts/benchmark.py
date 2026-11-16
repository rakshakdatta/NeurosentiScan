"""Run a reproducible benchmark on the three-class EEG feature CSV."""

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
ML_MODELS = ("dummy", "logistic", "svm", "knn", "naive_bayes", "tree", "random_forest", "random_forest_100", "xgboost")
DL_MODELS = ("cnn", "lstm", "cnn_lstm", "bilstm", "eegnet_style")


def load_csv(path, group_column=None):
    frame = pd.read_csv(path)
    if "label" not in frame or set(frame["label"]) != set(LABELS):
        raise ValueError(f"Expected a label column with these classes: {LABELS}")
    if group_column == "label":
        raise ValueError("The label column cannot identify participants.")
    groups = None
    if group_column:
        groups = frame.pop(group_column)
        if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
            raise ValueError("Every row needs a participant identifier.")
    features = frame.drop(columns="label").astype("float32")
    if features.empty or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Expected a nonempty matrix of finite numeric features.")
    return features, LabelEncoder().fit(LABELS).transform(frame["label"]), groups


def split_rows(features, labels, groups=None, seed=SEED):
    """Split first and keep exact duplicate feature rows together."""
    fingerprints = pd.util.hash_pandas_object(features.replace(-0.0, 0.0), index=False)
    if groups is None and fingerprints.duplicated().any():
        groups = fingerprints
        protocol = "row-level, grouped by exact feature duplicates"
    elif groups is None:
        protocol = "stratified row-level"
    else:
        protocol = "participant-held-out"
        pairing = pd.DataFrame({"fingerprint": fingerprints, "group": groups})
        if pairing.groupby("fingerprint")["group"].nunique().gt(1).any():
            raise ValueError("Duplicate feature rows span participants. Resolve these before splitting.")

    rows = np.arange(len(labels))

    def take_fold(pool, folds):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for keep, hold in splitter.split(pool, labels[pool], np.asarray(groups)[pool]):
            if len(np.unique(labels[pool[keep]])) == len(LABELS) and len(np.unique(labels[pool[hold]])) == len(LABELS):
                yield pool[keep], pool[hold]

    if groups is None:
        rest, test = train_test_split(rows, test_size=0.2, stratify=labels, random_state=seed)
        train, validation = train_test_split(rest, test_size=0.25, stratify=labels[rest], random_state=seed)
    else:
        for rest, test in take_fold(rows, 5):
            valid = next(take_fold(rest, 4), None) if len(set(np.asarray(groups)[rest])) >= 4 else None
            if valid is not None:
                train, validation = valid
                break
        else:
            raise ValueError("Cannot form three group-separated partitions covering all classes.")
    return train, validation, test, protocol


def classical_model(name):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=4, random_state=SEED, n_jobs=2)
    estimators = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1500, random_state=SEED),
        "svm": SVC(),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "tree": DecisionTreeClassifier(random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=2),
    }
    if name == "random_forest_100":
        return make_pipeline(SelectKBest(f_classif, k=100), estimators["random_forest"])
    estimator = estimators[name]
    if name in {"logistic", "svm", "knn"}:
        return make_pipeline(StandardScaler(), estimator)
    return estimator


def train_neural(name, tf, X_train, y_train, X_val, y_val, epochs):
    keras = tf.keras
    layers = keras.layers
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    inputs = keras.Input(shape=(49, 52))
    if name == "eegnet_style":
        x = layers.Reshape((49, 52, 1))(inputs)
        x = layers.Conv2D(8, (1, 7), padding="same", activation="elu")(x)
        x = layers.DepthwiseConv2D((49, 1), activation="elu")(x)
        x = layers.SeparableConv2D(16, (1, 7), padding="same", activation="elu")(x)
        x = layers.AveragePooling2D((1, 2))(x)
    elif name == "bilstm":
        x = layers.Bidirectional(layers.LSTM(32))(inputs)
    elif name == "lstm":
        x = layers.LSTM(48)(inputs)
    else:
        x = layers.Conv1D(48, 3, activation="relu")(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(32, 3, activation="relu")(x)
        if name == "cnn_lstm":
            x = layers.LSTM(32)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(48, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    model = keras.Model(inputs, layers.Dense(len(LABELS), activation="softmax")(x))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Explicit datasets keep thread usage bounded in CPU-only environments.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2

    def batches(X, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if training:
            ds = ds.shuffle(len(y), seed=SEED)
        return ds.batch(64).with_options(options)

    model.fit(batches(X_train, y_train, True), validation_data=batches(X_val, y_val),
              epochs=epochs, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    return model


def paired_summary(predictions, actual):
    rows = []
    for (name_a, a), (name_b, b) in combinations(predictions.items(), 2):
        a_correct = a == actual
        b_correct = b == actual
        a_only = int(np.sum(a_correct & ~b_correct))
        b_only = int(np.sum(b_correct & ~a_correct))
        discordant = a_only + b_only
        p = binomtest(a_only, discordant).pvalue if discordant else 1.0
        rows.append((name_a, name_b, a_only, b_only, p))
    result = pd.DataFrame(rows, columns=["model_a", "model_b", "a_only_correct", "b_only_correct", "p_exact_mcnemar"])
    if not result.empty:
        order = result["p_exact_mcnemar"].sort_values().index
        adjusted = np.maximum.accumulate(result.loc[order, "p_exact_mcnemar"].to_numpy() * np.arange(len(result), 0, -1))
        result.loc[order, "p_holm"] = np.minimum(1, adjusted)
    return result


def plot_reports(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = result.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 5.6), layout="constrained")
    colors = data["family"].map({"baseline": "#84939c", "ml": "#166a61", "dl": "#485c9f", "ensemble": "#9d582d"})
    bars = ax.barh(data["model"], data["macro_f1"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.set(xlabel="Held-out macro F1", xlim=(0, 1.12), title="NeuroSentiScan model benchmark")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "benchmark.png", dpi=160)
    plt.close(fig)


def explain_forest(model, test, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    samples = test.sample(n=min(64, len(test)), random_state=SEED)
    explanation = shap.TreeExplainer(model)(samples)
    importance = pd.DataFrame({
        "feature": test.columns,
        "mean_absolute_shap": np.abs(explanation.values).mean(axis=(0, 2)),
    }).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(output / "shap_importance.csv", index=False)
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.barh(top["feature"], top["mean_absolute_shap"], color="#166a61")
    ax.set(title="Random Forest SHAP on held-out EEG features",
           xlabel="Mean absolute SHAP value across samples and classes")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "shap_eeg.png", dpi=160)
    plt.close(fig)


def run(args):
    X, y, groups = load_csv(args.csv, args.group_column)
    train, validation, test, protocol = split_rows(X, y, groups)
    models = list(dict.fromkeys(args.models))
    neural = any(name in DL_MODELS for name in models)
    if neural and X.shape[1] != 2548:
        raise ValueError("Neural models require the 2,548-column feature layout (49 x 52).")

    if neural:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.keras.utils.set_random_seed(SEED)
        tf.config.experimental.enable_op_determinism()
        scaler = StandardScaler().fit(X.iloc[train])
        X_train, X_val, X_test = [
            scaler.transform(X.iloc[rows]).reshape(-1, 49, 52) for rows in (train, validation, test)
        ]

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "protocol": protocol, "group_column": args.group_column, "seed": SEED,
        "rows": len(X), "features": X.shape[1], "epochs_limit": args.epochs,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "models": models, "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib")},
    }
    for package, enabled in (("tensorflow", neural), ("keras", neural), ("xgboost", "xgboost" in models), ("shap", args.shap)):
        if enabled:
            metadata["versions"][package] = version(package)
    splits = pd.DataFrame({"row_id": np.concatenate((train, validation, test)),
                           "partition": np.repeat(["train", "validation", "test"], [len(train), len(validation), len(test)])})
    splits.sort_values("row_id").to_csv(args.output / "split.csv", index=False)
    if args.save_models:
        model_dir = args.output / "models"
        model_dir.mkdir(exist_ok=True)
        if neural:
            joblib.dump(scaler, model_dir / "neural_scaler.joblib")

    rows, predictions, reports = [], {}, {}
    actual = y[test]
    def record(name, family, pred):
        predictions[name] = pred
        reports[name] = classification_report(actual, pred, labels=range(len(LABELS)), target_names=LABELS,
                                              output_dict=True, zero_division=0)
        rows.append({"model": name, "family": family, "accuracy": accuracy_score(actual, pred),
                     "macro_f1": f1_score(actual, pred, average="macro"), "test_rows": len(test)})


    for name in models:
        print(f"Training {name}", flush=True)
        if name in DL_MODELS:
            family = "dl"
            model = train_neural(name, tf, X_train, y[train], X_val, y[validation], args.epochs)
            pred = model(X_test, training=False).numpy().argmax(axis=1)
            if args.save_models:
                model.save(model_dir / f"{name}.keras")
        else:
            family = "baseline" if name == "dummy" else "ml"
            model = classical_model(name).fit(X.iloc[train], y[train])
            pred = model.predict(X.iloc[test])
            if args.save_models:
                joblib.dump(model, model_dir / f"{name}.joblib")
            if name == "random_forest" and args.shap:
                explain_forest(model, X.iloc[test], args.output)
        record(name, family, pred)

    members = [name for name in models if name != "dummy"]
    if len(members) >= 3:
        votes = np.stack([predictions[name] for name in members])
        counts = np.stack([(votes == label).sum(axis=0) for label in range(len(LABELS))])
        record("majority_vote", "ensemble", counts.argmax(axis=0))
        metadata["ensemble_members"] = members

    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result.to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame({"row_id": test, "actual": actual, **predictions}).to_csv(args.output / "predictions.csv", index=False)
    paired_summary(predictions, actual).to_csv(args.output / "paired_comparisons.csv", index=False)
    (args.output / "reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_reports(result, args.output)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-column", help="Column containing verified participant IDs")
    parser.add_argument("--models", nargs="+", choices=ML_MODELS + DL_MODELS, default=ML_MODELS + DL_MODELS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--shap", action="store_true", help="Explain the trained Random Forest on test rows")
    parser.add_argument("--save-models", action="store_true", help="Save fitted models and the neural scaler locally")
    args = parser.parse_args()
    if args.epochs < 1 or (args.shap and "random_forest" not in args.models):
        parser.error("Use at least one epoch and include random_forest when requesting SHAP.")
    run(args)
