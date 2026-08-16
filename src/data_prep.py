
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_PATH = os.path.join("data", "creditcard.csv")
OUT_DIR = os.path.join("data", "processed")


IMBALANCE_RATIOS = {
    "1to1": 1,
    "1to10": 10,
    "1to100": 100,
    "full": None,  
}


RANDOM_STATE = 42


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}"
        )
    return pd.read_csv(path)


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
    return df


def stratified_split(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_ratio_subset(X_train, y_train, ratio_multiplier):
    
    fraud_idx = y_train[y_train == 1].index

    legit_idx = y_train[y_train == 0].index

    n_fraud = len(fraud_idx)

    if ratio_multiplier is None:
    
        keep_legit_idx = legit_idx
    else:

        n_legit_keep = min(len(legit_idx), n_fraud * ratio_multiplier)

        rng = np.random.RandomState(RANDOM_STATE)

        keep_legit_idx = rng.choice(legit_idx, size=n_legit_keep, replace=False)

    keep_idx = np.concatenate([fraud_idx, keep_legit_idx])

    rng = np.random.RandomState(RANDOM_STATE)

    rng.shuffle(keep_idx)

    return X_train.loc[keep_idx], y_train.loc[keep_idx]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)


    df = load_raw()
    print(f"Loaded {len(df):,} rows, {df['Class'].sum():,} fraud "
          f"({100 * df['Class'].mean():.4f}%)")

    df = scale_features(df)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(df)

    X_val.to_csv(os.path.join(OUT_DIR, "X_val.csv"), index=False)

    y_val.to_csv(os.path.join(OUT_DIR, "y_val.csv"), index=False)

    X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

    for name, ratio in IMBALANCE_RATIOS.items():
        X_sub, y_sub = build_ratio_subset(X_train, y_train, ratio)
        X_sub.to_csv(os.path.join(OUT_DIR, f"X_train_{name}.csv"), index=False)
        y_sub.to_csv(os.path.join(OUT_DIR, f"y_train_{name}.csv"), index=False)
        print(f"  [{name}] train size={len(X_sub):,}, "
              f"fraud={int(y_sub.sum())}, legit={len(y_sub) - int(y_sub.sum())}, "
              f"fraud%={100 * y_sub.mean():.3f}%")

    print(f"\n Processed splits written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
