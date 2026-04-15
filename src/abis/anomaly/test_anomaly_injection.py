import joblib
import pandas as pd
from abis.utils.config import SETTINGS
from abis.preprocess.make_features import load_and_prepare_features


def main():
    scaler = joblib.load("models/scaler.joblib")
    model = joblib.load("models/isolation_forest.joblib")
    feature_names = joblib.load("models/feature_names.joblib")

    X, _ = load_and_prepare_features(str(SETTINGS.data_path))

    normal = X.iloc[[0]].copy()
    crazy = normal.copy() * 10  # make it obviously abnormal

    def score(df, name):
        df = df.reindex(columns=feature_names, fill_value=0)
        xs = scaler.transform(df)
        s = float(model.decision_function(xs)[0])
        p = int(model.predict(xs)[0])
        print(f"{name}: score={s:.4f}, anomaly={p==-1}")

    score(normal, "NORMAL")
    score(crazy, "CRAZY (x10)")

if __name__ == "__main__":
    main()
