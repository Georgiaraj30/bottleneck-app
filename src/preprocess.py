import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data():

    # 1. Load raw datasets
    df_server = pd.read_csv("data/raw/system.csv")
    df_net = pd.read_csv("data/raw/network.csv")

    # 2. Convert timestamps
    df_server["timestamp"] = pd.to_datetime(
        df_server["timestamp"],
        unit="s"
    )

    df_net["datetime"] = pd.to_datetime(
        df_net["datetime"]
    ).dt.tz_localize(None)

    # 3. Merge server and network data
    df = pd.merge_asof(
        df_server.sort_values("timestamp"),
        df_net.sort_values("datetime"),
        left_on="timestamp",
        right_on="datetime",
        direction="nearest"
    )

    # 4. Create bottleneck target
    df["bottleneck"] = (
        (df["cpu-user"] > 0.7) |
        (df["cpu-iowait"] > 0.3) |
        (df["disk-io-time"] > 0.02)
    ).astype(int)

    # 5. Drop unnecessary columns
    drop_cols = [
        "timestamp",
        "datetime",
        "uid",
        "src_ip",
        "dest_ip",
        "community_id",
        "history"
    ]

    df.drop(
        columns=[col for col in drop_cols if col in df.columns],
        inplace=True
    )

    # 6. Encode categorical columns
    cat_cols = [
        "service",
        "protocol",
        "conn_state",
        "mitre_attack_tactics"
    ]

    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # 7. Convert Boolean columns
    bool_cols = [
        "local_resp",
        "local_orig"
    ]

    df[bool_cols] = df[bool_cols].astype(int)

    # 8. Separate features and target
    X = df.drop(columns=["bottleneck"])
    y = df["bottleneck"]

    # 9. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, X, y


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, X, y = preprocess_data()

    print("Preprocessing completed successfully!")
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Target distribution:")
    print(y.value_counts())