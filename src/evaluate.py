from preprocess import preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_models():

    # Get preprocessed data
    X_train, X_test, y_train, y_test, X, y = preprocess_data()

    # Define the three models
    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    }

    print("\n========================================")
    print("MODEL EVALUATION")
    print("========================================")

    # Evaluate each model
    for name, model in models.items():

        print(f"\n{name}")
        print("-" * len(name))

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(y_test, y_pred)

        print("Accuracy:", accuracy)

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(report)

    print("\n========================================")
    print("All 3 models evaluated successfully!")
    print("========================================")


if __name__ == "__main__":
    evaluate_models()