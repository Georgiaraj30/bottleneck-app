from preprocess import preprocess_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import mlflow
import mlflow.sklearn
import joblib


def train_model():

    # Get preprocessed data
    X_train, X_test, y_train, y_test, X, y = preprocess_data()

    # MLflow experiment
    mlflow.set_experiment("IT_Bottleneck_Detection")

    # Model parameters
    n_estimators = 100
    random_state = 42

    with mlflow.start_run():

        # Random Forest model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model to MLflow
        mlflow.sklearn.log_model(
   	model,
    	name="model",
   	skops_trusted_types=["sklearn.tree._tree.Tree"]
	)
        # Also save model for Streamlit
        joblib.dump(model, "bottleneck_model.pkl")

        print("Model trained successfully!")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Model saved as bottleneck_model.pkl")
        print("MLflow run completed successfully!")

        return model


if __name__ == "__main__":
    train_model()