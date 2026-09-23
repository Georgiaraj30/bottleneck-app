from preprocess import preprocess_data
from train import train_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_model():

    # Get preprocessed data
    X_train, X_test, y_train, y_test, X, y = preprocess_data()

    # Train/load the model
    model = train_model()

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    evaluate_model()