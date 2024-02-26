import pandas as pd
import sys

from sklearn.metrics import accuracy_score
from utils import check_files_existence, run_command


def prepare_dataset_for_prediction(dataset_path, output_path):
    """
    Prepares the dataset by removing the target column and saving it for prediction.

    Parameters:
    - dataset_path (str): Path to the dataset file.
    - output_path (str): Path where the modified dataset will be saved.
    """
    df = pd.read_csv(dataset_path)
    # Drop 'Hogwarts House' column and save the dataset for prediction
    df.drop("Hogwarts House", axis=1, inplace=True)
    # Add empty 'Hogwarts House' column
    df.insert(loc=1, column="Hogwarts House", value="")
    # Save the modified dataset
    df.to_csv(output_path, index=False)


def compute_accuracy(real_dataset_path, predicted_dataset_path):
    """
    Computes and prints the accuracy of the predictions.

    Parameters:
    - real_dataset_path (str): Path to the dataset with real labels.
    - predicted_dataset_path (str): Path to the dataset with predicted labels.
    """
    # Load datasets
    real_df = pd.read_csv(real_dataset_path)
    # Load the predicted dataset
    prediction_df = pd.read_csv(predicted_dataset_path)
    # Compute and print accuracy
    accuracy = 100 * accuracy_score(
        real_df["Hogwarts House"], prediction_df["Hogwarts House"]
    )
    print(f"Accuracy on training dataset: {accuracy}%")


if __name__ == "__main__":
    # Define file paths
    dataset_train_path = "../../datasets/dataset_train.csv"
    to_predict_path = "to_predict.csv"
    weights_path = "weights.csv"
    predicted_houses_path = "houses.csv"
    # Check if the 'dataset_train.csv' file exists
    if not check_files_existence([dataset_train_path]):
        sys.exit(
            "Terminating due to missing training dataset 'dataset_train.csv' file."
        )
    # Train the logistic regression model and predict Hogwarts Houses
    run_command("python3 logreg_train.py ../../datasets/dataset_train.csv")
    # Prepare dataset for prediction
    prepare_dataset_for_prediction(dataset_train_path, to_predict_path)
    # Check if the 'weights.csv' file exists
    if not check_files_existence([weights_path]):
        sys.exit("Terminating due to missing model weights 'weights.csv' file.")
    run_command("python3 logreg_predict.py to_predict.csv weights.csv")
    # Check if the 'houses.csv' file exists
    if not check_files_existence([predicted_houses_path]):
        sys.exit("Terminating due to missing predicted houses 'houses.csv' file.")
    # Compute and print accuracy
    compute_accuracy(dataset_train_path, predicted_houses_path)
