import numpy as np
import pandas as pd
import argparse

from utils import linear_interpolation, standardize_features, sigmoid_func


class HogwartsHousePredictor:
    """Predicts the Hogwarts House using logistic regression model."""

    def __init__(self, weights_path):
        """
        Initializes the HogwartsHousePredictor with model weights.

        Parameters:
        - weights_path (str): Path to the CSV file containing model weights.
        """
        # Load model weights and house names
        weights_df = pd.read_csv(weights_path)
        # Extract house names and weights
        self.house_names = weights_df.columns.tolist()
        self.weights = weights_df.to_numpy()

    def prepare_features(self, df):
        """
        Prepares the dataset for prediction.

        Parameters:
        - df (pd.DataFrame): The dataset containing features.

        Returns:
        - pd.DataFrame: The processed dataset ready for prediction.
        """
        # Drop irrelevant columns
        df = df.drop(
            [
                "Index",
                "First Name",
                "Last Name",
                "Birthday",
                "Best Hand",
                "Care of Magical Creatures",
                "Astronomy",
                "Hogwarts House",
            ],
            axis=1,
            errors="ignore",
        )
        # Interpolate missing values using linear interpolation
        for column in df.columns:
            df[column] = linear_interpolation(df[column])
        # Standardize features
        df = standardize_features(df)
        # Add a bias term
        df["Bias"] = np.ones(df.shape[0])
        return df

    def predict_houses(self, df):
        """
        Predicts the Hogwarts House for each entry in the dataset.

        Parameters:
        - df (pd.DataFrame): The dataset with prepared features.

        Returns:
        - list: Predicted Hogwarts Houses.
        """
        # Convert the DataFrame to numpy array for computation
        X = df.values
        # Apply the sigmoid function to the dot product of the features and weights
        predictions = sigmoid_func(np.dot(X, self.weights))
        # Select the class with the highest probability
        predicted_classes = np.argmax(predictions, axis=1)
        # Map the class indices to house names
        return [self.house_names[i] for i in predicted_classes]


def main(test_dataset_file, weights_path):
    """
    Main function to predict Hogwarts Houses and save the predictions.

    Parameters:
    - test_dataset_file (str): Path to the test dataset file.
    - weights_path (str): Path to the model weights file.
    """
    # Load and prepare test dataset
    df_test = pd.read_csv(test_dataset_file)
    # Save the 'Index' as first column for the final prediction file
    index = df_test["Index"]
    # Initialize predictor
    predictor = HogwartsHousePredictor(weights_path)
    # Prepare the data
    df_prepared = predictor.prepare_features(df_test)
    # Predict and save Hogwarts Houses
    predicted_houses = predictor.predict_houses(df_prepared)
    # Create and save the prediction file
    predictions_df = pd.DataFrame({"Index": index, "Hogwarts House": predicted_houses})
    predictions_df.to_csv("houses.csv", index=False)
    print("Prediction complete. Results saved to 'houses.csv'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Hogwarts House for individuals in the test dataset using logistic regression model."
    )
    parser.add_argument(
        "test_dataset_file", type=str, help="Path to the test dataset file."
    )
    parser.add_argument(
        "weights_path", type=str, help="Path to the model weights file."
    )
    args = parser.parse_args()

    # Check for specific file names
    if not (
        args.test_dataset_file.endswith("dataset_test.csv")
        or args.test_dataset_file.endswith("to_predict.csv")
    ):
        print(
            "Error: The script requires 'dataset_test.csv' from 'datasets.tgz' or 'to_predict.csv' from 'training_acc.py'."
        )
    elif not args.weights_path.endswith("weights.csv"):
        print("Error: The script requires 'weights.csv' created by 'logreg_train.py'.")
    else:
        main(args.test_dataset_file, args.weights_path)
