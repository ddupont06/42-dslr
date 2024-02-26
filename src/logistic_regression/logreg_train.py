import numpy as np
import pandas as pd
import argparse

from utils import linear_interpolation, standardize_features, sigmoid_func


class LogisticRegressionGD:
    """This class implements logistic regression using gradient descent."""

    def __init__(self):
        """Initializes the LogisticRegressionGD with default weights as None."""
        self.weights = None

    def prepare_training(self, df):
        """
        Prepares the dataset for training the logistic regression model.

        Parameters:
        - df (pd.DataFrame): The dataset containing features and the target variable.

        Performs feature selection, data imputation, feature standardization, and one-hot encoding for the target variable.
        """
        # Drop unnecessary columns
            # Drop Care of Magical Creatures since it has homogeneous score distribution between all four houses
            # Drop Astronomy since it is negatively correlated with Care of Magical Creatures
        df = df.drop(
            [
                "Index",
                "First Name",
                "Last Name",
                "Birthday",
                "Best Hand",
                "Care of Magical Creatures",
                "Astronomy",
            ],
            axis=1,
        )
        # Apply linear interpolation for each value in the DataFrame
        for column in df.columns:
            if df[column].isnull().any():
                df[column] = linear_interpolation(df[column].copy())
        # Drop the target column
        target = df.drop("Hogwarts House", axis=1)
        # Standardize the features
        df.update(standardize_features(target))
        # Add bias term column to allow the model to fit the data better
        df["Bias"] = np.ones(df.shape[0])
        # Drop the target column to make the feature matrix
            # [1600, 12] -> [1600 training examples, 11 features + 1 bias] matrix filled with standardized values
        self.X = df.drop("Hogwarts House", axis=1).values
        # One-hot encode the target variable 'Hogwarts House' into a binary matrix that represents each class
        # It enables multi-class classification with the logistic regression model via the one-vs-all method
            # [1600, 4] -> [1600 training examples, 4 houses] matrix filled with either 0 or 1
        self.y = pd.get_dummies(df["Hogwarts House"]).values
        # Initialize weights with random values that will be updated during the training process
            # [12, 4] -> [11 features + 1 bias, 4 houses] matrix filled with random probability values
        self.weights = np.random.randn(self.X.shape[1], self.y.shape[1]) * 0.01

    # def cost_fun(self, weights):
    #     """
    #     Calculates the cost function for logistic regression.
    #     The cost function quantifies the error between predicted values and actual values in the training data.

    #     Parameters:
    #     - weights (np.ndarray): The model weights.

    #     Returns:
    #     - float: The cost of the model using the current weights.
    #     """
    #     # Reshape weights to ensure correct matrix multiplication
    #     weights = weights.reshape((self.X.shape[1], -1))
    #     # Number of training examples
    #     m = self.y.shape[0]
    #     # Compute the prediction for every example
    #     predictions = sigmoid_func(np.dot(self.X, weights))
    #     # Calculate the cost using the logistic regression cost function formula
    #     cost = (
    #         -1
    #         / m
    #         * np.sum(
    #             self.y * np.log(predictions) + (1 - self.y) * np.log(1 - predictions)
    #         )
    #     )
    #     return cost

    def gradient_descent(self, learning_rate=0.01, iterations=10000):
        """
        Updates the model weights using gradient descent algorithm.

        Parameters:
        - learning_rate (float): Adjusts update speed. High values may overshoot; low values slow convergence.
        - iterations (int): How many times to adjust the model to improve its predictions.
        """
        # Number of individual training examples
        m = self.X.shape[0]
        print("Starting gradient descent...")
        for iteration in range(iterations):
            # Compute the prediction (hypothesis) for every example
                # [1600, 4] -> [1600 training examples, 4 houses] matrix filled with values holding the probability to belong to each house
            predictions = sigmoid_func(np.dot(self.X, self.weights))
            # Compute the gradient of the cost function with respect to each weight. It is the direction to move the weights to minimize the cost function
                # The error matrix '(predictions - self.y)' computes the difference between predicted and actual values
                # self.X.T transposes the feature matrix from [1600, 12] to [12, 1600] for matrix multiplication
                # The transposed feature matrix [12, 1600] is then multiplied by the error matrix [1600, 4] producing a [12, 4] matrix
                # [12, 4] -> [11 features + 1 bias, 4 houses] filled with values holding the gradient of the cost function
            gradient = np.dot(self.X.T, (predictions - self.y)) / m
            # Update weights by taking a step proportional to the gradient of the cost function
                # A positive gradient indicates that the cost function is increasing, so the weights should be decreased
                # A negative gradient indicates that the cost function is decreasing, so the weights should be increased
            self.weights -= learning_rate * gradient
            # Print the cost every 1000 iterations to monitor progress
            # if iteration % 1000 == 0:
            #     print(f"Cost at iteration {iteration}: {self.cost_fun(self.weights)}")

    def save_model(self, filename="weights.csv"):
        """
        Saves the model weights to a CSV file.

        Parameters:
        - filename (str): The name of the file to save the weights to.

        Returns:
        - str: The filename of the saved weights.
        """
        # Convert weights to DataFrame
        weights_df = pd.DataFrame(self.weights)
        # Set house names as the first row of the DataFrame
        house_names = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
        weights_df.columns = house_names
        # Save the weights to a CSV file
        weights_df.to_csv(filename, index=False)
        # Return the filename
        return filename


def main(dataset_file):
    """Trains the logistic regression model using the specified dataset and saves the weights to a CSV file.

    Parameters:
    - dataset_file (str): The path to the dataset file.
    """
    # Load the dataset
    df = pd.read_csv(dataset_file)

    # Prepare and train the model
    logreg = LogisticRegressionGD()
    logreg.prepare_training(df)
    logreg.gradient_descent()

    # Save the model weights and get the save location
    model_filepath = logreg.save_model("weights.csv")

    print(f"Model training complete. Weights saved to '{model_filepath}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a logistic regression by running the gradient descent."
    )
    parser.add_argument(
        "dataset_file",
        type=str,
        help="Path to the dataset file (e.g., dataset_train.csv)",
    )
    args = parser.parse_args()

    # Validate that the specified dataset is 'dataset_train.csv'
    if args.dataset_file.endswith("dataset_train.csv"):
        main(args.dataset_file)
    else:
        print("Error: The script requires 'dataset_train.csv' from 'datasets.tgz'.")
