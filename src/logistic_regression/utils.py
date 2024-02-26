import numpy as np
import os


def linear_interpolation(series):
    """
    Fills missing values in a pandas Series using linear interpolation.
    The missing values will be filled with the mean of the previous and next non-missing values.

    Parameters:
    - series (pd.Series): The series with missing values to fill.

    Returns:
    - pd.Series: The series with missing values filled via linear interpolation.
    """
    # Find indices of non-NaN values
    non_nan_indices = np.where(~np.isnan(series))[0]
    for i in range(len(series)):
        if np.isnan(series[i]):
            # Find the closest non-NaN indices before and after the NaN
            prev_index = max(filter(lambda x: x < i, non_nan_indices), default=None)
            next_index = min(filter(lambda x: x > i, non_nan_indices), default=None)
            # Compute interpolated value if both surrounding indices are found
            if prev_index is not None and next_index is not None:
                series[i] = series[prev_index] + (
                    series[next_index] - series[prev_index]
                ) * ((i - prev_index) / (next_index - prev_index))
            # Fill with the previous value if only the next index is found
            # Case when the first value is NaN
            elif prev_index is not None:
                series[i] = series[prev_index]
            # Fill with the next value if only the previous index is found
            # Case when the last value is NaN
            elif next_index is not None:
                series[i] = series[next_index]
    return series


def standardize_features(df):
    """
    Standardizes each feature in the DataFrame by subtracting the mean and dividing by the standard deviation.
    This normalization ensures that each feature has a mean of 0 and a standard deviation of 1

    Parameters:
    - df (pd.DataFrame): The DataFrame to standardize.

    Returns:
    - pd.DataFrame: The standardized DataFrame.
    """
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df


def sigmoid_func(z):
    """
    Computes the sigmoid function.

    Parameters:
    - z (np.ndarray): The input to the sigmoid function.

    Returns:
    - np.ndarray: The output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))


def check_files_existence(files):
    """
    Check if the required files exist.

    Parameters:
    - files (list of str): A list of file paths to check.

    Returns:
    - bool: True if all files exist, False otherwise.
    """
    for file in files:
        if not os.path.exists(file):
            print(f"Error: '{file}' does not exist.")
            return False
    return True


def run_command(command):
    """
    Executes a system command.

    Parameters:
    - command (str): The command to execute.
    """
    os.system(command)
