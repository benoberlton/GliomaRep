import os
from sklearn.preprocessing import StandardScaler
import numpy as np

def make_directory(directory: str = None):
    """Creates a directory at the specified path if one doesn't exist.

    Parameters
    ----------
    directory : str
        A string specifying the directory path.

    Returns
    -------
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def standardize(x =  None):
    """Standardizes data by removing the mean and scaling to unit variance.

    Parameters
    x: pd.DataFrame (default = None)
        data matrix (dimensions = cells x features)
    ----------

    Returns
    X: pd.DataFrame
        standardized data matrix (dimensions = cells x features)
    ----------
    """
    scaler = StandardScaler(with_mean = True, with_std = True)
    X = scaler.fit_transform(x)
    return X

def compute_percentile(df, p = 70):
    """
    Computes the specified percentile of values in a given DataFrame, ignoring any NaN values.

    Parameters:
        df (pd.DataFrame): DataFrame containing numerical data from which to calculate the percentile.
        p (int, optional): The percentile to compute. Must be between 0 and 100. Default is 70.

    Returns:
        float: The computed percentile value from the provided data.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'scores': [1, 2, 3, np.nan, 5]})
        >>> compute_percentile(df['scores'], 50)
        2.5
    """
    scores = df.values
    scores = scores[~np.isnan(scores)]
    perc = np.percentile(scores, p)
    return perc