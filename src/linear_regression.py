import numpy as np

class SimpleLinearRegression:
    """
    Simple Linear Regression implemented using
    the closed-form Ordinary Least Squares Solution.
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted_ = False
    
    def fit(self, X, y):
        """
        Fit the linear regression model using OLS.

        Parameters
        ----------
        X : array-like
            1D feature array

        y : array-like
            Target values

        Returns
        -------
        self
            Fitted model.
        """

        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of observations")
        
        if len(X) == 0:
            raise ValueError("X and y cannot be empty")

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        if denominator == 0:
            raise ValueError("The feature must contain more than one unique value")

        self.coef = numerator / denominator
        self.intercept_ = y_mean - self.coef * x_mean

        self.is_fitted_ = True
    
    def predict(self, X):
        """
        Generate predictions using the fitted model
        """

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=float).reshape(-1)

        return self.intercept_ + self.coef_ * X