import numpy as np

class CustomModel:
    
    def __init__(self):
        pass

    def gradient_descent(self, X, y, loss_func, learning_rate=0.01, num_iterations=1000, epsilon=1e-6):
        """
        Performs gradient descent optimization to minimize the loss function.

        Args:
            X (numpy.ndarray): Input features of shape (num_samples, num_features).
            y (numpy.ndarray): Target values of shape (num_samples,).
            loss_func (callable): Loss function that takes predictions and targets as input and returns the loss.
            learning_rate (float): Learning rate or step size for gradient descent (default: 0.01).
            num_iterations (int): Maximum number of iterations (default: 1000).
            epsilon (float): Minimum change in loss for convergence (default: 1e-6).

        Returns:
            numpy.ndarray: Optimized weights of shape (num_features,).
            float: Final loss.
            int: Number of iterations performed.
        """
        X = np.array(X)
        y = np.array(y)
        num_samples, num_features = X.shape
        weights = np.zeros(num_features)
        loss = np.inf
        iteration = 0

        while iteration < num_iterations:
            predictions = np.dot(X, weights)
            error = predictions - y
            gradient = np.dot(X.T, error) / num_samples
            new_weights = weights - learning_rate * gradient
            new_loss = loss_func(predictions, y)

            if abs(loss - new_loss) < epsilon:
                break

            weights = new_weights
            loss = new_loss
            iteration += 1

        return weights, loss, iteration

    def linear_regression_loss(self, predictions, targets):
        """
        Computes the mean squared error (MSE) loss for linear regression.

        Args:
            predictions (numpy.ndarray): Predicted values.
            targets (numpy.ndarray): Target values.

        Returns:
            float: Loss value.
        """
        error = predictions - targets
        mse = np.mean(np.square(error))
        return mse
    
    def l0_loss(self, predictions, targets):
        """
        Computes the L0 loss, which counts the number of non-zero elements in the error vector.

        Args:
            predictions (numpy.ndarray): Predicted values.
            targets (numpy.ndarray): Target values.

        Returns:
            int: Number of non-zero elements in the error vector.
        """
        error = predictions - targets
        num_nonzero = np.count_nonzero(error)
        return num_nonzero
