import numpy as np
import math

class CustomModel:
    
    def __init__(self, lossType):
        self.lossType = lossType

    def gradient_descent(self, X, y, loss_func, alpha, learning_rate=0.01, num_iterations=100000, epsilon=1e-6):
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
            #change to logistic -> DONE
            if(self.lossType == "logistic"):
                predictions = 1/(1 + math.e**(-1 * np.dot(X, weights)))
            elif(self.lossType == "linear"):
                predictions = np.dot(X, weights)
            error = predictions - y
            if(self.lossType == "logistic"):
                pass
            elif(self.lossType == "linear"):
                self.getGradientLinearRegressionPlusL2(X, y, weights, alpha)
            gradient = np.dot(X.T, error) / num_samples
            new_weights = weights - learning_rate * gradient
            new_loss = loss_func(predictions, y, new_weights, alpha)

            if abs(loss - new_loss) < epsilon:
                break

            weights = new_weights
            loss = new_loss
            iteration += 1

        self.weights = weights
        return weights, loss, iteration
    
    def getGradientLinearRegressionPlusL2(self, X, y, weights, alpha):
        sum = np.zeros(len(X[0]))
        for i in range(len(X)):
            difference = self.predictOne(weights, X[i]) - y[i]
            newX = self.multiplyList(X[i],difference*2)
            for j in range(len(X[0])):
                sum[j] += newX[j]
        newNewX = self.divideList(sum, len(X))
        newNewX = np.array(newNewX)
        weights = np.array(weights)
        weights *= 2*alpha
        newNewX += weights
        return newNewX
        
        
    
    def linear_regression_loss_plus_l0(self, predictions, targets, weights, alpha):
        return self.linear_regression_loss(predictions, targets) + (alpha * self.l0_loss(weights))
    
    def linear_regression_loss_plus_l1(self, predictions, targets, weights, alpha):
        return self.linear_regression_loss(predictions, targets) + (alpha * self.l1_loss(weights))
    
    def linear_regression_loss_plus_l2(self, predictions, targets, weights, alpha):
        return self.linear_regression_loss(predictions, targets) + (alpha * self.l2_loss(weights))
    
    def logistic_loss_plus_l0(self, predictions, targets, weights, alpha):
        return self.logistic_loss(predictions, targets) + (alpha * self.l0_loss(weights))
    
    def logistic_loss_plus_l1(self, predictions, targets, weights, alpha):
        return self.logistic_loss(predictions, targets) + (alpha * self.l1_loss(weights))
    
    def logistic_loss_plus_l2(self, predictions, targets, weights, alpha):
        return self.logistic_loss(predictions, targets) + (alpha * self.l2_loss(weights))

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
    
    def logistic_loss(self, predictions, targets):
        """
        Computes the logistic loss, also known as binary cross-entropy loss.

        Args:
            predictions (numpy.ndarray): Predicted probabilities or scores.
            targets (numpy.ndarray): True labels (0 or 1).

        Returns:
            float: Loss value.
        """
        epsilon = 1e-10  # small value added to avoid division by zero
        loss = -np.mean(targets * np.log(predictions + epsilon) + (1 - targets) * np.log(1 - predictions + epsilon))
        return loss
    
    #The weights that are not 0
    def l0_loss(self, weights):
        """
        Computes the L0 loss, which counts the number of non-zero elements in the error vector.

        Args:
            predictions (numpy.ndarray): Predicted values.
            targets (numpy.ndarray): Target values.

        Returns:
            int: Number of non-zero elements in the error vector.
        """
        num_nonzero = np.count_nonzero(weights)
        return num_nonzero
    
    #The absolute value of the weights added up
    def l1_loss(self, weights):
        """
        Computes the L1 loss, also known as mean absolute error (MAE).

        Args:
            predictions (numpy.ndarray): Predicted values.
            targets (numpy.ndarray): True values.

        Returns:
            float: Loss value.
        """
        error = np.abs(weights)
        loss = np.mean(error)
        return loss
    
    #Absolute Value of Weight Squared all added up and square rooted
    def l2_loss(self, weights):
        """
        Computes the L2 loss, also known as mean squared error (MSE).

        Args:
            predictions (numpy.ndarray): Predicted values.
            targets (numpy.ndarray): True values.

        Returns:
            float: Loss value.
        """
        error = np.square(weights)
        loss = np.sum(error)
        endLoss = np.sqrt(loss)
        return endLoss
    
    def predictOne(self, weights, X):
        sum = 0
        for i in range(len(weights)):
            sum += X[i] * weights[i]
        return sum
    
    def predict(self, weights, X):
        predictions = []
        for dataPoint in X:
            sum = 0
            for i in range(len(weights)):
                sum += dataPoint[i] * weights[i]
            if(self.lossType == "logistic"):
                prediction = 1/(1 + math.e**(-1 * sum))
            elif(self.lossType == "linear"):
                prediction = sum
            predictions.append(prediction)
        return predictions
    
    def multiplyList(self, list_to_multiply, constant):
        new_list = []
        for element in list_to_multiply:
            new_element = element * constant
            new_list.append(new_element)
        return new_list
    
    def divideList(self, list_to_divide, constant):
        new_list = []
        for element in list_to_divide:
            new_element = element / constant
            new_list.append(new_element)
        return new_list

