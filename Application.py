from DataCollector import *
from CustomModel import *
# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 5, 7])

model = CustomModel()

#weights, loss, iterations = model.gradient_descent(X, y, model.linear_regression_loss)
weights, loss, iterations = model.gradient_descent(X, y, model.l0_loss)

print("Optimized weights:", weights)
print("Final loss:", loss)
print("Iterations performed:", iterations)
#app = DataCollector("AdultData.txt","l2", modelType="GradientBoostingClassifier", pointDomination=False)
#app.graphData(0.1)