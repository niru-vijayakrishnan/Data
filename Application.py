from DataCollector import *
from CustomModel import *

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 0, 1])

model = CustomModel("linear")

weights, loss, iterations = model.gradient_descent(X, y, model.linear_regression_loss_plus_l2,1, num_iterations = 10000000)
#weights, loss, iterations = model.gradient_descent(X, y, model.l0_loss)
print(model.predict(weights, X))

print("Optimized weights:", weights)
print("Final loss:", loss)
print("Iterations performed:", iterations)

''''
app = DataCollector("GermanData.txt","l2", modelType="CustomModel", pointDomination=False)
#app.writeExcel()
app.graphData(0.1)
'''
'''
app = DataCollector("GermanData.txt","l2", modelType="CustomModel", pointDomination=False)
X = app.X_DATA
y = app.Y_DATA
model = CustomModel()
weights, loss, iterations = model.gradient_descent(X, y, model.linear_regression_loss_plus_l2,.1)
predictedY = model.predict(weights, X)
print(predictedY)
count = 0
for i in range(len(y)):
    if(y[i] == round(predictedY[i])):
        count += 1
print("Accuracy: " + str(count/len(y)))
'''
