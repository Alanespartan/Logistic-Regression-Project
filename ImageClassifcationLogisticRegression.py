# Juan Arturo Cruz Cardona - A01701804
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    #Find the number of training data
    m = X.shape[1]
    
    #Calculate the predicted output
    A = sigmoid(np.dot(w.T, X) + b)
    
    #Calculate the cost function 
    cost = -1/m * np.sum(Y*np.log(A) + (1-Y) * np.log(1-A))
    
    #Calculate the gradients
    dw = 1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum(A-Y)
        
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []
    
    #propagate function will run for a number of iterations    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)    
        dw = grads["dw"]
        db = grads["db"]
        
        #Updating w and b by deducting the dw 
        #and db times learning rate from the previous w and b    
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #Record the cost function value for each 100 iterations        
        if i % 100 == 0:
            costs.append(cost)
            
    #The final updated parameters     
    params = {"w": w, "b": b}
    #The final updated gradients    
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    w = w.reshape(X.shape[0], 1)
    
    #Initializing an aray of zeros which has a size of the input
    #These zeros will be replaced by the predicted output 
    Y_prediction = np.zeros((1, m))    
    
    #Calculating the predicted output using the sigmoid function
    #This will return the values from 0 to 1
    A = sigmoid(np.dot(w.T, X) + b)
    
    #Iterating through A and predict an 1 if the value of A
    #is greater than 0.5 and zero otherwise
    for i in range(A.shape[1]):
        Y_prediction[:, i] = (A[:, i] > 0.5) * 1
        
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    #Initializing the w and b as zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    
    #Best fit the training data
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    w = parameters["w"]
    b = parameters["b"]
    
    # Predicting the output for both test and training set 
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
        
    #Calculating the training and test set accuracy by comparing
    #the predicted output and the original output
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
  
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

# Set up of data
df_x= pd.read_excel('dataset.xlsx', 'X', header=None)
df_x.head()

df_y= pd.read_excel('dataset.xlsx', 'Y', header=None)
df_y.head()

y = df_y[0]
for i in range(len(y)):
    if y[i] != 1:
        y[i] = 0
y = pd.DataFrame(y)

x_train = df_x.iloc[0:4000].T
x_test = df_x.iloc[4000:].T
x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = y.iloc[0:4000].T
y_test = y.iloc[4000:].T
y_train = np.array(y_train)
y_test = np.array(y_test)

ni = 2000 # num iterations 
lr = 0.015 # learning rate
d = model(x_train, y_train, x_test, y_test, ni, lr)

#Plot how cost function changed each updated w's and b's
plt.figure(figsize=(7,5))
plt.scatter(x = range(len(d['costs'])), y = d['costs'], color='black')
plt.title('Scatter Plot of Cost Functions', fontsize=18)
plt.ylabel('Costs', fontsize=12)
plt.show()

#plt.imshow(np.array(df_x.iloc[500, :]).reshape(20,20))
#plt.show()