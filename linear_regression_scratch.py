
import numpy as np


class Sam_Linear_Regression() :
    def __init__(self,learning_rate,number_of_iterations):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations

    def fit(self,X,Y):
        self.m,self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X =X
        self.Y =Y

        for i in range(self.number_of_iterations):
            self.update_weights()

    def update_weights(self):

        Y_prediction = self.predict(self.X)

        dw = -(2*(self.X.T).dot(self.Y - Y_prediction)) / self.m

        db = -2 * np.sum(self.Y-Y_prediction)

        self.w = self.w - self.learning_rate * dw
        
        self.b = self.b - self.learning_rate * db

    def predict(self,X):

        return X.dot(self.w) + self.b


