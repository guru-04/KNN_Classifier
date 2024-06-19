import numpy as np
from collections import Counter

class KNN_Classifier:

    def __init__(self,n_neighbours=5):
        self.n_neighbours = n_neighbours

    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        print("Training Done")

    def predict(self,x_test):
        y_pred = []

        for i in x_test:
            distances = []
            for j in self.x_train:
                distances.append(self.calculate_distance(i,j))
            distances = sorted(list(enumerate(distances)),key = lambda x:x[1])[:self.n_neighbours]

            y_pred.append(self.majority_lable(distances))
                
        return y_pred

    def calculate_distance(self,point_a,point_b):
        return np.linalg.norm(point_a-point_b)
    
    def majority_lable(self,neighbours):
        lables = []
        for k in neighbours:
                lables.append(self.y_train[k[0]])

        return Counter(lables).most_common()[0][0]