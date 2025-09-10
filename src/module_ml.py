# Librerías
import pandas as pd

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


class Model():

    def __init__(self, X:pd.DataFrame, y:pd.Series, seed:int=42):
        self.X = X
        self.y = y
        self.seed = seed
    
    def split(self, train_size:float=0.8):
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            train_size=train_size,
                                                            random_state=self.seed
                                                            )
        return X_train, X_test, y_train, y_test

    def evaluate(self, model):
        X_train, X_test, y_train, y_test = self.split()
        model.fit(X_train, y_train)
        print("Entrenamiento completado")
        y_pred = model.predict(X_test)
        print("Metricas relevantes")
        accuracy = accuracy_score(y_test, y_pred)
        roc_score = roc_auc_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Roc_Score: {roc_score}")