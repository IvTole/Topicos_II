# Importación de librerías
import numpy as np
import pprint as pp

# Scikit-learn
from sklearn.linear_model import LogisticRegression

# Módulos propios
from module_data import Dataset # class Dataset
from module_ml import Model


def main():

    data = Dataset()
    X,y = data.load_xy_scaled()
    
    # Model
    ml = Model(X=X, y=y, seed=42)
    ml.evaluate(LogisticRegression(max_iter=5000))

if __name__ == "__main__":
    main()


