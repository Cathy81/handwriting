import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from dataprep import read
import numpy as np
#Step2: complete the following function, which accepts a 28 X28 array for a handwriting digit.
#THe function outputs the recognized digit from the model built in mnist_model_build.py
#Hint: similar to testing.py

def digitRecog(imageArray):
    "*** YOUR CODE HERE ***"
    mlp = pickle.load(open("mlp", 'rb'))
    nX = np.array(imageArray)

    X_test =[nX.flatten()/255.]
    predictions = mlp.predict(X_test)
    print(" Letter written is:  ", predictions)
    return predictions