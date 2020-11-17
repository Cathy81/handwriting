
print(__doc__)

import matplotlib.pyplot as plt
#from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from dataprep import read
import numpy as np
import pickle

# rescale the data, use the traditional train/test split
y_train=[]
X=[]
mnist = list(read("training"))
for item in mnist:
    label,x=item
    nX=np.array(x)
    y_train.append(label)
    X.append(nX.flatten())
#y_train, X=mnist[:60000]
X_train=[img/255 for img in X]  # normalize the data,

print(len(X_train))
print(len(y_train))
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train) #build the model

print("Training set score: %f" % mlp.score(X_train, y_train))

predictionsTraining=mlp.predict(X_train)
print("Preticted results on first100:", predictionsTraining[:100])
print("real      results on first100:", y_train[:100])
print(confusion_matrix(y_train, predictionsTraining))
pickle.dump(mlp, open("mlp", 'wb')) # save the model
