
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from dataprep import read
import numpy as np
import pickle

mlp = pickle.load(open("mlp", 'rb'))


y_test=[]
XTest=[]
mTest = list(read("testing"))
for item in mTest:
  #  label,x=mTest[100]
    label, x = item
    nX=np.array(x)
    y_test.append(label)
    XTest.append(nX.flatten())
    #y_train, X=mnist[:60000]
    X_test=[img/255 for img in XTest]  # normalize the data,
    #X_test=[XTest[0]/255 ]
    # print(len(X_test))
    # print(len(y_test))


predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print("test set score: %f" % mlp.score(X_test, y_test))
print("target:",y_test)
print("predict:", predictions)
