import scipy
import scipy.io
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# Read train input
print "\nReading training data..."
data = scipy.io.loadmat('trainData.mat', squeeze_me=True)
X = np.array(data['trainVectors'])
T = np.array(data['trainLabels'])
print "Got " + str(len(T)) + " training examples."
print "Number of features: " + str(len(X[0])) + "."

# Split data into train-val
print "\nSplitting train data into train/val..."
Xtrain, Xval, Ttrain, Tval = train_test_split(X, T, test_size=0.0001, \
    random_state=42)
print "Train size: " + str(len(Ttrain))
print "Validation size: " + str(len(Tval))

# Reduce dimensionality
print "\nReducing dimensionality..."
feature_clf = LogisticRegression(C=0.015, penalty="l1", dual=False)
feature_clf = feature_clf.fit(Xtrain, Ttrain)
# print feature_clf.feature_importances_  
reducer = SelectFromModel(feature_clf, prefit=True)
Xtrain = reducer.transform(Xtrain)


# Scale data -- apparently it is recommended
scaler = StandardScaler()  
scaler = scaler.fit(Xtrain)  
Xtrain = scaler.transform(Xtrain)


print "New feature dimension: " + str(len(Xtrain[0]))

# Train a classifier
print "\nTraining MLPClassifier on train data..."
clf = MLPClassifier(
    alpha=1e-5, 
    hidden_layer_sizes=(1000),
    verbose=True,
    # learning_rate='adaptive',
    activation='relu',
    # batch_size=256,
    validation_fraction=0.3,
    tol=1e-6)
clf.fit(Xtrain, Ttrain)

# See results on validation
print "\nPredicting validation data..."
Xval = reducer.transform(Xval)
Xval = scaler.transform(Xval)
Tnet = clf.predict(Xval)
confusion = confusion_matrix(Tval, Tnet)
print "\nConfusion matrix:"
print confusion
accuracy = np.trace(confusion) * 1.0 / len(Tval)
print "Accuracy: " + str(accuracy)

# Read test data
print "\nReading test data..."
data = scipy.io.loadmat('testData.mat', squeeze_me=True)
Xtest = np.array(data['testVectors'])
Xtest = reducer.transform(Xtest)
Xtest = scaler.transform(Xtest)
print "Got " + str(len(Xtest)) + " test items."  

# Predict
print "\nPredicting test data..."
Ytest = clf.predict(Xtest)

# Save csv
print "\nSaving csv..."
first_column = np.arange(1, len(Ytest) + 1)
second_column = Ytest
np.savetxt(
    "results.csv", 
    zip(first_column,second_column), 
    header='Id,Prediction', 
    comments='',
    fmt="%.0f",
    delimiter=',')




