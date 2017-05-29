import scipy
import scipy.io
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 

# Read train input
print "Reading training data..."
data = scipy.io.loadmat('trainData.mat', squeeze_me=True)
X = np.array(data['trainVectors'])
T = np.array(data['trainLabels'])



# Split data into train-val
print "\nSplitting train data into train/val..."
Xtrain, Xval, Ttrain, Tval = train_test_split(X, T, test_size=0.001, random_state=42)
print "Train size: " + str(len(Ttrain))
print "Validation size: " + str(len(Tval))

scaler = StandardScaler()  
# Scale data -- apparently it is recommended
scaler.fit(Xtrain)  
Xtrain = scaler.transform(Xtrain)

# Train a classifier
print "Training MLPClassifier..."
clf = MLPClassifier(
    alpha=1e-5, 
    hidden_layer_sizes=(1200),
    verbose=True,
    validation_fraction=0.1,
    tol=-1,
    max_iter=1000)

Xval = scaler.transform(Xval)
clf.fit(Xtrain, Ttrain)

# See results on validation
print "\nPredicting validation data..."
Tnet = clf.predict(Xval)
confusion = confusion_matrix(Tval, Tnet)
print "\nConfusion matrix:"
print confusion
accuracy = np.trace(confusion) * 1.0 / len(Tval)
print "Accuracy: " + str(accuracy)


# Read test data
print "Reading test data..."
data = scipy.io.loadmat('testData.mat', squeeze_me=True)
Xtest = np.array(data['testVectors'])
Xtest = scaler.transform(Xtest)  

# Predict
print "Predicting..."
Ytest = clf.predict(Xtest)

# Save csv
print "Saving..."
first_column = np.arange(1, len(Ytest) + 1)
second_column = Ytest
np.savetxt(
    "results2.csv", 
    zip(first_column,second_column), 
    header='Id,Prediction', 
    comments='',
    fmt="%.0f",
    delimiter=',')




