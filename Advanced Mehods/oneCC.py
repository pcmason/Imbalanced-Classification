"""
Example code using sklearn methods to run One-Class SVM, Isolation Forest, Minimum Covariance Determinant & Local
Outlier Factor on a synthetic imbalanced binary classification dataset.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
import numpy as np


# Method to fit a model on only the majority class of the training set
def fit_majority(model, trainX, trainY):
    train_x = trainX[trainY == 0]
    model.fit(train_x)
    return model


# Make prediction with a lof model
def lof_predict(model, trainX, testX):
    # Create one large dataset
    composite = np.vstack((trainX, testX))
    # Make prediction on composite data
    yhat = model.fit_predict(composite)
    # Return just the predictions on the test set
    return yhat[len(trainX):]


# Method to calculate f1_score based on anomaly detection
def outlier_detection_score(model, testX, testY, trainX, lof_method=False):
    # Add condition for if method is lof or not as that uses above lof_predict method for yhat
    if lof_method:
        yhat = lof_predict(model, trainX, testX)
    else:
        # Detect outliers in the test set
        yhat = model.predict(testX)
    # Mark inliers 1, outliers -1
    testY[testY == 1] = -1
    testY[testY == 0] = 1
    # Calculate F-Measure score
    score = f1_score(testY, yhat, pos_label=-1)
    print('F1 Score: %.3f' % score)


if __name__ == '__main__':
    # Generate synthetic imbalanced classification dataset
    x, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.99], flip_y=0,random_state=4)
    # Split into train & test splits
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.5, random_state=2, stratify=y)

    # Define outlier detection model One-Class SVM
    oc_svm = OneClassSVM(gamma='scale', nu=0.01)
    # Fit on majority class
    oc_svm = fit_majority(oc_svm, train_x, train_y)
    # Evaluate One-Class SVM model
    outlier_detection_score(oc_svm, test_x, test_y, train_x)

    # Define outlier detection model Isolation Forest
    iso_forest = IsolationForest(contamination=0.01)
    # Fit on majority class
    iso_forest = fit_majority(iso_forest, train_x, train_y)
    # Evaluate Isolation Forest model
    outlier_detection_score(iso_forest, test_x, test_y, train_x)

    # Define outlier detection for Minimum Covariance Determinant
    ell_env = EllipticEnvelope(contamination=0.01)
    # Fit on the majority
    ell_env = fit_majority(ell_env, train_x, train_y)
    # Evaluate MCD model
    outlier_detection_score(ell_env, test_x, test_y, train_x)

    # Define outlier detection model for Local Outlier Factor
    lof = LocalOutlierFactor(contamination=0.01)
    # Can run LOF method just by using outlier_detection_score & setting lof_method to True
    outlier_detection_score(lof, test_x, test_y, train_x, lof_method=True)


