# Project 1: Naive Bayes
# Author: Jawad Ahmad Khan
# Purpose: Predict a student's result using study hours.

from sklearn.naive_bayes import GaussianNB

# Training data: hours studied
X = [[1], [2], [3], [4], [5]]

# Labels: 0 = Fail, 1 = Pass
y = [0, 0, 1, 1, 1]

# Create the Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X, y)

# Predict the result for a student studying 3 hours
prediction = model.predict([[3]])

if prediction[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")
