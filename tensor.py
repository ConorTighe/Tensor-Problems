import os
import numpy as np
import tensorflow as tf
import pandas as pd

# I use this line to suppress warnings, tensorflow tells me that my machine can run tensorflow faster with some configuration.
# This line is not necessary but makes the output cleaner
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# IRIS data file name set up
IRIS = "IRIS.csv"

def main():
  # Load data from csv.
  print("Loading IRIS..")
  fulldata = pd.read_csv(IRIS, header=None)
  
  # Prepare data for modeling by checking the columns with tf.feature_column.numeric_column.
  # We have the Petal length + width and the Speal length + width so we set the shape to 4 since there are 4 columns to feature
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Fit our data to a Deep Neural Network Classifier Model and make a temp directory to hold it
  print("Creating Model..")
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="Model")
  
  # Display the full dataset to the user and
  # Split our mixed up IRIS data into training and testing
  print("IRIS Dataset:")
  print(fulldata)
  training_set = pd.read_csv(IRIS, header=None, nrows=100 , names = ["100", "4", "setosa", "versicolor", "virginica"])
  test_set = pd.read_csv(IRIS, header=None, skiprows=100, nrows=151, names = ["51", "4", "setosa", "versicolor", "virginica"])

  # Set the file name for both CSV
  train_data = "trainIRIS.csv"
  test_data = "testIRIS.csv"

  # Save them to individual CSV files
  training_set.to_csv(train_data, index = False, header=True)
  test_set.to_csv(test_data, index = False, header=True)

  # Create training dataset
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=train_data,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Create testing dataset
  test = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=test_data,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Define the training inputs
  print("Preparing training with inputs..")
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  print("Training model..")
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test.data)},
      y=np.array(test.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy using test input on classifer model.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  # Print model accuracy
  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Predict the test array
  print("Predicting data..")
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(test.data)},
  y=np.array(test.target),
  num_epochs=1,
  shuffle=False)

  # Get predictied classes results from output
  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  # Display predicted classes
  print("Prediction results:")
  print('[%s]' % ', '.join(map(str, predicted_classes)))


if __name__ == "__main__":
    main()