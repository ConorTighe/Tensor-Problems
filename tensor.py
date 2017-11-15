import os
import numpy as np
import tensorflow as tf
import random as rd
import pandas as pd
import csv

# I use this line to suppress warnings, tensorflow tells me that my machine can run tensorflow faster with some configuration.
# This line is not necessary but makes the output cleaner
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# IRIS data set
IRIS = "IRIS.csv"

def main():
  # Load data from csv.
  print("Loading IRIS..")
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Prepare data for modeling by checking the columns with tf.feature_column.numeric_column.
  # We have the Petal length + width and the Speal length + width so we set the shape to 4 since there are 4 columns to feature
  print("Checking values..")
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Fit our data to a Deep Neural Network Classifier Model and make a temp directory to hold it
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="Model")
  # Define the training inputs
  print("Preparing training with inputs..")
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)
  
  IRIS_data = "IRIS_data.csv"
  irisd = pd.read_csv(IRIS_data, delimiter=',')
  irisd = irisd.values
  print(irisd)

  # Mix up data so we can test our training set
  train_input_fn.train.shuffle_batch()
  print("Shuffled:")
  print(train_input_fn)
  sp1, sp2 = train_input_fn.split()

  # Split our mixed up IRIS data into training and testing
  train_data = irisd[:50]
  test_data = irisd[50:]

  # Train model.
  print("Training model..")
  classifier.train(input_fn=train_input_fn, steps=2000)

  #print("Train shuffled and split")
  #print(train_data)
  #print("Test shuffled and split")
  #print(test_data)

if __name__ == "__main__":
    main()