import os
import numpy as np
import tensorflow as tf

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

  # Train model.
  print("Training model..")
  classifier.train(input_fn=train_input_fn, steps=2000)

if __name__ == "__main__":
    main()