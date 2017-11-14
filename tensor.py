import tensorflow as tf
import numpy as np

# Csv Name
IRIS_TRAINING = "IRIS.csv"

# Load datasets.
print("Loading dataset...")

Iris_set = tf.contrib.learn.datasets.base.load_csv_with_header(
filename=IRIS_TRAINING,
target_dtype=np.int,
features_dtype=np.float32)

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(Iris_set.data)},
    y=np.array(Iris_set.target),
    num_epochs=None,
    shuffle=True)

# Train model.
classifier.train(input_fn=train_input_fn, steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=train_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))