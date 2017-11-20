# TensorFlow Problem sheet
The following is the solutions to the [problem sheet](https://github.com/emerging-technologies/emerging-technologies.github.io/blob/master/problems/tensorflow.md) that is part of the emerging technologies module.

## Set up:
To run the application you must set up tensorflow which can be done using anaconda, the offical tensorflow tutorial can be found [here](https://www.tensorflow.org/install/). You must also install [Python 3.6](https://anaconda.org/anaconda/python). The Iris dataset is also required for this app and is included in the repository.

Once you've downloaded the directory navigate to it in the cmd and use the input below to launch the application.
```
    python tensor.py
```

## IRIS:
Originally published at UCI Machine Learning Repository: Iris Data Set, this small dataset from 1936 is often used for testing out machine learning algorithms and visualizations. Each row of the table represents an iris flower, including its species and dimensions of its botanical parts, sepal and petal, in centimeters.

## Imports:
    - Numpy: We import this library for complex math equations and for multi-dimensional arrays and matrices that the library allows us to use, these will be useful when manipulating our data.
    - Tensorflow: Import tensorflow for machine learning algorithms 
    - Pandas: This is a library for mainpulating csvs and is what we will use to seperate the data. 
    
---

# Overview:
This program uses machine learning through tensor flow to identify flowers based on the IRIS dataset. First we import the dataset using pandas read_csv without headers because we already have our own ones made within the csv. We then create the feature columns for the model and set the shape as 4 since there are 4 columns our model will use. A DNNClassifier Model is created for comparing our dataset to and testing its accuacy. We then print the whole dataset to the user before using pandas to split the dataset in half. Once the dataset is split we save the training and testing sets to there own csvs. We then use the csvs to create seperate datasets and input. After testing these we compare the test to our model and print the accuracy. It should print out a result between 0.9-1.0. Then we use the test set to predict the type of flowers in the data.