# TensorFlow Problem sheet
The following is the solutions to the [problem sheet](https://github.com/emerging-technologies/emerging-technologies.github.io/blob/master/problems/tensorflow.md) that is part of the emerging technologies module. There is a .py file containing just the code and its comments so it can be run instantly and a notebook version which goes through each section of the program explaining each answer to the problem sheet parts.

## Set up:
To run the application you must set up tensorflow which can be done using anaconda, the offical tensorflow tutorial can be found [here](https://www.tensorflow.org/install/). You must also install [Python 3.6](https://anaconda.org/anaconda/python). The Iris dataset is also required for this app and is included in the repository.

Once you've downloaded the directory navigate to it in the cmd and use the input below to launch the application.
```
    python tensor.py
```

If you are running the jupyter notebook version instead install the following listed above then navigate to the directory you are storing the tensor.ipynb file in then run the following.
```
    Jupyter notebook
```

The file should be avalible in the jupyter app window then.

---

### IRIS:
Originally published at UCI Machine Learning Repository: Iris Data Set, this small dataset from 1936 is often used for testing out machine learning algorithms and visualizations. Each row of the table represents an iris flower, including its species and dimensions of its botanical parts, sepal and petal, in centimeters.
[More here](https://archive.ics.uci.edu/ml/datasets/iris)

### Numpy: 
This is a python library for complex math equations and for multi-dimensional arrays and matrices. This libray is often used with tensorflow as it makes some calculations required in machine learning much easier and less time consuming.
[More here](http://www.numpy.org/)

### Tensorflow: 
This is a programming library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.
[More here](https://www.tensorflow.org/)

### Pandas: 
This is a library for creating data structures and using data analysis tools with Python. It is one of the most popular libraries for retrieving and saving data to a CSV.
[More here](https://pandas.pydata.org/)

### Jupyter: 
An application that allows you to create and share documents containing code in a easy to read format, making sharing code documentation with others simple and efficent.
[More here](http://jupyter.org/)
    
---

# Summary:
This program uses machine learning through tensor flow to identify flowers based on the IRIS dataset. First we import the dataset using pandas read_csv without headers because we already have our own ones made within the csv. We then create the feature columns for the model and set the shape as 4 since there are 4 columns our model will use. A DNNClassifier Model is created for comparing our dataset to and testing its accuacy. We then print the whole dataset to the user before using pandas to split the dataset in half. Once the dataset is split we save the training and testing sets to there own csvs. We then use the csvs to create seperate datasets and input. After testing these we compare the test to our model and print the accuracy. It should print out a result between 0.9-1.0. Then we use the test set to predict the type of flowers in the data.

### Notebook headers:
- Use Tensorflow to create model.
- Split the data into training and testing.
- Train the model.
- Test the model
- Conclusion

### References:
    - https://www.tensorflow.org/get_started/estimator
    - https://stackoverflow.com/questions/32206661/selecting-only-certain-rows-while-opening-text-file-in-pandas
    - https://deeplearning4j.org/neuralnet-overview