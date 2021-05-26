# Wine-Classification
Here is the [Wine Dataset](https://archive.ics.uci.edu/ml/datasets/Wine) that was used in this project.

### SYSTEM REQUIREMENTS
1. Python3 needs to be installed on the PC.
2. Important statistical libraries such as NumPy, pandas, scikit-learn, etc.
3. The code can be run in Jupyter-notebook if installed otherwise, Google Colab can be used

### DATA SET INFORMATION:
These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. The data is Multivariate.
The attributes are:
1) Alcohol 
2) Malic acid 
3) Ash 
4) Alkalinity of ash 
5) Magnesium 
6) Total phenols 
7) Flavanoids
8) Non Flavonoid phenols
9) Proanthocyanins
10) Color intensity
11) Hue
12) OD280/OD315 of diluted wines
13) Proline

![image](https://github.com/daksh6171/Wine-Classification/blob/main/Images/Reading%20data.png)
![datset](https://github.com/daksh6171/Wine-Classification/blob/main/Images/Dataset.png)

### ATTRIBUTE AND CLASS INFORMATION
There are 13 attributes in total. All attributes are continuous data. Dataset was verified and contained no missing attribute values.

![no_missing_attribute](https://github.com/daksh6171/Wine-Classification/blob/main/Images/No%20missing%20attribute.png)

There are a total of 3 classes and a total of 178 instances.:
* Class 1 - 59 instances
* Class 2 - 71 instances
* Class 3 - 48 instances

### DATA PREPROCESSING
We have calculated the measures of central tendencies and it can be summarized in the contingency table below:

![central_tendency](https://github.com/daksh6171/Wine-Classification/blob/main/Images/measure%20of%20the%20central%20tendency.png)

### PRELIMINARY ANALYSIS OF DATA
We can observe that the data is sorted based on class labels. So we infer that we must randomize it before splitting. By observing the data description, we can infer that the features are not closely related to each other. For e.g. Proline values dominate overall central tendency measures over attributes such as Ash content. Hence, we can infer that there is a need for normalizing the attribute values to be contained in a similar domain. We decided to normalize the data after splitting it into test and training set.

### TRAINING DATA vs TEST DATA
Here we are using a simple holdout method where we are keeping 25% of the data for test and 75% for training. The data has been randomized before split as observed in the preliminary stage.

Train data:

![train_data](https://github.com/daksh6171/Wine-Classification-Data-Science-Project/blob/main/Images/75%25%20of%20the%20data%20is%20training%20dataset.png)

Test data:

![test_data](https://github.com/daksh6171/Wine-Classification-Data-Science-Project/blob/main/Images/25%25%20of%20the%20dataset%20is%20test%20dataset.png)

Random values from training data:

![train_data_stats](https://github.com/daksh6171/Wine-Classification/blob/main/Images/train%20data%20size.png)

Random values from test data:

![test_data_stats](https://github.com/daksh6171/Wine-Classification/blob/main/Images/test%20data%20stats.png)

### NORMALIZATION OF TRAINING AND TEST DATA
We performed two types of scaling:

1. Standard Scaling

![standard](https://github.com/daksh6171/Wine-Classification/blob/main/Images/standard%20scaling.png)

2. Minmax Scaling

![minmax](https://github.com/daksh6171/Wine-Classification/blob/main/Images/minmax%20scaling.png)

We have decided to use Min-Max Scaling over Standard Scaling. Since the values are much closer to each other in min-max and since we know that classifiers such as SVM depend on how good the scaling is performed, min-max dominates over standard scaling.

### RE-ANALYSIS OF DATA AFTER PARTITIONING (TRAINING AND TEST SETS) AND NORMALIZING
1. Class Distribution
* Class distribution in Original Dataset

![original](https://github.com/daksh6171/Wine-Classification/blob/main/Images/class%20distribution%20in%20original%20dataset.png)

* Class distribution in Training Dataset

![training](https://github.com/daksh6171/Wine-Classification/blob/main/Images/class%20distribution%20in%20training%20dataset.png)

* Class distribution in Testing Dataset

![testing](https://github.com/daksh6171/Wine-Classification/blob/main/Images/class%20distribution%20in%20testing%20dataset.png)

We can see that test-class distribution is roughly equivalent in all three datasets. This means accuracy is a good way of measuring classifiers (due to the absence of bias).

2. Class as a funtion of different attributes

![sharing](https://github.com/daksh6171/Wine-Classification/blob/main/Images/sharing%20attributes%20x%20per%20y.png)

These are the subplots of various important attributes and the relation between attribute-values and label classes.

![graph](https://github.com/daksh6171/Wine-Classification/blob/main/Images/contents%20to%20drop.png)

Attributes and their influence on classification have been calculated in order to drop those attributes which are least important like Non Flavonoid phenols, Ash.

### CLASSIFICATION AND CHOOSING APPROPRIATE CLASSIFIER
We have used SVM (with the linear kernel), Naive Bias Classifiers and Random Forest (decision tree) classifiers.

![result](https://github.com/daksh6171/Wine-Classification/blob/main/Images/conclusion%20table.png)

### CONCLUSION
Although it may appear counter-intuitive, we conclude naive bias classifiers may be the best classifier in this case. This is because the classifiers are showing extremely high accuracy and we must try to avoid overfitting

### References
* [SVM](https://scikit-learn.org/stable/modules/svm.html)
* [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [Matplotlib](https://matplotlib.org/)
* [Feature Selection Techniques in Machine Learning with Python](https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e)
