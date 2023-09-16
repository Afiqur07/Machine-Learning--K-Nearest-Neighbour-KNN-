# Machine-Learning / Data Science--K-Nearest-Neighbour-KNN-Project
An effective Machine Learning model predicting Dementia using K-Nearest Neighbours.

The Problem: The goal of this project is to compare the glucose metabolism measurements taken from the isthmuscingulate and precuneus regions of the brain between healthy individuals or stable normal controls (sNC) and individuals with advanced or stable DAT (sDAT). Specifically, I attempted to build a kNN classifier that can predict if an individual belongs to the sNC group or the sDAT group based on their brain glucose metabolism signatures.

![image](https://github.com/Afiqur07/Machine-Learning--K-Nearest-Neighbour-KNN-/assets/27920239/d347350e-467d-496d-85cc-960a1a784e7c)


Methods Used: Euclidean and Manhattan.

The Data: The training dataset consists of glucose metabolism features taken from the above mentioned two brain regions across 200 sNC and 200 sDAT individuals given in the train.sNC.csv and train.sDAT.csv files respectively. The test.sNC.csv and test.sDAT.csv files correspond to a test dataset with the same brain glucose metabolism features taken from another 100 sNC and 100 sDAT individuals respectively.

A snippet of one of the Data files:

![image](https://github.com/Afiqur07/Machine-Learning--K-Nearest-Neighbour-KNN-/assets/27920239/5fb4bf9d-c396-4f5a-9bc8-0b0eec8fb238)

Visualization of the Data for Euclidean method: 

![image](https://github.com/Afiqur07/Machine-Learning--K-Nearest-Neighbour-KNN-/assets/27920239/aed6c8d8-c48c-4fc8-802f-533e4f9447da)

Results in the context of over(under)fitting, bias and variance: Overfitting occurs when the decision boundary closely matches the training data for very low values of K (such as K=1). This is characterized by a low training error rate but a high testing error rate, demonstrating the model's poor generalization to unseen, new data. The decision border smooths out as K grows, and the model generalizes to new data more well. Nevertheless, once K grows too big (for instance, K=200), the decision boundary smooths out and becomes extremely straightforward, which leads to underfitting. A low incidence of training and testing errors indicates that the model is too basic to accurately represent the complexity of the underlying data.
When choosing the value of k for a KNN model, bias and variance are generally trade-offs. A smaller k might lead to a simpler model with higher bias, whereas a bigger k could lead to a more complicated model with more variance. The ideal value of k will vary depending on the particular facts and the issue at hand. In our case, K=30 gave the best results when building the classifier using Euclidean distance metric.

Visualization of the Data for Manhattan method: 

![image](https://github.com/Afiqur07/Machine-Learning--K-Nearest-Neighbour-KNN-/assets/27920239/e52a7b9e-2004-4727-a1a2-96da3242c5cf)

Results in context of comparison to the Euclidean method: The performance of the classifier in the context of k-NN can be significantly impacted by the
distance metric that is selected. By comparing the Manhattan distance metric to the Euclidean
distance metric, we can see a clear difference in the classification borders between the two
produced pictures.
By employing the Manhattan distance metric instead of the Euclidean distance metric, the
classification border is less smooth and has a more step-like pattern. This is predicted since the
Manhattan distance, which is calculated by adding the absolute differences between two points'
attributes, frequently results in such borders. While Euclidean distance tends to generate more
rounded borders, it is calculated by taking the square root of the sum of squared differences.
The bias and variance of the KNN classifier can be impacted by the distance metric selection. The
classifier may match the training data better but may also be more sensitive to changes in the data
if a more sophisticated distance measure, such as Euclidean distance, is used. The classifier may
be less sensitive to changes in the data, but it may also underfit the training data, if you choose a
simpler distance measure, such the Manhattan distance, which has a greater bias and a smaller
variance.
In general, the particular situation and the properties of the data should be taken into consideration
while selecting the distance measure. Using a simpler distance measure may be more acceptable
in some situations while a more complicated distance metric may be more appropriate in others. It
is also typical to experiment with several distance measurements and evaluate their effectiveness
before choosing one.

More Experimentation: Based on the experiments in Experiment 1 and Experiment 2, I selected the distance metric (i.e., Euclidean or Manhattan) that
leads to a lower test error rate. Using this chosen distance metric I generated the “Error rate versus Model capacity”
plot. I parameterized “Model capacity” as “ 1k” and explored the parameter space from “0.01” to “1.00”. The “x-axis” was plotted using the “log-scale” and the training and test
rate error curves. 

The Plot: 

![image](https://github.com/Afiqur07/Machine-Learning--K-Nearest-Neighbour-KNN-/assets/27920239/b7881d3b-d7f1-4b59-8b86-0ab579996d1d)

I Discuss the trend of the training and test error rate curves in the context of model capacity, bias and variance. I commented on the over(under)fitting zones in the plot :- 

When the model capacity rises, or as the value of 1/k lowers, we can observe from the figure that
the training and testing error rates both drop. This tendency is anticipated since a more complicated
model that can better match the training data results from a larger model capacity.
When we get to the right side of the figure, the pace of decrease in the error rates begins to slow
down, showing that, beyond a certain point, expanding the model capacity no longer significantly
improves the model's performance. Alternatively said, the model begins to overfit the training set
of data. The underfitting zone, where the model capacity is low and both the training and testing
error rates are high, can be seen on the left side of the figure. This shows that the model is
underperforming on both the training and testing datasets, indicating that it is too simplistic to
capture the underlying patterns in the data.
On the right-hand side of the figure, when the model capacity is large, the training error rate is
extremely low, but the testing error rate is high, is where the overfitting zone can be seen. This
suggests that the model is overly complicated, which causes it to begin capturing noise and random
changes in the training data, leading to subpar performance on the hidden testing data. In other
words, rather than discovering underlying patterns, the model is just memorizing the training data.
The point at which the testing error rate is lowest—in this example, about 1/k = 0.3—is the sweet
spot, or the ideal value of the model capacity. The model is sophisticated enough to capture the
underlying patterns in the data but not so complicated that it starts to overfit the training data. This
illustrates the ideal balance between bias and variance.
