# Text-Classification-using-weka(naive bayes) and python for rochio and KNN 
Weka library has been used in this project to classify using naive bayes mathod. If you want to run this project, you have to copy all the depending libraries in project path.
For rochio and KNN, python code is written, you have to place data folders accordingly.

#Datasets
=========
https://archive.ics.uci.edu/ml/datasets/DBWorld+e-mails
https://archive.ics.uci.edu/ml/datasets/Sentence+Classification

Naive Bayes Classifier
======================
This is a simple (naive) classification method based on Bayes rule. It relies on a very simple representation of the document called the bag of words representation.
Imagine we have 2 classes, positive and negative, and our input is a text representing a review of a movie. We want to know whether the review was positive or negative. So we may have a bag of positive words e.g. love, amazing,hilarious, great, and a bag of negative words e.g. hate, terrible.
We may then count the number of times each of those words appears in the document, in order to classify the document as positive or negative.
This technique works well for topic classification; say we have a set of academic papers, and we want to classify them into different topics like computer science, biology, mathematics.
For a document d and a class c, and using Bayes’ rule,
P( c | d ) = [ P( d | c ) x P( c ) ] / [ P( d ) ]
P( c ) is the total probability of a class. In the case of classes positive and negative, we would be calculating the probability that any given review is positive or negative, without actually analyzing the current input document.
This is calculated by counting the relative frequencies of each class in a corpus.
To calculate the Naive Bayes probability, P( d | c ) x P( c ), we calculate P( xi | c ) for each xi in d, and multiply them together.


K Nearest Neighbor
==================
This is a simple Knn classification method based. It relies on a very simple representation of the document called the bag of words representation. 
We have applied knn on sentence classification. I computes the BOW of each document and then it computes tf-idf of the documents, it then applies knn which select neighbors based on distance as metric. It classifies the document based on minimum distance.


Rocchio Classification
======================
Text classification using Rocchio's algorithm. Each document is represented in a vector space. In the training phase, centroids for each class of documents are found. In the testing phase, test document's distance to each centroid is calculated and document is assigned to the closest centroid's class. 
Sentence classification dataset is used we have pre-processed the dataset removed the stop words and dataset contains some words like Citations, Number and Symbol have been removed with increase the precision, recall and F1 score.

Precision, Recall & F-measure
============================
We need more accurate measure than contingency table True, false positive and negative.
Precision: % of selected items that are correct. Tp/ Tp + Fp
Recall: % of correct items that are selected. Tp / Tp + Fn
There is a tradeoff between precision & recall. A standard way proposed to combine this measure is F-measure. F-Measure is weighted harmonic mean.
Mostly balanced F measure (F1 measure) is used.
F1=2PR/P+R
