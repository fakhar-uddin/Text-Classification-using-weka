import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sys
import os
import glob


def bagOfWords(files_data):
    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    return count_vector.fit_transform(files_data)


def main():

    dir_path = 'processedDatasetSentence'

    # load data
    print ('Loading files into memory')
    files = sklearn.datasets.load_files(dir_path)


    # calculate the BOW representation
    print ('Calculating BOW')
    word_counts = bagOfWords(files.data)


    # TFIDF
    print ('Calculating TFIDF')
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts)
    
    n_neighbors = 3
    weights = 'distance'
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

    # test the classifier
    print '\n\n'
    print ('Testing classifier with train-test split')
    test_classifier(X, files.target, clf, test_size=0.1, y_names=files.target_names, confusion=False)



def test_classifier(X, y, clf, test_size=0.4, y_names=None, confusion=False):
    # train-test split
    print 'test size is: %2.0f%%' % (test_size * 100)
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

    clf.fit(X_train, y_train)
    
    y_predicted = clf.predict(X_test)
    #print X_test

    if not confusion:
        print ('Classification report:')
        print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
    else:
        print ('Confusion Matrix:')
        print sklearn.metrics.confusion_matrix(y_test, y_predicted)

if __name__ == '__main__':
    main()
