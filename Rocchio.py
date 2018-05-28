from __future__ import division
import numpy as np
import scipy as sc
from prettyprint import pp
import os
import re
from datetime import datetime as dt
from nltk.corpus import stopwords

#index label in the dictionary
idx_lbl = 'idx'
dfreq_lbl = "docfreq"



pattern = re.compile(r'([a-zA-Z]+|[0-9]+(\.[0-9]+)?)')

def tokenizeDoc(doc_address, min_len = 0, remove_numerics=True):
    from string import punctuation, digits
    tokens = []
    try:
        f = open(doc_address)
        raw = f.read().lower()
        text = pattern.sub(r' \1 ', raw.replace('\n', ' '))
        text_translated = ''
        if remove_numerics:
            text_translated = text.translate(None, punctuation + digits)
        else:
            text_translated = text.translate(None, punctuation)
        tokens = [word for word in text_translated.split(' ') if (word and len(word) > min_len)]
        tokens=[word for word in tokens if word not in set(stopwords.words("english"))]


        f.close()
    except:
        print "Error: %s couldn't be opened!", doc_address
    finally:
        return tokens




def createDictionary(classes, tokens_pool):
   
    token_dict = {}
    idx = 0 #a unique index for words in dictionary
    for cl in classes:
        for tokens_list in tokens_pool[cl]:
            for token in tokens_list:
                if token in token_dict:            
                    if cl in token_dict[token]:
                        token_dict[token][cl] += 1
                    else:
                        token_dict[token][cl] = 1
                else:
                    token_dict[token] = {}
                    token_dict[token][idx_lbl] = idx
                    idx += 1
                    token_dict[token][cl] = 1
    return token_dict



def createTokenPool(classes, paths):
    token_pool = {}
    for cl in classes:
        token_pool[cl] = []
        for path in paths[cl]:
            token_pool[cl].append(tokenizeDoc(path))

    return token_pool



def train_test_split(ratio, classes, files):
    train_dict = {}
    test_dict = {}
    for cl in classes:
        train_cnt = int(ratio * len(files[cl]))
        train_dict[cl] = files[cl][:train_cnt]
        test_dict[cl] = files[cl][train_cnt:]
    return train_dict, test_dict


class Rocchio:
    
    def __init__(self, class_labels, tdict):
        
        self.k = len(class_labels)
       
        self.centroids = []
        self.lbl_dict = dict(zip(class_labels, range(self.k)))
        self.class_labels = class_labels
        self.tdict = tdict
        self.ctermcnt = np.zeros((self.k, 1))           # total number of terms in a class


    def train(self, token_pool, tfidf_but_smoothing = True):
        
        if len(token_pool) != len(self.class_labels):
            print "error! number of classes don't match"
            return

        # now find the term frequency for each class
        for term, data in self.tdict.items():
            for cl in self.lbl_dict:
                if cl in data:
                    self.ctermcnt[self.lbl_dict[cl], 0] += data[cl]

        # now normalize each input vector and add it to its corresponding centroid vector
        for cl in self.class_labels:
            self.centroids.append(np.zeros((len(self.tdict), 1)))
            for doc in token_pool[cl]:
                vec = self.__createNormalizedVectorRepresentation(doc, cl)
                self.centroids[self.lbl_dict[cl]] += vec

            self.centroids[self.lbl_dict[cl]] /= len(token_pool[cl])


    def predict(self, doc):
        
        doc_vec = self.__createNormalizedVectorRepresentation(doc, None)

        distances = []
        for i in range(self.k):
            distances.append(np.linalg.norm(doc_vec - self.centroids[i]))



        return self.class_labels[distances.index(min(distances))]


    def predictPool(self, doc_collection):
        lbl_pool = {}
        for cl in self.class_labels:
            lbl_pool[cl] = []
            for doc in doc_collection[cl]:
                lbl_pool[cl].append(self.predict(doc))

        return lbl_pool


    def __createNormalizedVectorRepresentation(self, tokens_list, cl = None, tfidf = False):
       
        vec = np.zeros((len(self.tdict), 1))
        for token in tokens_list:
            if token in self.tdict:
                vec[self.tdict[token][idx_lbl], 0] += 1

        token_set = set(tokens_list)
        if tfidf:
            if cl != None:
                for term in token_set:
                    if cl in self.tdict[term]:
                        vec[self.tdict[term][idx_lbl], 0] *= np.log(self.ctermcnt[self.lbl_dict[cl], 0] * 1.0 / self.tdict[term][cl])


        norm_vec = np.linalg.norm(vec)
        vec = (vec / (norm_vec + 1e-14))
        return vec


def calculateMetrics(class_labels, lbl_pool):
    
    metrics = {}
    for cl in class_labels:
        metrics[cl] = {}
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for lbl in lbl_pool[cl]:
            if lbl == cl:
                tp += 1
            else:
                fp += 1
        for ncl in class_labels:
            if ncl != cl:
                for lbl in lbl_pool[ncl]:
                    if lbl == cl:
                        fn += 1
                    else:
                        tn += 1

        metrics[cl]["tp"] = tp
        metrics[cl]["tn"] = tn
        metrics[cl]["fp"] = fp
        metrics[cl]["fn"] = fn

    return metrics

def preProcessDataset(root_path):
    if os.path.exists('processedDatasetSentence/'):
        return 'processedDatasetSentence/'
    else:
        os.makedirs('processedDatasetSentence')
        os.makedirs('processedDatasetSentence/AIMX')
        os.makedirs('processedDatasetSentence/BASE')
        os.makedirs('processedDatasetSentence/CONT')
        os.makedirs('processedDatasetSentence/MISC')
        os.makedirs('processedDatasetSentence/OWNX')

        #top_view folders
        folders = [root_path + folder + '/' for folder in os.listdir(root_path)]

        #there are only 4 classes
        class_titles = os.listdir(root_path)
        txtFiles = os.listdir(root_path)
        #print (txtFiles)

        counter=0
        for file in txtFiles:
            if file.endswith(".txt"):
                with open(root_path+file) as f:
                   print (root_path+file+"\n")
                   content = f.readlines()           
                   for line in content:
                    tokens=line.split( )

                    if(tokens[0]=='###'):

                        tokens.pop(0)
                        tokens.pop(0)
                        tokens.pop(0)
                    else:
                        outFile= open('processedDataset/'+tokens[0]+'/'+str(counter)+".txt","w+")
                        tokens.pop(0)
                    counter+=1
                    tokens=[word for word in tokens if word not in set(stopwords.words("english"))]
                    tokens=[word for word in tokens if word not in ('CITATION','NUMBER','SYMBOL')]
                    for token in tokens:
                        outFile.write (token+" ")
    return 'processedDatasetSentence/'

def main():

    # remove files  jdm_annotate4_220_*.txt

    root_path = 'legal/' #sentenceCorpusProcessed #legal  emails

    print "Root Path: ", root_path



    folders = [root_path + folder + '/' for folder in os.listdir(root_path)]
    class_titles = os.listdir(root_path)
    
    files = {}
    for folder, title in zip(folders, class_titles):
        files[title] = [folder + f for f in os.listdir(folder)]
    
    train_test_ratio = 0.8

    train, test = train_test_split(train_test_ratio, class_titles, files)

    pool = createTokenPool(class_titles, train)
    print len(pool[class_titles[0]])
    tdict = createDictionary(class_titles, pool)
    print len(tdict)

    print "Rocchio's turn"

    rocchio = Rocchio(class_titles, tdict)

    start = dt.now()
    rocchio.train(pool)
    end = dt.now()


    print 'elapsed time for training rocchio'
    print end - start
   # print (rocchio.centroids)
    for c in rocchio.centroids:
        print np.linalg.norm(c - rocchio.centroids[0])
    id = 0
    

    start = dt.now()
    lbl = rocchio.predict(tokenizeDoc(test[class_titles[id]][3]))
    end = dt.now()
    

    print 'elapsed time for testing rocchio'
    print end - start


    print lbl 
    

    test_pool = createTokenPool(class_titles, test)
    
    start = dt.now()
    test_lbl_pool = rocchio.predictPool(test_pool)
    end = dt.now()

    print 'elapsed time for testing a pool of documents'
    print end - start


    metrics = calculateMetrics(class_titles, test_lbl_pool)
    total_F = 0
    for cl in class_titles:
        print cl
        P = (metrics[cl]["tp"] * 1.0 / (metrics[cl]["tp"] + metrics[cl]["fp"]))
        R = (metrics[cl]["tp"] * 1.0 / (metrics[cl]["tp"] + metrics[cl]["fn"]))
        print 'precision = ', P
        print 'recall = ', R
        if P!=0 and R!=0:
            Acc = ((metrics[cl]["tp"] + metrics[cl]["tn"])* 1.0 / (metrics[cl]["tp"] + metrics[cl]["fp"] + metrics[cl]["fn"] + metrics[cl]["tn"]))
            F_1 = 2 * R * P / (R + P)
        else:
            Acc=0
            F_1=0
        total_F += F_1
       
        print 'accuracy = ', Acc
        print ' '

    print 'macro-averaged F measure', (total_F / len(class_titles))



if __name__ == "__main__":
    main()
