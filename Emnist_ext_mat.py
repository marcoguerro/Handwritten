# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:08:27 2018

@author: alessio

@title: EMNIST - Support Vector Machine
"""

import scipy
import random
import matplotlib.pyplot as plt
import matplotlib
import sklearn.svm as svm
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import time


#file in matlab format
data = scipy.io.loadmat('C:/Users/alessio/Desktop/gzip/matlab/emnist-balanced.mat')

############################  STUDIO DEL DATASET  #############################
len(data['dataset']) # 1 ???
len(data['dataset'][0]) # 1 ???
len(data['dataset'][0][0]) # 3 !!n° datasets!!
# data['dataset'][0][0][0] = dataset train 
# data['dataset'][0][0][1] = dataset test
# data['dataset'][0][0][2] = dataset mapping (47x2 double)
len(data['dataset'][0][0][0]) # 1
len(data['dataset'][0][0][0][0]) #n° datasets
len(data['dataset'][0][0][0][0][0]) # 3 !!n° subsets!!
# data['dataset'][0][0][0][0][0][0] = subset images (nx784 unit8) where n depends on the input
# data['dataset'][0][0][0][0][0][1] = subset labels (nx1 double) where n depends on the input
# data['dataset'][0][0][0][0][0][2] = subset writes (nx1 double) where n depends on the input
len(data['dataset'][0][0][0][0][0][0]) #n° immagini nel train_images
len(data['dataset'][0][0][0][0][0][0][8]) #lunghezza di un immagine in forma vettoriale
for i in ('none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning',
          'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'):
    # così andiamo a vedere le immagini dei vai tipi di modelli
    a = data['dataset'][0][0][0][0][0][0][55]
    some_digit_image = a.reshape(28, 28)
    plt.imshow(np.transpose(some_digit_image), cmap = matplotlib.cm.binary,
           interpolation=i)
    plt.axis("off")
    plt.show()
    print(i,data['dataset'][0][0][0][0][0][1][55])
    massimo = max(a)
    some_digit_image_rescaled = a.reshape(28, 28)/massimo
    plt.imshow(np.transpose(some_digit_image), cmap = matplotlib.cm.binary,
           interpolation=i)
    plt.axis("off")
    plt.show()
    print(i,data['dataset'][0][0][0][0][0][1][55])
###############################################################################
train_set, test_set, mapping_set = data['dataset'][0][0]

class prova():
    def esempio(dat):
        rand = random.sample(range(len(dat[0][0][0])),1)
        a = dat[0][0][0][rand]
        massimo = max(a)
        some_digit_image = a.reshape(28, 28)/massimo
        some_digit_image_transpose = some_digit_image.transpose()
        plt.imshow(some_digit_image_transpose, cmap = matplotlib.cm.binary,
           interpolation="nearest")
        plt.axis("off")
        plt.show()
        print(dat[0][0][1][rand])
        
    def trans_image(dat):
        vect=[]
        for i in range(len(dat[0][0][0])):
            a = dat[0][0][0][i]
            massimo = max(a)
            some_digit_image = a.reshape(28, 28)/massimo
            some_digit_image_transpose = some_digit_image.transpose()
            vect.append(some_digit_image_transpose.ravel())
        return np.array(vect)
    
    def labels(dat):
        return(dat[0][0][1].ravel())
        
#    def out(image, SVC = SVC):
#        
#        predicted = SVC.predict(prova_test_images[995].reshape(1, -1))
#        return predicted

train_images = prova.trans_image(train_set)
train_labels = prova.labels(train_set)
test_images = prova.trans_image(test_set)
test_labels = prova.labels(test_set)

SVC = svm.SVC(C=125,gamma=0.015) #the best: C=125 ,gamma=0.015
print('Inizio fit -> ',time.localtime().tm_hour,':',time.localtime().tm_min,':',time.localtime().tm_sec)
SVC.fit(train_images, train_labels)
print('Fine fit -> ',time.localtime().tm_hour,':',time.localtime().tm_min,':',time.localtime().tm_sec)
expected = test_labels
print('Inizio predict -> ',time.localtime().tm_hour,':',time.localtime().tm_min,':',time.localtime().tm_sec)
predicted = SVC.predict(test_images)
print('Fine predict -> ',time.localtime().tm_hour,':',time.localtime().tm_min,':',time.localtime().tm_sec)
print(metrics.accuracy_score(expected, predicted))
print("Classification report for classifier %s:\n%s\n"
      % (SVC, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


## Utilizzando un campione omogeneo e ridotto
bomber = train_labels[94000:] 
SVC.fit(train_images[94000:], bomber) 
expected = test_labels
predicted = SVC.predict(test_images)
print(metrics.accuracy_score(expected, predicted))

## Plot a nice confsion matrix
import seaborn as sn

labels = list(range(10))
labels.extend(list(map(chr, range(65, 91))))
labels.extend(['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'])
cnf_matrix = metrics.confusion_matrix(expected, predicted)
cnf_matrix_pd = pd.DataFrame(cnf_matrix, index=labels, columns=labels)
plt.figure(figsize = (20,14))
sn.set(font_scale=1.4) #for label size
sn.heatmap(cnf_matrix_pd, annot=True,annot_kws={"size": 5}, fmt='g')    # font size




###############################################################################
#################################  THEANO  ####################################
import theano
import theano.tensor as T
train_set, test_set, mapping_set = data['dataset'][0][0]
train_set_images = train_set[0][0][0]
train_set_labels = train_set[0][0][1]
train_set_writers = train_set[0][0][2]
test_set_images = test_set[0][0][0]
test_set_labels = test_set[0][0][1]
test_set_writers = test_set[0][0][2]

def shared_dataset(data_x, data_y):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')

train_set_x, train_set_y = shared_dataset(train_set_images,train_set_labels)
test_set_x, test_set_y = shared_dataset(test_set_images,test_set_labels)

SVC = svm.SVC(gamma=0.001)
SVC.fit(train_set_x, train_set_y.ravel())
expected = test_set_y.ravel()
predicted = SVC.predict(test_set_x)
print("Classification report for classifier %s:\n%s\n"
      % (SVC, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))



