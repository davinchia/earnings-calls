import numpy as np
import os
import nltk
import time
import sys 

from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from dbn import SupervisedDBNClassification

# DEFAULT VALUES
np.random.seed(1337)  # for reproducibility
smlTrnUrl = "./latest/small-set/"
smlYUrl = smlTrnUrl + "/YES/"
smlNUrl = smlTrnUrl + "/NO/"
allTrnUrl = "./latest/split-transcripts"
allYUrl = allTrnUrl + "/YES/"
allNUrl = allTrnUrl + "/NO/"
stopset = set(stopwords.words('english'))

def getFile(fileurl, arr, lblarr, label): 
  listOfFiles = os.listdir(fileurl)
  while len(listOfFiles) > 0:
    filename = listOfFiles.pop(0)
    if (filename != ".DS_Store"):
      with open(fileurl + filename) as f:
        file = f.read() # returns a string of document
        file = file.replace("\xc2\xa0", " ").lower().decode("utf-8", "replace")
        arr.append(file)
        lblarr.append(label)
  return (arr, lblarr)

# Create our main processors
mf = 10000
max_df = 0.8
min_df = 5
print("  Vectoriser Parameters - max_features: %i, max_df: %.3f, min_df: %i" % (mf, max_df, min_df))
vectorizer = TfidfVectorizer(max_features = mf, max_df = max_df,
                             min_df = min_df, use_idf = True, 
                             ngram_range = (1, 3), stop_words = stopset)
dimension = 1000
print("  SVD Parameters - # Dimension: %i" % (dimension))
svd = TruncatedSVD(dimension)
lsa = make_pipeline(svd, Normalizer(copy=False))

# Grab the list of files
# Y = 1, N = 0
textlist, Y = getFile(allYUrl, [], [], 1)
textlist, Y = getFile(allNUrl, textlist, Y, 0)

# Split up data
train_url, test_url, Y_train, Y_test = train_test_split(textlist, Y, test_size=0.1, random_state=0)

# If-idf with training data
t0 = time.time()
X_train_ifidf = vectorizer.fit_transform(train_url)
print("  Train If-idf done in %.3fsec" % (time.time() - t0))

# LSA with training data
t0 = time.time()
X_train = lsa.fit_transform(X_train_ifidf)
print("  Train LSA done in %.3fsec" % (time.time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

# Now apply the transformations to the test data as well.
t0 = time.time()
X_test_tfidf = vectorizer.transform(test_url)
print("  Test If-idf done in %.3fsec" % (time.time() - t0))
t0 = time.time()
X_test = lsa.transform(X_test_tfidf)
print("  Test LSA done in %.3fsec" % (time.time() - t0))

hls = [800, 500, 250]
l_r_rmb = 0.05
lr = 0.1
n_e_rbm = 30
n_i_backprop = 500
bs = 1000
dp_p = 0.2
# Training
print("  DBN Parameters:" )
print("    hidden_layers_structure: ")
print(hls)
print("    learning_rate_rbm: %.3f. learning_rate: %.3f, n_epochs_rbm: %i, n_iter_backprop: %i, batch_size: %i, dropout_p: %.3f"
       %(l_r_rmb, lr, n_e_rbm, n_i_backprop, bs, dp_p))
classifier = SupervisedDBNClassification(hidden_layers_structure=hls,
                                         learning_rate_rbm=l_r_rmb,
                                         learning_rate=lr,
                                         n_epochs_rbm=n_e_rbm,
                                         n_iter_backprop=n_i_backprop,
                                         batch_size=bs,
                                         activation_function='relu',
                                         dropout_p=dp_p)
classifier.fit(X_train, Y_train)

# # Test
Y_pred = classifier.predict(X_test)
print 'Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred)

