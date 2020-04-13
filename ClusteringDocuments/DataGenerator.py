from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import joblib
from time import time
import numpy as np
import pandas as pd
import os

import Constants as conts

def computeHashVectorizer():
    hasher = HashingVectorizer(n_features=conts.HASH_CONSTS["NUM_FEATURES"],
                                stop_words=conts.HASH_CONSTS["STOP_WORDS"],
                                alternate_sign=conts.HASH_CONSTS["ALTERNATE_SIGN"],
                                norm=conts.HASH_CONSTS["NORM"])
    if conts.HASH_CONSTS["USE_IDF"]:
        # Perform an IDF normalization on the output of HashingVectorizer
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = hasher
    return vectorizer

def computeTfidfVectorizer():
    vectorizer = TfidfVectorizer(max_df=conts.TF_IDF_CONSTS["MAX_DF"],
                                max_features=conts.TF_IDF_CONSTS["NUM_FEATURES"],
                                min_df=conts.TF_IDF_CONSTS["MIN_DF"],
                                stop_words=conts.TF_IDF_CONSTS["STOP_WORDS"],
                                use_idf=conts.TF_IDF_CONSTS["USE_IDF"])
    return vectorizer

def applyLsaProcessing(X, true_k):
    print("Starting to perform dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(true_k)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    print("Run time for LSA processing operation is: {0:0.2f} seconds".format((time() - t0)) )
    return X, svd

def getNewsCategories():
  categories = [
      'alt.atheism',
      'comp.graphics',
      'talk.religion.misc',
      'sci.space',
  ]
  return categories

def isDataExist():
    print("In isDataExist()..")
    t0=time()
    status = False
    result = None
    if conts.IS_KMEANS_MODEL and conts.IS_CACHE_PRE_PROCESS_DATA and \
        os.path.exists(conts.CACHE_PRE_PRO_DATA_PATH) and \
        not conts.IS_APPLY_LSA_DIM_REDUCTION:
        print(f"Pre-process data already exists and will be obtained from the cache path: {conts.CACHE_PRE_PRO_DATA_PATH}")
        print(f"Run time of this operation is: {(time() - t0):0.2f} seconds" )
        status = True
        result = joblib.load(conts.CACHE_PRE_PRO_DATA_PATH)
    elif conts.IS_AUTOENCODER_MODEL and conts.IS_CACHE_PRE_PROCESS_DATA and \
        os.path.exists(conts.CACHE_PRE_PRO_DATA_PATH):
        print(f"Pre-process data already exists and will be obtained from the cache path: {conts.CACHE_PRE_PRO_DATA_PATH}")
        print(f"Run time of this operation is: {(time() - t0):0.2f} seconds" )
        status = True
        result = joblib.load(conts.CACHE_PRE_PRO_DATA_PATH)
    elif conts.IS_KMEANS_MODEL and conts.IS_CACHE_PRE_PROCESS_DATA and \
        os.path.exists(conts.CACHE_PRE_PRO_DATA_WITH_REDUCED_DIM_PATH) and \
        conts.IS_APPLY_LSA_DIM_REDUCTION:
        print(f"Pre-process data already exists and will be obtained from the cache path: {conts.CACHE_PRE_PRO_DATA_WITH_REDUCED_DIM_PATH}")
        print(f"Run time of this operation is: {(time() - t0):0.2f} seconds" )
        status = True
        result = joblib.load(conts.CACHE_PRE_PRO_DATA_WITH_REDUCED_DIM_PATH) 
    return status, result

def prepareData():
    is_results_already_cached, results = isDataExist()
    if is_results_already_cached:
        return results
    t0 = time()
    categories = getNewsCategories()
    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                shuffle=True, random_state=conts.K_MEANS_CONTS["SEED"])
    labels = dataset.target
    true_k = np.unique(labels).shape[0]
    print("There are {} documents".format(len(dataset.data)))
    print("There are {} categories".format(len(dataset.target_names)) )
    print()

    vectorizer = None
    if conts.IS_USE_HASH_VECTORIZER:
        vectorizer = computeHashVectorizer()
    else:
        vectorizer = computeTfidfVectorizer()
    train_dataset_X = dataset.data
    train_dataset_y = labels

    X = vectorizer.fit_transform(train_dataset_X)
    print("X.shape post-TFIDF is: {}".format(X.shape))
    print("Run time of this operation is: {0:0.2f} seconds".format((time() - t0)) )
    print("n_samples: {0}, n_features: {1}".format(X.shape[0], X.shape[1]))
    print()
    svd = None
    if conts.IS_APPLY_LSA_DIM_REDUCTION:
        X, svd = applyLsaProcessing(X, true_k)
        print("X.shape post-SVD is: {}".format(X.shape))
    results = {
            "X": X,
            "SVD": svd,
            "TRUE_K": true_k,
            "VECTORIZER": vectorizer,
            "TRAIN_DATA_X":train_dataset_X,
            "TRAIN_DATA_y":train_dataset_y,
            "CLUSTER_NAMES": dataset.target_names,
            "DOC_FILENAME": dataset.filenames,
        }
    if conts.IS_CACHE_PRE_PROCESS_DATA:
        if conts.IS_APPLY_LSA_DIM_REDUCTION: 
            joblib.dump(results, conts.CACHE_PRE_PRO_DATA_WITH_REDUCED_DIM_PATH)
        else:
            joblib.dump(results, conts.CACHE_PRE_PRO_DATA_PATH)
    return results

def testDataGeneration():
    results = prepareData()
    print("Shape of X is: {}".format(results["X"].shape))
    results2 = joblib.load(conts.CACHE_PRE_PRO_DATA_PATH)
    print("Shape of X is: {}".format(results2["X"].shape))


if __name__ == '__main__':
    print("Starting..")
    testDataGeneration()


