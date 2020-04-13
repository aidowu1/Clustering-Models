from sklearn.cluster import KMeans, MiniBatchKMeans
import os
from time import time

import joblib

import Constants as c

def computeClustersUsingKmeans(X, true_k):
  t0 = time()
  model_path = None
  if c.IS_APPLY_LSA_DIM_REDUCTION:
    model_path = c.CACHE_KMEANS_MODEL_WITH_REDUCED_DIM_PATH
  else:
    model_path = c.CACHE_KMEANS_MODEL_PATH
  print("Starting to compute the K-means clustering algorithm..")
  if c.K_MEANS_CONTS["IS_USE_CACHE_MODEL"] and os.path.exists(model_path):
    print(f"Model already exists and will be obtained from the cache path: {model_path}")
    print(f"Run time of this operation is: {(time() - t0):0.2f} seconds" )
    model = joblib.load(model_path)
    return model
  if c.K_MEANS_CONTS["IS_MINI_BATCH"]:
    model = MiniBatchKMeans(n_clusters=true_k, init=c.K_MEANS_CONTS["INIT"], 
                            n_init=c.K_MEANS_CONTS["N_INIT"],
                            init_size=c.K_MEANS_CONTS["INIT_SIZE"], 
                            batch_size=c.K_MEANS_CONTS["BATCH_SIZE"], 
                            verbose=c.K_MEANS_CONTS["IS_VERBOSE"])    
  else:
    model = KMeans(n_clusters=true_k)

  model.fit(X)
  print(f"Run time of this operation is: {(time() - t0):0.2f} seconds" )
  print(f"Saving model to cache path: {model_path}")
  joblib.dump(model,  model_path)
  return model


if __name__ == "__main__":
    pass