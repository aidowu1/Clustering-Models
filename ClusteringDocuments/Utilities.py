from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import cpu_count
from sklearn.manifold import MDS
from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
import numpy as np
from tabulate import tabulate

import Constants as co


def createClusterLabelMap(train_dataset_y, target_names):
  d = {}
  for v in target_names:
    for k in train_dataset_y:
      if k not in d:
        d[k] = v
        break
  return d
  
def computeDocDistance(X):
  dist = 1 - cosine_similarity(X)
  return dist

def computeMDS(dist):
  mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
  pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
  return pos

def computeTSNE(dist):
  tsne =  TSNE(n_jobs=cpu_count())
  pos = tsne.fit_transform(dist)  # shape (n_components, n_samples)
  return pos



def reportTopNClutersByWords( model,
                              svd, 
                              vectorizer, 
                              cluster_to_label_map
                      ):
  print("Top terms per cluster:")
  cluster_labels = cluster_to_label_map.keys()
  print(f"cluster_to_label_map: {cluster_to_label_map}")
  def getCentroids():
    if co.IS_APPLY_LSA_DIM_REDUCTION:
        original_space_centroids = svd.inverse_transform(model.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    return order_centroids
  order_centroids = getCentroids()
  terms = vectorizer.get_feature_names()
  for i in cluster_labels:
      news_category = cluster_to_label_map[i]
      print("Cluster [{} {}]:".format(i, news_category), end='')
      for ind in order_centroids[i, :co.NUM_TOP_WORDS]:
          print(' %s' % terms[ind], end='')
      print()
  
def reportTopNClutersByDocs(model,
                    train_dataset_y, 
                    train_docs_filename,
                    num_samples_per_cluster=10
                    ):
  print("\n\nClustering Results:")
  results = {
      "Docs": train_docs_filename,
      "predicted_clusters": model.labels_.tolist(),
      "actual_clusters": train_dataset_y
  }
  results_df = pd.DataFrame(results)
  true_k = np.unique(train_dataset_y).shape[0]
  for i in range(true_k):
    results_k_df = results_df[results_df.predicted_clusters == i].head(num_samples_per_cluster) 
    print(tabulate(results_k_df, headers='keys', tablefmt='psql'))
  print("\n")
  return results_df




if __name__ == "__main__":
    pass
