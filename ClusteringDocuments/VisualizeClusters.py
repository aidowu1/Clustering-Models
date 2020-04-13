import matplotlib.pyplot as plt
import matplotlib as mpl
import mpld3
import pandas as pd
import numpy as np

import Utilities
import Constants as c

np.random.seed(100)

def getClusterColorMap():
  cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a'}
  return cluster_colors


def plotClustersIn2D(df, cluster_names, cluster_colors):
  print("Plotting the clusters in 2D space..\n\n")
  # Shuffle the dataframe and only sample/plot a fraction of the data
  df2 = df.sample(n=c.NUM_PLOT_SAMPLES)
  df2.reset_index()

  #group by cluster
  groups = df2.groupby('label')
  
  # set up plot
  fig, ax = plt.subplots(figsize=(17, 9)) # set size
  ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

  #iterate through groups to layer the plot
  #note that I use the cluster_name and cluster_color dicts with the 'name' 
  #lookup to return the appropriate color/label
  for name, group in groups:
      ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
              label=cluster_names[name], color=cluster_colors[name], 
              mec='none')
      ax.set_aspect('auto')
      ax.tick_params(\
          axis= 'x',          # changes apply to the x-axis
          which='both',      # both major and minor ticks are affected
          bottom='off',      # ticks along the bottom edge are off
          top='off',         # ticks along the top edge are off
          labelbottom='off')
      ax.tick_params(\
          axis= 'y',         # changes apply to the y-axis
          which='both',      # both major and minor ticks are affected
          left='off',      # ticks along the bottom edge are off
          top='off',         # ticks along the top edge are off
          labelleft='off')
      
  ax.legend(numpoints=1)  #show legend with only 1 point

  #add label in x,y position with the label as the news category
  for i in range(len(df2)):
      ax.text(df2.iloc[i]['x'], df2.iloc[i]['y'], df2.iloc[i]['title'], size=8)
  #plt.show() #show the plot
  return fig

def reduceDimnesions2D_V2(X):  
  dist = Utilities.computeDocDistance(X)
  pos = dist[:]
  print(f"dist.shape: {dist.shape}")
  if c.IS_KMEANS_MODEL:    
    if c.IS_USE_TSNE:
      pos = Utilities.computeTSNE(dist)
    else:
      pos = Utilities.computeMDS(dist)
  return pos

def reduceDimnesions2D(X):  
  dist = Utilities.computeDocDistance(X)
  print(f"dist.shape: {dist.shape}")    
  if c.IS_USE_TSNE:
    pos = Utilities.computeTSNE(dist)
  else:
    pos = Utilities.computeMDS(dist)
  return pos


def visualizeClusteringResults(data_results, results_df, cluster_to_label_map):
  X = data_results["X"]
  pos = reduceDimnesions2D(X)
  xs, ys = pos[:, 0], pos[:, 1]
  clusters = results_df.predicted_clusters.tolist()
  doc_ids = range(len(results_df.Docs.tolist()))
  #create data frame that has the result of the MDS plus the cluster numbers and doc_ids
  df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=doc_ids))
  cluster_colors = getClusterColorMap()
  fig = plotClustersIn2D(df, cluster_to_label_map, cluster_colors)
  return fig


if __name__ == "__main__":
    pass
