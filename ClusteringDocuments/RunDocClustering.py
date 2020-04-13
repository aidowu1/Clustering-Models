from sklearn import metrics

import Constants as co
import DataGenerator as dg
import KmeansClusteringModel as km
import AutoencoderClusteringModel as am
import VisualizeClusters as vc
import Utilities as ut


def predictKemeansClusters(model, data_results):
  train_dataset_X = data_results["TRAIN_DATA_X"]
  y_test_pred = model.labels_
  y_test_expected = data_results["TRAIN_DATA_y"]
  accuracy = metrics.accuracy_score(y_test_expected, y_test_pred)
  homo_score = metrics.homogeneity_score(y_test_expected, y_test_pred)
  data_results["ACCURACY"] = accuracy
  data_results["HOMOGENEITY"] = homo_score
  data_results["TEST_DATA_yhat"] = y_test_pred 
  return data_results

def run():
    data_results = dg.prepareData()
    X = data_results["X"]
    print(f"data_results[X].shape: {X.shape}")
    cluster_to_label_map = ut.createClusterLabelMap(data_results["TRAIN_DATA_y"], 
                                                data_results["CLUSTER_NAMES"])

    model = None
    if co.IS_KMEANS_MODEL:
      model = km.computeClustersUsingKmeans(X, data_results["TRUE_K"])
    else:
      model = am.computeClustersUsingAutoencoder(X, data_results["TRUE_K"])
    print(co.LINE_BREAK)
    ut.reportTopNClutersByWords(model, 
                            data_results["SVD"],
                            data_results["VECTORIZER"], 
                            cluster_to_label_map)
    print(co.LINE_BREAK)
    data_results = predictKemeansClusters(model, data_results)
    kmeans_accuracy = data_results["ACCURACY"]
    kmeans_homogeniety = data_results["HOMOGENEITY"]
    print(f"K-means prediction accuracy is: {kmeans_accuracy:0.2f}")
    print(f"K-means prediction homogeniety score is: {kmeans_homogeniety:0.2f}")
    print(co.LINE_BREAK)
    results_df = ut.reportTopNClutersByDocs(model,
                        data_results["TRAIN_DATA_y"], 
                        data_results["DOC_FILENAME"])
    fig = vc.visualizeClusteringResults(data_results, results_df, cluster_to_label_map)
    return fig
    


if __name__ == "__main__":
    run()
