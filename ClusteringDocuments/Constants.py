import multiprocessing

IS_KMEANS_MODEL = False
IS_AUTOENCODER_MODEL = True
IS_USE_TSNE = True
NEW_LINE = "\n"
NUM_TOP_WORDS = 10
NUM_PLOT_SAMPLES = 500
NUM_TEST_SAMPLES = 100
LINE_BREAK = "---"*20
IS_APPLY_LSA_DIM_REDUCTION = False
IS_USE_HASH_VECTORIZER = False
IS_CACHE_PRE_PROCESS_DATA = True
CACHE_KMEANS_MODEL_PATH = "Cache_kmeans_model.pl"
CACHE_KMEANS_MODEL_WITH_REDUCED_DIM_PATH = "Cache_kmeans_model_with_reduced_dim.pl"
CACHE_AUTOENCODER_MODEL_PATH = "Cache_autoencoder_model.h"
CACHE_PRE_PRO_DATA_WITH_REDUCED_DIM_PATH = "Cache_pre_process_data_with_reduced_dim.pl"
CACHE_PRE_PRO_DATA_PATH = "Cache_pre_process_data.pl"
IS_INTERACTIVE_PLOT = False 

TF_IDF_CONSTS = {
   "NUM_FEATURES" : 10000,
   "MAX_DF" : 0.5,
   "MIN_DF" : 2,
   "STOP_WORDS": "english",
   "USE_IDF":True
}

HASH_CONSTS = {
   "NUM_FEATURES" : 10000,
   "ALTERNATE_SIGN" : False,
   "NORM" : None,
   "STOP_WORDS": "english",
   "USE_IDF":True,
}

K_MEANS_CONTS = {
    "MAX_ITER": 500,
    "INIT": "k-means++",
    "N_INIT": 1,
    "INIT_SIZE": 1000,
    "BATCH_SIZE": 1000,
    "SEED":100,
    "NUM_CORES": multiprocessing.cpu_count(),
    "IS_MINI_BATCH" : True,
    "IS_VERBOSE": False,
    "IS_USE_CACHE_MODEL": True,
}

class AUTOENCODER_CONSTS:
    n_layers = 4
    n_inputs = TF_IDF_CONSTS['NUM_FEATURES']
    n_hidden_1 = n_inputs // 2
    n_hidden_2 = n_hidden_1 // 2
    n_hidden_3 = n_hidden_2 // 2
    n_code = 2
    batch_size = 500
    epoch_size = 5

LAYERS = {
    'n_inputs': AUTOENCODER_CONSTS.n_inputs,
    'n_hidden_1':AUTOENCODER_CONSTS.n_hidden_1,
    'n_hidden_2':AUTOENCODER_CONSTS.n_hidden_2,
    'n_hidden_3':AUTOENCODER_CONSTS.n_hidden_3,
    'n_code':AUTOENCODER_CONSTS.n_code,
}


if __name__ == '__main__':
    print(f"AUTOENCODER.n_layers: {AUTOENCODER_CONSTS.n_layers}")