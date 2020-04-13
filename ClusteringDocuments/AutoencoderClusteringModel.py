# Checkpoint the weights for best model on validation accuracy
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy as np

import Constants as c
import KmeansClusteringModel as km

def createAutoEnconcoderModel():
	model = Sequential()
	# Encoder
	model.add(Dense(c.LAYERS['n_hidden_1'], input_dim=c.LAYERS['n_inputs'], activation='relu'))
	model.add(Dense(c.LAYERS['n_hidden_2'], activation='relu'))
	model.add(Dense(c.LAYERS['n_hidden_3'], activation='relu'))
	model.add(Dense(c.LAYERS['n_code'],  activation='relu', name='embedding'))
	# Decoder
	model.add(Dense(c.LAYERS['n_hidden_3'], activation='relu'))
	model.add(Dense(c.LAYERS['n_hidden_2'], activation='relu'))
	model.add(Dense(c.LAYERS['n_hidden_1'], activation='relu'))
	model.add(Dense(c.LAYERS['n_inputs'], activation='sigmoid'))
	model.summary()
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# checkpoint
	filepath="weights.best.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	return model, callbacks_list

def computeClustersUsingAutoencoder(X, true_k):
	print(f"X.shape: {X.shape}")
	model, callbacks_list = createAutoEnconcoderModel()
	model.fit(
		X, 
		X, 
		batch_size=c.AUTOENCODER_CONSTS.batch_size, 
		epochs=c.AUTOENCODER_CONSTS.epoch_size, 
		callbacks=callbacks_list)
	model.save(c.CACHE_AUTOENCODER_MODEL_PATH)
    # extract features
	feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
	features = feature_model.predict(X)
	print(f'Dimensionality reduction from {X.shape} to {features.shape} using Autoencoder model ')
	features = np.reshape(features, newshape=(features.shape[0], -1))
	km_model = km.computeClustersUsingKmeans(features, true_k)
	return km_model

