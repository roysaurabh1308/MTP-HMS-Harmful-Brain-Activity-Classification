# -*- coding: utf-8 -*-
"""Hilbert_hms-v3_CE.ipynb

# HMS - Harmful Brain Activity Classification

### IMPORTING THE NECESSARY LIBRARIES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')

"""## LOADING THE DATASET"""

BASE_DIR = "/home/m1/23CS60R76/MTP_Works/HMS_Brain_Activity/Harmful_Brain_Activity/"

df = pd.read_csv(f"{BASE_DIR}train.csv")
df.head()

votes_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
df_votes = df[votes_columns].melt(var_name='Brain Activity', value_name='Votes')
df['Brain Activity'] = df[votes_columns].idxmax(axis=1).apply(lambda x: x.replace('_vote', ''))

EEGid_label_list = df[["eeg_id", "Brain Activity", "eeg_label_offset_seconds"]].values.tolist()

"""## MODEL CREATION"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from scipy.signal import hilbert
from keras.utils import to_categorical

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # Only allocate as much GPU memory as needed
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix

def data_generator(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def createDataset(EEGid_label_list, index_list):
    X = []
    y = []
    prev_eegId = ""; idx=0
    brain_activities = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
    activity_mapping = {activity.lower(): idx for idx, activity in enumerate(brain_activities)}
    for x in EEGid_label_list:
        eeg_id, label, offset = x
        if(eeg_id!=prev_eegId):
            temp_df = pd.read_parquet(f'{BASE_DIR}train_eegs/{eeg_id}.parquet')
            C = ['Fp1', 'T3', 'P4']
            temp_arr = temp_df[C].to_numpy().T
            temp_arr[np.isnan(temp_arr)] = 0
        start = 200*int(offset)
        if(idx in index_list):
            X.append(temp_arr[:, start:start+10000])
            y.append(activity_mapping[label])
        # print(eeg_id, activity_mapping[label], temp_arr.shape)
        prev_eegId = eeg_id
        idx += 1
    X = np.array(X)
    y = np.array(y)
    print("X:", X.shape)
    print("y:", y.shape)
    # CNN will be used. Channel dimension is added.
    X = X[:, :, :, np.newaxis]
    return data_generator(X, to_categorical(y))

# # Custom Hilbert Transform Layer (previously defined)
# class HilbertLayer(layers.Layer):
#     def call(self, inputs):
#         # Implementation of Hilbert Transform (simplified)
#         inputs_complex = tf.cast(inputs, tf.complex64)
#         fft = tf.signal.fft(inputs_complex)
#         # Hilbert mask application assumed
#         return tf.abs(tf.signal.ifft(fft))  # Simplified example

#     def compute_output_shape(self, input_shape):
#         return input_shape

class HilbertLayer(layers.Layer):
    def call(self, inputs):
        # Assuming input is real, convert to complex for FFT
        inputs_complex = tf.cast(inputs, tf.complex64)
        # Compute FFT
        fft = tf.signal.fft(inputs_complex)
        # Construct Hilbert mask to zero out negative frequencies
        hilbert_mask = tf.concat([tf.ones_like(fft[..., :1]), 2*tf.ones_like(fft[..., 1:fft.shape[-1]//2]), tf.zeros_like(fft[..., fft.shape[-1]//2:])], axis=-1)
        # Apply Hilbert mask
        filtered_fft = fft * hilbert_mask
        # Compute IFFT
        ifft = tf.signal.ifft(filtered_fft)
        # Return magnitude (absolute value)
        return tf.abs(ifft)

def createModel(ip_shape):
    saved_model_path = '/home/m1/23CS60R76/MTP_Works/BestSavedModels/H_HMS_model5_v3.keras'

    if(os.path.isfile(saved_model_path)):
        model = keras.models.load_model(saved_model_path, custom_objects={'HilbertLayer': HilbertLayer})
        print("Model Loaded :", saved_model_path)
    else:
        model = keras.models.Sequential()

        model.add(layers.Conv2D(filters=64, kernel_size=(2, 4), padding='same', activation='relu', input_shape=ip_shape))
        model.add(layers.Conv2D(filters=64, kernel_size=(2, 4), strides=(1, 2), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((1, 2)))

        model.add(layers.Conv2D(filters=128, kernel_size=(2, 4), padding='same', activation='relu'))
        model.add(layers.Conv2D(filters=128, kernel_size=(2, 4), strides=(1, 2), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(filters=256, kernel_size=(4, 4), padding='same', activation='relu'))
        model.add(layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(1, 2), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((1, 2)))

        # Inserting Hilbert Layer after all feature extraction
        model.add(HilbertLayer())

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(6, activation='softmax'))
        print("Model Created.")

    LEARNING_RATE = 1e-5
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# EEGid_label_list = EEGid_label_list[:70000]

indices = list(range(len(EEGid_label_list)))
train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)

print("\n--Training:")
train_dataset = createDataset(EEGid_label_list, train_indices)
# train_dataset = data_generator(X_train, y_train, batch_size=64)

print("\n--Validation:")
val_dataset = createDataset(EEGid_label_list, val_indices)
# val_dataset = data_generator(X_val, y_val, batch_size=64)

model = createModel((3,10000,1))
model.summary()

# callbacks
VERBOSE=0
#lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, verbose=VERBOSE, min_le=1e-8)
es = EarlyStopping(monitor='val_loss', patience=50, verbose=VERBOSE, mode='auto', restore_best_weights=True)
checkpoint_filepath = '/home/m1/23CS60R76/MTP_Works/BestSavedModels/H_HMS_arch3_model5_v3.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
callbacks = [es, model_checkpoint_callback]
hist = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
# plt.show()

# Saving the figure.
plt.savefig("hilbert_training_history_A3.jpg")

# beta_model = keras.models.load_model('/home/m1/23CS60R76/MTP_Works/BestSavedModels/H_HMS_model5_v3.keras')
# y_pred = beta_model.predict(X_test)

# predicted_categories = np.argmax(y_pred, axis = 1)
# actual_categories = np.argmax(y_test_one_hot, axis = 1)
# report = classification_report(actual_categories, predicted_categories)
# print("Classification Report:\n", report)
# conf_matrix = confusion_matrix(actual_categories, predicted_categories)
# print("Confusion Matrix:\n", conf_matrix)

# import scikitplot as skplt
# skplt.metrics.plot_confusion_matrix(actual_categories, predicted_categories, normalize=True)
# plt.savefig("hilbert_conf_mat.jpg")