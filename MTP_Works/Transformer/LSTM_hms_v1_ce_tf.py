"""Success_hms-v3_KLD-TF.ipynb

# HMS - Harmful Brain Activity Classification

### IMPORTING THE NECESSARY LIBRARIES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import warnings
warnings.filterwarnings('ignore')

"""## LOADING THE DATASET"""

BASE_DIR = "/home/m1/23CS60R76/MTP_Works/HMS_Brain_Activity/Harmful_Brain_Activity/"

df = pd.read_csv(f"{BASE_DIR}train.csv")

votes_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
df_votes = df[votes_columns].melt(var_name='Brain Activity', value_name='Votes')
df['Brain Activity'] = df[votes_columns].idxmax(axis=1).apply(lambda x: x.replace('_vote', ''))

EEGid_label_list = df[["eeg_id", "Brain Activity", "eeg_label_offset_seconds"]].values.tolist()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix

def data_generator(X, y, batch_size=64):
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
            C1 = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fp1', 'Fp2','F3', 'F4', 'C3', 'C4', 'P3', 'P4']
            C2 = ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2']
            temp_arr1 = temp_df[C1].to_numpy()
            temp_arr1[np.isnan(temp_arr1)] = 0
            temp_arr2 = temp_df[C2].to_numpy()
            temp_arr2[np.isnan(temp_arr2)] = 0
            temp_arr = temp_arr1 - temp_arr2
        start = 200*int(offset)
        if(idx in index_list):
            X.append(temp_arr[start:start+10000])
            y.append(activity_mapping[label])
        prev_eegId = eeg_id
        idx+=1
    X = np.array(X)
    y = np.array(y)
    print("X:", X.shape)
    print("y:", y.shape)
    
    return data_generator(X, y)

# X_train, X_val, X_test, y_train, y_val, y_test = createDataset(EEGid_label_list[:50000])

# indices = list(range(len(EEGid_label_list)))
indices = list(range(50000))
train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)

print("\n--Training:")
train_dataset = createDataset(EEGid_label_list, train_indices)
# train_dataset = data_generator(X_train, y_train, batch_size=64)

print("\n--Validation:")
val_dataset = createDataset(EEGid_label_list, val_indices)
# val_dataset = data_generator(X_val, y_val, batch_size=64)

"""## MODEL CREATION"""

## deep learning model
def createModel(ip_shape, num_classes):
    saved_model_path = '/home/m1/23CS60R76/MTP_Works/BestSavedModels/LSTM_HMS_model_v1.keras'

    if(os.path.isfile(saved_model_path)):
        model = keras.models.load_model(saved_model_path)
        print("Model Loaded :", saved_model_path)
    else:
        model = keras.models.Sequential()
        model.add(layers.LSTM(128, input_shape=ip_shape, return_sequences=True))
        model.add(layers.LSTM(64))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))
        print("Model Created.")
    LEARNING_RATE = 1e-5
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=OPTIMIZER, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model

# Define parameters
input_shape = (10000,16)  # (sequence_length, num_features)
num_classes = 6
model = createModel(input_shape,num_classes)
model.summary()

# callbacks
VERBOSE=0
#lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, verbose=VERBOSE, min_le=1e-8)
es = EarlyStopping(monitor='val_loss', patience=50, verbose=VERBOSE, mode='auto', restore_best_weights=True)
checkpoint_filepath = '/home/m1/23CS60R76/MTP_Works/BestSavedModels/LSTM_HMS_model_v1.keras'
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
    epochs=200,
    batch_size=64,
    callbacks=callbacks
)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(hist.history['sparse_categorical_accuracy'])
plt.plot(hist.history['val_sparse_categorical_accuracy'])
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
plt.show()

# Saving the figure.
plt.savefig("lstm_training_history-0.jpg")