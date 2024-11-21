import tensorflow as tf
from tensorflow import keras
# from keras import layers
# from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

saved_model_path = '/home/m1/23CS60R76/MTP_Works/BestSavedModels/HMS_model7_v1.keras'

model = keras.models.load_model(saved_model_path)
model.summary()
model.save('/home/m1/23CS60R76/MTP_Works/BestSavedModels/HMS_model7_v2.h5')