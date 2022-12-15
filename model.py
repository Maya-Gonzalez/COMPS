import glob
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

import keras


from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# specify path to dataset
data_dir = '/Users/mayagonzalez/Desktop/ML-to-detect-Alzheimer-s-/Dataset/'
# convert directory to list 
Mild_Demented = list(glob.glob(os.path.join(data_dir, 'Mild_Demented/*'))) 
Moderate_Demented = list(glob.glob(os.path.join(data_dir,'Moderate_Demented/*')))
Non_Demented = list(glob.glob(os.path.join(data_dir, 'Non_Demented/*')))
Very_Mild_Demented = list(glob.glob(os.path.join(data_dir,'Very_Mild_Demented/*')))

# -------- Print out images
def printImageTest():
  img = PIL.Image.open(str(Mild_Demented[0]))
  img.show(img)
  img = PIL.Image.open(str(Moderate_Demented[0]))
  img.show()
  img = PIL.Image.open(str(Non_Demented[0]))
  img.show(img)
  img = PIL.Image.open(str(Very_Mild_Demented[-1]))
  img.show()

# ------- Get details about classes
def printDetails():

  # identify num of images within each class of --> print len of each folder
  print('Len of Mild ds: ', len(Mild_Demented))
  print('Len of Moderate ds: ', len(Moderate_Demented))
  print('Len of Non ds: ', len(Non_Demented))
  print('Len of Very_Mild ds: ',len(Very_Mild_Demented),'\n')

  img = PIL.Image.open(str(Mild_Demented[0]))
  img_height = img.height 
  img_width = img.width 
  print('height : ', img_height)
  print('width : ', img_width)


# ------- Init model
# create training and validation sets
batch_size = 256
img = PIL.Image.open(str(Mild_Demented[0]))
img_height = img.height
img_width = img.width

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

# cache the data 
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------  Visualize dataset
def visualizeDS():
  # plot sample images from the train set
  plt.figure(figsize=(10, 10))
  for images, labels in train_ds.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")
  
# -------  Callback Method to Stop Training 
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = "min", patience = 3, restore_best_weights = True)

# -------  Instantiate Model
num_classes = len(class_names)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), # normalization layer
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'), # fully connected layer
  layers.Dense(num_classes)
])

# optimize model 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# -------  Visualize Model Layers (before  Running Model)
model.summary()

printImageTest()
printDetails()
visualizeDS()


# -------  Train Model
epochs= 100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks = [earlyStopping_callback]
)


# ------- Visualize training results 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# adjust epoch range if callbacks are utilized
epochs_range = earlyStopping_callback.stopped_epoch + 1 

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
