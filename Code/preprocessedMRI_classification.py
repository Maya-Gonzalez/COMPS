import glob
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

import keras
from nibabel.testing import data_path
import nibabel as nib

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# load dataset
data_dir = '/Users/mayagonzalez/Desktop/ML-to-detect-Alzheimer-s-/Dataset/ADNI'
metadata = '/Users/mayagonzalez/Desktop/ML-to-detect-Alzheimer-s-/Dataset/ADNI1_Baseline_3T_11_10_2022.csv'
#try opening the image
for root, dirs, files in os.walk(data_dir):
  print('dirpath',root)
  print('dirnames',dirs)
  print('filenames',files)
  for file in files:
    print('here')
    if(file.endswith('.nii')):
      print('here2')
      file_path = os.path.join(root, file)
      print('FILE PATH', file_path)
      img = nib.load(file)
      print(img.shape)
      # put into new folder
  if root is not data_dir:
    continue
print(s)
example_filename = os.path.join(root, 'example4d.nii.gz')
img = nib.load(example_filename)
print(img.shape)
# for image name in csv, get name and classification
# create folders for each classification
# for image in data_dir, assign to appropriate folder based on classification
# display image
Mild_Demented = list(glob.glob(os.path.join(data_dir, 'Mild_Demented/*'))) # convert to list for sake of accessing first image to print out
Moderate_Demented = list(glob.glob(os.path.join(data_dir,'Moderate_Demented/*')))
Non_Demented = list(glob.glob(os.path.join(data_dir, 'Non_Demented/*')))
Very_Mild_Demented = list(glob.glob(os.path.join(data_dir,'Very_Mild_Demented/*')))

# -------- Print out images
img = PIL.Image.open(str(Mild_Demented[0]))
# img.show(img)
# img = PIL.Image.open(str(Moderate_Demented[0]))
# img.show()
# img = PIL.Image.open(str(Non_Demented[0]))
# img.show(img)
# img = PIL.Image.open(str(Very_Mild_Demented[-1]))
# img.show()


# ------- Get details about classes

# identify num of images within each class of --> print len of each folder
print('Len of Mild ds: ', len(Mild_Demented))
print('Len of Moderate ds: ', len(Moderate_Demented))
print('Len of Non ds: ', len(Non_Demented))
print('Len of Very_Mild ds: ',len(Very_Mild_Demented),'\n')

# ------- Init model

# create training and validation sets
batch_size = 256
img_height = img.height 
img_width = img.width 
print('height : ', img_height)
print('width : ', img_width)
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
print(class_names)


# cache the data (store it in local memory)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------  Visualize dataset

# plot sample images from the train set
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# print shapes of images and labels used in each partition of the train set
for image_batch, labels_batch in train_ds:
  print('image batch size: ',image_batch.shape)
  print('labels batch size:', labels_batch.shape)
  break
  
# -------  Callback Method to Stop Training 
accuracy_threshold = 0.24
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

# optimize model between different epochs
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# -------  Visualize Model Layers (b4 Running Model)
model.summary()

# -------  Train Model
epochs= 2
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

epochs_range = range(epochs)
# FIXME: update epochs_range (for plotting) when training terminates before set epochs
#epochs_range = earlyStopping_callback.stopped_epoch + 1 # FIXME: manually add one to mathc len of accuracy list?
print('stopping at: ', accuracy_threshold)
print('epochs_range: ', epochs_range )
print('accuracy: ', acc)
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
