# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:19:40 2021

@author: Sharanya
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:08:35 2021

@author: Sharanya
"""

#VGG16 model
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras import layers, models, optimizers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

 

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
 
# re-size all the images to this
IMAGE_SIZE = [224, 224, 3]

 
#training_path
training_path = 'H:/newdatasetMulticlass/multi/multi/training_set/'

 

#validation_path 
validation_path = 'H:/newdatasetMulticlass/multi/multi/validation_set/'
                

 



 

train_datagen = ImageDataGenerator(#validation_split = 0.15,
                                    rescale = 1./255,
                                   shear_range = 0.2, ### Choose a shear_range
                                  # zoom_range = 0.2, ### Choose a zoom range
                                  # horizontal_flip = True
                                  ) ### Assign the Horizontal flip 
training_set = train_datagen.flow_from_directory('H:/newdatasetMulticlass/multi/multi/split/train/',
                                                 subset = 'training',
                                                 target_size = (224, 224),
                                                 batch_size = 32, ### Choose the batch size
                                                 class_mode = 'categorical')

 

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_set = validation_datagen.flow_from_directory('H:/newdatasetMulticlass/multi/multi/split/val/',
                                                 #subset = 'validation',
                                                 target_size = (224, 224),
                                                 batch_size = 32, ### Choose the batch size
                                                 class_mode = 'categorical')

 

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

  

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(6, activation='softmax')) #softmax
model.summary()

 

conv_base.trainable = False

 

model.compile(loss='categorical_crossentropy',
          optimizer='Adam',
          metrics=['acc'])

 

 

history = model.fit_generator(
    training_set,
    steps_per_epoch=20, # batches in the generator are 50, so it takes 320 batches to get to 16000 images
    epochs=25,
    validation_data=validation_set,
    validation_steps=len(validation_set))

 
print(history.history['acc'])
print(history.history['loss'])
print("Training Done")
model.save("foodbinary.h5")

 

new_model = tf.keras.models.load_model('foodbinary.h5')

 

new_model.summary()

 

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Latest Plot Lossvalue")
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
plt.plot(history.history['acc'], label='train acc')
plt.plot(history.history['val_acc'], label='val acc')
plt.title("Latest Plot accuracy")
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

 

collection = "H:/newdatasetMulticlass/test1/"
test_length = len(os.listdir(collection))

 

path = "H:/newdatasetMulticlass/test1/"
#path_len=-len(os.listdir(path))

#vegitables_details="" 


for i in range(0,test_length+1):
    #var = path + "/test_image_" + 
    test_image = image.load_img(path +"/" + str(i) + ".jpg", target_size = (224, 224)) ### TRY Your Own Image!
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = new_model.predict(test_image)
    training_set.class_indices
    
                                                                  
                                                                                   
    
    
    if result[0][0] == 1:
      prediction = 'tomato'
      vegetables_details="tomato"
    elif result[0][1] == 1:
      prediction = 'potato'
      vegetables_details="potato"
    elif result[0][2] == 1:
        prediction = 'carrot'
        vegetables_details="carrot"
    elif result[0][3] == 1:
        prediction = 'radish'
        vegetables_details="radish"
    elif result[0][4] == 1:
        prediction = 'cucumber'
        vegetables_details="cucumber"
    elif result[0][5] == 1:
        prediction = 'beetroot'
        vegetables_details="beetroot"

 

       print(str(i) + " " + prediction+'\n'+vegetables_details)