# Convolutional Neural Network

# Installing Keras
# conda install -c conda-forge keras
'''
The datasets are from Kaggle
The files' names are already named in format.
10000 images in total, 8000 for training and 2000 for testing, dogs/ cats are 4000 and 1000
'''

# We don't need data preproccessing since it's done automatically

#%% Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
'''
There are two classes to initialize neural networks:
    1. sequential NN (here, CNN is a sequence of layers)
    2. as a gragh
'''
from keras.layers import Convolution2D
'''to create convolutional layer, 2D to deal with images'''
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
'''to create fully connected layer'''

# Initialising the CNN
classifier = Sequential()
'''# Arguments
layers: list of layers to add to the model.'''

#%% Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
'''
Convolution2D(number_of_output_filters, filter_rows, filter_columns)
a filter is a feature detector, we choose 3*3 filters
input_shape = (3, 256, 256) is default, w.r.t 3 channels and 256*256 for 2d image
while here we use tensorflow, it's a backend order, which is different with the default (endback order)
and also, with CPU, to get faster result, we don't want set too large values here.
rectifier to have non-linearity
'''

#%% Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
'''argument: pool_size(vertical, horizontal) and strides'''

#%% Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
'''Add a fully-connected layer will also improve the neural network.
We don't need to include input_shape, since we already have previous layer set.
'''

#%% Step 3 - Flattening
classifier.add(Flatten())

#%% Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
'''This is the fully connected layer
128 is a good number from past experience'''
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
'''this is the output layer'''

#%% Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
'''binary_crossentropy: binary outcome and cross-entropy method'''

#%% Part 2 - Fitting the CNN to the images
# Check out Keras Documentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
