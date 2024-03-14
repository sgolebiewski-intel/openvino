From Training to Deployment with TensorFlow and OpenVINO™
=========================================================

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `TensorFlow Image Classification
   Training <#tensorflow-image-classification-training>`__
-  `Import TensorFlow and Other
   Libraries <#import-tensorflow-and-other-libraries>`__
-  `Download and Explore the
   Dataset <#download-and-explore-the-dataset>`__
-  `Load Using keras.preprocessing <#load-using-keras-preprocessing>`__
-  `Create a Dataset <#create-a-dataset>`__
-  `Visualize the Data <#visualize-the-data>`__
-  `Configure the Dataset for
   Performance <#configure-the-dataset-for-performance>`__
-  `Standardize the Data <#standardize-the-data>`__
-  `Create the Model <#create-the-model>`__
-  `Compile the Model <#compile-the-model>`__
-  `Model Summary <#model-summary>`__
-  `Train the Model <#train-the-model>`__
-  `Visualize Training Results <#visualize-training-results>`__
-  `Overfitting <#overfitting>`__
-  `Data Augmentation <#data-augmentation>`__
-  `Dropout <#dropout>`__
-  `Compile and Train the Model <#compile-and-train-the-model>`__
-  `Visualize Training Results <#visualize-training-results>`__
-  `Predict on New Data <#predict-on-new-data>`__
-  `Save the TensorFlow Model <#save-the-tensorflow-model>`__
-  `Convert the TensorFlow model with OpenVINO Model Conversion
   API <#convert-the-tensorflow-model-with-openvino-model-conversion-api>`__
-  `Preprocessing Image Function <#preprocessing-image-function>`__
-  `OpenVINO Runtime Setup <#openvino-runtime-setup>`__

   -  `Select inference device <#select-inference-device>`__

-  `Run the Inference Step <#run-the-inference-step>`__
-  `The Next Steps <#the-next-steps>`__

.. code:: ipython3

    # @title Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    
    # Copyright 2018 The TensorFlow Authors
    #
    # Modified for OpenVINO Notebooks

This tutorial demonstrates how to train, convert, and deploy an image
classification model with TensorFlow and OpenVINO. This particular
notebook shows the process where we perform the inference step on the
freshly trained model that is converted to OpenVINO IR with model
conversion API. For faster inference speed on the model created in this
notebook, check out the `Post-Training Quantization with TensorFlow
Classification Model <301-tensorflow-training-openvino-nncf-with-output.html>`__
notebook.

This training code comprises the official `TensorFlow Image
Classification
Tutorial <https://www.tensorflow.org/tutorials/images/classification>`__
in its entirety.

The ``flower_ir.bin`` and ``flower_ir.xml`` (pre-trained models) can be
obtained by executing the code with ‘Runtime->Run All’ or the
``Ctrl+F9`` command.

.. code:: ipython3

    %pip install -q "openvino>=2023.1.0"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


TensorFlow Image Classification Training
----------------------------------------

`back to top ⬆️ <#table-of-contents>`__

The first part of the tutorial shows how to classify images of flowers
(based on the TensorFlow’s official tutorial). It creates an image
classifier using a ``keras.Sequential`` model, and loads data using
``preprocessing.image_dataset_from_directory``. You will gain practical
experience with the following concepts:

-  Efficiently loading a dataset off disk.
-  Identifying overfitting and applying techniques to mitigate it,
   including data augmentation and Dropout.

This tutorial follows a basic machine learning workflow:

1. Examine and understand data
2. Build an input pipeline
3. Build the model
4. Train the model
5. Test the model

Import TensorFlow and Other Libraries
-------------------------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    import os
    import sys
    from pathlib import Path
    
    import PIL
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    import openvino as ov
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2024-03-14 01:04:59.279897: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-14 01:04:59.314679: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-14 01:04:59.825615: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Download and Explore the Dataset
--------------------------------

`back to top ⬆️ <#table-of-contents>`__

This tutorial uses a dataset of about 3,700 photos of flowers. The
dataset contains 5 sub-directories, one per class:

::

   flower_photo/
     daisy/
     dandelion/
     roses/
     sunflowers/
     tulips/

.. code:: ipython3

    import pathlib
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

After downloading, you should now have a copy of the dataset available.
There are 3,670 total images:

.. code:: ipython3

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)


.. parsed-literal::

    3670


Here are some roses:

.. code:: ipython3

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_14_0.png



.. code:: ipython3

    PIL.Image.open(str(roses[1]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_15_0.png



And some tulips:

.. code:: ipython3

    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_17_0.png



.. code:: ipython3

    PIL.Image.open(str(tulips[1]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_18_0.png



Load Using keras.preprocessing
------------------------------

`back to top ⬆️ <#table-of-contents>`__

Let’s load these images off disk using the helpful
`image_dataset_from_directory <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory>`__
utility. This will take you from a directory of images on disk to a
``tf.data.Dataset`` in just a couple lines of code. If you like, you can
also write your own data loading code from scratch by visiting the `load
images <https://www.tensorflow.org/tutorials/load_data/images>`__
tutorial.

Create a Dataset
----------------

`back to top ⬆️ <#table-of-contents>`__

Define some parameters for the loader:

.. code:: ipython3

    batch_size = 32
    img_height = 180
    img_width = 180

It’s good practice to use a validation split when developing your model.
Let’s use 80% of the images for training, and 20% for validation.

.. code:: ipython3

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 2936 files for training.


.. parsed-literal::

    2024-03-14 01:05:02.881268: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-14 01:05:02.881298: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-03-14 01:05:02.881303: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-03-14 01:05:02.881429: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-03-14 01:05:02.881452: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-03-14 01:05:02.881456: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. code:: ipython3

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.


You can find the class names in the ``class_names`` attribute on these
datasets. These correspond to the directory names in alphabetical order.

.. code:: ipython3

    class_names = train_ds.class_names
    print(class_names)


.. parsed-literal::

    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


Visualize the Data
------------------

`back to top ⬆️ <#table-of-contents>`__

Here are the first 9 images from the training dataset.

.. code:: ipython3

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


.. parsed-literal::

    2024-03-14 01:05:03.206226: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:05:03.206656: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_29_1.png


You will train a model using these datasets by passing them to
``model.fit`` in a moment. If you like, you can also manually iterate
over the dataset and retrieve batches of images:

.. code:: ipython3

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    2024-03-14 01:05:04.036011: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-14 01:05:04.036303: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


The ``image_batch`` is a tensor of the shape ``(32, 180, 180, 3)``. This
is a batch of 32 images of shape ``180x180x3`` (the last dimension
refers to color channels RGB). The ``label_batch`` is a tensor of the
shape ``(32,)``, these are corresponding labels to the 32 images.

You can call ``.numpy()`` on the ``image_batch`` and ``labels_batch``
tensors to convert them to a ``numpy.ndarray``.

Configure the Dataset for Performance
-------------------------------------

`back to top ⬆️ <#table-of-contents>`__

Let’s make sure to use buffered prefetching so you can yield data from
disk without having I/O become blocking. These are two important methods
you should use when loading data.

``Dataset.cache()`` keeps the images in memory after they’re loaded off
disk during the first epoch. This will ensure the dataset does not
become a bottleneck while training your model. If your dataset is too
large to fit into memory, you can also use this method to create a
performant on-disk cache.

``Dataset.prefetch()`` overlaps data preprocessing and model execution
while training.

Interested readers can learn more about both methods, as well as how to
cache data to disk in the `data performance
guide <https://www.tensorflow.org/guide/data_performance#prefetching>`__.

.. code:: ipython3

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

Standardize the Data
--------------------

`back to top ⬆️ <#table-of-contents>`__

The RGB channel values are in the ``[0, 255]`` range. This is not ideal
for a neural network; in general you should seek to make your input
values small. Here, you will standardize values to be in the ``[0, 1]``
range by using a Rescaling layer.

.. code:: ipython3

    normalization_layer = layers.Rescaling(1./255)

Note: The Keras Preprocessing utilities and layers introduced in this
section are currently experimental and may change.

There are two ways to use this layer. You can apply it to the dataset by
calling map:

.. code:: ipython3

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image)) 


.. parsed-literal::

    2024-03-14 01:05:04.223853: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:05:04.224428: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    0.0161317 1.0


Or, you can include the layer inside your model definition, which can
simplify deployment. Let’s use the second approach here.

Note: you previously resized images using the ``image_size`` argument of
``image_dataset_from_directory``. If you want to include the resizing
logic in your model as well, you can use the
`Resizing <https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing>`__
layer.

Create the Model
----------------

`back to top ⬆️ <#table-of-contents>`__

The model consists of three convolution blocks with a max pool layer in
each of them. There’s a fully connected layer with 128 units on top of
it that is activated by a ``relu`` activation function. This model has
not been tuned for high accuracy, the goal of this tutorial is to show a
standard approach.

.. code:: ipython3

    num_classes = 5
    
    model = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

Compile the Model
-----------------

`back to top ⬆️ <#table-of-contents>`__

For this tutorial, choose the ``optimizers.Adam`` optimizer and
``losses.SparseCategoricalCrossentropy`` loss function. To view training
and validation accuracy for each training epoch, pass the ``metrics``
argument.

.. code:: ipython3

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

Model Summary
-------------

`back to top ⬆️ <#table-of-contents>`__

View all the layers of the network using the model’s ``summary`` method.

   **NOTE:** This section is commented out for performance reasons.
   Please feel free to uncomment these to compare the results.

.. code:: ipython3

    # model.summary()

Train the Model
---------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    # epochs=10
    # history = model.fit(
    #   train_ds,
    #   validation_data=val_ds,
    #   epochs=epochs
    # )

Visualize Training Results
--------------------------

`back to top ⬆️ <#table-of-contents>`__

Create plots of loss and accuracy on the training and validation sets.

.. code:: ipython3

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    
    # epochs_range = range(epochs)
    
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

As you can see from the plots, training accuracy and validation accuracy
are off by large margin and the model has achieved only around 60%
accuracy on the validation set.

Let’s look at what went wrong and try to increase the overall
performance of the model.

Overfitting
-----------

`back to top ⬆️ <#table-of-contents>`__

In the plots above, the training accuracy is increasing linearly over
time, whereas validation accuracy stalls around 60% in the training
process. Also, the difference in accuracy between training and
validation accuracy is noticeable — a sign of
`overfitting <https://www.tensorflow.org/tutorials/keras/overfit_and_underfit>`__.

When there are a small number of training examples, the model sometimes
learns from noises or unwanted details from training examples—to an
extent that it negatively impacts the performance of the model on new
examples. This phenomenon is known as overfitting. It means that the
model will have a difficult time generalizing on a new dataset.

There are multiple ways to fight overfitting in the training process. In
this tutorial, you’ll use *data augmentation* and add *Dropout* to your
model.

Data Augmentation
-----------------

`back to top ⬆️ <#table-of-contents>`__

Overfitting generally occurs when there are a small number of training
examples. `Data
augmentation <https://www.tensorflow.org/tutorials/images/data_augmentation>`__
takes the approach of generating additional training data from your
existing examples by augmenting them using random transformations that
yield believable-looking images. This helps expose the model to more
aspects of the data and generalize better.

You will implement data augmentation using the layers from
``tf.keras.layers.experimental.preprocessing``. These can be included
inside your model like other layers, and run on the GPU.

.. code:: ipython3

    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )

Let’s visualize what a few augmented examples look like by applying data
augmentation to the same image several times:

.. code:: ipython3

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")


.. parsed-literal::

    2024-03-14 01:05:05.126486: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:05:05.127286: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_57_1.png


You will use data augmentation to train a model in a moment.

Dropout
-------

`back to top ⬆️ <#table-of-contents>`__

Another technique to reduce overfitting is to introduce
`Dropout <https://developers.google.com/machine-learning/glossary#dropout_regularization>`__
to the network, a form of *regularization*.

When you apply Dropout to a layer it randomly drops out (by setting the
activation to zero) a number of output units from the layer during the
training process. Dropout takes a fractional number as its input value,
in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20%
or 40% of the output units randomly from the applied layer.

Let’s create a new neural network using ``layers.Dropout``, then train
it using augmented images.

.. code:: ipython3

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

Compile and Train the Model
---------------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

.. code:: ipython3

    model.summary()


.. parsed-literal::

    Model: "sequential_2"


.. parsed-literal::

    _________________________________________________________________


.. parsed-literal::

     Layer (type)                Output Shape              Param #   


.. parsed-literal::

    =================================================================


.. parsed-literal::

     sequential_1 (Sequential)   (None, 180, 180, 3)       0         




                                                                     


.. parsed-literal::

     rescaling_2 (Rescaling)     (None, 180, 180, 3)       0         




                                                                     


.. parsed-literal::

     conv2d_3 (Conv2D)           (None, 180, 180, 16)      448       




                                                                     


.. parsed-literal::

     max_pooling2d_3 (MaxPooling  (None, 90, 90, 16)       0         


.. parsed-literal::

     2D)                                                             




                                                                     


.. parsed-literal::

     conv2d_4 (Conv2D)           (None, 90, 90, 32)        4640      




                                                                     


.. parsed-literal::

     max_pooling2d_4 (MaxPooling  (None, 45, 45, 32)       0         


.. parsed-literal::

     2D)                                                             




                                                                     


.. parsed-literal::

     conv2d_5 (Conv2D)           (None, 45, 45, 64)        18496     




                                                                     


.. parsed-literal::

     max_pooling2d_5 (MaxPooling  (None, 22, 22, 64)       0         


.. parsed-literal::

     2D)                                                             




                                                                     


.. parsed-literal::

     dropout (Dropout)           (None, 22, 22, 64)        0         




                                                                     


.. parsed-literal::

     flatten_1 (Flatten)         (None, 30976)             0         




                                                                     


.. parsed-literal::

     dense_2 (Dense)             (None, 128)               3965056   




                                                                     


.. parsed-literal::

     outputs (Dense)             (None, 5)                 645       




                                                                     


.. parsed-literal::

    =================================================================


.. parsed-literal::

    Total params: 3,989,285


.. parsed-literal::

    Trainable params: 3,989,285


.. parsed-literal::

    Non-trainable params: 0


.. parsed-literal::

    _________________________________________________________________


.. code:: ipython3

    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


.. parsed-literal::

    Epoch 1/15


.. parsed-literal::

    2024-03-14 01:05:06.178653: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:05:06.179041: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
 1/92 [..............................] - ETA: 1:29 - loss: 1.5955 - accuracy: 0.2188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 2.4613 - accuracy: 0.1875  

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 2.4754 - accuracy: 0.2083

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 2.2985 - accuracy: 0.2344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 2.2089 - accuracy: 0.2438

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 2.1044 - accuracy: 0.2552

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 2.0336 - accuracy: 0.2545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.9780 - accuracy: 0.2578

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.9297 - accuracy: 0.2604

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.8929 - accuracy: 0.2562

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.8686 - accuracy: 0.2500

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.8428 - accuracy: 0.2552

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.8241 - accuracy: 0.2500

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.8049 - accuracy: 0.2567

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.7848 - accuracy: 0.2688

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.7720 - accuracy: 0.2598

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.7604 - accuracy: 0.2592

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.7505 - accuracy: 0.2604

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.7404 - accuracy: 0.2582

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.7234 - accuracy: 0.2656

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.7170 - accuracy: 0.2679

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.7025 - accuracy: 0.2727

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.6990 - accuracy: 0.2690

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 1.6930 - accuracy: 0.2708

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.6868 - accuracy: 0.2700

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.6819 - accuracy: 0.2716

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.6743 - accuracy: 0.2766

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.6714 - accuracy: 0.2768

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.6647 - accuracy: 0.2791

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.6572 - accuracy: 0.2844

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.6524 - accuracy: 0.2853

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.6448 - accuracy: 0.2900

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.6396 - accuracy: 0.2917

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.6309 - accuracy: 0.2960

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.6227 - accuracy: 0.3027

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.6168 - accuracy: 0.3056

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.6147 - accuracy: 0.3024

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.6116 - accuracy: 0.3043

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.6021 - accuracy: 0.3101

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.5996 - accuracy: 0.3086

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 1.5937 - accuracy: 0.3095

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.5887 - accuracy: 0.3073

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.5815 - accuracy: 0.3121

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.5735 - accuracy: 0.3156

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.5675 - accuracy: 0.3169

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.5615 - accuracy: 0.3195

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.5613 - accuracy: 0.3200

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.5600 - accuracy: 0.3212

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.5557 - accuracy: 0.3241

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.5494 - accuracy: 0.3276

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.5404 - accuracy: 0.3321

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.5364 - accuracy: 0.3329

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.5296 - accuracy: 0.3384

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.5277 - accuracy: 0.3402

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.5266 - accuracy: 0.3380

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.5263 - accuracy: 0.3387

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.5201 - accuracy: 0.3409

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.5144 - accuracy: 0.3426

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.5092 - accuracy: 0.3457

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.5032 - accuracy: 0.3452

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.4995 - accuracy: 0.3451

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.4946 - accuracy: 0.3461

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.4907 - accuracy: 0.3480

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.4875 - accuracy: 0.3480

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.4833 - accuracy: 0.3493

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.4778 - accuracy: 0.3521

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.4737 - accuracy: 0.3529

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.4678 - accuracy: 0.3555

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.4621 - accuracy: 0.3598

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.4592 - accuracy: 0.3609

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.4585 - accuracy: 0.3598

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.4544 - accuracy: 0.3617

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.4481 - accuracy: 0.3648

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.4485 - accuracy: 0.3650

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.4456 - accuracy: 0.3659

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.4432 - accuracy: 0.3664

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.4383 - accuracy: 0.3682

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.4357 - accuracy: 0.3690

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.4323 - accuracy: 0.3703

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.4304 - accuracy: 0.3711

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.4287 - accuracy: 0.3727

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.4275 - accuracy: 0.3739

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.4261 - accuracy: 0.3757

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.4231 - accuracy: 0.3802

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.4221 - accuracy: 0.3816

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.4194 - accuracy: 0.3826

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.4148 - accuracy: 0.3857

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.4104 - accuracy: 0.3887

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.4064 - accuracy: 0.3914

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.4030 - accuracy: 0.3929

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.4002 - accuracy: 0.3958

.. parsed-literal::

    2024-03-14 01:05:12.510620: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-03-14 01:05:12.510913: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.4002 - accuracy: 0.3958 - val_loss: 1.1011 - val_accuracy: 0.5463


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 1.0250 - accuracy: 0.5312

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1375 - accuracy: 0.4844

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0591 - accuracy: 0.5104

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0560 - accuracy: 0.5156

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0542 - accuracy: 0.5500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.0575 - accuracy: 0.5417

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.0445 - accuracy: 0.5536

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0724 - accuracy: 0.5312

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0746 - accuracy: 0.5243

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.1210 - accuracy: 0.5125

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.1403 - accuracy: 0.5085

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.1466 - accuracy: 0.5078

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.1269 - accuracy: 0.5168

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.1164 - accuracy: 0.5201

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.1350 - accuracy: 0.5167

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.1475 - accuracy: 0.5117

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.1385 - accuracy: 0.5165

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.1340 - accuracy: 0.5208

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.1235 - accuracy: 0.5263

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.1218 - accuracy: 0.5281

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.1264 - accuracy: 0.5268

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.1239 - accuracy: 0.5284

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.1272 - accuracy: 0.5299

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.1413 - accuracy: 0.5234

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.1327 - accuracy: 0.5263

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.1352 - accuracy: 0.5240

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.1436 - accuracy: 0.5197

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.1427 - accuracy: 0.5190

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.1428 - accuracy: 0.5172

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.1471 - accuracy: 0.5146

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.1436 - accuracy: 0.5171

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.1460 - accuracy: 0.5156

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.1465 - accuracy: 0.5161

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.1460 - accuracy: 0.5165

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.1450 - accuracy: 0.5161

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.1494 - accuracy: 0.5165

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.1489 - accuracy: 0.5169

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.1429 - accuracy: 0.5197

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.1415 - accuracy: 0.5184

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.1496 - accuracy: 0.5156

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.1497 - accuracy: 0.5137

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.1448 - accuracy: 0.5164

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.1393 - accuracy: 0.5203

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.1406 - accuracy: 0.5220

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.1398 - accuracy: 0.5229

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.1439 - accuracy: 0.5251

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.1399 - accuracy: 0.5279

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.1396 - accuracy: 0.5273

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.1359 - accuracy: 0.5287

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.1334 - accuracy: 0.5288

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.1306 - accuracy: 0.5331

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.1250 - accuracy: 0.5367

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.1235 - accuracy: 0.5342

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.1203 - accuracy: 0.5347

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.1161 - accuracy: 0.5381

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.1148 - accuracy: 0.5396

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.1124 - accuracy: 0.5417

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.1057 - accuracy: 0.5447

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.1030 - accuracy: 0.5461

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.1051 - accuracy: 0.5453

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.1032 - accuracy: 0.5451

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.1001 - accuracy: 0.5464

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0974 - accuracy: 0.5491

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0975 - accuracy: 0.5513

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0949 - accuracy: 0.5524

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0908 - accuracy: 0.5543

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0893 - accuracy: 0.5554

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0899 - accuracy: 0.5559

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0868 - accuracy: 0.5560

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0876 - accuracy: 0.5548

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0855 - accuracy: 0.5553

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0829 - accuracy: 0.5576

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0814 - accuracy: 0.5581

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0798 - accuracy: 0.5598

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0817 - accuracy: 0.5594

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0818 - accuracy: 0.5578

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0814 - accuracy: 0.5579

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0803 - accuracy: 0.5583

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0788 - accuracy: 0.5588

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0786 - accuracy: 0.5584

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0742 - accuracy: 0.5600

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0773 - accuracy: 0.5585

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0750 - accuracy: 0.5604

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0748 - accuracy: 0.5597

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0749 - accuracy: 0.5598

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0754 - accuracy: 0.5594

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0729 - accuracy: 0.5613

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0718 - accuracy: 0.5620

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0715 - accuracy: 0.5616

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0712 - accuracy: 0.5620

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0701 - accuracy: 0.5630

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.0701 - accuracy: 0.5630 - val_loss: 1.0037 - val_accuracy: 0.6049


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 1.4123 - accuracy: 0.2500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.2866 - accuracy: 0.3906

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.1346 - accuracy: 0.5104

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.1366 - accuracy: 0.5078

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0929 - accuracy: 0.5250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.1013 - accuracy: 0.5365

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.1221 - accuracy: 0.5402

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.0853 - accuracy: 0.5508

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0866 - accuracy: 0.5417

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0820 - accuracy: 0.5437

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0732 - accuracy: 0.5511

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0595 - accuracy: 0.5599

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0598 - accuracy: 0.5625

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0663 - accuracy: 0.5603

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.0791 - accuracy: 0.5521

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0733 - accuracy: 0.5547

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0601 - accuracy: 0.5643

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0502 - accuracy: 0.5712

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0489 - accuracy: 0.5740

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0411 - accuracy: 0.5828

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0325 - accuracy: 0.5878

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0385 - accuracy: 0.5838

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.0342 - accuracy: 0.5870

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 1.0351 - accuracy: 0.5885

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0326 - accuracy: 0.5863

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0258 - accuracy: 0.5877

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0334 - accuracy: 0.5868

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0239 - accuracy: 0.5893

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0221 - accuracy: 0.5884

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0134 - accuracy: 0.5938

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0087 - accuracy: 0.5978

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0191 - accuracy: 0.5918

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0184 - accuracy: 0.5928

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0188 - accuracy: 0.5919

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0116 - accuracy: 0.5991

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0204 - accuracy: 0.5964

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0205 - accuracy: 0.5988

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0231 - accuracy: 0.5970

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0227 - accuracy: 0.5962

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0214 - accuracy: 0.5961

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 1.0222 - accuracy: 0.5960

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0206 - accuracy: 0.5945

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0256 - accuracy: 0.5923

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0208 - accuracy: 0.5952

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0225 - accuracy: 0.5944

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0208 - accuracy: 0.5951

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0230 - accuracy: 0.5938

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0247 - accuracy: 0.5931

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0271 - accuracy: 0.5912

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0275 - accuracy: 0.5906

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0276 - accuracy: 0.5907

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0280 - accuracy: 0.5913

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0272 - accuracy: 0.5914

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0249 - accuracy: 0.5926

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0214 - accuracy: 0.5943

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0208 - accuracy: 0.5938

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0225 - accuracy: 0.5948

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0230 - accuracy: 0.5970

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0176 - accuracy: 0.6006

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0145 - accuracy: 0.6010

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0102 - accuracy: 0.6035

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0122 - accuracy: 0.6023

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0110 - accuracy: 0.6027

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0103 - accuracy: 0.6040

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0061 - accuracy: 0.6062

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0008 - accuracy: 0.6084

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9998 - accuracy: 0.6082

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9989 - accuracy: 0.6080

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9976 - accuracy: 0.6091

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9999 - accuracy: 0.6085

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9979 - accuracy: 0.6100

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9963 - accuracy: 0.6115

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9925 - accuracy: 0.6122

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9904 - accuracy: 0.6137

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9892 - accuracy: 0.6134

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9875 - accuracy: 0.6136

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9886 - accuracy: 0.6137

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9876 - accuracy: 0.6139

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9850 - accuracy: 0.6160

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9843 - accuracy: 0.6165

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9831 - accuracy: 0.6177

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9793 - accuracy: 0.6182

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9771 - accuracy: 0.6194

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9760 - accuracy: 0.6202

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9789 - accuracy: 0.6188

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9778 - accuracy: 0.6182

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9748 - accuracy: 0.6197

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9732 - accuracy: 0.6197

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9751 - accuracy: 0.6187

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9724 - accuracy: 0.6191

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9719 - accuracy: 0.6196

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.9719 - accuracy: 0.6196 - val_loss: 0.8646 - val_accuracy: 0.6649


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6668 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6856 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8735 - accuracy: 0.6354

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8807 - accuracy: 0.6484

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8740 - accuracy: 0.6500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.8638 - accuracy: 0.6562

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9007 - accuracy: 0.6384

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9270 - accuracy: 0.6289

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9217 - accuracy: 0.6285

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9068 - accuracy: 0.6344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8945 - accuracy: 0.6449

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9122 - accuracy: 0.6458

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9018 - accuracy: 0.6514

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9281 - accuracy: 0.6429

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9156 - accuracy: 0.6458

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9029 - accuracy: 0.6484

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8984 - accuracy: 0.6526

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8928 - accuracy: 0.6528

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8802 - accuracy: 0.6562

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8764 - accuracy: 0.6536

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.8879 - accuracy: 0.6523

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8826 - accuracy: 0.6552

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8878 - accuracy: 0.6526

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8930 - accuracy: 0.6515

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9037 - accuracy: 0.6529

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8913 - accuracy: 0.6600

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8987 - accuracy: 0.6588

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8952 - accuracy: 0.6576

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8951 - accuracy: 0.6565

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8939 - accuracy: 0.6565

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8944 - accuracy: 0.6565

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8897 - accuracy: 0.6594

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8846 - accuracy: 0.6602

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8876 - accuracy: 0.6592

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8905 - accuracy: 0.6573

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8881 - accuracy: 0.6599

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8925 - accuracy: 0.6581

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8967 - accuracy: 0.6589

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.9018 - accuracy: 0.6557

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9037 - accuracy: 0.6541

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9096 - accuracy: 0.6504

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9076 - accuracy: 0.6520

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9022 - accuracy: 0.6550

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9011 - accuracy: 0.6557

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8987 - accuracy: 0.6564

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8946 - accuracy: 0.6571

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8956 - accuracy: 0.6571

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8954 - accuracy: 0.6571

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8960 - accuracy: 0.6564

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8972 - accuracy: 0.6558

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8970 - accuracy: 0.6540

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8950 - accuracy: 0.6558

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8904 - accuracy: 0.6581

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8885 - accuracy: 0.6592

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8881 - accuracy: 0.6598

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8870 - accuracy: 0.6608

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8840 - accuracy: 0.6623

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8824 - accuracy: 0.6638

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8816 - accuracy: 0.6647

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8837 - accuracy: 0.6641

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8792 - accuracy: 0.6665

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8792 - accuracy: 0.6658

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8791 - accuracy: 0.6672

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8778 - accuracy: 0.6680

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8771 - accuracy: 0.6673

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8789 - accuracy: 0.6662

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8780 - accuracy: 0.6665

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8802 - accuracy: 0.6645

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8850 - accuracy: 0.6653

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8861 - accuracy: 0.6643

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8859 - accuracy: 0.6642

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8860 - accuracy: 0.6649

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8875 - accuracy: 0.6636

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8862 - accuracy: 0.6626

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8857 - accuracy: 0.6625

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8829 - accuracy: 0.6629

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8815 - accuracy: 0.6636

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8799 - accuracy: 0.6631

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8809 - accuracy: 0.6618

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8806 - accuracy: 0.6618

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8822 - accuracy: 0.6609

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8820 - accuracy: 0.6620

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8807 - accuracy: 0.6627

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8790 - accuracy: 0.6637

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8764 - accuracy: 0.6640

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8766 - accuracy: 0.6635

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8735 - accuracy: 0.6652

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8748 - accuracy: 0.6637

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8750 - accuracy: 0.6643

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8719 - accuracy: 0.6649

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8738 - accuracy: 0.6642

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8738 - accuracy: 0.6642 - val_loss: 0.8186 - val_accuracy: 0.6826


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 8s - loss: 0.8166 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7572 - accuracy: 0.6875

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7499 - accuracy: 0.6979

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7817 - accuracy: 0.6953

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7692 - accuracy: 0.7063

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7811 - accuracy: 0.6927

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7387 - accuracy: 0.7143

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7451 - accuracy: 0.7148

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7804 - accuracy: 0.6944

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7773 - accuracy: 0.6875

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7710 - accuracy: 0.6847

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7884 - accuracy: 0.6771

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7934 - accuracy: 0.6779

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8058 - accuracy: 0.6786

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7883 - accuracy: 0.6854

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7994 - accuracy: 0.6836

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8000 - accuracy: 0.6838

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8093 - accuracy: 0.6840

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8087 - accuracy: 0.6859

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8003 - accuracy: 0.6906

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8092 - accuracy: 0.6905

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8114 - accuracy: 0.6903

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8168 - accuracy: 0.6929

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8144 - accuracy: 0.6927

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8095 - accuracy: 0.6963

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8083 - accuracy: 0.6995

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8094 - accuracy: 0.6968

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8099 - accuracy: 0.6975

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8025 - accuracy: 0.7004

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8040 - accuracy: 0.7010

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7999 - accuracy: 0.6998

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7990 - accuracy: 0.6985

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7969 - accuracy: 0.6981

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7908 - accuracy: 0.6978

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7916 - accuracy: 0.6949

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7909 - accuracy: 0.6947

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7930 - accuracy: 0.6978

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7930 - accuracy: 0.6976

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7913 - accuracy: 0.6989

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7869 - accuracy: 0.7002

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7879 - accuracy: 0.6999

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7914 - accuracy: 0.6996

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7884 - accuracy: 0.7000

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7934 - accuracy: 0.6990

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7948 - accuracy: 0.6988

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7915 - accuracy: 0.6992

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7893 - accuracy: 0.7016

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7897 - accuracy: 0.7006

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7874 - accuracy: 0.7029

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7867 - accuracy: 0.7044

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7867 - accuracy: 0.7053

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7892 - accuracy: 0.7050

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7889 - accuracy: 0.7052

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7883 - accuracy: 0.7055

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7911 - accuracy: 0.7029

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7918 - accuracy: 0.7015

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7906 - accuracy: 0.7018

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7951 - accuracy: 0.6995

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8016 - accuracy: 0.6946

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8032 - accuracy: 0.6929

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8074 - accuracy: 0.6918

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8060 - accuracy: 0.6927

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8029 - accuracy: 0.6956

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8012 - accuracy: 0.6964

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8034 - accuracy: 0.6958

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8050 - accuracy: 0.6962

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8060 - accuracy: 0.6956

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8107 - accuracy: 0.6941

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8133 - accuracy: 0.6949

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8143 - accuracy: 0.6948

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8118 - accuracy: 0.6960

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8126 - accuracy: 0.6959

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8119 - accuracy: 0.6962

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8124 - accuracy: 0.6957

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8158 - accuracy: 0.6939

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8142 - accuracy: 0.6950

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8124 - accuracy: 0.6961

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8102 - accuracy: 0.6972

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8099 - accuracy: 0.6971

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8110 - accuracy: 0.6970

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8095 - accuracy: 0.6972

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8096 - accuracy: 0.6968

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8070 - accuracy: 0.6970

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8047 - accuracy: 0.6984

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8039 - accuracy: 0.6997

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8021 - accuracy: 0.7006

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7988 - accuracy: 0.7023

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7968 - accuracy: 0.7025

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7974 - accuracy: 0.7026

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7975 - accuracy: 0.7018

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8003 - accuracy: 0.6996

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8003 - accuracy: 0.6996 - val_loss: 0.8621 - val_accuracy: 0.6866


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6894 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6421 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7591 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7074 - accuracy: 0.7500

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7436 - accuracy: 0.7312

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7346 - accuracy: 0.7344

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7722 - accuracy: 0.7098

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8112 - accuracy: 0.6914

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8111 - accuracy: 0.6910

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8219 - accuracy: 0.6906

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8116 - accuracy: 0.6875

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8174 - accuracy: 0.6875

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8066 - accuracy: 0.6947

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8010 - accuracy: 0.6897

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8055 - accuracy: 0.6917

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8073 - accuracy: 0.6914

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8009 - accuracy: 0.6949

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7975 - accuracy: 0.6944

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8102 - accuracy: 0.6908

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8024 - accuracy: 0.6891

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8150 - accuracy: 0.6860

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8092 - accuracy: 0.6889

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7987 - accuracy: 0.6929

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8059 - accuracy: 0.6901

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8058 - accuracy: 0.6913

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7961 - accuracy: 0.6959

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8005 - accuracy: 0.6956

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7977 - accuracy: 0.6975

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8021 - accuracy: 0.6929

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8019 - accuracy: 0.6938

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7992 - accuracy: 0.6946

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8015 - accuracy: 0.6924

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7932 - accuracy: 0.6960

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7908 - accuracy: 0.6976

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7872 - accuracy: 0.7009

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7821 - accuracy: 0.7023

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7782 - accuracy: 0.7035

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7788 - accuracy: 0.7048

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7842 - accuracy: 0.7051

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7801 - accuracy: 0.7047

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7814 - accuracy: 0.7058

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7758 - accuracy: 0.7076

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7761 - accuracy: 0.7078

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7731 - accuracy: 0.7081

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7705 - accuracy: 0.7090

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7733 - accuracy: 0.7086

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7681 - accuracy: 0.7101

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7636 - accuracy: 0.7103

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7648 - accuracy: 0.7092

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7701 - accuracy: 0.7075

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7705 - accuracy: 0.7077

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7720 - accuracy: 0.7067

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7747 - accuracy: 0.7046

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7740 - accuracy: 0.7054

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7750 - accuracy: 0.7028

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7744 - accuracy: 0.7026

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7817 - accuracy: 0.6990

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7829 - accuracy: 0.6988

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7822 - accuracy: 0.6986

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7827 - accuracy: 0.6974

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7810 - accuracy: 0.6988

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7790 - accuracy: 0.6981

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7787 - accuracy: 0.6989

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7769 - accuracy: 0.6987

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7778 - accuracy: 0.6981

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7760 - accuracy: 0.6979

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7757 - accuracy: 0.6982

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7746 - accuracy: 0.6985

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7735 - accuracy: 0.6984

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7697 - accuracy: 0.6996

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7707 - accuracy: 0.6994

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7704 - accuracy: 0.6992

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7703 - accuracy: 0.6991

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7740 - accuracy: 0.6964

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7786 - accuracy: 0.6950

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7803 - accuracy: 0.6957

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7786 - accuracy: 0.6960

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7778 - accuracy: 0.6959

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7791 - accuracy: 0.6966

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7782 - accuracy: 0.6969

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7810 - accuracy: 0.6952

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7837 - accuracy: 0.6947

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7867 - accuracy: 0.6920

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7832 - accuracy: 0.6931

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7816 - accuracy: 0.6945

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7826 - accuracy: 0.6944

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7817 - accuracy: 0.6950

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7824 - accuracy: 0.6950

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7815 - accuracy: 0.6957

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7819 - accuracy: 0.6952

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7824 - accuracy: 0.6955

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7824 - accuracy: 0.6955 - val_loss: 0.8560 - val_accuracy: 0.6485


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.7046 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6812 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7064 - accuracy: 0.7604

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7372 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7193 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7336 - accuracy: 0.7500

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7254 - accuracy: 0.7455

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7356 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7193 - accuracy: 0.7500

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7135 - accuracy: 0.7563

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7291 - accuracy: 0.7528

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7101 - accuracy: 0.7552

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7058 - accuracy: 0.7596

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7152 - accuracy: 0.7500

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7018 - accuracy: 0.7579

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7067 - accuracy: 0.7500

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7120 - accuracy: 0.7500

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6971 - accuracy: 0.7567

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7064 - accuracy: 0.7500

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7138 - accuracy: 0.7425

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7113 - accuracy: 0.7414

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7117 - accuracy: 0.7404

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7199 - accuracy: 0.7395

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7271 - accuracy: 0.7348

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7301 - accuracy: 0.7342

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7272 - accuracy: 0.7348

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7248 - accuracy: 0.7365

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7217 - accuracy: 0.7380

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7175 - accuracy: 0.7395

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7190 - accuracy: 0.7348

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7188 - accuracy: 0.7352

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7196 - accuracy: 0.7338

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7188 - accuracy: 0.7333

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7151 - accuracy: 0.7338

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7146 - accuracy: 0.7325

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7110 - accuracy: 0.7338

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7143 - accuracy: 0.7343

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7157 - accuracy: 0.7331

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7197 - accuracy: 0.7327

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7178 - accuracy: 0.7331

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7126 - accuracy: 0.7358

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7150 - accuracy: 0.7339

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7149 - accuracy: 0.7350

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7105 - accuracy: 0.7367

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7112 - accuracy: 0.7357

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7145 - accuracy: 0.7353

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7138 - accuracy: 0.7356

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7136 - accuracy: 0.7353

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7159 - accuracy: 0.7349

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7137 - accuracy: 0.7365

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7138 - accuracy: 0.7373

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7136 - accuracy: 0.7370

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7161 - accuracy: 0.7355

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7142 - accuracy: 0.7352

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7155 - accuracy: 0.7360

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7138 - accuracy: 0.7368

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7193 - accuracy: 0.7348

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7191 - accuracy: 0.7340

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7191 - accuracy: 0.7343

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7191 - accuracy: 0.7341

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7237 - accuracy: 0.7308

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7261 - accuracy: 0.7291

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7288 - accuracy: 0.7275

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7281 - accuracy: 0.7278

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7347 - accuracy: 0.7248

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7375 - accuracy: 0.7233

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7349 - accuracy: 0.7242

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7369 - accuracy: 0.7236

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7348 - accuracy: 0.7249

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7316 - accuracy: 0.7257

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7305 - accuracy: 0.7269

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7283 - accuracy: 0.7290

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7303 - accuracy: 0.7288

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7312 - accuracy: 0.7278

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7304 - accuracy: 0.7273

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7303 - accuracy: 0.7272

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7305 - accuracy: 0.7271

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7360 - accuracy: 0.7242

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7348 - accuracy: 0.7237

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7340 - accuracy: 0.7248

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7326 - accuracy: 0.7255

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7340 - accuracy: 0.7251

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7367 - accuracy: 0.7235

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7363 - accuracy: 0.7242

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7358 - accuracy: 0.7241

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7389 - accuracy: 0.7223

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7394 - accuracy: 0.7215

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7370 - accuracy: 0.7225

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7383 - accuracy: 0.7218

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7415 - accuracy: 0.7200

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7427 - accuracy: 0.7193

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7427 - accuracy: 0.7193 - val_loss: 0.7820 - val_accuracy: 0.6826


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 6s - loss: 0.5143 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5858 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6064 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6099 - accuracy: 0.7734

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5806 - accuracy: 0.7937

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5820 - accuracy: 0.7812

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5661 - accuracy: 0.7902

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5837 - accuracy: 0.7812

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6178 - accuracy: 0.7604

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6491 - accuracy: 0.7500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6407 - accuracy: 0.7585

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6423 - accuracy: 0.7630

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6417 - accuracy: 0.7644

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6478 - accuracy: 0.7634

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6599 - accuracy: 0.7563

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6661 - accuracy: 0.7539

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6670 - accuracy: 0.7500

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6770 - accuracy: 0.7448

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6745 - accuracy: 0.7434

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6895 - accuracy: 0.7437

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6934 - accuracy: 0.7411

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6937 - accuracy: 0.7401

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6945 - accuracy: 0.7418

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6882 - accuracy: 0.7435

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6846 - accuracy: 0.7450

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6847 - accuracy: 0.7442

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6783 - accuracy: 0.7432

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6779 - accuracy: 0.7413

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6755 - accuracy: 0.7405

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6744 - accuracy: 0.7429

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6714 - accuracy: 0.7461

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6782 - accuracy: 0.7414

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6779 - accuracy: 0.7407

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6762 - accuracy: 0.7419

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6710 - accuracy: 0.7456

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6685 - accuracy: 0.7474

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6704 - accuracy: 0.7467

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6785 - accuracy: 0.7444

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6850 - accuracy: 0.7437

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6894 - accuracy: 0.7439

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6917 - accuracy: 0.7448

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6897 - accuracy: 0.7442

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6882 - accuracy: 0.7450

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6909 - accuracy: 0.7416

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6910 - accuracy: 0.7404

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6995 - accuracy: 0.7353

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7005 - accuracy: 0.7336

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6986 - accuracy: 0.7346

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6980 - accuracy: 0.7324

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6957 - accuracy: 0.7340

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6958 - accuracy: 0.7349

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6942 - accuracy: 0.7352

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6955 - accuracy: 0.7343

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6971 - accuracy: 0.7340

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6951 - accuracy: 0.7360

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6925 - accuracy: 0.7373

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6899 - accuracy: 0.7381

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6905 - accuracy: 0.7383

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6913 - accuracy: 0.7374

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6962 - accuracy: 0.7356

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6966 - accuracy: 0.7343

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6993 - accuracy: 0.7321

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7027 - accuracy: 0.7314

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7004 - accuracy: 0.7326

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6973 - accuracy: 0.7329

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6960 - accuracy: 0.7336

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6986 - accuracy: 0.7320

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6991 - accuracy: 0.7323

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7040 - accuracy: 0.7294

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7069 - accuracy: 0.7270

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7032 - accuracy: 0.7287

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7039 - accuracy: 0.7290

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7014 - accuracy: 0.7284

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7028 - accuracy: 0.7274

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7059 - accuracy: 0.7269

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7092 - accuracy: 0.7252

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7085 - accuracy: 0.7255

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7065 - accuracy: 0.7258

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7076 - accuracy: 0.7265

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7098 - accuracy: 0.7245

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7095 - accuracy: 0.7244

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7083 - accuracy: 0.7243

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7077 - accuracy: 0.7246

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7067 - accuracy: 0.7253

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7071 - accuracy: 0.7259

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7077 - accuracy: 0.7262

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7049 - accuracy: 0.7272

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7024 - accuracy: 0.7282

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7014 - accuracy: 0.7288

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7014 - accuracy: 0.7283

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7026 - accuracy: 0.7282

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7026 - accuracy: 0.7282 - val_loss: 0.7725 - val_accuracy: 0.6989


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5637 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7595 - accuracy: 0.6562

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7703 - accuracy: 0.6458

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7610 - accuracy: 0.6484

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6969 - accuracy: 0.6938

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6940 - accuracy: 0.7031

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7242 - accuracy: 0.7009

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7173 - accuracy: 0.7070

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7023 - accuracy: 0.7222

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6885 - accuracy: 0.7250

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6845 - accuracy: 0.7216

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6654 - accuracy: 0.7344

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6494 - accuracy: 0.7356

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6584 - accuracy: 0.7321

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6429 - accuracy: 0.7417

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6358 - accuracy: 0.7441

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6351 - accuracy: 0.7445

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6426 - accuracy: 0.7378

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6465 - accuracy: 0.7418

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6476 - accuracy: 0.7406

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6596 - accuracy: 0.7351

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6699 - accuracy: 0.7358

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6699 - accuracy: 0.7351

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6654 - accuracy: 0.7370

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6685 - accuracy: 0.7337

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6718 - accuracy: 0.7272

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6646 - accuracy: 0.7326

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6694 - accuracy: 0.7288

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6695 - accuracy: 0.7306

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6675 - accuracy: 0.7312

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6734 - accuracy: 0.7268

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6712 - accuracy: 0.7285

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6684 - accuracy: 0.7311

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6678 - accuracy: 0.7325

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6717 - accuracy: 0.7295

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6686 - accuracy: 0.7318

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6803 - accuracy: 0.7238

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6770 - accuracy: 0.7262

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6820 - accuracy: 0.7244

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6765 - accuracy: 0.7273

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6740 - accuracy: 0.7294

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6756 - accuracy: 0.7292

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6744 - accuracy: 0.7311

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6755 - accuracy: 0.7308

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6772 - accuracy: 0.7306

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6734 - accuracy: 0.7317

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6705 - accuracy: 0.7334

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6696 - accuracy: 0.7344

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6649 - accuracy: 0.7366

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6602 - accuracy: 0.7394

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6587 - accuracy: 0.7402

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6623 - accuracy: 0.7398

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6580 - accuracy: 0.7412

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6609 - accuracy: 0.7390

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6635 - accuracy: 0.7398

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6632 - accuracy: 0.7388

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6612 - accuracy: 0.7390

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6593 - accuracy: 0.7403

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6578 - accuracy: 0.7421

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6558 - accuracy: 0.7422

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6562 - accuracy: 0.7413

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6541 - accuracy: 0.7424

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6542 - accuracy: 0.7440

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6577 - accuracy: 0.7427

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6570 - accuracy: 0.7437

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6561 - accuracy: 0.7443

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6592 - accuracy: 0.7435

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6609 - accuracy: 0.7436

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6608 - accuracy: 0.7441

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6613 - accuracy: 0.7455

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6624 - accuracy: 0.7447

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6662 - accuracy: 0.7431

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6664 - accuracy: 0.7436

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6661 - accuracy: 0.7433

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6649 - accuracy: 0.7434

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6659 - accuracy: 0.7431

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6643 - accuracy: 0.7432

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6627 - accuracy: 0.7437

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6635 - accuracy: 0.7437

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6651 - accuracy: 0.7430

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6710 - accuracy: 0.7416

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6726 - accuracy: 0.7402

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6726 - accuracy: 0.7399

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6726 - accuracy: 0.7404

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6738 - accuracy: 0.7394

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6734 - accuracy: 0.7392

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6741 - accuracy: 0.7386

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6760 - accuracy: 0.7373

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6787 - accuracy: 0.7361

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6787 - accuracy: 0.7362

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6796 - accuracy: 0.7360

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6796 - accuracy: 0.7360 - val_loss: 0.7852 - val_accuracy: 0.6839


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6974 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7623 - accuracy: 0.7031

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7011 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6527 - accuracy: 0.7500

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6627 - accuracy: 0.7312

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6613 - accuracy: 0.7396

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.6486 - accuracy: 0.7411

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6457 - accuracy: 0.7539

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6517 - accuracy: 0.7674

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6442 - accuracy: 0.7656

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6407 - accuracy: 0.7614

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6448 - accuracy: 0.7604

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6355 - accuracy: 0.7620

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6397 - accuracy: 0.7567

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6390 - accuracy: 0.7540

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6369 - accuracy: 0.7500

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6332 - accuracy: 0.7553

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6311 - accuracy: 0.7583

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6247 - accuracy: 0.7611

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6217 - accuracy: 0.7605

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6145 - accuracy: 0.7644

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6078 - accuracy: 0.7692

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6101 - accuracy: 0.7684

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6053 - accuracy: 0.7727

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6067 - accuracy: 0.7743

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6089 - accuracy: 0.7699

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6150 - accuracy: 0.7714

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6108 - accuracy: 0.7717

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6139 - accuracy: 0.7679

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6095 - accuracy: 0.7693

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6089 - accuracy: 0.7707

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6108 - accuracy: 0.7710

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6104 - accuracy: 0.7704

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6078 - accuracy: 0.7698

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6027 - accuracy: 0.7727

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6078 - accuracy: 0.7704

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6044 - accuracy: 0.7707

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6180 - accuracy: 0.7661

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6127 - accuracy: 0.7673

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6130 - accuracy: 0.7669

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6163 - accuracy: 0.7657

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6232 - accuracy: 0.7610

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6272 - accuracy: 0.7593

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6322 - accuracy: 0.7584

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6344 - accuracy: 0.7575

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6342 - accuracy: 0.7574

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6313 - accuracy: 0.7585

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6356 - accuracy: 0.7564

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6325 - accuracy: 0.7582

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6322 - accuracy: 0.7586

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6340 - accuracy: 0.7579

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6354 - accuracy: 0.7559

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6339 - accuracy: 0.7564

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6414 - accuracy: 0.7529

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6382 - accuracy: 0.7556

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6369 - accuracy: 0.7561

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6405 - accuracy: 0.7538

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6429 - accuracy: 0.7521

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6468 - accuracy: 0.7505

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6456 - accuracy: 0.7510

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6459 - accuracy: 0.7510

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6490 - accuracy: 0.7510

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6511 - accuracy: 0.7500

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6499 - accuracy: 0.7519

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6506 - accuracy: 0.7505

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6511 - accuracy: 0.7505

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6504 - accuracy: 0.7509

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6502 - accuracy: 0.7527

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6510 - accuracy: 0.7531

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6481 - accuracy: 0.7535

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6454 - accuracy: 0.7552

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6466 - accuracy: 0.7552

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6454 - accuracy: 0.7555

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6446 - accuracy: 0.7559

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6440 - accuracy: 0.7562

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6432 - accuracy: 0.7569

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6427 - accuracy: 0.7576

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6444 - accuracy: 0.7560

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6448 - accuracy: 0.7547

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6454 - accuracy: 0.7543

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6463 - accuracy: 0.7534

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6460 - accuracy: 0.7538

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6480 - accuracy: 0.7526

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6474 - accuracy: 0.7529

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6469 - accuracy: 0.7533

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6491 - accuracy: 0.7529

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6480 - accuracy: 0.7536

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6474 - accuracy: 0.7532

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6477 - accuracy: 0.7528

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6450 - accuracy: 0.7534

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6427 - accuracy: 0.7541

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6427 - accuracy: 0.7541 - val_loss: 0.7060 - val_accuracy: 0.7398


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.4816 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5658 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6230 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5988 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5603 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5591 - accuracy: 0.7812

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5467 - accuracy: 0.7857

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5659 - accuracy: 0.7773

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5584 - accuracy: 0.7882

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5414 - accuracy: 0.8000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5685 - accuracy: 0.7841

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5737 - accuracy: 0.7786

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5911 - accuracy: 0.7668

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5822 - accuracy: 0.7679

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6027 - accuracy: 0.7604

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6038 - accuracy: 0.7617

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6151 - accuracy: 0.7574

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5984 - accuracy: 0.7691

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6013 - accuracy: 0.7664

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5939 - accuracy: 0.7734

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5976 - accuracy: 0.7738

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5858 - accuracy: 0.7770

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5967 - accuracy: 0.7731

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6018 - accuracy: 0.7721

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6210 - accuracy: 0.7600

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6244 - accuracy: 0.7560

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6228 - accuracy: 0.7581

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6154 - accuracy: 0.7623

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6124 - accuracy: 0.7629

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6160 - accuracy: 0.7604

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6119 - accuracy: 0.7641

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6122 - accuracy: 0.7666

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6103 - accuracy: 0.7680

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6082 - accuracy: 0.7721

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6054 - accuracy: 0.7732

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6003 - accuracy: 0.7752

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6032 - accuracy: 0.7728

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6073 - accuracy: 0.7722

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6065 - accuracy: 0.7732

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6033 - accuracy: 0.7766

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6046 - accuracy: 0.7759

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6116 - accuracy: 0.7731

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6083 - accuracy: 0.7754

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6040 - accuracy: 0.7770

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6093 - accuracy: 0.7750

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6129 - accuracy: 0.7724

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6106 - accuracy: 0.7739

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6122 - accuracy: 0.7728

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6164 - accuracy: 0.7710

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6149 - accuracy: 0.7731

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6164 - accuracy: 0.7696

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6150 - accuracy: 0.7704

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6119 - accuracy: 0.7730

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6106 - accuracy: 0.7743

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6101 - accuracy: 0.7744

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6127 - accuracy: 0.7729

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6112 - accuracy: 0.7727

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6090 - accuracy: 0.7734

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6103 - accuracy: 0.7741

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6130 - accuracy: 0.7731

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6136 - accuracy: 0.7728

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6160 - accuracy: 0.7709

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6132 - accuracy: 0.7721

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6131 - accuracy: 0.7717

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6124 - accuracy: 0.7723

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6098 - accuracy: 0.7734

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6110 - accuracy: 0.7726

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6101 - accuracy: 0.7727

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6090 - accuracy: 0.7737

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6079 - accuracy: 0.7739

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6075 - accuracy: 0.7722

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6092 - accuracy: 0.7715

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6081 - accuracy: 0.7725

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6093 - accuracy: 0.7722

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6091 - accuracy: 0.7731

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6069 - accuracy: 0.7740

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6071 - accuracy: 0.7741

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6103 - accuracy: 0.7734

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6132 - accuracy: 0.7727

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6166 - accuracy: 0.7709

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6164 - accuracy: 0.7706

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6183 - accuracy: 0.7693

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6171 - accuracy: 0.7698

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6214 - accuracy: 0.7670

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6200 - accuracy: 0.7679

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6214 - accuracy: 0.7673

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6231 - accuracy: 0.7664

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6214 - accuracy: 0.7665

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6231 - accuracy: 0.7657

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6236 - accuracy: 0.7658

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6231 - accuracy: 0.7660

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6231 - accuracy: 0.7660 - val_loss: 0.6830 - val_accuracy: 0.7207


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.4528 - accuracy: 0.9062

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5887 - accuracy: 0.8438

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5979 - accuracy: 0.8229

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6200 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5897 - accuracy: 0.7875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6050 - accuracy: 0.7812

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6150 - accuracy: 0.7723

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6369 - accuracy: 0.7695

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6385 - accuracy: 0.7604

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6384 - accuracy: 0.7625

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6179 - accuracy: 0.7784

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6273 - accuracy: 0.7708

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6234 - accuracy: 0.7716

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6208 - accuracy: 0.7790

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6146 - accuracy: 0.7771

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6132 - accuracy: 0.7715

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6065 - accuracy: 0.7739

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6062 - accuracy: 0.7708

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6032 - accuracy: 0.7697

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5955 - accuracy: 0.7703

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5881 - accuracy: 0.7723

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5906 - accuracy: 0.7699

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5878 - accuracy: 0.7731

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5907 - accuracy: 0.7695

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5836 - accuracy: 0.7700

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5908 - accuracy: 0.7656

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5874 - accuracy: 0.7685

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5827 - accuracy: 0.7712

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5743 - accuracy: 0.7748

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5769 - accuracy: 0.7760

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5813 - accuracy: 0.7742

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5775 - accuracy: 0.7754

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5762 - accuracy: 0.7765

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5750 - accuracy: 0.7767

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5716 - accuracy: 0.7777

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5788 - accuracy: 0.7769

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5791 - accuracy: 0.7770

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5815 - accuracy: 0.7730

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5884 - accuracy: 0.7692

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5886 - accuracy: 0.7719

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5873 - accuracy: 0.7752

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5860 - accuracy: 0.7753

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5916 - accuracy: 0.7754

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5890 - accuracy: 0.7777

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5877 - accuracy: 0.7764

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5925 - accuracy: 0.7738

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5947 - accuracy: 0.7726

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5963 - accuracy: 0.7708

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5925 - accuracy: 0.7704

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5906 - accuracy: 0.7713

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5922 - accuracy: 0.7714

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5914 - accuracy: 0.7722

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5947 - accuracy: 0.7712

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5919 - accuracy: 0.7731

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5946 - accuracy: 0.7713

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5980 - accuracy: 0.7698

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5938 - accuracy: 0.7711

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5914 - accuracy: 0.7718

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5916 - accuracy: 0.7714

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5879 - accuracy: 0.7742

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5873 - accuracy: 0.7743

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5902 - accuracy: 0.7734

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5862 - accuracy: 0.7745

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5847 - accuracy: 0.7756

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5830 - accuracy: 0.7761

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5840 - accuracy: 0.7762

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5836 - accuracy: 0.7758

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5825 - accuracy: 0.7773

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5815 - accuracy: 0.7782

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5830 - accuracy: 0.7774

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5809 - accuracy: 0.7779

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5780 - accuracy: 0.7792

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5819 - accuracy: 0.7775

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5805 - accuracy: 0.7784

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5846 - accuracy: 0.7772

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5839 - accuracy: 0.7777

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5872 - accuracy: 0.7765

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5838 - accuracy: 0.7778

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5818 - accuracy: 0.7786

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5848 - accuracy: 0.7775

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5819 - accuracy: 0.7787

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5814 - accuracy: 0.7795

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5848 - accuracy: 0.7784

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5871 - accuracy: 0.7773

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5858 - accuracy: 0.7777

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5870 - accuracy: 0.7774

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5874 - accuracy: 0.7774

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5869 - accuracy: 0.7778

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5893 - accuracy: 0.7761

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5891 - accuracy: 0.7765

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5891 - accuracy: 0.7766

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5891 - accuracy: 0.7766 - val_loss: 0.7268 - val_accuracy: 0.7125


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5683 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5409 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5283 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5545 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5757 - accuracy: 0.7688

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5677 - accuracy: 0.7760

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.5731 - accuracy: 0.7768

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5936 - accuracy: 0.7695

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6000 - accuracy: 0.7639

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5784 - accuracy: 0.7656

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5672 - accuracy: 0.7699

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5886 - accuracy: 0.7604

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5901 - accuracy: 0.7572

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5805 - accuracy: 0.7656

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5675 - accuracy: 0.7729

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5608 - accuracy: 0.7812

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5555 - accuracy: 0.7831

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5488 - accuracy: 0.7865

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5547 - accuracy: 0.7862

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5460 - accuracy: 0.7891

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5671 - accuracy: 0.7887

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5661 - accuracy: 0.7869

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5705 - accuracy: 0.7840

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5710 - accuracy: 0.7865

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5750 - accuracy: 0.7850

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5801 - accuracy: 0.7812

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5785 - accuracy: 0.7801

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5800 - accuracy: 0.7812

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5798 - accuracy: 0.7812

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5783 - accuracy: 0.7823

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5755 - accuracy: 0.7853

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5785 - accuracy: 0.7852

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5774 - accuracy: 0.7831

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5724 - accuracy: 0.7868

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5688 - accuracy: 0.7875

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5783 - accuracy: 0.7821

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5860 - accuracy: 0.7770

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5877 - accuracy: 0.7763

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5845 - accuracy: 0.7764

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5822 - accuracy: 0.7781

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5845 - accuracy: 0.7767

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5821 - accuracy: 0.7790

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5820 - accuracy: 0.7798

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5780 - accuracy: 0.7827

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5794 - accuracy: 0.7826

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5797 - accuracy: 0.7826

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5807 - accuracy: 0.7819

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5813 - accuracy: 0.7826

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5827 - accuracy: 0.7825

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5835 - accuracy: 0.7825

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5834 - accuracy: 0.7819

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5814 - accuracy: 0.7837

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5798 - accuracy: 0.7848

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5778 - accuracy: 0.7859

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5735 - accuracy: 0.7869

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5733 - accuracy: 0.7874

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5704 - accuracy: 0.7878

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5752 - accuracy: 0.7872

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5772 - accuracy: 0.7876

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5781 - accuracy: 0.7865

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5763 - accuracy: 0.7869

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5747 - accuracy: 0.7878

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5747 - accuracy: 0.7868

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5783 - accuracy: 0.7867

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5829 - accuracy: 0.7866

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5817 - accuracy: 0.7870

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5861 - accuracy: 0.7851

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5836 - accuracy: 0.7855

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5825 - accuracy: 0.7863

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5832 - accuracy: 0.7853

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5841 - accuracy: 0.7840

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5877 - accuracy: 0.7818

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5888 - accuracy: 0.7805

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5900 - accuracy: 0.7797

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5909 - accuracy: 0.7805

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5950 - accuracy: 0.7789

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5940 - accuracy: 0.7793

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5932 - accuracy: 0.7798

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5929 - accuracy: 0.7798

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5924 - accuracy: 0.7806

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5901 - accuracy: 0.7817

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5944 - accuracy: 0.7798

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5931 - accuracy: 0.7802

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5959 - accuracy: 0.7791

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5957 - accuracy: 0.7792

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5965 - accuracy: 0.7785

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6008 - accuracy: 0.7771

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5998 - accuracy: 0.7778

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6002 - accuracy: 0.7782

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5994 - accuracy: 0.7779

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6023 - accuracy: 0.7766

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6023 - accuracy: 0.7766 - val_loss: 0.7342 - val_accuracy: 0.7125


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5249 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4749 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5423 - accuracy: 0.7812

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5309 - accuracy: 0.7969

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5513 - accuracy: 0.7875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5336 - accuracy: 0.8073

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5205 - accuracy: 0.8036

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5102 - accuracy: 0.8086

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5108 - accuracy: 0.8125

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5029 - accuracy: 0.8094

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5258 - accuracy: 0.8040

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5489 - accuracy: 0.7943

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5512 - accuracy: 0.7909

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5371 - accuracy: 0.7991

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5276 - accuracy: 0.8042

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5366 - accuracy: 0.8008

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5399 - accuracy: 0.7996

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5463 - accuracy: 0.7934

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5414 - accuracy: 0.7961

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5484 - accuracy: 0.7891

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5405 - accuracy: 0.7932

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5426 - accuracy: 0.7955

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5506 - accuracy: 0.7948

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5631 - accuracy: 0.7904

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5610 - accuracy: 0.7912

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5654 - accuracy: 0.7921

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5699 - accuracy: 0.7894

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5687 - accuracy: 0.7913

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5746 - accuracy: 0.7888

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5681 - accuracy: 0.7917

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5661 - accuracy: 0.7903

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5646 - accuracy: 0.7900

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5605 - accuracy: 0.7898

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5652 - accuracy: 0.7877

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5614 - accuracy: 0.7902

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5589 - accuracy: 0.7899

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5510 - accuracy: 0.7939

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5544 - accuracy: 0.7919

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5583 - accuracy: 0.7901

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5622 - accuracy: 0.7867

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5576 - accuracy: 0.7889

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5574 - accuracy: 0.7894

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5590 - accuracy: 0.7900

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5561 - accuracy: 0.7919

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5570 - accuracy: 0.7931

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5590 - accuracy: 0.7921

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5577 - accuracy: 0.7925

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5536 - accuracy: 0.7936

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5539 - accuracy: 0.7940

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5514 - accuracy: 0.7950

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5497 - accuracy: 0.7953

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5544 - accuracy: 0.7932

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5524 - accuracy: 0.7948

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5495 - accuracy: 0.7951

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5483 - accuracy: 0.7954

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5478 - accuracy: 0.7952

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5472 - accuracy: 0.7944

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5473 - accuracy: 0.7952

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5456 - accuracy: 0.7960

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5451 - accuracy: 0.7953

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5457 - accuracy: 0.7955

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5492 - accuracy: 0.7938

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5529 - accuracy: 0.7922

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5542 - accuracy: 0.7920

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5502 - accuracy: 0.7947

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5504 - accuracy: 0.7949

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5499 - accuracy: 0.7952

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5481 - accuracy: 0.7955

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5496 - accuracy: 0.7953

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5533 - accuracy: 0.7942

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5506 - accuracy: 0.7957

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5517 - accuracy: 0.7951

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5501 - accuracy: 0.7958

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5507 - accuracy: 0.7960

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5530 - accuracy: 0.7941

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5563 - accuracy: 0.7919

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5583 - accuracy: 0.7902

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5586 - accuracy: 0.7897

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5582 - accuracy: 0.7896

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5564 - accuracy: 0.7895

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5540 - accuracy: 0.7905

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5557 - accuracy: 0.7897

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5580 - accuracy: 0.7884

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5586 - accuracy: 0.7887

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5577 - accuracy: 0.7886

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5576 - accuracy: 0.7885

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5613 - accuracy: 0.7870

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5656 - accuracy: 0.7852

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5655 - accuracy: 0.7862

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5625 - accuracy: 0.7875

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5631 - accuracy: 0.7875

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5631 - accuracy: 0.7875 - val_loss: 0.7228 - val_accuracy: 0.7262


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6605 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5627 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5660 - accuracy: 0.7812

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5433 - accuracy: 0.7891

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5414 - accuracy: 0.7937

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5389 - accuracy: 0.7969

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5724 - accuracy: 0.7679

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5766 - accuracy: 0.7773

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5701 - accuracy: 0.7812

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5559 - accuracy: 0.7906

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5540 - accuracy: 0.7926

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5424 - accuracy: 0.7943

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5560 - accuracy: 0.7885

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5635 - accuracy: 0.7857

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5578 - accuracy: 0.7875

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5638 - accuracy: 0.7852

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5539 - accuracy: 0.7868

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5587 - accuracy: 0.7917

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5593 - accuracy: 0.7878

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5621 - accuracy: 0.7875

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5518 - accuracy: 0.7917

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5467 - accuracy: 0.7926

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5472 - accuracy: 0.7908

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5368 - accuracy: 0.7969

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5310 - accuracy: 0.7975

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5351 - accuracy: 0.7957

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5447 - accuracy: 0.7894

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5400 - accuracy: 0.7879

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5420 - accuracy: 0.7866

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5519 - accuracy: 0.7823

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5541 - accuracy: 0.7853

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5498 - accuracy: 0.7881

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5449 - accuracy: 0.7907

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5391 - accuracy: 0.7932

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5354 - accuracy: 0.7973

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5366 - accuracy: 0.7969

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5392 - accuracy: 0.7965

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5346 - accuracy: 0.7977

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5339 - accuracy: 0.7989

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5307 - accuracy: 0.8000

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5438 - accuracy: 0.7965

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5453 - accuracy: 0.7954

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5435 - accuracy: 0.7958

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5494 - accuracy: 0.7940

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5449 - accuracy: 0.7958

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5440 - accuracy: 0.7962

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5483 - accuracy: 0.7932

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5479 - accuracy: 0.7936

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5437 - accuracy: 0.7953

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5465 - accuracy: 0.7956

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5483 - accuracy: 0.7953

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5488 - accuracy: 0.7945

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5549 - accuracy: 0.7925

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5522 - accuracy: 0.7934

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5584 - accuracy: 0.7909

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5559 - accuracy: 0.7919

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5540 - accuracy: 0.7911

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5580 - accuracy: 0.7888

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5591 - accuracy: 0.7892

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5604 - accuracy: 0.7880

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5608 - accuracy: 0.7884

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5606 - accuracy: 0.7883

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5610 - accuracy: 0.7882

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5588 - accuracy: 0.7886

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5613 - accuracy: 0.7871

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5637 - accuracy: 0.7856

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5600 - accuracy: 0.7869

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5596 - accuracy: 0.7868

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5601 - accuracy: 0.7867

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5569 - accuracy: 0.7889

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5580 - accuracy: 0.7883

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5562 - accuracy: 0.7887

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5564 - accuracy: 0.7898

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5560 - accuracy: 0.7901

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5560 - accuracy: 0.7904

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5575 - accuracy: 0.7899

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5563 - accuracy: 0.7906

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5536 - accuracy: 0.7917

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5525 - accuracy: 0.7923

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5485 - accuracy: 0.7945

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5488 - accuracy: 0.7940

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5492 - accuracy: 0.7934

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5478 - accuracy: 0.7929

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5504 - accuracy: 0.7909

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5538 - accuracy: 0.7894

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5543 - accuracy: 0.7893

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5539 - accuracy: 0.7892

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5517 - accuracy: 0.7901

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5507 - accuracy: 0.7907

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5557 - accuracy: 0.7889

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5547 - accuracy: 0.7899

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5547 - accuracy: 0.7899 - val_loss: 0.7232 - val_accuracy: 0.7234


Visualize Training Results
--------------------------

`back to top ⬆️ <#table-of-contents>`__

After applying data augmentation and Dropout, there is less overfitting
than before, and training and validation accuracy are closer aligned.

.. code:: ipython3

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
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



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_66_0.png


Predict on New Data
-------------------

`back to top ⬆️ <#table-of-contents>`__

Finally, let us use the model to classify an image that was not included
in the training or validation sets.

   **Note**: Data augmentation and Dropout layers are inactive at
   inference time.

.. code:: ipython3

    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    
    img = keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


.. parsed-literal::

    
1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
1/1 [==============================] - 0s 74ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 92.99 percent confidence.


Save the TensorFlow Model
-------------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    #save the trained model - a new folder flower will be created
    #and the file "saved_model.pb" is the pre-trained model
    model_dir = "model"
    saved_model_dir = f"{model_dir}/flower/saved_model"
    model.save(saved_model_dir)


.. parsed-literal::

    2024-03-14 01:06:36.025267: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-14 01:06:36.111994: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.121837: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-14 01:06:36.132704: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.139639: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.146452: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.157227: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.219928: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-03-14 01:06:36.287595: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.308029: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-03-14 01:06:36.347207: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.372283: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.444815: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-14 01:06:36.587856: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.725931: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.759998: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-14 01:06:36.787825: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-14 01:06:36.834957: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 4 of 4). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


Convert the TensorFlow model with OpenVINO Model Conversion API
---------------------------------------------------------------

`back to top ⬆️ <#table-of-contents>`__ To convert the model to
OpenVINO IR with ``FP16`` precision, use model conversion Python API.

.. code:: ipython3

    # Convert the model to ir model format and save it.
    ir_model_path = Path("model/flower")
    ir_model_path.mkdir(parents=True, exist_ok=True)
    ir_model = ov.convert_model(saved_model_dir, input=[1,180,180,3])
    ov.save_model(ir_model, ir_model_path / "flower_ir.xml")

Preprocessing Image Function
----------------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    def pre_process_image(imagePath, img_height=180):
        # Model input format
        n, h, w, c = [1, img_height, img_height, 3]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)
    
        # Convert to array and change data layout from HWC to CHW
        image = np.array(image)
        input_image = image.reshape((n, h, w, c))
    
        return input_image

OpenVINO Runtime Setup
----------------------

`back to top ⬆️ <#table-of-contents>`__

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    # Initialize OpenVINO runtime
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    class_names=["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    
    compiled_model = core.compile_model(model=ir_model, device_name=device.value)
    
    del ir_model
    
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

Run the Inference Step
----------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    # Run inference on the input image...
    inp_img_url = "https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg"
    OUTPUT_DIR = "output"
    inp_file_name = f"A_Close_Up_Photo_of_a_Dandelion.jpg"
    file_path = Path(OUTPUT_DIR)/Path(inp_file_name)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download the image
    download_file(inp_img_url, inp_file_name, directory=OUTPUT_DIR)
    
    # Pre-process the image and get it ready for inference.
    input_image = pre_process_image(file_path)
    
    print(input_image.shape)
    print(input_layer.shape)
    res = compiled_model([input_image])[output_layer]
    
    score = tf.nn.softmax(res[0])
    
    # Show the results
    image = Image.open(file_path)
    plt.imshow(image)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


.. parsed-literal::

    'output/A_Close_Up_Photo_of_a_Dandelion.jpg' already exists.
    (1, 180, 180, 3)
    [1,180,180,3]
    This image most likely belongs to dandelion with a 99.75 percent confidence.



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_79_1.png


The Next Steps
--------------

`back to top ⬆️ <#table-of-contents>`__

This tutorial showed how to train a TensorFlow model, how to convert
that model to OpenVINO’s IR format, and how to do inference on the
converted model. For faster inference speed, you can quantize the IR
model. To see how to quantize this model with OpenVINO’s `Post-training
Quantization with NNCF
Tool <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__,
check out the `Post-Training Quantization with TensorFlow Classification
Model <301-tensorflow-training-openvino-nncf-with-output.html>`__ notebook.
