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
Classification Model <./tensorflow-training-openvino-nncf.ipynb>`__
notebook.

This training code comprises the official `TensorFlow Image
Classification
Tutorial <https://www.tensorflow.org/tutorials/images/classification>`__
in its entirety.

The ``flower_ir.bin`` and ``flower_ir.xml`` (pre-trained models) can be
obtained by executing the code with ‘Runtime->Run All’ or the
``Ctrl+F9`` command.

.. code:: ipython3

    import platform
    
    %pip install -q "openvino>=2023.1.0" "pillow"
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"
    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow-macos>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version <= '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version <= '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform != 'darwin' and python_version <= '3.8'"
    %pip install -q tf_keras


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


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
    
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    
    import PIL
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    import openvino as ov
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2024-03-27 15:05:39.353545: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-27 15:05:39.388084: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-27 15:05:39.931777: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_14_0.png



.. code:: ipython3

    PIL.Image.open(str(roses[1]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_15_0.png



And some tulips:

.. code:: ipython3

    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_17_0.png



.. code:: ipython3

    PIL.Image.open(str(tulips[1]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_18_0.png



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

    2024-03-27 15:05:43.037307: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-27 15:05:43.037341: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-03-27 15:05:43.037345: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-03-27 15:05:43.037483: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-03-27 15:05:43.037498: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-03-27 15:05:43.037502: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


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

    2024-03-27 15:05:43.369451: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-27 15:05:43.369773: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_29_1.png


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

    2024-03-27 15:05:44.206486: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-27 15:05:44.206892: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


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

    normalization_layer = tf.keras.layers.Rescaling(1./255)

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

    2024-03-27 15:05:44.383697: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-27 15:05:44.384075: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    0.0 0.9791725


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
    
    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes)
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

    data_augmentation = tf.keras.Sequential(
      [
        tf.keras.layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
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

    2024-03-27 15:05:45.139326: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-27 15:05:45.140213: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_57_1.png


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

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, name="outputs")
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

    2024-03-27 15:05:46.353212: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-27 15:05:46.353777: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    
 1/92 [..............................] - ETA: 1:29 - loss: 1.5762 - accuracy: 0.2188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 2.9148 - accuracy: 0.2031  

.. parsed-literal::

    
 3/92 [..............................] - ETA: 6s - loss: 2.6059 - accuracy: 0.1875

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 2.4100 - accuracy: 0.2266

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 2.3016 - accuracy: 0.2188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 2.2073 - accuracy: 0.2396

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 2.1357 - accuracy: 0.2454

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 2.0602 - accuracy: 0.2621

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 5s - loss: 2.0043 - accuracy: 0.2750

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.9629 - accuracy: 0.2788

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.9273 - accuracy: 0.2674

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.8966 - accuracy: 0.2793

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.8686 - accuracy: 0.2868

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.8420 - accuracy: 0.2932

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.8229 - accuracy: 0.2924

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.8025 - accuracy: 0.2937

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.7830 - accuracy: 0.3097

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.7593 - accuracy: 0.3169

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.7394 - accuracy: 0.3200

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.7219 - accuracy: 0.3259

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.7047 - accuracy: 0.3343

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.6880 - accuracy: 0.3376

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.6820 - accuracy: 0.3324

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.6651 - accuracy: 0.3382

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.6428 - accuracy: 0.3485

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.6314 - accuracy: 0.3495

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.6181 - accuracy: 0.3540

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.6088 - accuracy: 0.3581

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.5982 - accuracy: 0.3641

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.5905 - accuracy: 0.3666

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.5906 - accuracy: 0.3679

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.5817 - accuracy: 0.3711

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.5777 - accuracy: 0.3712

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.5649 - accuracy: 0.3750

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.5579 - accuracy: 0.3804

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.5478 - accuracy: 0.3837

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.5364 - accuracy: 0.3844

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.5273 - accuracy: 0.3858

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.5191 - accuracy: 0.3911

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.5129 - accuracy: 0.3923

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.5047 - accuracy: 0.3926

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.5009 - accuracy: 0.3930

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.4945 - accuracy: 0.3947

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.4866 - accuracy: 0.4000

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.4842 - accuracy: 0.3980

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.4788 - accuracy: 0.3996

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.4739 - accuracy: 0.4031

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.4651 - accuracy: 0.4051

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.4607 - accuracy: 0.4077

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.4617 - accuracy: 0.4045

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.4542 - accuracy: 0.4083

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.4509 - accuracy: 0.4094

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.4428 - accuracy: 0.4147

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.4357 - accuracy: 0.4180

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.4354 - accuracy: 0.4178

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.4305 - accuracy: 0.4187

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.4249 - accuracy: 0.4202

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.4168 - accuracy: 0.4226

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.4140 - accuracy: 0.4234

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.4095 - accuracy: 0.4257

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.4053 - accuracy: 0.4280

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.4008 - accuracy: 0.4302

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.4012 - accuracy: 0.4293

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.3980 - accuracy: 0.4314

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.3962 - accuracy: 0.4315

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.3923 - accuracy: 0.4325

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.3883 - accuracy: 0.4326

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.3851 - accuracy: 0.4327

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.3809 - accuracy: 0.4332

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.3762 - accuracy: 0.4359

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.3730 - accuracy: 0.4382

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.3676 - accuracy: 0.4395

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.3632 - accuracy: 0.4412

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.3610 - accuracy: 0.4415

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.3571 - accuracy: 0.4419

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.3552 - accuracy: 0.4414

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.3531 - accuracy: 0.4406

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.3490 - accuracy: 0.4421

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.3451 - accuracy: 0.4433

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.3400 - accuracy: 0.4451

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.3362 - accuracy: 0.4466

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.3339 - accuracy: 0.4476

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.3314 - accuracy: 0.4486

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.3315 - accuracy: 0.4489

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.3303 - accuracy: 0.4491

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.3264 - accuracy: 0.4508

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.3249 - accuracy: 0.4510

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.3226 - accuracy: 0.4516

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.3165 - accuracy: 0.4553

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.3145 - accuracy: 0.4547

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.3131 - accuracy: 0.4542

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.3116 - accuracy: 0.4554

.. parsed-literal::

    2024-03-27 15:05:52.614987: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-03-27 15:05:52.615370: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 65ms/step - loss: 1.3116 - accuracy: 0.4554 - val_loss: 1.0949 - val_accuracy: 0.5586


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 1.1130 - accuracy: 0.5312

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1233 - accuracy: 0.5938

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0375 - accuracy: 0.6146

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0203 - accuracy: 0.5938

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9950 - accuracy: 0.6375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 1.0445 - accuracy: 0.6198

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.0279 - accuracy: 0.6295

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0125 - accuracy: 0.6211

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0556 - accuracy: 0.5868

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0721 - accuracy: 0.5781

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0709 - accuracy: 0.5710

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0644 - accuracy: 0.5651

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0591 - accuracy: 0.5697

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0680 - accuracy: 0.5647

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.0649 - accuracy: 0.5688

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0538 - accuracy: 0.5664

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0563 - accuracy: 0.5680

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0501 - accuracy: 0.5694

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0496 - accuracy: 0.5707

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0418 - accuracy: 0.5781

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0418 - accuracy: 0.5818

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0428 - accuracy: 0.5753

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 1.0363 - accuracy: 0.5815

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.0309 - accuracy: 0.5872

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0297 - accuracy: 0.5900

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0328 - accuracy: 0.5889

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0300 - accuracy: 0.5903

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0332 - accuracy: 0.5882

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0353 - accuracy: 0.5851

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0291 - accuracy: 0.5884

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0326 - accuracy: 0.5896

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0306 - accuracy: 0.5878

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0377 - accuracy: 0.5815

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0338 - accuracy: 0.5809

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0423 - accuracy: 0.5787

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0375 - accuracy: 0.5765

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0400 - accuracy: 0.5753

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0431 - accuracy: 0.5726

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 1.0490 - accuracy: 0.5700

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0542 - accuracy: 0.5667

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0540 - accuracy: 0.5644

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0505 - accuracy: 0.5665

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0472 - accuracy: 0.5686

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0489 - accuracy: 0.5663

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0520 - accuracy: 0.5635

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0544 - accuracy: 0.5628

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0526 - accuracy: 0.5648

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0490 - accuracy: 0.5679

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0510 - accuracy: 0.5672

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0496 - accuracy: 0.5690

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0502 - accuracy: 0.5688

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0459 - accuracy: 0.5723

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0447 - accuracy: 0.5727

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0477 - accuracy: 0.5719

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0468 - accuracy: 0.5717

.. parsed-literal::

    
57/92 [=================>............] - ETA: 1s - loss: 1.0450 - accuracy: 0.5727

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0415 - accuracy: 0.5747

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0416 - accuracy: 0.5766

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0441 - accuracy: 0.5769

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0425 - accuracy: 0.5782

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0509 - accuracy: 0.5734

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0480 - accuracy: 0.5737

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0461 - accuracy: 0.5740

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0461 - accuracy: 0.5743

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0491 - accuracy: 0.5737

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0470 - accuracy: 0.5763

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0479 - accuracy: 0.5775

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0493 - accuracy: 0.5782

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0476 - accuracy: 0.5793

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0459 - accuracy: 0.5799

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0432 - accuracy: 0.5814

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0417 - accuracy: 0.5838

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0415 - accuracy: 0.5835

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0380 - accuracy: 0.5853

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0395 - accuracy: 0.5837

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0398 - accuracy: 0.5851

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0395 - accuracy: 0.5844

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0376 - accuracy: 0.5857

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0397 - accuracy: 0.5850

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0377 - accuracy: 0.5859

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0396 - accuracy: 0.5849

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0397 - accuracy: 0.5846

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0402 - accuracy: 0.5840

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0409 - accuracy: 0.5844

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0408 - accuracy: 0.5842

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0423 - accuracy: 0.5829

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0401 - accuracy: 0.5848

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0399 - accuracy: 0.5842

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0382 - accuracy: 0.5843

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0383 - accuracy: 0.5844

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0404 - accuracy: 0.5831

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 1.0404 - accuracy: 0.5831 - val_loss: 0.9702 - val_accuracy: 0.6022


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.8891 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9463 - accuracy: 0.6250

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9708 - accuracy: 0.6250

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9735 - accuracy: 0.6172

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9685 - accuracy: 0.6187

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9682 - accuracy: 0.6094

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9516 - accuracy: 0.6161

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9563 - accuracy: 0.6172

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9450 - accuracy: 0.6319

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9262 - accuracy: 0.6438

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9211 - accuracy: 0.6449

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9142 - accuracy: 0.6406

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9165 - accuracy: 0.6370

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9048 - accuracy: 0.6339

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9038 - accuracy: 0.6313

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9017 - accuracy: 0.6270

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8992 - accuracy: 0.6287

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9111 - accuracy: 0.6337

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9110 - accuracy: 0.6266

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9194 - accuracy: 0.6219

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9145 - accuracy: 0.6235

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9173 - accuracy: 0.6264

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.9168 - accuracy: 0.6250

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9182 - accuracy: 0.6276

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9118 - accuracy: 0.6300

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9157 - accuracy: 0.6298

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9153 - accuracy: 0.6273

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9143 - accuracy: 0.6272

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9265 - accuracy: 0.6228

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9302 - accuracy: 0.6229

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9346 - accuracy: 0.6190

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9377 - accuracy: 0.6152

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9342 - accuracy: 0.6155

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9301 - accuracy: 0.6186

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9291 - accuracy: 0.6170

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9232 - accuracy: 0.6207

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9293 - accuracy: 0.6199

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9291 - accuracy: 0.6209

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9310 - accuracy: 0.6218

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.9339 - accuracy: 0.6234

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9381 - accuracy: 0.6181

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9407 - accuracy: 0.6168

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9371 - accuracy: 0.6177

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9366 - accuracy: 0.6193

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9358 - accuracy: 0.6215

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9386 - accuracy: 0.6196

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9326 - accuracy: 0.6237

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9327 - accuracy: 0.6243

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9326 - accuracy: 0.6256

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9340 - accuracy: 0.6256

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9349 - accuracy: 0.6250

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9345 - accuracy: 0.6256

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9315 - accuracy: 0.6256

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9361 - accuracy: 0.6233

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9346 - accuracy: 0.6233

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9337 - accuracy: 0.6222

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9347 - accuracy: 0.6228

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9376 - accuracy: 0.6218

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9377 - accuracy: 0.6224

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9418 - accuracy: 0.6214

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9412 - accuracy: 0.6230

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9394 - accuracy: 0.6240

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9384 - accuracy: 0.6240

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9360 - accuracy: 0.6260

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9364 - accuracy: 0.6255

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9355 - accuracy: 0.6255

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9317 - accuracy: 0.6273

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9301 - accuracy: 0.6278

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9362 - accuracy: 0.6259

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9357 - accuracy: 0.6259

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9363 - accuracy: 0.6263

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9383 - accuracy: 0.6263

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9383 - accuracy: 0.6271

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9368 - accuracy: 0.6280

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9346 - accuracy: 0.6296

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9351 - accuracy: 0.6308

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9344 - accuracy: 0.6315

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9329 - accuracy: 0.6334

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9330 - accuracy: 0.6337

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9300 - accuracy: 0.6344

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9256 - accuracy: 0.6370

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9275 - accuracy: 0.6361

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9257 - accuracy: 0.6370

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9246 - accuracy: 0.6380

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9231 - accuracy: 0.6390

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9243 - accuracy: 0.6380

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9229 - accuracy: 0.6378

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9240 - accuracy: 0.6359

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9216 - accuracy: 0.6365

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9234 - accuracy: 0.6357

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9244 - accuracy: 0.6352

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.9244 - accuracy: 0.6352 - val_loss: 0.9094 - val_accuracy: 0.6444


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 6s - loss: 0.9623 - accuracy: 0.5938

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9813 - accuracy: 0.5938

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9152 - accuracy: 0.6250

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9429 - accuracy: 0.6250

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9188 - accuracy: 0.6313

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9344 - accuracy: 0.6302

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9240 - accuracy: 0.6339

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9090 - accuracy: 0.6406

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9063 - accuracy: 0.6424

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9191 - accuracy: 0.6375

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9276 - accuracy: 0.6506

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9142 - accuracy: 0.6536

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8880 - accuracy: 0.6614

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8812 - accuracy: 0.6653

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8872 - accuracy: 0.6627

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8911 - accuracy: 0.6604

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8959 - accuracy: 0.6602

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8909 - accuracy: 0.6650

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8860 - accuracy: 0.6661

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8813 - accuracy: 0.6687

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.8709 - accuracy: 0.6724

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8635 - accuracy: 0.6717

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8620 - accuracy: 0.6671

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8603 - accuracy: 0.6692

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8620 - accuracy: 0.6711

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8589 - accuracy: 0.6717

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8676 - accuracy: 0.6689

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8728 - accuracy: 0.6696

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8720 - accuracy: 0.6681

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8772 - accuracy: 0.6687

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8791 - accuracy: 0.6673

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8838 - accuracy: 0.6670

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8856 - accuracy: 0.6657

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8890 - accuracy: 0.6646

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8890 - accuracy: 0.6652

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8923 - accuracy: 0.6633

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8885 - accuracy: 0.6656

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8882 - accuracy: 0.6653

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8845 - accuracy: 0.6682

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8921 - accuracy: 0.6626

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8919 - accuracy: 0.6632

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8900 - accuracy: 0.6623

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8865 - accuracy: 0.6629

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8829 - accuracy: 0.6655

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8817 - accuracy: 0.6660

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8809 - accuracy: 0.6651

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8786 - accuracy: 0.6662

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8792 - accuracy: 0.6660

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8785 - accuracy: 0.6665

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8761 - accuracy: 0.6687

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8801 - accuracy: 0.6655

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8783 - accuracy: 0.6659

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8819 - accuracy: 0.6657

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8835 - accuracy: 0.6667

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8890 - accuracy: 0.6631

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8831 - accuracy: 0.6635

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8822 - accuracy: 0.6629

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8803 - accuracy: 0.6638

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8830 - accuracy: 0.6621

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8874 - accuracy: 0.6595

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8842 - accuracy: 0.6609

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8852 - accuracy: 0.6609

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8844 - accuracy: 0.6608

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8856 - accuracy: 0.6597

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8844 - accuracy: 0.6606

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8858 - accuracy: 0.6606

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8850 - accuracy: 0.6605

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8823 - accuracy: 0.6609

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8843 - accuracy: 0.6595

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8827 - accuracy: 0.6590

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8817 - accuracy: 0.6594

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8818 - accuracy: 0.6594

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8783 - accuracy: 0.6606

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8765 - accuracy: 0.6614

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8759 - accuracy: 0.6625

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8743 - accuracy: 0.6641

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8740 - accuracy: 0.6640

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8720 - accuracy: 0.6651

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8678 - accuracy: 0.6665

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8681 - accuracy: 0.6672

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8718 - accuracy: 0.6663

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8740 - accuracy: 0.6654

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8751 - accuracy: 0.6653

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8736 - accuracy: 0.6670

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8739 - accuracy: 0.6665

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8714 - accuracy: 0.6679

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8696 - accuracy: 0.6695

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8698 - accuracy: 0.6704

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8745 - accuracy: 0.6682

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8724 - accuracy: 0.6691

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8739 - accuracy: 0.6676

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8739 - accuracy: 0.6676 - val_loss: 0.8190 - val_accuracy: 0.6921


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.9012 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8187 - accuracy: 0.6875

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8477 - accuracy: 0.6667

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8356 - accuracy: 0.6562

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7930 - accuracy: 0.6750

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7877 - accuracy: 0.6823

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7903 - accuracy: 0.6830

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7786 - accuracy: 0.6953

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7698 - accuracy: 0.6910

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7601 - accuracy: 0.6938

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7734 - accuracy: 0.6960

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7838 - accuracy: 0.6953

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8038 - accuracy: 0.6851

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8216 - accuracy: 0.6830

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8214 - accuracy: 0.6833

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8463 - accuracy: 0.6719

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8480 - accuracy: 0.6691

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8329 - accuracy: 0.6736

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8349 - accuracy: 0.6694

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8434 - accuracy: 0.6656

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8485 - accuracy: 0.6607

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8649 - accuracy: 0.6534

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8595 - accuracy: 0.6576

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8583 - accuracy: 0.6562

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8528 - accuracy: 0.6612

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8628 - accuracy: 0.6554

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8630 - accuracy: 0.6577

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8644 - accuracy: 0.6576

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8576 - accuracy: 0.6618

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8531 - accuracy: 0.6616

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8452 - accuracy: 0.6663

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8495 - accuracy: 0.6641

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8478 - accuracy: 0.6639

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8410 - accuracy: 0.6664

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8355 - accuracy: 0.6696

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8394 - accuracy: 0.6667

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8371 - accuracy: 0.6689

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8350 - accuracy: 0.6694

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8438 - accuracy: 0.6651

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8449 - accuracy: 0.6641

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8426 - accuracy: 0.6662

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8414 - accuracy: 0.6659

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8466 - accuracy: 0.6636

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8407 - accuracy: 0.6683

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8401 - accuracy: 0.6680

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8395 - accuracy: 0.6684

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8403 - accuracy: 0.6688

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8398 - accuracy: 0.6686

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8358 - accuracy: 0.6715

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8379 - accuracy: 0.6712

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8353 - accuracy: 0.6721

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8360 - accuracy: 0.6748

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8331 - accuracy: 0.6762

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8312 - accuracy: 0.6781

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8323 - accuracy: 0.6788

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8327 - accuracy: 0.6784

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8307 - accuracy: 0.6797

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8286 - accuracy: 0.6798

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8270 - accuracy: 0.6810

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8241 - accuracy: 0.6821

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8277 - accuracy: 0.6827

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8273 - accuracy: 0.6823

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8249 - accuracy: 0.6824

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8218 - accuracy: 0.6824

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8213 - accuracy: 0.6825

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8191 - accuracy: 0.6826

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8175 - accuracy: 0.6822

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8196 - accuracy: 0.6823

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8198 - accuracy: 0.6837

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8196 - accuracy: 0.6842

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8183 - accuracy: 0.6847

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8171 - accuracy: 0.6851

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8183 - accuracy: 0.6847

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8156 - accuracy: 0.6865

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8158 - accuracy: 0.6877

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8156 - accuracy: 0.6877

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8131 - accuracy: 0.6881

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8125 - accuracy: 0.6885

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8111 - accuracy: 0.6889

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8109 - accuracy: 0.6889

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8083 - accuracy: 0.6904

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8053 - accuracy: 0.6922

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8080 - accuracy: 0.6914

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8063 - accuracy: 0.6925

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8067 - accuracy: 0.6921

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8076 - accuracy: 0.6916

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8048 - accuracy: 0.6927

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8041 - accuracy: 0.6930

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8049 - accuracy: 0.6925

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8064 - accuracy: 0.6911

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8048 - accuracy: 0.6921

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8048 - accuracy: 0.6921 - val_loss: 0.8279 - val_accuracy: 0.6812


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5554 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6369 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7214 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7274 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6952 - accuracy: 0.7625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6629 - accuracy: 0.7708

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7300 - accuracy: 0.7366

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7128 - accuracy: 0.7383

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7146 - accuracy: 0.7396

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7207 - accuracy: 0.7344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7376 - accuracy: 0.7273

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7338 - accuracy: 0.7344

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7416 - accuracy: 0.7260

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7509 - accuracy: 0.7210

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7665 - accuracy: 0.7146

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7636 - accuracy: 0.7168

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7706 - accuracy: 0.7188

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7682 - accuracy: 0.7170

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7605 - accuracy: 0.7171

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7636 - accuracy: 0.7156

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7588 - accuracy: 0.7188

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.7624 - accuracy: 0.7173

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7674 - accuracy: 0.7106

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7591 - accuracy: 0.7122

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7591 - accuracy: 0.7113

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7596 - accuracy: 0.7091

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7656 - accuracy: 0.7060

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7627 - accuracy: 0.7087

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7567 - accuracy: 0.7080

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7625 - accuracy: 0.7042

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7770 - accuracy: 0.6966

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7765 - accuracy: 0.6963

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7744 - accuracy: 0.6960

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7720 - accuracy: 0.6985

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7720 - accuracy: 0.6991

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7739 - accuracy: 0.6962

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7767 - accuracy: 0.6951

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7817 - accuracy: 0.6965

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7774 - accuracy: 0.6979

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.7742 - accuracy: 0.6984

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7710 - accuracy: 0.7005

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7695 - accuracy: 0.7009

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7712 - accuracy: 0.7020

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7715 - accuracy: 0.7010

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7726 - accuracy: 0.7014

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7722 - accuracy: 0.7024

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7709 - accuracy: 0.7021

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7695 - accuracy: 0.7025

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7689 - accuracy: 0.7009

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7657 - accuracy: 0.7038

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7649 - accuracy: 0.7040

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7694 - accuracy: 0.7031

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7705 - accuracy: 0.7028

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7686 - accuracy: 0.7031

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7668 - accuracy: 0.7045

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7637 - accuracy: 0.7059

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7589 - accuracy: 0.7072

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7573 - accuracy: 0.7074

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7567 - accuracy: 0.7071

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7546 - accuracy: 0.7078

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7587 - accuracy: 0.7054

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7588 - accuracy: 0.7072

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7569 - accuracy: 0.7073

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7544 - accuracy: 0.7080

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7526 - accuracy: 0.7087

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7573 - accuracy: 0.7079

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7572 - accuracy: 0.7076

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7565 - accuracy: 0.7082

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7545 - accuracy: 0.7083

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7546 - accuracy: 0.7089

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7542 - accuracy: 0.7086

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7557 - accuracy: 0.7083

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7548 - accuracy: 0.7093

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7557 - accuracy: 0.7086

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7608 - accuracy: 0.7063

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7620 - accuracy: 0.7056

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7610 - accuracy: 0.7058

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7580 - accuracy: 0.7063

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7591 - accuracy: 0.7057

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7607 - accuracy: 0.7039

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7605 - accuracy: 0.7049

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7574 - accuracy: 0.7058

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7576 - accuracy: 0.7067

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7593 - accuracy: 0.7061

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7597 - accuracy: 0.7059

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7616 - accuracy: 0.7046

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7618 - accuracy: 0.7044

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7633 - accuracy: 0.7046

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7631 - accuracy: 0.7037

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7657 - accuracy: 0.7014

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7638 - accuracy: 0.7030

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7638 - accuracy: 0.7030 - val_loss: 0.7524 - val_accuracy: 0.7016


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.8011 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8125 - accuracy: 0.6406

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7809 - accuracy: 0.6875

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7733 - accuracy: 0.6875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7514 - accuracy: 0.6875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7662 - accuracy: 0.6875

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8026 - accuracy: 0.6786

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7884 - accuracy: 0.6758

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7983 - accuracy: 0.6701

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7878 - accuracy: 0.6781

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7829 - accuracy: 0.6790

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7795 - accuracy: 0.6849

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7708 - accuracy: 0.6875

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7570 - accuracy: 0.6920

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7563 - accuracy: 0.6979

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7561 - accuracy: 0.7031

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7413 - accuracy: 0.7077

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7249 - accuracy: 0.7153

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7190 - accuracy: 0.7155

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7198 - accuracy: 0.7172

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7254 - accuracy: 0.7173

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7290 - accuracy: 0.7188

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7351 - accuracy: 0.7147

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7360 - accuracy: 0.7122

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7370 - accuracy: 0.7100

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7339 - accuracy: 0.7103

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7359 - accuracy: 0.7072

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7362 - accuracy: 0.7065

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7486 - accuracy: 0.7026

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7457 - accuracy: 0.7021

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7417 - accuracy: 0.7056

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7389 - accuracy: 0.7061

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7376 - accuracy: 0.7045

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7362 - accuracy: 0.7068

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7412 - accuracy: 0.7080

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7365 - accuracy: 0.7118

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7311 - accuracy: 0.7154

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7276 - accuracy: 0.7171

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7252 - accuracy: 0.7196

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.7232 - accuracy: 0.7219

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7200 - accuracy: 0.7241

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7193 - accuracy: 0.7247

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7207 - accuracy: 0.7253

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7236 - accuracy: 0.7244

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7211 - accuracy: 0.7257

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7160 - accuracy: 0.7283

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7121 - accuracy: 0.7301

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7147 - accuracy: 0.7298

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7156 - accuracy: 0.7283

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7141 - accuracy: 0.7275

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7190 - accuracy: 0.7255

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7123 - accuracy: 0.7290

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7130 - accuracy: 0.7276

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7126 - accuracy: 0.7280

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7100 - accuracy: 0.7295

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7106 - accuracy: 0.7294

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7154 - accuracy: 0.7270

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7160 - accuracy: 0.7274

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7185 - accuracy: 0.7256

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7208 - accuracy: 0.7255

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7189 - accuracy: 0.7259

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7210 - accuracy: 0.7248

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7176 - accuracy: 0.7272

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7148 - accuracy: 0.7285

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7210 - accuracy: 0.7264

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7213 - accuracy: 0.7268

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7233 - accuracy: 0.7262

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7220 - accuracy: 0.7270

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7199 - accuracy: 0.7287

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7192 - accuracy: 0.7281

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7201 - accuracy: 0.7276

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7205 - accuracy: 0.7283

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7217 - accuracy: 0.7282

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7247 - accuracy: 0.7276

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7220 - accuracy: 0.7292

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7206 - accuracy: 0.7290

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7200 - accuracy: 0.7281

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7183 - accuracy: 0.7284

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7169 - accuracy: 0.7298

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7149 - accuracy: 0.7305

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7151 - accuracy: 0.7303

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7145 - accuracy: 0.7306

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7187 - accuracy: 0.7278

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7183 - accuracy: 0.7281

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7175 - accuracy: 0.7268

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7178 - accuracy: 0.7264

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7156 - accuracy: 0.7270

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7145 - accuracy: 0.7269

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7157 - accuracy: 0.7267

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7179 - accuracy: 0.7262

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7204 - accuracy: 0.7262

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7204 - accuracy: 0.7262 - val_loss: 0.8592 - val_accuracy: 0.6880


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.7000 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7225 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7459 - accuracy: 0.7188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7055 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6559 - accuracy: 0.7563

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6284 - accuracy: 0.7760

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6690 - accuracy: 0.7500

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6676 - accuracy: 0.7461

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6470 - accuracy: 0.7569

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6676 - accuracy: 0.7469

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6829 - accuracy: 0.7415

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6788 - accuracy: 0.7422

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6759 - accuracy: 0.7380

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6680 - accuracy: 0.7388

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6670 - accuracy: 0.7396

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6659 - accuracy: 0.7441

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6859 - accuracy: 0.7335

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6799 - accuracy: 0.7344

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6810 - accuracy: 0.7385

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6772 - accuracy: 0.7422

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6764 - accuracy: 0.7426

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6823 - accuracy: 0.7386

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6882 - accuracy: 0.7391

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6935 - accuracy: 0.7331

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6892 - accuracy: 0.7350

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6893 - accuracy: 0.7344

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6825 - accuracy: 0.7373

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6820 - accuracy: 0.7355

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6784 - accuracy: 0.7371

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6742 - accuracy: 0.7375

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6766 - accuracy: 0.7369

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6741 - accuracy: 0.7354

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6839 - accuracy: 0.7330

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6865 - accuracy: 0.7289

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6802 - accuracy: 0.7312

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6772 - accuracy: 0.7326

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6824 - accuracy: 0.7314

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6840 - accuracy: 0.7327

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6880 - accuracy: 0.7308

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6925 - accuracy: 0.7320

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6917 - accuracy: 0.7340

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6916 - accuracy: 0.7329

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6912 - accuracy: 0.7340

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6904 - accuracy: 0.7344

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6887 - accuracy: 0.7354

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6901 - accuracy: 0.7344

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6870 - accuracy: 0.7347

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6881 - accuracy: 0.7350

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6823 - accuracy: 0.7372

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6816 - accuracy: 0.7369

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6794 - accuracy: 0.7384

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6841 - accuracy: 0.7344

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6815 - accuracy: 0.7347

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6799 - accuracy: 0.7355

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6782 - accuracy: 0.7358

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6770 - accuracy: 0.7366

.. parsed-literal::

    
57/92 [=================>............] - ETA: 1s - loss: 0.6777 - accuracy: 0.7363

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6806 - accuracy: 0.7333

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6777 - accuracy: 0.7346

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6737 - accuracy: 0.7359

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6761 - accuracy: 0.7357

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6801 - accuracy: 0.7339

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6840 - accuracy: 0.7321

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6853 - accuracy: 0.7329

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6841 - accuracy: 0.7322

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6876 - accuracy: 0.7306

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6878 - accuracy: 0.7309

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6854 - accuracy: 0.7312

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6881 - accuracy: 0.7287

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6891 - accuracy: 0.7281

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6865 - accuracy: 0.7289

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6914 - accuracy: 0.7270

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6911 - accuracy: 0.7265

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6924 - accuracy: 0.7259

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6951 - accuracy: 0.7250

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7003 - accuracy: 0.7233

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7001 - accuracy: 0.7240

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6992 - accuracy: 0.7248

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7006 - accuracy: 0.7243

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6989 - accuracy: 0.7250

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6997 - accuracy: 0.7249

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7012 - accuracy: 0.7241

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7011 - accuracy: 0.7240

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7007 - accuracy: 0.7240

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7022 - accuracy: 0.7243

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7008 - accuracy: 0.7253

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7035 - accuracy: 0.7251

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7032 - accuracy: 0.7261

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7014 - accuracy: 0.7267

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7018 - accuracy: 0.7262

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7026 - accuracy: 0.7262

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7026 - accuracy: 0.7262 - val_loss: 0.7449 - val_accuracy: 0.7044


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6527 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5894 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5417 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6002 - accuracy: 0.7500

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6160 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6129 - accuracy: 0.7396

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6263 - accuracy: 0.7366

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6326 - accuracy: 0.7305

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6175 - accuracy: 0.7465

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5959 - accuracy: 0.7563

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5944 - accuracy: 0.7585

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6152 - accuracy: 0.7474

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5959 - accuracy: 0.7596

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5987 - accuracy: 0.7567

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6023 - accuracy: 0.7563

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6035 - accuracy: 0.7539

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6026 - accuracy: 0.7555

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6049 - accuracy: 0.7535

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6015 - accuracy: 0.7549

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6023 - accuracy: 0.7594

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5992 - accuracy: 0.7634

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5958 - accuracy: 0.7628

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5974 - accuracy: 0.7622

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5937 - accuracy: 0.7656

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6014 - accuracy: 0.7638

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6177 - accuracy: 0.7548

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6207 - accuracy: 0.7523

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6252 - accuracy: 0.7522

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6156 - accuracy: 0.7597

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6240 - accuracy: 0.7573

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6236 - accuracy: 0.7571

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6261 - accuracy: 0.7559

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6245 - accuracy: 0.7557

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6307 - accuracy: 0.7528

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6275 - accuracy: 0.7536

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6260 - accuracy: 0.7543

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6405 - accuracy: 0.7483

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6432 - accuracy: 0.7500

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6402 - accuracy: 0.7492

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6390 - accuracy: 0.7508

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6360 - accuracy: 0.7508

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6320 - accuracy: 0.7530

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6331 - accuracy: 0.7536

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6320 - accuracy: 0.7543

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6338 - accuracy: 0.7542

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6368 - accuracy: 0.7527

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6379 - accuracy: 0.7533

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6417 - accuracy: 0.7520

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6385 - accuracy: 0.7532

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6382 - accuracy: 0.7538

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6403 - accuracy: 0.7543

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6374 - accuracy: 0.7548

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6383 - accuracy: 0.7547

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6428 - accuracy: 0.7535

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6431 - accuracy: 0.7528

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6434 - accuracy: 0.7522

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6418 - accuracy: 0.7544

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6428 - accuracy: 0.7532

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6422 - accuracy: 0.7537

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6395 - accuracy: 0.7557

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6393 - accuracy: 0.7561

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6384 - accuracy: 0.7566

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6349 - accuracy: 0.7578

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6335 - accuracy: 0.7587

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6352 - accuracy: 0.7581

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6359 - accuracy: 0.7584

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6378 - accuracy: 0.7578

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6365 - accuracy: 0.7577

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6366 - accuracy: 0.7572

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6406 - accuracy: 0.7549

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6402 - accuracy: 0.7561

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6396 - accuracy: 0.7569

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6405 - accuracy: 0.7576

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6425 - accuracy: 0.7567

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6428 - accuracy: 0.7570

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6440 - accuracy: 0.7573

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6449 - accuracy: 0.7560

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6468 - accuracy: 0.7552

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6466 - accuracy: 0.7551

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6473 - accuracy: 0.7554

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6469 - accuracy: 0.7557

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6490 - accuracy: 0.7542

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6479 - accuracy: 0.7549

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6503 - accuracy: 0.7537

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6521 - accuracy: 0.7533

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6519 - accuracy: 0.7529

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6556 - accuracy: 0.7518

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6574 - accuracy: 0.7507

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6583 - accuracy: 0.7490

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6588 - accuracy: 0.7493

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6584 - accuracy: 0.7493

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6584 - accuracy: 0.7493 - val_loss: 0.7375 - val_accuracy: 0.7003


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 5s - loss: 0.4908 - accuracy: 0.8333

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7146 - accuracy: 0.7321

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6723 - accuracy: 0.7273

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6717 - accuracy: 0.7167

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6913 - accuracy: 0.7105

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6604 - accuracy: 0.7283

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6625 - accuracy: 0.7361

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6895 - accuracy: 0.7379

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6902 - accuracy: 0.7286

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6734 - accuracy: 0.7372

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6809 - accuracy: 0.7355

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6716 - accuracy: 0.7394

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6632 - accuracy: 0.7426

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6555 - accuracy: 0.7455

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6418 - accuracy: 0.7500

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6410 - accuracy: 0.7460

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6405 - accuracy: 0.7463

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6426 - accuracy: 0.7465

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6549 - accuracy: 0.7383

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6601 - accuracy: 0.7389

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6696 - accuracy: 0.7334

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6661 - accuracy: 0.7342

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6599 - accuracy: 0.7363

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6517 - accuracy: 0.7421

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6497 - accuracy: 0.7449

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6478 - accuracy: 0.7439

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6482 - accuracy: 0.7430

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6446 - accuracy: 0.7432

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6579 - accuracy: 0.7348

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6561 - accuracy: 0.7363

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6565 - accuracy: 0.7368

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6541 - accuracy: 0.7382

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6518 - accuracy: 0.7405

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6468 - accuracy: 0.7435

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6487 - accuracy: 0.7437

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6449 - accuracy: 0.7439

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6501 - accuracy: 0.7415

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6518 - accuracy: 0.7384

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6550 - accuracy: 0.7387

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6526 - accuracy: 0.7406

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6512 - accuracy: 0.7408

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6487 - accuracy: 0.7425

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6438 - accuracy: 0.7456

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6454 - accuracy: 0.7436

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6512 - accuracy: 0.7409

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6444 - accuracy: 0.7439

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6471 - accuracy: 0.7420

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6444 - accuracy: 0.7435

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6401 - accuracy: 0.7468

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6398 - accuracy: 0.7481

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6409 - accuracy: 0.7463

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6371 - accuracy: 0.7476

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6380 - accuracy: 0.7476

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6382 - accuracy: 0.7477

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6421 - accuracy: 0.7454

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6426 - accuracy: 0.7461

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6425 - accuracy: 0.7461

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6460 - accuracy: 0.7457

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6462 - accuracy: 0.7452

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6437 - accuracy: 0.7453

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6444 - accuracy: 0.7459

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6466 - accuracy: 0.7434

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6521 - accuracy: 0.7415

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6520 - accuracy: 0.7402

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6510 - accuracy: 0.7403

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6555 - accuracy: 0.7386

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6517 - accuracy: 0.7406

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6487 - accuracy: 0.7426

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6511 - accuracy: 0.7414

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6543 - accuracy: 0.7401

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6552 - accuracy: 0.7394

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6542 - accuracy: 0.7395

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6540 - accuracy: 0.7405

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6544 - accuracy: 0.7403

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6535 - accuracy: 0.7404

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6540 - accuracy: 0.7405

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6540 - accuracy: 0.7402

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6529 - accuracy: 0.7412

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6529 - accuracy: 0.7417

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6500 - accuracy: 0.7422

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6498 - accuracy: 0.7430

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6506 - accuracy: 0.7424

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6519 - accuracy: 0.7421

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6541 - accuracy: 0.7414

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6515 - accuracy: 0.7426

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6491 - accuracy: 0.7431

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6491 - accuracy: 0.7432

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6496 - accuracy: 0.7432

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6493 - accuracy: 0.7433

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6496 - accuracy: 0.7430

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6499 - accuracy: 0.7431

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6488 - accuracy: 0.7439

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6488 - accuracy: 0.7439 - val_loss: 0.7257 - val_accuracy: 0.7180


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.6773 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6881 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7145 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7175 - accuracy: 0.7344

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7213 - accuracy: 0.7174

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7067 - accuracy: 0.7083

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7200 - accuracy: 0.7016

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6916 - accuracy: 0.7143

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6723 - accuracy: 0.7212

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6574 - accuracy: 0.7267

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6506 - accuracy: 0.7340

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6444 - accuracy: 0.7402

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6329 - accuracy: 0.7455

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6196 - accuracy: 0.7479

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6066 - accuracy: 0.7560

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5988 - accuracy: 0.7575

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5874 - accuracy: 0.7623

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5793 - accuracy: 0.7650

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5708 - accuracy: 0.7658

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5638 - accuracy: 0.7666

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.5768 - accuracy: 0.7644

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5735 - accuracy: 0.7651

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5643 - accuracy: 0.7684

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5640 - accuracy: 0.7715

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5623 - accuracy: 0.7731

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5712 - accuracy: 0.7722

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5762 - accuracy: 0.7691

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5883 - accuracy: 0.7641

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5890 - accuracy: 0.7637

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5822 - accuracy: 0.7673

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5836 - accuracy: 0.7677

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5857 - accuracy: 0.7681

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5806 - accuracy: 0.7704

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5889 - accuracy: 0.7680

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5826 - accuracy: 0.7727

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5844 - accuracy: 0.7730

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5880 - accuracy: 0.7715

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5926 - accuracy: 0.7718

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5940 - accuracy: 0.7720

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5911 - accuracy: 0.7722

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5968 - accuracy: 0.7687

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5938 - accuracy: 0.7697

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5939 - accuracy: 0.7686

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5954 - accuracy: 0.7675

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5935 - accuracy: 0.7678

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5891 - accuracy: 0.7701

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5918 - accuracy: 0.7690

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5940 - accuracy: 0.7686

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5932 - accuracy: 0.7688

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5927 - accuracy: 0.7691

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5920 - accuracy: 0.7699

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5975 - accuracy: 0.7690

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5935 - accuracy: 0.7709

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5918 - accuracy: 0.7711

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5912 - accuracy: 0.7719

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5914 - accuracy: 0.7709

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5921 - accuracy: 0.7706

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5946 - accuracy: 0.7691

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5965 - accuracy: 0.7694

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5962 - accuracy: 0.7695

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5957 - accuracy: 0.7707

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5958 - accuracy: 0.7704

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6002 - accuracy: 0.7696

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6026 - accuracy: 0.7688

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6030 - accuracy: 0.7700

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6052 - accuracy: 0.7692

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6060 - accuracy: 0.7680

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6041 - accuracy: 0.7691

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6035 - accuracy: 0.7697

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6027 - accuracy: 0.7699

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6017 - accuracy: 0.7700

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5998 - accuracy: 0.7710

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5971 - accuracy: 0.7725

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5962 - accuracy: 0.7734

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5964 - accuracy: 0.7731

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5954 - accuracy: 0.7732

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5945 - accuracy: 0.7737

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5930 - accuracy: 0.7738

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5952 - accuracy: 0.7739

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5934 - accuracy: 0.7736

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5943 - accuracy: 0.7729

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5939 - accuracy: 0.7730

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5931 - accuracy: 0.7735

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5944 - accuracy: 0.7729

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5931 - accuracy: 0.7726

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5930 - accuracy: 0.7723

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5925 - accuracy: 0.7724

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5900 - accuracy: 0.7736

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5919 - accuracy: 0.7733

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5916 - accuracy: 0.7727

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5937 - accuracy: 0.7701

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5937 - accuracy: 0.7701 - val_loss: 0.6982 - val_accuracy: 0.7357


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5082 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6058 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6027 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5658 - accuracy: 0.7656

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5691 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5479 - accuracy: 0.7865

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5315 - accuracy: 0.7902

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5383 - accuracy: 0.7891

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5259 - accuracy: 0.7951

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5221 - accuracy: 0.8000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5177 - accuracy: 0.8068

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5077 - accuracy: 0.8099

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5239 - accuracy: 0.8029

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5387 - accuracy: 0.7969

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5358 - accuracy: 0.7979

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5442 - accuracy: 0.7949

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5503 - accuracy: 0.7923

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5431 - accuracy: 0.7951

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5506 - accuracy: 0.7928

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5566 - accuracy: 0.7875

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5636 - accuracy: 0.7857

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5531 - accuracy: 0.7898

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5495 - accuracy: 0.7894

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5480 - accuracy: 0.7917

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5473 - accuracy: 0.7937

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5473 - accuracy: 0.7969

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5443 - accuracy: 0.7963

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5386 - accuracy: 0.7991

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5434 - accuracy: 0.7963

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5448 - accuracy: 0.7958

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5469 - accuracy: 0.7964

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5572 - accuracy: 0.7920

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5558 - accuracy: 0.7917

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5659 - accuracy: 0.7858

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5618 - accuracy: 0.7866

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5623 - accuracy: 0.7847

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5663 - accuracy: 0.7838

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5752 - accuracy: 0.7812

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5753 - accuracy: 0.7812

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5749 - accuracy: 0.7820

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5718 - accuracy: 0.7828

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5708 - accuracy: 0.7827

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5769 - accuracy: 0.7805

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5787 - accuracy: 0.7798

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5826 - accuracy: 0.7785

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5868 - accuracy: 0.7761

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5833 - accuracy: 0.7768

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5859 - accuracy: 0.7763

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5895 - accuracy: 0.7732

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5880 - accuracy: 0.7734

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5913 - accuracy: 0.7717

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5941 - accuracy: 0.7707

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5910 - accuracy: 0.7721

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5890 - accuracy: 0.7734

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5902 - accuracy: 0.7741

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5938 - accuracy: 0.7715

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5941 - accuracy: 0.7700

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5948 - accuracy: 0.7691

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5950 - accuracy: 0.7683

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5915 - accuracy: 0.7701

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5948 - accuracy: 0.7682

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5935 - accuracy: 0.7699

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5923 - accuracy: 0.7706

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5903 - accuracy: 0.7708

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5924 - accuracy: 0.7704

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5967 - accuracy: 0.7687

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5961 - accuracy: 0.7698

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5980 - accuracy: 0.7700

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5979 - accuracy: 0.7702

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5967 - accuracy: 0.7708

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5970 - accuracy: 0.7713

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5946 - accuracy: 0.7719

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5968 - accuracy: 0.7695

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5950 - accuracy: 0.7705

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5915 - accuracy: 0.7719

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5888 - accuracy: 0.7728

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5916 - accuracy: 0.7713

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5915 - accuracy: 0.7718

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5912 - accuracy: 0.7712

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5901 - accuracy: 0.7724

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5891 - accuracy: 0.7729

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5889 - accuracy: 0.7727

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5892 - accuracy: 0.7724

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5884 - accuracy: 0.7729

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5895 - accuracy: 0.7722

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5895 - accuracy: 0.7723

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5889 - accuracy: 0.7724

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5878 - accuracy: 0.7725

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5870 - accuracy: 0.7716

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5904 - accuracy: 0.7693

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5881 - accuracy: 0.7701

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5881 - accuracy: 0.7701 - val_loss: 0.7623 - val_accuracy: 0.7112


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.3620 - accuracy: 0.9062

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5030 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4800 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4509 - accuracy: 0.8359

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4599 - accuracy: 0.8188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.4787 - accuracy: 0.8177

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4884 - accuracy: 0.8125

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5217 - accuracy: 0.8047

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5201 - accuracy: 0.8056

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5127 - accuracy: 0.8125

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5331 - accuracy: 0.7926

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5277 - accuracy: 0.7943

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5343 - accuracy: 0.7909

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5457 - accuracy: 0.7902

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5527 - accuracy: 0.7854

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5490 - accuracy: 0.7852

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5450 - accuracy: 0.7868

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5447 - accuracy: 0.7865

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5503 - accuracy: 0.7895

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5381 - accuracy: 0.7984

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5385 - accuracy: 0.7976

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5466 - accuracy: 0.7955

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5450 - accuracy: 0.7962

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5432 - accuracy: 0.7943

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5456 - accuracy: 0.7900

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5427 - accuracy: 0.7909

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5398 - accuracy: 0.7940

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5527 - accuracy: 0.7857

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5591 - accuracy: 0.7856

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5500 - accuracy: 0.7896

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5459 - accuracy: 0.7903

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5413 - accuracy: 0.7930

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5518 - accuracy: 0.7888

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5503 - accuracy: 0.7904

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5473 - accuracy: 0.7937

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5515 - accuracy: 0.7934

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5524 - accuracy: 0.7948

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5520 - accuracy: 0.7944

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5527 - accuracy: 0.7925

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5547 - accuracy: 0.7906

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5560 - accuracy: 0.7896

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5521 - accuracy: 0.7917

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5580 - accuracy: 0.7892

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5575 - accuracy: 0.7891

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5575 - accuracy: 0.7889

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5559 - accuracy: 0.7908

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5542 - accuracy: 0.7906

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5526 - accuracy: 0.7917

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5513 - accuracy: 0.7927

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5525 - accuracy: 0.7913

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5527 - accuracy: 0.7917

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5532 - accuracy: 0.7915

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5546 - accuracy: 0.7907

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5566 - accuracy: 0.7900

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5594 - accuracy: 0.7881

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5598 - accuracy: 0.7880

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5607 - accuracy: 0.7879

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5607 - accuracy: 0.7872

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5673 - accuracy: 0.7856

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5665 - accuracy: 0.7855

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5686 - accuracy: 0.7844

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5702 - accuracy: 0.7839

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5668 - accuracy: 0.7848

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5656 - accuracy: 0.7852

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5637 - accuracy: 0.7856

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5622 - accuracy: 0.7860

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5636 - accuracy: 0.7855

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5641 - accuracy: 0.7845

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5601 - accuracy: 0.7863

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5599 - accuracy: 0.7862

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5607 - accuracy: 0.7875

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5617 - accuracy: 0.7874

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5631 - accuracy: 0.7877

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5685 - accuracy: 0.7864

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5686 - accuracy: 0.7855

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5680 - accuracy: 0.7862

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5692 - accuracy: 0.7842

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5681 - accuracy: 0.7837

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5669 - accuracy: 0.7845

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5656 - accuracy: 0.7856

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5660 - accuracy: 0.7856

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5671 - accuracy: 0.7851

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5654 - accuracy: 0.7866

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5648 - accuracy: 0.7861

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5626 - accuracy: 0.7883

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5626 - accuracy: 0.7875

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5631 - accuracy: 0.7877

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5642 - accuracy: 0.7877

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5655 - accuracy: 0.7876

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5646 - accuracy: 0.7879

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5660 - accuracy: 0.7868

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5660 - accuracy: 0.7868 - val_loss: 0.7577 - val_accuracy: 0.7180


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 6s - loss: 0.3940 - accuracy: 0.9375

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5227 - accuracy: 0.8281

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5732 - accuracy: 0.8021

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.5119 - accuracy: 0.8158

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5109 - accuracy: 0.8261

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4735 - accuracy: 0.8426

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4546 - accuracy: 0.8427

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4669 - accuracy: 0.8429

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4700 - accuracy: 0.8397

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4525 - accuracy: 0.8459

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4632 - accuracy: 0.8404

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4887 - accuracy: 0.8260

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4810 - accuracy: 0.8295

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4754 - accuracy: 0.8347

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4907 - accuracy: 0.8313

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4990 - accuracy: 0.8284

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4877 - accuracy: 0.8345

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4982 - accuracy: 0.8283

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4973 - accuracy: 0.8259

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5019 - accuracy: 0.8223

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.5000 - accuracy: 0.8190

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5115 - accuracy: 0.8146

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5115 - accuracy: 0.8079

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5078 - accuracy: 0.8093

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5052 - accuracy: 0.8107

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5042 - accuracy: 0.8119

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4983 - accuracy: 0.8153

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5003 - accuracy: 0.8130

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4996 - accuracy: 0.8141

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.4970 - accuracy: 0.8150

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4920 - accuracy: 0.8159

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4941 - accuracy: 0.8177

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4853 - accuracy: 0.8231

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.4863 - accuracy: 0.8228

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.4888 - accuracy: 0.8208

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4996 - accuracy: 0.8172

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5032 - accuracy: 0.8162

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5022 - accuracy: 0.8169

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5068 - accuracy: 0.8160

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5068 - accuracy: 0.8167

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5058 - accuracy: 0.8181

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5118 - accuracy: 0.8158

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5143 - accuracy: 0.8136

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5139 - accuracy: 0.8142

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5158 - accuracy: 0.8128

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5223 - accuracy: 0.8115

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5244 - accuracy: 0.8102

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5252 - accuracy: 0.8090

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5292 - accuracy: 0.8078

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5246 - accuracy: 0.8091

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5229 - accuracy: 0.8098

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5222 - accuracy: 0.8098

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5224 - accuracy: 0.8081

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5195 - accuracy: 0.8094

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5188 - accuracy: 0.8089

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5225 - accuracy: 0.8084

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5206 - accuracy: 0.8090

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5231 - accuracy: 0.8064

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5219 - accuracy: 0.8060

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5213 - accuracy: 0.8061

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5200 - accuracy: 0.8067

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5185 - accuracy: 0.8068

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5184 - accuracy: 0.8064

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5236 - accuracy: 0.8050

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5260 - accuracy: 0.8056

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5234 - accuracy: 0.8066

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5259 - accuracy: 0.8054

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5264 - accuracy: 0.8050

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5260 - accuracy: 0.8047

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5235 - accuracy: 0.8057

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5253 - accuracy: 0.8040

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5280 - accuracy: 0.8024

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5278 - accuracy: 0.8030

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5283 - accuracy: 0.8018

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5306 - accuracy: 0.8003

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5323 - accuracy: 0.7993

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5336 - accuracy: 0.7990

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5357 - accuracy: 0.7984

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5363 - accuracy: 0.7982

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5382 - accuracy: 0.7972

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5402 - accuracy: 0.7959

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5401 - accuracy: 0.7965

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5402 - accuracy: 0.7959

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5395 - accuracy: 0.7968

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5387 - accuracy: 0.7970

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5377 - accuracy: 0.7979

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5366 - accuracy: 0.7977

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5359 - accuracy: 0.7979

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5379 - accuracy: 0.7974

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5396 - accuracy: 0.7972

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5377 - accuracy: 0.7977

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5377 - accuracy: 0.7977 - val_loss: 0.7492 - val_accuracy: 0.7125


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::

    
 1/92 [..............................] - ETA: 7s - loss: 0.5375 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5815 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5542 - accuracy: 0.8229

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6554 - accuracy: 0.7656

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6445 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6618 - accuracy: 0.7500

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6351 - accuracy: 0.7679

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6186 - accuracy: 0.7695

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6197 - accuracy: 0.7708

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6079 - accuracy: 0.7781

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6152 - accuracy: 0.7699

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6088 - accuracy: 0.7734

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5991 - accuracy: 0.7716

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5937 - accuracy: 0.7790

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5880 - accuracy: 0.7792

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5797 - accuracy: 0.7832

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5821 - accuracy: 0.7794

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5884 - accuracy: 0.7812

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5874 - accuracy: 0.7780

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5843 - accuracy: 0.7797

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5751 - accuracy: 0.7827

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5660 - accuracy: 0.7841

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5576 - accuracy: 0.7867

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5562 - accuracy: 0.7878

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5507 - accuracy: 0.7912

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5478 - accuracy: 0.7933

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5477 - accuracy: 0.7928

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5465 - accuracy: 0.7924

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5397 - accuracy: 0.7953

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5333 - accuracy: 0.7969

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5341 - accuracy: 0.7964

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5357 - accuracy: 0.7959

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5360 - accuracy: 0.7963

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5411 - accuracy: 0.7950

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5434 - accuracy: 0.7937

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5429 - accuracy: 0.7942

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5458 - accuracy: 0.7947

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5478 - accuracy: 0.7935

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5465 - accuracy: 0.7956

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5449 - accuracy: 0.7960

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5439 - accuracy: 0.7972

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5411 - accuracy: 0.7982

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5395 - accuracy: 0.8000

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5437 - accuracy: 0.7982

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5391 - accuracy: 0.8005

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5351 - accuracy: 0.8021

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5375 - accuracy: 0.8030

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5342 - accuracy: 0.8045

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5362 - accuracy: 0.8034

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5360 - accuracy: 0.8023

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5359 - accuracy: 0.8025

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5313 - accuracy: 0.8045

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5374 - accuracy: 0.8017

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5354 - accuracy: 0.8025

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5318 - accuracy: 0.8038

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5274 - accuracy: 0.8056

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5261 - accuracy: 0.8063

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5259 - accuracy: 0.8059

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5271 - accuracy: 0.8049

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5252 - accuracy: 0.8066

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5269 - accuracy: 0.8067

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5255 - accuracy: 0.8078

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5237 - accuracy: 0.8083

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5272 - accuracy: 0.8069

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5304 - accuracy: 0.8061

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5297 - accuracy: 0.8071

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5269 - accuracy: 0.8086

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5265 - accuracy: 0.8086

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5243 - accuracy: 0.8100

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5254 - accuracy: 0.8092

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5254 - accuracy: 0.8092

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5232 - accuracy: 0.8101

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5255 - accuracy: 0.8089

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5265 - accuracy: 0.8081

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5265 - accuracy: 0.8078

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5258 - accuracy: 0.8078

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5264 - accuracy: 0.8075

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5282 - accuracy: 0.8056

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5293 - accuracy: 0.8056

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5278 - accuracy: 0.8061

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5266 - accuracy: 0.8062

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5249 - accuracy: 0.8070

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5278 - accuracy: 0.8063

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5281 - accuracy: 0.8060

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5297 - accuracy: 0.8050

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5279 - accuracy: 0.8051

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5295 - accuracy: 0.8041

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5272 - accuracy: 0.8046

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5262 - accuracy: 0.8050

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5256 - accuracy: 0.8051

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5280 - accuracy: 0.8042

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5280 - accuracy: 0.8042 - val_loss: 0.7093 - val_accuracy: 0.7357


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



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_66_0.png


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
    
    img = tf.keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
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

    This image most likely belongs to sunflowers with a 99.59 percent confidence.


Save the TensorFlow Model
-------------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    #save the trained model - a new folder flower will be created
    #and the file "saved_model.h5" is the pre-trained model
    model_dir = "model"
    saved_model_path = f"{model_dir}/flower/saved_model"
    model.save(saved_model_path)


.. parsed-literal::

    2024-03-27 15:07:15.273905: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-27 15:07:15.369970: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:15.380490: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-27 15:07:15.392149: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:15.399687: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:15.407048: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:15.418682: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:15.460237: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-03-27 15:07:15.532876: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:15.554840: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-03-27 15:07:15.596852: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:15.622044: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:15.696655: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-27 15:07:15.853478: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:16.005739: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:16.042205: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-27 15:07:16.073858: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-27 15:07:16.124925: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    ir_model = ov.convert_model(saved_model_path, input=[1,180,180,3])
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
    This image most likely belongs to dandelion with a 99.54 percent confidence.



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_79_1.png


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
Model <./tensorflow-training-openvino-nncf.ipynb>`__ notebook.
