Person Tracking with OpenVINO™
==============================

This notebook demonstrates live person tracking with OpenVINO: it reads
frames from an input video sequence, detects people in the frames,
uniquely identifies each one of them and tracks all of them until they
leave the frame. We will use the `Deep
SORT <https://arxiv.org/abs/1703.07402>`__ algorithm to perform object
tracking, an extension to SORT (Simple Online and Realtime Tracking).

Detection vs Tracking
---------------------

-  In object detection, we detect an object in a frame, put a bounding
   box or a mask around it, and classify the object. Note that, the job
   of the detector ends here. It processes each frame independently and
   identifies numerous objects in that particular frame.
-  An object tracker on the other hand needs to track a particular
   object across the entire video. If the detector detects three cars in
   the frame, the object tracker has to identify the three separate
   detections and needs to track it across the subsequent frames (with
   the help of a unique ID).

Deep SORT
---------

`Deep SORT <https://arxiv.org/abs/1703.07402>`__ can be defined as the
tracking algorithm which tracks objects not only based on the velocity
and motion of the object but also the appearance of the object. It is
made of three key components which are as follows: |deepsort|

1. **Detection**

   This is the first step in the tracking module. In this step, a deep
   learning model will be used to detect the objects in the frame that
   are to be tracked. These detections are then passed on to the next
   step.

2. **Prediction**

   In this step, we use Kalman filter [1] framework to predict a target
   bounding box of each tracking object in the next frame. There are two
   states of prediction output: ``confirmed`` and ``unconfirmed``. A new
   track comes with a state of ``unconfirmed`` by default, and it can be
   turned into ``confirmed`` when a certain number of consecutive
   detections are matched with this new track. Meanwhile, if a matched
   track is missed over a specific time, it will be deleted as well.

3. **Data association and update**

   Now, we have to match the target bounding box with the detected
   bounding box, and update track identities. A conventional way to
   solve the association between the predicted Kalman states and newly
   arrived measurements is to build an assignment problem with the
   Hungarian algorithm [2]. In this problem formulation, we integrate
   motion and appearance information through a combination of two
   appropriate metrics. The cost used for the first matching step is set
   as a combination of the Mahalanobis and the cosine distances. The
   `Mahalanobis
   distance <https://en.wikipedia.org/wiki/Mahalanobis_distance>`__ is
   used to incorporate motion information and the cosine distance is
   used to calculate similarity between two objects. Cosine distance is
   a metric that helps the tracker recover identities in case of
   long-term occlusion and motion estimation also fails. For this
   purposes, a reidentification model will be implemented to produce a
   vector in high-dimensional space that represents the appearance of
   the object. Using these simple things can make the tracker even more
   powerful and accurate.

   In the second matching stage, we will run intersection over
   union(IOU) association as proposed in the original SORT algorithm [3]
   on the set of unconfirmed and unmatched tracks from the previous
   step. If the IOU of detection and target is less than a certain
   threshold value called ``IOUmin`` then that assignment is rejected.
   This helps to account for sudden appearance changes, for example, due
   to partial occlusion with static scene geometry, and to increase
   robustness against erroneous.

   When detection result is associated with a target, the detected
   bounding box is used to update the target state.

--------------

[1] R. Kalman, “A New Approach to Linear Filtering and Prediction
Problems”, Journal of Basic Engineering, vol. 82, no. Series D,
pp. 35-45, 1960.

[2] H. W. Kuhn, “The Hungarian method for the assignment problem”, Naval
Research Logistics Quarterly, vol. 2, pp. 83-97, 1955.

[3] A. Bewley, G. Zongyuan, F. Ramos, and B. Upcroft, “Simple online and
realtime tracking,” in ICIP, 2016, pp. 3464–3468.

.. |deepsort| image:: https://user-images.githubusercontent.com/91237924/221744683-0042eff8-2c41-43b8-b3ad-b5929bafb60b.png

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports <#imports>`__
-  `Download the Model <#download-the-model>`__
-  `Load model <#load-model>`__

   -  `Select inference device <#select-inference-device>`__

-  `Data Processing <#data-processing>`__
-  `Test person reidentification
   model <#test-person-reidentification-model>`__

   -  `Visualize data <#visualize-data>`__
   -  `Compare two persons <#compare-two-persons>`__

-  `Main Processing Function <#main-processing-function>`__
-  `Run <#run>`__

   -  `Initialize tracker <#initialize-tracker>`__
   -  `Run Live Person Tracking <#run-live-person-tracking>`__

.. code:: ipython3

    import platform
    
    %pip install -q "openvino-dev>=2024.0.0"
    %pip install -q opencv-python requests scipy
    
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    import collections
    from pathlib import Path
    import sys
    import time
    
    import numpy as np
    import cv2
    from IPython import display
    import matplotlib.pyplot as plt
    import openvino as ov

.. code:: ipython3

    # Import local modules
    
    utils_file_path = Path('../utils/notebook_utils.py')
    notebook_directory_path = Path('.')
    
    if not utils_file_path.exists():
        !git clone --depth 1 https://github.com/igor-davidyuk/openvino_notebooks.git -b moving_data_to_cloud openvino_notebooks
        utils_file_path = Path('./openvino_notebooks/notebooks/utils/notebook_utils.py')
        notebook_directory_path = Path('./openvino_notebooks/notebooks/person-tracking-webcam/')
    
    sys.path.append(str(utils_file_path.parent))
    sys.path.append(str(notebook_directory_path))
    
    import notebook_utils as utils
    from deepsort_utils.tracker import Tracker
    from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
    from deepsort_utils.detection import Detection, compute_color_for_labels, xywh_to_xyxy, xywh_to_tlwh, tlwh_to_xyxy

Download the Model
------------------

`back to top ⬆️ <#table-of-contents>`__

We will use pre-trained models from OpenVINO’s `Open Model
Zoo <https://docs.openvino.ai/2024/documentation/legacy-features/model-zoo.html>`__
to start the test.

Use ``omz_downloader``, which is a command-line tool from the
``openvino-dev`` package. It automatically creates a directory structure
and downloads the selected model. This step is skipped if the model is
already downloaded. The selected model comes from the public directory,
which means it must be converted into OpenVINO Intermediate
Representation (OpenVINO IR).

   **NOTE**: Using a model outside the list can require different pre-
   and post-processing.

In this case, `person detection
model <https://docs.openvino.ai/2024/omz_models_model_person_detection_0202.html>`__
is deployed to detect the person in each frame of the video, and
`reidentification
model <https://docs.openvino.ai/2024/omz_models_model_person_reidentification_retail_0287.html>`__
is used to output embedding vector to match a pair of images of a person
by the cosine distance.

If you want to download another model (``person-detection-xxx`` from
`Object Detection Models
list <https://docs.openvino.ai/2024/omz_models_group_intel.html#object-detection-models>`__,
``person-reidentification-retail-xxx`` from `Reidentification Models
list <https://docs.openvino.ai/2024/omz_models_group_intel.html#reidentification-models>`__),
replace the name of the model in the code below.

.. code:: ipython3

    # A directory where the model will be downloaded.
    base_model_dir = "model"
    precision = "FP16"
    # The name of the model from Open Model Zoo
    detection_model_name = "person-detection-0202"
    
    download_command = f"omz_downloader " \
                       f"--name {detection_model_name} " \
                       f"--precisions {precision} " \
                       f"--output_dir {base_model_dir} " \
                       f"--cache_dir {base_model_dir}"
    ! $download_command
    
    detection_model_path = f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"
    
    
    reidentification_model_name = "person-reidentification-retail-0287"
    
    download_command = f"omz_downloader " \
                       f"--name {reidentification_model_name} " \
                       f"--precisions {precision} " \
                       f"--output_dir {base_model_dir} " \
                       f"--cache_dir {base_model_dir}"
    ! $download_command
    
    reidentification_model_path = f"model/intel/{reidentification_model_name}/{precision}/{reidentification_model_name}.xml"


.. parsed-literal::

    ################|| Downloading person-detection-0202 ||################
    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.xml


.. parsed-literal::

    ... 12%, 32 KB, 1001 KB/s, 0 seconds passed
... 25%, 64 KB, 976 KB/s, 0 seconds passed

.. parsed-literal::

    ... 38%, 96 KB, 1387 KB/s, 0 seconds passed
... 51%, 128 KB, 1705 KB/s, 0 seconds passed
... 64%, 160 KB, 1623 KB/s, 0 seconds passed
... 77%, 192 KB, 1915 KB/s, 0 seconds passed
... 89%, 224 KB, 2200 KB/s, 0 seconds passed
... 100%, 248 KB, 2430 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.bin


.. parsed-literal::

    ... 0%, 32 KB, 921 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 64 KB, 904 KB/s, 0 seconds passed
... 2%, 96 KB, 1335 KB/s, 0 seconds passed
... 3%, 128 KB, 1749 KB/s, 0 seconds passed
... 4%, 160 KB, 1483 KB/s, 0 seconds passed
... 5%, 192 KB, 1760 KB/s, 0 seconds passed
... 6%, 224 KB, 2027 KB/s, 0 seconds passed
... 7%, 256 KB, 2277 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 288 KB, 1992 KB/s, 0 seconds passed
... 9%, 320 KB, 2198 KB/s, 0 seconds passed
... 9%, 352 KB, 2397 KB/s, 0 seconds passed
... 10%, 384 KB, 2590 KB/s, 0 seconds passed
... 11%, 416 KB, 2780 KB/s, 0 seconds passed
... 12%, 448 KB, 2960 KB/s, 0 seconds passed
... 13%, 480 KB, 3151 KB/s, 0 seconds passed
... 14%, 512 KB, 3324 KB/s, 0 seconds passed
... 15%, 544 KB, 3502 KB/s, 0 seconds passed
... 16%, 576 KB, 3687 KB/s, 0 seconds passed

.. parsed-literal::

    ... 17%, 608 KB, 3359 KB/s, 0 seconds passed
... 18%, 640 KB, 3527 KB/s, 0 seconds passed
... 18%, 672 KB, 3688 KB/s, 0 seconds passed
... 19%, 704 KB, 3801 KB/s, 0 seconds passed
... 20%, 736 KB, 3961 KB/s, 0 seconds passed
... 21%, 768 KB, 4124 KB/s, 0 seconds passed
... 22%, 800 KB, 4272 KB/s, 0 seconds passed
... 23%, 832 KB, 4433 KB/s, 0 seconds passed
... 24%, 864 KB, 4593 KB/s, 0 seconds passed
... 25%, 896 KB, 4755 KB/s, 0 seconds passed
... 26%, 928 KB, 4913 KB/s, 0 seconds passed
... 27%, 960 KB, 5014 KB/s, 0 seconds passed
... 27%, 992 KB, 5155 KB/s, 0 seconds passed
... 28%, 1024 KB, 5309 KB/s, 0 seconds passed
... 29%, 1056 KB, 5458 KB/s, 0 seconds passed
... 30%, 1088 KB, 5611 KB/s, 0 seconds passed
... 31%, 1120 KB, 5764 KB/s, 0 seconds passed
... 32%, 1152 KB, 5918 KB/s, 0 seconds passed
... 33%, 1184 KB, 6070 KB/s, 0 seconds passed
... 34%, 1216 KB, 5603 KB/s, 0 seconds passed
... 35%, 1248 KB, 5737 KB/s, 0 seconds passed
... 36%, 1280 KB, 5872 KB/s, 0 seconds passed
... 36%, 1312 KB, 6008 KB/s, 0 seconds passed
... 37%, 1344 KB, 6143 KB/s, 0 seconds passed
... 38%, 1376 KB, 6278 KB/s, 0 seconds passed
... 39%, 1408 KB, 6413 KB/s, 0 seconds passed
... 40%, 1440 KB, 6548 KB/s, 0 seconds passed
... 41%, 1472 KB, 6682 KB/s, 0 seconds passed
... 42%, 1504 KB, 6815 KB/s, 0 seconds passed
... 43%, 1536 KB, 6948 KB/s, 0 seconds passed
... 44%, 1568 KB, 7081 KB/s, 0 seconds passed

.. parsed-literal::

    ... 45%, 1600 KB, 7212 KB/s, 0 seconds passed
... 45%, 1632 KB, 7343 KB/s, 0 seconds passed
... 46%, 1664 KB, 7474 KB/s, 0 seconds passed
... 47%, 1696 KB, 7605 KB/s, 0 seconds passed
... 48%, 1728 KB, 7737 KB/s, 0 seconds passed
... 49%, 1760 KB, 7869 KB/s, 0 seconds passed
... 50%, 1792 KB, 8001 KB/s, 0 seconds passed
... 51%, 1824 KB, 8024 KB/s, 0 seconds passed
... 52%, 1856 KB, 8149 KB/s, 0 seconds passed
... 53%, 1888 KB, 8275 KB/s, 0 seconds passed
... 54%, 1920 KB, 8400 KB/s, 0 seconds passed
... 54%, 1952 KB, 8527 KB/s, 0 seconds passed
... 55%, 1984 KB, 8652 KB/s, 0 seconds passed
... 56%, 2016 KB, 8778 KB/s, 0 seconds passed
... 57%, 2048 KB, 8902 KB/s, 0 seconds passed
... 58%, 2080 KB, 9026 KB/s, 0 seconds passed
... 59%, 2112 KB, 9150 KB/s, 0 seconds passed
... 60%, 2144 KB, 9276 KB/s, 0 seconds passed
... 61%, 2176 KB, 9401 KB/s, 0 seconds passed
... 62%, 2208 KB, 9526 KB/s, 0 seconds passed
... 63%, 2240 KB, 9650 KB/s, 0 seconds passed
... 64%, 2272 KB, 9774 KB/s, 0 seconds passed
... 64%, 2304 KB, 9026 KB/s, 0 seconds passed
... 65%, 2336 KB, 9133 KB/s, 0 seconds passed
... 66%, 2368 KB, 9239 KB/s, 0 seconds passed
... 67%, 2400 KB, 9349 KB/s, 0 seconds passed
... 68%, 2432 KB, 9460 KB/s, 0 seconds passed
... 69%, 2464 KB, 9570 KB/s, 0 seconds passed
... 70%, 2496 KB, 9679 KB/s, 0 seconds passed
... 71%, 2528 KB, 9788 KB/s, 0 seconds passed
... 72%, 2560 KB, 9898 KB/s, 0 seconds passed
... 73%, 2592 KB, 10006 KB/s, 0 seconds passed
... 73%, 2624 KB, 10115 KB/s, 0 seconds passed
... 74%, 2656 KB, 10224 KB/s, 0 seconds passed
... 75%, 2688 KB, 10332 KB/s, 0 seconds passed
... 76%, 2720 KB, 10440 KB/s, 0 seconds passed
... 77%, 2752 KB, 10548 KB/s, 0 seconds passed
... 78%, 2784 KB, 10655 KB/s, 0 seconds passed
... 79%, 2816 KB, 10762 KB/s, 0 seconds passed
... 80%, 2848 KB, 10866 KB/s, 0 seconds passed
... 81%, 2880 KB, 10972 KB/s, 0 seconds passed
... 82%, 2912 KB, 11079 KB/s, 0 seconds passed
... 82%, 2944 KB, 11184 KB/s, 0 seconds passed
... 83%, 2976 KB, 11289 KB/s, 0 seconds passed
... 84%, 3008 KB, 11395 KB/s, 0 seconds passed
... 85%, 3040 KB, 11500 KB/s, 0 seconds passed
... 86%, 3072 KB, 11604 KB/s, 0 seconds passed
... 87%, 3104 KB, 11708 KB/s, 0 seconds passed
... 88%, 3136 KB, 11812 KB/s, 0 seconds passed
... 89%, 3168 KB, 11915 KB/s, 0 seconds passed
... 90%, 3200 KB, 12016 KB/s, 0 seconds passed
... 91%, 3232 KB, 12121 KB/s, 0 seconds passed
... 91%, 3264 KB, 12227 KB/s, 0 seconds passed
... 92%, 3296 KB, 12334 KB/s, 0 seconds passed
... 93%, 3328 KB, 12440 KB/s, 0 seconds passed
... 94%, 3360 KB, 12546 KB/s, 0 seconds passed
... 95%, 3392 KB, 12652 KB/s, 0 seconds passed
... 96%, 3424 KB, 12758 KB/s, 0 seconds passed
... 97%, 3456 KB, 12863 KB/s, 0 seconds passed
... 98%, 3488 KB, 12968 KB/s, 0 seconds passed
... 99%, 3520 KB, 13073 KB/s, 0 seconds passed
... 100%, 3549 KB, 13165 KB/s, 0 seconds passed





    


.. parsed-literal::

    ################|| Downloading person-reidentification-retail-0287 ||################
    
    ========== Downloading model/intel/person-reidentification-retail-0287/person-reidentification-retail-0267.onnx


.. parsed-literal::

    ... 0%, 32 KB, 902 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 64 KB, 880 KB/s, 0 seconds passed
... 2%, 96 KB, 1260 KB/s, 0 seconds passed
... 3%, 128 KB, 1619 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 160 KB, 1476 KB/s, 0 seconds passed
... 5%, 192 KB, 1741 KB/s, 0 seconds passed
... 6%, 224 KB, 1983 KB/s, 0 seconds passed
... 7%, 256 KB, 2236 KB/s, 0 seconds passed
... 8%, 288 KB, 2457 KB/s, 0 seconds passed
... 9%, 320 KB, 2363 KB/s, 0 seconds passed

.. parsed-literal::

    ... 10%, 352 KB, 2037 KB/s, 0 seconds passed
... 11%, 384 KB, 2002 KB/s, 0 seconds passed
... 11%, 416 KB, 2022 KB/s, 0 seconds passed

.. parsed-literal::

    ... 12%, 448 KB, 2065 KB/s, 0 seconds passed
... 13%, 480 KB, 2126 KB/s, 0 seconds passed
... 14%, 512 KB, 2197 KB/s, 0 seconds passed
... 15%, 544 KB, 2269 KB/s, 0 seconds passed
... 16%, 576 KB, 2351 KB/s, 0 seconds passed
... 17%, 608 KB, 2452 KB/s, 0 seconds passed
... 18%, 640 KB, 2532 KB/s, 0 seconds passed
... 19%, 672 KB, 2617 KB/s, 0 seconds passed

.. parsed-literal::

    ... 20%, 704 KB, 2698 KB/s, 0 seconds passed
... 21%, 736 KB, 2777 KB/s, 0 seconds passed
... 22%, 768 KB, 2868 KB/s, 0 seconds passed
... 22%, 800 KB, 2944 KB/s, 0 seconds passed
... 23%, 832 KB, 3037 KB/s, 0 seconds passed
... 24%, 864 KB, 3121 KB/s, 0 seconds passed
... 25%, 896 KB, 3214 KB/s, 0 seconds passed
... 26%, 928 KB, 3292 KB/s, 0 seconds passed
... 27%, 960 KB, 3381 KB/s, 0 seconds passed
... 28%, 992 KB, 3456 KB/s, 0 seconds passed
... 29%, 1024 KB, 3542 KB/s, 0 seconds passed
... 30%, 1056 KB, 3626 KB/s, 0 seconds passed
... 31%, 1088 KB, 3710 KB/s, 0 seconds passed
... 32%, 1120 KB, 3812 KB/s, 0 seconds passed
... 33%, 1152 KB, 3893 KB/s, 0 seconds passed
... 33%, 1184 KB, 3974 KB/s, 0 seconds passed
... 34%, 1216 KB, 4054 KB/s, 0 seconds passed
... 35%, 1248 KB, 4136 KB/s, 0 seconds passed
... 36%, 1280 KB, 4214 KB/s, 0 seconds passed
... 37%, 1312 KB, 4302 KB/s, 0 seconds passed
... 38%, 1344 KB, 4380 KB/s, 0 seconds passed
... 39%, 1376 KB, 4465 KB/s, 0 seconds passed
... 40%, 1408 KB, 4560 KB/s, 0 seconds passed
... 41%, 1440 KB, 4641 KB/s, 0 seconds passed

.. parsed-literal::

    ... 42%, 1472 KB, 4721 KB/s, 0 seconds passed
... 43%, 1504 KB, 4808 KB/s, 0 seconds passed
... 44%, 1536 KB, 4878 KB/s, 0 seconds passed
... 44%, 1568 KB, 4962 KB/s, 0 seconds passed
... 45%, 1600 KB, 5048 KB/s, 0 seconds passed
... 46%, 1632 KB, 5132 KB/s, 0 seconds passed
... 47%, 1664 KB, 5215 KB/s, 0 seconds passed
... 48%, 1696 KB, 5300 KB/s, 0 seconds passed
... 49%, 1728 KB, 5382 KB/s, 0 seconds passed
... 50%, 1760 KB, 5466 KB/s, 0 seconds passed
... 51%, 1792 KB, 5547 KB/s, 0 seconds passed
... 52%, 1824 KB, 5630 KB/s, 0 seconds passed
... 53%, 1856 KB, 5709 KB/s, 0 seconds passed
... 54%, 1888 KB, 5787 KB/s, 0 seconds passed
... 55%, 1920 KB, 5873 KB/s, 0 seconds passed
... 55%, 1952 KB, 5952 KB/s, 0 seconds passed
... 56%, 1984 KB, 6033 KB/s, 0 seconds passed
... 57%, 2016 KB, 6112 KB/s, 0 seconds passed
... 58%, 2048 KB, 6190 KB/s, 0 seconds passed
... 59%, 2080 KB, 6268 KB/s, 0 seconds passed
... 60%, 2112 KB, 6348 KB/s, 0 seconds passed
... 61%, 2144 KB, 6436 KB/s, 0 seconds passed
... 62%, 2176 KB, 6506 KB/s, 0 seconds passed
... 63%, 2208 KB, 6590 KB/s, 0 seconds passed
... 64%, 2240 KB, 6665 KB/s, 0 seconds passed
... 65%, 2272 KB, 6753 KB/s, 0 seconds passed
... 66%, 2304 KB, 6837 KB/s, 0 seconds passed
... 66%, 2336 KB, 6914 KB/s, 0 seconds passed
... 67%, 2368 KB, 6994 KB/s, 0 seconds passed
... 68%, 2400 KB, 7078 KB/s, 0 seconds passed
... 69%, 2432 KB, 7155 KB/s, 0 seconds passed
... 70%, 2464 KB, 7229 KB/s, 0 seconds passed
... 71%, 2496 KB, 7314 KB/s, 0 seconds passed
... 72%, 2528 KB, 7391 KB/s, 0 seconds passed
... 73%, 2560 KB, 7463 KB/s, 0 seconds passed
... 74%, 2592 KB, 7548 KB/s, 0 seconds passed
... 75%, 2624 KB, 7624 KB/s, 0 seconds passed
... 76%, 2656 KB, 7700 KB/s, 0 seconds passed
... 77%, 2688 KB, 7785 KB/s, 0 seconds passed
... 77%, 2720 KB, 7860 KB/s, 0 seconds passed
... 78%, 2752 KB, 7931 KB/s, 0 seconds passed
... 79%, 2784 KB, 8013 KB/s, 0 seconds passed
... 80%, 2816 KB, 8091 KB/s, 0 seconds passed
... 81%, 2848 KB, 8174 KB/s, 0 seconds passed
... 82%, 2880 KB, 8248 KB/s, 0 seconds passed
... 83%, 2912 KB, 8328 KB/s, 0 seconds passed
... 84%, 2944 KB, 8410 KB/s, 0 seconds passed
... 85%, 2976 KB, 8483 KB/s, 0 seconds passed
... 86%, 3008 KB, 8565 KB/s, 0 seconds passed
... 87%, 3040 KB, 8639 KB/s, 0 seconds passed
... 88%, 3072 KB, 8720 KB/s, 0 seconds passed
... 88%, 3104 KB, 8794 KB/s, 0 seconds passed
... 89%, 3136 KB, 8861 KB/s, 0 seconds passed
... 90%, 3168 KB, 8943 KB/s, 0 seconds passed
... 91%, 3200 KB, 9017 KB/s, 0 seconds passed
... 92%, 3232 KB, 9098 KB/s, 0 seconds passed
... 93%, 3264 KB, 9173 KB/s, 0 seconds passed
... 94%, 3296 KB, 9253 KB/s, 0 seconds passed
... 95%, 3328 KB, 9328 KB/s, 0 seconds passed
... 96%, 3360 KB, 9408 KB/s, 0 seconds passed
... 97%, 3392 KB, 9480 KB/s, 0 seconds passed
... 98%, 3424 KB, 9560 KB/s, 0 seconds passed
... 99%, 3456 KB, 9641 KB/s, 0 seconds passed
... 100%, 3487 KB, 9715 KB/s, 0 seconds passed



.. parsed-literal::

    
    ========== Downloading model/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml


.. parsed-literal::

    ... 5%, 32 KB, 940 KB/s, 0 seconds passed
... 10%, 64 KB, 939 KB/s, 0 seconds passed

.. parsed-literal::

    ... 15%, 96 KB, 1399 KB/s, 0 seconds passed
... 21%, 128 KB, 1250 KB/s, 0 seconds passed
... 26%, 160 KB, 1555 KB/s, 0 seconds passed
... 31%, 192 KB, 1857 KB/s, 0 seconds passed
... 37%, 224 KB, 2158 KB/s, 0 seconds passed
... 42%, 256 KB, 2458 KB/s, 0 seconds passed
... 47%, 288 KB, 2737 KB/s, 0 seconds passed

.. parsed-literal::

    ... 53%, 320 KB, 2334 KB/s, 0 seconds passed
... 58%, 352 KB, 2559 KB/s, 0 seconds passed
... 63%, 384 KB, 2780 KB/s, 0 seconds passed
... 69%, 416 KB, 3003 KB/s, 0 seconds passed
... 74%, 448 KB, 3225 KB/s, 0 seconds passed
... 79%, 480 KB, 3444 KB/s, 0 seconds passed
... 85%, 512 KB, 3662 KB/s, 0 seconds passed
... 90%, 544 KB, 3881 KB/s, 0 seconds passed
... 95%, 576 KB, 4100 KB/s, 0 seconds passed

.. parsed-literal::

    ... 100%, 600 KB, 3503 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.bin


.. parsed-literal::

    ... 2%, 32 KB, 905 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 64 KB, 908 KB/s, 0 seconds passed
... 8%, 96 KB, 1352 KB/s, 0 seconds passed
... 11%, 128 KB, 1793 KB/s, 0 seconds passed
... 13%, 160 KB, 1490 KB/s, 0 seconds passed
... 16%, 192 KB, 1780 KB/s, 0 seconds passed
... 19%, 224 KB, 2069 KB/s, 0 seconds passed
... 22%, 256 KB, 2356 KB/s, 0 seconds passed
... 24%, 288 KB, 2643 KB/s, 0 seconds passed

.. parsed-literal::

    ... 27%, 320 KB, 2246 KB/s, 0 seconds passed
... 30%, 352 KB, 2462 KB/s, 0 seconds passed
... 33%, 384 KB, 2677 KB/s, 0 seconds passed
... 36%, 416 KB, 2891 KB/s, 0 seconds passed
... 38%, 448 KB, 3105 KB/s, 0 seconds passed
... 41%, 480 KB, 3318 KB/s, 0 seconds passed
... 44%, 512 KB, 3529 KB/s, 0 seconds passed
... 47%, 544 KB, 3741 KB/s, 0 seconds passed
... 49%, 576 KB, 3953 KB/s, 0 seconds passed
... 52%, 608 KB, 4164 KB/s, 0 seconds passed

.. parsed-literal::

    ... 55%, 640 KB, 3568 KB/s, 0 seconds passed
... 58%, 672 KB, 3736 KB/s, 0 seconds passed
... 61%, 704 KB, 3871 KB/s, 0 seconds passed
... 63%, 736 KB, 4038 KB/s, 0 seconds passed
... 66%, 768 KB, 4205 KB/s, 0 seconds passed
... 69%, 800 KB, 4371 KB/s, 0 seconds passed
... 72%, 832 KB, 4537 KB/s, 0 seconds passed
... 74%, 864 KB, 4702 KB/s, 0 seconds passed
... 77%, 896 KB, 4866 KB/s, 0 seconds passed
... 80%, 928 KB, 5029 KB/s, 0 seconds passed
... 83%, 960 KB, 5192 KB/s, 0 seconds passed
... 86%, 992 KB, 5354 KB/s, 0 seconds passed
... 88%, 1024 KB, 5516 KB/s, 0 seconds passed
... 91%, 1056 KB, 5679 KB/s, 0 seconds passed
... 94%, 1088 KB, 5843 KB/s, 0 seconds passed
... 97%, 1120 KB, 6005 KB/s, 0 seconds passed
... 99%, 1152 KB, 6157 KB/s, 0 seconds passed
... 100%, 1153 KB, 6153 KB/s, 0 seconds passed

    


Load model
----------

`back to top ⬆️ <#table-of-contents>`__

Define a common class for model loading and predicting.

There are four main steps for OpenVINO model initialization, and they
are required to run for only once before inference loop. 1. Initialize
OpenVINO Runtime. 2. Read the network from ``*.bin`` and ``*.xml`` files
(weights and architecture). 3. Compile the model for device. 4. Get
input and output names of nodes.

In this case, we can put them all in a class constructor function.

To let OpenVINO automatically select the best device for inference just
use ``AUTO``. In most cases, the best device to use is ``GPU`` (better
performance, but slightly longer startup time).

.. code:: ipython3

    core = ov.Core()
    
    
    class Model:
        """
        This class represents a OpenVINO model object.
    
        """
        def __init__(self, model_path, batchsize=1, device="AUTO"):
            """
            Initialize the model object
            
            Parameters
            ----------
            model_path: path of inference model
            batchsize: batch size of input data
            device: device used to run inference
            """
            self.model = core.read_model(model=model_path)
            self.input_layer = self.model.input(0)
            self.input_shape = self.input_layer.shape
            self.height = self.input_shape[2]
            self.width = self.input_shape[3]
    
            for layer in self.model.inputs:
                input_shape = layer.partial_shape
                input_shape[0] = batchsize
                self.model.reshape({layer: input_shape})
            self.compiled_model = core.compile_model(model=self.model, device_name=device)
            self.output_layer = self.compiled_model.output(0)
    
        def predict(self, input):
            """
            Run inference
            
            Parameters
            ----------
            input: array of input data
            """
            result = self.compiled_model(input)[self.output_layer]
            return result

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
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

    detector = Model(detection_model_path, device=device.value)
    # since the number of detection object is uncertain, the input batch size of reid model should be dynamic
    extractor = Model(reidentification_model_path, -1, device.value)

Data Processing
---------------

`back to top ⬆️ <#table-of-contents>`__

Data Processing includes data preprocess and postprocess functions. -
Data preprocess function is used to change the layout and shape of input
data, according to requirement of the network input format. - Data
postprocess function is used to extract the useful information from
network’s original output and visualize it.

.. code:: ipython3

    def preprocess(frame, height, width):
        """
        Preprocess a single image
        
        Parameters
        ----------
        frame: input frame
        height: height of model input data
        width: width of model input data
        """
        resized_image = cv2.resize(frame, (width, height))
        resized_image = resized_image.transpose((2, 0, 1))
        input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
        return input_image
    
    
    def batch_preprocess(img_crops, height, width):
        """
        Preprocess batched images
        
        Parameters
        ----------
        img_crops: batched input images
        height: height of model input data
        width: width of model input data
        """
        img_batch = np.concatenate([
            preprocess(img, height, width)
            for img in img_crops
        ], axis=0)
        return img_batch
    
    
    def process_results(h, w, results, thresh=0.5):
        """
        postprocess detection results
        
        Parameters
        ----------
        h, w: original height and width of input image
        results: raw detection network output
        thresh: threshold for low confidence filtering
        """
        # The 'results' variable is a [1, 1, N, 7] tensor.
        detections = results.reshape(-1, 7)
        boxes = []
        labels = []
        scores = []
        for i, detection in enumerate(detections):
            _, label, score, xmin, ymin, xmax, ymax = detection
            # Filter detected objects.
            if score > thresh:
                # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
                boxes.append(
                    [(xmin + xmax) / 2 * w, (ymin + ymax) / 2 * h, (xmax - xmin) * w, (ymax - ymin) * h]
                )
                labels.append(int(label))
                scores.append(float(score))
    
        if len(boxes) == 0:
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])
            labels = np.array([])
        return np.array(boxes), np.array(scores), np.array(labels)
    
    
    def draw_boxes(img, bbox, identities=None):
        """
        Draw bounding box in original image
        
        Parameters
        ----------
        img: original image
        bbox: coordinate of bounding box
        identities: identities IDs
        """
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(
                img,
                label,
                (x1, y1 + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN,
                1.6,
                [255, 255, 255],
                2
            )
        return img
    
    
    def cosin_metric(x1, x2):
        """
        Calculate the consin distance of two vector
        
        Parameters
        ----------
        x1, x2: input vectors
        """
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

Test person reidentification model
----------------------------------

`back to top ⬆️ <#table-of-contents>`__

The reidentification network outputs a blob with the ``(1, 256)`` shape
named ``reid_embedding``, which can be compared with other descriptors
using the cosine distance.

Visualize data
~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    base_file_link = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/person_'
    image_indices = ['1_1.png', '1_2.png', '2_1.png']
    image_paths = [utils.download_file(base_file_link + image_index, directory='data') for image_index in image_indices]
    image1, image2, image3 = [cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) for image_path in image_paths]
    
    # Define titles with images.
    data = {"Person 1": image1, "Person 2": image2, "Person 3": image3}
    
    # Create a subplot to visualize images.
    fig, axs = plt.subplots(1, len(data.items()), figsize=(5, 5))
    
    # Fill the subplot.
    for ax, (name, image) in zip(axs, data.items()):
        ax.axis('off')
        ax.set_title(name)
        ax.imshow(image)
    
    # Display an image.
    plt.show(fig)



.. parsed-literal::

    data/person_1_1.png:   0%|          | 0.00/68.3k [00:00<?, ?B/s]



.. parsed-literal::

    data/person_1_2.png:   0%|          | 0.00/68.9k [00:00<?, ?B/s]



.. parsed-literal::

    data/person_2_1.png:   0%|          | 0.00/70.3k [00:00<?, ?B/s]



.. image:: person-tracking-with-output_files/person-tracking-with-output_17_3.png


Compare two persons
~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    # Metric parameters
    MAX_COSINE_DISTANCE = 0.6  # threshold of matching object
    input_data = [image2, image3]
    img_batch = batch_preprocess(input_data, extractor.height, extractor.width)
    features = extractor.predict(img_batch)
    sim = cosin_metric(features[0], features[1])
    if sim >= 1 - MAX_COSINE_DISTANCE:
        print(f'Same person (confidence: {sim})')
    else:
        print(f'Different person (confidence: {sim})')


.. parsed-literal::

    Different person (confidence: 0.02726624347269535)


Main Processing Function
------------------------

`back to top ⬆️ <#table-of-contents>`__

Run person tracking on the specified source. Either a webcam feed or a
video file.

.. code:: ipython3

    # Main processing function to run person tracking.
    def run_person_tracking(source=0, flip=False, use_popup=False, skip_first_frames=0):
        """
        Main function to run the person tracking:
        1. Create a video player to play with target fps (utils.VideoPlayer).
        2. Prepare a set of frames for person tracking.
        3. Run AI inference for person tracking.
        4. Visualize the results.
    
        Parameters:
        ----------
            source: The webcam number to feed the video stream with primary webcam set to "0", or the video path.  
            flip: To be used by VideoPlayer function for flipping capture image.
            use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
            skip_first_frames: Number of frames to skip at the beginning of the video. 
        """
        player = None
        try:
            # Create a video player to play with target fps.
            player = utils.VideoPlayer(
                source=source, size=(700, 450), flip=flip, fps=24, skip_first_frames=skip_first_frames
            )
            # Start capturing.
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(
                    winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
                )
    
            processing_times = collections.deque()
            while True:
                # Grab the frame.
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break
                # If the frame is larger than full HD, reduce size to improve the performance.
    
                # Resize the image and change dims to fit neural network input.
                h, w = frame.shape[:2]
                input_image = preprocess(frame, detector.height, detector.width)
    
                # Measure processing time.
                start_time = time.time()
                # Get the results.
                output = detector.predict(input_image)
                stop_time = time.time()
                processing_times.append(stop_time - start_time)
                if len(processing_times) > 200:
                    processing_times.popleft()
    
                _, f_width = frame.shape[:2]
                # Mean processing time [ms].
                processing_time = np.mean(processing_times) * 1100
                fps = 1000 / processing_time
    
                # Get poses from detection results.
                bbox_xywh, score, label = process_results(h, w, results=output)
                
                img_crops = []
                for box in bbox_xywh:
                    x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
                    img = frame[y1:y2, x1:x2]
                    img_crops.append(img)
    
                # Get reidentification feature of each person.
                if img_crops:
                    # preprocess
                    img_batch = batch_preprocess(img_crops, extractor.height, extractor.width)
                    features = extractor.predict(img_batch)
                else:
                    features = np.array([])
    
                # Wrap the detection and reidentification results together
                bbox_tlwh = xywh_to_tlwh(bbox_xywh)
                detections = [
                    Detection(bbox_tlwh[i], features[i])
                    for i in range(features.shape[0])
                ]
    
                # predict the position of tracking target 
                tracker.predict()
    
                # update tracker
                tracker.update(detections)
    
                # update bbox identities
                outputs = []
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    box = track.to_tlwh()
                    x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
                    track_id = track.track_id
                    outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))
                if len(outputs) > 0:
                    outputs = np.stack(outputs, axis=0)
    
                # draw box for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    frame = draw_boxes(frame, bbox_xyxy, identities)
    
                cv2.putText(
                    img=frame,
                    text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    org=(20, 40),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=f_width / 1000,
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                
                if use_popup:
                    cv2.imshow(winname=title, mat=frame)
                    key = cv2.waitKey(1)
                    # escape = 27
                    if key == 27:
                        break
                else:
                    # Encode numpy array to jpg.
                    _, encoded_img = cv2.imencode(
                        ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                    )
                    # Create an IPython image.
                    i = display.Image(data=encoded_img)
                    # Display the image in this notebook.
                    display.clear_output(wait=True)
                    display.display(i)
    
        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
        # any different error
        except RuntimeError as e:
            print(e)
        finally:
            if player is not None:
                # Stop capturing.
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()

Run
---

`back to top ⬆️ <#table-of-contents>`__

Initialize tracker
~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Before running a new tracking task, we have to reinitialize a Tracker
object

.. code:: ipython3

    NN_BUDGET = 100
    MAX_COSINE_DISTANCE = 0.6  # threshold of matching object
    metric = NearestNeighborDistanceMetric(
        "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
    )
    tracker = Tracker(
        metric,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3
    )

Run Live Person Tracking
~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Use a webcam as the video input. By default, the primary webcam is set
with ``source=0``. If you have multiple webcams, each one will be
assigned a consecutive number starting at 0. Set ``flip=True`` when
using a front-facing camera. Some web browsers, especially Mozilla
Firefox, may cause flickering. If you experience flickering, set
``use_popup=True``.

If you do not have a webcam, you can still run this demo with a video
file. Any `format supported by
OpenCV <https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
will work.

.. code:: ipython3

    USE_WEBCAM = False
    
    cam_id = 0
    video_file = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4'
    source = cam_id if USE_WEBCAM else video_file
    
    run_person_tracking(source=source, flip=USE_WEBCAM, use_popup=False)



.. image:: person-tracking-with-output_files/person-tracking-with-output_25_0.png


.. parsed-literal::

    Source ended

