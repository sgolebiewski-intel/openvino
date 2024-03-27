Image Colorization with OpenVINO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This notebook demonstrates how to colorize images with OpenVINO using
the Colorization model
`colorization-v2 <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/colorization-v2/README.md>`__
or
`colorization-siggraph <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-siggraph>`__
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md>`__
based on the paper `Colorful Image
Colorization <https://arxiv.org/abs/1603.08511>`__ models from Open
Model Zoo.

.. figure:: https://user-images.githubusercontent.com/18904157/180923280-9caefaf1-742b-4d2f-8943-5d4a6126e2fc.png
   :alt: Let there be color

   Let there be color

Given a grayscale image as input, the model generates colorized version
of the image as the output.

About Colorization-v2
^^^^^^^^^^^^^^^^^^^^^

-  The colorization-v2 model is one of the colorization group of models
   designed to perform image colorization.
-  Model trained on the ImageNet dataset.
-  Model consumes L-channel of LAB-image as input and produces predict
   A- and B-channels of LAB-image as output.

About Colorization-siggraph
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  The colorization-siggraph model is one of the colorization group of
   models designed to real-time user-guided image colorization.
-  Model trained on the ImageNet dataset with synthetically generated
   user interaction.
-  Model consumes L-channel of LAB-image as input and produces predict
   A- and B-channels of LAB-image as output.

See the `colorization <https://github.com/richzhang/colorization>`__
repository for more details.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports <#imports>`__
-  `Configurations <#configurations>`__

   -  `Select inference device <#select-inference-device>`__

-  `Download the model <#download-the-model>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__
-  `Loading the Model <#loading-the-model>`__
-  `Utility Functions <#utility-functions>`__
-  `Load the Image <#load-the-image>`__
-  `Display Colorized Image <#display-colorized-image>`__

.. code:: ipython3

    %pip install "openvino-dev>=2024.0.0"


.. parsed-literal::

    Collecting openvino-dev>=2024.0.0
      Using cached openvino_dev-2024.0.0-14509-py3-none-any.whl.metadata (16 kB)


.. parsed-literal::

    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (0.7.1)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.8.8)
    Requirement already satisfied: numpy>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (1.23.5)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2023.2.1)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (24.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.31.0)
    Requirement already satisfied: openvino==2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.0.0)


.. parsed-literal::

    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2024.2.2)
    Using cached openvino_dev-2024.0.0-14509-py3-none-any.whl (4.7 MB)


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Installing collected packages: openvino-dev


.. parsed-literal::

    Successfully installed openvino-dev-2024.0.0


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    import os
    import sys
    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import openvino as ov
    
    sys.path.append("../utils")
    import notebook_utils as utils

Configurations
--------------

`back to top ⬆️ <#table-of-contents>`__

-  ``PRECISION`` - {FP16, FP32}, default: FP16.
-  ``MODEL_DIR`` - directory where the model is to be stored, default:
   public.
-  ``MODEL_NAME`` - name of the model used for inference, default:
   colorization-v2.
-  ``DATA_DIR`` - directory where test images are stored, default: data.

.. code:: ipython3

    PRECISION = "FP16"
    MODEL_DIR = "models"
    MODEL_NAME = "colorization-v2"
    # MODEL_NAME="colorization-siggraph"
    MODEL_PATH = f"{MODEL_DIR}/public/{MODEL_NAME}/{PRECISION}/{MODEL_NAME}.xml"
    DATA_DIR = "data"

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
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



Download the model
------------------

`back to top ⬆️ <#table-of-contents>`__

``omz_downloader`` downloads model files from online sources and, if
necessary, patches them to make them more usable with Model Converter.

In this case, ``omz_downloader`` downloads the checkpoint and pytorch
model of
`colorization-v2 <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/colorization-v2/README.md>`__
or
`colorization-siggraph <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-siggraph>`__
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md>`__
and saves it under ``MODEL_DIR``, as specified in the configuration
above.

.. code:: ipython3

    download_command = (
        f"omz_downloader "
        f"--name {MODEL_NAME} "
        f"--output_dir {MODEL_DIR} "
        f"--cache_dir {MODEL_DIR}"
    )
    ! $download_command


.. parsed-literal::

    ################|| Downloading colorization-v2 ||################
    
    ========== Downloading models/public/colorization-v2/ckpt/colorization-v2-eccv16.pth


.. parsed-literal::

    ... 0%, 32 KB, 922 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 969 KB/s, 0 seconds passed
... 0%, 96 KB, 1410 KB/s, 0 seconds passed
... 0%, 128 KB, 1832 KB/s, 0 seconds passed
... 0%, 160 KB, 1595 KB/s, 0 seconds passed
... 0%, 192 KB, 1885 KB/s, 0 seconds passed
... 0%, 224 KB, 2169 KB/s, 0 seconds passed
... 0%, 256 KB, 2430 KB/s, 0 seconds passed
... 0%, 288 KB, 2695 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 320 KB, 2377 KB/s, 0 seconds passed
... 0%, 352 KB, 2596 KB/s, 0 seconds passed
... 0%, 384 KB, 2818 KB/s, 0 seconds passed
... 0%, 416 KB, 3019 KB/s, 0 seconds passed
... 0%, 448 KB, 3239 KB/s, 0 seconds passed
... 0%, 480 KB, 3437 KB/s, 0 seconds passed
... 0%, 512 KB, 3571 KB/s, 0 seconds passed
... 0%, 544 KB, 3771 KB/s, 0 seconds passed
... 0%, 576 KB, 3959 KB/s, 0 seconds passed
... 0%, 608 KB, 4133 KB/s, 0 seconds passed
... 0%, 640 KB, 3830 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 672 KB, 3986 KB/s, 0 seconds passed
... 0%, 704 KB, 4164 KB/s, 0 seconds passed
... 0%, 736 KB, 4269 KB/s, 0 seconds passed
... 0%, 768 KB, 4444 KB/s, 0 seconds passed
... 0%, 800 KB, 4597 KB/s, 0 seconds passed
... 0%, 832 KB, 4766 KB/s, 0 seconds passed
... 0%, 864 KB, 4924 KB/s, 0 seconds passed
... 0%, 896 KB, 5093 KB/s, 0 seconds passed
... 0%, 928 KB, 5263 KB/s, 0 seconds passed
... 0%, 960 KB, 5431 KB/s, 0 seconds passed
... 0%, 992 KB, 5600 KB/s, 0 seconds passed
... 0%, 1024 KB, 5768 KB/s, 0 seconds passed
... 0%, 1056 KB, 5935 KB/s, 0 seconds passed
... 0%, 1088 KB, 6100 KB/s, 0 seconds passed
... 0%, 1120 KB, 6268 KB/s, 0 seconds passed
... 0%, 1152 KB, 6435 KB/s, 0 seconds passed
... 0%, 1184 KB, 6599 KB/s, 0 seconds passed
... 0%, 1216 KB, 6762 KB/s, 0 seconds passed
... 0%, 1248 KB, 6928 KB/s, 0 seconds passed
... 1%, 1280 KB, 7094 KB/s, 0 seconds passed
... 1%, 1312 KB, 6529 KB/s, 0 seconds passed
... 1%, 1344 KB, 6621 KB/s, 0 seconds passed
... 1%, 1376 KB, 6762 KB/s, 0 seconds passed
... 1%, 1408 KB, 6904 KB/s, 0 seconds passed
... 1%, 1440 KB, 7048 KB/s, 0 seconds passed
... 1%, 1472 KB, 7190 KB/s, 0 seconds passed
... 1%, 1504 KB, 7333 KB/s, 0 seconds passed
... 1%, 1536 KB, 7477 KB/s, 0 seconds passed
... 1%, 1568 KB, 7612 KB/s, 0 seconds passed
... 1%, 1600 KB, 7735 KB/s, 0 seconds passed
... 1%, 1632 KB, 7874 KB/s, 0 seconds passed
... 1%, 1664 KB, 8006 KB/s, 0 seconds passed
... 1%, 1696 KB, 8144 KB/s, 0 seconds passed
... 1%, 1728 KB, 8259 KB/s, 0 seconds passed
... 1%, 1760 KB, 8395 KB/s, 0 seconds passed
... 1%, 1792 KB, 8533 KB/s, 0 seconds passed
... 1%, 1824 KB, 8671 KB/s, 0 seconds passed
... 1%, 1856 KB, 8806 KB/s, 0 seconds passed
... 1%, 1888 KB, 8914 KB/s, 0 seconds passed
... 1%, 1920 KB, 9046 KB/s, 0 seconds passed
... 1%, 1952 KB, 9179 KB/s, 0 seconds passed
... 1%, 1984 KB, 9313 KB/s, 0 seconds passed
... 1%, 2016 KB, 9433 KB/s, 0 seconds passed
... 1%, 2048 KB, 9553 KB/s, 0 seconds passed
... 1%, 2080 KB, 9683 KB/s, 0 seconds passed
... 1%, 2112 KB, 9814 KB/s, 0 seconds passed
... 1%, 2144 KB, 9947 KB/s, 0 seconds passed
... 1%, 2176 KB, 10061 KB/s, 0 seconds passed
... 1%, 2208 KB, 10111 KB/s, 0 seconds passed
... 1%, 2240 KB, 10238 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 2272 KB, 10366 KB/s, 0 seconds passed
... 1%, 2304 KB, 10494 KB/s, 0 seconds passed
... 1%, 2336 KB, 10621 KB/s, 0 seconds passed
... 1%, 2368 KB, 10748 KB/s, 0 seconds passed
... 1%, 2400 KB, 10876 KB/s, 0 seconds passed
... 1%, 2432 KB, 10998 KB/s, 0 seconds passed
... 1%, 2464 KB, 11124 KB/s, 0 seconds passed
... 1%, 2496 KB, 11251 KB/s, 0 seconds passed
... 2%, 2528 KB, 11364 KB/s, 0 seconds passed
... 2%, 2560 KB, 11489 KB/s, 0 seconds passed
... 2%, 2592 KB, 11598 KB/s, 0 seconds passed
... 2%, 2624 KB, 11719 KB/s, 0 seconds passed
... 2%, 2656 KB, 11305 KB/s, 0 seconds passed
... 2%, 2688 KB, 11339 KB/s, 0 seconds passed
... 2%, 2720 KB, 11455 KB/s, 0 seconds passed
... 2%, 2752 KB, 11558 KB/s, 0 seconds passed
... 2%, 2784 KB, 11673 KB/s, 0 seconds passed
... 2%, 2816 KB, 11785 KB/s, 0 seconds passed
... 2%, 2848 KB, 11900 KB/s, 0 seconds passed
... 2%, 2880 KB, 12006 KB/s, 0 seconds passed
... 2%, 2912 KB, 12120 KB/s, 0 seconds passed
... 2%, 2944 KB, 12234 KB/s, 0 seconds passed
... 2%, 2976 KB, 12332 KB/s, 0 seconds passed
... 2%, 3008 KB, 12445 KB/s, 0 seconds passed
... 2%, 3040 KB, 12548 KB/s, 0 seconds passed
... 2%, 3072 KB, 12660 KB/s, 0 seconds passed
... 2%, 3104 KB, 12763 KB/s, 0 seconds passed
... 2%, 3136 KB, 12874 KB/s, 0 seconds passed
... 2%, 3168 KB, 12985 KB/s, 0 seconds passed
... 2%, 3200 KB, 13092 KB/s, 0 seconds passed
... 2%, 3232 KB, 13203 KB/s, 0 seconds passed
... 2%, 3264 KB, 13301 KB/s, 0 seconds passed
... 2%, 3296 KB, 13411 KB/s, 0 seconds passed
... 2%, 3328 KB, 13510 KB/s, 0 seconds passed
... 2%, 3360 KB, 13619 KB/s, 0 seconds passed
... 2%, 3392 KB, 13729 KB/s, 0 seconds passed
... 2%, 3424 KB, 13838 KB/s, 0 seconds passed
... 2%, 3456 KB, 13945 KB/s, 0 seconds passed
... 2%, 3488 KB, 14041 KB/s, 0 seconds passed
... 2%, 3520 KB, 14152 KB/s, 0 seconds passed
... 2%, 3552 KB, 14254 KB/s, 0 seconds passed
... 2%, 3584 KB, 14364 KB/s, 0 seconds passed
... 2%, 3616 KB, 14465 KB/s, 0 seconds passed
... 2%, 3648 KB, 14575 KB/s, 0 seconds passed
... 2%, 3680 KB, 14685 KB/s, 0 seconds passed
... 2%, 3712 KB, 14787 KB/s, 0 seconds passed
... 2%, 3744 KB, 14896 KB/s, 0 seconds passed
... 2%, 3776 KB, 14985 KB/s, 0 seconds passed
... 3%, 3808 KB, 15093 KB/s, 0 seconds passed
... 3%, 3840 KB, 15195 KB/s, 0 seconds passed
... 3%, 3872 KB, 15303 KB/s, 0 seconds passed
... 3%, 3904 KB, 15410 KB/s, 0 seconds passed
... 3%, 3936 KB, 15505 KB/s, 0 seconds passed
... 3%, 3968 KB, 15612 KB/s, 0 seconds passed
... 3%, 4000 KB, 15718 KB/s, 0 seconds passed
... 3%, 4032 KB, 15824 KB/s, 0 seconds passed
... 3%, 4064 KB, 15377 KB/s, 0 seconds passed
... 3%, 4096 KB, 15474 KB/s, 0 seconds passed
... 3%, 4128 KB, 15572 KB/s, 0 seconds passed
... 3%, 4160 KB, 15552 KB/s, 0 seconds passed
... 3%, 4192 KB, 15647 KB/s, 0 seconds passed
... 3%, 4224 KB, 15745 KB/s, 0 seconds passed
... 3%, 4256 KB, 15842 KB/s, 0 seconds passed
... 3%, 4288 KB, 15939 KB/s, 0 seconds passed
... 3%, 4320 KB, 16036 KB/s, 0 seconds passed
... 3%, 4352 KB, 16133 KB/s, 0 seconds passed
... 3%, 4384 KB, 16228 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 4416 KB, 16325 KB/s, 0 seconds passed
... 3%, 4448 KB, 16419 KB/s, 0 seconds passed
... 3%, 4480 KB, 16514 KB/s, 0 seconds passed
... 3%, 4512 KB, 16609 KB/s, 0 seconds passed
... 3%, 4544 KB, 16701 KB/s, 0 seconds passed
... 3%, 4576 KB, 16794 KB/s, 0 seconds passed
... 3%, 4608 KB, 16889 KB/s, 0 seconds passed
... 3%, 4640 KB, 16982 KB/s, 0 seconds passed
... 3%, 4672 KB, 17076 KB/s, 0 seconds passed
... 3%, 4704 KB, 17169 KB/s, 0 seconds passed
... 3%, 4736 KB, 17261 KB/s, 0 seconds passed
... 3%, 4768 KB, 17351 KB/s, 0 seconds passed
... 3%, 4800 KB, 17443 KB/s, 0 seconds passed
... 3%, 4832 KB, 17536 KB/s, 0 seconds passed
... 3%, 4864 KB, 17626 KB/s, 0 seconds passed
... 3%, 4896 KB, 17717 KB/s, 0 seconds passed
... 3%, 4928 KB, 17809 KB/s, 0 seconds passed
... 3%, 4960 KB, 17900 KB/s, 0 seconds passed
... 3%, 4992 KB, 17991 KB/s, 0 seconds passed
... 3%, 5024 KB, 18083 KB/s, 0 seconds passed
... 4%, 5056 KB, 18176 KB/s, 0 seconds passed
... 4%, 5088 KB, 18278 KB/s, 0 seconds passed
... 4%, 5120 KB, 18379 KB/s, 0 seconds passed
... 4%, 5152 KB, 18478 KB/s, 0 seconds passed
... 4%, 5184 KB, 18576 KB/s, 0 seconds passed
... 4%, 5216 KB, 18675 KB/s, 0 seconds passed
... 4%, 5248 KB, 18773 KB/s, 0 seconds passed
... 4%, 5280 KB, 18870 KB/s, 0 seconds passed
... 4%, 5312 KB, 18968 KB/s, 0 seconds passed
... 4%, 5344 KB, 19065 KB/s, 0 seconds passed
... 4%, 5376 KB, 19161 KB/s, 0 seconds passed
... 4%, 5408 KB, 19259 KB/s, 0 seconds passed
... 4%, 5440 KB, 19354 KB/s, 0 seconds passed
... 4%, 5472 KB, 19452 KB/s, 0 seconds passed
... 4%, 5504 KB, 19549 KB/s, 0 seconds passed
... 4%, 5536 KB, 19645 KB/s, 0 seconds passed
... 4%, 5568 KB, 19742 KB/s, 0 seconds passed
... 4%, 5600 KB, 19838 KB/s, 0 seconds passed
... 4%, 5632 KB, 19935 KB/s, 0 seconds passed
... 4%, 5664 KB, 20031 KB/s, 0 seconds passed
... 4%, 5696 KB, 20128 KB/s, 0 seconds passed
... 4%, 5728 KB, 20224 KB/s, 0 seconds passed
... 4%, 5760 KB, 20320 KB/s, 0 seconds passed
... 4%, 5792 KB, 20415 KB/s, 0 seconds passed
... 4%, 5824 KB, 20509 KB/s, 0 seconds passed
... 4%, 5856 KB, 20604 KB/s, 0 seconds passed
... 4%, 5888 KB, 20699 KB/s, 0 seconds passed
... 4%, 5920 KB, 20794 KB/s, 0 seconds passed
... 4%, 5952 KB, 20889 KB/s, 0 seconds passed
... 4%, 5984 KB, 20984 KB/s, 0 seconds passed
... 4%, 6016 KB, 21078 KB/s, 0 seconds passed
... 4%, 6048 KB, 21173 KB/s, 0 seconds passed
... 4%, 6080 KB, 21267 KB/s, 0 seconds passed
... 4%, 6112 KB, 21362 KB/s, 0 seconds passed
... 4%, 6144 KB, 21455 KB/s, 0 seconds passed
... 4%, 6176 KB, 21549 KB/s, 0 seconds passed
... 4%, 6208 KB, 21643 KB/s, 0 seconds passed
... 4%, 6240 KB, 21737 KB/s, 0 seconds passed
... 4%, 6272 KB, 21830 KB/s, 0 seconds passed
... 5%, 6304 KB, 21923 KB/s, 0 seconds passed
... 5%, 6336 KB, 22015 KB/s, 0 seconds passed
... 5%, 6368 KB, 22108 KB/s, 0 seconds passed
... 5%, 6400 KB, 22200 KB/s, 0 seconds passed
... 5%, 6432 KB, 22293 KB/s, 0 seconds passed
... 5%, 6464 KB, 22385 KB/s, 0 seconds passed
... 5%, 6496 KB, 22477 KB/s, 0 seconds passed
... 5%, 6528 KB, 22575 KB/s, 0 seconds passed
... 5%, 6560 KB, 22673 KB/s, 0 seconds passed
... 5%, 6592 KB, 22771 KB/s, 0 seconds passed
... 5%, 6624 KB, 22870 KB/s, 0 seconds passed
... 5%, 6656 KB, 22965 KB/s, 0 seconds passed
... 5%, 6688 KB, 23050 KB/s, 0 seconds passed
... 5%, 6720 KB, 23140 KB/s, 0 seconds passed
... 5%, 6752 KB, 23229 KB/s, 0 seconds passed
... 5%, 6784 KB, 23318 KB/s, 0 seconds passed
... 5%, 6816 KB, 23402 KB/s, 0 seconds passed
... 5%, 6848 KB, 23491 KB/s, 0 seconds passed
... 5%, 6880 KB, 23578 KB/s, 0 seconds passed
... 5%, 6912 KB, 23668 KB/s, 0 seconds passed
... 5%, 6944 KB, 23751 KB/s, 0 seconds passed
... 5%, 6976 KB, 23839 KB/s, 0 seconds passed
... 5%, 7008 KB, 23927 KB/s, 0 seconds passed
... 5%, 7040 KB, 24011 KB/s, 0 seconds passed
... 5%, 7072 KB, 24098 KB/s, 0 seconds passed
... 5%, 7104 KB, 24186 KB/s, 0 seconds passed
... 5%, 7136 KB, 24273 KB/s, 0 seconds passed
... 5%, 7168 KB, 24356 KB/s, 0 seconds passed
... 5%, 7200 KB, 24443 KB/s, 0 seconds passed
... 5%, 7232 KB, 24525 KB/s, 0 seconds passed
... 5%, 7264 KB, 24616 KB/s, 0 seconds passed
... 5%, 7296 KB, 24698 KB/s, 0 seconds passed
... 5%, 7328 KB, 24784 KB/s, 0 seconds passed
... 5%, 7360 KB, 24870 KB/s, 0 seconds passed
... 5%, 7392 KB, 24956 KB/s, 0 seconds passed
... 5%, 7424 KB, 25038 KB/s, 0 seconds passed
... 5%, 7456 KB, 25123 KB/s, 0 seconds passed
... 5%, 7488 KB, 25209 KB/s, 0 seconds passed
... 5%, 7520 KB, 25285 KB/s, 0 seconds passed
... 5%, 7552 KB, 25362 KB/s, 0 seconds passed
... 6%, 7584 KB, 25434 KB/s, 0 seconds passed
... 6%, 7616 KB, 25509 KB/s, 0 seconds passed
... 6%, 7648 KB, 25602 KB/s, 0 seconds passed
... 6%, 7680 KB, 25695 KB/s, 0 seconds passed
... 6%, 7712 KB, 25787 KB/s, 0 seconds passed
... 6%, 7744 KB, 25868 KB/s, 0 seconds passed
... 6%, 7776 KB, 25948 KB/s, 0 seconds passed
... 6%, 7808 KB, 26025 KB/s, 0 seconds passed
... 6%, 7840 KB, 26104 KB/s, 0 seconds passed
... 6%, 7872 KB, 26183 KB/s, 0 seconds passed
... 6%, 7904 KB, 26262 KB/s, 0 seconds passed
... 6%, 7936 KB, 26342 KB/s, 0 seconds passed
... 6%, 7968 KB, 26421 KB/s, 0 seconds passed
... 6%, 8000 KB, 26500 KB/s, 0 seconds passed
... 6%, 8032 KB, 26579 KB/s, 0 seconds passed
... 6%, 8064 KB, 26657 KB/s, 0 seconds passed
... 6%, 8096 KB, 26736 KB/s, 0 seconds passed
... 6%, 8128 KB, 26814 KB/s, 0 seconds passed
... 6%, 8160 KB, 26891 KB/s, 0 seconds passed
... 6%, 8192 KB, 26969 KB/s, 0 seconds passed
... 6%, 8224 KB, 27043 KB/s, 0 seconds passed
... 6%, 8256 KB, 27119 KB/s, 0 seconds passed
... 6%, 8288 KB, 27197 KB/s, 0 seconds passed
... 6%, 8320 KB, 27275 KB/s, 0 seconds passed
... 6%, 8352 KB, 27351 KB/s, 0 seconds passed
... 6%, 8384 KB, 27428 KB/s, 0 seconds passed
... 6%, 8416 KB, 27504 KB/s, 0 seconds passed
... 6%, 8448 KB, 27581 KB/s, 0 seconds passed
... 6%, 8480 KB, 27658 KB/s, 0 seconds passed
... 6%, 8512 KB, 27735 KB/s, 0 seconds passed
... 6%, 8544 KB, 27811 KB/s, 0 seconds passed
... 6%, 8576 KB, 27886 KB/s, 0 seconds passed
... 6%, 8608 KB, 27960 KB/s, 0 seconds passed
... 6%, 8640 KB, 28033 KB/s, 0 seconds passed
... 6%, 8672 KB, 28108 KB/s, 0 seconds passed
... 6%, 8704 KB, 28183 KB/s, 0 seconds passed
... 6%, 8736 KB, 28264 KB/s, 0 seconds passed
... 6%, 8768 KB, 28348 KB/s, 0 seconds passed
... 6%, 8800 KB, 28432 KB/s, 0 seconds passed
... 7%, 8832 KB, 28516 KB/s, 0 seconds passed
... 7%, 8864 KB, 28601 KB/s, 0 seconds passed
... 7%, 8896 KB, 28685 KB/s, 0 seconds passed
... 7%, 8928 KB, 28769 KB/s, 0 seconds passed
... 7%, 8960 KB, 28854 KB/s, 0 seconds passed
... 7%, 8992 KB, 28938 KB/s, 0 seconds passed
... 7%, 9024 KB, 29021 KB/s, 0 seconds passed
... 7%, 9056 KB, 29105 KB/s, 0 seconds passed
... 7%, 9088 KB, 29188 KB/s, 0 seconds passed
... 7%, 9120 KB, 29271 KB/s, 0 seconds passed
... 7%, 9152 KB, 29351 KB/s, 0 seconds passed
... 7%, 9184 KB, 29434 KB/s, 0 seconds passed
... 7%, 9216 KB, 29517 KB/s, 0 seconds passed
... 7%, 9248 KB, 29596 KB/s, 0 seconds passed
... 7%, 9280 KB, 29673 KB/s, 0 seconds passed
... 7%, 9312 KB, 29750 KB/s, 0 seconds passed
... 7%, 9344 KB, 29827 KB/s, 0 seconds passed
... 7%, 9376 KB, 29899 KB/s, 0 seconds passed
... 7%, 9408 KB, 29976 KB/s, 0 seconds passed
... 7%, 9440 KB, 30052 KB/s, 0 seconds passed
... 7%, 9472 KB, 30125 KB/s, 0 seconds passed
... 7%, 9504 KB, 30201 KB/s, 0 seconds passed
... 7%, 9536 KB, 30277 KB/s, 0 seconds passed
... 7%, 9568 KB, 30353 KB/s, 0 seconds passed
... 7%, 9600 KB, 30425 KB/s, 0 seconds passed
... 7%, 9632 KB, 30498 KB/s, 0 seconds passed
... 7%, 9664 KB, 30503 KB/s, 0 seconds passed
... 7%, 9696 KB, 30574 KB/s, 0 seconds passed
... 7%, 9728 KB, 30600 KB/s, 0 seconds passed
... 7%, 9760 KB, 30659 KB/s, 0 seconds passed
... 7%, 9792 KB, 30727 KB/s, 0 seconds passed
... 7%, 9824 KB, 30807 KB/s, 0 seconds passed
... 7%, 9856 KB, 30881 KB/s, 0 seconds passed
... 7%, 9888 KB, 30954 KB/s, 0 seconds passed
... 7%, 9920 KB, 31021 KB/s, 0 seconds passed
... 7%, 9952 KB, 31092 KB/s, 0 seconds passed
... 7%, 9984 KB, 31162 KB/s, 0 seconds passed
... 7%, 10016 KB, 31234 KB/s, 0 seconds passed
... 7%, 10048 KB, 31304 KB/s, 0 seconds passed
... 8%, 10080 KB, 31375 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 10112 KB, 31444 KB/s, 0 seconds passed
... 8%, 10144 KB, 31515 KB/s, 0 seconds passed
... 8%, 10176 KB, 31585 KB/s, 0 seconds passed
... 8%, 10208 KB, 31656 KB/s, 0 seconds passed
... 8%, 10240 KB, 31725 KB/s, 0 seconds passed
... 8%, 10272 KB, 31796 KB/s, 0 seconds passed
... 8%, 10304 KB, 31860 KB/s, 0 seconds passed
... 8%, 10336 KB, 31929 KB/s, 0 seconds passed
... 8%, 10368 KB, 31997 KB/s, 0 seconds passed
... 8%, 10400 KB, 32066 KB/s, 0 seconds passed
... 8%, 10432 KB, 32130 KB/s, 0 seconds passed
... 8%, 10464 KB, 32196 KB/s, 0 seconds passed
... 8%, 10496 KB, 32265 KB/s, 0 seconds passed
... 8%, 10528 KB, 32333 KB/s, 0 seconds passed
... 8%, 10560 KB, 32402 KB/s, 0 seconds passed
... 8%, 10592 KB, 32471 KB/s, 0 seconds passed
... 8%, 10624 KB, 32541 KB/s, 0 seconds passed
... 8%, 10656 KB, 32607 KB/s, 0 seconds passed
... 8%, 10688 KB, 32674 KB/s, 0 seconds passed
... 8%, 10720 KB, 32742 KB/s, 0 seconds passed
... 8%, 10752 KB, 32811 KB/s, 0 seconds passed
... 8%, 10784 KB, 32876 KB/s, 0 seconds passed
... 8%, 10816 KB, 32943 KB/s, 0 seconds passed
... 8%, 10848 KB, 33010 KB/s, 0 seconds passed
... 8%, 10880 KB, 33077 KB/s, 0 seconds passed
... 8%, 10912 KB, 33155 KB/s, 0 seconds passed
... 8%, 10944 KB, 33232 KB/s, 0 seconds passed
... 8%, 10976 KB, 33310 KB/s, 0 seconds passed
... 8%, 11008 KB, 33388 KB/s, 0 seconds passed
... 8%, 11040 KB, 33466 KB/s, 0 seconds passed
... 8%, 11072 KB, 33543 KB/s, 0 seconds passed
... 8%, 11104 KB, 33621 KB/s, 0 seconds passed
... 8%, 11136 KB, 33699 KB/s, 0 seconds passed
... 8%, 11168 KB, 33776 KB/s, 0 seconds passed
... 8%, 11200 KB, 33853 KB/s, 0 seconds passed
... 8%, 11232 KB, 33930 KB/s, 0 seconds passed
... 8%, 11264 KB, 34004 KB/s, 0 seconds passed
... 8%, 11296 KB, 34069 KB/s, 0 seconds passed
... 8%, 11328 KB, 34122 KB/s, 0 seconds passed
... 9%, 11360 KB, 34175 KB/s, 0 seconds passed
... 9%, 11392 KB, 34229 KB/s, 0 seconds passed
... 9%, 11424 KB, 34305 KB/s, 0 seconds passed
... 9%, 11456 KB, 34381 KB/s, 0 seconds passed
... 9%, 11488 KB, 34457 KB/s, 0 seconds passed
... 9%, 11520 KB, 34533 KB/s, 0 seconds passed
... 9%, 11552 KB, 34610 KB/s, 0 seconds passed
... 9%, 11584 KB, 34671 KB/s, 0 seconds passed
... 9%, 11616 KB, 34736 KB/s, 0 seconds passed
... 9%, 11648 KB, 34810 KB/s, 0 seconds passed
... 9%, 11680 KB, 34878 KB/s, 0 seconds passed
... 9%, 11712 KB, 34946 KB/s, 0 seconds passed
... 9%, 11744 KB, 35014 KB/s, 0 seconds passed
... 9%, 11776 KB, 35077 KB/s, 0 seconds passed
... 9%, 11808 KB, 35144 KB/s, 0 seconds passed
... 9%, 11840 KB, 35212 KB/s, 0 seconds passed
... 9%, 11872 KB, 35274 KB/s, 0 seconds passed
... 9%, 11904 KB, 35336 KB/s, 0 seconds passed
... 9%, 11936 KB, 35403 KB/s, 0 seconds passed
... 9%, 11968 KB, 35465 KB/s, 0 seconds passed
... 9%, 12000 KB, 35532 KB/s, 0 seconds passed
... 9%, 12032 KB, 35599 KB/s, 0 seconds passed
... 9%, 12064 KB, 35666 KB/s, 0 seconds passed
... 9%, 12096 KB, 35727 KB/s, 0 seconds passed
... 9%, 12128 KB, 35794 KB/s, 0 seconds passed
... 9%, 12160 KB, 35860 KB/s, 0 seconds passed
... 9%, 12192 KB, 35927 KB/s, 0 seconds passed
... 9%, 12224 KB, 35993 KB/s, 0 seconds passed
... 9%, 12256 KB, 36054 KB/s, 0 seconds passed
... 9%, 12288 KB, 36120 KB/s, 0 seconds passed
... 9%, 12320 KB, 36186 KB/s, 0 seconds passed
... 9%, 12352 KB, 36247 KB/s, 0 seconds passed
... 9%, 12384 KB, 36313 KB/s, 0 seconds passed
... 9%, 12416 KB, 35106 KB/s, 0 seconds passed
... 9%, 12448 KB, 35156 KB/s, 0 seconds passed
... 9%, 12480 KB, 35206 KB/s, 0 seconds passed
... 9%, 12512 KB, 35261 KB/s, 0 seconds passed
... 9%, 12544 KB, 35315 KB/s, 0 seconds passed
... 9%, 12576 KB, 35367 KB/s, 0 seconds passed
... 10%, 12608 KB, 35418 KB/s, 0 seconds passed
... 10%, 12640 KB, 35472 KB/s, 0 seconds passed
... 10%, 12672 KB, 35525 KB/s, 0 seconds passed
... 10%, 12704 KB, 35577 KB/s, 0 seconds passed
... 10%, 12736 KB, 35633 KB/s, 0 seconds passed
... 10%, 12768 KB, 35693 KB/s, 0 seconds passed
... 10%, 12800 KB, 35754 KB/s, 0 seconds passed
... 10%, 12832 KB, 35757 KB/s, 0 seconds passed
... 10%, 12864 KB, 35806 KB/s, 0 seconds passed
... 10%, 12896 KB, 35854 KB/s, 0 seconds passed
... 10%, 12928 KB, 35904 KB/s, 0 seconds passed
... 10%, 12960 KB, 35955 KB/s, 0 seconds passed
... 10%, 12992 KB, 36007 KB/s, 0 seconds passed
... 10%, 13024 KB, 36060 KB/s, 0 seconds passed
... 10%, 13056 KB, 36110 KB/s, 0 seconds passed
... 10%, 13088 KB, 36163 KB/s, 0 seconds passed
... 10%, 13120 KB, 36216 KB/s, 0 seconds passed
... 10%, 13152 KB, 36268 KB/s, 0 seconds passed
... 10%, 13184 KB, 36319 KB/s, 0 seconds passed
... 10%, 13216 KB, 36371 KB/s, 0 seconds passed
... 10%, 13248 KB, 36423 KB/s, 0 seconds passed
... 10%, 13280 KB, 36472 KB/s, 0 seconds passed
... 10%, 13312 KB, 36523 KB/s, 0 seconds passed
... 10%, 13344 KB, 36574 KB/s, 0 seconds passed
... 10%, 13376 KB, 36626 KB/s, 0 seconds passed
... 10%, 13408 KB, 36678 KB/s, 0 seconds passed
... 10%, 13440 KB, 36729 KB/s, 0 seconds passed
... 10%, 13472 KB, 36779 KB/s, 0 seconds passed
... 10%, 13504 KB, 36828 KB/s, 0 seconds passed
... 10%, 13536 KB, 36879 KB/s, 0 seconds passed
... 10%, 13568 KB, 36930 KB/s, 0 seconds passed
... 10%, 13600 KB, 36979 KB/s, 0 seconds passed
... 10%, 13632 KB, 37026 KB/s, 0 seconds passed
... 10%, 13664 KB, 37077 KB/s, 0 seconds passed
... 10%, 13696 KB, 37126 KB/s, 0 seconds passed
... 10%, 13728 KB, 37188 KB/s, 0 seconds passed
... 10%, 13760 KB, 37250 KB/s, 0 seconds passed
... 10%, 13792 KB, 37312 KB/s, 0 seconds passed
... 10%, 13824 KB, 37375 KB/s, 0 seconds passed
... 11%, 13856 KB, 37436 KB/s, 0 seconds passed
... 11%, 13888 KB, 37498 KB/s, 0 seconds passed
... 11%, 13920 KB, 37560 KB/s, 0 seconds passed
... 11%, 13952 KB, 37620 KB/s, 0 seconds passed
... 11%, 13984 KB, 37682 KB/s, 0 seconds passed
... 11%, 14016 KB, 37744 KB/s, 0 seconds passed
... 11%, 14048 KB, 37805 KB/s, 0 seconds passed
... 11%, 14080 KB, 37865 KB/s, 0 seconds passed
... 11%, 14112 KB, 37926 KB/s, 0 seconds passed
... 11%, 14144 KB, 37988 KB/s, 0 seconds passed
... 11%, 14176 KB, 38049 KB/s, 0 seconds passed
... 11%, 14208 KB, 38109 KB/s, 0 seconds passed

.. parsed-literal::

    ... 11%, 14240 KB, 38170 KB/s, 0 seconds passed
... 11%, 14272 KB, 38231 KB/s, 0 seconds passed
... 11%, 14304 KB, 38290 KB/s, 0 seconds passed
... 11%, 14336 KB, 38349 KB/s, 0 seconds passed
... 11%, 14368 KB, 38409 KB/s, 0 seconds passed
... 11%, 14400 KB, 38470 KB/s, 0 seconds passed
... 11%, 14432 KB, 38525 KB/s, 0 seconds passed
... 11%, 14464 KB, 38582 KB/s, 0 seconds passed
... 11%, 14496 KB, 38642 KB/s, 0 seconds passed
... 11%, 14528 KB, 38702 KB/s, 0 seconds passed
... 11%, 14560 KB, 38760 KB/s, 0 seconds passed
... 11%, 14592 KB, 38817 KB/s, 0 seconds passed
... 11%, 14624 KB, 38877 KB/s, 0 seconds passed
... 11%, 14656 KB, 38935 KB/s, 0 seconds passed
... 11%, 14688 KB, 38994 KB/s, 0 seconds passed
... 11%, 14720 KB, 39053 KB/s, 0 seconds passed
... 11%, 14752 KB, 39111 KB/s, 0 seconds passed
... 11%, 14784 KB, 39171 KB/s, 0 seconds passed
... 11%, 14816 KB, 39231 KB/s, 0 seconds passed
... 11%, 14848 KB, 39290 KB/s, 0 seconds passed
... 11%, 14880 KB, 39349 KB/s, 0 seconds passed
... 11%, 14912 KB, 39408 KB/s, 0 seconds passed
... 11%, 14944 KB, 39467 KB/s, 0 seconds passed
... 11%, 14976 KB, 39524 KB/s, 0 seconds passed
... 11%, 15008 KB, 39584 KB/s, 0 seconds passed
... 11%, 15040 KB, 39651 KB/s, 0 seconds passed
... 11%, 15072 KB, 39718 KB/s, 0 seconds passed
... 11%, 15104 KB, 39786 KB/s, 0 seconds passed
... 12%, 15136 KB, 39852 KB/s, 0 seconds passed
... 12%, 15168 KB, 39918 KB/s, 0 seconds passed
... 12%, 15200 KB, 39984 KB/s, 0 seconds passed
... 12%, 15232 KB, 40050 KB/s, 0 seconds passed
... 12%, 15264 KB, 40116 KB/s, 0 seconds passed
... 12%, 15296 KB, 40183 KB/s, 0 seconds passed
... 12%, 15328 KB, 40251 KB/s, 0 seconds passed

.. parsed-literal::

    ... 12%, 15360 KB, 35278 KB/s, 0 seconds passed
... 12%, 15392 KB, 35312 KB/s, 0 seconds passed
... 12%, 15424 KB, 34729 KB/s, 0 seconds passed
... 12%, 15456 KB, 34763 KB/s, 0 seconds passed
... 12%, 15488 KB, 34803 KB/s, 0 seconds passed
... 12%, 15520 KB, 34844 KB/s, 0 seconds passed
... 12%, 15552 KB, 34883 KB/s, 0 seconds passed
... 12%, 15584 KB, 34931 KB/s, 0 seconds passed
... 12%, 15616 KB, 34560 KB/s, 0 seconds passed
... 12%, 15648 KB, 34597 KB/s, 0 seconds passed
... 12%, 15680 KB, 34579 KB/s, 0 seconds passed
... 12%, 15712 KB, 34619 KB/s, 0 seconds passed
... 12%, 15744 KB, 34660 KB/s, 0 seconds passed
... 12%, 15776 KB, 34702 KB/s, 0 seconds passed
... 12%, 15808 KB, 34743 KB/s, 0 seconds passed
... 12%, 15840 KB, 34785 KB/s, 0 seconds passed
... 12%, 15872 KB, 34826 KB/s, 0 seconds passed
... 12%, 15904 KB, 34864 KB/s, 0 seconds passed
... 12%, 15936 KB, 34905 KB/s, 0 seconds passed
... 12%, 15968 KB, 34945 KB/s, 0 seconds passed
... 12%, 16000 KB, 34985 KB/s, 0 seconds passed
... 12%, 16032 KB, 35026 KB/s, 0 seconds passed
... 12%, 16064 KB, 35067 KB/s, 0 seconds passed
... 12%, 16096 KB, 35108 KB/s, 0 seconds passed
... 12%, 16128 KB, 35149 KB/s, 0 seconds passed
... 12%, 16160 KB, 35188 KB/s, 0 seconds passed
... 12%, 16192 KB, 35229 KB/s, 0 seconds passed
... 12%, 16224 KB, 35266 KB/s, 0 seconds passed
... 12%, 16256 KB, 35307 KB/s, 0 seconds passed
... 12%, 16288 KB, 35347 KB/s, 0 seconds passed
... 12%, 16320 KB, 35388 KB/s, 0 seconds passed
... 12%, 16352 KB, 35428 KB/s, 0 seconds passed
... 13%, 16384 KB, 35467 KB/s, 0 seconds passed
... 13%, 16416 KB, 35508 KB/s, 0 seconds passed
... 13%, 16448 KB, 35547 KB/s, 0 seconds passed
... 13%, 16480 KB, 35586 KB/s, 0 seconds passed
... 13%, 16512 KB, 35626 KB/s, 0 seconds passed
... 13%, 16544 KB, 35661 KB/s, 0 seconds passed
... 13%, 16576 KB, 35697 KB/s, 0 seconds passed
... 13%, 16608 KB, 35738 KB/s, 0 seconds passed
... 13%, 16640 KB, 35776 KB/s, 0 seconds passed
... 13%, 16672 KB, 35817 KB/s, 0 seconds passed
... 13%, 16704 KB, 35856 KB/s, 0 seconds passed
... 13%, 16736 KB, 35895 KB/s, 0 seconds passed
... 13%, 16768 KB, 35935 KB/s, 0 seconds passed
... 13%, 16800 KB, 35973 KB/s, 0 seconds passed
... 13%, 16832 KB, 36012 KB/s, 0 seconds passed
... 13%, 16864 KB, 36050 KB/s, 0 seconds passed
... 13%, 16896 KB, 36089 KB/s, 0 seconds passed
... 13%, 16928 KB, 36127 KB/s, 0 seconds passed
... 13%, 16960 KB, 36167 KB/s, 0 seconds passed
... 13%, 16992 KB, 36205 KB/s, 0 seconds passed
... 13%, 17024 KB, 36243 KB/s, 0 seconds passed
... 13%, 17056 KB, 36282 KB/s, 0 seconds passed
... 13%, 17088 KB, 36321 KB/s, 0 seconds passed
... 13%, 17120 KB, 36371 KB/s, 0 seconds passed
... 13%, 17152 KB, 36419 KB/s, 0 seconds passed
... 13%, 17184 KB, 36469 KB/s, 0 seconds passed
... 13%, 17216 KB, 36518 KB/s, 0 seconds passed
... 13%, 17248 KB, 36565 KB/s, 0 seconds passed
... 13%, 17280 KB, 36613 KB/s, 0 seconds passed
... 13%, 17312 KB, 36661 KB/s, 0 seconds passed
... 13%, 17344 KB, 36711 KB/s, 0 seconds passed
... 13%, 17376 KB, 36760 KB/s, 0 seconds passed
... 13%, 17408 KB, 36809 KB/s, 0 seconds passed
... 13%, 17440 KB, 36858 KB/s, 0 seconds passed
... 13%, 17472 KB, 36906 KB/s, 0 seconds passed
... 13%, 17504 KB, 36955 KB/s, 0 seconds passed
... 13%, 17536 KB, 37003 KB/s, 0 seconds passed
... 13%, 17568 KB, 37052 KB/s, 0 seconds passed
... 13%, 17600 KB, 37101 KB/s, 0 seconds passed
... 13%, 17632 KB, 37150 KB/s, 0 seconds passed
... 14%, 17664 KB, 37197 KB/s, 0 seconds passed
... 14%, 17696 KB, 37244 KB/s, 0 seconds passed
... 14%, 17728 KB, 37292 KB/s, 0 seconds passed

.. parsed-literal::

    ... 14%, 17760 KB, 37340 KB/s, 0 seconds passed
... 14%, 17792 KB, 37387 KB/s, 0 seconds passed
... 14%, 17824 KB, 37434 KB/s, 0 seconds passed
... 14%, 17856 KB, 37482 KB/s, 0 seconds passed
... 14%, 17888 KB, 37531 KB/s, 0 seconds passed
... 14%, 17920 KB, 37578 KB/s, 0 seconds passed
... 14%, 17952 KB, 37625 KB/s, 0 seconds passed
... 14%, 17984 KB, 37670 KB/s, 0 seconds passed
... 14%, 18016 KB, 37717 KB/s, 0 seconds passed
... 14%, 18048 KB, 37765 KB/s, 0 seconds passed
... 14%, 18080 KB, 37813 KB/s, 0 seconds passed
... 14%, 18112 KB, 37861 KB/s, 0 seconds passed
... 14%, 18144 KB, 37908 KB/s, 0 seconds passed
... 14%, 18176 KB, 37957 KB/s, 0 seconds passed
... 14%, 18208 KB, 38005 KB/s, 0 seconds passed
... 14%, 18240 KB, 38052 KB/s, 0 seconds passed
... 14%, 18272 KB, 38100 KB/s, 0 seconds passed
... 14%, 18304 KB, 38146 KB/s, 0 seconds passed
... 14%, 18336 KB, 38193 KB/s, 0 seconds passed
... 14%, 18368 KB, 38246 KB/s, 0 seconds passed
... 14%, 18400 KB, 38299 KB/s, 0 seconds passed
... 14%, 18432 KB, 38353 KB/s, 0 seconds passed
... 14%, 18464 KB, 38406 KB/s, 0 seconds passed
... 14%, 18496 KB, 38458 KB/s, 0 seconds passed
... 14%, 18528 KB, 38511 KB/s, 0 seconds passed
... 14%, 18560 KB, 38564 KB/s, 0 seconds passed
... 14%, 18592 KB, 38617 KB/s, 0 seconds passed
... 14%, 18624 KB, 38670 KB/s, 0 seconds passed
... 14%, 18656 KB, 38722 KB/s, 0 seconds passed
... 14%, 18688 KB, 38775 KB/s, 0 seconds passed
... 14%, 18720 KB, 38828 KB/s, 0 seconds passed
... 14%, 18752 KB, 38880 KB/s, 0 seconds passed
... 14%, 18784 KB, 38933 KB/s, 0 seconds passed
... 14%, 18816 KB, 38986 KB/s, 0 seconds passed
... 14%, 18848 KB, 39038 KB/s, 0 seconds passed
... 14%, 18880 KB, 39090 KB/s, 0 seconds passed
... 15%, 18912 KB, 39141 KB/s, 0 seconds passed
... 15%, 18944 KB, 39194 KB/s, 0 seconds passed
... 15%, 18976 KB, 39247 KB/s, 0 seconds passed
... 15%, 19008 KB, 39299 KB/s, 0 seconds passed
... 15%, 19040 KB, 39351 KB/s, 0 seconds passed
... 15%, 19072 KB, 39404 KB/s, 0 seconds passed
... 15%, 19104 KB, 39456 KB/s, 0 seconds passed
... 15%, 19136 KB, 39509 KB/s, 0 seconds passed
... 15%, 19168 KB, 39561 KB/s, 0 seconds passed
... 15%, 19200 KB, 39614 KB/s, 0 seconds passed
... 15%, 19232 KB, 39666 KB/s, 0 seconds passed
... 15%, 19264 KB, 39719 KB/s, 0 seconds passed
... 15%, 19296 KB, 39771 KB/s, 0 seconds passed
... 15%, 19328 KB, 39824 KB/s, 0 seconds passed
... 15%, 19360 KB, 39875 KB/s, 0 seconds passed
... 15%, 19392 KB, 39922 KB/s, 0 seconds passed
... 15%, 19424 KB, 39967 KB/s, 0 seconds passed
... 15%, 19456 KB, 40010 KB/s, 0 seconds passed
... 15%, 19488 KB, 40050 KB/s, 0 seconds passed
... 15%, 19520 KB, 40095 KB/s, 0 seconds passed
... 15%, 19552 KB, 40139 KB/s, 0 seconds passed
... 15%, 19584 KB, 40178 KB/s, 0 seconds passed
... 15%, 19616 KB, 40223 KB/s, 0 seconds passed
... 15%, 19648 KB, 40266 KB/s, 0 seconds passed
... 15%, 19680 KB, 40310 KB/s, 0 seconds passed
... 15%, 19712 KB, 40350 KB/s, 0 seconds passed
... 15%, 19744 KB, 40394 KB/s, 0 seconds passed
... 15%, 19776 KB, 40433 KB/s, 0 seconds passed
... 15%, 19808 KB, 40477 KB/s, 0 seconds passed
... 15%, 19840 KB, 40521 KB/s, 0 seconds passed
... 15%, 19872 KB, 40560 KB/s, 0 seconds passed
... 15%, 19904 KB, 40603 KB/s, 0 seconds passed
... 15%, 19936 KB, 40647 KB/s, 0 seconds passed
... 15%, 19968 KB, 40686 KB/s, 0 seconds passed
... 15%, 20000 KB, 40730 KB/s, 0 seconds passed
... 15%, 20032 KB, 40773 KB/s, 0 seconds passed
... 15%, 20064 KB, 40812 KB/s, 0 seconds passed
... 15%, 20096 KB, 40856 KB/s, 0 seconds passed
... 15%, 20128 KB, 40899 KB/s, 0 seconds passed
... 16%, 20160 KB, 40942 KB/s, 0 seconds passed
... 16%, 20192 KB, 40981 KB/s, 0 seconds passed
... 16%, 20224 KB, 41024 KB/s, 0 seconds passed
... 16%, 20256 KB, 41063 KB/s, 0 seconds passed
... 16%, 20288 KB, 41106 KB/s, 0 seconds passed
... 16%, 20320 KB, 41149 KB/s, 0 seconds passed
... 16%, 20352 KB, 41188 KB/s, 0 seconds passed
... 16%, 20384 KB, 41231 KB/s, 0 seconds passed
... 16%, 20416 KB, 41274 KB/s, 0 seconds passed
... 16%, 20448 KB, 41317 KB/s, 0 seconds passed
... 16%, 20480 KB, 39300 KB/s, 0 seconds passed
... 16%, 20512 KB, 39250 KB/s, 0 seconds passed
... 16%, 20544 KB, 39283 KB/s, 0 seconds passed
... 16%, 20576 KB, 39316 KB/s, 0 seconds passed
... 16%, 20608 KB, 39349 KB/s, 0 seconds passed
... 16%, 20640 KB, 39380 KB/s, 0 seconds passed
... 16%, 20672 KB, 39413 KB/s, 0 seconds passed
... 16%, 20704 KB, 39443 KB/s, 0 seconds passed
... 16%, 20736 KB, 39475 KB/s, 0 seconds passed
... 16%, 20768 KB, 39513 KB/s, 0 seconds passed

.. parsed-literal::

    ... 16%, 20800 KB, 39097 KB/s, 0 seconds passed
... 16%, 20832 KB, 39123 KB/s, 0 seconds passed
... 16%, 20864 KB, 38801 KB/s, 0 seconds passed
... 16%, 20896 KB, 38825 KB/s, 0 seconds passed
... 16%, 20928 KB, 38615 KB/s, 0 seconds passed
... 16%, 20960 KB, 38639 KB/s, 0 seconds passed
... 16%, 20992 KB, 38669 KB/s, 0 seconds passed
... 16%, 21024 KB, 38705 KB/s, 0 seconds passed
... 16%, 21056 KB, 38731 KB/s, 0 seconds passed
... 16%, 21088 KB, 38758 KB/s, 0 seconds passed
... 16%, 21120 KB, 38770 KB/s, 0 seconds passed
... 16%, 21152 KB, 38801 KB/s, 0 seconds passed
... 16%, 21184 KB, 38832 KB/s, 0 seconds passed
... 16%, 21216 KB, 38863 KB/s, 0 seconds passed
... 16%, 21248 KB, 38894 KB/s, 0 seconds passed
... 16%, 21280 KB, 38927 KB/s, 0 seconds passed
... 16%, 21312 KB, 38958 KB/s, 0 seconds passed
... 16%, 21344 KB, 38988 KB/s, 0 seconds passed
... 16%, 21376 KB, 39019 KB/s, 0 seconds passed
... 16%, 21408 KB, 39049 KB/s, 0 seconds passed
... 17%, 21440 KB, 39080 KB/s, 0 seconds passed
... 17%, 21472 KB, 39111 KB/s, 0 seconds passed
... 17%, 21504 KB, 39141 KB/s, 0 seconds passed
... 17%, 21536 KB, 39171 KB/s, 0 seconds passed
... 17%, 21568 KB, 39203 KB/s, 0 seconds passed
... 17%, 21600 KB, 39235 KB/s, 0 seconds passed
... 17%, 21632 KB, 39265 KB/s, 0 seconds passed
... 17%, 21664 KB, 39296 KB/s, 0 seconds passed
... 17%, 21696 KB, 39329 KB/s, 0 seconds passed
... 17%, 21728 KB, 39363 KB/s, 0 seconds passed
... 17%, 21760 KB, 39398 KB/s, 0 seconds passed
... 17%, 21792 KB, 39435 KB/s, 0 seconds passed
... 17%, 21824 KB, 39470 KB/s, 0 seconds passed
... 17%, 21856 KB, 39507 KB/s, 0 seconds passed
... 17%, 21888 KB, 39542 KB/s, 0 seconds passed
... 17%, 21920 KB, 39578 KB/s, 0 seconds passed
... 17%, 21952 KB, 39613 KB/s, 0 seconds passed
... 17%, 21984 KB, 39648 KB/s, 0 seconds passed
... 17%, 22016 KB, 39683 KB/s, 0 seconds passed
... 17%, 22048 KB, 39718 KB/s, 0 seconds passed
... 17%, 22080 KB, 39754 KB/s, 0 seconds passed
... 17%, 22112 KB, 39790 KB/s, 0 seconds passed
... 17%, 22144 KB, 39825 KB/s, 0 seconds passed
... 17%, 22176 KB, 39861 KB/s, 0 seconds passed
... 17%, 22208 KB, 39895 KB/s, 0 seconds passed
... 17%, 22240 KB, 39931 KB/s, 0 seconds passed
... 17%, 22272 KB, 39966 KB/s, 0 seconds passed
... 17%, 22304 KB, 40002 KB/s, 0 seconds passed
... 17%, 22336 KB, 40038 KB/s, 0 seconds passed
... 17%, 22368 KB, 40072 KB/s, 0 seconds passed
... 17%, 22400 KB, 40108 KB/s, 0 seconds passed
... 17%, 22432 KB, 40144 KB/s, 0 seconds passed
... 17%, 22464 KB, 40179 KB/s, 0 seconds passed
... 17%, 22496 KB, 40215 KB/s, 0 seconds passed
... 17%, 22528 KB, 40250 KB/s, 0 seconds passed
... 17%, 22560 KB, 40283 KB/s, 0 seconds passed
... 17%, 22592 KB, 40318 KB/s, 0 seconds passed
... 17%, 22624 KB, 40353 KB/s, 0 seconds passed
... 17%, 22656 KB, 40389 KB/s, 0 seconds passed
... 18%, 22688 KB, 40425 KB/s, 0 seconds passed
... 18%, 22720 KB, 40461 KB/s, 0 seconds passed
... 18%, 22752 KB, 40501 KB/s, 0 seconds passed
... 18%, 22784 KB, 40543 KB/s, 0 seconds passed
... 18%, 22816 KB, 40585 KB/s, 0 seconds passed
... 18%, 22848 KB, 40628 KB/s, 0 seconds passed
... 18%, 22880 KB, 40670 KB/s, 0 seconds passed
... 18%, 22912 KB, 40712 KB/s, 0 seconds passed
... 18%, 22944 KB, 40753 KB/s, 0 seconds passed
... 18%, 22976 KB, 40796 KB/s, 0 seconds passed
... 18%, 23008 KB, 40837 KB/s, 0 seconds passed
... 18%, 23040 KB, 40879 KB/s, 0 seconds passed
... 18%, 23072 KB, 40920 KB/s, 0 seconds passed
... 18%, 23104 KB, 40963 KB/s, 0 seconds passed
... 18%, 23136 KB, 41005 KB/s, 0 seconds passed
... 18%, 23168 KB, 41046 KB/s, 0 seconds passed
... 18%, 23200 KB, 41088 KB/s, 0 seconds passed
... 18%, 23232 KB, 41129 KB/s, 0 seconds passed
... 18%, 23264 KB, 41170 KB/s, 0 seconds passed
... 18%, 23296 KB, 41211 KB/s, 0 seconds passed
... 18%, 23328 KB, 41253 KB/s, 0 seconds passed
... 18%, 23360 KB, 41294 KB/s, 0 seconds passed
... 18%, 23392 KB, 41335 KB/s, 0 seconds passed
... 18%, 23424 KB, 41377 KB/s, 0 seconds passed
... 18%, 23456 KB, 41419 KB/s, 0 seconds passed
... 18%, 23488 KB, 41459 KB/s, 0 seconds passed
... 18%, 23520 KB, 41501 KB/s, 0 seconds passed
... 18%, 23552 KB, 41543 KB/s, 0 seconds passed
... 18%, 23584 KB, 41585 KB/s, 0 seconds passed
... 18%, 23616 KB, 41626 KB/s, 0 seconds passed
... 18%, 23648 KB, 41668 KB/s, 0 seconds passed
... 18%, 23680 KB, 41710 KB/s, 0 seconds passed
... 18%, 23712 KB, 41749 KB/s, 0 seconds passed
... 18%, 23744 KB, 41791 KB/s, 0 seconds passed
... 18%, 23776 KB, 41832 KB/s, 0 seconds passed
... 18%, 23808 KB, 41874 KB/s, 0 seconds passed
... 18%, 23840 KB, 41915 KB/s, 0 seconds passed
... 18%, 23872 KB, 41956 KB/s, 0 seconds passed
... 18%, 23904 KB, 41997 KB/s, 0 seconds passed
... 19%, 23936 KB, 42039 KB/s, 0 seconds passed
... 19%, 23968 KB, 42081 KB/s, 0 seconds passed
... 19%, 24000 KB, 42122 KB/s, 0 seconds passed
... 19%, 24032 KB, 42155 KB/s, 0 seconds passed
... 19%, 24064 KB, 42191 KB/s, 0 seconds passed
... 19%, 24096 KB, 42224 KB/s, 0 seconds passed
... 19%, 24128 KB, 42261 KB/s, 0 seconds passed
... 19%, 24160 KB, 42297 KB/s, 0 seconds passed
... 19%, 24192 KB, 42330 KB/s, 0 seconds passed
... 19%, 24224 KB, 42371 KB/s, 0 seconds passed
... 19%, 24256 KB, 42404 KB/s, 0 seconds passed
... 19%, 24288 KB, 42440 KB/s, 0 seconds passed
... 19%, 24320 KB, 42477 KB/s, 0 seconds passed
... 19%, 24352 KB, 42509 KB/s, 0 seconds passed
... 19%, 24384 KB, 42545 KB/s, 0 seconds passed
... 19%, 24416 KB, 42578 KB/s, 0 seconds passed
... 19%, 24448 KB, 42614 KB/s, 0 seconds passed
... 19%, 24480 KB, 42650 KB/s, 0 seconds passed
... 19%, 24512 KB, 42683 KB/s, 0 seconds passed
... 19%, 24544 KB, 42719 KB/s, 0 seconds passed
... 19%, 24576 KB, 42755 KB/s, 0 seconds passed
... 19%, 24608 KB, 42787 KB/s, 0 seconds passed
... 19%, 24640 KB, 42824 KB/s, 0 seconds passed
... 19%, 24672 KB, 42860 KB/s, 0 seconds passed
... 19%, 24704 KB, 42892 KB/s, 0 seconds passed
... 19%, 24736 KB, 42928 KB/s, 0 seconds passed

.. parsed-literal::

    ... 19%, 24768 KB, 41324 KB/s, 0 seconds passed
... 19%, 24800 KB, 41348 KB/s, 0 seconds passed
... 19%, 24832 KB, 41375 KB/s, 0 seconds passed
... 19%, 24864 KB, 41403 KB/s, 0 seconds passed
... 19%, 24896 KB, 41429 KB/s, 0 seconds passed
... 19%, 24928 KB, 41457 KB/s, 0 seconds passed
... 19%, 24960 KB, 41485 KB/s, 0 seconds passed
... 19%, 24992 KB, 41513 KB/s, 0 seconds passed
... 19%, 25024 KB, 41541 KB/s, 0 seconds passed
... 19%, 25056 KB, 41569 KB/s, 0 seconds passed
... 19%, 25088 KB, 41595 KB/s, 0 seconds passed
... 19%, 25120 KB, 41622 KB/s, 0 seconds passed
... 19%, 25152 KB, 41646 KB/s, 0 seconds passed
... 19%, 25184 KB, 41674 KB/s, 0 seconds passed
... 20%, 25216 KB, 41701 KB/s, 0 seconds passed
... 20%, 25248 KB, 41728 KB/s, 0 seconds passed
... 20%, 25280 KB, 41754 KB/s, 0 seconds passed
... 20%, 25312 KB, 41781 KB/s, 0 seconds passed
... 20%, 25344 KB, 41809 KB/s, 0 seconds passed
... 20%, 25376 KB, 41839 KB/s, 0 seconds passed
... 20%, 25408 KB, 41870 KB/s, 0 seconds passed
... 20%, 25440 KB, 41903 KB/s, 0 seconds passed
... 20%, 25472 KB, 41936 KB/s, 0 seconds passed
... 20%, 25504 KB, 41970 KB/s, 0 seconds passed
... 20%, 25536 KB, 42001 KB/s, 0 seconds passed
... 20%, 25568 KB, 42034 KB/s, 0 seconds passed

.. parsed-literal::

    ... 20%, 25600 KB, 37645 KB/s, 0 seconds passed
... 20%, 25632 KB, 37661 KB/s, 0 seconds passed

.. parsed-literal::

    ... 20%, 25664 KB, 37685 KB/s, 0 seconds passed
... 20%, 25696 KB, 37712 KB/s, 0 seconds passed
... 20%, 25728 KB, 37221 KB/s, 0 seconds passed
... 20%, 25760 KB, 37162 KB/s, 0 seconds passed
... 20%, 25792 KB, 37180 KB/s, 0 seconds passed
... 20%, 25824 KB, 37202 KB/s, 0 seconds passed
... 20%, 25856 KB, 37227 KB/s, 0 seconds passed
... 20%, 25888 KB, 37253 KB/s, 0 seconds passed
... 20%, 25920 KB, 37279 KB/s, 0 seconds passed
... 20%, 25952 KB, 37305 KB/s, 0 seconds passed
... 20%, 25984 KB, 37329 KB/s, 0 seconds passed
... 20%, 26016 KB, 37356 KB/s, 0 seconds passed
... 20%, 26048 KB, 37381 KB/s, 0 seconds passed
... 20%, 26080 KB, 37409 KB/s, 0 seconds passed
... 20%, 26112 KB, 37439 KB/s, 0 seconds passed
... 20%, 26144 KB, 37467 KB/s, 0 seconds passed
... 20%, 26176 KB, 37494 KB/s, 0 seconds passed
... 20%, 26208 KB, 37456 KB/s, 0 seconds passed
... 20%, 26240 KB, 37473 KB/s, 0 seconds passed
... 20%, 26272 KB, 37497 KB/s, 0 seconds passed
... 20%, 26304 KB, 37521 KB/s, 0 seconds passed
... 20%, 26336 KB, 37547 KB/s, 0 seconds passed
... 20%, 26368 KB, 37571 KB/s, 0 seconds passed
... 20%, 26400 KB, 37597 KB/s, 0 seconds passed
... 20%, 26432 KB, 37624 KB/s, 0 seconds passed
... 21%, 26464 KB, 37651 KB/s, 0 seconds passed
... 21%, 26496 KB, 37678 KB/s, 0 seconds passed
... 21%, 26528 KB, 37704 KB/s, 0 seconds passed
... 21%, 26560 KB, 37730 KB/s, 0 seconds passed
... 21%, 26592 KB, 37760 KB/s, 0 seconds passed
... 21%, 26624 KB, 37787 KB/s, 0 seconds passed
... 21%, 26656 KB, 37814 KB/s, 0 seconds passed
... 21%, 26688 KB, 37841 KB/s, 0 seconds passed
... 21%, 26720 KB, 37868 KB/s, 0 seconds passed
... 21%, 26752 KB, 37897 KB/s, 0 seconds passed
... 21%, 26784 KB, 37924 KB/s, 0 seconds passed
... 21%, 26816 KB, 37951 KB/s, 0 seconds passed
... 21%, 26848 KB, 37978 KB/s, 0 seconds passed
... 21%, 26880 KB, 38005 KB/s, 0 seconds passed
... 21%, 26912 KB, 38032 KB/s, 0 seconds passed
... 21%, 26944 KB, 38058 KB/s, 0 seconds passed
... 21%, 26976 KB, 38062 KB/s, 0 seconds passed
... 21%, 27008 KB, 38089 KB/s, 0 seconds passed
... 21%, 27040 KB, 38116 KB/s, 0 seconds passed
... 21%, 27072 KB, 38142 KB/s, 0 seconds passed
... 21%, 27104 KB, 38168 KB/s, 0 seconds passed
... 21%, 27136 KB, 38195 KB/s, 0 seconds passed
... 21%, 27168 KB, 38221 KB/s, 0 seconds passed
... 21%, 27200 KB, 38247 KB/s, 0 seconds passed
... 21%, 27232 KB, 38274 KB/s, 0 seconds passed
... 21%, 27264 KB, 38307 KB/s, 0 seconds passed
... 21%, 27296 KB, 38338 KB/s, 0 seconds passed
... 21%, 27328 KB, 38370 KB/s, 0 seconds passed
... 21%, 27360 KB, 38404 KB/s, 0 seconds passed
... 21%, 27392 KB, 37768 KB/s, 0 seconds passed
... 21%, 27424 KB, 37763 KB/s, 0 seconds passed
... 21%, 27456 KB, 37776 KB/s, 0 seconds passed
... 21%, 27488 KB, 37799 KB/s, 0 seconds passed
... 21%, 27520 KB, 37824 KB/s, 0 seconds passed
... 21%, 27552 KB, 37846 KB/s, 0 seconds passed
... 21%, 27584 KB, 37870 KB/s, 0 seconds passed
... 21%, 27616 KB, 37891 KB/s, 0 seconds passed
... 21%, 27648 KB, 37915 KB/s, 0 seconds passed
... 21%, 27680 KB, 37939 KB/s, 0 seconds passed
... 22%, 27712 KB, 37964 KB/s, 0 seconds passed
... 22%, 27744 KB, 37989 KB/s, 0 seconds passed
... 22%, 27776 KB, 38013 KB/s, 0 seconds passed

.. parsed-literal::

    ... 22%, 27808 KB, 37975 KB/s, 0 seconds passed
... 22%, 27840 KB, 37996 KB/s, 0 seconds passed
... 22%, 27872 KB, 38018 KB/s, 0 seconds passed
... 22%, 27904 KB, 38042 KB/s, 0 seconds passed
... 22%, 27936 KB, 38063 KB/s, 0 seconds passed
... 22%, 27968 KB, 38087 KB/s, 0 seconds passed
... 22%, 28000 KB, 38111 KB/s, 0 seconds passed
... 22%, 28032 KB, 38135 KB/s, 0 seconds passed
... 22%, 28064 KB, 38159 KB/s, 0 seconds passed
... 22%, 28096 KB, 38181 KB/s, 0 seconds passed
... 22%, 28128 KB, 38205 KB/s, 0 seconds passed
... 22%, 28160 KB, 38232 KB/s, 0 seconds passed
... 22%, 28192 KB, 38259 KB/s, 0 seconds passed
... 22%, 28224 KB, 38283 KB/s, 0 seconds passed
... 22%, 28256 KB, 38308 KB/s, 0 seconds passed
... 22%, 28288 KB, 38335 KB/s, 0 seconds passed
... 22%, 28320 KB, 38363 KB/s, 0 seconds passed
... 22%, 28352 KB, 38311 KB/s, 0 seconds passed
... 22%, 28384 KB, 38333 KB/s, 0 seconds passed
... 22%, 28416 KB, 38317 KB/s, 0 seconds passed
... 22%, 28448 KB, 38337 KB/s, 0 seconds passed
... 22%, 28480 KB, 38360 KB/s, 0 seconds passed
... 22%, 28512 KB, 38383 KB/s, 0 seconds passed
... 22%, 28544 KB, 38406 KB/s, 0 seconds passed
... 22%, 28576 KB, 38430 KB/s, 0 seconds passed
... 22%, 28608 KB, 38451 KB/s, 0 seconds passed
... 22%, 28640 KB, 38474 KB/s, 0 seconds passed
... 22%, 28672 KB, 38499 KB/s, 0 seconds passed
... 22%, 28704 KB, 38523 KB/s, 0 seconds passed
... 22%, 28736 KB, 38546 KB/s, 0 seconds passed
... 22%, 28768 KB, 38569 KB/s, 0 seconds passed
... 22%, 28800 KB, 38593 KB/s, 0 seconds passed
... 22%, 28832 KB, 38615 KB/s, 0 seconds passed
... 22%, 28864 KB, 38638 KB/s, 0 seconds passed
... 22%, 28896 KB, 38662 KB/s, 0 seconds passed
... 22%, 28928 KB, 38685 KB/s, 0 seconds passed
... 22%, 28960 KB, 38707 KB/s, 0 seconds passed
... 23%, 28992 KB, 38730 KB/s, 0 seconds passed
... 23%, 29024 KB, 38753 KB/s, 0 seconds passed
... 23%, 29056 KB, 38776 KB/s, 0 seconds passed
... 23%, 29088 KB, 38799 KB/s, 0 seconds passed
... 23%, 29120 KB, 38821 KB/s, 0 seconds passed
... 23%, 29152 KB, 38844 KB/s, 0 seconds passed
... 23%, 29184 KB, 38868 KB/s, 0 seconds passed
... 23%, 29216 KB, 38890 KB/s, 0 seconds passed
... 23%, 29248 KB, 38913 KB/s, 0 seconds passed
... 23%, 29280 KB, 38935 KB/s, 0 seconds passed
... 23%, 29312 KB, 38958 KB/s, 0 seconds passed
... 23%, 29344 KB, 38981 KB/s, 0 seconds passed
... 23%, 29376 KB, 39010 KB/s, 0 seconds passed
... 23%, 29408 KB, 39040 KB/s, 0 seconds passed
... 23%, 29440 KB, 39070 KB/s, 0 seconds passed
... 23%, 29472 KB, 39099 KB/s, 0 seconds passed
... 23%, 29504 KB, 39128 KB/s, 0 seconds passed
... 23%, 29536 KB, 39158 KB/s, 0 seconds passed
... 23%, 29568 KB, 39187 KB/s, 0 seconds passed
... 23%, 29600 KB, 39217 KB/s, 0 seconds passed
... 23%, 29632 KB, 39246 KB/s, 0 seconds passed
... 23%, 29664 KB, 39276 KB/s, 0 seconds passed
... 23%, 29696 KB, 39306 KB/s, 0 seconds passed
... 23%, 29728 KB, 39336 KB/s, 0 seconds passed
... 23%, 29760 KB, 39365 KB/s, 0 seconds passed
... 23%, 29792 KB, 39395 KB/s, 0 seconds passed
... 23%, 29824 KB, 39424 KB/s, 0 seconds passed
... 23%, 29856 KB, 39454 KB/s, 0 seconds passed
... 23%, 29888 KB, 39483 KB/s, 0 seconds passed
... 23%, 29920 KB, 39513 KB/s, 0 seconds passed
... 23%, 29952 KB, 39542 KB/s, 0 seconds passed
... 23%, 29984 KB, 39571 KB/s, 0 seconds passed
... 23%, 30016 KB, 39601 KB/s, 0 seconds passed
... 23%, 30048 KB, 39631 KB/s, 0 seconds passed
... 23%, 30080 KB, 39660 KB/s, 0 seconds passed
... 23%, 30112 KB, 39690 KB/s, 0 seconds passed
... 23%, 30144 KB, 39719 KB/s, 0 seconds passed
... 23%, 30176 KB, 39748 KB/s, 0 seconds passed
... 23%, 30208 KB, 39777 KB/s, 0 seconds passed
... 24%, 30240 KB, 39807 KB/s, 0 seconds passed
... 24%, 30272 KB, 39835 KB/s, 0 seconds passed
... 24%, 30304 KB, 39865 KB/s, 0 seconds passed
... 24%, 30336 KB, 39893 KB/s, 0 seconds passed
... 24%, 30368 KB, 39923 KB/s, 0 seconds passed
... 24%, 30400 KB, 39952 KB/s, 0 seconds passed
... 24%, 30432 KB, 39981 KB/s, 0 seconds passed
... 24%, 30464 KB, 40010 KB/s, 0 seconds passed
... 24%, 30496 KB, 40040 KB/s, 0 seconds passed
... 24%, 30528 KB, 40072 KB/s, 0 seconds passed
... 24%, 30560 KB, 40104 KB/s, 0 seconds passed
... 24%, 30592 KB, 40136 KB/s, 0 seconds passed
... 24%, 30624 KB, 40167 KB/s, 0 seconds passed
... 24%, 30656 KB, 40200 KB/s, 0 seconds passed
... 24%, 30688 KB, 40234 KB/s, 0 seconds passed

.. parsed-literal::

    ... 24%, 30720 KB, 37318 KB/s, 0 seconds passed
... 24%, 30752 KB, 37333 KB/s, 0 seconds passed
... 24%, 30784 KB, 37352 KB/s, 0 seconds passed
... 24%, 30816 KB, 37374 KB/s, 0 seconds passed
... 24%, 30848 KB, 37399 KB/s, 0 seconds passed
... 24%, 30880 KB, 37395 KB/s, 0 seconds passed
... 24%, 30912 KB, 37413 KB/s, 0 seconds passed
... 24%, 30944 KB, 37433 KB/s, 0 seconds passed
... 24%, 30976 KB, 37454 KB/s, 0 seconds passed
... 24%, 31008 KB, 37476 KB/s, 0 seconds passed
... 24%, 31040 KB, 37496 KB/s, 0 seconds passed
... 24%, 31072 KB, 37517 KB/s, 0 seconds passed
... 24%, 31104 KB, 37538 KB/s, 0 seconds passed
... 24%, 31136 KB, 37559 KB/s, 0 seconds passed
... 24%, 31168 KB, 37581 KB/s, 0 seconds passed
... 24%, 31200 KB, 37603 KB/s, 0 seconds passed
... 24%, 31232 KB, 37624 KB/s, 0 seconds passed
... 24%, 31264 KB, 37645 KB/s, 0 seconds passed
... 24%, 31296 KB, 37666 KB/s, 0 seconds passed
... 24%, 31328 KB, 37687 KB/s, 0 seconds passed
... 24%, 31360 KB, 37709 KB/s, 0 seconds passed
... 24%, 31392 KB, 37728 KB/s, 0 seconds passed
... 24%, 31424 KB, 37749 KB/s, 0 seconds passed
... 24%, 31456 KB, 37771 KB/s, 0 seconds passed
... 24%, 31488 KB, 37792 KB/s, 0 seconds passed
... 25%, 31520 KB, 37816 KB/s, 0 seconds passed
... 25%, 31552 KB, 37840 KB/s, 0 seconds passed
... 25%, 31584 KB, 37864 KB/s, 0 seconds passed
... 25%, 31616 KB, 37889 KB/s, 0 seconds passed
... 25%, 31648 KB, 37913 KB/s, 0 seconds passed

.. parsed-literal::

    ... 25%, 31680 KB, 37937 KB/s, 0 seconds passed
... 25%, 31712 KB, 37961 KB/s, 0 seconds passed
... 25%, 31744 KB, 37985 KB/s, 0 seconds passed
... 25%, 31776 KB, 38009 KB/s, 0 seconds passed
... 25%, 31808 KB, 38032 KB/s, 0 seconds passed
... 25%, 31840 KB, 38054 KB/s, 0 seconds passed
... 25%, 31872 KB, 38078 KB/s, 0 seconds passed
... 25%, 31904 KB, 38101 KB/s, 0 seconds passed
... 25%, 31936 KB, 38125 KB/s, 0 seconds passed
... 25%, 31968 KB, 38149 KB/s, 0 seconds passed
... 25%, 32000 KB, 38173 KB/s, 0 seconds passed
... 25%, 32032 KB, 38197 KB/s, 0 seconds passed
... 25%, 32064 KB, 38221 KB/s, 0 seconds passed
... 25%, 32096 KB, 38243 KB/s, 0 seconds passed
... 25%, 32128 KB, 38267 KB/s, 0 seconds passed
... 25%, 32160 KB, 38290 KB/s, 0 seconds passed
... 25%, 32192 KB, 38314 KB/s, 0 seconds passed
... 25%, 32224 KB, 38338 KB/s, 0 seconds passed
... 25%, 32256 KB, 38363 KB/s, 0 seconds passed
... 25%, 32288 KB, 38387 KB/s, 0 seconds passed
... 25%, 32320 KB, 38411 KB/s, 0 seconds passed
... 25%, 32352 KB, 38436 KB/s, 0 seconds passed
... 25%, 32384 KB, 38460 KB/s, 0 seconds passed
... 25%, 32416 KB, 38484 KB/s, 0 seconds passed
... 25%, 32448 KB, 38507 KB/s, 0 seconds passed
... 25%, 32480 KB, 38532 KB/s, 0 seconds passed
... 25%, 32512 KB, 38556 KB/s, 0 seconds passed
... 25%, 32544 KB, 38580 KB/s, 0 seconds passed
... 25%, 32576 KB, 38609 KB/s, 0 seconds passed
... 25%, 32608 KB, 38636 KB/s, 0 seconds passed
... 25%, 32640 KB, 38666 KB/s, 0 seconds passed
... 25%, 32672 KB, 38696 KB/s, 0 seconds passed
... 25%, 32704 KB, 38726 KB/s, 0 seconds passed
... 25%, 32736 KB, 38756 KB/s, 0 seconds passed
... 26%, 32768 KB, 38786 KB/s, 0 seconds passed
... 26%, 32800 KB, 38816 KB/s, 0 seconds passed
... 26%, 32832 KB, 38847 KB/s, 0 seconds passed
... 26%, 32864 KB, 38877 KB/s, 0 seconds passed
... 26%, 32896 KB, 38907 KB/s, 0 seconds passed
... 26%, 32928 KB, 37835 KB/s, 0 seconds passed

.. parsed-literal::

    ... 26%, 32960 KB, 37058 KB/s, 0 seconds passed
... 26%, 32992 KB, 36692 KB/s, 0 seconds passed
... 26%, 33024 KB, 36708 KB/s, 0 seconds passed
... 26%, 33056 KB, 36343 KB/s, 0 seconds passed
... 26%, 33088 KB, 36359 KB/s, 0 seconds passed
... 26%, 33120 KB, 36377 KB/s, 0 seconds passed
... 26%, 33152 KB, 36398 KB/s, 0 seconds passed
... 26%, 33184 KB, 36179 KB/s, 0 seconds passed
... 26%, 33216 KB, 36193 KB/s, 0 seconds passed
... 26%, 33248 KB, 36212 KB/s, 0 seconds passed
... 26%, 33280 KB, 36232 KB/s, 0 seconds passed
... 26%, 33312 KB, 36255 KB/s, 0 seconds passed
... 26%, 33344 KB, 36263 KB/s, 0 seconds passed
... 26%, 33376 KB, 36280 KB/s, 0 seconds passed
... 26%, 33408 KB, 36300 KB/s, 0 seconds passed
... 26%, 33440 KB, 36322 KB/s, 0 seconds passed
... 26%, 33472 KB, 36061 KB/s, 0 seconds passed
... 26%, 33504 KB, 36076 KB/s, 0 seconds passed
... 26%, 33536 KB, 36096 KB/s, 0 seconds passed
... 26%, 33568 KB, 36116 KB/s, 0 seconds passed
... 26%, 33600 KB, 36134 KB/s, 0 seconds passed
... 26%, 33632 KB, 36154 KB/s, 0 seconds passed
... 26%, 33664 KB, 36173 KB/s, 0 seconds passed
... 26%, 33696 KB, 36193 KB/s, 0 seconds passed
... 26%, 33728 KB, 36212 KB/s, 0 seconds passed
... 26%, 33760 KB, 36230 KB/s, 0 seconds passed
... 26%, 33792 KB, 36248 KB/s, 0 seconds passed
... 26%, 33824 KB, 36270 KB/s, 0 seconds passed
... 26%, 33856 KB, 36291 KB/s, 0 seconds passed
... 26%, 33888 KB, 36313 KB/s, 0 seconds passed
... 26%, 33920 KB, 36336 KB/s, 0 seconds passed

.. parsed-literal::

    ... 26%, 33952 KB, 36145 KB/s, 0 seconds passed
... 26%, 33984 KB, 36159 KB/s, 0 seconds passed
... 27%, 34016 KB, 36178 KB/s, 0 seconds passed
... 27%, 34048 KB, 36198 KB/s, 0 seconds passed
... 27%, 34080 KB, 36219 KB/s, 0 seconds passed
... 27%, 34112 KB, 36238 KB/s, 0 seconds passed
... 27%, 34144 KB, 36259 KB/s, 0 seconds passed
... 27%, 34176 KB, 36280 KB/s, 0 seconds passed
... 27%, 34208 KB, 36300 KB/s, 0 seconds passed
... 27%, 34240 KB, 36322 KB/s, 0 seconds passed
... 27%, 34272 KB, 36342 KB/s, 0 seconds passed
... 27%, 34304 KB, 36363 KB/s, 0 seconds passed
... 27%, 34336 KB, 36384 KB/s, 0 seconds passed
... 27%, 34368 KB, 36401 KB/s, 0 seconds passed
... 27%, 34400 KB, 36423 KB/s, 0 seconds passed
... 27%, 34432 KB, 36443 KB/s, 0 seconds passed
... 27%, 34464 KB, 36464 KB/s, 0 seconds passed
... 27%, 34496 KB, 36485 KB/s, 0 seconds passed
... 27%, 34528 KB, 36504 KB/s, 0 seconds passed
... 27%, 34560 KB, 36525 KB/s, 0 seconds passed
... 27%, 34592 KB, 36546 KB/s, 0 seconds passed
... 27%, 34624 KB, 36566 KB/s, 0 seconds passed
... 27%, 34656 KB, 36587 KB/s, 0 seconds passed
... 27%, 34688 KB, 36607 KB/s, 0 seconds passed
... 27%, 34720 KB, 36626 KB/s, 0 seconds passed
... 27%, 34752 KB, 36646 KB/s, 0 seconds passed
... 27%, 34784 KB, 36666 KB/s, 0 seconds passed
... 27%, 34816 KB, 36686 KB/s, 0 seconds passed
... 27%, 34848 KB, 36706 KB/s, 0 seconds passed
... 27%, 34880 KB, 36726 KB/s, 0 seconds passed
... 27%, 34912 KB, 36747 KB/s, 0 seconds passed
... 27%, 34944 KB, 36766 KB/s, 0 seconds passed
... 27%, 34976 KB, 36787 KB/s, 0 seconds passed
... 27%, 35008 KB, 36807 KB/s, 0 seconds passed
... 27%, 35040 KB, 36827 KB/s, 0 seconds passed
... 27%, 35072 KB, 36846 KB/s, 0 seconds passed
... 27%, 35104 KB, 36865 KB/s, 0 seconds passed
... 27%, 35136 KB, 36886 KB/s, 0 seconds passed
... 27%, 35168 KB, 36906 KB/s, 0 seconds passed
... 27%, 35200 KB, 36925 KB/s, 0 seconds passed
... 27%, 35232 KB, 36944 KB/s, 0 seconds passed
... 27%, 35264 KB, 36963 KB/s, 0 seconds passed
... 28%, 35296 KB, 36984 KB/s, 0 seconds passed
... 28%, 35328 KB, 37005 KB/s, 0 seconds passed
... 28%, 35360 KB, 37023 KB/s, 0 seconds passed
... 28%, 35392 KB, 37047 KB/s, 0 seconds passed
... 28%, 35424 KB, 37071 KB/s, 0 seconds passed
... 28%, 35456 KB, 37096 KB/s, 0 seconds passed
... 28%, 35488 KB, 37121 KB/s, 0 seconds passed
... 28%, 35520 KB, 37147 KB/s, 0 seconds passed
... 28%, 35552 KB, 37173 KB/s, 0 seconds passed
... 28%, 35584 KB, 37199 KB/s, 0 seconds passed
... 28%, 35616 KB, 37224 KB/s, 0 seconds passed
... 28%, 35648 KB, 37250 KB/s, 0 seconds passed
... 28%, 35680 KB, 37276 KB/s, 0 seconds passed
... 28%, 35712 KB, 37302 KB/s, 0 seconds passed
... 28%, 35744 KB, 37328 KB/s, 0 seconds passed
... 28%, 35776 KB, 37354 KB/s, 0 seconds passed
... 28%, 35808 KB, 37380 KB/s, 0 seconds passed

.. parsed-literal::

    ... 28%, 35840 KB, 34264 KB/s, 1 seconds passed
... 28%, 35872 KB, 34279 KB/s, 1 seconds passed
... 28%, 35904 KB, 34287 KB/s, 1 seconds passed
... 28%, 35936 KB, 34304 KB/s, 1 seconds passed
... 28%, 35968 KB, 33902 KB/s, 1 seconds passed
... 28%, 36000 KB, 33917 KB/s, 1 seconds passed
... 28%, 36032 KB, 33935 KB/s, 1 seconds passed
... 28%, 36064 KB, 33950 KB/s, 1 seconds passed
... 28%, 36096 KB, 33966 KB/s, 1 seconds passed
... 28%, 36128 KB, 33984 KB/s, 1 seconds passed
... 28%, 36160 KB, 34002 KB/s, 1 seconds passed
... 28%, 36192 KB, 34018 KB/s, 1 seconds passed
... 28%, 36224 KB, 34035 KB/s, 1 seconds passed
... 28%, 36256 KB, 34053 KB/s, 1 seconds passed
... 28%, 36288 KB, 34071 KB/s, 1 seconds passed
... 28%, 36320 KB, 34088 KB/s, 1 seconds passed
... 28%, 36352 KB, 34106 KB/s, 1 seconds passed
... 28%, 36384 KB, 34123 KB/s, 1 seconds passed
... 28%, 36416 KB, 34141 KB/s, 1 seconds passed
... 28%, 36448 KB, 34158 KB/s, 1 seconds passed
... 28%, 36480 KB, 34176 KB/s, 1 seconds passed
... 28%, 36512 KB, 34193 KB/s, 1 seconds passed
... 29%, 36544 KB, 34213 KB/s, 1 seconds passed
... 29%, 36576 KB, 34234 KB/s, 1 seconds passed
... 29%, 36608 KB, 34254 KB/s, 1 seconds passed
... 29%, 36640 KB, 34272 KB/s, 1 seconds passed
... 29%, 36672 KB, 34289 KB/s, 1 seconds passed
... 29%, 36704 KB, 34307 KB/s, 1 seconds passed
... 29%, 36736 KB, 34325 KB/s, 1 seconds passed
... 29%, 36768 KB, 34342 KB/s, 1 seconds passed
... 29%, 36800 KB, 34359 KB/s, 1 seconds passed
... 29%, 36832 KB, 34377 KB/s, 1 seconds passed
... 29%, 36864 KB, 34394 KB/s, 1 seconds passed
... 29%, 36896 KB, 34411 KB/s, 1 seconds passed
... 29%, 36928 KB, 34428 KB/s, 1 seconds passed
... 29%, 36960 KB, 34446 KB/s, 1 seconds passed
... 29%, 36992 KB, 34463 KB/s, 1 seconds passed
... 29%, 37024 KB, 34482 KB/s, 1 seconds passed
... 29%, 37056 KB, 34500 KB/s, 1 seconds passed
... 29%, 37088 KB, 34517 KB/s, 1 seconds passed
... 29%, 37120 KB, 34534 KB/s, 1 seconds passed
... 29%, 37152 KB, 34551 KB/s, 1 seconds passed
... 29%, 37184 KB, 34568 KB/s, 1 seconds passed
... 29%, 37216 KB, 34584 KB/s, 1 seconds passed
... 29%, 37248 KB, 34602 KB/s, 1 seconds passed
... 29%, 37280 KB, 34621 KB/s, 1 seconds passed
... 29%, 37312 KB, 34643 KB/s, 1 seconds passed
... 29%, 37344 KB, 34665 KB/s, 1 seconds passed
... 29%, 37376 KB, 34686 KB/s, 1 seconds passed
... 29%, 37408 KB, 34708 KB/s, 1 seconds passed
... 29%, 37440 KB, 34730 KB/s, 1 seconds passed
... 29%, 37472 KB, 34751 KB/s, 1 seconds passed
... 29%, 37504 KB, 34773 KB/s, 1 seconds passed
... 29%, 37536 KB, 34795 KB/s, 1 seconds passed
... 29%, 37568 KB, 34817 KB/s, 1 seconds passed
... 29%, 37600 KB, 34839 KB/s, 1 seconds passed
... 29%, 37632 KB, 34861 KB/s, 1 seconds passed
... 29%, 37664 KB, 34883 KB/s, 1 seconds passed
... 29%, 37696 KB, 34903 KB/s, 1 seconds passed
... 29%, 37728 KB, 34926 KB/s, 1 seconds passed
... 29%, 37760 KB, 34949 KB/s, 1 seconds passed
... 30%, 37792 KB, 34972 KB/s, 1 seconds passed
... 30%, 37824 KB, 34995 KB/s, 1 seconds passed
... 30%, 37856 KB, 35019 KB/s, 1 seconds passed

.. parsed-literal::

    ... 30%, 37888 KB, 33978 KB/s, 1 seconds passed
... 30%, 37920 KB, 33992 KB/s, 1 seconds passed
... 30%, 37952 KB, 33731 KB/s, 1 seconds passed
... 30%, 37984 KB, 33457 KB/s, 1 seconds passed
... 30%, 38016 KB, 33468 KB/s, 1 seconds passed
... 30%, 38048 KB, 33485 KB/s, 1 seconds passed

.. parsed-literal::

    ... 30%, 38080 KB, 33298 KB/s, 1 seconds passed
... 30%, 38112 KB, 33260 KB/s, 1 seconds passed
... 30%, 38144 KB, 33274 KB/s, 1 seconds passed
... 30%, 38176 KB, 33290 KB/s, 1 seconds passed
... 30%, 38208 KB, 33309 KB/s, 1 seconds passed
... 30%, 38240 KB, 33126 KB/s, 1 seconds passed
... 30%, 38272 KB, 33141 KB/s, 1 seconds passed
... 30%, 38304 KB, 33157 KB/s, 1 seconds passed
... 30%, 38336 KB, 33174 KB/s, 1 seconds passed
... 30%, 38368 KB, 33190 KB/s, 1 seconds passed
... 30%, 38400 KB, 33206 KB/s, 1 seconds passed
... 30%, 38432 KB, 33223 KB/s, 1 seconds passed
... 30%, 38464 KB, 33058 KB/s, 1 seconds passed
... 30%, 38496 KB, 33071 KB/s, 1 seconds passed
... 30%, 38528 KB, 33086 KB/s, 1 seconds passed
... 30%, 38560 KB, 33056 KB/s, 1 seconds passed
... 30%, 38592 KB, 33070 KB/s, 1 seconds passed
... 30%, 38624 KB, 33085 KB/s, 1 seconds passed
... 30%, 38656 KB, 33100 KB/s, 1 seconds passed
... 30%, 38688 KB, 33117 KB/s, 1 seconds passed
... 30%, 38720 KB, 33133 KB/s, 1 seconds passed
... 30%, 38752 KB, 33150 KB/s, 1 seconds passed
... 30%, 38784 KB, 33166 KB/s, 1 seconds passed
... 30%, 38816 KB, 33182 KB/s, 1 seconds passed
... 30%, 38848 KB, 33198 KB/s, 1 seconds passed
... 30%, 38880 KB, 33215 KB/s, 1 seconds passed
... 30%, 38912 KB, 33230 KB/s, 1 seconds passed
... 30%, 38944 KB, 33249 KB/s, 1 seconds passed
... 30%, 38976 KB, 33268 KB/s, 1 seconds passed
... 30%, 39008 KB, 33283 KB/s, 1 seconds passed
... 30%, 39040 KB, 33300 KB/s, 1 seconds passed
... 31%, 39072 KB, 33319 KB/s, 1 seconds passed
... 31%, 39104 KB, 33338 KB/s, 1 seconds passed
... 31%, 39136 KB, 33354 KB/s, 1 seconds passed
... 31%, 39168 KB, 33370 KB/s, 1 seconds passed
... 31%, 39200 KB, 33199 KB/s, 1 seconds passed
... 31%, 39232 KB, 33213 KB/s, 1 seconds passed
... 31%, 39264 KB, 33106 KB/s, 1 seconds passed
... 31%, 39296 KB, 33119 KB/s, 1 seconds passed
... 31%, 39328 KB, 33134 KB/s, 1 seconds passed
... 31%, 39360 KB, 33150 KB/s, 1 seconds passed
... 31%, 39392 KB, 33165 KB/s, 1 seconds passed
... 31%, 39424 KB, 33181 KB/s, 1 seconds passed
... 31%, 39456 KB, 33197 KB/s, 1 seconds passed
... 31%, 39488 KB, 33214 KB/s, 1 seconds passed
... 31%, 39520 KB, 33230 KB/s, 1 seconds passed
... 31%, 39552 KB, 33246 KB/s, 1 seconds passed
... 31%, 39584 KB, 33263 KB/s, 1 seconds passed
... 31%, 39616 KB, 33278 KB/s, 1 seconds passed
... 31%, 39648 KB, 33295 KB/s, 1 seconds passed
... 31%, 39680 KB, 33311 KB/s, 1 seconds passed
... 31%, 39712 KB, 33328 KB/s, 1 seconds passed
... 31%, 39744 KB, 33344 KB/s, 1 seconds passed
... 31%, 39776 KB, 33359 KB/s, 1 seconds passed
... 31%, 39808 KB, 33375 KB/s, 1 seconds passed
... 31%, 39840 KB, 33392 KB/s, 1 seconds passed
... 31%, 39872 KB, 33409 KB/s, 1 seconds passed
... 31%, 39904 KB, 33424 KB/s, 1 seconds passed
... 31%, 39936 KB, 33440 KB/s, 1 seconds passed

.. parsed-literal::

    ... 31%, 39968 KB, 33456 KB/s, 1 seconds passed
... 31%, 40000 KB, 33472 KB/s, 1 seconds passed
... 31%, 40032 KB, 33489 KB/s, 1 seconds passed
... 31%, 40064 KB, 33505 KB/s, 1 seconds passed
... 31%, 40096 KB, 33520 KB/s, 1 seconds passed
... 31%, 40128 KB, 33535 KB/s, 1 seconds passed
... 31%, 40160 KB, 33551 KB/s, 1 seconds passed
... 31%, 40192 KB, 33567 KB/s, 1 seconds passed
... 31%, 40224 KB, 33584 KB/s, 1 seconds passed
... 31%, 40256 KB, 33604 KB/s, 1 seconds passed
... 31%, 40288 KB, 33623 KB/s, 1 seconds passed
... 32%, 40320 KB, 33643 KB/s, 1 seconds passed
... 32%, 40352 KB, 33663 KB/s, 1 seconds passed
... 32%, 40384 KB, 33683 KB/s, 1 seconds passed
... 32%, 40416 KB, 33703 KB/s, 1 seconds passed
... 32%, 40448 KB, 33723 KB/s, 1 seconds passed
... 32%, 40480 KB, 33743 KB/s, 1 seconds passed
... 32%, 40512 KB, 33761 KB/s, 1 seconds passed
... 32%, 40544 KB, 33781 KB/s, 1 seconds passed
... 32%, 40576 KB, 33800 KB/s, 1 seconds passed
... 32%, 40608 KB, 33820 KB/s, 1 seconds passed
... 32%, 40640 KB, 33841 KB/s, 1 seconds passed
... 32%, 40672 KB, 33862 KB/s, 1 seconds passed
... 32%, 40704 KB, 33883 KB/s, 1 seconds passed
... 32%, 40736 KB, 33905 KB/s, 1 seconds passed
... 32%, 40768 KB, 33926 KB/s, 1 seconds passed
... 32%, 40800 KB, 33947 KB/s, 1 seconds passed
... 32%, 40832 KB, 33968 KB/s, 1 seconds passed
... 32%, 40864 KB, 33989 KB/s, 1 seconds passed
... 32%, 40896 KB, 34010 KB/s, 1 seconds passed
... 32%, 40928 KB, 34031 KB/s, 1 seconds passed
... 32%, 40960 KB, 33946 KB/s, 1 seconds passed
... 32%, 40992 KB, 33814 KB/s, 1 seconds passed
... 32%, 41024 KB, 33828 KB/s, 1 seconds passed
... 32%, 41056 KB, 33846 KB/s, 1 seconds passed
... 32%, 41088 KB, 33865 KB/s, 1 seconds passed
... 32%, 41120 KB, 33884 KB/s, 1 seconds passed
... 32%, 41152 KB, 33904 KB/s, 1 seconds passed
... 32%, 41184 KB, 33922 KB/s, 1 seconds passed
... 32%, 41216 KB, 33942 KB/s, 1 seconds passed
... 32%, 41248 KB, 33961 KB/s, 1 seconds passed
... 32%, 41280 KB, 33981 KB/s, 1 seconds passed
... 32%, 41312 KB, 34000 KB/s, 1 seconds passed
... 32%, 41344 KB, 34019 KB/s, 1 seconds passed
... 32%, 41376 KB, 33793 KB/s, 1 seconds passed
... 32%, 41408 KB, 33806 KB/s, 1 seconds passed
... 32%, 41440 KB, 33821 KB/s, 1 seconds passed
... 32%, 41472 KB, 33838 KB/s, 1 seconds passed
... 32%, 41504 KB, 33853 KB/s, 1 seconds passed
... 32%, 41536 KB, 33869 KB/s, 1 seconds passed
... 33%, 41568 KB, 33885 KB/s, 1 seconds passed
... 33%, 41600 KB, 33900 KB/s, 1 seconds passed
... 33%, 41632 KB, 33916 KB/s, 1 seconds passed
... 33%, 41664 KB, 33931 KB/s, 1 seconds passed
... 33%, 41696 KB, 33947 KB/s, 1 seconds passed
... 33%, 41728 KB, 33962 KB/s, 1 seconds passed
... 33%, 41760 KB, 33977 KB/s, 1 seconds passed
... 33%, 41792 KB, 33993 KB/s, 1 seconds passed
... 33%, 41824 KB, 34009 KB/s, 1 seconds passed
... 33%, 41856 KB, 34024 KB/s, 1 seconds passed
... 33%, 41888 KB, 34040 KB/s, 1 seconds passed
... 33%, 41920 KB, 34055 KB/s, 1 seconds passed
... 33%, 41952 KB, 34070 KB/s, 1 seconds passed
... 33%, 41984 KB, 34086 KB/s, 1 seconds passed
... 33%, 42016 KB, 34100 KB/s, 1 seconds passed
... 33%, 42048 KB, 34115 KB/s, 1 seconds passed
... 33%, 42080 KB, 34130 KB/s, 1 seconds passed
... 33%, 42112 KB, 34146 KB/s, 1 seconds passed
... 33%, 42144 KB, 34161 KB/s, 1 seconds passed
... 33%, 42176 KB, 34176 KB/s, 1 seconds passed
... 33%, 42208 KB, 34192 KB/s, 1 seconds passed
... 33%, 42240 KB, 34207 KB/s, 1 seconds passed
... 33%, 42272 KB, 34222 KB/s, 1 seconds passed
... 33%, 42304 KB, 34238 KB/s, 1 seconds passed
... 33%, 42336 KB, 34253 KB/s, 1 seconds passed
... 33%, 42368 KB, 34269 KB/s, 1 seconds passed
... 33%, 42400 KB, 34284 KB/s, 1 seconds passed
... 33%, 42432 KB, 34299 KB/s, 1 seconds passed
... 33%, 42464 KB, 34314 KB/s, 1 seconds passed
... 33%, 42496 KB, 34330 KB/s, 1 seconds passed
... 33%, 42528 KB, 34345 KB/s, 1 seconds passed
... 33%, 42560 KB, 34360 KB/s, 1 seconds passed
... 33%, 42592 KB, 34376 KB/s, 1 seconds passed
... 33%, 42624 KB, 34396 KB/s, 1 seconds passed
... 33%, 42656 KB, 34415 KB/s, 1 seconds passed
... 33%, 42688 KB, 34434 KB/s, 1 seconds passed
... 33%, 42720 KB, 34452 KB/s, 1 seconds passed
... 33%, 42752 KB, 34470 KB/s, 1 seconds passed
... 33%, 42784 KB, 34489 KB/s, 1 seconds passed
... 33%, 42816 KB, 34509 KB/s, 1 seconds passed
... 34%, 42848 KB, 34528 KB/s, 1 seconds passed
... 34%, 42880 KB, 34547 KB/s, 1 seconds passed
... 34%, 42912 KB, 34566 KB/s, 1 seconds passed
... 34%, 42944 KB, 34585 KB/s, 1 seconds passed
... 34%, 42976 KB, 34604 KB/s, 1 seconds passed
... 34%, 43008 KB, 34623 KB/s, 1 seconds passed
... 34%, 43040 KB, 34641 KB/s, 1 seconds passed
... 34%, 43072 KB, 34661 KB/s, 1 seconds passed
... 34%, 43104 KB, 34680 KB/s, 1 seconds passed
... 34%, 43136 KB, 34699 KB/s, 1 seconds passed
... 34%, 43168 KB, 34718 KB/s, 1 seconds passed
... 34%, 43200 KB, 34736 KB/s, 1 seconds passed
... 34%, 43232 KB, 34755 KB/s, 1 seconds passed
... 34%, 43264 KB, 34774 KB/s, 1 seconds passed
... 34%, 43296 KB, 34792 KB/s, 1 seconds passed
... 34%, 43328 KB, 34812 KB/s, 1 seconds passed
... 34%, 43360 KB, 34831 KB/s, 1 seconds passed
... 34%, 43392 KB, 34850 KB/s, 1 seconds passed
... 34%, 43424 KB, 34868 KB/s, 1 seconds passed
... 34%, 43456 KB, 34887 KB/s, 1 seconds passed

.. parsed-literal::

    ... 34%, 43488 KB, 34906 KB/s, 1 seconds passed
... 34%, 43520 KB, 34925 KB/s, 1 seconds passed
... 34%, 43552 KB, 34944 KB/s, 1 seconds passed
... 34%, 43584 KB, 34963 KB/s, 1 seconds passed
... 34%, 43616 KB, 34982 KB/s, 1 seconds passed
... 34%, 43648 KB, 35001 KB/s, 1 seconds passed
... 34%, 43680 KB, 35019 KB/s, 1 seconds passed
... 34%, 43712 KB, 35037 KB/s, 1 seconds passed
... 34%, 43744 KB, 35055 KB/s, 1 seconds passed
... 34%, 43776 KB, 35073 KB/s, 1 seconds passed
... 34%, 43808 KB, 35093 KB/s, 1 seconds passed
... 34%, 43840 KB, 35113 KB/s, 1 seconds passed
... 34%, 43872 KB, 35132 KB/s, 1 seconds passed
... 34%, 43904 KB, 35154 KB/s, 1 seconds passed
... 34%, 43936 KB, 35176 KB/s, 1 seconds passed
... 34%, 43968 KB, 35198 KB/s, 1 seconds passed
... 34%, 44000 KB, 35220 KB/s, 1 seconds passed
... 34%, 44032 KB, 35242 KB/s, 1 seconds passed
... 34%, 44064 KB, 35120 KB/s, 1 seconds passed
... 35%, 44096 KB, 35136 KB/s, 1 seconds passed
... 35%, 44128 KB, 35153 KB/s, 1 seconds passed
... 35%, 44160 KB, 35171 KB/s, 1 seconds passed
... 35%, 44192 KB, 35189 KB/s, 1 seconds passed
... 35%, 44224 KB, 35210 KB/s, 1 seconds passed
... 35%, 44256 KB, 35227 KB/s, 1 seconds passed
... 35%, 44288 KB, 35245 KB/s, 1 seconds passed
... 35%, 44320 KB, 35260 KB/s, 1 seconds passed
... 35%, 44352 KB, 35280 KB/s, 1 seconds passed
... 35%, 44384 KB, 35147 KB/s, 1 seconds passed
... 35%, 44416 KB, 35161 KB/s, 1 seconds passed
... 35%, 44448 KB, 35175 KB/s, 1 seconds passed
... 35%, 44480 KB, 35184 KB/s, 1 seconds passed
... 35%, 44512 KB, 35199 KB/s, 1 seconds passed
... 35%, 44544 KB, 35214 KB/s, 1 seconds passed
... 35%, 44576 KB, 35229 KB/s, 1 seconds passed
... 35%, 44608 KB, 35244 KB/s, 1 seconds passed
... 35%, 44640 KB, 35259 KB/s, 1 seconds passed
... 35%, 44672 KB, 35274 KB/s, 1 seconds passed
... 35%, 44704 KB, 35289 KB/s, 1 seconds passed
... 35%, 44736 KB, 35304 KB/s, 1 seconds passed
... 35%, 44768 KB, 35319 KB/s, 1 seconds passed
... 35%, 44800 KB, 35333 KB/s, 1 seconds passed
... 35%, 44832 KB, 35348 KB/s, 1 seconds passed
... 35%, 44864 KB, 35363 KB/s, 1 seconds passed
... 35%, 44896 KB, 35378 KB/s, 1 seconds passed
... 35%, 44928 KB, 35393 KB/s, 1 seconds passed
... 35%, 44960 KB, 35409 KB/s, 1 seconds passed
... 35%, 44992 KB, 35426 KB/s, 1 seconds passed
... 35%, 45024 KB, 35443 KB/s, 1 seconds passed
... 35%, 45056 KB, 35460 KB/s, 1 seconds passed
... 35%, 45088 KB, 35477 KB/s, 1 seconds passed

.. parsed-literal::

    ... 35%, 45120 KB, 33667 KB/s, 1 seconds passed
... 35%, 45152 KB, 33678 KB/s, 1 seconds passed
... 35%, 45184 KB, 33691 KB/s, 1 seconds passed
... 35%, 45216 KB, 33705 KB/s, 1 seconds passed
... 35%, 45248 KB, 33718 KB/s, 1 seconds passed
... 35%, 45280 KB, 33733 KB/s, 1 seconds passed
... 35%, 45312 KB, 33747 KB/s, 1 seconds passed
... 36%, 45344 KB, 33761 KB/s, 1 seconds passed
... 36%, 45376 KB, 33775 KB/s, 1 seconds passed
... 36%, 45408 KB, 33788 KB/s, 1 seconds passed
... 36%, 45440 KB, 33802 KB/s, 1 seconds passed
... 36%, 45472 KB, 33816 KB/s, 1 seconds passed
... 36%, 45504 KB, 33830 KB/s, 1 seconds passed
... 36%, 45536 KB, 33845 KB/s, 1 seconds passed
... 36%, 45568 KB, 33860 KB/s, 1 seconds passed
... 36%, 45600 KB, 33875 KB/s, 1 seconds passed
... 36%, 45632 KB, 33890 KB/s, 1 seconds passed
... 36%, 45664 KB, 33907 KB/s, 1 seconds passed
... 36%, 45696 KB, 33915 KB/s, 1 seconds passed
... 36%, 45728 KB, 33929 KB/s, 1 seconds passed
... 36%, 45760 KB, 33943 KB/s, 1 seconds passed

.. parsed-literal::

    ... 36%, 45792 KB, 33957 KB/s, 1 seconds passed
... 36%, 45824 KB, 33972 KB/s, 1 seconds passed
... 36%, 45856 KB, 33986 KB/s, 1 seconds passed
... 36%, 45888 KB, 34000 KB/s, 1 seconds passed
... 36%, 45920 KB, 34015 KB/s, 1 seconds passed
... 36%, 45952 KB, 34030 KB/s, 1 seconds passed
... 36%, 45984 KB, 34046 KB/s, 1 seconds passed
... 36%, 46016 KB, 34064 KB/s, 1 seconds passed
... 36%, 46048 KB, 34081 KB/s, 1 seconds passed

.. parsed-literal::

    ... 36%, 46080 KB, 32695 KB/s, 1 seconds passed
... 36%, 46112 KB, 32707 KB/s, 1 seconds passed
... 36%, 46144 KB, 32682 KB/s, 1 seconds passed
... 36%, 46176 KB, 32692 KB/s, 1 seconds passed
... 36%, 46208 KB, 32705 KB/s, 1 seconds passed
... 36%, 46240 KB, 32718 KB/s, 1 seconds passed
... 36%, 46272 KB, 32731 KB/s, 1 seconds passed
... 36%, 46304 KB, 32745 KB/s, 1 seconds passed
... 36%, 46336 KB, 32759 KB/s, 1 seconds passed
... 36%, 46368 KB, 32773 KB/s, 1 seconds passed
... 36%, 46400 KB, 32787 KB/s, 1 seconds passed
... 36%, 46432 KB, 32801 KB/s, 1 seconds passed
... 36%, 46464 KB, 32814 KB/s, 1 seconds passed
... 36%, 46496 KB, 32828 KB/s, 1 seconds passed
... 36%, 46528 KB, 32842 KB/s, 1 seconds passed
... 36%, 46560 KB, 32856 KB/s, 1 seconds passed
... 36%, 46592 KB, 32869 KB/s, 1 seconds passed
... 37%, 46624 KB, 32883 KB/s, 1 seconds passed
... 37%, 46656 KB, 32897 KB/s, 1 seconds passed
... 37%, 46688 KB, 32910 KB/s, 1 seconds passed
... 37%, 46720 KB, 32924 KB/s, 1 seconds passed
... 37%, 46752 KB, 32939 KB/s, 1 seconds passed
... 37%, 46784 KB, 32955 KB/s, 1 seconds passed
... 37%, 46816 KB, 32971 KB/s, 1 seconds passed
... 37%, 46848 KB, 32988 KB/s, 1 seconds passed
... 37%, 46880 KB, 33005 KB/s, 1 seconds passed
... 37%, 46912 KB, 33022 KB/s, 1 seconds passed
... 37%, 46944 KB, 33039 KB/s, 1 seconds passed
... 37%, 46976 KB, 32981 KB/s, 1 seconds passed
... 37%, 47008 KB, 32992 KB/s, 1 seconds passed
... 37%, 47040 KB, 33007 KB/s, 1 seconds passed
... 37%, 47072 KB, 33022 KB/s, 1 seconds passed
... 37%, 47104 KB, 33037 KB/s, 1 seconds passed
... 37%, 47136 KB, 33052 KB/s, 1 seconds passed
... 37%, 47168 KB, 33068 KB/s, 1 seconds passed
... 37%, 47200 KB, 33083 KB/s, 1 seconds passed
... 37%, 47232 KB, 33098 KB/s, 1 seconds passed
... 37%, 47264 KB, 33114 KB/s, 1 seconds passed
... 37%, 47296 KB, 33128 KB/s, 1 seconds passed
... 37%, 47328 KB, 33143 KB/s, 1 seconds passed
... 37%, 47360 KB, 33158 KB/s, 1 seconds passed
... 37%, 47392 KB, 33173 KB/s, 1 seconds passed
... 37%, 47424 KB, 33188 KB/s, 1 seconds passed
... 37%, 47456 KB, 33203 KB/s, 1 seconds passed
... 37%, 47488 KB, 33218 KB/s, 1 seconds passed
... 37%, 47520 KB, 33234 KB/s, 1 seconds passed
... 37%, 47552 KB, 33249 KB/s, 1 seconds passed
... 37%, 47584 KB, 33265 KB/s, 1 seconds passed
... 37%, 47616 KB, 33282 KB/s, 1 seconds passed
... 37%, 47648 KB, 33297 KB/s, 1 seconds passed
... 37%, 47680 KB, 33313 KB/s, 1 seconds passed
... 37%, 47712 KB, 33329 KB/s, 1 seconds passed
... 37%, 47744 KB, 33345 KB/s, 1 seconds passed
... 37%, 47776 KB, 33360 KB/s, 1 seconds passed
... 37%, 47808 KB, 33376 KB/s, 1 seconds passed
... 37%, 47840 KB, 33392 KB/s, 1 seconds passed
... 38%, 47872 KB, 33408 KB/s, 1 seconds passed
... 38%, 47904 KB, 33424 KB/s, 1 seconds passed
... 38%, 47936 KB, 33440 KB/s, 1 seconds passed
... 38%, 47968 KB, 33455 KB/s, 1 seconds passed
... 38%, 48000 KB, 33471 KB/s, 1 seconds passed
... 38%, 48032 KB, 33487 KB/s, 1 seconds passed
... 38%, 48064 KB, 33502 KB/s, 1 seconds passed
... 38%, 48096 KB, 33518 KB/s, 1 seconds passed
... 38%, 48128 KB, 33534 KB/s, 1 seconds passed
... 38%, 48160 KB, 33550 KB/s, 1 seconds passed
... 38%, 48192 KB, 33566 KB/s, 1 seconds passed
... 38%, 48224 KB, 33581 KB/s, 1 seconds passed
... 38%, 48256 KB, 33598 KB/s, 1 seconds passed
... 38%, 48288 KB, 33615 KB/s, 1 seconds passed
... 38%, 48320 KB, 33348 KB/s, 1 seconds passed
... 38%, 48352 KB, 33359 KB/s, 1 seconds passed
... 38%, 48384 KB, 33374 KB/s, 1 seconds passed
... 38%, 48416 KB, 33373 KB/s, 1 seconds passed

.. parsed-literal::

    ... 38%, 48448 KB, 33385 KB/s, 1 seconds passed
... 38%, 48480 KB, 33398 KB/s, 1 seconds passed
... 38%, 48512 KB, 33412 KB/s, 1 seconds passed
... 38%, 48544 KB, 33426 KB/s, 1 seconds passed
... 38%, 48576 KB, 33441 KB/s, 1 seconds passed
... 38%, 48608 KB, 33456 KB/s, 1 seconds passed
... 38%, 48640 KB, 33469 KB/s, 1 seconds passed
... 38%, 48672 KB, 33483 KB/s, 1 seconds passed
... 38%, 48704 KB, 33497 KB/s, 1 seconds passed
... 38%, 48736 KB, 33510 KB/s, 1 seconds passed
... 38%, 48768 KB, 33525 KB/s, 1 seconds passed
... 38%, 48800 KB, 33540 KB/s, 1 seconds passed
... 38%, 48832 KB, 33554 KB/s, 1 seconds passed
... 38%, 48864 KB, 33564 KB/s, 1 seconds passed
... 38%, 48896 KB, 33578 KB/s, 1 seconds passed
... 38%, 48928 KB, 33592 KB/s, 1 seconds passed
... 38%, 48960 KB, 33606 KB/s, 1 seconds passed
... 38%, 48992 KB, 33621 KB/s, 1 seconds passed
... 38%, 49024 KB, 33633 KB/s, 1 seconds passed
... 38%, 49056 KB, 33648 KB/s, 1 seconds passed
... 38%, 49088 KB, 33662 KB/s, 1 seconds passed
... 38%, 49120 KB, 33677 KB/s, 1 seconds passed
... 39%, 49152 KB, 33691 KB/s, 1 seconds passed
... 39%, 49184 KB, 33706 KB/s, 1 seconds passed
... 39%, 49216 KB, 33720 KB/s, 1 seconds passed
... 39%, 49248 KB, 33734 KB/s, 1 seconds passed
... 39%, 49280 KB, 33748 KB/s, 1 seconds passed
... 39%, 49312 KB, 33765 KB/s, 1 seconds passed
... 39%, 49344 KB, 33781 KB/s, 1 seconds passed
... 39%, 49376 KB, 33797 KB/s, 1 seconds passed
... 39%, 49408 KB, 33813 KB/s, 1 seconds passed
... 39%, 49440 KB, 33830 KB/s, 1 seconds passed
... 39%, 49472 KB, 33846 KB/s, 1 seconds passed
... 39%, 49504 KB, 33862 KB/s, 1 seconds passed
... 39%, 49536 KB, 33878 KB/s, 1 seconds passed
... 39%, 49568 KB, 33894 KB/s, 1 seconds passed
... 39%, 49600 KB, 33911 KB/s, 1 seconds passed
... 39%, 49632 KB, 33927 KB/s, 1 seconds passed
... 39%, 49664 KB, 33943 KB/s, 1 seconds passed
... 39%, 49696 KB, 33959 KB/s, 1 seconds passed
... 39%, 49728 KB, 33975 KB/s, 1 seconds passed
... 39%, 49760 KB, 33991 KB/s, 1 seconds passed
... 39%, 49792 KB, 34007 KB/s, 1 seconds passed
... 39%, 49824 KB, 34022 KB/s, 1 seconds passed
... 39%, 49856 KB, 34038 KB/s, 1 seconds passed
... 39%, 49888 KB, 34053 KB/s, 1 seconds passed
... 39%, 49920 KB, 34069 KB/s, 1 seconds passed
... 39%, 49952 KB, 34084 KB/s, 1 seconds passed
... 39%, 49984 KB, 34099 KB/s, 1 seconds passed
... 39%, 50016 KB, 34115 KB/s, 1 seconds passed
... 39%, 50048 KB, 34130 KB/s, 1 seconds passed
... 39%, 50080 KB, 34146 KB/s, 1 seconds passed
... 39%, 50112 KB, 34160 KB/s, 1 seconds passed
... 39%, 50144 KB, 34176 KB/s, 1 seconds passed
... 39%, 50176 KB, 34192 KB/s, 1 seconds passed
... 39%, 50208 KB, 34206 KB/s, 1 seconds passed
... 39%, 50240 KB, 34221 KB/s, 1 seconds passed
... 39%, 50272 KB, 34237 KB/s, 1 seconds passed
... 39%, 50304 KB, 34253 KB/s, 1 seconds passed
... 39%, 50336 KB, 34267 KB/s, 1 seconds passed
... 39%, 50368 KB, 34283 KB/s, 1 seconds passed
... 40%, 50400 KB, 34298 KB/s, 1 seconds passed
... 40%, 50432 KB, 34314 KB/s, 1 seconds passed
... 40%, 50464 KB, 34330 KB/s, 1 seconds passed
... 40%, 50496 KB, 34345 KB/s, 1 seconds passed
... 40%, 50528 KB, 34142 KB/s, 1 seconds passed
... 40%, 50560 KB, 34154 KB/s, 1 seconds passed
... 40%, 50592 KB, 34166 KB/s, 1 seconds passed
... 40%, 50624 KB, 34179 KB/s, 1 seconds passed
... 40%, 50656 KB, 34192 KB/s, 1 seconds passed
... 40%, 50688 KB, 34205 KB/s, 1 seconds passed
... 40%, 50720 KB, 34218 KB/s, 1 seconds passed
... 40%, 50752 KB, 34231 KB/s, 1 seconds passed
... 40%, 50784 KB, 34244 KB/s, 1 seconds passed
... 40%, 50816 KB, 34257 KB/s, 1 seconds passed
... 40%, 50848 KB, 34270 KB/s, 1 seconds passed
... 40%, 50880 KB, 34282 KB/s, 1 seconds passed
... 40%, 50912 KB, 34295 KB/s, 1 seconds passed
... 40%, 50944 KB, 34308 KB/s, 1 seconds passed
... 40%, 50976 KB, 34320 KB/s, 1 seconds passed
... 40%, 51008 KB, 34333 KB/s, 1 seconds passed
... 40%, 51040 KB, 34346 KB/s, 1 seconds passed
... 40%, 51072 KB, 34359 KB/s, 1 seconds passed
... 40%, 51104 KB, 34372 KB/s, 1 seconds passed
... 40%, 51136 KB, 34386 KB/s, 1 seconds passed
... 40%, 51168 KB, 34400 KB/s, 1 seconds passed
... 40%, 51200 KB, 34411 KB/s, 1 seconds passed
... 40%, 51232 KB, 34424 KB/s, 1 seconds passed
... 40%, 51264 KB, 34439 KB/s, 1 seconds passed
... 40%, 51296 KB, 34452 KB/s, 1 seconds passed
... 40%, 51328 KB, 34467 KB/s, 1 seconds passed
... 40%, 51360 KB, 34482 KB/s, 1 seconds passed
... 40%, 51392 KB, 34497 KB/s, 1 seconds passed
... 40%, 51424 KB, 34504 KB/s, 1 seconds passed
... 40%, 51456 KB, 34516 KB/s, 1 seconds passed
... 40%, 51488 KB, 34530 KB/s, 1 seconds passed
... 40%, 51520 KB, 34546 KB/s, 1 seconds passed
... 40%, 51552 KB, 34562 KB/s, 1 seconds passed
... 40%, 51584 KB, 34577 KB/s, 1 seconds passed
... 40%, 51616 KB, 34593 KB/s, 1 seconds passed
... 41%, 51648 KB, 34608 KB/s, 1 seconds passed
... 41%, 51680 KB, 34624 KB/s, 1 seconds passed
... 41%, 51712 KB, 34639 KB/s, 1 seconds passed
... 41%, 51744 KB, 34653 KB/s, 1 seconds passed
... 41%, 51776 KB, 34668 KB/s, 1 seconds passed
... 41%, 51808 KB, 34683 KB/s, 1 seconds passed
... 41%, 51840 KB, 34697 KB/s, 1 seconds passed
... 41%, 51872 KB, 34712 KB/s, 1 seconds passed
... 41%, 51904 KB, 34728 KB/s, 1 seconds passed
... 41%, 51936 KB, 34743 KB/s, 1 seconds passed
... 41%, 51968 KB, 34758 KB/s, 1 seconds passed
... 41%, 52000 KB, 34772 KB/s, 1 seconds passed
... 41%, 52032 KB, 34639 KB/s, 1 seconds passed

.. parsed-literal::

    ... 41%, 52064 KB, 34649 KB/s, 1 seconds passed
... 41%, 52096 KB, 34660 KB/s, 1 seconds passed
... 41%, 52128 KB, 34673 KB/s, 1 seconds passed
... 41%, 52160 KB, 34684 KB/s, 1 seconds passed
... 41%, 52192 KB, 34696 KB/s, 1 seconds passed
... 41%, 52224 KB, 34708 KB/s, 1 seconds passed
... 41%, 52256 KB, 34719 KB/s, 1 seconds passed
... 41%, 52288 KB, 34732 KB/s, 1 seconds passed
... 41%, 52320 KB, 34744 KB/s, 1 seconds passed
... 41%, 52352 KB, 34757 KB/s, 1 seconds passed
... 41%, 52384 KB, 34769 KB/s, 1 seconds passed
... 41%, 52416 KB, 34782 KB/s, 1 seconds passed
... 41%, 52448 KB, 34795 KB/s, 1 seconds passed
... 41%, 52480 KB, 34807 KB/s, 1 seconds passed
... 41%, 52512 KB, 34819 KB/s, 1 seconds passed
... 41%, 52544 KB, 34831 KB/s, 1 seconds passed
... 41%, 52576 KB, 34844 KB/s, 1 seconds passed
... 41%, 52608 KB, 34856 KB/s, 1 seconds passed
... 41%, 52640 KB, 34869 KB/s, 1 seconds passed
... 41%, 52672 KB, 34881 KB/s, 1 seconds passed
... 41%, 52704 KB, 34893 KB/s, 1 seconds passed
... 41%, 52736 KB, 34906 KB/s, 1 seconds passed
... 41%, 52768 KB, 34919 KB/s, 1 seconds passed
... 41%, 52800 KB, 34933 KB/s, 1 seconds passed
... 41%, 52832 KB, 34947 KB/s, 1 seconds passed
... 41%, 52864 KB, 34961 KB/s, 1 seconds passed
... 41%, 52896 KB, 34974 KB/s, 1 seconds passed
... 42%, 52928 KB, 34988 KB/s, 1 seconds passed
... 42%, 52960 KB, 35002 KB/s, 1 seconds passed
... 42%, 52992 KB, 35016 KB/s, 1 seconds passed
... 42%, 53024 KB, 35030 KB/s, 1 seconds passed
... 42%, 53056 KB, 35044 KB/s, 1 seconds passed
... 42%, 53088 KB, 35058 KB/s, 1 seconds passed
... 42%, 53120 KB, 35072 KB/s, 1 seconds passed
... 42%, 53152 KB, 35086 KB/s, 1 seconds passed
... 42%, 53184 KB, 35100 KB/s, 1 seconds passed
... 42%, 53216 KB, 35114 KB/s, 1 seconds passed
... 42%, 53248 KB, 35128 KB/s, 1 seconds passed
... 42%, 53280 KB, 35141 KB/s, 1 seconds passed
... 42%, 53312 KB, 35155 KB/s, 1 seconds passed
... 42%, 53344 KB, 35169 KB/s, 1 seconds passed
... 42%, 53376 KB, 35183 KB/s, 1 seconds passed
... 42%, 53408 KB, 35198 KB/s, 1 seconds passed
... 42%, 53440 KB, 35212 KB/s, 1 seconds passed
... 42%, 53472 KB, 35226 KB/s, 1 seconds passed
... 42%, 53504 KB, 35240 KB/s, 1 seconds passed
... 42%, 53536 KB, 35254 KB/s, 1 seconds passed
... 42%, 53568 KB, 35268 KB/s, 1 seconds passed
... 42%, 53600 KB, 35282 KB/s, 1 seconds passed
... 42%, 53632 KB, 35296 KB/s, 1 seconds passed
... 42%, 53664 KB, 35311 KB/s, 1 seconds passed
... 42%, 53696 KB, 35326 KB/s, 1 seconds passed
... 42%, 53728 KB, 35341 KB/s, 1 seconds passed
... 42%, 53760 KB, 35356 KB/s, 1 seconds passed
... 42%, 53792 KB, 35372 KB/s, 1 seconds passed
... 42%, 53824 KB, 35388 KB/s, 1 seconds passed
... 42%, 53856 KB, 35404 KB/s, 1 seconds passed
... 42%, 53888 KB, 35422 KB/s, 1 seconds passed
... 42%, 53920 KB, 35439 KB/s, 1 seconds passed
... 42%, 53952 KB, 35456 KB/s, 1 seconds passed
... 42%, 53984 KB, 35474 KB/s, 1 seconds passed
... 42%, 54016 KB, 35491 KB/s, 1 seconds passed
... 42%, 54048 KB, 35508 KB/s, 1 seconds passed
... 42%, 54080 KB, 35253 KB/s, 1 seconds passed
... 42%, 54112 KB, 35155 KB/s, 1 seconds passed
... 42%, 54144 KB, 35165 KB/s, 1 seconds passed
... 43%, 54176 KB, 35175 KB/s, 1 seconds passed
... 43%, 54208 KB, 35187 KB/s, 1 seconds passed
... 43%, 54240 KB, 35199 KB/s, 1 seconds passed
... 43%, 54272 KB, 35210 KB/s, 1 seconds passed
... 43%, 54304 KB, 35222 KB/s, 1 seconds passed
... 43%, 54336 KB, 35234 KB/s, 1 seconds passed
... 43%, 54368 KB, 35247 KB/s, 1 seconds passed
... 43%, 54400 KB, 35259 KB/s, 1 seconds passed
... 43%, 54432 KB, 35271 KB/s, 1 seconds passed
... 43%, 54464 KB, 35284 KB/s, 1 seconds passed
... 43%, 54496 KB, 35295 KB/s, 1 seconds passed
... 43%, 54528 KB, 35306 KB/s, 1 seconds passed
... 43%, 54560 KB, 35318 KB/s, 1 seconds passed
... 43%, 54592 KB, 35331 KB/s, 1 seconds passed
... 43%, 54624 KB, 35344 KB/s, 1 seconds passed
... 43%, 54656 KB, 35358 KB/s, 1 seconds passed
... 43%, 54688 KB, 35370 KB/s, 1 seconds passed
... 43%, 54720 KB, 35382 KB/s, 1 seconds passed
... 43%, 54752 KB, 35394 KB/s, 1 seconds passed
... 43%, 54784 KB, 35406 KB/s, 1 seconds passed
... 43%, 54816 KB, 35418 KB/s, 1 seconds passed
... 43%, 54848 KB, 35429 KB/s, 1 seconds passed
... 43%, 54880 KB, 35441 KB/s, 1 seconds passed
... 43%, 54912 KB, 35454 KB/s, 1 seconds passed
... 43%, 54944 KB, 35465 KB/s, 1 seconds passed
... 43%, 54976 KB, 35478 KB/s, 1 seconds passed
... 43%, 55008 KB, 35490 KB/s, 1 seconds passed
... 43%, 55040 KB, 35501 KB/s, 1 seconds passed
... 43%, 55072 KB, 35513 KB/s, 1 seconds passed
... 43%, 55104 KB, 35526 KB/s, 1 seconds passed
... 43%, 55136 KB, 35541 KB/s, 1 seconds passed
... 43%, 55168 KB, 35556 KB/s, 1 seconds passed
... 43%, 55200 KB, 35571 KB/s, 1 seconds passed
... 43%, 55232 KB, 35586 KB/s, 1 seconds passed
... 43%, 55264 KB, 35601 KB/s, 1 seconds passed
... 43%, 55296 KB, 35615 KB/s, 1 seconds passed
... 43%, 55328 KB, 35630 KB/s, 1 seconds passed
... 43%, 55360 KB, 35645 KB/s, 1 seconds passed
... 43%, 55392 KB, 35660 KB/s, 1 seconds passed
... 44%, 55424 KB, 35675 KB/s, 1 seconds passed
... 44%, 55456 KB, 35690 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 55488 KB, 35705 KB/s, 1 seconds passed
... 44%, 55520 KB, 35720 KB/s, 1 seconds passed
... 44%, 55552 KB, 35735 KB/s, 1 seconds passed
... 44%, 55584 KB, 35750 KB/s, 1 seconds passed
... 44%, 55616 KB, 35765 KB/s, 1 seconds passed
... 44%, 55648 KB, 35780 KB/s, 1 seconds passed
... 44%, 55680 KB, 35794 KB/s, 1 seconds passed
... 44%, 55712 KB, 35809 KB/s, 1 seconds passed
... 44%, 55744 KB, 35823 KB/s, 1 seconds passed
... 44%, 55776 KB, 35838 KB/s, 1 seconds passed
... 44%, 55808 KB, 35853 KB/s, 1 seconds passed
... 44%, 55840 KB, 35868 KB/s, 1 seconds passed
... 44%, 55872 KB, 35883 KB/s, 1 seconds passed
... 44%, 55904 KB, 35898 KB/s, 1 seconds passed
... 44%, 55936 KB, 35913 KB/s, 1 seconds passed
... 44%, 55968 KB, 35928 KB/s, 1 seconds passed
... 44%, 56000 KB, 35943 KB/s, 1 seconds passed
... 44%, 56032 KB, 35958 KB/s, 1 seconds passed
... 44%, 56064 KB, 35973 KB/s, 1 seconds passed
... 44%, 56096 KB, 35988 KB/s, 1 seconds passed
... 44%, 56128 KB, 36003 KB/s, 1 seconds passed
... 44%, 56160 KB, 36018 KB/s, 1 seconds passed
... 44%, 56192 KB, 36034 KB/s, 1 seconds passed
... 44%, 56224 KB, 36050 KB/s, 1 seconds passed
... 44%, 56256 KB, 36066 KB/s, 1 seconds passed
... 44%, 56288 KB, 36082 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 56320 KB, 34253 KB/s, 1 seconds passed
... 44%, 56352 KB, 34262 KB/s, 1 seconds passed
... 44%, 56384 KB, 34252 KB/s, 1 seconds passed
... 44%, 56416 KB, 34261 KB/s, 1 seconds passed
... 44%, 56448 KB, 34272 KB/s, 1 seconds passed
... 44%, 56480 KB, 34283 KB/s, 1 seconds passed
... 44%, 56512 KB, 34293 KB/s, 1 seconds passed
... 44%, 56544 KB, 34306 KB/s, 1 seconds passed
... 44%, 56576 KB, 34319 KB/s, 1 seconds passed
... 44%, 56608 KB, 34326 KB/s, 1 seconds passed
... 44%, 56640 KB, 34338 KB/s, 1 seconds passed
... 44%, 56672 KB, 34349 KB/s, 1 seconds passed
... 45%, 56704 KB, 34361 KB/s, 1 seconds passed
... 45%, 56736 KB, 34372 KB/s, 1 seconds passed
... 45%, 56768 KB, 34384 KB/s, 1 seconds passed
... 45%, 56800 KB, 34395 KB/s, 1 seconds passed
... 45%, 56832 KB, 34406 KB/s, 1 seconds passed
... 45%, 56864 KB, 34416 KB/s, 1 seconds passed
... 45%, 56896 KB, 34428 KB/s, 1 seconds passed
... 45%, 56928 KB, 34439 KB/s, 1 seconds passed
... 45%, 56960 KB, 34450 KB/s, 1 seconds passed
... 45%, 56992 KB, 34464 KB/s, 1 seconds passed
... 45%, 57024 KB, 34477 KB/s, 1 seconds passed
... 45%, 57056 KB, 34490 KB/s, 1 seconds passed

.. parsed-literal::

    ... 45%, 57088 KB, 34461 KB/s, 1 seconds passed
... 45%, 57120 KB, 34471 KB/s, 1 seconds passed
... 45%, 57152 KB, 34483 KB/s, 1 seconds passed
... 45%, 57184 KB, 34495 KB/s, 1 seconds passed
... 45%, 57216 KB, 34506 KB/s, 1 seconds passed
... 45%, 57248 KB, 34517 KB/s, 1 seconds passed
... 45%, 57280 KB, 34528 KB/s, 1 seconds passed
... 45%, 57312 KB, 34539 KB/s, 1 seconds passed
... 45%, 57344 KB, 34550 KB/s, 1 seconds passed
... 45%, 57376 KB, 34561 KB/s, 1 seconds passed
... 45%, 57408 KB, 34572 KB/s, 1 seconds passed
... 45%, 57440 KB, 34584 KB/s, 1 seconds passed
... 45%, 57472 KB, 34595 KB/s, 1 seconds passed
... 45%, 57504 KB, 34607 KB/s, 1 seconds passed
... 45%, 57536 KB, 34618 KB/s, 1 seconds passed
... 45%, 57568 KB, 34630 KB/s, 1 seconds passed
... 45%, 57600 KB, 34643 KB/s, 1 seconds passed
... 45%, 57632 KB, 34656 KB/s, 1 seconds passed
... 45%, 57664 KB, 34669 KB/s, 1 seconds passed
... 45%, 57696 KB, 34682 KB/s, 1 seconds passed
... 45%, 57728 KB, 34695 KB/s, 1 seconds passed
... 45%, 57760 KB, 34709 KB/s, 1 seconds passed
... 45%, 57792 KB, 34723 KB/s, 1 seconds passed
... 45%, 57824 KB, 34737 KB/s, 1 seconds passed
... 45%, 57856 KB, 34751 KB/s, 1 seconds passed
... 45%, 57888 KB, 34766 KB/s, 1 seconds passed
... 45%, 57920 KB, 34465 KB/s, 1 seconds passed
... 46%, 57952 KB, 34474 KB/s, 1 seconds passed
... 46%, 57984 KB, 34484 KB/s, 1 seconds passed
... 46%, 58016 KB, 34495 KB/s, 1 seconds passed
... 46%, 58048 KB, 34507 KB/s, 1 seconds passed
... 46%, 58080 KB, 34520 KB/s, 1 seconds passed
... 46%, 58112 KB, 34363 KB/s, 1 seconds passed
... 46%, 58144 KB, 34370 KB/s, 1 seconds passed
... 46%, 58176 KB, 34380 KB/s, 1 seconds passed
... 46%, 58208 KB, 34391 KB/s, 1 seconds passed
... 46%, 58240 KB, 34402 KB/s, 1 seconds passed
... 46%, 58272 KB, 34414 KB/s, 1 seconds passed
... 46%, 58304 KB, 34425 KB/s, 1 seconds passed
... 46%, 58336 KB, 34437 KB/s, 1 seconds passed
... 46%, 58368 KB, 34449 KB/s, 1 seconds passed
... 46%, 58400 KB, 34462 KB/s, 1 seconds passed
... 46%, 58432 KB, 34475 KB/s, 1 seconds passed
... 46%, 58464 KB, 34316 KB/s, 1 seconds passed
... 46%, 58496 KB, 34325 KB/s, 1 seconds passed
... 46%, 58528 KB, 34334 KB/s, 1 seconds passed
... 46%, 58560 KB, 34345 KB/s, 1 seconds passed
... 46%, 58592 KB, 34356 KB/s, 1 seconds passed
... 46%, 58624 KB, 34367 KB/s, 1 seconds passed
... 46%, 58656 KB, 34378 KB/s, 1 seconds passed
... 46%, 58688 KB, 34389 KB/s, 1 seconds passed
... 46%, 58720 KB, 34400 KB/s, 1 seconds passed
... 46%, 58752 KB, 34411 KB/s, 1 seconds passed

.. parsed-literal::

    ... 46%, 58784 KB, 34423 KB/s, 1 seconds passed
... 46%, 58816 KB, 34433 KB/s, 1 seconds passed
... 46%, 58848 KB, 34444 KB/s, 1 seconds passed
... 46%, 58880 KB, 34455 KB/s, 1 seconds passed
... 46%, 58912 KB, 34466 KB/s, 1 seconds passed
... 46%, 58944 KB, 34476 KB/s, 1 seconds passed
... 46%, 58976 KB, 34487 KB/s, 1 seconds passed
... 46%, 59008 KB, 34498 KB/s, 1 seconds passed
... 46%, 59040 KB, 34509 KB/s, 1 seconds passed
... 46%, 59072 KB, 34520 KB/s, 1 seconds passed
... 46%, 59104 KB, 34532 KB/s, 1 seconds passed
... 46%, 59136 KB, 34542 KB/s, 1 seconds passed
... 46%, 59168 KB, 34553 KB/s, 1 seconds passed
... 47%, 59200 KB, 34564 KB/s, 1 seconds passed
... 47%, 59232 KB, 34575 KB/s, 1 seconds passed
... 47%, 59264 KB, 34588 KB/s, 1 seconds passed
... 47%, 59296 KB, 34600 KB/s, 1 seconds passed
... 47%, 59328 KB, 34613 KB/s, 1 seconds passed
... 47%, 59360 KB, 34626 KB/s, 1 seconds passed
... 47%, 59392 KB, 34639 KB/s, 1 seconds passed
... 47%, 59424 KB, 34652 KB/s, 1 seconds passed
... 47%, 59456 KB, 34665 KB/s, 1 seconds passed
... 47%, 59488 KB, 34678 KB/s, 1 seconds passed
... 47%, 59520 KB, 34691 KB/s, 1 seconds passed
... 47%, 59552 KB, 34703 KB/s, 1 seconds passed
... 47%, 59584 KB, 34716 KB/s, 1 seconds passed
... 47%, 59616 KB, 34729 KB/s, 1 seconds passed
... 47%, 59648 KB, 34742 KB/s, 1 seconds passed
... 47%, 59680 KB, 34754 KB/s, 1 seconds passed
... 47%, 59712 KB, 34766 KB/s, 1 seconds passed
... 47%, 59744 KB, 34779 KB/s, 1 seconds passed
... 47%, 59776 KB, 34793 KB/s, 1 seconds passed
... 47%, 59808 KB, 34805 KB/s, 1 seconds passed
... 47%, 59840 KB, 34818 KB/s, 1 seconds passed
... 47%, 59872 KB, 34831 KB/s, 1 seconds passed
... 47%, 59904 KB, 34844 KB/s, 1 seconds passed
... 47%, 59936 KB, 34857 KB/s, 1 seconds passed
... 47%, 59968 KB, 34870 KB/s, 1 seconds passed
... 47%, 60000 KB, 34883 KB/s, 1 seconds passed
... 47%, 60032 KB, 34895 KB/s, 1 seconds passed
... 47%, 60064 KB, 34908 KB/s, 1 seconds passed
... 47%, 60096 KB, 34921 KB/s, 1 seconds passed
... 47%, 60128 KB, 34934 KB/s, 1 seconds passed
... 47%, 60160 KB, 34947 KB/s, 1 seconds passed
... 47%, 60192 KB, 34959 KB/s, 1 seconds passed
... 47%, 60224 KB, 34972 KB/s, 1 seconds passed
... 47%, 60256 KB, 34986 KB/s, 1 seconds passed
... 47%, 60288 KB, 35000 KB/s, 1 seconds passed
... 47%, 60320 KB, 35014 KB/s, 1 seconds passed
... 47%, 60352 KB, 35028 KB/s, 1 seconds passed
... 47%, 60384 KB, 35043 KB/s, 1 seconds passed
... 47%, 60416 KB, 35058 KB/s, 1 seconds passed
... 47%, 60448 KB, 35074 KB/s, 1 seconds passed
... 48%, 60480 KB, 35089 KB/s, 1 seconds passed
... 48%, 60512 KB, 35105 KB/s, 1 seconds passed
... 48%, 60544 KB, 35120 KB/s, 1 seconds passed
... 48%, 60576 KB, 35135 KB/s, 1 seconds passed
... 48%, 60608 KB, 35134 KB/s, 1 seconds passed
... 48%, 60640 KB, 35147 KB/s, 1 seconds passed
... 48%, 60672 KB, 35159 KB/s, 1 seconds passed
... 48%, 60704 KB, 35173 KB/s, 1 seconds passed
... 48%, 60736 KB, 35187 KB/s, 1 seconds passed
... 48%, 60768 KB, 35159 KB/s, 1 seconds passed
... 48%, 60800 KB, 35171 KB/s, 1 seconds passed
... 48%, 60832 KB, 35185 KB/s, 1 seconds passed
... 48%, 60864 KB, 35199 KB/s, 1 seconds passed
... 48%, 60896 KB, 35211 KB/s, 1 seconds passed
... 48%, 60928 KB, 35225 KB/s, 1 seconds passed
... 48%, 60960 KB, 35238 KB/s, 1 seconds passed
... 48%, 60992 KB, 35250 KB/s, 1 seconds passed
... 48%, 61024 KB, 35263 KB/s, 1 seconds passed
... 48%, 61056 KB, 35276 KB/s, 1 seconds passed
... 48%, 61088 KB, 35289 KB/s, 1 seconds passed
... 48%, 61120 KB, 35301 KB/s, 1 seconds passed
... 48%, 61152 KB, 35314 KB/s, 1 seconds passed
... 48%, 61184 KB, 35326 KB/s, 1 seconds passed
... 48%, 61216 KB, 35339 KB/s, 1 seconds passed
... 48%, 61248 KB, 35350 KB/s, 1 seconds passed
... 48%, 61280 KB, 35363 KB/s, 1 seconds passed
... 48%, 61312 KB, 35376 KB/s, 1 seconds passed
... 48%, 61344 KB, 35390 KB/s, 1 seconds passed
... 48%, 61376 KB, 35404 KB/s, 1 seconds passed
... 48%, 61408 KB, 35415 KB/s, 1 seconds passed

.. parsed-literal::

    ... 48%, 61440 KB, 33490 KB/s, 1 seconds passed
... 48%, 61472 KB, 33468 KB/s, 1 seconds passed
... 48%, 61504 KB, 33477 KB/s, 1 seconds passed
... 48%, 61536 KB, 33487 KB/s, 1 seconds passed
... 48%, 61568 KB, 33456 KB/s, 1 seconds passed
... 48%, 61600 KB, 33463 KB/s, 1 seconds passed
... 48%, 61632 KB, 33473 KB/s, 1 seconds passed
... 48%, 61664 KB, 33483 KB/s, 1 seconds passed
... 48%, 61696 KB, 33494 KB/s, 1 seconds passed
... 49%, 61728 KB, 33504 KB/s, 1 seconds passed
... 49%, 61760 KB, 33514 KB/s, 1 seconds passed
... 49%, 61792 KB, 33525 KB/s, 1 seconds passed
... 49%, 61824 KB, 33535 KB/s, 1 seconds passed
... 49%, 61856 KB, 33545 KB/s, 1 seconds passed
... 49%, 61888 KB, 33555 KB/s, 1 seconds passed
... 49%, 61920 KB, 33566 KB/s, 1 seconds passed
... 49%, 61952 KB, 33576 KB/s, 1 seconds passed
... 49%, 61984 KB, 33586 KB/s, 1 seconds passed
... 49%, 62016 KB, 33597 KB/s, 1 seconds passed
... 49%, 62048 KB, 33607 KB/s, 1 seconds passed
... 49%, 62080 KB, 33617 KB/s, 1 seconds passed
... 49%, 62112 KB, 33628 KB/s, 1 seconds passed
... 49%, 62144 KB, 33638 KB/s, 1 seconds passed
... 49%, 62176 KB, 33648 KB/s, 1 seconds passed
... 49%, 62208 KB, 33658 KB/s, 1 seconds passed
... 49%, 62240 KB, 33668 KB/s, 1 seconds passed
... 49%, 62272 KB, 33679 KB/s, 1 seconds passed
... 49%, 62304 KB, 33689 KB/s, 1 seconds passed
... 49%, 62336 KB, 33699 KB/s, 1 seconds passed
... 49%, 62368 KB, 33710 KB/s, 1 seconds passed
... 49%, 62400 KB, 33720 KB/s, 1 seconds passed
... 49%, 62432 KB, 33731 KB/s, 1 seconds passed
... 49%, 62464 KB, 33741 KB/s, 1 seconds passed
... 49%, 62496 KB, 33752 KB/s, 1 seconds passed
... 49%, 62528 KB, 33762 KB/s, 1 seconds passed
... 49%, 62560 KB, 33772 KB/s, 1 seconds passed
... 49%, 62592 KB, 33782 KB/s, 1 seconds passed
... 49%, 62624 KB, 33792 KB/s, 1 seconds passed
... 49%, 62656 KB, 33802 KB/s, 1 seconds passed
... 49%, 62688 KB, 33815 KB/s, 1 seconds passed
... 49%, 62720 KB, 33828 KB/s, 1 seconds passed
... 49%, 62752 KB, 33841 KB/s, 1 seconds passed
... 49%, 62784 KB, 33854 KB/s, 1 seconds passed
... 49%, 62816 KB, 33866 KB/s, 1 seconds passed
... 49%, 62848 KB, 33879 KB/s, 1 seconds passed
... 49%, 62880 KB, 33892 KB/s, 1 seconds passed
... 49%, 62912 KB, 33904 KB/s, 1 seconds passed
... 49%, 62944 KB, 33917 KB/s, 1 seconds passed
... 49%, 62976 KB, 33930 KB/s, 1 seconds passed
... 50%, 63008 KB, 33942 KB/s, 1 seconds passed
... 50%, 63040 KB, 33955 KB/s, 1 seconds passed
... 50%, 63072 KB, 33968 KB/s, 1 seconds passed
... 50%, 63104 KB, 33980 KB/s, 1 seconds passed
... 50%, 63136 KB, 33993 KB/s, 1 seconds passed
... 50%, 63168 KB, 34006 KB/s, 1 seconds passed
... 50%, 63200 KB, 34019 KB/s, 1 seconds passed
... 50%, 63232 KB, 34032 KB/s, 1 seconds passed
... 50%, 63264 KB, 34046 KB/s, 1 seconds passed
... 50%, 63296 KB, 34060 KB/s, 1 seconds passed
... 50%, 63328 KB, 34073 KB/s, 1 seconds passed
... 50%, 63360 KB, 34087 KB/s, 1 seconds passed
... 50%, 63392 KB, 34101 KB/s, 1 seconds passed
... 50%, 63424 KB, 34114 KB/s, 1 seconds passed

.. parsed-literal::

    ... 50%, 63456 KB, 33583 KB/s, 1 seconds passed
... 50%, 63488 KB, 33591 KB/s, 1 seconds passed
... 50%, 63520 KB, 33601 KB/s, 1 seconds passed
... 50%, 63552 KB, 33611 KB/s, 1 seconds passed
... 50%, 63584 KB, 33620 KB/s, 1 seconds passed
... 50%, 63616 KB, 33631 KB/s, 1 seconds passed
... 50%, 63648 KB, 33640 KB/s, 1 seconds passed
... 50%, 63680 KB, 33650 KB/s, 1 seconds passed
... 50%, 63712 KB, 33660 KB/s, 1 seconds passed
... 50%, 63744 KB, 33670 KB/s, 1 seconds passed
... 50%, 63776 KB, 33681 KB/s, 1 seconds passed
... 50%, 63808 KB, 33691 KB/s, 1 seconds passed
... 50%, 63840 KB, 33701 KB/s, 1 seconds passed
... 50%, 63872 KB, 33711 KB/s, 1 seconds passed
... 50%, 63904 KB, 33723 KB/s, 1 seconds passed
... 50%, 63936 KB, 33735 KB/s, 1 seconds passed
... 50%, 63968 KB, 33745 KB/s, 1 seconds passed
... 50%, 64000 KB, 33756 KB/s, 1 seconds passed
... 50%, 64032 KB, 33767 KB/s, 1 seconds passed
... 50%, 64064 KB, 33757 KB/s, 1 seconds passed
... 50%, 64096 KB, 33764 KB/s, 1 seconds passed
... 50%, 64128 KB, 33773 KB/s, 1 seconds passed
... 50%, 64160 KB, 33783 KB/s, 1 seconds passed
... 50%, 64192 KB, 33793 KB/s, 1 seconds passed
... 50%, 64224 KB, 33802 KB/s, 1 seconds passed
... 51%, 64256 KB, 33812 KB/s, 1 seconds passed
... 51%, 64288 KB, 33823 KB/s, 1 seconds passed
... 51%, 64320 KB, 33835 KB/s, 1 seconds passed
... 51%, 64352 KB, 33846 KB/s, 1 seconds passed
... 51%, 64384 KB, 33857 KB/s, 1 seconds passed
... 51%, 64416 KB, 33859 KB/s, 1 seconds passed
... 51%, 64448 KB, 33869 KB/s, 1 seconds passed
... 51%, 64480 KB, 33879 KB/s, 1 seconds passed
... 51%, 64512 KB, 33889 KB/s, 1 seconds passed
... 51%, 64544 KB, 33898 KB/s, 1 seconds passed
... 51%, 64576 KB, 33908 KB/s, 1 seconds passed
... 51%, 64608 KB, 33919 KB/s, 1 seconds passed
... 51%, 64640 KB, 33931 KB/s, 1 seconds passed
... 51%, 64672 KB, 33943 KB/s, 1 seconds passed
... 51%, 64704 KB, 33954 KB/s, 1 seconds passed
... 51%, 64736 KB, 33966 KB/s, 1 seconds passed
... 51%, 64768 KB, 33978 KB/s, 1 seconds passed
... 51%, 64800 KB, 33990 KB/s, 1 seconds passed
... 51%, 64832 KB, 34001 KB/s, 1 seconds passed
... 51%, 64864 KB, 34012 KB/s, 1 seconds passed
... 51%, 64896 KB, 34024 KB/s, 1 seconds passed
... 51%, 64928 KB, 34036 KB/s, 1 seconds passed
... 51%, 64960 KB, 34047 KB/s, 1 seconds passed
... 51%, 64992 KB, 34059 KB/s, 1 seconds passed
... 51%, 65024 KB, 34070 KB/s, 1 seconds passed
... 51%, 65056 KB, 34082 KB/s, 1 seconds passed
... 51%, 65088 KB, 34093 KB/s, 1 seconds passed
... 51%, 65120 KB, 34104 KB/s, 1 seconds passed
... 51%, 65152 KB, 34115 KB/s, 1 seconds passed
... 51%, 65184 KB, 34127 KB/s, 1 seconds passed
... 51%, 65216 KB, 34138 KB/s, 1 seconds passed
... 51%, 65248 KB, 34150 KB/s, 1 seconds passed
... 51%, 65280 KB, 34161 KB/s, 1 seconds passed
... 51%, 65312 KB, 34173 KB/s, 1 seconds passed
... 51%, 65344 KB, 34184 KB/s, 1 seconds passed
... 51%, 65376 KB, 34195 KB/s, 1 seconds passed
... 51%, 65408 KB, 34207 KB/s, 1 seconds passed
... 51%, 65440 KB, 34218 KB/s, 1 seconds passed
... 51%, 65472 KB, 34229 KB/s, 1 seconds passed

.. parsed-literal::

    ... 52%, 65504 KB, 34241 KB/s, 1 seconds passed
... 52%, 65536 KB, 34252 KB/s, 1 seconds passed
... 52%, 65568 KB, 34264 KB/s, 1 seconds passed
... 52%, 65600 KB, 34276 KB/s, 1 seconds passed
... 52%, 65632 KB, 34289 KB/s, 1 seconds passed
... 52%, 65664 KB, 34301 KB/s, 1 seconds passed
... 52%, 65696 KB, 34313 KB/s, 1 seconds passed

.. parsed-literal::

    ... 52%, 65728 KB, 33443 KB/s, 1 seconds passed
... 52%, 65760 KB, 33450 KB/s, 1 seconds passed
... 52%, 65792 KB, 33459 KB/s, 1 seconds passed
... 52%, 65824 KB, 33469 KB/s, 1 seconds passed
... 52%, 65856 KB, 33478 KB/s, 1 seconds passed
... 52%, 65888 KB, 33488 KB/s, 1 seconds passed
... 52%, 65920 KB, 33497 KB/s, 1 seconds passed
... 52%, 65952 KB, 33506 KB/s, 1 seconds passed
... 52%, 65984 KB, 33516 KB/s, 1 seconds passed
... 52%, 66016 KB, 33525 KB/s, 1 seconds passed
... 52%, 66048 KB, 33534 KB/s, 1 seconds passed
... 52%, 66080 KB, 33544 KB/s, 1 seconds passed
... 52%, 66112 KB, 33554 KB/s, 1 seconds passed
... 52%, 66144 KB, 33563 KB/s, 1 seconds passed
... 52%, 66176 KB, 33573 KB/s, 1 seconds passed
... 52%, 66208 KB, 33583 KB/s, 1 seconds passed
... 52%, 66240 KB, 33592 KB/s, 1 seconds passed
... 52%, 66272 KB, 33602 KB/s, 1 seconds passed
... 52%, 66304 KB, 33612 KB/s, 1 seconds passed
... 52%, 66336 KB, 33623 KB/s, 1 seconds passed
... 52%, 66368 KB, 33634 KB/s, 1 seconds passed
... 52%, 66400 KB, 33645 KB/s, 1 seconds passed
... 52%, 66432 KB, 33656 KB/s, 1 seconds passed
... 52%, 66464 KB, 33668 KB/s, 1 seconds passed
... 52%, 66496 KB, 33679 KB/s, 1 seconds passed
... 52%, 66528 KB, 33691 KB/s, 1 seconds passed

.. parsed-literal::

    ... 52%, 66560 KB, 31506 KB/s, 2 seconds passed
... 52%, 66592 KB, 31514 KB/s, 2 seconds passed

.. parsed-literal::

    ... 52%, 66624 KB, 31450 KB/s, 2 seconds passed
... 52%, 66656 KB, 31457 KB/s, 2 seconds passed
... 52%, 66688 KB, 31466 KB/s, 2 seconds passed
... 52%, 66720 KB, 31475 KB/s, 2 seconds passed
... 52%, 66752 KB, 31482 KB/s, 2 seconds passed
... 53%, 66784 KB, 31491 KB/s, 2 seconds passed
... 53%, 66816 KB, 31501 KB/s, 2 seconds passed
... 53%, 66848 KB, 31510 KB/s, 2 seconds passed
... 53%, 66880 KB, 31519 KB/s, 2 seconds passed
... 53%, 66912 KB, 31528 KB/s, 2 seconds passed
... 53%, 66944 KB, 31537 KB/s, 2 seconds passed
... 53%, 66976 KB, 31547 KB/s, 2 seconds passed
... 53%, 67008 KB, 31556 KB/s, 2 seconds passed
... 53%, 67040 KB, 31565 KB/s, 2 seconds passed
... 53%, 67072 KB, 31574 KB/s, 2 seconds passed
... 53%, 67104 KB, 31583 KB/s, 2 seconds passed
... 53%, 67136 KB, 31592 KB/s, 2 seconds passed
... 53%, 67168 KB, 31602 KB/s, 2 seconds passed
... 53%, 67200 KB, 31611 KB/s, 2 seconds passed
... 53%, 67232 KB, 31620 KB/s, 2 seconds passed
... 53%, 67264 KB, 31630 KB/s, 2 seconds passed
... 53%, 67296 KB, 31640 KB/s, 2 seconds passed
... 53%, 67328 KB, 31649 KB/s, 2 seconds passed
... 53%, 67360 KB, 31659 KB/s, 2 seconds passed
... 53%, 67392 KB, 31667 KB/s, 2 seconds passed
... 53%, 67424 KB, 31677 KB/s, 2 seconds passed
... 53%, 67456 KB, 31687 KB/s, 2 seconds passed
... 53%, 67488 KB, 31697 KB/s, 2 seconds passed
... 53%, 67520 KB, 31706 KB/s, 2 seconds passed
... 53%, 67552 KB, 31716 KB/s, 2 seconds passed
... 53%, 67584 KB, 31726 KB/s, 2 seconds passed
... 53%, 67616 KB, 31735 KB/s, 2 seconds passed
... 53%, 67648 KB, 31745 KB/s, 2 seconds passed
... 53%, 67680 KB, 31755 KB/s, 2 seconds passed
... 53%, 67712 KB, 31764 KB/s, 2 seconds passed
... 53%, 67744 KB, 31774 KB/s, 2 seconds passed
... 53%, 67776 KB, 31783 KB/s, 2 seconds passed
... 53%, 67808 KB, 31793 KB/s, 2 seconds passed
... 53%, 67840 KB, 31803 KB/s, 2 seconds passed
... 53%, 67872 KB, 31813 KB/s, 2 seconds passed
... 53%, 67904 KB, 31822 KB/s, 2 seconds passed
... 53%, 67936 KB, 31832 KB/s, 2 seconds passed
... 53%, 67968 KB, 31843 KB/s, 2 seconds passed
... 53%, 68000 KB, 31854 KB/s, 2 seconds passed
... 54%, 68032 KB, 31865 KB/s, 2 seconds passed
... 54%, 68064 KB, 31876 KB/s, 2 seconds passed
... 54%, 68096 KB, 31887 KB/s, 2 seconds passed
... 54%, 68128 KB, 31897 KB/s, 2 seconds passed
... 54%, 68160 KB, 31908 KB/s, 2 seconds passed
... 54%, 68192 KB, 31919 KB/s, 2 seconds passed
... 54%, 68224 KB, 31930 KB/s, 2 seconds passed
... 54%, 68256 KB, 31941 KB/s, 2 seconds passed
... 54%, 68288 KB, 31954 KB/s, 2 seconds passed
... 54%, 68320 KB, 31966 KB/s, 2 seconds passed
... 54%, 68352 KB, 31978 KB/s, 2 seconds passed
... 54%, 68384 KB, 31991 KB/s, 2 seconds passed
... 54%, 68416 KB, 32003 KB/s, 2 seconds passed
... 54%, 68448 KB, 31932 KB/s, 2 seconds passed
... 54%, 68480 KB, 31939 KB/s, 2 seconds passed
... 54%, 68512 KB, 31947 KB/s, 2 seconds passed
... 54%, 68544 KB, 31958 KB/s, 2 seconds passed
... 54%, 68576 KB, 31969 KB/s, 2 seconds passed
... 54%, 68608 KB, 31981 KB/s, 2 seconds passed
... 54%, 68640 KB, 31992 KB/s, 2 seconds passed
... 54%, 68672 KB, 32004 KB/s, 2 seconds passed
... 54%, 68704 KB, 32015 KB/s, 2 seconds passed
... 54%, 68736 KB, 32026 KB/s, 2 seconds passed
... 54%, 68768 KB, 32037 KB/s, 2 seconds passed
... 54%, 68800 KB, 32048 KB/s, 2 seconds passed
... 54%, 68832 KB, 32059 KB/s, 2 seconds passed
... 54%, 68864 KB, 32069 KB/s, 2 seconds passed
... 54%, 68896 KB, 32080 KB/s, 2 seconds passed
... 54%, 68928 KB, 32090 KB/s, 2 seconds passed
... 54%, 68960 KB, 32101 KB/s, 2 seconds passed
... 54%, 68992 KB, 32112 KB/s, 2 seconds passed
... 54%, 69024 KB, 32123 KB/s, 2 seconds passed
... 54%, 69056 KB, 32134 KB/s, 2 seconds passed
... 54%, 69088 KB, 32145 KB/s, 2 seconds passed
... 54%, 69120 KB, 32155 KB/s, 2 seconds passed
... 54%, 69152 KB, 32166 KB/s, 2 seconds passed
... 54%, 69184 KB, 32177 KB/s, 2 seconds passed
... 54%, 69216 KB, 32188 KB/s, 2 seconds passed
... 54%, 69248 KB, 32199 KB/s, 2 seconds passed
... 55%, 69280 KB, 32209 KB/s, 2 seconds passed
... 55%, 69312 KB, 32220 KB/s, 2 seconds passed
... 55%, 69344 KB, 32231 KB/s, 2 seconds passed
... 55%, 69376 KB, 32242 KB/s, 2 seconds passed
... 55%, 69408 KB, 32252 KB/s, 2 seconds passed
... 55%, 69440 KB, 32263 KB/s, 2 seconds passed
... 55%, 69472 KB, 32274 KB/s, 2 seconds passed
... 55%, 69504 KB, 32284 KB/s, 2 seconds passed
... 55%, 69536 KB, 32295 KB/s, 2 seconds passed
... 55%, 69568 KB, 32306 KB/s, 2 seconds passed
... 55%, 69600 KB, 32316 KB/s, 2 seconds passed
... 55%, 69632 KB, 32326 KB/s, 2 seconds passed
... 55%, 69664 KB, 32337 KB/s, 2 seconds passed
... 55%, 69696 KB, 32347 KB/s, 2 seconds passed
... 55%, 69728 KB, 32358 KB/s, 2 seconds passed
... 55%, 69760 KB, 32369 KB/s, 2 seconds passed
... 55%, 69792 KB, 32380 KB/s, 2 seconds passed
... 55%, 69824 KB, 32391 KB/s, 2 seconds passed
... 55%, 69856 KB, 32401 KB/s, 2 seconds passed
... 55%, 69888 KB, 32412 KB/s, 2 seconds passed
... 55%, 69920 KB, 32422 KB/s, 2 seconds passed
... 55%, 69952 KB, 32433 KB/s, 2 seconds passed
... 55%, 69984 KB, 32444 KB/s, 2 seconds passed
... 55%, 70016 KB, 32455 KB/s, 2 seconds passed
... 55%, 70048 KB, 32465 KB/s, 2 seconds passed
... 55%, 70080 KB, 32476 KB/s, 2 seconds passed
... 55%, 70112 KB, 32487 KB/s, 2 seconds passed
... 55%, 70144 KB, 32497 KB/s, 2 seconds passed
... 55%, 70176 KB, 32507 KB/s, 2 seconds passed
... 55%, 70208 KB, 32518 KB/s, 2 seconds passed
... 55%, 70240 KB, 32529 KB/s, 2 seconds passed
... 55%, 70272 KB, 32540 KB/s, 2 seconds passed
... 55%, 70304 KB, 32550 KB/s, 2 seconds passed
... 55%, 70336 KB, 32561 KB/s, 2 seconds passed
... 55%, 70368 KB, 32572 KB/s, 2 seconds passed
... 55%, 70400 KB, 32582 KB/s, 2 seconds passed
... 55%, 70432 KB, 32593 KB/s, 2 seconds passed
... 55%, 70464 KB, 32604 KB/s, 2 seconds passed
... 55%, 70496 KB, 32614 KB/s, 2 seconds passed
... 55%, 70528 KB, 32624 KB/s, 2 seconds passed
... 56%, 70560 KB, 32635 KB/s, 2 seconds passed
... 56%, 70592 KB, 32646 KB/s, 2 seconds passed
... 56%, 70624 KB, 32656 KB/s, 2 seconds passed
... 56%, 70656 KB, 32667 KB/s, 2 seconds passed
... 56%, 70688 KB, 32678 KB/s, 2 seconds passed
... 56%, 70720 KB, 32688 KB/s, 2 seconds passed
... 56%, 70752 KB, 32699 KB/s, 2 seconds passed
... 56%, 70784 KB, 32709 KB/s, 2 seconds passed
... 56%, 70816 KB, 32720 KB/s, 2 seconds passed

.. parsed-literal::

    ... 56%, 70848 KB, 31577 KB/s, 2 seconds passed
... 56%, 70880 KB, 31584 KB/s, 2 seconds passed
... 56%, 70912 KB, 31592 KB/s, 2 seconds passed
... 56%, 70944 KB, 31600 KB/s, 2 seconds passed
... 56%, 70976 KB, 31609 KB/s, 2 seconds passed
... 56%, 71008 KB, 31618 KB/s, 2 seconds passed
... 56%, 71040 KB, 31627 KB/s, 2 seconds passed
... 56%, 71072 KB, 31636 KB/s, 2 seconds passed
... 56%, 71104 KB, 31645 KB/s, 2 seconds passed
... 56%, 71136 KB, 31654 KB/s, 2 seconds passed
... 56%, 71168 KB, 31662 KB/s, 2 seconds passed
... 56%, 71200 KB, 31670 KB/s, 2 seconds passed
... 56%, 71232 KB, 31679 KB/s, 2 seconds passed
... 56%, 71264 KB, 31688 KB/s, 2 seconds passed
... 56%, 71296 KB, 31696 KB/s, 2 seconds passed
... 56%, 71328 KB, 31705 KB/s, 2 seconds passed
... 56%, 71360 KB, 31714 KB/s, 2 seconds passed
... 56%, 71392 KB, 31723 KB/s, 2 seconds passed
... 56%, 71424 KB, 31732 KB/s, 2 seconds passed
... 56%, 71456 KB, 31742 KB/s, 2 seconds passed
... 56%, 71488 KB, 31753 KB/s, 2 seconds passed
... 56%, 71520 KB, 31763 KB/s, 2 seconds passed
... 56%, 71552 KB, 31772 KB/s, 2 seconds passed
... 56%, 71584 KB, 31782 KB/s, 2 seconds passed
... 56%, 71616 KB, 31792 KB/s, 2 seconds passed
... 56%, 71648 KB, 31802 KB/s, 2 seconds passed

.. parsed-literal::

    ... 56%, 71680 KB, 31325 KB/s, 2 seconds passed
... 56%, 71712 KB, 31332 KB/s, 2 seconds passed
... 56%, 71744 KB, 31340 KB/s, 2 seconds passed
... 56%, 71776 KB, 31349 KB/s, 2 seconds passed
... 57%, 71808 KB, 31357 KB/s, 2 seconds passed
... 57%, 71840 KB, 31366 KB/s, 2 seconds passed
... 57%, 71872 KB, 31374 KB/s, 2 seconds passed
... 57%, 71904 KB, 31383 KB/s, 2 seconds passed
... 57%, 71936 KB, 31392 KB/s, 2 seconds passed
... 57%, 71968 KB, 31400 KB/s, 2 seconds passed
... 57%, 72000 KB, 31409 KB/s, 2 seconds passed
... 57%, 72032 KB, 31417 KB/s, 2 seconds passed
... 57%, 72064 KB, 31426 KB/s, 2 seconds passed
... 57%, 72096 KB, 31434 KB/s, 2 seconds passed
... 57%, 72128 KB, 31443 KB/s, 2 seconds passed
... 57%, 72160 KB, 31451 KB/s, 2 seconds passed
... 57%, 72192 KB, 31459 KB/s, 2 seconds passed
... 57%, 72224 KB, 31468 KB/s, 2 seconds passed
... 57%, 72256 KB, 31476 KB/s, 2 seconds passed
... 57%, 72288 KB, 31485 KB/s, 2 seconds passed
... 57%, 72320 KB, 31493 KB/s, 2 seconds passed
... 57%, 72352 KB, 31502 KB/s, 2 seconds passed
... 57%, 72384 KB, 31511 KB/s, 2 seconds passed
... 57%, 72416 KB, 31519 KB/s, 2 seconds passed
... 57%, 72448 KB, 31528 KB/s, 2 seconds passed
... 57%, 72480 KB, 31537 KB/s, 2 seconds passed
... 57%, 72512 KB, 31546 KB/s, 2 seconds passed
... 57%, 72544 KB, 31556 KB/s, 2 seconds passed
... 57%, 72576 KB, 31566 KB/s, 2 seconds passed
... 57%, 72608 KB, 31577 KB/s, 2 seconds passed
... 57%, 72640 KB, 31586 KB/s, 2 seconds passed
... 57%, 72672 KB, 31596 KB/s, 2 seconds passed
... 57%, 72704 KB, 31606 KB/s, 2 seconds passed
... 57%, 72736 KB, 31615 KB/s, 2 seconds passed
... 57%, 72768 KB, 31625 KB/s, 2 seconds passed
... 57%, 72800 KB, 31635 KB/s, 2 seconds passed
... 57%, 72832 KB, 31645 KB/s, 2 seconds passed
... 57%, 72864 KB, 31655 KB/s, 2 seconds passed
... 57%, 72896 KB, 31665 KB/s, 2 seconds passed
... 57%, 72928 KB, 31675 KB/s, 2 seconds passed
... 57%, 72960 KB, 31686 KB/s, 2 seconds passed
... 57%, 72992 KB, 31695 KB/s, 2 seconds passed
... 57%, 73024 KB, 31705 KB/s, 2 seconds passed
... 58%, 73056 KB, 31715 KB/s, 2 seconds passed
... 58%, 73088 KB, 31725 KB/s, 2 seconds passed
... 58%, 73120 KB, 31734 KB/s, 2 seconds passed
... 58%, 73152 KB, 31744 KB/s, 2 seconds passed
... 58%, 73184 KB, 31754 KB/s, 2 seconds passed
... 58%, 73216 KB, 31764 KB/s, 2 seconds passed
... 58%, 73248 KB, 31774 KB/s, 2 seconds passed
... 58%, 73280 KB, 31784 KB/s, 2 seconds passed
... 58%, 73312 KB, 31794 KB/s, 2 seconds passed
... 58%, 73344 KB, 31805 KB/s, 2 seconds passed
... 58%, 73376 KB, 31816 KB/s, 2 seconds passed
... 58%, 73408 KB, 31827 KB/s, 2 seconds passed
... 58%, 73440 KB, 31838 KB/s, 2 seconds passed
... 58%, 73472 KB, 31848 KB/s, 2 seconds passed

.. parsed-literal::

    ... 58%, 73504 KB, 31636 KB/s, 2 seconds passed
... 58%, 73536 KB, 31643 KB/s, 2 seconds passed
... 58%, 73568 KB, 31612 KB/s, 2 seconds passed
... 58%, 73600 KB, 31621 KB/s, 2 seconds passed
... 58%, 73632 KB, 31626 KB/s, 2 seconds passed
... 58%, 73664 KB, 31635 KB/s, 2 seconds passed
... 58%, 73696 KB, 31608 KB/s, 2 seconds passed
... 58%, 73728 KB, 31605 KB/s, 2 seconds passed
... 58%, 73760 KB, 31612 KB/s, 2 seconds passed
... 58%, 73792 KB, 31614 KB/s, 2 seconds passed
... 58%, 73824 KB, 31623 KB/s, 2 seconds passed
... 58%, 73856 KB, 31633 KB/s, 2 seconds passed
... 58%, 73888 KB, 31642 KB/s, 2 seconds passed
... 58%, 73920 KB, 31652 KB/s, 2 seconds passed
... 58%, 73952 KB, 31661 KB/s, 2 seconds passed
... 58%, 73984 KB, 31671 KB/s, 2 seconds passed
... 58%, 74016 KB, 31680 KB/s, 2 seconds passed
... 58%, 74048 KB, 31690 KB/s, 2 seconds passed
... 58%, 74080 KB, 31700 KB/s, 2 seconds passed
... 58%, 74112 KB, 31709 KB/s, 2 seconds passed
... 58%, 74144 KB, 31719 KB/s, 2 seconds passed
... 58%, 74176 KB, 31728 KB/s, 2 seconds passed
... 58%, 74208 KB, 31684 KB/s, 2 seconds passed
... 58%, 74240 KB, 31691 KB/s, 2 seconds passed
... 58%, 74272 KB, 31700 KB/s, 2 seconds passed
... 58%, 74304 KB, 31710 KB/s, 2 seconds passed
... 59%, 74336 KB, 31719 KB/s, 2 seconds passed
... 59%, 74368 KB, 31728 KB/s, 2 seconds passed
... 59%, 74400 KB, 31738 KB/s, 2 seconds passed
... 59%, 74432 KB, 31747 KB/s, 2 seconds passed
... 59%, 74464 KB, 31757 KB/s, 2 seconds passed
... 59%, 74496 KB, 31767 KB/s, 2 seconds passed
... 59%, 74528 KB, 31776 KB/s, 2 seconds passed
... 59%, 74560 KB, 31786 KB/s, 2 seconds passed
... 59%, 74592 KB, 31795 KB/s, 2 seconds passed
... 59%, 74624 KB, 31804 KB/s, 2 seconds passed
... 59%, 74656 KB, 31814 KB/s, 2 seconds passed
... 59%, 74688 KB, 31823 KB/s, 2 seconds passed
... 59%, 74720 KB, 31833 KB/s, 2 seconds passed
... 59%, 74752 KB, 31842 KB/s, 2 seconds passed
... 59%, 74784 KB, 31852 KB/s, 2 seconds passed
... 59%, 74816 KB, 31861 KB/s, 2 seconds passed
... 59%, 74848 KB, 31871 KB/s, 2 seconds passed
... 59%, 74880 KB, 31880 KB/s, 2 seconds passed
... 59%, 74912 KB, 31889 KB/s, 2 seconds passed
... 59%, 74944 KB, 31899 KB/s, 2 seconds passed
... 59%, 74976 KB, 31908 KB/s, 2 seconds passed
... 59%, 75008 KB, 31918 KB/s, 2 seconds passed
... 59%, 75040 KB, 31927 KB/s, 2 seconds passed
... 59%, 75072 KB, 31937 KB/s, 2 seconds passed
... 59%, 75104 KB, 31946 KB/s, 2 seconds passed
... 59%, 75136 KB, 31956 KB/s, 2 seconds passed
... 59%, 75168 KB, 31965 KB/s, 2 seconds passed
... 59%, 75200 KB, 31975 KB/s, 2 seconds passed
... 59%, 75232 KB, 31984 KB/s, 2 seconds passed
... 59%, 75264 KB, 31994 KB/s, 2 seconds passed
... 59%, 75296 KB, 32004 KB/s, 2 seconds passed
... 59%, 75328 KB, 32013 KB/s, 2 seconds passed
... 59%, 75360 KB, 32023 KB/s, 2 seconds passed
... 59%, 75392 KB, 32032 KB/s, 2 seconds passed
... 59%, 75424 KB, 32042 KB/s, 2 seconds passed
... 59%, 75456 KB, 32051 KB/s, 2 seconds passed
... 59%, 75488 KB, 32060 KB/s, 2 seconds passed
... 59%, 75520 KB, 32070 KB/s, 2 seconds passed
... 59%, 75552 KB, 32058 KB/s, 2 seconds passed
... 60%, 75584 KB, 32066 KB/s, 2 seconds passed
... 60%, 75616 KB, 32057 KB/s, 2 seconds passed
... 60%, 75648 KB, 32065 KB/s, 2 seconds passed
... 60%, 75680 KB, 32050 KB/s, 2 seconds passed
... 60%, 75712 KB, 32057 KB/s, 2 seconds passed
... 60%, 75744 KB, 32066 KB/s, 2 seconds passed
... 60%, 75776 KB, 32076 KB/s, 2 seconds passed
... 60%, 75808 KB, 32086 KB/s, 2 seconds passed
... 60%, 75840 KB, 32095 KB/s, 2 seconds passed
... 60%, 75872 KB, 32105 KB/s, 2 seconds passed
... 60%, 75904 KB, 32115 KB/s, 2 seconds passed
... 60%, 75936 KB, 32125 KB/s, 2 seconds passed
... 60%, 75968 KB, 32135 KB/s, 2 seconds passed
... 60%, 76000 KB, 32145 KB/s, 2 seconds passed
... 60%, 76032 KB, 32155 KB/s, 2 seconds passed
... 60%, 76064 KB, 32165 KB/s, 2 seconds passed
... 60%, 76096 KB, 32175 KB/s, 2 seconds passed
... 60%, 76128 KB, 32185 KB/s, 2 seconds passed
... 60%, 76160 KB, 32195 KB/s, 2 seconds passed
... 60%, 76192 KB, 32205 KB/s, 2 seconds passed
... 60%, 76224 KB, 32215 KB/s, 2 seconds passed
... 60%, 76256 KB, 32225 KB/s, 2 seconds passed
... 60%, 76288 KB, 32234 KB/s, 2 seconds passed
... 60%, 76320 KB, 32244 KB/s, 2 seconds passed
... 60%, 76352 KB, 32255 KB/s, 2 seconds passed
... 60%, 76384 KB, 32264 KB/s, 2 seconds passed
... 60%, 76416 KB, 32273 KB/s, 2 seconds passed
... 60%, 76448 KB, 32283 KB/s, 2 seconds passed
... 60%, 76480 KB, 32293 KB/s, 2 seconds passed
... 60%, 76512 KB, 32303 KB/s, 2 seconds passed
... 60%, 76544 KB, 32313 KB/s, 2 seconds passed
... 60%, 76576 KB, 32323 KB/s, 2 seconds passed
... 60%, 76608 KB, 32333 KB/s, 2 seconds passed
... 60%, 76640 KB, 32341 KB/s, 2 seconds passed
... 60%, 76672 KB, 32352 KB/s, 2 seconds passed
... 60%, 76704 KB, 32361 KB/s, 2 seconds passed
... 60%, 76736 KB, 32370 KB/s, 2 seconds passed
... 60%, 76768 KB, 32380 KB/s, 2 seconds passed

.. parsed-literal::

    ... 60%, 76800 KB, 31195 KB/s, 2 seconds passed
... 61%, 76832 KB, 31203 KB/s, 2 seconds passed
... 61%, 76864 KB, 31211 KB/s, 2 seconds passed
... 61%, 76896 KB, 31220 KB/s, 2 seconds passed
... 61%, 76928 KB, 31226 KB/s, 2 seconds passed
... 61%, 76960 KB, 31235 KB/s, 2 seconds passed
... 61%, 76992 KB, 31243 KB/s, 2 seconds passed
... 61%, 77024 KB, 31252 KB/s, 2 seconds passed
... 61%, 77056 KB, 31245 KB/s, 2 seconds passed
... 61%, 77088 KB, 31252 KB/s, 2 seconds passed
... 61%, 77120 KB, 31261 KB/s, 2 seconds passed
... 61%, 77152 KB, 31270 KB/s, 2 seconds passed
... 61%, 77184 KB, 31279 KB/s, 2 seconds passed
... 61%, 77216 KB, 31287 KB/s, 2 seconds passed
... 61%, 77248 KB, 31296 KB/s, 2 seconds passed
... 61%, 77280 KB, 31305 KB/s, 2 seconds passed
... 61%, 77312 KB, 31315 KB/s, 2 seconds passed
... 61%, 77344 KB, 31324 KB/s, 2 seconds passed
... 61%, 77376 KB, 31327 KB/s, 2 seconds passed
... 61%, 77408 KB, 31336 KB/s, 2 seconds passed
... 61%, 77440 KB, 31345 KB/s, 2 seconds passed
... 61%, 77472 KB, 31354 KB/s, 2 seconds passed
... 61%, 77504 KB, 31305 KB/s, 2 seconds passed
... 61%, 77536 KB, 31312 KB/s, 2 seconds passed

.. parsed-literal::

    ... 61%, 77568 KB, 31304 KB/s, 2 seconds passed
... 61%, 77600 KB, 31311 KB/s, 2 seconds passed
... 61%, 77632 KB, 31320 KB/s, 2 seconds passed
... 61%, 77664 KB, 31329 KB/s, 2 seconds passed
... 61%, 77696 KB, 31338 KB/s, 2 seconds passed
... 61%, 77728 KB, 31346 KB/s, 2 seconds passed
... 61%, 77760 KB, 31355 KB/s, 2 seconds passed
... 61%, 77792 KB, 31364 KB/s, 2 seconds passed
... 61%, 77824 KB, 31373 KB/s, 2 seconds passed
... 61%, 77856 KB, 31382 KB/s, 2 seconds passed
... 61%, 77888 KB, 31391 KB/s, 2 seconds passed
... 61%, 77920 KB, 31401 KB/s, 2 seconds passed
... 61%, 77952 KB, 31410 KB/s, 2 seconds passed
... 61%, 77984 KB, 31419 KB/s, 2 seconds passed
... 61%, 78016 KB, 31428 KB/s, 2 seconds passed
... 61%, 78048 KB, 31437 KB/s, 2 seconds passed
... 61%, 78080 KB, 31446 KB/s, 2 seconds passed
... 62%, 78112 KB, 31455 KB/s, 2 seconds passed
... 62%, 78144 KB, 31464 KB/s, 2 seconds passed
... 62%, 78176 KB, 31473 KB/s, 2 seconds passed
... 62%, 78208 KB, 31482 KB/s, 2 seconds passed
... 62%, 78240 KB, 31491 KB/s, 2 seconds passed
... 62%, 78272 KB, 31500 KB/s, 2 seconds passed
... 62%, 78304 KB, 31509 KB/s, 2 seconds passed
... 62%, 78336 KB, 31518 KB/s, 2 seconds passed
... 62%, 78368 KB, 31527 KB/s, 2 seconds passed
... 62%, 78400 KB, 31536 KB/s, 2 seconds passed
... 62%, 78432 KB, 31545 KB/s, 2 seconds passed
... 62%, 78464 KB, 31554 KB/s, 2 seconds passed
... 62%, 78496 KB, 31563 KB/s, 2 seconds passed
... 62%, 78528 KB, 31572 KB/s, 2 seconds passed
... 62%, 78560 KB, 31581 KB/s, 2 seconds passed
... 62%, 78592 KB, 31590 KB/s, 2 seconds passed
... 62%, 78624 KB, 31599 KB/s, 2 seconds passed
... 62%, 78656 KB, 31608 KB/s, 2 seconds passed
... 62%, 78688 KB, 31617 KB/s, 2 seconds passed
... 62%, 78720 KB, 31626 KB/s, 2 seconds passed
... 62%, 78752 KB, 31635 KB/s, 2 seconds passed
... 62%, 78784 KB, 31645 KB/s, 2 seconds passed
... 62%, 78816 KB, 31655 KB/s, 2 seconds passed
... 62%, 78848 KB, 31665 KB/s, 2 seconds passed
... 62%, 78880 KB, 31675 KB/s, 2 seconds passed
... 62%, 78912 KB, 31686 KB/s, 2 seconds passed
... 62%, 78944 KB, 31696 KB/s, 2 seconds passed
... 62%, 78976 KB, 31706 KB/s, 2 seconds passed
... 62%, 79008 KB, 31716 KB/s, 2 seconds passed
... 62%, 79040 KB, 31727 KB/s, 2 seconds passed
... 62%, 79072 KB, 31737 KB/s, 2 seconds passed
... 62%, 79104 KB, 31747 KB/s, 2 seconds passed
... 62%, 79136 KB, 31757 KB/s, 2 seconds passed
... 62%, 79168 KB, 31768 KB/s, 2 seconds passed
... 62%, 79200 KB, 31778 KB/s, 2 seconds passed
... 62%, 79232 KB, 31788 KB/s, 2 seconds passed
... 62%, 79264 KB, 31798 KB/s, 2 seconds passed
... 62%, 79296 KB, 31809 KB/s, 2 seconds passed
... 62%, 79328 KB, 31818 KB/s, 2 seconds passed
... 63%, 79360 KB, 31827 KB/s, 2 seconds passed
... 63%, 79392 KB, 31837 KB/s, 2 seconds passed
... 63%, 79424 KB, 31846 KB/s, 2 seconds passed
... 63%, 79456 KB, 31855 KB/s, 2 seconds passed
... 63%, 79488 KB, 31864 KB/s, 2 seconds passed
... 63%, 79520 KB, 31874 KB/s, 2 seconds passed
... 63%, 79552 KB, 31883 KB/s, 2 seconds passed
... 63%, 79584 KB, 31609 KB/s, 2 seconds passed
... 63%, 79616 KB, 31616 KB/s, 2 seconds passed
... 63%, 79648 KB, 31625 KB/s, 2 seconds passed
... 63%, 79680 KB, 31633 KB/s, 2 seconds passed
... 63%, 79712 KB, 31642 KB/s, 2 seconds passed
... 63%, 79744 KB, 31651 KB/s, 2 seconds passed
... 63%, 79776 KB, 31660 KB/s, 2 seconds passed
... 63%, 79808 KB, 31668 KB/s, 2 seconds passed
... 63%, 79840 KB, 31677 KB/s, 2 seconds passed
... 63%, 79872 KB, 31686 KB/s, 2 seconds passed
... 63%, 79904 KB, 31695 KB/s, 2 seconds passed
... 63%, 79936 KB, 31704 KB/s, 2 seconds passed
... 63%, 79968 KB, 31712 KB/s, 2 seconds passed
... 63%, 80000 KB, 31721 KB/s, 2 seconds passed
... 63%, 80032 KB, 31730 KB/s, 2 seconds passed
... 63%, 80064 KB, 31739 KB/s, 2 seconds passed
... 63%, 80096 KB, 31748 KB/s, 2 seconds passed
... 63%, 80128 KB, 31757 KB/s, 2 seconds passed
... 63%, 80160 KB, 31765 KB/s, 2 seconds passed
... 63%, 80192 KB, 31774 KB/s, 2 seconds passed
... 63%, 80224 KB, 31783 KB/s, 2 seconds passed
... 63%, 80256 KB, 31792 KB/s, 2 seconds passed
... 63%, 80288 KB, 31801 KB/s, 2 seconds passed
... 63%, 80320 KB, 31810 KB/s, 2 seconds passed
... 63%, 80352 KB, 31819 KB/s, 2 seconds passed
... 63%, 80384 KB, 31828 KB/s, 2 seconds passed
... 63%, 80416 KB, 31837 KB/s, 2 seconds passed
... 63%, 80448 KB, 31846 KB/s, 2 seconds passed
... 63%, 80480 KB, 31854 KB/s, 2 seconds passed
... 63%, 80512 KB, 31863 KB/s, 2 seconds passed
... 63%, 80544 KB, 31872 KB/s, 2 seconds passed
... 63%, 80576 KB, 31880 KB/s, 2 seconds passed
... 63%, 80608 KB, 31889 KB/s, 2 seconds passed
... 64%, 80640 KB, 31898 KB/s, 2 seconds passed

.. parsed-literal::

    ... 64%, 80672 KB, 31907 KB/s, 2 seconds passed
... 64%, 80704 KB, 31916 KB/s, 2 seconds passed
... 64%, 80736 KB, 31925 KB/s, 2 seconds passed
... 64%, 80768 KB, 31933 KB/s, 2 seconds passed
... 64%, 80800 KB, 31942 KB/s, 2 seconds passed
... 64%, 80832 KB, 31952 KB/s, 2 seconds passed
... 64%, 80864 KB, 31962 KB/s, 2 seconds passed
... 64%, 80896 KB, 31972 KB/s, 2 seconds passed
... 64%, 80928 KB, 31982 KB/s, 2 seconds passed
... 64%, 80960 KB, 31992 KB/s, 2 seconds passed
... 64%, 80992 KB, 32002 KB/s, 2 seconds passed
... 64%, 81024 KB, 32012 KB/s, 2 seconds passed
... 64%, 81056 KB, 32022 KB/s, 2 seconds passed
... 64%, 81088 KB, 31443 KB/s, 2 seconds passed
... 64%, 81120 KB, 31450 KB/s, 2 seconds passed

.. parsed-literal::

    ... 64%, 81152 KB, 31458 KB/s, 2 seconds passed
... 64%, 81184 KB, 31467 KB/s, 2 seconds passed
... 64%, 81216 KB, 31476 KB/s, 2 seconds passed
... 64%, 81248 KB, 31485 KB/s, 2 seconds passed
... 64%, 81280 KB, 31494 KB/s, 2 seconds passed
... 64%, 81312 KB, 31502 KB/s, 2 seconds passed
... 64%, 81344 KB, 31511 KB/s, 2 seconds passed
... 64%, 81376 KB, 31519 KB/s, 2 seconds passed
... 64%, 81408 KB, 31528 KB/s, 2 seconds passed
... 64%, 81440 KB, 31537 KB/s, 2 seconds passed
... 64%, 81472 KB, 31545 KB/s, 2 seconds passed
... 64%, 81504 KB, 31554 KB/s, 2 seconds passed
... 64%, 81536 KB, 31563 KB/s, 2 seconds passed
... 64%, 81568 KB, 31571 KB/s, 2 seconds passed
... 64%, 81600 KB, 31580 KB/s, 2 seconds passed
... 64%, 81632 KB, 31589 KB/s, 2 seconds passed
... 64%, 81664 KB, 31597 KB/s, 2 seconds passed
... 64%, 81696 KB, 31606 KB/s, 2 seconds passed
... 64%, 81728 KB, 31614 KB/s, 2 seconds passed
... 64%, 81760 KB, 31623 KB/s, 2 seconds passed
... 64%, 81792 KB, 31632 KB/s, 2 seconds passed
... 64%, 81824 KB, 31641 KB/s, 2 seconds passed
... 64%, 81856 KB, 31649 KB/s, 2 seconds passed
... 65%, 81888 KB, 31658 KB/s, 2 seconds passed

.. parsed-literal::

    ... 65%, 81920 KB, 30454 KB/s, 2 seconds passed
... 65%, 81952 KB, 30460 KB/s, 2 seconds passed
... 65%, 81984 KB, 30468 KB/s, 2 seconds passed
... 65%, 82016 KB, 30476 KB/s, 2 seconds passed
... 65%, 82048 KB, 30484 KB/s, 2 seconds passed
... 65%, 82080 KB, 30493 KB/s, 2 seconds passed
... 65%, 82112 KB, 30501 KB/s, 2 seconds passed
... 65%, 82144 KB, 30509 KB/s, 2 seconds passed
... 65%, 82176 KB, 30518 KB/s, 2 seconds passed
... 65%, 82208 KB, 30526 KB/s, 2 seconds passed
... 65%, 82240 KB, 30535 KB/s, 2 seconds passed
... 65%, 82272 KB, 30543 KB/s, 2 seconds passed
... 65%, 82304 KB, 30552 KB/s, 2 seconds passed
... 65%, 82336 KB, 30560 KB/s, 2 seconds passed
... 65%, 82368 KB, 30568 KB/s, 2 seconds passed
... 65%, 82400 KB, 30577 KB/s, 2 seconds passed
... 65%, 82432 KB, 30585 KB/s, 2 seconds passed
... 65%, 82464 KB, 30594 KB/s, 2 seconds passed
... 65%, 82496 KB, 30602 KB/s, 2 seconds passed
... 65%, 82528 KB, 30610 KB/s, 2 seconds passed
... 65%, 82560 KB, 30619 KB/s, 2 seconds passed
... 65%, 82592 KB, 30627 KB/s, 2 seconds passed
... 65%, 82624 KB, 30635 KB/s, 2 seconds passed
... 65%, 82656 KB, 30644 KB/s, 2 seconds passed
... 65%, 82688 KB, 30652 KB/s, 2 seconds passed
... 65%, 82720 KB, 30660 KB/s, 2 seconds passed
... 65%, 82752 KB, 30669 KB/s, 2 seconds passed
... 65%, 82784 KB, 30677 KB/s, 2 seconds passed
... 65%, 82816 KB, 30686 KB/s, 2 seconds passed
... 65%, 82848 KB, 30694 KB/s, 2 seconds passed
... 65%, 82880 KB, 30703 KB/s, 2 seconds passed
... 65%, 82912 KB, 30630 KB/s, 2 seconds passed
... 65%, 82944 KB, 30637 KB/s, 2 seconds passed
... 65%, 82976 KB, 30645 KB/s, 2 seconds passed
... 65%, 83008 KB, 30653 KB/s, 2 seconds passed
... 65%, 83040 KB, 30661 KB/s, 2 seconds passed
... 65%, 83072 KB, 30670 KB/s, 2 seconds passed
... 65%, 83104 KB, 30679 KB/s, 2 seconds passed
... 66%, 83136 KB, 30688 KB/s, 2 seconds passed
... 66%, 83168 KB, 30696 KB/s, 2 seconds passed
... 66%, 83200 KB, 30705 KB/s, 2 seconds passed
... 66%, 83232 KB, 30714 KB/s, 2 seconds passed
... 66%, 83264 KB, 30722 KB/s, 2 seconds passed
... 66%, 83296 KB, 30731 KB/s, 2 seconds passed
... 66%, 83328 KB, 30740 KB/s, 2 seconds passed
... 66%, 83360 KB, 30749 KB/s, 2 seconds passed
... 66%, 83392 KB, 30757 KB/s, 2 seconds passed
... 66%, 83424 KB, 30766 KB/s, 2 seconds passed
... 66%, 83456 KB, 30774 KB/s, 2 seconds passed
... 66%, 83488 KB, 30783 KB/s, 2 seconds passed
... 66%, 83520 KB, 30744 KB/s, 2 seconds passed
... 66%, 83552 KB, 30650 KB/s, 2 seconds passed
... 66%, 83584 KB, 30657 KB/s, 2 seconds passed
... 66%, 83616 KB, 30664 KB/s, 2 seconds passed
... 66%, 83648 KB, 30673 KB/s, 2 seconds passed

.. parsed-literal::

    ... 66%, 83680 KB, 30585 KB/s, 2 seconds passed
... 66%, 83712 KB, 30570 KB/s, 2 seconds passed
... 66%, 83744 KB, 30577 KB/s, 2 seconds passed
... 66%, 83776 KB, 30585 KB/s, 2 seconds passed
... 66%, 83808 KB, 30593 KB/s, 2 seconds passed
... 66%, 83840 KB, 30601 KB/s, 2 seconds passed
... 66%, 83872 KB, 30609 KB/s, 2 seconds passed
... 66%, 83904 KB, 30576 KB/s, 2 seconds passed
... 66%, 83936 KB, 30573 KB/s, 2 seconds passed
... 66%, 83968 KB, 30575 KB/s, 2 seconds passed
... 66%, 84000 KB, 30583 KB/s, 2 seconds passed
... 66%, 84032 KB, 30591 KB/s, 2 seconds passed
... 66%, 84064 KB, 30599 KB/s, 2 seconds passed
... 66%, 84096 KB, 30607 KB/s, 2 seconds passed
... 66%, 84128 KB, 30615 KB/s, 2 seconds passed
... 66%, 84160 KB, 30623 KB/s, 2 seconds passed
... 66%, 84192 KB, 30631 KB/s, 2 seconds passed
... 66%, 84224 KB, 30639 KB/s, 2 seconds passed
... 66%, 84256 KB, 30648 KB/s, 2 seconds passed
... 66%, 84288 KB, 30655 KB/s, 2 seconds passed
... 66%, 84320 KB, 30664 KB/s, 2 seconds passed
... 66%, 84352 KB, 30672 KB/s, 2 seconds passed
... 66%, 84384 KB, 30680 KB/s, 2 seconds passed
... 67%, 84416 KB, 30689 KB/s, 2 seconds passed
... 67%, 84448 KB, 30697 KB/s, 2 seconds passed
... 67%, 84480 KB, 30665 KB/s, 2 seconds passed
... 67%, 84512 KB, 30672 KB/s, 2 seconds passed
... 67%, 84544 KB, 30680 KB/s, 2 seconds passed
... 67%, 84576 KB, 30676 KB/s, 2 seconds passed
... 67%, 84608 KB, 30683 KB/s, 2 seconds passed
... 67%, 84640 KB, 30692 KB/s, 2 seconds passed
... 67%, 84672 KB, 30700 KB/s, 2 seconds passed
... 67%, 84704 KB, 30708 KB/s, 2 seconds passed
... 67%, 84736 KB, 30716 KB/s, 2 seconds passed
... 67%, 84768 KB, 30724 KB/s, 2 seconds passed
... 67%, 84800 KB, 30732 KB/s, 2 seconds passed
... 67%, 84832 KB, 30741 KB/s, 2 seconds passed
... 67%, 84864 KB, 30749 KB/s, 2 seconds passed
... 67%, 84896 KB, 30757 KB/s, 2 seconds passed
... 67%, 84928 KB, 30765 KB/s, 2 seconds passed
... 67%, 84960 KB, 30773 KB/s, 2 seconds passed
... 67%, 84992 KB, 30781 KB/s, 2 seconds passed
... 67%, 85024 KB, 30789 KB/s, 2 seconds passed
... 67%, 85056 KB, 30798 KB/s, 2 seconds passed
... 67%, 85088 KB, 30806 KB/s, 2 seconds passed
... 67%, 85120 KB, 30814 KB/s, 2 seconds passed
... 67%, 85152 KB, 30822 KB/s, 2 seconds passed
... 67%, 85184 KB, 30830 KB/s, 2 seconds passed
... 67%, 85216 KB, 30838 KB/s, 2 seconds passed
... 67%, 85248 KB, 30846 KB/s, 2 seconds passed
... 67%, 85280 KB, 30854 KB/s, 2 seconds passed
... 67%, 85312 KB, 30862 KB/s, 2 seconds passed
... 67%, 85344 KB, 30870 KB/s, 2 seconds passed
... 67%, 85376 KB, 30878 KB/s, 2 seconds passed
... 67%, 85408 KB, 30887 KB/s, 2 seconds passed
... 67%, 85440 KB, 30895 KB/s, 2 seconds passed
... 67%, 85472 KB, 30903 KB/s, 2 seconds passed
... 67%, 85504 KB, 30911 KB/s, 2 seconds passed
... 67%, 85536 KB, 30919 KB/s, 2 seconds passed
... 67%, 85568 KB, 30927 KB/s, 2 seconds passed
... 67%, 85600 KB, 30935 KB/s, 2 seconds passed
... 67%, 85632 KB, 30943 KB/s, 2 seconds passed
... 68%, 85664 KB, 30952 KB/s, 2 seconds passed
... 68%, 85696 KB, 30960 KB/s, 2 seconds passed
... 68%, 85728 KB, 30968 KB/s, 2 seconds passed
... 68%, 85760 KB, 30976 KB/s, 2 seconds passed
... 68%, 85792 KB, 30984 KB/s, 2 seconds passed
... 68%, 85824 KB, 30992 KB/s, 2 seconds passed
... 68%, 85856 KB, 31000 KB/s, 2 seconds passed
... 68%, 85888 KB, 31008 KB/s, 2 seconds passed
... 68%, 85920 KB, 31016 KB/s, 2 seconds passed
... 68%, 85952 KB, 31025 KB/s, 2 seconds passed
... 68%, 85984 KB, 31033 KB/s, 2 seconds passed
... 68%, 86016 KB, 31041 KB/s, 2 seconds passed
... 68%, 86048 KB, 31049 KB/s, 2 seconds passed
... 68%, 86080 KB, 31057 KB/s, 2 seconds passed
... 68%, 86112 KB, 31065 KB/s, 2 seconds passed
... 68%, 86144 KB, 31074 KB/s, 2 seconds passed
... 68%, 86176 KB, 31084 KB/s, 2 seconds passed
... 68%, 86208 KB, 30991 KB/s, 2 seconds passed
... 68%, 86240 KB, 30999 KB/s, 2 seconds passed
... 68%, 86272 KB, 31007 KB/s, 2 seconds passed
... 68%, 86304 KB, 31014 KB/s, 2 seconds passed
... 68%, 86336 KB, 31020 KB/s, 2 seconds passed
... 68%, 86368 KB, 31028 KB/s, 2 seconds passed
... 68%, 86400 KB, 31036 KB/s, 2 seconds passed
... 68%, 86432 KB, 31045 KB/s, 2 seconds passed
... 68%, 86464 KB, 31050 KB/s, 2 seconds passed

.. parsed-literal::

    ... 68%, 86496 KB, 31057 KB/s, 2 seconds passed
... 68%, 86528 KB, 31064 KB/s, 2 seconds passed
... 68%, 86560 KB, 31071 KB/s, 2 seconds passed
... 68%, 86592 KB, 31075 KB/s, 2 seconds passed
... 68%, 86624 KB, 31083 KB/s, 2 seconds passed
... 68%, 86656 KB, 31091 KB/s, 2 seconds passed
... 68%, 86688 KB, 31099 KB/s, 2 seconds passed
... 68%, 86720 KB, 31104 KB/s, 2 seconds passed
... 68%, 86752 KB, 31111 KB/s, 2 seconds passed
... 68%, 86784 KB, 31117 KB/s, 2 seconds passed
... 68%, 86816 KB, 31125 KB/s, 2 seconds passed
... 68%, 86848 KB, 31133 KB/s, 2 seconds passed
... 68%, 86880 KB, 31142 KB/s, 2 seconds passed
... 69%, 86912 KB, 31149 KB/s, 2 seconds passed
... 69%, 86944 KB, 31157 KB/s, 2 seconds passed
... 69%, 86976 KB, 31165 KB/s, 2 seconds passed
... 69%, 87008 KB, 31173 KB/s, 2 seconds passed

.. parsed-literal::

    ... 69%, 87040 KB, 30373 KB/s, 2 seconds passed
... 69%, 87072 KB, 30379 KB/s, 2 seconds passed
... 69%, 87104 KB, 30387 KB/s, 2 seconds passed
... 69%, 87136 KB, 30395 KB/s, 2 seconds passed
... 69%, 87168 KB, 30393 KB/s, 2 seconds passed
... 69%, 87200 KB, 30400 KB/s, 2 seconds passed
... 69%, 87232 KB, 30408 KB/s, 2 seconds passed
... 69%, 87264 KB, 30416 KB/s, 2 seconds passed
... 69%, 87296 KB, 30424 KB/s, 2 seconds passed
... 69%, 87328 KB, 30432 KB/s, 2 seconds passed
... 69%, 87360 KB, 30440 KB/s, 2 seconds passed
... 69%, 87392 KB, 30448 KB/s, 2 seconds passed
... 69%, 87424 KB, 30456 KB/s, 2 seconds passed
... 69%, 87456 KB, 30464 KB/s, 2 seconds passed
... 69%, 87488 KB, 30472 KB/s, 2 seconds passed
... 69%, 87520 KB, 30479 KB/s, 2 seconds passed
... 69%, 87552 KB, 30487 KB/s, 2 seconds passed
... 69%, 87584 KB, 30495 KB/s, 2 seconds passed
... 69%, 87616 KB, 30503 KB/s, 2 seconds passed
... 69%, 87648 KB, 30511 KB/s, 2 seconds passed
... 69%, 87680 KB, 30519 KB/s, 2 seconds passed
... 69%, 87712 KB, 30526 KB/s, 2 seconds passed
... 69%, 87744 KB, 30534 KB/s, 2 seconds passed
... 69%, 87776 KB, 30542 KB/s, 2 seconds passed
... 69%, 87808 KB, 30550 KB/s, 2 seconds passed
... 69%, 87840 KB, 30558 KB/s, 2 seconds passed
... 69%, 87872 KB, 30566 KB/s, 2 seconds passed
... 69%, 87904 KB, 30574 KB/s, 2 seconds passed
... 69%, 87936 KB, 30582 KB/s, 2 seconds passed
... 69%, 87968 KB, 30590 KB/s, 2 seconds passed
... 69%, 88000 KB, 30597 KB/s, 2 seconds passed
... 69%, 88032 KB, 30605 KB/s, 2 seconds passed
... 69%, 88064 KB, 30613 KB/s, 2 seconds passed
... 69%, 88096 KB, 30621 KB/s, 2 seconds passed
... 69%, 88128 KB, 30629 KB/s, 2 seconds passed
... 69%, 88160 KB, 30637 KB/s, 2 seconds passed
... 70%, 88192 KB, 30645 KB/s, 2 seconds passed
... 70%, 88224 KB, 30653 KB/s, 2 seconds passed
... 70%, 88256 KB, 30661 KB/s, 2 seconds passed
... 70%, 88288 KB, 30668 KB/s, 2 seconds passed
... 70%, 88320 KB, 30676 KB/s, 2 seconds passed
... 70%, 88352 KB, 30684 KB/s, 2 seconds passed
... 70%, 88384 KB, 30692 KB/s, 2 seconds passed
... 70%, 88416 KB, 30700 KB/s, 2 seconds passed
... 70%, 88448 KB, 30708 KB/s, 2 seconds passed
... 70%, 88480 KB, 30715 KB/s, 2 seconds passed
... 70%, 88512 KB, 30723 KB/s, 2 seconds passed
... 70%, 88544 KB, 30731 KB/s, 2 seconds passed
... 70%, 88576 KB, 30739 KB/s, 2 seconds passed
... 70%, 88608 KB, 30747 KB/s, 2 seconds passed
... 70%, 88640 KB, 30755 KB/s, 2 seconds passed
... 70%, 88672 KB, 30763 KB/s, 2 seconds passed
... 70%, 88704 KB, 30771 KB/s, 2 seconds passed
... 70%, 88736 KB, 30779 KB/s, 2 seconds passed
... 70%, 88768 KB, 30776 KB/s, 2 seconds passed
... 70%, 88800 KB, 30785 KB/s, 2 seconds passed
... 70%, 88832 KB, 30792 KB/s, 2 seconds passed
... 70%, 88864 KB, 30800 KB/s, 2 seconds passed
... 70%, 88896 KB, 30809 KB/s, 2 seconds passed

.. parsed-literal::

    ... 70%, 88928 KB, 30535 KB/s, 2 seconds passed
... 70%, 88960 KB, 30542 KB/s, 2 seconds passed
... 70%, 88992 KB, 30549 KB/s, 2 seconds passed
... 70%, 89024 KB, 30557 KB/s, 2 seconds passed
... 70%, 89056 KB, 30565 KB/s, 2 seconds passed
... 70%, 89088 KB, 30571 KB/s, 2 seconds passed
... 70%, 89120 KB, 30577 KB/s, 2 seconds passed
... 70%, 89152 KB, 30580 KB/s, 2 seconds passed
... 70%, 89184 KB, 30587 KB/s, 2 seconds passed
... 70%, 89216 KB, 30583 KB/s, 2 seconds passed
... 70%, 89248 KB, 30590 KB/s, 2 seconds passed
... 70%, 89280 KB, 30597 KB/s, 2 seconds passed
... 70%, 89312 KB, 30605 KB/s, 2 seconds passed
... 70%, 89344 KB, 30613 KB/s, 2 seconds passed
... 70%, 89376 KB, 30621 KB/s, 2 seconds passed
... 70%, 89408 KB, 30628 KB/s, 2 seconds passed
... 71%, 89440 KB, 30636 KB/s, 2 seconds passed
... 71%, 89472 KB, 30644 KB/s, 2 seconds passed
... 71%, 89504 KB, 30652 KB/s, 2 seconds passed
... 71%, 89536 KB, 30645 KB/s, 2 seconds passed
... 71%, 89568 KB, 30647 KB/s, 2 seconds passed
... 71%, 89600 KB, 30641 KB/s, 2 seconds passed
... 71%, 89632 KB, 30648 KB/s, 2 seconds passed
... 71%, 89664 KB, 30649 KB/s, 2 seconds passed
... 71%, 89696 KB, 30656 KB/s, 2 seconds passed
... 71%, 89728 KB, 30664 KB/s, 2 seconds passed
... 71%, 89760 KB, 30672 KB/s, 2 seconds passed
... 71%, 89792 KB, 30679 KB/s, 2 seconds passed
... 71%, 89824 KB, 30687 KB/s, 2 seconds passed
... 71%, 89856 KB, 30695 KB/s, 2 seconds passed
... 71%, 89888 KB, 30702 KB/s, 2 seconds passed
... 71%, 89920 KB, 30710 KB/s, 2 seconds passed
... 71%, 89952 KB, 30718 KB/s, 2 seconds passed
... 71%, 89984 KB, 30725 KB/s, 2 seconds passed
... 71%, 90016 KB, 30733 KB/s, 2 seconds passed
... 71%, 90048 KB, 30741 KB/s, 2 seconds passed
... 71%, 90080 KB, 30749 KB/s, 2 seconds passed
... 71%, 90112 KB, 30756 KB/s, 2 seconds passed
... 71%, 90144 KB, 30764 KB/s, 2 seconds passed
... 71%, 90176 KB, 30772 KB/s, 2 seconds passed
... 71%, 90208 KB, 30780 KB/s, 2 seconds passed
... 71%, 90240 KB, 30788 KB/s, 2 seconds passed
... 71%, 90272 KB, 30796 KB/s, 2 seconds passed
... 71%, 90304 KB, 30803 KB/s, 2 seconds passed
... 71%, 90336 KB, 30811 KB/s, 2 seconds passed
... 71%, 90368 KB, 30818 KB/s, 2 seconds passed
... 71%, 90400 KB, 30826 KB/s, 2 seconds passed
... 71%, 90432 KB, 30834 KB/s, 2 seconds passed
... 71%, 90464 KB, 30842 KB/s, 2 seconds passed
... 71%, 90496 KB, 30849 KB/s, 2 seconds passed
... 71%, 90528 KB, 30857 KB/s, 2 seconds passed
... 71%, 90560 KB, 30865 KB/s, 2 seconds passed
... 71%, 90592 KB, 30873 KB/s, 2 seconds passed
... 71%, 90624 KB, 30882 KB/s, 2 seconds passed
... 71%, 90656 KB, 30890 KB/s, 2 seconds passed
... 72%, 90688 KB, 30899 KB/s, 2 seconds passed
... 72%, 90720 KB, 30907 KB/s, 2 seconds passed
... 72%, 90752 KB, 30916 KB/s, 2 seconds passed
... 72%, 90784 KB, 30924 KB/s, 2 seconds passed
... 72%, 90816 KB, 30932 KB/s, 2 seconds passed
... 72%, 90848 KB, 30941 KB/s, 2 seconds passed
... 72%, 90880 KB, 30945 KB/s, 2 seconds passed
... 72%, 90912 KB, 30953 KB/s, 2 seconds passed
... 72%, 90944 KB, 30961 KB/s, 2 seconds passed
... 72%, 90976 KB, 30969 KB/s, 2 seconds passed
... 72%, 91008 KB, 30977 KB/s, 2 seconds passed
... 72%, 91040 KB, 30985 KB/s, 2 seconds passed
... 72%, 91072 KB, 30993 KB/s, 2 seconds passed
... 72%, 91104 KB, 31001 KB/s, 2 seconds passed

.. parsed-literal::

    ... 72%, 91136 KB, 31009 KB/s, 2 seconds passed
... 72%, 91168 KB, 31017 KB/s, 2 seconds passed
... 72%, 91200 KB, 31025 KB/s, 2 seconds passed
... 72%, 91232 KB, 31033 KB/s, 2 seconds passed
... 72%, 91264 KB, 31041 KB/s, 2 seconds passed
... 72%, 91296 KB, 31048 KB/s, 2 seconds passed
... 72%, 91328 KB, 31057 KB/s, 2 seconds passed
... 72%, 91360 KB, 31065 KB/s, 2 seconds passed
... 72%, 91392 KB, 31073 KB/s, 2 seconds passed
... 72%, 91424 KB, 31081 KB/s, 2 seconds passed
... 72%, 91456 KB, 31089 KB/s, 2 seconds passed
... 72%, 91488 KB, 31097 KB/s, 2 seconds passed
... 72%, 91520 KB, 31105 KB/s, 2 seconds passed
... 72%, 91552 KB, 31112 KB/s, 2 seconds passed
... 72%, 91584 KB, 31121 KB/s, 2 seconds passed
... 72%, 91616 KB, 31129 KB/s, 2 seconds passed
... 72%, 91648 KB, 31136 KB/s, 2 seconds passed
... 72%, 91680 KB, 31144 KB/s, 2 seconds passed
... 72%, 91712 KB, 31152 KB/s, 2 seconds passed
... 72%, 91744 KB, 31160 KB/s, 2 seconds passed
... 72%, 91776 KB, 31168 KB/s, 2 seconds passed
... 72%, 91808 KB, 31176 KB/s, 2 seconds passed
... 72%, 91840 KB, 31184 KB/s, 2 seconds passed
... 72%, 91872 KB, 31192 KB/s, 2 seconds passed
... 72%, 91904 KB, 31200 KB/s, 2 seconds passed
... 72%, 91936 KB, 31208 KB/s, 2 seconds passed
... 73%, 91968 KB, 31216 KB/s, 2 seconds passed
... 73%, 92000 KB, 31223 KB/s, 2 seconds passed
... 73%, 92032 KB, 31232 KB/s, 2 seconds passed
... 73%, 92064 KB, 31240 KB/s, 2 seconds passed
... 73%, 92096 KB, 31248 KB/s, 2 seconds passed
... 73%, 92128 KB, 31256 KB/s, 2 seconds passed

.. parsed-literal::

    ... 73%, 92160 KB, 30550 KB/s, 3 seconds passed
... 73%, 92192 KB, 30556 KB/s, 3 seconds passed
... 73%, 92224 KB, 30558 KB/s, 3 seconds passed
... 73%, 92256 KB, 30566 KB/s, 3 seconds passed
... 73%, 92288 KB, 30573 KB/s, 3 seconds passed
... 73%, 92320 KB, 30580 KB/s, 3 seconds passed
... 73%, 92352 KB, 30588 KB/s, 3 seconds passed
... 73%, 92384 KB, 30595 KB/s, 3 seconds passed
... 73%, 92416 KB, 30603 KB/s, 3 seconds passed
... 73%, 92448 KB, 30610 KB/s, 3 seconds passed
... 73%, 92480 KB, 30618 KB/s, 3 seconds passed
... 73%, 92512 KB, 30625 KB/s, 3 seconds passed
... 73%, 92544 KB, 30633 KB/s, 3 seconds passed
... 73%, 92576 KB, 30641 KB/s, 3 seconds passed
... 73%, 92608 KB, 30648 KB/s, 3 seconds passed
... 73%, 92640 KB, 30656 KB/s, 3 seconds passed
... 73%, 92672 KB, 30663 KB/s, 3 seconds passed
... 73%, 92704 KB, 30671 KB/s, 3 seconds passed
... 73%, 92736 KB, 30678 KB/s, 3 seconds passed
... 73%, 92768 KB, 30685 KB/s, 3 seconds passed
... 73%, 92800 KB, 30693 KB/s, 3 seconds passed
... 73%, 92832 KB, 30700 KB/s, 3 seconds passed
... 73%, 92864 KB, 30708 KB/s, 3 seconds passed
... 73%, 92896 KB, 30715 KB/s, 3 seconds passed
... 73%, 92928 KB, 30723 KB/s, 3 seconds passed
... 73%, 92960 KB, 30730 KB/s, 3 seconds passed
... 73%, 92992 KB, 30738 KB/s, 3 seconds passed
... 73%, 93024 KB, 30746 KB/s, 3 seconds passed
... 73%, 93056 KB, 30753 KB/s, 3 seconds passed
... 73%, 93088 KB, 30761 KB/s, 3 seconds passed
... 73%, 93120 KB, 30769 KB/s, 3 seconds passed
... 73%, 93152 KB, 30776 KB/s, 3 seconds passed
... 73%, 93184 KB, 30784 KB/s, 3 seconds passed
... 74%, 93216 KB, 30792 KB/s, 3 seconds passed
... 74%, 93248 KB, 30800 KB/s, 3 seconds passed
... 74%, 93280 KB, 30807 KB/s, 3 seconds passed
... 74%, 93312 KB, 30815 KB/s, 3 seconds passed
... 74%, 93344 KB, 30822 KB/s, 3 seconds passed
... 74%, 93376 KB, 30830 KB/s, 3 seconds passed
... 74%, 93408 KB, 30838 KB/s, 3 seconds passed
... 74%, 93440 KB, 30845 KB/s, 3 seconds passed
... 74%, 93472 KB, 30853 KB/s, 3 seconds passed
... 74%, 93504 KB, 30860 KB/s, 3 seconds passed
... 74%, 93536 KB, 30868 KB/s, 3 seconds passed
... 74%, 93568 KB, 30876 KB/s, 3 seconds passed
... 74%, 93600 KB, 30884 KB/s, 3 seconds passed
... 74%, 93632 KB, 30891 KB/s, 3 seconds passed
... 74%, 93664 KB, 30896 KB/s, 3 seconds passed

.. parsed-literal::

    ... 74%, 93696 KB, 30778 KB/s, 3 seconds passed
... 74%, 93728 KB, 30785 KB/s, 3 seconds passed
... 74%, 93760 KB, 30793 KB/s, 3 seconds passed
... 74%, 93792 KB, 30800 KB/s, 3 seconds passed
... 74%, 93824 KB, 30807 KB/s, 3 seconds passed
... 74%, 93856 KB, 30805 KB/s, 3 seconds passed
... 74%, 93888 KB, 30812 KB/s, 3 seconds passed
... 74%, 93920 KB, 30819 KB/s, 3 seconds passed
... 74%, 93952 KB, 30826 KB/s, 3 seconds passed
... 74%, 93984 KB, 30833 KB/s, 3 seconds passed
... 74%, 94016 KB, 30841 KB/s, 3 seconds passed
... 74%, 94048 KB, 30848 KB/s, 3 seconds passed
... 74%, 94080 KB, 30855 KB/s, 3 seconds passed
... 74%, 94112 KB, 30863 KB/s, 3 seconds passed
... 74%, 94144 KB, 30870 KB/s, 3 seconds passed
... 74%, 94176 KB, 30876 KB/s, 3 seconds passed
... 74%, 94208 KB, 30883 KB/s, 3 seconds passed
... 74%, 94240 KB, 30890 KB/s, 3 seconds passed
... 74%, 94272 KB, 30897 KB/s, 3 seconds passed
... 74%, 94304 KB, 30904 KB/s, 3 seconds passed
... 74%, 94336 KB, 30864 KB/s, 3 seconds passed
... 74%, 94368 KB, 30870 KB/s, 3 seconds passed
... 74%, 94400 KB, 30878 KB/s, 3 seconds passed
... 74%, 94432 KB, 30885 KB/s, 3 seconds passed
... 74%, 94464 KB, 30892 KB/s, 3 seconds passed
... 75%, 94496 KB, 30899 KB/s, 3 seconds passed
... 75%, 94528 KB, 30906 KB/s, 3 seconds passed
... 75%, 94560 KB, 30913 KB/s, 3 seconds passed
... 75%, 94592 KB, 30920 KB/s, 3 seconds passed
... 75%, 94624 KB, 30928 KB/s, 3 seconds passed
... 75%, 94656 KB, 30935 KB/s, 3 seconds passed
... 75%, 94688 KB, 30942 KB/s, 3 seconds passed
... 75%, 94720 KB, 30950 KB/s, 3 seconds passed
... 75%, 94752 KB, 30957 KB/s, 3 seconds passed
... 75%, 94784 KB, 30965 KB/s, 3 seconds passed
... 75%, 94816 KB, 30972 KB/s, 3 seconds passed
... 75%, 94848 KB, 30980 KB/s, 3 seconds passed
... 75%, 94880 KB, 30987 KB/s, 3 seconds passed
... 75%, 94912 KB, 30994 KB/s, 3 seconds passed
... 75%, 94944 KB, 31002 KB/s, 3 seconds passed
... 75%, 94976 KB, 31009 KB/s, 3 seconds passed
... 75%, 95008 KB, 31017 KB/s, 3 seconds passed
... 75%, 95040 KB, 31024 KB/s, 3 seconds passed
... 75%, 95072 KB, 31031 KB/s, 3 seconds passed
... 75%, 95104 KB, 31039 KB/s, 3 seconds passed
... 75%, 95136 KB, 31046 KB/s, 3 seconds passed
... 75%, 95168 KB, 31054 KB/s, 3 seconds passed
... 75%, 95200 KB, 31061 KB/s, 3 seconds passed
... 75%, 95232 KB, 31069 KB/s, 3 seconds passed
... 75%, 95264 KB, 31077 KB/s, 3 seconds passed
... 75%, 95296 KB, 31085 KB/s, 3 seconds passed
... 75%, 95328 KB, 31092 KB/s, 3 seconds passed
... 75%, 95360 KB, 31100 KB/s, 3 seconds passed
... 75%, 95392 KB, 31108 KB/s, 3 seconds passed
... 75%, 95424 KB, 31116 KB/s, 3 seconds passed
... 75%, 95456 KB, 31124 KB/s, 3 seconds passed
... 75%, 95488 KB, 31132 KB/s, 3 seconds passed
... 75%, 95520 KB, 31140 KB/s, 3 seconds passed
... 75%, 95552 KB, 31148 KB/s, 3 seconds passed
... 75%, 95584 KB, 31156 KB/s, 3 seconds passed
... 75%, 95616 KB, 31163 KB/s, 3 seconds passed
... 75%, 95648 KB, 31171 KB/s, 3 seconds passed
... 75%, 95680 KB, 31179 KB/s, 3 seconds passed
... 75%, 95712 KB, 31186 KB/s, 3 seconds passed
... 76%, 95744 KB, 31194 KB/s, 3 seconds passed
... 76%, 95776 KB, 31201 KB/s, 3 seconds passed
... 76%, 95808 KB, 31209 KB/s, 3 seconds passed
... 76%, 95840 KB, 31217 KB/s, 3 seconds passed
... 76%, 95872 KB, 31225 KB/s, 3 seconds passed
... 76%, 95904 KB, 31232 KB/s, 3 seconds passed
... 76%, 95936 KB, 31240 KB/s, 3 seconds passed
... 76%, 95968 KB, 31247 KB/s, 3 seconds passed
... 76%, 96000 KB, 31255 KB/s, 3 seconds passed
... 76%, 96032 KB, 31263 KB/s, 3 seconds passed
... 76%, 96064 KB, 31270 KB/s, 3 seconds passed
... 76%, 96096 KB, 31277 KB/s, 3 seconds passed
... 76%, 96128 KB, 31285 KB/s, 3 seconds passed
... 76%, 96160 KB, 31293 KB/s, 3 seconds passed
... 76%, 96192 KB, 31301 KB/s, 3 seconds passed
... 76%, 96224 KB, 31308 KB/s, 3 seconds passed
... 76%, 96256 KB, 31315 KB/s, 3 seconds passed
... 76%, 96288 KB, 31323 KB/s, 3 seconds passed
... 76%, 96320 KB, 31331 KB/s, 3 seconds passed
... 76%, 96352 KB, 31338 KB/s, 3 seconds passed
... 76%, 96384 KB, 31346 KB/s, 3 seconds passed
... 76%, 96416 KB, 31353 KB/s, 3 seconds passed

.. parsed-literal::

    ... 76%, 96448 KB, 30307 KB/s, 3 seconds passed
... 76%, 96480 KB, 30312 KB/s, 3 seconds passed
... 76%, 96512 KB, 30319 KB/s, 3 seconds passed
... 76%, 96544 KB, 30326 KB/s, 3 seconds passed
... 76%, 96576 KB, 30333 KB/s, 3 seconds passed
... 76%, 96608 KB, 30340 KB/s, 3 seconds passed
... 76%, 96640 KB, 30347 KB/s, 3 seconds passed
... 76%, 96672 KB, 30354 KB/s, 3 seconds passed
... 76%, 96704 KB, 30361 KB/s, 3 seconds passed
... 76%, 96736 KB, 30368 KB/s, 3 seconds passed
... 76%, 96768 KB, 30375 KB/s, 3 seconds passed
... 76%, 96800 KB, 30383 KB/s, 3 seconds passed
... 76%, 96832 KB, 30390 KB/s, 3 seconds passed
... 76%, 96864 KB, 30397 KB/s, 3 seconds passed
... 76%, 96896 KB, 30404 KB/s, 3 seconds passed
... 76%, 96928 KB, 30411 KB/s, 3 seconds passed
... 76%, 96960 KB, 30418 KB/s, 3 seconds passed
... 77%, 96992 KB, 30425 KB/s, 3 seconds passed
... 77%, 97024 KB, 30433 KB/s, 3 seconds passed
... 77%, 97056 KB, 30438 KB/s, 3 seconds passed
... 77%, 97088 KB, 30445 KB/s, 3 seconds passed
... 77%, 97120 KB, 30452 KB/s, 3 seconds passed
... 77%, 97152 KB, 30460 KB/s, 3 seconds passed
... 77%, 97184 KB, 30467 KB/s, 3 seconds passed
... 77%, 97216 KB, 30474 KB/s, 3 seconds passed
... 77%, 97248 KB, 30481 KB/s, 3 seconds passed

.. parsed-literal::

    ... 77%, 97280 KB, 29920 KB/s, 3 seconds passed
... 77%, 97312 KB, 29923 KB/s, 3 seconds passed
... 77%, 97344 KB, 29930 KB/s, 3 seconds passed
... 77%, 97376 KB, 29936 KB/s, 3 seconds passed
... 77%, 97408 KB, 29928 KB/s, 3 seconds passed
... 77%, 97440 KB, 29935 KB/s, 3 seconds passed
... 77%, 97472 KB, 29942 KB/s, 3 seconds passed
... 77%, 97504 KB, 29950 KB/s, 3 seconds passed
... 77%, 97536 KB, 29956 KB/s, 3 seconds passed
... 77%, 97568 KB, 29964 KB/s, 3 seconds passed
... 77%, 97600 KB, 29971 KB/s, 3 seconds passed
... 77%, 97632 KB, 29978 KB/s, 3 seconds passed
... 77%, 97664 KB, 29984 KB/s, 3 seconds passed
... 77%, 97696 KB, 29991 KB/s, 3 seconds passed
... 77%, 97728 KB, 29999 KB/s, 3 seconds passed
... 77%, 97760 KB, 30006 KB/s, 3 seconds passed
... 77%, 97792 KB, 30013 KB/s, 3 seconds passed
... 77%, 97824 KB, 30020 KB/s, 3 seconds passed
... 77%, 97856 KB, 30027 KB/s, 3 seconds passed
... 77%, 97888 KB, 30033 KB/s, 3 seconds passed
... 77%, 97920 KB, 30040 KB/s, 3 seconds passed
... 77%, 97952 KB, 30047 KB/s, 3 seconds passed
... 77%, 97984 KB, 30054 KB/s, 3 seconds passed
... 77%, 98016 KB, 30061 KB/s, 3 seconds passed
... 77%, 98048 KB, 30069 KB/s, 3 seconds passed
... 77%, 98080 KB, 30076 KB/s, 3 seconds passed
... 77%, 98112 KB, 30082 KB/s, 3 seconds passed
... 77%, 98144 KB, 30089 KB/s, 3 seconds passed
... 77%, 98176 KB, 30096 KB/s, 3 seconds passed
... 77%, 98208 KB, 30103 KB/s, 3 seconds passed
... 77%, 98240 KB, 30110 KB/s, 3 seconds passed
... 78%, 98272 KB, 30117 KB/s, 3 seconds passed
... 78%, 98304 KB, 30100 KB/s, 3 seconds passed
... 78%, 98336 KB, 30105 KB/s, 3 seconds passed
... 78%, 98368 KB, 30111 KB/s, 3 seconds passed
... 78%, 98400 KB, 30118 KB/s, 3 seconds passed
... 78%, 98432 KB, 30125 KB/s, 3 seconds passed
... 78%, 98464 KB, 30132 KB/s, 3 seconds passed
... 78%, 98496 KB, 30140 KB/s, 3 seconds passed
... 78%, 98528 KB, 30147 KB/s, 3 seconds passed
... 78%, 98560 KB, 30154 KB/s, 3 seconds passed
... 78%, 98592 KB, 30161 KB/s, 3 seconds passed
... 78%, 98624 KB, 30168 KB/s, 3 seconds passed
... 78%, 98656 KB, 30176 KB/s, 3 seconds passed
... 78%, 98688 KB, 30183 KB/s, 3 seconds passed
... 78%, 98720 KB, 30190 KB/s, 3 seconds passed
... 78%, 98752 KB, 30198 KB/s, 3 seconds passed
... 78%, 98784 KB, 30205 KB/s, 3 seconds passed
... 78%, 98816 KB, 30212 KB/s, 3 seconds passed
... 78%, 98848 KB, 30219 KB/s, 3 seconds passed
... 78%, 98880 KB, 30227 KB/s, 3 seconds passed
... 78%, 98912 KB, 30234 KB/s, 3 seconds passed
... 78%, 98944 KB, 30241 KB/s, 3 seconds passed
... 78%, 98976 KB, 30249 KB/s, 3 seconds passed
... 78%, 99008 KB, 30256 KB/s, 3 seconds passed
... 78%, 99040 KB, 30263 KB/s, 3 seconds passed
... 78%, 99072 KB, 30271 KB/s, 3 seconds passed
... 78%, 99104 KB, 30278 KB/s, 3 seconds passed
... 78%, 99136 KB, 30285 KB/s, 3 seconds passed
... 78%, 99168 KB, 30293 KB/s, 3 seconds passed
... 78%, 99200 KB, 30300 KB/s, 3 seconds passed
... 78%, 99232 KB, 30307 KB/s, 3 seconds passed
... 78%, 99264 KB, 30101 KB/s, 3 seconds passed

.. parsed-literal::

    ... 78%, 99296 KB, 30106 KB/s, 3 seconds passed
... 78%, 99328 KB, 30113 KB/s, 3 seconds passed
... 78%, 99360 KB, 30114 KB/s, 3 seconds passed
... 78%, 99392 KB, 30121 KB/s, 3 seconds passed
... 78%, 99424 KB, 30127 KB/s, 3 seconds passed
... 78%, 99456 KB, 30134 KB/s, 3 seconds passed
... 78%, 99488 KB, 30141 KB/s, 3 seconds passed
... 79%, 99520 KB, 30148 KB/s, 3 seconds passed
... 79%, 99552 KB, 30155 KB/s, 3 seconds passed
... 79%, 99584 KB, 30162 KB/s, 3 seconds passed
... 79%, 99616 KB, 30169 KB/s, 3 seconds passed
... 79%, 99648 KB, 30176 KB/s, 3 seconds passed
... 79%, 99680 KB, 30182 KB/s, 3 seconds passed
... 79%, 99712 KB, 30189 KB/s, 3 seconds passed
... 79%, 99744 KB, 30196 KB/s, 3 seconds passed
... 79%, 99776 KB, 30203 KB/s, 3 seconds passed
... 79%, 99808 KB, 30210 KB/s, 3 seconds passed
... 79%, 99840 KB, 30216 KB/s, 3 seconds passed
... 79%, 99872 KB, 30223 KB/s, 3 seconds passed
... 79%, 99904 KB, 30230 KB/s, 3 seconds passed
... 79%, 99936 KB, 30237 KB/s, 3 seconds passed
... 79%, 99968 KB, 30244 KB/s, 3 seconds passed
... 79%, 100000 KB, 30251 KB/s, 3 seconds passed
... 79%, 100032 KB, 30258 KB/s, 3 seconds passed
... 79%, 100064 KB, 30265 KB/s, 3 seconds passed
... 79%, 100096 KB, 30271 KB/s, 3 seconds passed
... 79%, 100128 KB, 30278 KB/s, 3 seconds passed
... 79%, 100160 KB, 30285 KB/s, 3 seconds passed
... 79%, 100192 KB, 30292 KB/s, 3 seconds passed
... 79%, 100224 KB, 30299 KB/s, 3 seconds passed
... 79%, 100256 KB, 30306 KB/s, 3 seconds passed
... 79%, 100288 KB, 30313 KB/s, 3 seconds passed
... 79%, 100320 KB, 30321 KB/s, 3 seconds passed
... 79%, 100352 KB, 30328 KB/s, 3 seconds passed
... 79%, 100384 KB, 30336 KB/s, 3 seconds passed
... 79%, 100416 KB, 30343 KB/s, 3 seconds passed
... 79%, 100448 KB, 30351 KB/s, 3 seconds passed
... 79%, 100480 KB, 30358 KB/s, 3 seconds passed
... 79%, 100512 KB, 30366 KB/s, 3 seconds passed
... 79%, 100544 KB, 30373 KB/s, 3 seconds passed
... 79%, 100576 KB, 30381 KB/s, 3 seconds passed
... 79%, 100608 KB, 30388 KB/s, 3 seconds passed
... 79%, 100640 KB, 30396 KB/s, 3 seconds passed
... 79%, 100672 KB, 30403 KB/s, 3 seconds passed
... 79%, 100704 KB, 30411 KB/s, 3 seconds passed
... 79%, 100736 KB, 30418 KB/s, 3 seconds passed
... 80%, 100768 KB, 30425 KB/s, 3 seconds passed
... 80%, 100800 KB, 30433 KB/s, 3 seconds passed
... 80%, 100832 KB, 30440 KB/s, 3 seconds passed
... 80%, 100864 KB, 30448 KB/s, 3 seconds passed
... 80%, 100896 KB, 30455 KB/s, 3 seconds passed
... 80%, 100928 KB, 30463 KB/s, 3 seconds passed
... 80%, 100960 KB, 30470 KB/s, 3 seconds passed
... 80%, 100992 KB, 30478 KB/s, 3 seconds passed
... 80%, 101024 KB, 30485 KB/s, 3 seconds passed
... 80%, 101056 KB, 30492 KB/s, 3 seconds passed
... 80%, 101088 KB, 30499 KB/s, 3 seconds passed
... 80%, 101120 KB, 30507 KB/s, 3 seconds passed
... 80%, 101152 KB, 30513 KB/s, 3 seconds passed
... 80%, 101184 KB, 30521 KB/s, 3 seconds passed
... 80%, 101216 KB, 30528 KB/s, 3 seconds passed
... 80%, 101248 KB, 30535 KB/s, 3 seconds passed
... 80%, 101280 KB, 30542 KB/s, 3 seconds passed
... 80%, 101312 KB, 30549 KB/s, 3 seconds passed
... 80%, 101344 KB, 30556 KB/s, 3 seconds passed
... 80%, 101376 KB, 30563 KB/s, 3 seconds passed
... 80%, 101408 KB, 30570 KB/s, 3 seconds passed
... 80%, 101440 KB, 30577 KB/s, 3 seconds passed
... 80%, 101472 KB, 30584 KB/s, 3 seconds passed
... 80%, 101504 KB, 30592 KB/s, 3 seconds passed
... 80%, 101536 KB, 30599 KB/s, 3 seconds passed
... 80%, 101568 KB, 30412 KB/s, 3 seconds passed
... 80%, 101600 KB, 30416 KB/s, 3 seconds passed
... 80%, 101632 KB, 30422 KB/s, 3 seconds passed
... 80%, 101664 KB, 30428 KB/s, 3 seconds passed
... 80%, 101696 KB, 30435 KB/s, 3 seconds passed
... 80%, 101728 KB, 30442 KB/s, 3 seconds passed
... 80%, 101760 KB, 30449 KB/s, 3 seconds passed
... 80%, 101792 KB, 30456 KB/s, 3 seconds passed
... 80%, 101824 KB, 30462 KB/s, 3 seconds passed
... 80%, 101856 KB, 30469 KB/s, 3 seconds passed
... 80%, 101888 KB, 30476 KB/s, 3 seconds passed
... 80%, 101920 KB, 30483 KB/s, 3 seconds passed
... 80%, 101952 KB, 30490 KB/s, 3 seconds passed
... 80%, 101984 KB, 30496 KB/s, 3 seconds passed
... 80%, 102016 KB, 30503 KB/s, 3 seconds passed
... 81%, 102048 KB, 30510 KB/s, 3 seconds passed
... 81%, 102080 KB, 30517 KB/s, 3 seconds passed
... 81%, 102112 KB, 30524 KB/s, 3 seconds passed
... 81%, 102144 KB, 30530 KB/s, 3 seconds passed
... 81%, 102176 KB, 30537 KB/s, 3 seconds passed
... 81%, 102208 KB, 30544 KB/s, 3 seconds passed
... 81%, 102240 KB, 30551 KB/s, 3 seconds passed
... 81%, 102272 KB, 30558 KB/s, 3 seconds passed
... 81%, 102304 KB, 30564 KB/s, 3 seconds passed
... 81%, 102336 KB, 30571 KB/s, 3 seconds passed
... 81%, 102368 KB, 30578 KB/s, 3 seconds passed

.. parsed-literal::

    ... 81%, 102400 KB, 29514 KB/s, 3 seconds passed
... 81%, 102432 KB, 29498 KB/s, 3 seconds passed
... 81%, 102464 KB, 29503 KB/s, 3 seconds passed
... 81%, 102496 KB, 29509 KB/s, 3 seconds passed
... 81%, 102528 KB, 29516 KB/s, 3 seconds passed
... 81%, 102560 KB, 29523 KB/s, 3 seconds passed
... 81%, 102592 KB, 29529 KB/s, 3 seconds passed
... 81%, 102624 KB, 29536 KB/s, 3 seconds passed
... 81%, 102656 KB, 29542 KB/s, 3 seconds passed
... 81%, 102688 KB, 29549 KB/s, 3 seconds passed
... 81%, 102720 KB, 29555 KB/s, 3 seconds passed
... 81%, 102752 KB, 29562 KB/s, 3 seconds passed
... 81%, 102784 KB, 29568 KB/s, 3 seconds passed
... 81%, 102816 KB, 29575 KB/s, 3 seconds passed
... 81%, 102848 KB, 29572 KB/s, 3 seconds passed
... 81%, 102880 KB, 29578 KB/s, 3 seconds passed
... 81%, 102912 KB, 29579 KB/s, 3 seconds passed
... 81%, 102944 KB, 29585 KB/s, 3 seconds passed
... 81%, 102976 KB, 29592 KB/s, 3 seconds passed
... 81%, 103008 KB, 29598 KB/s, 3 seconds passed
... 81%, 103040 KB, 29605 KB/s, 3 seconds passed
... 81%, 103072 KB, 29611 KB/s, 3 seconds passed
... 81%, 103104 KB, 29618 KB/s, 3 seconds passed
... 81%, 103136 KB, 29625 KB/s, 3 seconds passed
... 81%, 103168 KB, 29631 KB/s, 3 seconds passed
... 81%, 103200 KB, 29638 KB/s, 3 seconds passed
... 81%, 103232 KB, 29644 KB/s, 3 seconds passed
... 81%, 103264 KB, 29651 KB/s, 3 seconds passed
... 82%, 103296 KB, 29657 KB/s, 3 seconds passed
... 82%, 103328 KB, 29664 KB/s, 3 seconds passed
... 82%, 103360 KB, 29670 KB/s, 3 seconds passed
... 82%, 103392 KB, 29677 KB/s, 3 seconds passed
... 82%, 103424 KB, 29683 KB/s, 3 seconds passed
... 82%, 103456 KB, 29690 KB/s, 3 seconds passed
... 82%, 103488 KB, 29697 KB/s, 3 seconds passed
... 82%, 103520 KB, 29703 KB/s, 3 seconds passed
... 82%, 103552 KB, 29710 KB/s, 3 seconds passed
... 82%, 103584 KB, 29716 KB/s, 3 seconds passed
... 82%, 103616 KB, 29723 KB/s, 3 seconds passed
... 82%, 103648 KB, 29730 KB/s, 3 seconds passed
... 82%, 103680 KB, 29736 KB/s, 3 seconds passed
... 82%, 103712 KB, 29743 KB/s, 3 seconds passed
... 82%, 103744 KB, 29749 KB/s, 3 seconds passed
... 82%, 103776 KB, 29756 KB/s, 3 seconds passed
... 82%, 103808 KB, 29763 KB/s, 3 seconds passed
... 82%, 103840 KB, 29769 KB/s, 3 seconds passed
... 82%, 103872 KB, 29776 KB/s, 3 seconds passed
... 82%, 103904 KB, 29782 KB/s, 3 seconds passed
... 82%, 103936 KB, 29789 KB/s, 3 seconds passed
... 82%, 103968 KB, 29795 KB/s, 3 seconds passed
... 82%, 104000 KB, 29802 KB/s, 3 seconds passed
... 82%, 104032 KB, 29809 KB/s, 3 seconds passed
... 82%, 104064 KB, 29817 KB/s, 3 seconds passed
... 82%, 104096 KB, 29824 KB/s, 3 seconds passed
... 82%, 104128 KB, 29831 KB/s, 3 seconds passed
... 82%, 104160 KB, 29838 KB/s, 3 seconds passed
... 82%, 104192 KB, 29846 KB/s, 3 seconds passed
... 82%, 104224 KB, 29853 KB/s, 3 seconds passed
... 82%, 104256 KB, 29860 KB/s, 3 seconds passed
... 82%, 104288 KB, 29867 KB/s, 3 seconds passed
... 82%, 104320 KB, 29874 KB/s, 3 seconds passed
... 82%, 104352 KB, 29882 KB/s, 3 seconds passed
... 82%, 104384 KB, 29889 KB/s, 3 seconds passed
... 82%, 104416 KB, 29896 KB/s, 3 seconds passed
... 82%, 104448 KB, 29903 KB/s, 3 seconds passed
... 82%, 104480 KB, 29910 KB/s, 3 seconds passed
... 82%, 104512 KB, 29918 KB/s, 3 seconds passed
... 83%, 104544 KB, 29925 KB/s, 3 seconds passed
... 83%, 104576 KB, 29932 KB/s, 3 seconds passed
... 83%, 104608 KB, 29939 KB/s, 3 seconds passed
... 83%, 104640 KB, 29946 KB/s, 3 seconds passed
... 83%, 104672 KB, 29954 KB/s, 3 seconds passed
... 83%, 104704 KB, 29960 KB/s, 3 seconds passed
... 83%, 104736 KB, 29967 KB/s, 3 seconds passed
... 83%, 104768 KB, 29974 KB/s, 3 seconds passed
... 83%, 104800 KB, 29981 KB/s, 3 seconds passed
... 83%, 104832 KB, 29988 KB/s, 3 seconds passed
... 83%, 104864 KB, 29994 KB/s, 3 seconds passed
... 83%, 104896 KB, 30001 KB/s, 3 seconds passed
... 83%, 104928 KB, 30008 KB/s, 3 seconds passed
... 83%, 104960 KB, 30015 KB/s, 3 seconds passed
... 83%, 104992 KB, 30021 KB/s, 3 seconds passed
... 83%, 105024 KB, 30028 KB/s, 3 seconds passed
... 83%, 105056 KB, 30035 KB/s, 3 seconds passed

.. parsed-literal::

    ... 83%, 105088 KB, 29882 KB/s, 3 seconds passed
... 83%, 105120 KB, 29886 KB/s, 3 seconds passed
... 83%, 105152 KB, 29891 KB/s, 3 seconds passed
... 83%, 105184 KB, 29895 KB/s, 3 seconds passed
... 83%, 105216 KB, 29901 KB/s, 3 seconds passed
... 83%, 105248 KB, 29902 KB/s, 3 seconds passed
... 83%, 105280 KB, 29878 KB/s, 3 seconds passed
... 83%, 105312 KB, 29884 KB/s, 3 seconds passed
... 83%, 105344 KB, 29891 KB/s, 3 seconds passed
... 83%, 105376 KB, 29897 KB/s, 3 seconds passed
... 83%, 105408 KB, 29904 KB/s, 3 seconds passed
... 83%, 105440 KB, 29910 KB/s, 3 seconds passed
... 83%, 105472 KB, 29917 KB/s, 3 seconds passed
... 83%, 105504 KB, 29923 KB/s, 3 seconds passed
... 83%, 105536 KB, 29930 KB/s, 3 seconds passed
... 83%, 105568 KB, 29936 KB/s, 3 seconds passed
... 83%, 105600 KB, 29943 KB/s, 3 seconds passed
... 83%, 105632 KB, 29949 KB/s, 3 seconds passed
... 83%, 105664 KB, 29956 KB/s, 3 seconds passed
... 83%, 105696 KB, 29962 KB/s, 3 seconds passed
... 83%, 105728 KB, 29969 KB/s, 3 seconds passed
... 83%, 105760 KB, 29975 KB/s, 3 seconds passed
... 83%, 105792 KB, 29982 KB/s, 3 seconds passed
... 84%, 105824 KB, 29988 KB/s, 3 seconds passed
... 84%, 105856 KB, 29995 KB/s, 3 seconds passed
... 84%, 105888 KB, 30001 KB/s, 3 seconds passed
... 84%, 105920 KB, 30008 KB/s, 3 seconds passed
... 84%, 105952 KB, 30014 KB/s, 3 seconds passed
... 84%, 105984 KB, 30021 KB/s, 3 seconds passed
... 84%, 106016 KB, 30027 KB/s, 3 seconds passed
... 84%, 106048 KB, 30034 KB/s, 3 seconds passed
... 84%, 106080 KB, 30040 KB/s, 3 seconds passed
... 84%, 106112 KB, 30047 KB/s, 3 seconds passed
... 84%, 106144 KB, 30053 KB/s, 3 seconds passed
... 84%, 106176 KB, 30060 KB/s, 3 seconds passed
... 84%, 106208 KB, 30066 KB/s, 3 seconds passed
... 84%, 106240 KB, 30073 KB/s, 3 seconds passed
... 84%, 106272 KB, 30079 KB/s, 3 seconds passed
... 84%, 106304 KB, 30086 KB/s, 3 seconds passed
... 84%, 106336 KB, 30092 KB/s, 3 seconds passed
... 84%, 106368 KB, 30099 KB/s, 3 seconds passed
... 84%, 106400 KB, 30105 KB/s, 3 seconds passed
... 84%, 106432 KB, 30112 KB/s, 3 seconds passed
... 84%, 106464 KB, 30119 KB/s, 3 seconds passed
... 84%, 106496 KB, 30125 KB/s, 3 seconds passed
... 84%, 106528 KB, 30131 KB/s, 3 seconds passed
... 84%, 106560 KB, 30138 KB/s, 3 seconds passed
... 84%, 106592 KB, 30144 KB/s, 3 seconds passed
... 84%, 106624 KB, 30151 KB/s, 3 seconds passed
... 84%, 106656 KB, 30157 KB/s, 3 seconds passed
... 84%, 106688 KB, 30163 KB/s, 3 seconds passed
... 84%, 106720 KB, 30170 KB/s, 3 seconds passed
... 84%, 106752 KB, 30176 KB/s, 3 seconds passed
... 84%, 106784 KB, 30183 KB/s, 3 seconds passed
... 84%, 106816 KB, 30189 KB/s, 3 seconds passed
... 84%, 106848 KB, 30196 KB/s, 3 seconds passed
... 84%, 106880 KB, 30202 KB/s, 3 seconds passed
... 84%, 106912 KB, 30209 KB/s, 3 seconds passed
... 84%, 106944 KB, 30215 KB/s, 3 seconds passed
... 84%, 106976 KB, 30222 KB/s, 3 seconds passed
... 84%, 107008 KB, 30229 KB/s, 3 seconds passed
... 84%, 107040 KB, 30236 KB/s, 3 seconds passed
... 85%, 107072 KB, 30243 KB/s, 3 seconds passed
... 85%, 107104 KB, 30251 KB/s, 3 seconds passed
... 85%, 107136 KB, 30258 KB/s, 3 seconds passed
... 85%, 107168 KB, 30265 KB/s, 3 seconds passed
... 85%, 107200 KB, 30272 KB/s, 3 seconds passed
... 85%, 107232 KB, 30280 KB/s, 3 seconds passed
... 85%, 107264 KB, 30287 KB/s, 3 seconds passed
... 85%, 107296 KB, 30294 KB/s, 3 seconds passed
... 85%, 107328 KB, 30301 KB/s, 3 seconds passed
... 85%, 107360 KB, 30309 KB/s, 3 seconds passed
... 85%, 107392 KB, 30316 KB/s, 3 seconds passed
... 85%, 107424 KB, 30323 KB/s, 3 seconds passed
... 85%, 107456 KB, 30331 KB/s, 3 seconds passed
... 85%, 107488 KB, 30338 KB/s, 3 seconds passed

.. parsed-literal::

    ... 85%, 107520 KB, 30073 KB/s, 3 seconds passed
... 85%, 107552 KB, 30078 KB/s, 3 seconds passed
... 85%, 107584 KB, 30084 KB/s, 3 seconds passed
... 85%, 107616 KB, 30090 KB/s, 3 seconds passed
... 85%, 107648 KB, 30096 KB/s, 3 seconds passed
... 85%, 107680 KB, 30103 KB/s, 3 seconds passed
... 85%, 107712 KB, 30109 KB/s, 3 seconds passed
... 85%, 107744 KB, 30116 KB/s, 3 seconds passed
... 85%, 107776 KB, 30122 KB/s, 3 seconds passed
... 85%, 107808 KB, 30128 KB/s, 3 seconds passed
... 85%, 107840 KB, 30135 KB/s, 3 seconds passed
... 85%, 107872 KB, 30109 KB/s, 3 seconds passed
... 85%, 107904 KB, 30114 KB/s, 3 seconds passed
... 85%, 107936 KB, 30120 KB/s, 3 seconds passed
... 85%, 107968 KB, 30126 KB/s, 3 seconds passed
... 85%, 108000 KB, 30133 KB/s, 3 seconds passed
... 85%, 108032 KB, 30139 KB/s, 3 seconds passed
... 85%, 108064 KB, 30145 KB/s, 3 seconds passed
... 85%, 108096 KB, 30152 KB/s, 3 seconds passed
... 85%, 108128 KB, 30158 KB/s, 3 seconds passed
... 85%, 108160 KB, 30164 KB/s, 3 seconds passed
... 85%, 108192 KB, 30171 KB/s, 3 seconds passed
... 85%, 108224 KB, 30177 KB/s, 3 seconds passed
... 85%, 108256 KB, 30184 KB/s, 3 seconds passed
... 85%, 108288 KB, 30190 KB/s, 3 seconds passed
... 86%, 108320 KB, 30196 KB/s, 3 seconds passed
... 86%, 108352 KB, 30203 KB/s, 3 seconds passed
... 86%, 108384 KB, 30209 KB/s, 3 seconds passed
... 86%, 108416 KB, 30215 KB/s, 3 seconds passed
... 86%, 108448 KB, 30222 KB/s, 3 seconds passed
... 86%, 108480 KB, 30228 KB/s, 3 seconds passed
... 86%, 108512 KB, 30234 KB/s, 3 seconds passed
... 86%, 108544 KB, 30241 KB/s, 3 seconds passed
... 86%, 108576 KB, 30247 KB/s, 3 seconds passed
... 86%, 108608 KB, 30253 KB/s, 3 seconds passed
... 86%, 108640 KB, 30260 KB/s, 3 seconds passed
... 86%, 108672 KB, 30266 KB/s, 3 seconds passed
... 86%, 108704 KB, 30272 KB/s, 3 seconds passed
... 86%, 108736 KB, 30279 KB/s, 3 seconds passed
... 86%, 108768 KB, 30285 KB/s, 3 seconds passed
... 86%, 108800 KB, 30291 KB/s, 3 seconds passed
... 86%, 108832 KB, 30297 KB/s, 3 seconds passed
... 86%, 108864 KB, 30304 KB/s, 3 seconds passed
... 86%, 108896 KB, 30310 KB/s, 3 seconds passed
... 86%, 108928 KB, 30316 KB/s, 3 seconds passed
... 86%, 108960 KB, 30323 KB/s, 3 seconds passed
... 86%, 108992 KB, 30329 KB/s, 3 seconds passed
... 86%, 109024 KB, 30335 KB/s, 3 seconds passed
... 86%, 109056 KB, 30342 KB/s, 3 seconds passed
... 86%, 109088 KB, 30348 KB/s, 3 seconds passed
... 86%, 109120 KB, 30354 KB/s, 3 seconds passed
... 86%, 109152 KB, 30361 KB/s, 3 seconds passed
... 86%, 109184 KB, 30367 KB/s, 3 seconds passed
... 86%, 109216 KB, 30373 KB/s, 3 seconds passed
... 86%, 109248 KB, 30380 KB/s, 3 seconds passed
... 86%, 109280 KB, 30386 KB/s, 3 seconds passed
... 86%, 109312 KB, 30392 KB/s, 3 seconds passed
... 86%, 109344 KB, 30399 KB/s, 3 seconds passed
... 86%, 109376 KB, 30405 KB/s, 3 seconds passed
... 86%, 109408 KB, 30411 KB/s, 3 seconds passed
... 86%, 109440 KB, 30417 KB/s, 3 seconds passed
... 86%, 109472 KB, 30424 KB/s, 3 seconds passed
... 86%, 109504 KB, 30430 KB/s, 3 seconds passed
... 86%, 109536 KB, 30437 KB/s, 3 seconds passed
... 86%, 109568 KB, 30443 KB/s, 3 seconds passed
... 87%, 109600 KB, 30449 KB/s, 3 seconds passed
... 87%, 109632 KB, 30456 KB/s, 3 seconds passed
... 87%, 109664 KB, 30462 KB/s, 3 seconds passed
... 87%, 109696 KB, 30468 KB/s, 3 seconds passed
... 87%, 109728 KB, 30475 KB/s, 3 seconds passed
... 87%, 109760 KB, 30482 KB/s, 3 seconds passed
... 87%, 109792 KB, 30489 KB/s, 3 seconds passed
... 87%, 109824 KB, 30497 KB/s, 3 seconds passed
... 87%, 109856 KB, 30504 KB/s, 3 seconds passed
... 87%, 109888 KB, 30511 KB/s, 3 seconds passed
... 87%, 109920 KB, 30518 KB/s, 3 seconds passed
... 87%, 109952 KB, 30525 KB/s, 3 seconds passed
... 87%, 109984 KB, 30533 KB/s, 3 seconds passed
... 87%, 110016 KB, 30540 KB/s, 3 seconds passed
... 87%, 110048 KB, 30547 KB/s, 3 seconds passed
... 87%, 110080 KB, 30554 KB/s, 3 seconds passed
... 87%, 110112 KB, 30561 KB/s, 3 seconds passed
... 87%, 110144 KB, 30569 KB/s, 3 seconds passed
... 87%, 110176 KB, 30576 KB/s, 3 seconds passed
... 87%, 110208 KB, 30583 KB/s, 3 seconds passed
... 87%, 110240 KB, 30590 KB/s, 3 seconds passed
... 87%, 110272 KB, 30597 KB/s, 3 seconds passed
... 87%, 110304 KB, 30604 KB/s, 3 seconds passed
... 87%, 110336 KB, 30611 KB/s, 3 seconds passed
... 87%, 110368 KB, 30618 KB/s, 3 seconds passed
... 87%, 110400 KB, 30625 KB/s, 3 seconds passed
... 87%, 110432 KB, 30632 KB/s, 3 seconds passed
... 87%, 110464 KB, 30638 KB/s, 3 seconds passed

.. parsed-literal::

    ... 87%, 110496 KB, 30644 KB/s, 3 seconds passed
... 87%, 110528 KB, 30651 KB/s, 3 seconds passed
... 87%, 110560 KB, 30657 KB/s, 3 seconds passed
... 87%, 110592 KB, 30664 KB/s, 3 seconds passed
... 87%, 110624 KB, 30671 KB/s, 3 seconds passed
... 87%, 110656 KB, 30677 KB/s, 3 seconds passed
... 87%, 110688 KB, 30683 KB/s, 3 seconds passed
... 87%, 110720 KB, 30690 KB/s, 3 seconds passed
... 87%, 110752 KB, 30696 KB/s, 3 seconds passed
... 87%, 110784 KB, 30703 KB/s, 3 seconds passed
... 87%, 110816 KB, 30709 KB/s, 3 seconds passed
... 88%, 110848 KB, 30716 KB/s, 3 seconds passed
... 88%, 110880 KB, 30723 KB/s, 3 seconds passed
... 88%, 110912 KB, 30729 KB/s, 3 seconds passed
... 88%, 110944 KB, 30735 KB/s, 3 seconds passed
... 88%, 110976 KB, 30741 KB/s, 3 seconds passed
... 88%, 111008 KB, 30747 KB/s, 3 seconds passed
... 88%, 111040 KB, 30754 KB/s, 3 seconds passed
... 88%, 111072 KB, 30761 KB/s, 3 seconds passed
... 88%, 111104 KB, 30768 KB/s, 3 seconds passed
... 88%, 111136 KB, 30774 KB/s, 3 seconds passed
... 88%, 111168 KB, 30781 KB/s, 3 seconds passed
... 88%, 111200 KB, 30786 KB/s, 3 seconds passed
... 88%, 111232 KB, 30790 KB/s, 3 seconds passed
... 88%, 111264 KB, 30795 KB/s, 3 seconds passed
... 88%, 111296 KB, 30800 KB/s, 3 seconds passed
... 88%, 111328 KB, 30806 KB/s, 3 seconds passed
... 88%, 111360 KB, 30811 KB/s, 3 seconds passed
... 88%, 111392 KB, 30816 KB/s, 3 seconds passed
... 88%, 111424 KB, 30822 KB/s, 3 seconds passed
... 88%, 111456 KB, 30827 KB/s, 3 seconds passed
... 88%, 111488 KB, 30832 KB/s, 3 seconds passed
... 88%, 111520 KB, 30837 KB/s, 3 seconds passed
... 88%, 111552 KB, 30843 KB/s, 3 seconds passed
... 88%, 111584 KB, 30848 KB/s, 3 seconds passed
... 88%, 111616 KB, 30854 KB/s, 3 seconds passed
... 88%, 111648 KB, 30860 KB/s, 3 seconds passed
... 88%, 111680 KB, 30866 KB/s, 3 seconds passed
... 88%, 111712 KB, 30872 KB/s, 3 seconds passed
... 88%, 111744 KB, 30879 KB/s, 3 seconds passed
... 88%, 111776 KB, 30885 KB/s, 3 seconds passed
... 88%, 111808 KB, 30891 KB/s, 3 seconds passed
... 88%, 111840 KB, 30896 KB/s, 3 seconds passed
... 88%, 111872 KB, 30902 KB/s, 3 seconds passed
... 88%, 111904 KB, 30907 KB/s, 3 seconds passed
... 88%, 111936 KB, 30913 KB/s, 3 seconds passed
... 88%, 111968 KB, 30918 KB/s, 3 seconds passed
... 88%, 112000 KB, 30924 KB/s, 3 seconds passed
... 88%, 112032 KB, 30930 KB/s, 3 seconds passed
... 88%, 112064 KB, 30936 KB/s, 3 seconds passed
... 88%, 112096 KB, 30941 KB/s, 3 seconds passed
... 89%, 112128 KB, 30943 KB/s, 3 seconds passed
... 89%, 112160 KB, 30948 KB/s, 3 seconds passed
... 89%, 112192 KB, 30953 KB/s, 3 seconds passed
... 89%, 112224 KB, 30958 KB/s, 3 seconds passed
... 89%, 112256 KB, 30964 KB/s, 3 seconds passed
... 89%, 112288 KB, 30969 KB/s, 3 seconds passed
... 89%, 112320 KB, 30975 KB/s, 3 seconds passed
... 89%, 112352 KB, 30980 KB/s, 3 seconds passed
... 89%, 112384 KB, 30986 KB/s, 3 seconds passed
... 89%, 112416 KB, 30991 KB/s, 3 seconds passed
... 89%, 112448 KB, 30997 KB/s, 3 seconds passed
... 89%, 112480 KB, 31002 KB/s, 3 seconds passed
... 89%, 112512 KB, 31009 KB/s, 3 seconds passed
... 89%, 112544 KB, 31015 KB/s, 3 seconds passed
... 89%, 112576 KB, 31021 KB/s, 3 seconds passed
... 89%, 112608 KB, 31028 KB/s, 3 seconds passed

.. parsed-literal::

    ... 89%, 112640 KB, 30335 KB/s, 3 seconds passed
... 89%, 112672 KB, 30331 KB/s, 3 seconds passed
... 89%, 112704 KB, 30336 KB/s, 3 seconds passed
... 89%, 112736 KB, 30342 KB/s, 3 seconds passed
... 89%, 112768 KB, 30266 KB/s, 3 seconds passed
... 89%, 112800 KB, 30232 KB/s, 3 seconds passed
... 89%, 112832 KB, 30237 KB/s, 3 seconds passed
... 89%, 112864 KB, 30242 KB/s, 3 seconds passed
... 89%, 112896 KB, 30248 KB/s, 3 seconds passed
... 89%, 112928 KB, 30253 KB/s, 3 seconds passed
... 89%, 112960 KB, 30258 KB/s, 3 seconds passed
... 89%, 112992 KB, 30264 KB/s, 3 seconds passed
... 89%, 113024 KB, 30270 KB/s, 3 seconds passed
... 89%, 113056 KB, 30275 KB/s, 3 seconds passed
... 89%, 113088 KB, 30280 KB/s, 3 seconds passed
... 89%, 113120 KB, 30286 KB/s, 3 seconds passed
... 89%, 113152 KB, 30291 KB/s, 3 seconds passed
... 89%, 113184 KB, 30296 KB/s, 3 seconds passed
... 89%, 113216 KB, 30302 KB/s, 3 seconds passed
... 89%, 113248 KB, 30307 KB/s, 3 seconds passed
... 89%, 113280 KB, 30312 KB/s, 3 seconds passed
... 89%, 113312 KB, 30318 KB/s, 3 seconds passed
... 89%, 113344 KB, 30323 KB/s, 3 seconds passed
... 90%, 113376 KB, 30329 KB/s, 3 seconds passed
... 90%, 113408 KB, 30334 KB/s, 3 seconds passed
... 90%, 113440 KB, 30340 KB/s, 3 seconds passed
... 90%, 113472 KB, 30346 KB/s, 3 seconds passed
... 90%, 113504 KB, 30351 KB/s, 3 seconds passed
... 90%, 113536 KB, 30356 KB/s, 3 seconds passed
... 90%, 113568 KB, 30362 KB/s, 3 seconds passed
... 90%, 113600 KB, 30367 KB/s, 3 seconds passed
... 90%, 113632 KB, 30373 KB/s, 3 seconds passed
... 90%, 113664 KB, 30378 KB/s, 3 seconds passed
... 90%, 113696 KB, 30384 KB/s, 3 seconds passed
... 90%, 113728 KB, 30390 KB/s, 3 seconds passed
... 90%, 113760 KB, 30397 KB/s, 3 seconds passed
... 90%, 113792 KB, 30403 KB/s, 3 seconds passed
... 90%, 113824 KB, 30410 KB/s, 3 seconds passed
... 90%, 113856 KB, 30417 KB/s, 3 seconds passed
... 90%, 113888 KB, 30423 KB/s, 3 seconds passed
... 90%, 113920 KB, 30430 KB/s, 3 seconds passed
... 90%, 113952 KB, 30436 KB/s, 3 seconds passed
... 90%, 113984 KB, 30442 KB/s, 3 seconds passed
... 90%, 114016 KB, 30449 KB/s, 3 seconds passed
... 90%, 114048 KB, 30455 KB/s, 3 seconds passed
... 90%, 114080 KB, 30462 KB/s, 3 seconds passed
... 90%, 114112 KB, 30468 KB/s, 3 seconds passed
... 90%, 114144 KB, 30475 KB/s, 3 seconds passed
... 90%, 114176 KB, 30482 KB/s, 3 seconds passed
... 90%, 114208 KB, 30488 KB/s, 3 seconds passed
... 90%, 114240 KB, 30495 KB/s, 3 seconds passed
... 90%, 114272 KB, 30501 KB/s, 3 seconds passed
... 90%, 114304 KB, 30508 KB/s, 3 seconds passed
... 90%, 114336 KB, 30515 KB/s, 3 seconds passed
... 90%, 114368 KB, 30521 KB/s, 3 seconds passed
... 90%, 114400 KB, 30528 KB/s, 3 seconds passed
... 90%, 114432 KB, 30534 KB/s, 3 seconds passed
... 90%, 114464 KB, 30541 KB/s, 3 seconds passed
... 90%, 114496 KB, 30547 KB/s, 3 seconds passed
... 90%, 114528 KB, 30554 KB/s, 3 seconds passed
... 90%, 114560 KB, 30561 KB/s, 3 seconds passed
... 90%, 114592 KB, 30567 KB/s, 3 seconds passed
... 91%, 114624 KB, 30574 KB/s, 3 seconds passed
... 91%, 114656 KB, 30580 KB/s, 3 seconds passed
... 91%, 114688 KB, 30587 KB/s, 3 seconds passed
... 91%, 114720 KB, 30593 KB/s, 3 seconds passed
... 91%, 114752 KB, 30600 KB/s, 3 seconds passed
... 91%, 114784 KB, 30607 KB/s, 3 seconds passed
... 91%, 114816 KB, 30614 KB/s, 3 seconds passed
... 91%, 114848 KB, 30621 KB/s, 3 seconds passed
... 91%, 114880 KB, 30628 KB/s, 3 seconds passed
... 91%, 114912 KB, 30635 KB/s, 3 seconds passed

.. parsed-literal::

    ... 91%, 114944 KB, 30477 KB/s, 3 seconds passed
... 91%, 114976 KB, 30482 KB/s, 3 seconds passed
... 91%, 115008 KB, 30487 KB/s, 3 seconds passed
... 91%, 115040 KB, 30493 KB/s, 3 seconds passed
... 91%, 115072 KB, 30499 KB/s, 3 seconds passed
... 91%, 115104 KB, 30505 KB/s, 3 seconds passed
... 91%, 115136 KB, 30511 KB/s, 3 seconds passed
... 91%, 115168 KB, 30517 KB/s, 3 seconds passed
... 91%, 115200 KB, 30523 KB/s, 3 seconds passed
... 91%, 115232 KB, 30529 KB/s, 3 seconds passed
... 91%, 115264 KB, 30535 KB/s, 3 seconds passed
... 91%, 115296 KB, 30541 KB/s, 3 seconds passed
... 91%, 115328 KB, 30547 KB/s, 3 seconds passed
... 91%, 115360 KB, 30553 KB/s, 3 seconds passed
... 91%, 115392 KB, 30559 KB/s, 3 seconds passed
... 91%, 115424 KB, 30565 KB/s, 3 seconds passed
... 91%, 115456 KB, 30571 KB/s, 3 seconds passed
... 91%, 115488 KB, 30577 KB/s, 3 seconds passed
... 91%, 115520 KB, 30583 KB/s, 3 seconds passed
... 91%, 115552 KB, 30589 KB/s, 3 seconds passed
... 91%, 115584 KB, 30595 KB/s, 3 seconds passed
... 91%, 115616 KB, 30601 KB/s, 3 seconds passed
... 91%, 115648 KB, 30607 KB/s, 3 seconds passed
... 91%, 115680 KB, 30613 KB/s, 3 seconds passed
... 91%, 115712 KB, 30619 KB/s, 3 seconds passed
... 91%, 115744 KB, 30625 KB/s, 3 seconds passed
... 91%, 115776 KB, 30631 KB/s, 3 seconds passed
... 91%, 115808 KB, 30636 KB/s, 3 seconds passed
... 91%, 115840 KB, 30642 KB/s, 3 seconds passed
... 91%, 115872 KB, 30648 KB/s, 3 seconds passed
... 92%, 115904 KB, 30654 KB/s, 3 seconds passed
... 92%, 115936 KB, 30660 KB/s, 3 seconds passed
... 92%, 115968 KB, 30666 KB/s, 3 seconds passed
... 92%, 116000 KB, 30673 KB/s, 3 seconds passed
... 92%, 116032 KB, 30680 KB/s, 3 seconds passed
... 92%, 116064 KB, 30686 KB/s, 3 seconds passed
... 92%, 116096 KB, 30693 KB/s, 3 seconds passed
... 92%, 116128 KB, 30700 KB/s, 3 seconds passed
... 92%, 116160 KB, 30707 KB/s, 3 seconds passed
... 92%, 116192 KB, 30713 KB/s, 3 seconds passed
... 92%, 116224 KB, 30720 KB/s, 3 seconds passed
... 92%, 116256 KB, 30727 KB/s, 3 seconds passed
... 92%, 116288 KB, 30733 KB/s, 3 seconds passed
... 92%, 116320 KB, 30740 KB/s, 3 seconds passed
... 92%, 116352 KB, 30747 KB/s, 3 seconds passed
... 92%, 116384 KB, 30753 KB/s, 3 seconds passed
... 92%, 116416 KB, 30760 KB/s, 3 seconds passed
... 92%, 116448 KB, 30767 KB/s, 3 seconds passed
... 92%, 116480 KB, 30773 KB/s, 3 seconds passed
... 92%, 116512 KB, 30780 KB/s, 3 seconds passed
... 92%, 116544 KB, 30787 KB/s, 3 seconds passed
... 92%, 116576 KB, 30793 KB/s, 3 seconds passed
... 92%, 116608 KB, 30800 KB/s, 3 seconds passed
... 92%, 116640 KB, 30806 KB/s, 3 seconds passed
... 92%, 116672 KB, 30812 KB/s, 3 seconds passed
... 92%, 116704 KB, 30818 KB/s, 3 seconds passed
... 92%, 116736 KB, 30824 KB/s, 3 seconds passed
... 92%, 116768 KB, 30831 KB/s, 3 seconds passed
... 92%, 116800 KB, 30837 KB/s, 3 seconds passed
... 92%, 116832 KB, 30843 KB/s, 3 seconds passed
... 92%, 116864 KB, 30849 KB/s, 3 seconds passed
... 92%, 116896 KB, 30855 KB/s, 3 seconds passed

.. parsed-literal::

    ... 92%, 116928 KB, 30330 KB/s, 3 seconds passed
... 92%, 116960 KB, 30335 KB/s, 3 seconds passed
... 92%, 116992 KB, 30340 KB/s, 3 seconds passed
... 92%, 117024 KB, 30345 KB/s, 3 seconds passed
... 92%, 117056 KB, 30350 KB/s, 3 seconds passed
... 92%, 117088 KB, 30350 KB/s, 3 seconds passed
... 92%, 117120 KB, 30355 KB/s, 3 seconds passed
... 93%, 117152 KB, 30360 KB/s, 3 seconds passed
... 93%, 117184 KB, 30365 KB/s, 3 seconds passed
... 93%, 117216 KB, 30370 KB/s, 3 seconds passed
... 93%, 117248 KB, 30375 KB/s, 3 seconds passed
... 93%, 117280 KB, 30381 KB/s, 3 seconds passed
... 93%, 117312 KB, 30386 KB/s, 3 seconds passed
... 93%, 117344 KB, 30391 KB/s, 3 seconds passed
... 93%, 117376 KB, 30396 KB/s, 3 seconds passed

.. parsed-literal::

    ... 93%, 117408 KB, 30402 KB/s, 3 seconds passed
... 93%, 117440 KB, 30407 KB/s, 3 seconds passed
... 93%, 117472 KB, 30412 KB/s, 3 seconds passed
... 93%, 117504 KB, 30417 KB/s, 3 seconds passed
... 93%, 117536 KB, 30423 KB/s, 3 seconds passed
... 93%, 117568 KB, 30428 KB/s, 3 seconds passed
... 93%, 117600 KB, 30434 KB/s, 3 seconds passed
... 93%, 117632 KB, 30440 KB/s, 3 seconds passed
... 93%, 117664 KB, 30446 KB/s, 3 seconds passed
... 93%, 117696 KB, 30452 KB/s, 3 seconds passed
... 93%, 117728 KB, 30458 KB/s, 3 seconds passed

.. parsed-literal::

    ... 93%, 117760 KB, 29848 KB/s, 3 seconds passed
... 93%, 117792 KB, 29852 KB/s, 3 seconds passed
... 93%, 117824 KB, 29857 KB/s, 3 seconds passed
... 93%, 117856 KB, 29862 KB/s, 3 seconds passed

.. parsed-literal::

    ... 93%, 117888 KB, 29718 KB/s, 3 seconds passed
... 93%, 117920 KB, 29722 KB/s, 3 seconds passed
... 93%, 117952 KB, 29727 KB/s, 3 seconds passed
... 93%, 117984 KB, 29732 KB/s, 3 seconds passed
... 93%, 118016 KB, 29737 KB/s, 3 seconds passed
... 93%, 118048 KB, 29742 KB/s, 3 seconds passed
... 93%, 118080 KB, 29748 KB/s, 3 seconds passed
... 93%, 118112 KB, 29754 KB/s, 3 seconds passed
... 93%, 118144 KB, 29759 KB/s, 3 seconds passed
... 93%, 118176 KB, 29757 KB/s, 3 seconds passed
... 93%, 118208 KB, 29761 KB/s, 3 seconds passed
... 93%, 118240 KB, 29766 KB/s, 3 seconds passed
... 93%, 118272 KB, 29771 KB/s, 3 seconds passed
... 93%, 118304 KB, 29777 KB/s, 3 seconds passed
... 93%, 118336 KB, 29782 KB/s, 3 seconds passed
... 93%, 118368 KB, 29787 KB/s, 3 seconds passed
... 94%, 118400 KB, 29792 KB/s, 3 seconds passed
... 94%, 118432 KB, 29797 KB/s, 3 seconds passed
... 94%, 118464 KB, 29803 KB/s, 3 seconds passed
... 94%, 118496 KB, 29808 KB/s, 3 seconds passed
... 94%, 118528 KB, 29814 KB/s, 3 seconds passed
... 94%, 118560 KB, 29820 KB/s, 3 seconds passed
... 94%, 118592 KB, 29782 KB/s, 3 seconds passed
... 94%, 118624 KB, 29787 KB/s, 3 seconds passed
... 94%, 118656 KB, 29792 KB/s, 3 seconds passed
... 94%, 118688 KB, 29797 KB/s, 3 seconds passed
... 94%, 118720 KB, 29802 KB/s, 3 seconds passed
... 94%, 118752 KB, 29807 KB/s, 3 seconds passed
... 94%, 118784 KB, 29812 KB/s, 3 seconds passed
... 94%, 118816 KB, 29817 KB/s, 3 seconds passed
... 94%, 118848 KB, 29822 KB/s, 3 seconds passed
... 94%, 118880 KB, 29827 KB/s, 3 seconds passed
... 94%, 118912 KB, 29832 KB/s, 3 seconds passed
... 94%, 118944 KB, 29837 KB/s, 3 seconds passed
... 94%, 118976 KB, 29843 KB/s, 3 seconds passed
... 94%, 119008 KB, 29848 KB/s, 3 seconds passed
... 94%, 119040 KB, 29853 KB/s, 3 seconds passed
... 94%, 119072 KB, 29858 KB/s, 3 seconds passed
... 94%, 119104 KB, 29863 KB/s, 3 seconds passed
... 94%, 119136 KB, 29868 KB/s, 3 seconds passed
... 94%, 119168 KB, 29873 KB/s, 3 seconds passed
... 94%, 119200 KB, 29878 KB/s, 3 seconds passed
... 94%, 119232 KB, 29884 KB/s, 3 seconds passed
... 94%, 119264 KB, 29889 KB/s, 3 seconds passed
... 94%, 119296 KB, 29894 KB/s, 3 seconds passed
... 94%, 119328 KB, 29899 KB/s, 3 seconds passed
... 94%, 119360 KB, 29904 KB/s, 3 seconds passed
... 94%, 119392 KB, 29909 KB/s, 3 seconds passed
... 94%, 119424 KB, 29914 KB/s, 3 seconds passed
... 94%, 119456 KB, 29919 KB/s, 3 seconds passed
... 94%, 119488 KB, 29924 KB/s, 3 seconds passed
... 94%, 119520 KB, 29930 KB/s, 3 seconds passed
... 94%, 119552 KB, 29935 KB/s, 3 seconds passed
... 94%, 119584 KB, 29940 KB/s, 3 seconds passed
... 94%, 119616 KB, 29946 KB/s, 3 seconds passed
... 94%, 119648 KB, 29953 KB/s, 3 seconds passed
... 95%, 119680 KB, 29959 KB/s, 3 seconds passed
... 95%, 119712 KB, 29965 KB/s, 3 seconds passed
... 95%, 119744 KB, 29971 KB/s, 3 seconds passed
... 95%, 119776 KB, 29977 KB/s, 3 seconds passed
... 95%, 119808 KB, 29983 KB/s, 3 seconds passed
... 95%, 119840 KB, 29989 KB/s, 3 seconds passed
... 95%, 119872 KB, 29995 KB/s, 3 seconds passed
... 95%, 119904 KB, 30002 KB/s, 3 seconds passed
... 95%, 119936 KB, 30008 KB/s, 3 seconds passed
... 95%, 119968 KB, 30014 KB/s, 3 seconds passed
... 95%, 120000 KB, 30020 KB/s, 3 seconds passed
... 95%, 120032 KB, 30026 KB/s, 3 seconds passed
... 95%, 120064 KB, 30033 KB/s, 3 seconds passed
... 95%, 120096 KB, 30039 KB/s, 3 seconds passed
... 95%, 120128 KB, 30045 KB/s, 3 seconds passed
... 95%, 120160 KB, 30051 KB/s, 3 seconds passed
... 95%, 120192 KB, 30057 KB/s, 3 seconds passed
... 95%, 120224 KB, 30064 KB/s, 3 seconds passed
... 95%, 120256 KB, 30070 KB/s, 3 seconds passed
... 95%, 120288 KB, 30076 KB/s, 3 seconds passed
... 95%, 120320 KB, 30082 KB/s, 3 seconds passed
... 95%, 120352 KB, 30088 KB/s, 3 seconds passed
... 95%, 120384 KB, 30094 KB/s, 4 seconds passed
... 95%, 120416 KB, 30101 KB/s, 4 seconds passed
... 95%, 120448 KB, 30107 KB/s, 4 seconds passed
... 95%, 120480 KB, 30113 KB/s, 4 seconds passed
... 95%, 120512 KB, 30119 KB/s, 4 seconds passed
... 95%, 120544 KB, 30125 KB/s, 4 seconds passed
... 95%, 120576 KB, 30132 KB/s, 4 seconds passed
... 95%, 120608 KB, 30138 KB/s, 4 seconds passed
... 95%, 120640 KB, 30144 KB/s, 4 seconds passed
... 95%, 120672 KB, 30150 KB/s, 4 seconds passed
... 95%, 120704 KB, 30157 KB/s, 4 seconds passed
... 95%, 120736 KB, 30163 KB/s, 4 seconds passed
... 95%, 120768 KB, 30169 KB/s, 4 seconds passed
... 95%, 120800 KB, 30175 KB/s, 4 seconds passed
... 95%, 120832 KB, 30181 KB/s, 4 seconds passed
... 95%, 120864 KB, 30187 KB/s, 4 seconds passed
... 95%, 120896 KB, 30193 KB/s, 4 seconds passed
... 96%, 120928 KB, 30199 KB/s, 4 seconds passed
... 96%, 120960 KB, 30205 KB/s, 4 seconds passed
... 96%, 120992 KB, 30212 KB/s, 4 seconds passed
... 96%, 121024 KB, 30219 KB/s, 4 seconds passed
... 96%, 121056 KB, 30226 KB/s, 4 seconds passed
... 96%, 121088 KB, 30233 KB/s, 4 seconds passed
... 96%, 121120 KB, 30239 KB/s, 4 seconds passed
... 96%, 121152 KB, 30246 KB/s, 4 seconds passed
... 96%, 121184 KB, 30253 KB/s, 4 seconds passed
... 96%, 121216 KB, 30260 KB/s, 4 seconds passed
... 96%, 121248 KB, 30267 KB/s, 4 seconds passed
... 96%, 121280 KB, 30274 KB/s, 4 seconds passed
... 96%, 121312 KB, 30211 KB/s, 4 seconds passed

.. parsed-literal::

    ... 96%, 121344 KB, 30216 KB/s, 4 seconds passed
... 96%, 121376 KB, 30221 KB/s, 4 seconds passed
... 96%, 121408 KB, 30221 KB/s, 4 seconds passed
... 96%, 121440 KB, 30226 KB/s, 4 seconds passed
... 96%, 121472 KB, 30231 KB/s, 4 seconds passed
... 96%, 121504 KB, 30236 KB/s, 4 seconds passed
... 96%, 121536 KB, 30242 KB/s, 4 seconds passed
... 96%, 121568 KB, 30247 KB/s, 4 seconds passed
... 96%, 121600 KB, 30252 KB/s, 4 seconds passed
... 96%, 121632 KB, 30258 KB/s, 4 seconds passed
... 96%, 121664 KB, 30263 KB/s, 4 seconds passed
... 96%, 121696 KB, 30268 KB/s, 4 seconds passed
... 96%, 121728 KB, 30274 KB/s, 4 seconds passed
... 96%, 121760 KB, 30279 KB/s, 4 seconds passed
... 96%, 121792 KB, 30284 KB/s, 4 seconds passed
... 96%, 121824 KB, 30290 KB/s, 4 seconds passed
... 96%, 121856 KB, 30295 KB/s, 4 seconds passed
... 96%, 121888 KB, 30301 KB/s, 4 seconds passed
... 96%, 121920 KB, 30306 KB/s, 4 seconds passed
... 96%, 121952 KB, 30311 KB/s, 4 seconds passed
... 96%, 121984 KB, 30317 KB/s, 4 seconds passed
... 96%, 122016 KB, 30322 KB/s, 4 seconds passed
... 96%, 122048 KB, 30327 KB/s, 4 seconds passed
... 96%, 122080 KB, 30333 KB/s, 4 seconds passed
... 96%, 122112 KB, 30339 KB/s, 4 seconds passed
... 96%, 122144 KB, 30344 KB/s, 4 seconds passed
... 97%, 122176 KB, 30350 KB/s, 4 seconds passed
... 97%, 122208 KB, 30339 KB/s, 4 seconds passed
... 97%, 122240 KB, 30340 KB/s, 4 seconds passed
... 97%, 122272 KB, 30345 KB/s, 4 seconds passed
... 97%, 122304 KB, 30351 KB/s, 4 seconds passed
... 97%, 122336 KB, 30356 KB/s, 4 seconds passed
... 97%, 122368 KB, 30362 KB/s, 4 seconds passed
... 97%, 122400 KB, 30367 KB/s, 4 seconds passed
... 97%, 122432 KB, 30373 KB/s, 4 seconds passed
... 97%, 122464 KB, 30379 KB/s, 4 seconds passed
... 97%, 122496 KB, 30384 KB/s, 4 seconds passed
... 97%, 122528 KB, 30389 KB/s, 4 seconds passed
... 97%, 122560 KB, 30395 KB/s, 4 seconds passed
... 97%, 122592 KB, 30400 KB/s, 4 seconds passed
... 97%, 122624 KB, 30406 KB/s, 4 seconds passed
... 97%, 122656 KB, 30411 KB/s, 4 seconds passed
... 97%, 122688 KB, 30417 KB/s, 4 seconds passed
... 97%, 122720 KB, 30422 KB/s, 4 seconds passed
... 97%, 122752 KB, 30428 KB/s, 4 seconds passed
... 97%, 122784 KB, 30434 KB/s, 4 seconds passed
... 97%, 122816 KB, 30439 KB/s, 4 seconds passed
... 97%, 122848 KB, 30445 KB/s, 4 seconds passed

.. parsed-literal::

    ... 97%, 122880 KB, 29864 KB/s, 4 seconds passed
... 97%, 122912 KB, 29869 KB/s, 4 seconds passed
... 97%, 122944 KB, 29872 KB/s, 4 seconds passed
... 97%, 122976 KB, 29877 KB/s, 4 seconds passed
... 97%, 123008 KB, 29882 KB/s, 4 seconds passed
... 97%, 123040 KB, 29887 KB/s, 4 seconds passed
... 97%, 123072 KB, 29892 KB/s, 4 seconds passed
... 97%, 123104 KB, 29897 KB/s, 4 seconds passed
... 97%, 123136 KB, 29902 KB/s, 4 seconds passed
... 97%, 123168 KB, 29907 KB/s, 4 seconds passed

.. parsed-literal::

    ... 97%, 123200 KB, 29912 KB/s, 4 seconds passed
... 97%, 123232 KB, 29917 KB/s, 4 seconds passed
... 97%, 123264 KB, 29922 KB/s, 4 seconds passed
... 97%, 123296 KB, 29926 KB/s, 4 seconds passed
... 97%, 123328 KB, 29931 KB/s, 4 seconds passed
... 97%, 123360 KB, 29936 KB/s, 4 seconds passed
... 97%, 123392 KB, 29941 KB/s, 4 seconds passed
... 97%, 123424 KB, 29946 KB/s, 4 seconds passed
... 98%, 123456 KB, 29951 KB/s, 4 seconds passed
... 98%, 123488 KB, 29956 KB/s, 4 seconds passed
... 98%, 123520 KB, 29962 KB/s, 4 seconds passed
... 98%, 123552 KB, 29968 KB/s, 4 seconds passed
... 98%, 123584 KB, 29973 KB/s, 4 seconds passed
... 98%, 123616 KB, 29978 KB/s, 4 seconds passed
... 98%, 123648 KB, 29983 KB/s, 4 seconds passed
... 98%, 123680 KB, 29988 KB/s, 4 seconds passed
... 98%, 123712 KB, 29994 KB/s, 4 seconds passed
... 98%, 123744 KB, 29999 KB/s, 4 seconds passed
... 98%, 123776 KB, 30004 KB/s, 4 seconds passed
... 98%, 123808 KB, 30010 KB/s, 4 seconds passed
... 98%, 123840 KB, 30016 KB/s, 4 seconds passed
... 98%, 123872 KB, 30022 KB/s, 4 seconds passed
... 98%, 123904 KB, 30028 KB/s, 4 seconds passed
... 98%, 123936 KB, 30034 KB/s, 4 seconds passed
... 98%, 123968 KB, 30040 KB/s, 4 seconds passed
... 98%, 124000 KB, 30046 KB/s, 4 seconds passed
... 98%, 124032 KB, 30052 KB/s, 4 seconds passed
... 98%, 124064 KB, 30058 KB/s, 4 seconds passed
... 98%, 124096 KB, 30064 KB/s, 4 seconds passed
... 98%, 124128 KB, 30070 KB/s, 4 seconds passed
... 98%, 124160 KB, 30076 KB/s, 4 seconds passed
... 98%, 124192 KB, 30081 KB/s, 4 seconds passed
... 98%, 124224 KB, 30087 KB/s, 4 seconds passed
... 98%, 124256 KB, 30093 KB/s, 4 seconds passed
... 98%, 124288 KB, 30099 KB/s, 4 seconds passed
... 98%, 124320 KB, 30104 KB/s, 4 seconds passed
... 98%, 124352 KB, 30110 KB/s, 4 seconds passed
... 98%, 124384 KB, 30116 KB/s, 4 seconds passed
... 98%, 124416 KB, 30122 KB/s, 4 seconds passed
... 98%, 124448 KB, 30127 KB/s, 4 seconds passed
... 98%, 124480 KB, 30133 KB/s, 4 seconds passed
... 98%, 124512 KB, 30139 KB/s, 4 seconds passed
... 98%, 124544 KB, 30145 KB/s, 4 seconds passed
... 98%, 124576 KB, 30150 KB/s, 4 seconds passed
... 98%, 124608 KB, 30155 KB/s, 4 seconds passed
... 98%, 124640 KB, 30161 KB/s, 4 seconds passed
... 98%, 124672 KB, 30167 KB/s, 4 seconds passed
... 99%, 124704 KB, 30173 KB/s, 4 seconds passed
... 99%, 124736 KB, 30178 KB/s, 4 seconds passed
... 99%, 124768 KB, 30184 KB/s, 4 seconds passed
... 99%, 124800 KB, 30190 KB/s, 4 seconds passed
... 99%, 124832 KB, 30196 KB/s, 4 seconds passed
... 99%, 124864 KB, 30201 KB/s, 4 seconds passed
... 99%, 124896 KB, 30207 KB/s, 4 seconds passed
... 99%, 124928 KB, 30213 KB/s, 4 seconds passed
... 99%, 124960 KB, 30218 KB/s, 4 seconds passed

.. parsed-literal::

    ... 99%, 124992 KB, 29619 KB/s, 4 seconds passed
... 99%, 125024 KB, 29622 KB/s, 4 seconds passed
... 99%, 125056 KB, 29627 KB/s, 4 seconds passed

.. parsed-literal::

    ... 99%, 125088 KB, 29632 KB/s, 4 seconds passed
... 99%, 125120 KB, 29637 KB/s, 4 seconds passed
... 99%, 125152 KB, 29642 KB/s, 4 seconds passed
... 99%, 125184 KB, 29647 KB/s, 4 seconds passed
... 99%, 125216 KB, 29652 KB/s, 4 seconds passed
... 99%, 125248 KB, 29657 KB/s, 4 seconds passed
... 99%, 125280 KB, 29662 KB/s, 4 seconds passed
... 99%, 125312 KB, 29666 KB/s, 4 seconds passed
... 99%, 125344 KB, 29671 KB/s, 4 seconds passed
... 99%, 125376 KB, 29676 KB/s, 4 seconds passed
... 99%, 125408 KB, 29681 KB/s, 4 seconds passed
... 99%, 125440 KB, 29686 KB/s, 4 seconds passed
... 99%, 125472 KB, 29691 KB/s, 4 seconds passed
... 99%, 125504 KB, 29696 KB/s, 4 seconds passed
... 99%, 125536 KB, 29701 KB/s, 4 seconds passed
... 99%, 125568 KB, 29706 KB/s, 4 seconds passed
... 99%, 125600 KB, 29710 KB/s, 4 seconds passed
... 99%, 125632 KB, 29715 KB/s, 4 seconds passed
... 99%, 125664 KB, 29720 KB/s, 4 seconds passed
... 99%, 125696 KB, 29725 KB/s, 4 seconds passed
... 99%, 125728 KB, 29730 KB/s, 4 seconds passed
... 99%, 125760 KB, 29735 KB/s, 4 seconds passed
... 99%, 125792 KB, 29740 KB/s, 4 seconds passed
... 99%, 125824 KB, 29745 KB/s, 4 seconds passed
... 99%, 125856 KB, 29750 KB/s, 4 seconds passed
... 99%, 125888 KB, 29754 KB/s, 4 seconds passed
... 99%, 125920 KB, 29760 KB/s, 4 seconds passed
... 99%, 125952 KB, 29765 KB/s, 4 seconds passed
... 100%, 125953 KB, 29763 KB/s, 4 seconds passed



.. parsed-literal::

    
    ========== Downloading models/public/colorization-v2/model/__init__.py


.. parsed-literal::

    ... 100%, 0 KB, 301 KB/s, 0 seconds passed



.. parsed-literal::

    
    ========== Downloading models/public/colorization-v2/model/base_color.py


.. parsed-literal::

    ... 100%, 0 KB, 1859 KB/s, 0 seconds passed

    
    ========== Downloading models/public/colorization-v2/model/eccv16.py


.. parsed-literal::

    ... 100%, 4 KB, 17298 KB/s, 0 seconds passed

    
    ========== Replacing text in models/public/colorization-v2/model/__init__.py
    ========== Replacing text in models/public/colorization-v2/model/__init__.py
    ========== Replacing text in models/public/colorization-v2/model/eccv16.py
    


Convert the model to OpenVINO IR
--------------------------------

`back to top ⬆️ <#table-of-contents>`__

``omz_converter`` converts the models that are not in the OpenVINO™ IR
format into that format using model conversion API.

The downloaded pytorch model is not in OpenVINO IR format which is
required for inference with OpenVINO runtime. ``omz_converter`` is used
to convert the downloaded pytorch model into ONNX and OpenVINO IR format
respectively

.. code:: ipython3

    if not os.path.exists(MODEL_PATH):
        convert_command = (
            f"omz_converter "
            f"--name {MODEL_NAME} "
            f"--download_dir {MODEL_DIR} "
            f"--precisions {PRECISION}"
        )
        ! $convert_command


.. parsed-literal::

    ========== Converting colorization-v2 to ONNX
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-path=models/public/colorization-v2 --model-name=ECCVGenerator --weights=models/public/colorization-v2/ckpt/colorization-v2-eccv16.pth --import-module=model --input-shape=1,1,256,256 --output-file=models/public/colorization-v2/colorization-v2-eccv16.onnx --input-names=data_l --output-names=color_ab
    


.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::

    
    ========== Converting colorization-v2 to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=models/public/colorization-v2/FP16 --model_name=colorization-v2 --input=data_l --output=color_ab --input_model=models/public/colorization-v2/colorization-v2-eccv16.onnx '--layout=data_l(NCHW)' '--input_shape=[1, 1, 256, 256]' --compress_to_fp16=True
    


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API. 
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


.. parsed-literal::

    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/notebooks/vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/notebooks/vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.bin


.. parsed-literal::

    


Loading the Model
-----------------

`back to top ⬆️ <#table-of-contents>`__

Load the model in OpenVINO Runtime with ``ie.read_model`` and compile it
for the specified device with ``ie.compile_model``.

.. code:: ipython3

    core = ov.Core()
    model = core.read_model(model=MODEL_PATH)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    N, C, H, W = list(input_layer.shape)

Utility Functions
-----------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    def read_image(impath: str) -> np.ndarray:
        """
        Returns an image as ndarra, given path to an image reads the
        (BGR) image using opencv's imread() API.
    
            Parameter:
                impath (string): Path of the image to be read and returned.
    
            Returns:
                image (ndarray): Numpy array representing the read image.
        """
    
        raw_image = cv2.imread(impath)
        if raw_image.shape[2] > 1:
            image = cv2.cvtColor(
                cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
        return image
    
    
    def plot_image(image: np.ndarray, title: str = "") -> None:
        """
        Given a image as ndarray and title as string, display it using
        matplotlib.
    
            Parameters:
                image (ndarray): Numpy array representing the image to be
                                 displayed.
                title (string): String representing the title of the plot.
    
            Returns:
                None
    
        """
    
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()
    
    
    def plot_output(gray_img: np.ndarray, color_img: np.ndarray) -> None:
        """
        Plots the original (bw or grayscale) image and colorized image
        on different column axes for comparing side by side.
    
            Parameters:
                gray_image (ndarray): Numpy array representing the original image.
                color_image (ndarray): Numpy array representing the model output.
    
            Returns:
                None
        """
    
        fig = plt.figure(figsize=(12, 12))
    
        ax1 = fig.add_subplot(1, 2, 1)
        plt.title("Input", fontsize=20)
        ax1.axis("off")
    
        ax2 = fig.add_subplot(1, 2, 2)
        plt.title("Colorized", fontsize=20)
        ax2.axis("off")
    
        ax1.imshow(gray_img)
        ax2.imshow(color_img)
    
        plt.show()

Load the Image
--------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    img_url_0 = "https://user-images.githubusercontent.com/18904157/180923287-20339d01-b1bf-493f-9a0d-55eff997aff1.jpg"
    img_url_1 = "https://user-images.githubusercontent.com/18904157/180923289-0bb71e09-25e1-46a6-aaf1-e8f666b62d26.jpg"
    
    image_file_0 = utils.download_file(
        img_url_0, filename="test_0.jpg", directory="data", show_progress=False, silent=True, timeout=30
    )
    assert Path(image_file_0).exists()
    
    image_file_1 = utils.download_file(
        img_url_1, filename="test_1.jpg", directory="data", show_progress=False, silent=True, timeout=30
    )
    assert Path(image_file_1).exists()
    
    test_img_0 = read_image("data/test_0.jpg")
    test_img_1 = read_image("data/test_1.jpg")

.. code:: ipython3

    def colorize(gray_img: np.ndarray) -> np.ndarray:
    
        """
        Given an image as ndarray for inference convert the image into LAB image, 
        the model consumes as input L-Channel of LAB image and provides output 
        A & B - Channels of LAB image. i.e returns a colorized image
    
            Parameters:
                gray_img (ndarray): Numpy array representing the original
                                    image.
    
            Returns:
                colorize_image (ndarray): Numpy arrray depicting the
                                          colorized version of the original
                                          image.
        """
        
        # Preprocess
        h_in, w_in, _ = gray_img.shape
        img_rgb = gray_img.astype(np.float32) / 255
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        img_l_rs = cv2.resize(img_lab.copy(), (W, H))[:, :, 0]
    
        # Inference
        inputs = np.expand_dims(img_l_rs, axis=[0, 1])
        res = compiled_model([inputs])[output_layer]
        update_res = np.squeeze(res)
    
        # Post-process
        out = update_res.transpose((1, 2, 0))
        out = cv2.resize(out, (w_in, h_in))
        img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis],
                                      out), axis=2)
        img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2RGB), 0, 1)
        colorized_image = (cv2.resize(img_bgr_out, (w_in, h_in))
                           * 255).astype(np.uint8)
        return colorized_image

.. code:: ipython3

    color_img_0 = colorize(test_img_0)
    color_img_1 = colorize(test_img_1)

Display Colorized Image
-----------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    plot_output(test_img_0, color_img_0)



.. image:: vision-image-colorization-with-output_files/vision-image-colorization-with-output_21_0.png


.. code:: ipython3

    plot_output(test_img_1, color_img_1)



.. image:: vision-image-colorization-with-output_files/vision-image-colorization-with-output_22_0.png

