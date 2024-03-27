Live 3D Human Pose Estimation with OpenVINO
===========================================

This notebook demonstrates live 3D Human Pose Estimation with OpenVINO
via a webcam. We utilize the model
`human-pose-estimation-3d-0001 <http/github.copenvinotoolkopen_model_ztrmastmodepublhuman-pose-estimation-3d-0001>`__
from `Open Model
Zoo <http/github.copenvinotoolkopen_model_z>`__. At the end
of this notebook, you will see live inference results from your webcam
(if available). Alternatively, you can also upload a video file to test
out the algorithms. **Make sure you have properly installed
the**\ `Jupyter
extension <http/github.cjupyter-widgepythreejs#jupyterlab>`__\ **and
been using JupyterLab to run the demo as suggested in the
``README.md``**

   **NOTE**: *To use a webcam, you must run this Jupyter notebook on a
   computer with a webcam. If you run on a remote server, the webcam
   will not work. However, you can still do inference on a video file in
   the final step. This demo utilizes the Python interface in
   ``Three.js`` integrated with WebGL to process data from the model
   inference. These results are processed and displayed in the
   notebook.*

*To ensure that the results are displayed correctly, run the code in a
recommended browser on one of the following operating systems:* *Ubuntu,
Windows: Chrome* *macOS: Safari*

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Imports <#imports>`__
-  `The model <#the-model>`__

   -  `Download the model <#download-the-model>`__
   -  `Convert Model to OpenVINO IR
      format <#convert-model-to-openvino-ir-format>`__
   -  `Select inference device <#select-inference-device>`__
   -  `Load the model <#load-the-model>`__

-  `Processing <#processing>`__

   -  `Model Inference <#model-inference>`__
   -  `Draw 2D Pose Overlays <#draw-2d-pose-overlays>`__
   -  `Main Processing Function <#main-processing-function>`__

-  `Run <#run>`__

Prerequisites
-------------



**The ``pythreejs`` extension may not display properly when using the
latest Jupyter Notebook release (2.4.1). Therefore, it is recommended to
use Jupyter Lab instead.**

.. code:: ipython3

    %pip install pythreejs "openvino-dev>=2024.0.0"


.. parsed-literal::

    Collecting pythreejs
      Using cached pythreejs-2.4.2-py3-none-any.whl.metadata (5.4 kB)
    Requirement already satisfied: openvino-dev>=2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2024.0.0)
    Requirement already satisfied: ipywidgets>=7.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (8.1.2)


.. parsed-literal::

    Collecting ipydatawidgets>=1.1.1 (from pythreejs)
      Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: numpy in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (1.23.5)
    Requirement already satisfied: traitlets in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (5.14.2)
    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (0.7.1)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (3.1)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2023.2.1)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (24.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.31.0)
    Requirement already satisfied: openvino==2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.0.0)


.. parsed-literal::

    Collecting traittypes>=0.2.0 (from ipydatawidgets>=1.1.1->pythreejs)
      Using cached traittypes-0.2.1-py2.py3-none-any.whl.metadata (1.0 kB)
    Requirement already satisfied: comm>=0.1.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (0.2.2)
    Requirement already satisfied: ipython>=6.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (8.12.3)
    Requirement already satisfied: widgetsnbextension~=4.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (4.0.10)
    Requirement already satisfied: jupyterlab-widgets~=3.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (3.0.10)


.. parsed-literal::

    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2024.2.2)


.. parsed-literal::

    Requirement already satisfied: backcall in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.0)
    Requirement already satisfied: decorator in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.19.1)
    Requirement already satisfied: matplotlib-inline in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.1.6)
    Requirement already satisfied: pickleshare in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (3.0.43)
    Requirement already satisfied: pygments>=2.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.17.2)
    Requirement already satisfied: stack-data in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.6.3)
    Requirement already satisfied: typing-extensions in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.10.0)
    Requirement already satisfied: pexpect>4.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.9.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.13)


.. parsed-literal::

    Requirement already satisfied: executing>=1.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.0.1)
    Requirement already satisfied: asttokens>=2.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.4.1)
    Requirement already satisfied: pure-eval in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.2)
    Requirement already satisfied: six>=1.12.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (1.16.0)


.. parsed-literal::

    Using cached pythreejs-2.4.2-py3-none-any.whl (3.4 MB)
    Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl (271 kB)
    Using cached traittypes-0.2.1-py2.py3-none-any.whl (8.6 kB)


.. parsed-literal::

    Installing collected packages: traittypes, ipydatawidgets, pythreejs


.. parsed-literal::

    Successfully installed ipydatawidgets-4.3.5 pythreejs-2.4.2 traittypes-0.2.1


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import collections
    import sys
    import time
    from pathlib import Path
    
    import cv2
    import ipywidgets as widgets
    import numpy as np
    from IPython.display import clear_output, display
    import openvino as ov
    
    sys.path.append("../utils")
    import notebook_utils as utils
    
    sys.path.append("./engine")
    import engine.engine3js as engine
    from engine.parse_poses import parse_poses

The model
---------



Download the model
~~~~~~~~~~~~~~~~~~



We use ``omz_downloader``, which is a command line tool from the
``openvino-dev`` package. ``omz_downloader`` automatically creates a
directory structure and downloads the selected model.

.. code:: ipython3

    # directory where model will be downloaded
    base_model_dir = "model"
    
    # model name as named in Open Model Zoo
    model_name = "human-pose-estimation-3d-0001"
    # selected precision (FP32, FP16)
    precision = "FP32"
    
    BASE_MODEL_NAME = f"{base_model_dir}/public/{model_name}/{model_name}"
    model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
    onnx_path = Path(BASE_MODEL_NAME).with_suffix(".onnx")
    
    ir_model_path = f"model/public/{model_name}/{precision}/{model_name}.xml"
    model_weights_path = f"model/public/{model_name}/{precision}/{model_name}.bin"
    
    if not model_path.exists():
        download_command = (
            f"omz_downloader " f"--name {model_name} " f"--output_dir {base_model_dir}"
        )
        ! $download_command


.. parsed-literal::

    ################|| Downloading human-pose-estimation-3d-0001 ||################
    
    ========== Downloading model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz


.. parsed-literal::

    ... 0%, 32 KB, 872 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 889 KB/s, 0 seconds passed
... 0%, 96 KB, 1300 KB/s, 0 seconds passed
... 0%, 128 KB, 1183 KB/s, 0 seconds passed
... 0%, 160 KB, 1449 KB/s, 0 seconds passed
... 1%, 192 KB, 1712 KB/s, 0 seconds passed
... 1%, 224 KB, 1949 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 256 KB, 2185 KB/s, 0 seconds passed
... 1%, 288 KB, 1982 KB/s, 0 seconds passed
... 1%, 320 KB, 2180 KB/s, 0 seconds passed
... 1%, 352 KB, 2385 KB/s, 0 seconds passed
... 2%, 384 KB, 2570 KB/s, 0 seconds passed
... 2%, 416 KB, 2755 KB/s, 0 seconds passed
... 2%, 448 KB, 2942 KB/s, 0 seconds passed
... 2%, 480 KB, 3100 KB/s, 0 seconds passed
... 2%, 512 KB, 3296 KB/s, 0 seconds passed
... 3%, 544 KB, 3464 KB/s, 0 seconds passed
... 3%, 576 KB, 3645 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 608 KB, 3336 KB/s, 0 seconds passed
... 3%, 640 KB, 3501 KB/s, 0 seconds passed
... 3%, 672 KB, 3668 KB/s, 0 seconds passed
... 3%, 704 KB, 3834 KB/s, 0 seconds passed
... 4%, 736 KB, 4000 KB/s, 0 seconds passed
... 4%, 768 KB, 4159 KB/s, 0 seconds passed
... 4%, 800 KB, 4319 KB/s, 0 seconds passed
... 4%, 832 KB, 4483 KB/s, 0 seconds passed
... 4%, 864 KB, 4647 KB/s, 0 seconds passed
... 4%, 896 KB, 4809 KB/s, 0 seconds passed
... 5%, 928 KB, 4970 KB/s, 0 seconds passed
... 5%, 960 KB, 5126 KB/s, 0 seconds passed
... 5%, 992 KB, 5285 KB/s, 0 seconds passed
... 5%, 1024 KB, 5444 KB/s, 0 seconds passed
... 5%, 1056 KB, 5515 KB/s, 0 seconds passed
... 6%, 1088 KB, 5670 KB/s, 0 seconds passed
... 6%, 1120 KB, 5813 KB/s, 0 seconds passed
... 6%, 1152 KB, 5966 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 1184 KB, 5406 KB/s, 0 seconds passed
... 6%, 1216 KB, 5538 KB/s, 0 seconds passed
... 6%, 1248 KB, 5673 KB/s, 0 seconds passed
... 7%, 1280 KB, 5809 KB/s, 0 seconds passed
... 7%, 1312 KB, 5938 KB/s, 0 seconds passed
... 7%, 1344 KB, 6072 KB/s, 0 seconds passed
... 7%, 1376 KB, 6205 KB/s, 0 seconds passed
... 7%, 1408 KB, 6338 KB/s, 0 seconds passed
... 8%, 1440 KB, 6470 KB/s, 0 seconds passed
... 8%, 1472 KB, 6602 KB/s, 0 seconds passed
... 8%, 1504 KB, 6733 KB/s, 0 seconds passed
... 8%, 1536 KB, 6865 KB/s, 0 seconds passed
... 8%, 1568 KB, 6996 KB/s, 0 seconds passed
... 8%, 1600 KB, 7127 KB/s, 0 seconds passed
... 9%, 1632 KB, 7257 KB/s, 0 seconds passed
... 9%, 1664 KB, 7387 KB/s, 0 seconds passed
... 9%, 1696 KB, 7516 KB/s, 0 seconds passed
... 9%, 1728 KB, 7644 KB/s, 0 seconds passed
... 9%, 1760 KB, 7773 KB/s, 0 seconds passed
... 9%, 1792 KB, 7901 KB/s, 0 seconds passed
... 10%, 1824 KB, 8028 KB/s, 0 seconds passed
... 10%, 1856 KB, 8155 KB/s, 0 seconds passed
... 10%, 1888 KB, 8282 KB/s, 0 seconds passed
... 10%, 1920 KB, 8409 KB/s, 0 seconds passed
... 10%, 1952 KB, 8535 KB/s, 0 seconds passed
... 11%, 1984 KB, 8663 KB/s, 0 seconds passed
... 11%, 2016 KB, 8792 KB/s, 0 seconds passed
... 11%, 2048 KB, 8917 KB/s, 0 seconds passed
... 11%, 2080 KB, 9042 KB/s, 0 seconds passed
... 11%, 2112 KB, 9166 KB/s, 0 seconds passed
... 11%, 2144 KB, 9290 KB/s, 0 seconds passed
... 12%, 2176 KB, 9414 KB/s, 0 seconds passed
... 12%, 2208 KB, 9538 KB/s, 0 seconds passed
... 12%, 2240 KB, 9663 KB/s, 0 seconds passed
... 12%, 2272 KB, 9789 KB/s, 0 seconds passed
... 12%, 2304 KB, 9914 KB/s, 0 seconds passed
... 12%, 2336 KB, 10040 KB/s, 0 seconds passed
... 13%, 2368 KB, 10165 KB/s, 0 seconds passed
... 13%, 2400 KB, 9422 KB/s, 0 seconds passed
... 13%, 2432 KB, 9387 KB/s, 0 seconds passed
... 13%, 2464 KB, 9493 KB/s, 0 seconds passed
... 13%, 2496 KB, 9572 KB/s, 0 seconds passed
... 14%, 2528 KB, 9674 KB/s, 0 seconds passed
... 14%, 2560 KB, 9781 KB/s, 0 seconds passed
... 14%, 2592 KB, 9888 KB/s, 0 seconds passed
... 14%, 2624 KB, 9995 KB/s, 0 seconds passed
... 14%, 2656 KB, 10102 KB/s, 0 seconds passed
... 14%, 2688 KB, 10210 KB/s, 0 seconds passed
... 15%, 2720 KB, 10316 KB/s, 0 seconds passed
... 15%, 2752 KB, 10423 KB/s, 0 seconds passed
... 15%, 2784 KB, 10529 KB/s, 0 seconds passed
... 15%, 2816 KB, 10635 KB/s, 0 seconds passed
... 15%, 2848 KB, 10740 KB/s, 0 seconds passed
... 16%, 2880 KB, 10845 KB/s, 0 seconds passed
... 16%, 2912 KB, 10949 KB/s, 0 seconds passed
... 16%, 2944 KB, 11053 KB/s, 0 seconds passed
... 16%, 2976 KB, 11157 KB/s, 0 seconds passed
... 16%, 3008 KB, 11262 KB/s, 0 seconds passed
... 16%, 3040 KB, 11365 KB/s, 0 seconds passed
... 17%, 3072 KB, 11467 KB/s, 0 seconds passed
... 17%, 3104 KB, 11570 KB/s, 0 seconds passed
... 17%, 3136 KB, 11673 KB/s, 0 seconds passed
... 17%, 3168 KB, 11776 KB/s, 0 seconds passed

.. parsed-literal::

    ... 17%, 3200 KB, 11878 KB/s, 0 seconds passed
... 17%, 3232 KB, 11979 KB/s, 0 seconds passed
... 18%, 3264 KB, 12081 KB/s, 0 seconds passed
... 18%, 3296 KB, 12182 KB/s, 0 seconds passed
... 18%, 3328 KB, 12283 KB/s, 0 seconds passed
... 18%, 3360 KB, 12384 KB/s, 0 seconds passed
... 18%, 3392 KB, 12484 KB/s, 0 seconds passed
... 19%, 3424 KB, 12585 KB/s, 0 seconds passed
... 19%, 3456 KB, 12683 KB/s, 0 seconds passed
... 19%, 3488 KB, 12783 KB/s, 0 seconds passed
... 19%, 3520 KB, 12883 KB/s, 0 seconds passed
... 19%, 3552 KB, 12987 KB/s, 0 seconds passed
... 19%, 3584 KB, 13091 KB/s, 0 seconds passed
... 20%, 3616 KB, 13196 KB/s, 0 seconds passed
... 20%, 3648 KB, 13301 KB/s, 0 seconds passed
... 20%, 3680 KB, 13406 KB/s, 0 seconds passed
... 20%, 3712 KB, 13510 KB/s, 0 seconds passed
... 20%, 3744 KB, 13615 KB/s, 0 seconds passed
... 20%, 3776 KB, 13719 KB/s, 0 seconds passed
... 21%, 3808 KB, 13823 KB/s, 0 seconds passed
... 21%, 3840 KB, 13927 KB/s, 0 seconds passed
... 21%, 3872 KB, 14031 KB/s, 0 seconds passed
... 21%, 3904 KB, 14134 KB/s, 0 seconds passed
... 21%, 3936 KB, 14238 KB/s, 0 seconds passed
... 22%, 3968 KB, 14341 KB/s, 0 seconds passed
... 22%, 4000 KB, 14444 KB/s, 0 seconds passed
... 22%, 4032 KB, 14547 KB/s, 0 seconds passed
... 22%, 4064 KB, 14650 KB/s, 0 seconds passed
... 22%, 4096 KB, 14752 KB/s, 0 seconds passed
... 22%, 4128 KB, 14854 KB/s, 0 seconds passed
... 23%, 4160 KB, 14955 KB/s, 0 seconds passed
... 23%, 4192 KB, 15057 KB/s, 0 seconds passed
... 23%, 4224 KB, 15159 KB/s, 0 seconds passed
... 23%, 4256 KB, 15261 KB/s, 0 seconds passed
... 23%, 4288 KB, 15362 KB/s, 0 seconds passed
... 24%, 4320 KB, 15463 KB/s, 0 seconds passed
... 24%, 4352 KB, 15565 KB/s, 0 seconds passed
... 24%, 4384 KB, 15666 KB/s, 0 seconds passed
... 24%, 4416 KB, 15766 KB/s, 0 seconds passed
... 24%, 4448 KB, 15866 KB/s, 0 seconds passed
... 24%, 4480 KB, 15967 KB/s, 0 seconds passed
... 25%, 4512 KB, 16069 KB/s, 0 seconds passed
... 25%, 4544 KB, 16172 KB/s, 0 seconds passed
... 25%, 4576 KB, 16274 KB/s, 0 seconds passed
... 25%, 4608 KB, 16377 KB/s, 0 seconds passed
... 25%, 4640 KB, 16480 KB/s, 0 seconds passed
... 25%, 4672 KB, 16581 KB/s, 0 seconds passed
... 26%, 4704 KB, 16683 KB/s, 0 seconds passed
... 26%, 4736 KB, 16785 KB/s, 0 seconds passed
... 26%, 4768 KB, 16887 KB/s, 0 seconds passed
... 26%, 4800 KB, 16278 KB/s, 0 seconds passed
... 26%, 4832 KB, 16359 KB/s, 0 seconds passed
... 27%, 4864 KB, 16445 KB/s, 0 seconds passed
... 27%, 4896 KB, 16532 KB/s, 0 seconds passed
... 27%, 4928 KB, 16620 KB/s, 0 seconds passed
... 27%, 4960 KB, 16711 KB/s, 0 seconds passed
... 27%, 4992 KB, 16367 KB/s, 0 seconds passed
... 27%, 5024 KB, 16444 KB/s, 0 seconds passed
... 28%, 5056 KB, 16525 KB/s, 0 seconds passed
... 28%, 5088 KB, 16608 KB/s, 0 seconds passed
... 28%, 5120 KB, 16690 KB/s, 0 seconds passed
... 28%, 5152 KB, 16774 KB/s, 0 seconds passed
... 28%, 5184 KB, 16857 KB/s, 0 seconds passed
... 28%, 5216 KB, 16940 KB/s, 0 seconds passed
... 29%, 5248 KB, 17023 KB/s, 0 seconds passed
... 29%, 5280 KB, 17106 KB/s, 0 seconds passed
... 29%, 5312 KB, 17189 KB/s, 0 seconds passed
... 29%, 5344 KB, 17272 KB/s, 0 seconds passed
... 29%, 5376 KB, 17353 KB/s, 0 seconds passed
... 30%, 5408 KB, 17434 KB/s, 0 seconds passed
... 30%, 5440 KB, 17516 KB/s, 0 seconds passed
... 30%, 5472 KB, 17597 KB/s, 0 seconds passed
... 30%, 5504 KB, 17678 KB/s, 0 seconds passed
... 30%, 5536 KB, 17759 KB/s, 0 seconds passed
... 30%, 5568 KB, 17839 KB/s, 0 seconds passed
... 31%, 5600 KB, 17920 KB/s, 0 seconds passed
... 31%, 5632 KB, 18001 KB/s, 0 seconds passed
... 31%, 5664 KB, 18082 KB/s, 0 seconds passed
... 31%, 5696 KB, 18165 KB/s, 0 seconds passed
... 31%, 5728 KB, 18248 KB/s, 0 seconds passed
... 32%, 5760 KB, 18331 KB/s, 0 seconds passed
... 32%, 5792 KB, 18414 KB/s, 0 seconds passed
... 32%, 5824 KB, 18498 KB/s, 0 seconds passed
... 32%, 5856 KB, 18581 KB/s, 0 seconds passed
... 32%, 5888 KB, 18664 KB/s, 0 seconds passed
... 32%, 5920 KB, 18748 KB/s, 0 seconds passed
... 33%, 5952 KB, 18831 KB/s, 0 seconds passed
... 33%, 5984 KB, 18913 KB/s, 0 seconds passed
... 33%, 6016 KB, 18996 KB/s, 0 seconds passed
... 33%, 6048 KB, 19078 KB/s, 0 seconds passed
... 33%, 6080 KB, 19160 KB/s, 0 seconds passed
... 33%, 6112 KB, 19242 KB/s, 0 seconds passed
... 34%, 6144 KB, 19323 KB/s, 0 seconds passed
... 34%, 6176 KB, 19407 KB/s, 0 seconds passed
... 34%, 6208 KB, 19490 KB/s, 0 seconds passed
... 34%, 6240 KB, 19571 KB/s, 0 seconds passed
... 34%, 6272 KB, 19653 KB/s, 0 seconds passed
... 35%, 6304 KB, 19734 KB/s, 0 seconds passed
... 35%, 6336 KB, 19815 KB/s, 0 seconds passed
... 35%, 6368 KB, 19897 KB/s, 0 seconds passed
... 35%, 6400 KB, 19978 KB/s, 0 seconds passed
... 35%, 6432 KB, 20060 KB/s, 0 seconds passed

.. parsed-literal::

    ... 35%, 6464 KB, 20140 KB/s, 0 seconds passed
... 36%, 6496 KB, 20220 KB/s, 0 seconds passed
... 36%, 6528 KB, 20300 KB/s, 0 seconds passed
... 36%, 6560 KB, 20378 KB/s, 0 seconds passed
... 36%, 6592 KB, 20457 KB/s, 0 seconds passed
... 36%, 6624 KB, 20536 KB/s, 0 seconds passed
... 36%, 6656 KB, 20615 KB/s, 0 seconds passed
... 37%, 6688 KB, 20695 KB/s, 0 seconds passed
... 37%, 6720 KB, 20775 KB/s, 0 seconds passed
... 37%, 6752 KB, 20861 KB/s, 0 seconds passed
... 37%, 6784 KB, 20946 KB/s, 0 seconds passed
... 37%, 6816 KB, 21032 KB/s, 0 seconds passed
... 38%, 6848 KB, 21118 KB/s, 0 seconds passed
... 38%, 6880 KB, 21204 KB/s, 0 seconds passed
... 38%, 6912 KB, 21290 KB/s, 0 seconds passed
... 38%, 6944 KB, 21376 KB/s, 0 seconds passed
... 38%, 6976 KB, 21461 KB/s, 0 seconds passed
... 38%, 7008 KB, 21546 KB/s, 0 seconds passed
... 39%, 7040 KB, 21631 KB/s, 0 seconds passed
... 39%, 7072 KB, 21717 KB/s, 0 seconds passed
... 39%, 7104 KB, 21801 KB/s, 0 seconds passed
... 39%, 7136 KB, 21886 KB/s, 0 seconds passed
... 39%, 7168 KB, 21971 KB/s, 0 seconds passed
... 40%, 7200 KB, 22056 KB/s, 0 seconds passed
... 40%, 7232 KB, 22140 KB/s, 0 seconds passed
... 40%, 7264 KB, 22224 KB/s, 0 seconds passed
... 40%, 7296 KB, 22309 KB/s, 0 seconds passed
... 40%, 7328 KB, 22393 KB/s, 0 seconds passed
... 40%, 7360 KB, 22478 KB/s, 0 seconds passed
... 41%, 7392 KB, 22562 KB/s, 0 seconds passed
... 41%, 7424 KB, 22647 KB/s, 0 seconds passed
... 41%, 7456 KB, 22731 KB/s, 0 seconds passed
... 41%, 7488 KB, 22814 KB/s, 0 seconds passed
... 41%, 7520 KB, 22896 KB/s, 0 seconds passed
... 41%, 7552 KB, 22980 KB/s, 0 seconds passed
... 42%, 7584 KB, 23064 KB/s, 0 seconds passed
... 42%, 7616 KB, 23148 KB/s, 0 seconds passed
... 42%, 7648 KB, 23231 KB/s, 0 seconds passed
... 42%, 7680 KB, 23315 KB/s, 0 seconds passed
... 42%, 7712 KB, 23398 KB/s, 0 seconds passed
... 43%, 7744 KB, 23480 KB/s, 0 seconds passed
... 43%, 7776 KB, 23563 KB/s, 0 seconds passed
... 43%, 7808 KB, 23646 KB/s, 0 seconds passed
... 43%, 7840 KB, 23729 KB/s, 0 seconds passed
... 43%, 7872 KB, 23812 KB/s, 0 seconds passed
... 43%, 7904 KB, 23895 KB/s, 0 seconds passed
... 44%, 7936 KB, 23978 KB/s, 0 seconds passed
... 44%, 7968 KB, 24055 KB/s, 0 seconds passed
... 44%, 8000 KB, 24132 KB/s, 0 seconds passed
... 44%, 8032 KB, 24206 KB/s, 0 seconds passed
... 44%, 8064 KB, 24283 KB/s, 0 seconds passed
... 45%, 8096 KB, 24360 KB/s, 0 seconds passed
... 45%, 8128 KB, 24437 KB/s, 0 seconds passed
... 45%, 8160 KB, 24514 KB/s, 0 seconds passed
... 45%, 8192 KB, 24586 KB/s, 0 seconds passed
... 45%, 8224 KB, 24663 KB/s, 0 seconds passed
... 45%, 8256 KB, 24739 KB/s, 0 seconds passed
... 46%, 8288 KB, 24812 KB/s, 0 seconds passed
... 46%, 8320 KB, 24888 KB/s, 0 seconds passed
... 46%, 8352 KB, 24965 KB/s, 0 seconds passed
... 46%, 8384 KB, 25041 KB/s, 0 seconds passed
... 46%, 8416 KB, 25117 KB/s, 0 seconds passed
... 46%, 8448 KB, 25188 KB/s, 0 seconds passed
... 47%, 8480 KB, 25264 KB/s, 0 seconds passed
... 47%, 8512 KB, 25340 KB/s, 0 seconds passed
... 47%, 8544 KB, 25411 KB/s, 0 seconds passed
... 47%, 8576 KB, 25486 KB/s, 0 seconds passed
... 47%, 8608 KB, 25561 KB/s, 0 seconds passed
... 48%, 8640 KB, 25636 KB/s, 0 seconds passed
... 48%, 8672 KB, 25711 KB/s, 0 seconds passed
... 48%, 8704 KB, 25778 KB/s, 0 seconds passed
... 48%, 8736 KB, 25848 KB/s, 0 seconds passed
... 48%, 8768 KB, 25909 KB/s, 0 seconds passed
... 48%, 8800 KB, 25975 KB/s, 0 seconds passed
... 49%, 8832 KB, 26059 KB/s, 0 seconds passed
... 49%, 8864 KB, 26143 KB/s, 0 seconds passed
... 49%, 8896 KB, 26218 KB/s, 0 seconds passed
... 49%, 8928 KB, 26288 KB/s, 0 seconds passed
... 49%, 8960 KB, 26362 KB/s, 0 seconds passed
... 49%, 8992 KB, 26436 KB/s, 0 seconds passed
... 50%, 9024 KB, 26509 KB/s, 0 seconds passed
... 50%, 9056 KB, 26583 KB/s, 0 seconds passed
... 50%, 9088 KB, 26652 KB/s, 0 seconds passed
... 50%, 9120 KB, 26725 KB/s, 0 seconds passed
... 50%, 9152 KB, 26798 KB/s, 0 seconds passed
... 51%, 9184 KB, 26871 KB/s, 0 seconds passed
... 51%, 9216 KB, 26940 KB/s, 0 seconds passed
... 51%, 9248 KB, 27013 KB/s, 0 seconds passed
... 51%, 9280 KB, 27090 KB/s, 0 seconds passed
... 51%, 9312 KB, 27158 KB/s, 0 seconds passed
... 51%, 9344 KB, 27231 KB/s, 0 seconds passed
... 52%, 9376 KB, 27303 KB/s, 0 seconds passed
... 52%, 9408 KB, 27371 KB/s, 0 seconds passed
... 52%, 9440 KB, 27444 KB/s, 0 seconds passed
... 52%, 9472 KB, 27516 KB/s, 0 seconds passed
... 52%, 9504 KB, 27583 KB/s, 0 seconds passed
... 53%, 9536 KB, 27642 KB/s, 0 seconds passed
... 53%, 9568 KB, 27701 KB/s, 0 seconds passed
... 53%, 9600 KB, 27764 KB/s, 0 seconds passed
... 53%, 9632 KB, 27126 KB/s, 0 seconds passed
... 53%, 9664 KB, 27185 KB/s, 0 seconds passed
... 53%, 9696 KB, 27245 KB/s, 0 seconds passed
... 54%, 9728 KB, 27305 KB/s, 0 seconds passed
... 54%, 9760 KB, 27367 KB/s, 0 seconds passed
... 54%, 9792 KB, 27429 KB/s, 0 seconds passed
... 54%, 9824 KB, 27492 KB/s, 0 seconds passed
... 54%, 9856 KB, 27554 KB/s, 0 seconds passed
... 54%, 9888 KB, 27614 KB/s, 0 seconds passed
... 55%, 9920 KB, 27675 KB/s, 0 seconds passed
... 55%, 9952 KB, 27737 KB/s, 0 seconds passed
... 55%, 9984 KB, 27799 KB/s, 0 seconds passed
... 55%, 10016 KB, 27860 KB/s, 0 seconds passed
... 55%, 10048 KB, 27920 KB/s, 0 seconds passed
... 56%, 10080 KB, 27980 KB/s, 0 seconds passed
... 56%, 10112 KB, 28041 KB/s, 0 seconds passed
... 56%, 10144 KB, 28101 KB/s, 0 seconds passed
... 56%, 10176 KB, 28162 KB/s, 0 seconds passed
... 56%, 10208 KB, 28225 KB/s, 0 seconds passed
... 56%, 10240 KB, 28290 KB/s, 0 seconds passed
... 57%, 10272 KB, 28356 KB/s, 0 seconds passed
... 57%, 10304 KB, 28422 KB/s, 0 seconds passed
... 57%, 10336 KB, 28489 KB/s, 0 seconds passed

.. parsed-literal::

    ... 57%, 10368 KB, 17538 KB/s, 0 seconds passed
... 57%, 10400 KB, 17319 KB/s, 0 seconds passed
... 57%, 10432 KB, 17081 KB/s, 0 seconds passed
... 58%, 10464 KB, 16880 KB/s, 0 seconds passed
... 58%, 10496 KB, 16918 KB/s, 0 seconds passed
... 58%, 10528 KB, 16958 KB/s, 0 seconds passed
... 58%, 10560 KB, 16999 KB/s, 0 seconds passed
... 58%, 10592 KB, 17040 KB/s, 0 seconds passed
... 59%, 10624 KB, 17080 KB/s, 0 seconds passed
... 59%, 10656 KB, 17121 KB/s, 0 seconds passed
... 59%, 10688 KB, 17162 KB/s, 0 seconds passed
... 59%, 10720 KB, 17202 KB/s, 0 seconds passed
... 59%, 10752 KB, 17243 KB/s, 0 seconds passed
... 59%, 10784 KB, 17283 KB/s, 0 seconds passed
... 60%, 10816 KB, 17324 KB/s, 0 seconds passed
... 60%, 10848 KB, 17365 KB/s, 0 seconds passed
... 60%, 10880 KB, 17406 KB/s, 0 seconds passed
... 60%, 10912 KB, 17446 KB/s, 0 seconds passed
... 60%, 10944 KB, 17486 KB/s, 0 seconds passed
... 61%, 10976 KB, 17526 KB/s, 0 seconds passed
... 61%, 11008 KB, 17567 KB/s, 0 seconds passed
... 61%, 11040 KB, 17607 KB/s, 0 seconds passed
... 61%, 11072 KB, 17647 KB/s, 0 seconds passed
... 61%, 11104 KB, 17688 KB/s, 0 seconds passed

.. parsed-literal::

    ... 61%, 11136 KB, 17728 KB/s, 0 seconds passed
... 62%, 11168 KB, 17768 KB/s, 0 seconds passed
... 62%, 11200 KB, 17809 KB/s, 0 seconds passed
... 62%, 11232 KB, 17849 KB/s, 0 seconds passed
... 62%, 11264 KB, 17890 KB/s, 0 seconds passed
... 62%, 11296 KB, 17929 KB/s, 0 seconds passed
... 62%, 11328 KB, 17969 KB/s, 0 seconds passed
... 63%, 11360 KB, 18009 KB/s, 0 seconds passed
... 63%, 11392 KB, 18049 KB/s, 0 seconds passed
... 63%, 11424 KB, 18089 KB/s, 0 seconds passed
... 63%, 11456 KB, 18128 KB/s, 0 seconds passed
... 63%, 11488 KB, 18168 KB/s, 0 seconds passed
... 64%, 11520 KB, 18208 KB/s, 0 seconds passed
... 64%, 11552 KB, 18248 KB/s, 0 seconds passed
... 64%, 11584 KB, 18287 KB/s, 0 seconds passed
... 64%, 11616 KB, 18326 KB/s, 0 seconds passed
... 64%, 11648 KB, 18365 KB/s, 0 seconds passed
... 64%, 11680 KB, 18405 KB/s, 0 seconds passed
... 65%, 11712 KB, 18444 KB/s, 0 seconds passed
... 65%, 11744 KB, 18484 KB/s, 0 seconds passed
... 65%, 11776 KB, 18524 KB/s, 0 seconds passed
... 65%, 11808 KB, 18563 KB/s, 0 seconds passed
... 65%, 11840 KB, 18603 KB/s, 0 seconds passed
... 65%, 11872 KB, 18644 KB/s, 0 seconds passed
... 66%, 11904 KB, 18687 KB/s, 0 seconds passed
... 66%, 11936 KB, 18730 KB/s, 0 seconds passed
... 66%, 11968 KB, 18773 KB/s, 0 seconds passed
... 66%, 12000 KB, 18816 KB/s, 0 seconds passed
... 66%, 12032 KB, 18859 KB/s, 0 seconds passed
... 67%, 12064 KB, 18901 KB/s, 0 seconds passed
... 67%, 12096 KB, 18944 KB/s, 0 seconds passed
... 67%, 12128 KB, 18987 KB/s, 0 seconds passed
... 67%, 12160 KB, 19030 KB/s, 0 seconds passed
... 67%, 12192 KB, 19073 KB/s, 0 seconds passed
... 67%, 12224 KB, 19116 KB/s, 0 seconds passed
... 68%, 12256 KB, 19159 KB/s, 0 seconds passed
... 68%, 12288 KB, 19201 KB/s, 0 seconds passed
... 68%, 12320 KB, 19244 KB/s, 0 seconds passed
... 68%, 12352 KB, 19287 KB/s, 0 seconds passed
... 68%, 12384 KB, 19329 KB/s, 0 seconds passed
... 69%, 12416 KB, 19372 KB/s, 0 seconds passed
... 69%, 12448 KB, 19414 KB/s, 0 seconds passed
... 69%, 12480 KB, 19457 KB/s, 0 seconds passed
... 69%, 12512 KB, 19499 KB/s, 0 seconds passed
... 69%, 12544 KB, 19541 KB/s, 0 seconds passed
... 69%, 12576 KB, 19583 KB/s, 0 seconds passed
... 70%, 12608 KB, 19626 KB/s, 0 seconds passed
... 70%, 12640 KB, 19669 KB/s, 0 seconds passed
... 70%, 12672 KB, 19711 KB/s, 0 seconds passed
... 70%, 12704 KB, 19753 KB/s, 0 seconds passed
... 70%, 12736 KB, 19795 KB/s, 0 seconds passed
... 70%, 12768 KB, 19838 KB/s, 0 seconds passed
... 71%, 12800 KB, 19880 KB/s, 0 seconds passed
... 71%, 12832 KB, 19922 KB/s, 0 seconds passed
... 71%, 12864 KB, 19964 KB/s, 0 seconds passed
... 71%, 12896 KB, 20006 KB/s, 0 seconds passed
... 71%, 12928 KB, 20049 KB/s, 0 seconds passed
... 72%, 12960 KB, 20091 KB/s, 0 seconds passed
... 72%, 12992 KB, 20133 KB/s, 0 seconds passed
... 72%, 13024 KB, 20174 KB/s, 0 seconds passed
... 72%, 13056 KB, 20215 KB/s, 0 seconds passed
... 72%, 13088 KB, 20257 KB/s, 0 seconds passed
... 72%, 13120 KB, 20298 KB/s, 0 seconds passed
... 73%, 13152 KB, 20340 KB/s, 0 seconds passed
... 73%, 13184 KB, 20382 KB/s, 0 seconds passed
... 73%, 13216 KB, 20426 KB/s, 0 seconds passed
... 73%, 13248 KB, 20470 KB/s, 0 seconds passed
... 73%, 13280 KB, 20514 KB/s, 0 seconds passed
... 73%, 13312 KB, 20558 KB/s, 0 seconds passed
... 74%, 13344 KB, 20602 KB/s, 0 seconds passed
... 74%, 13376 KB, 20646 KB/s, 0 seconds passed
... 74%, 13408 KB, 20690 KB/s, 0 seconds passed
... 74%, 13440 KB, 20734 KB/s, 0 seconds passed
... 74%, 13472 KB, 20778 KB/s, 0 seconds passed
... 75%, 13504 KB, 20822 KB/s, 0 seconds passed
... 75%, 13536 KB, 20866 KB/s, 0 seconds passed
... 75%, 13568 KB, 20910 KB/s, 0 seconds passed
... 75%, 13600 KB, 20954 KB/s, 0 seconds passed
... 75%, 13632 KB, 20997 KB/s, 0 seconds passed
... 75%, 13664 KB, 21041 KB/s, 0 seconds passed
... 76%, 13696 KB, 21085 KB/s, 0 seconds passed
... 76%, 13728 KB, 21129 KB/s, 0 seconds passed
... 76%, 13760 KB, 21172 KB/s, 0 seconds passed
... 76%, 13792 KB, 21216 KB/s, 0 seconds passed
... 76%, 13824 KB, 21260 KB/s, 0 seconds passed
... 77%, 13856 KB, 21303 KB/s, 0 seconds passed
... 77%, 13888 KB, 21347 KB/s, 0 seconds passed
... 77%, 13920 KB, 21391 KB/s, 0 seconds passed
... 77%, 13952 KB, 21435 KB/s, 0 seconds passed
... 77%, 13984 KB, 21479 KB/s, 0 seconds passed
... 77%, 14016 KB, 21523 KB/s, 0 seconds passed
... 78%, 14048 KB, 21566 KB/s, 0 seconds passed
... 78%, 14080 KB, 21610 KB/s, 0 seconds passed
... 78%, 14112 KB, 21654 KB/s, 0 seconds passed
... 78%, 14144 KB, 21696 KB/s, 0 seconds passed
... 78%, 14176 KB, 21736 KB/s, 0 seconds passed
... 78%, 14208 KB, 21777 KB/s, 0 seconds passed
... 79%, 14240 KB, 21815 KB/s, 0 seconds passed
... 79%, 14272 KB, 21855 KB/s, 0 seconds passed
... 79%, 14304 KB, 21896 KB/s, 0 seconds passed
... 79%, 14336 KB, 21936 KB/s, 0 seconds passed
... 79%, 14368 KB, 21973 KB/s, 0 seconds passed
... 80%, 14400 KB, 22014 KB/s, 0 seconds passed
... 80%, 14432 KB, 22054 KB/s, 0 seconds passed
... 80%, 14464 KB, 22092 KB/s, 0 seconds passed
... 80%, 14496 KB, 22132 KB/s, 0 seconds passed
... 80%, 14528 KB, 22172 KB/s, 0 seconds passed
... 80%, 14560 KB, 22210 KB/s, 0 seconds passed
... 81%, 14592 KB, 22250 KB/s, 0 seconds passed
... 81%, 14624 KB, 22290 KB/s, 0 seconds passed
... 81%, 14656 KB, 22328 KB/s, 0 seconds passed
... 81%, 14688 KB, 22368 KB/s, 0 seconds passed
... 81%, 14720 KB, 22408 KB/s, 0 seconds passed
... 82%, 14752 KB, 22446 KB/s, 0 seconds passed
... 82%, 14784 KB, 22485 KB/s, 0 seconds passed
... 82%, 14816 KB, 22525 KB/s, 0 seconds passed
... 82%, 14848 KB, 22565 KB/s, 0 seconds passed
... 82%, 14880 KB, 22603 KB/s, 0 seconds passed
... 82%, 14912 KB, 22642 KB/s, 0 seconds passed
... 83%, 14944 KB, 22682 KB/s, 0 seconds passed
... 83%, 14976 KB, 22720 KB/s, 0 seconds passed
... 83%, 15008 KB, 22759 KB/s, 0 seconds passed
... 83%, 15040 KB, 22798 KB/s, 0 seconds passed
... 83%, 15072 KB, 22836 KB/s, 0 seconds passed
... 83%, 15104 KB, 22875 KB/s, 0 seconds passed
... 84%, 15136 KB, 22915 KB/s, 0 seconds passed
... 84%, 15168 KB, 22952 KB/s, 0 seconds passed
... 84%, 15200 KB, 22992 KB/s, 0 seconds passed
... 84%, 15232 KB, 23031 KB/s, 0 seconds passed
... 84%, 15264 KB, 23068 KB/s, 0 seconds passed
... 85%, 15296 KB, 23107 KB/s, 0 seconds passed
... 85%, 15328 KB, 23147 KB/s, 0 seconds passed
... 85%, 15360 KB, 23186 KB/s, 0 seconds passed
... 85%, 15392 KB, 23223 KB/s, 0 seconds passed
... 85%, 15424 KB, 23262 KB/s, 0 seconds passed
... 85%, 15456 KB, 23300 KB/s, 0 seconds passed
... 86%, 15488 KB, 23339 KB/s, 0 seconds passed
... 86%, 15520 KB, 23378 KB/s, 0 seconds passed
... 86%, 15552 KB, 23415 KB/s, 0 seconds passed
... 86%, 15584 KB, 23449 KB/s, 0 seconds passed
... 86%, 15616 KB, 23488 KB/s, 0 seconds passed
... 86%, 15648 KB, 23531 KB/s, 0 seconds passed
... 87%, 15680 KB, 23568 KB/s, 0 seconds passed
... 87%, 15712 KB, 23607 KB/s, 0 seconds passed
... 87%, 15744 KB, 23645 KB/s, 0 seconds passed
... 87%, 15776 KB, 23682 KB/s, 0 seconds passed
... 87%, 15808 KB, 23721 KB/s, 0 seconds passed
... 88%, 15840 KB, 23759 KB/s, 0 seconds passed
... 88%, 15872 KB, 23798 KB/s, 0 seconds passed
... 88%, 15904 KB, 23835 KB/s, 0 seconds passed
... 88%, 15936 KB, 23873 KB/s, 0 seconds passed
... 88%, 15968 KB, 23912 KB/s, 0 seconds passed
... 88%, 16000 KB, 23950 KB/s, 0 seconds passed
... 89%, 16032 KB, 23989 KB/s, 0 seconds passed
... 89%, 16064 KB, 24025 KB/s, 0 seconds passed
... 89%, 16096 KB, 24064 KB/s, 0 seconds passed
... 89%, 16128 KB, 24102 KB/s, 0 seconds passed
... 89%, 16160 KB, 24139 KB/s, 0 seconds passed
... 90%, 16192 KB, 24177 KB/s, 0 seconds passed
... 90%, 16224 KB, 24215 KB/s, 0 seconds passed
... 90%, 16256 KB, 24253 KB/s, 0 seconds passed
... 90%, 16288 KB, 24291 KB/s, 0 seconds passed
... 90%, 16320 KB, 24328 KB/s, 0 seconds passed
... 90%, 16352 KB, 24366 KB/s, 0 seconds passed
... 91%, 16384 KB, 24404 KB/s, 0 seconds passed
... 91%, 16416 KB, 24440 KB/s, 0 seconds passed
... 91%, 16448 KB, 24479 KB/s, 0 seconds passed
... 91%, 16480 KB, 24517 KB/s, 0 seconds passed
... 91%, 16512 KB, 24555 KB/s, 0 seconds passed
... 91%, 16544 KB, 24591 KB/s, 0 seconds passed
... 92%, 16576 KB, 24629 KB/s, 0 seconds passed
... 92%, 16608 KB, 24667 KB/s, 0 seconds passed
... 92%, 16640 KB, 24703 KB/s, 0 seconds passed
... 92%, 16672 KB, 24736 KB/s, 0 seconds passed
... 92%, 16704 KB, 24769 KB/s, 0 seconds passed
... 93%, 16736 KB, 24802 KB/s, 0 seconds passed
... 93%, 16768 KB, 24845 KB/s, 0 seconds passed
... 93%, 16800 KB, 24887 KB/s, 0 seconds passed
... 93%, 16832 KB, 24921 KB/s, 0 seconds passed
... 93%, 16864 KB, 24954 KB/s, 0 seconds passed
... 93%, 16896 KB, 24993 KB/s, 0 seconds passed
... 94%, 16928 KB, 25035 KB/s, 0 seconds passed
... 94%, 16960 KB, 25071 KB/s, 0 seconds passed
... 94%, 16992 KB, 25109 KB/s, 0 seconds passed
... 94%, 17024 KB, 25147 KB/s, 0 seconds passed
... 94%, 17056 KB, 25182 KB/s, 0 seconds passed
... 94%, 17088 KB, 25220 KB/s, 0 seconds passed
... 95%, 17120 KB, 25257 KB/s, 0 seconds passed
... 95%, 17152 KB, 25293 KB/s, 0 seconds passed
... 95%, 17184 KB, 25330 KB/s, 0 seconds passed
... 95%, 17216 KB, 25367 KB/s, 0 seconds passed
... 95%, 17248 KB, 25405 KB/s, 0 seconds passed
... 96%, 17280 KB, 25440 KB/s, 0 seconds passed

.. parsed-literal::

    ... 96%, 17312 KB, 25477 KB/s, 0 seconds passed
... 96%, 17344 KB, 25513 KB/s, 0 seconds passed
... 96%, 17376 KB, 25550 KB/s, 0 seconds passed
... 96%, 17408 KB, 25587 KB/s, 0 seconds passed
... 96%, 17440 KB, 25622 KB/s, 0 seconds passed
... 97%, 17472 KB, 25659 KB/s, 0 seconds passed
... 97%, 17504 KB, 25696 KB/s, 0 seconds passed
... 97%, 17536 KB, 25734 KB/s, 0 seconds passed
... 97%, 17568 KB, 25768 KB/s, 0 seconds passed
... 97%, 17600 KB, 25806 KB/s, 0 seconds passed
... 98%, 17632 KB, 25841 KB/s, 0 seconds passed
... 98%, 17664 KB, 25878 KB/s, 0 seconds passed
... 98%, 17696 KB, 25915 KB/s, 0 seconds passed
... 98%, 17728 KB, 25950 KB/s, 0 seconds passed
... 98%, 17760 KB, 25987 KB/s, 0 seconds passed
... 98%, 17792 KB, 26023 KB/s, 0 seconds passed
... 99%, 17824 KB, 26060 KB/s, 0 seconds passed
... 99%, 17856 KB, 26097 KB/s, 0 seconds passed
... 99%, 17888 KB, 26132 KB/s, 0 seconds passed
... 99%, 17920 KB, 26167 KB/s, 0 seconds passed
... 99%, 17952 KB, 26203 KB/s, 0 seconds passed
... 99%, 17984 KB, 26242 KB/s, 0 seconds passed
... 100%, 17990 KB, 26245 KB/s, 0 seconds passed

    
    ========== Unpacking model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz


.. parsed-literal::

    


Convert Model to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The selected model comes from the public directory, which means it must
be converted into OpenVINO Intermediate Representation (OpenVINO IR). We
use ``omz_converter`` to convert the ONNX format model to the OpenVINO
IR format.

.. code:: ipython3

    if not onnx_path.exists():
        convert_command = (
            f"omz_converter "
            f"--name {model_name} "
            f"--precisions {precision} "
            f"--download_dir {base_model_dir} "
            f"--output_dir {base_model_dir}"
        )
        ! $convert_command


.. parsed-literal::

    ========== Converting human-pose-estimation-3d-0001 to ONNX
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-path=model/public/human-pose-estimation-3d-0001 --model-name=PoseEstimationWithMobileNet --model-param=is_convertible_by_mo=True --import-module=model --weights=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.pth --input-shape=1,3,256,448 --input-names=data --output-names=features,heatmaps,pafs --output-file=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx
    


.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::

    
    ========== Converting human-pose-estimation-3d-0001 to IR (FP32)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/human-pose-estimation-3d-0001/FP32 --model_name=human-pose-estimation-3d-0001 --input=data '--mean_values=data[128.0,128.0,128.0]' '--scale_values=data[255.0,255.0,255.0]' --output=features,heatmaps,pafs --input_model=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 256, 448]' --compress_to_fp16=False
    


.. parsed-literal::

    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API. 
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


.. parsed-literal::

    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/notebooks/3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-644/.workspace/scm/ov-notebook/notebooks/3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin


.. parsed-literal::

    


Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

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



Load the model
~~~~~~~~~~~~~~



Converted models are located in a fixed structure, which indicates
vendor, model name and precision.

First, initialize the inference engine, OpenVINO Runtime. Then, read the
network architecture and model weights from the ``.bin`` and ``.xml``
files to compile for the desired device. An inference request is then
created to infer the compiled model.

.. code:: ipython3

    # initialize inference engine
    core = ov.Core()
    # read the network and corresponding weights from file
    model = core.read_model(model=ir_model_path, weights=model_weights_path)
    # load the model on the specified device
    compiled_model = core.compile_model(model=model, device_name=device.value)
    infer_request = compiled_model.create_infer_request()
    input_tensor_name = model.inputs[0].get_any_name()
    
    # get input and output names of nodes
    input_layer = compiled_model.input(0)
    output_layers = list(compiled_model.outputs)

The input for the model is data from the input image and the outputs are
heat maps, PAF (part affinity fields) and features.

.. code:: ipython3

    input_layer.any_name, [o.any_name for o in output_layers]




.. parsed-literal::

    ('data', ['features', 'heatmaps', 'pafs'])



Processing
----------



Model Inference
~~~~~~~~~~~~~~~



Frames captured from video files or the live webcam are used as the
input for the 3D model. This is how you obtain the output heat maps, PAF
(part affinity fields) and features.

.. code:: ipython3

    def model_infer(scaled_img, stride):
        """
        Run model inference on the input image
    
        Parameters:
            scaled_img: resized image according to the input size of the model
            stride: int, the stride of the window
        """
    
        # Remove excess space from the picture
        img = scaled_img[
            0 : scaled_img.shape[0] - (scaled_img.shape[0] % stride),
            0 : scaled_img.shape[1] - (scaled_img.shape[1] % stride),
        ]
    
        img = np.transpose(img, (2, 0, 1))[
            None,
        ]
        infer_request.infer({input_tensor_name: img})
        # A set of three inference results is obtained
        results = {
            name: infer_request.get_tensor(name).data[:]
            for name in {"features", "heatmaps", "pafs"}
        }
        # Get the results
        results = (results["features"][0], results["heatmaps"][0], results["pafs"][0])
    
        return results

Draw 2D Pose Overlays
~~~~~~~~~~~~~~~~~~~~~



We need to define some connections between the joints in advance, so
that we can draw the structure of the human body in the resulting image
after obtaining the inference results. Joints are drawn as circles and
limbs are drawn as lines. The code is based on the `3D Human Pose
Estimation
Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/human_pose_estimation_3d_demo/python>`__
from Open Model Zoo.

.. code:: ipython3

    # 3D edge index array
    body_edges = np.array(
        [
            [0, 1], 
            [0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
            [0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
            [1, 15], [15, 16],            # nose - l_eye - l_ear
            [1, 17], [17, 18],            # nose - r_eye - r_ear
            [0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
            [0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
        ]
    )
    
    
    body_edges_2d = np.array(
        [
            [0, 1],                       # neck - nose
            [1, 16], [16, 18],            # nose - l_eye - l_ear
            [1, 15], [15, 17],            # nose - r_eye - r_ear
            [0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
            [0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
            [0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
            [0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
        ]
    )  
    
    
    def draw_poses(frame, poses_2d, scaled_img, use_popup):
        """
        Draw 2D pose overlays on the image to visualize estimated poses.
        Joints are drawn as circles and limbs are drawn as lines.
    
        :param frame: the input image
        :param poses_2d: array of human joint pairs
        """
        for pose in poses_2d:
            pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
            was_found = pose[2] > 0
    
            pose[0], pose[1] = (
                pose[0] * frame.shape[1] / scaled_img.shape[1],
                pose[1] * frame.shape[0] / scaled_img.shape[0],
            )
    
            # Draw joints.
            for edge in body_edges_2d:
                if was_found[edge[0]] and was_found[edge[1]]:
                    cv2.line(
                        frame,
                        tuple(pose[0:2, edge[0]].astype(np.int32)),
                        tuple(pose[0:2, edge[1]].astype(np.int32)),
                        (255, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
            # Draw limbs.
            for kpt_id in range(pose.shape[1]):
                if pose[2, kpt_id] != -1:
                    cv2.circle(
                        frame,
                        tuple(pose[0:2, kpt_id].astype(np.int32)),
                        3,
                        (0, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )
    
        return frame

Main Processing Function
~~~~~~~~~~~~~~~~~~~~~~~~



Run 3D pose estimation on the specified source. It could be either a
webcam feed or a video file.

.. code:: ipython3

    def run_pose_estimation(source=0, flip=False, use_popup=False, skip_frames=0):
        """
        2D image as input, using OpenVINO as inference backend,
        get joints 3D coordinates, and draw 3D human skeleton in the scene
    
        :param source:      The webcam number to feed the video stream with primary webcam set to "0", or the video path.
        :param flip:        To be used by VideoPlayer function for flipping capture image.
        :param use_popup:   False for showing encoded frames over this notebook, True for creating a popup window.
        :param skip_frames: Number of frames to skip at the beginning of the video.
        """
    
        focal_length = -1  # default
        stride = 8
        player = None
        skeleton_set = None
    
        try:
            # create video player to play with target fps  video_path
            # get the frame from camera
            # You can skip first N frames to fast forward video. change 'skip_first_frames'
            player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_frames)
            # start capturing
            player.start()
    
            input_image = player.next()
            # set the window size
            resize_scale = 450 / input_image.shape[1]
            windows_width = int(input_image.shape[1] * resize_scale)
            windows_height = int(input_image.shape[0] * resize_scale)
    
            # use visualization library
            engine3D = engine.Engine3js(grid=True, axis=True, view_width=windows_width, view_height=windows_height)
    
            if use_popup:
                # display the 3D human pose in this notebook, and origin frame in popup window
                display(engine3D.renderer)
                title = "Press ESC to Exit"
                cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE)
            else:
                # set the 2D image box, show both human pose and image in the notebook
                imgbox = widgets.Image(
                    format="jpg", height=windows_height, width=windows_width
                )
                display(widgets.HBox([engine3D.renderer, imgbox]))
    
            skeleton = engine.Skeleton(body_edges=body_edges)
    
            processing_times = collections.deque()
    
            while True:
                # grab the frame
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break
    
                # resize image and change dims to fit neural network input
                # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001)
                scaled_img = cv2.resize(frame, dsize=(model.inputs[0].shape[3], model.inputs[0].shape[2]))
    
                if focal_length < 0:  # Focal length is unknown
                    focal_length = np.float32(0.8 * scaled_img.shape[1])
    
                # inference start
                start_time = time.time()
                # get results
                inference_result = model_infer(scaled_img, stride)
    
                # inference stop
                stop_time = time.time()
                processing_times.append(stop_time - start_time)
                # Process the point to point coordinates of the data
                poses_3d, poses_2d = parse_poses(inference_result, 1, stride, focal_length, True)
    
                # use processing times from last 200 frames
                if len(processing_times) > 200:
                    processing_times.popleft()
    
                processing_time = np.mean(processing_times) * 1000
                fps = 1000 / processing_time
    
                if len(poses_3d) > 0:
                    # From here, you can rotate the 3D point positions using the function "draw_poses",
                    # or you can directly make the correct mapping below to properly display the object image on the screen
                    poses_3d_copy = poses_3d.copy()
                    x = poses_3d_copy[:, 0::4]
                    y = poses_3d_copy[:, 1::4]
                    z = poses_3d_copy[:, 2::4]
                    poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = (
                        -z + np.ones(poses_3d[:, 2::4].shape) * 200,
                        -y + np.ones(poses_3d[:, 2::4].shape) * 100,
                        -x,
                    )
    
                    poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
                    people = skeleton(poses_3d=poses_3d)
    
                    try:
                        engine3D.scene_remove(skeleton_set)
                    except Exception:
                        pass
    
                    engine3D.scene_add(people)
                    skeleton_set = people
    
                    # draw 2D
                    frame = draw_poses(frame, poses_2d, scaled_img, use_popup)
    
                else:
                    try:
                        engine3D.scene_remove(skeleton_set)
                        skeleton_set = None
                    except Exception:
                        pass
    
                cv2.putText(
                    frame,
                    f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
    
                if use_popup:
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(1)
                    # escape = 27, use ESC to exit
                    if key == 27:
                        break
                else:
                    # encode numpy array to jpg
                    imgbox.value = cv2.imencode(
                        ".jpg",
                        frame,
                        params=[cv2.IMWRITE_JPEG_QUALITY, 90],
                    )[1].tobytes()
    
                engine3D.renderer.render(engine3D.scene, engine3D.cam)
    
        except KeyboardInterrupt:
            print("Interrupted")
        except RuntimeError as e:
            print(e)
        finally:
            clear_output()
            if player is not None:
                # stop capturing
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()
            if skeleton_set:
                engine3D.scene_remove(skeleton_set)

Run
---



Run, using a webcam as the video input. By default, the primary webcam
is set with ``source=0``. If you have multiple webcams, each one will be
assigned a consecutive number starting at 0. Set ``flip=True`` when
using a front-facing camera. Some web browsers, especially Mozilla
Firefox, may cause flickering. If you experience flickering, set
``use_popup=True``.

   **NOTE**:

   *1. To use this notebook with a webcam, you need to run the notebook
   on a computer with a webcam. If you run the notebook on a server
   (e.g. Binder), the webcam will not work.*

   *2. Popup mode may not work if you run this notebook on a remote
   computer (e.g. Binder).*

If you do not have a webcam, you can still run this demo with a video
file. Any `format supported by
OpenCV <http/docs.opencv.o4.5dtutorial_py_video_display.html>`__
will work.

Using the following method, you can click and move your mouse over the
picture on the left to interact.

.. code:: ipython3

    USE_WEBCAM = False
    
    cam_id = 0
    video_path = "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"
    
    source = cam_id if USE_WEBCAM else video_path
    
    run_pose_estimation(source=source, flip=isinstance(source, int), use_popup=False)
