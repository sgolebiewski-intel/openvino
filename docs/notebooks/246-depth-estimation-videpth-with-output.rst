Monocular Visual-Inertial Depth Estimation using OpenVINO™
==========================================================

.. raw:: html

   <p align="center" width="100%">

.. raw:: html

   <figcaption>

The overall methodology. Diagram taken from the VI-Depth repository.

.. raw:: html

   </figcaption>

.. raw:: html

   </p>

A visual-inertial depth estimation pipeline that integrates monocular
depth estimation and visual-inertial odometry to produce dense depth
estimates with metric scale has been presented by the authors. The
approach consists of three stages:

1. input processing, where RGB and inertial measurement unit (IMU) data
   feed into monocular depth estimation alongside visual-inertial
   odometry,
2. global scale and shift alignment, where monocular depth estimates are
   fitted to sparse depth from visual inertial odometry (VIO) in a
   least-squares manner and
3. learning-based dense scale alignment, where globally-aligned depth is
   locally realigned using a dense scale map regressed by the
   ScaleMapLearner (SML).

The images at the bottom in the diagram above illustrate a Visual
Odometry with Inertial and Depth (VOID) sample being processed through
our pipeline; from left to right: the input RGB, ground truth depth,
sparse depth from VIO, globally-aligned depth, scale map scaffolding,
dense scale map regressed by SML, final depth output.

.. raw:: html

   <p align="center" width="100%">

.. raw:: html

   <figcaption>

An illustration of VOID samples being processed by the image pipeline.
Image taken from the VI-Depth repository.

.. raw:: html

   </figcaption>

.. raw:: html

   </p>

We will be consulting the `VI-Depth
repository <https://github.com/isl-org/VI-Depth>`__ for the
pre-processing, model transformations and basic utility code. A part of
it has already been kept as it is in the `utils <utils>`__ directory. At
the same time we will learn how to perform `model
conversion <https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-pytorch.html>`__
for converting a model in a different format to the standard OpenVINO™
IR model representation *via* another format.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports <#Imports>`__
-  `Loading models and checkpoints <#Loading-models-and-checkpoints>`__

   -  `Cleaning up the model
      directory <#Cleaning-up-the-model-directory>`__
   -  `Transformation of models <#Transformation-of-models>`__

      -  `Dummy input creation <#Dummy-input-creation>`__
      -  `Conversion of depth model to OpenVINO IR
         format <#Conversion-of-depth-model-to-OpenVINO-IR-format>`__

         -  `Select inference device <#Select-inference-device>`__
         -  `Compilation of depth model <#Compilation-of-depth-model>`__
         -  `Computation of scale and shift
            parameters <#Computation-of-scale-and-shift-parameters>`__

      -  `Conversion of Scale Map Learner model to OpenVINO IR
         format <#Conversion-of-Scale-Map-Learner-model-to-OpenVINO-IR-format>`__

         -  `Select inference device <#Select-inference-device>`__
         -  `Compilation of the ScaleMapLearner(SML)
            model <#Compilation-of-the-ScaleMapLearner(SML)-model>`__

      -  `Storing and visualizing dummy results
         obtained <#Storing-and-visualizing-dummy-results-obtained>`__

   -  `Running inference on a test
      image <#Running-inference-on-a-test-image>`__
   -  `Store and visualize Inference
      results <#Store-and-visualize-Inference-results>`__

      -  `Cleaning up the data
         directory <#Cleaning-up-the-data-directory>`__

   -  `Concluding notes <#Concluding-notes>`__

Imports
~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    # Import sys beforehand to inform of Python version <= 3.7 not being supported
    import sys
    
    if sys.version_info.minor < 8:
        print('Python3.7 is not supported. Some features might not work as expected')
        
    # Download the correct version of the PyTorch deep learning library associated with image models
    # alongside the lightning module
    %pip install -q "openvino>=2024.0.0" --extra-index-url https://download.pytorch.org/whl/cpu "pytorch-lightning" "timm>=0.6.12"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    googleapis-common-protos 1.63.0 requires protobuf!=3.20.0,!=3.20.1,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0.dev0,>=3.19.5, but you have protobuf 3.20.1 which is incompatible.
    onnx 1.15.0 requires protobuf>=3.20.2, but you have protobuf 3.20.1 which is incompatible.
    paddlepaddle 2.6.0 requires protobuf>=3.20.2; platform_system != "Windows", but you have protobuf 3.20.1 which is incompatible.
    tensorflow 2.12.0 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3, but you have protobuf 3.20.1 which is incompatible.
    tensorflow-metadata 1.14.0 requires protobuf<4.21,>=3.20.3, but you have protobuf 3.20.1 which is incompatible.
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import openvino as ov
    import torch
    import torchvision
    from pathlib import Path
    from shutil import rmtree
    from typing import Optional, Tuple
    
    sys.path.append('../utils')
    from notebook_utils import download_file
    
    sys.path.append('vi_depth_utils')
    import data_loader
    import modules.midas.transforms as transforms
    import modules.midas.utils as utils
    from modules.estimator import LeastSquaresEstimator
    from modules.interpolator import Interpolator2D
    from modules.midas.midas_net_custom import MidasNet_small_videpth

.. code:: ipython3

    # Ability to display images inline
    %matplotlib inline

Loading models and checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

The complete pipeline here requires only two models: one for depth
estimation and a ScaleMapLearner model which is responsible for
regressing a dense scale map. The table of models which has been given
in the original `VI-Depth repo <https://github.com/isl-org/VI-Depth>`__
has been presented as it is for the users to download from.
`VOID <https://github.com/alexklwong/void-dataset>`__ is the name of the
original dataset from on which these models have been trained. The
numbers after the word **VOID** represent the checkpoint in the model
obtained after training samples for sparse dense maps corresponding to
:math:`150`, :math:`500` and :math:`1500` levels in the density map.
Just *right-click* on any of the highlighted links and click on “Copy
link address”. We shall use this link in the next cell to download the
ScaleMapLearner model. *Interestingly*, the ScaleMapLearner decides the
depth prediction model as you will see.

================
===============================================================================================================================
===============================================================================================================================
================================================================================================================================
Depth Predictor  SML on VOID 150                                                                                                                 SML on VOID 500                                                                                                                 SML on VOID 1500
================
===============================================================================================================================
===============================================================================================================================
================================================================================================================================
DPT-BEiT-Large   `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_beit_large_512.nsamples.150.ckpt>`__  `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_beit_large_512.nsamples.500.ckpt>`__  `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_beit_large_512.nsamples.1500.ckpt>`__
DPT-SwinV2-Large `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_swin2_large_384.nsamples.150.ckpt>`__ `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_swin2_large_384.nsamples.500.ckpt>`__ `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_swin2_large_384.nsamples.1500.ckpt>`__
DPT-Large        `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_large.nsamples.150.ckpt>`__           `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_large.nsamples.500.ckpt>`__           `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_large.nsamples.1500.ckpt>`__
DPT-Hybrid       `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt>`__\ \*      `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_hybrid.nsamples.500.ckpt>`__          `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_hybrid.nsamples.1500.ckpt>`__
DPT-SwinV2-Tiny  `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_swin2_tiny_256.nsamples.150.ckpt>`__  `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_swin2_tiny_256.nsamples.500.ckpt>`__  `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_swin2_tiny_256.nsamples.1500.ckpt>`__
DPT-LeViT        `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_levit_224.nsamples.150.ckpt>`__       `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_levit_224.nsamples.500.ckpt>`__       `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_levit_224.nsamples.1500.ckpt>`__
MiDaS-small      `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.midas_small.nsamples.150.ckpt>`__         `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.midas_small.nsamples.500.ckpt>`__         `model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.midas_small.nsamples.1500.ckpt>`__
================
===============================================================================================================================
===============================================================================================================================
================================================================================================================================

\*Also available with pre-training on TartanAir:
`model <https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.dpt_hybrid.nsamples.150.pretrained.ckpt>`__

.. code:: ipython3

    # Base directory in which models would be stored as a pathlib.Path variable
    MODEL_DIR = Path('model')
    
    # Mapping between depth predictors and the corresponding scale map learners
    PREDICTOR_MODEL_MAP = {'dpt_beit_large_512': 'DPT_BEiT_L_512',
                           'dpt_swin2_large_384': 'DPT_SwinV2_L_384',
                           'dpt_large': 'DPT_Large',
                           'dpt_hybrid': 'DPT_Hybrid',
                           'dpt_swin2_tiny_256': 'DPT_SwinV2_T_256',
                           'dpt_levit_224': 'DPT_LeViT_224',
                           'midas_small': 'MiDaS_small'}

.. code:: ipython3

    # Create the model directory adjacent to the notebook and suppress errors if the directory already exists
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Here we will be downloading the SML model corresponding to the MiDaS-small depth predictor for 
    # the checkpoint captured after training on 1500 points of the density level. Suppress errors if the file already exists
    download_file('https://github.com/isl-org/VI-Depth/releases/download/v1/sml_model.dpredictor.midas_small.nsamples.1500.ckpt', directory=MODEL_DIR, silent=True)
    
    # Take a note of the samples. It would be of major use later on
    NSAMPLES = 1500



.. parsed-literal::

    model/sml_model.dpredictor.midas_small.nsamples.1500.ckpt:   0%|          | 0.00/208M [00:00<?, ?B/s]


.. code:: ipython3

    # Set the same model directory for downloading the depth predictor model which is available on
    # PyTorch hub
    torch.hub.set_dir(str(MODEL_DIR))
    
    
    # A utility function for utilising the mapping between depth predictors and 
    # scale map learners so as to download the former
    def get_model_for_predictor(depth_predictor: str, remote_repo: str = 'intel-isl/MiDaS') -> str:    
        """
        Download a model from the pre-validated 'isl-org/MiDaS:2.1' set of releases on the GitHub repo
        while simultaneously trusting the repo permanently
    
        :param: depth_predictor: Any depth estimation model amongst the ones given at https://github.com/isl-org/VI-Depth#setup
        :param: remote_repo: The remote GitHub repo from where the models will be downloaded
        :returns: A PyTorch model callable
        """    
        
        # Workaround for avoiding rate limit errors
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        
        return torch.hub.load(remote_repo, PREDICTOR_MODEL_MAP[depth_predictor], skip_validation=True, trust_repo=True)

.. code:: ipython3

    # Execute the above function so as to download the MiDaS-small model
    # and get the output of the model callable in return
    depth_model = get_model_for_predictor('midas_small')


.. parsed-literal::

    Downloading: "https://github.com/intel-isl/MiDaS/zipball/master" to model/master.zip


.. parsed-literal::

    Loading weights:  None


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-633/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/hub.py:294: UserWarning: You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to {calling_fn}(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or load(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use load(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour
      warnings.warn(
    Downloading: "https://github.com/rwightman/gen-efficientnet-pytorch/zipball/master" to model/master.zip


.. parsed-literal::

    Downloading: "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite3-b733e338.pth" to model/checkpoints/tf_efficientnet_lite3-b733e338.pth


.. parsed-literal::

    Downloading: "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt" to model/checkpoints/midas_v21_small_256.pt


.. parsed-literal::

      0%|          | 0.00/81.8M [00:00<?, ?B/s]

.. parsed-literal::

      0%|          | 288k/81.8M [00:00<00:29, 2.87MB/s]

.. parsed-literal::

      1%|          | 704k/81.8M [00:00<00:23, 3.64MB/s]

.. parsed-literal::

      1%|▏         | 1.04M/81.8M [00:00<00:23, 3.65MB/s]

.. parsed-literal::

      2%|▏         | 1.39M/81.8M [00:00<00:23, 3.59MB/s]

.. parsed-literal::

      2%|▏         | 1.73M/81.8M [00:00<00:25, 3.33MB/s]

.. parsed-literal::

      3%|▎         | 2.11M/81.8M [00:00<00:23, 3.51MB/s]

.. parsed-literal::

      3%|▎         | 2.45M/81.8M [00:00<00:24, 3.42MB/s]

.. parsed-literal::

      3%|▎         | 2.81M/81.8M [00:00<00:23, 3.50MB/s]

.. parsed-literal::

      4%|▍         | 3.17M/81.8M [00:00<00:23, 3.54MB/s]

.. parsed-literal::

      4%|▍         | 3.53M/81.8M [00:01<00:22, 3.57MB/s]

.. parsed-literal::

      5%|▍         | 3.89M/81.8M [00:01<00:22, 3.60MB/s]

.. parsed-literal::

      5%|▌         | 4.25M/81.8M [00:01<00:22, 3.59MB/s]

.. parsed-literal::

      6%|▌         | 4.61M/81.8M [00:01<00:22, 3.63MB/s]

.. parsed-literal::

      6%|▌         | 4.97M/81.8M [00:01<00:22, 3.61MB/s]

.. parsed-literal::

      7%|▋         | 5.34M/81.8M [00:01<00:21, 3.69MB/s]

.. parsed-literal::

      7%|▋         | 5.70M/81.8M [00:01<00:21, 3.70MB/s]

.. parsed-literal::

      7%|▋         | 6.06M/81.8M [00:01<00:22, 3.59MB/s]

.. parsed-literal::

      8%|▊         | 6.42M/81.8M [00:01<00:21, 3.64MB/s]

.. parsed-literal::

      8%|▊         | 6.78M/81.8M [00:01<00:21, 3.67MB/s]

.. parsed-literal::

      9%|▊         | 7.13M/81.8M [00:02<00:21, 3.66MB/s]

.. parsed-literal::

      9%|▉         | 7.52M/81.8M [00:02<00:21, 3.60MB/s]

.. parsed-literal::

     10%|▉         | 7.87M/81.8M [00:02<00:22, 3.45MB/s]

.. parsed-literal::

     10%|█         | 8.23M/81.8M [00:02<00:21, 3.56MB/s]

.. parsed-literal::

     10%|█         | 8.58M/81.8M [00:02<00:23, 3.26MB/s]

.. parsed-literal::

     11%|█         | 8.90M/81.8M [00:02<00:24, 3.18MB/s]

.. parsed-literal::

     11%|█▏        | 9.21M/81.8M [00:02<00:24, 3.09MB/s]

.. parsed-literal::

     12%|█▏        | 9.51M/81.8M [00:02<00:25, 3.02MB/s]

.. parsed-literal::

     12%|█▏        | 9.80M/81.8M [00:02<00:24, 3.05MB/s]

.. parsed-literal::

     12%|█▏        | 10.1M/81.8M [00:03<00:25, 2.93MB/s]

.. parsed-literal::

     13%|█▎        | 10.4M/81.8M [00:03<00:24, 3.03MB/s]

.. parsed-literal::

     13%|█▎        | 10.7M/81.8M [00:03<00:24, 3.07MB/s]

.. parsed-literal::

     13%|█▎        | 11.0M/81.8M [00:03<00:24, 3.08MB/s]

.. parsed-literal::

     14%|█▍        | 11.3M/81.8M [00:03<00:23, 3.13MB/s]

.. parsed-literal::

     14%|█▍        | 11.7M/81.8M [00:03<00:23, 3.18MB/s]

.. parsed-literal::

     15%|█▍        | 12.0M/81.8M [00:03<00:22, 3.22MB/s]

.. parsed-literal::

     15%|█▌        | 12.3M/81.8M [00:03<00:22, 3.24MB/s]

.. parsed-literal::

     15%|█▌        | 12.6M/81.8M [00:03<00:26, 2.71MB/s]

.. parsed-literal::

     16%|█▌        | 13.1M/81.8M [00:04<00:22, 3.19MB/s]

.. parsed-literal::

     16%|█▋        | 13.4M/81.8M [00:04<00:24, 2.97MB/s]

.. parsed-literal::

     17%|█▋        | 13.7M/81.8M [00:04<00:25, 2.77MB/s]

.. parsed-literal::

     17%|█▋        | 14.0M/81.8M [00:04<00:26, 2.70MB/s]

.. parsed-literal::

     17%|█▋        | 14.3M/81.8M [00:04<00:26, 2.66MB/s]

.. parsed-literal::

     18%|█▊        | 14.5M/81.8M [00:04<00:27, 2.61MB/s]

.. parsed-literal::

     18%|█▊        | 14.8M/81.8M [00:04<00:26, 2.63MB/s]

.. parsed-literal::

     18%|█▊        | 15.0M/81.8M [00:04<00:27, 2.58MB/s]

.. parsed-literal::

     19%|█▊        | 15.3M/81.8M [00:05<00:27, 2.57MB/s]

.. parsed-literal::

     19%|█▉        | 15.5M/81.8M [00:05<00:26, 2.60MB/s]

.. parsed-literal::

     19%|█▉        | 15.8M/81.8M [00:05<00:26, 2.62MB/s]

.. parsed-literal::

     20%|█▉        | 16.1M/81.8M [00:05<00:26, 2.61MB/s]

.. parsed-literal::

     20%|█▉        | 16.3M/81.8M [00:05<00:26, 2.56MB/s]

.. parsed-literal::

     20%|██        | 16.6M/81.8M [00:05<00:25, 2.71MB/s]

.. parsed-literal::

     21%|██        | 16.9M/81.8M [00:05<00:25, 2.72MB/s]

.. parsed-literal::

     21%|██        | 17.2M/81.8M [00:05<00:24, 2.74MB/s]

.. parsed-literal::

     21%|██▏       | 17.4M/81.8M [00:05<00:24, 2.72MB/s]

.. parsed-literal::

     22%|██▏       | 17.7M/81.8M [00:05<00:24, 2.74MB/s]

.. parsed-literal::

     22%|██▏       | 18.0M/81.8M [00:06<00:24, 2.73MB/s]

.. parsed-literal::

     22%|██▏       | 18.2M/81.8M [00:06<00:23, 2.80MB/s]

.. parsed-literal::

     23%|██▎       | 18.5M/81.8M [00:06<00:23, 2.82MB/s]

.. parsed-literal::

     23%|██▎       | 18.8M/81.8M [00:06<00:23, 2.79MB/s]

.. parsed-literal::

     23%|██▎       | 19.1M/81.8M [00:06<00:23, 2.81MB/s]

.. parsed-literal::

     24%|██▎       | 19.4M/81.8M [00:06<00:23, 2.81MB/s]

.. parsed-literal::

     24%|██▍       | 19.6M/81.8M [00:06<00:27, 2.37MB/s]

.. parsed-literal::

     24%|██▍       | 20.0M/81.8M [00:06<00:23, 2.73MB/s]

.. parsed-literal::

     25%|██▍       | 20.3M/81.8M [00:06<00:25, 2.51MB/s]

.. parsed-literal::

     25%|██▌       | 20.5M/81.8M [00:07<00:27, 2.37MB/s]

.. parsed-literal::

     25%|██▌       | 20.8M/81.8M [00:07<00:29, 2.20MB/s]

.. parsed-literal::

     26%|██▌       | 21.0M/81.8M [00:07<00:28, 2.21MB/s]

.. parsed-literal::

     26%|██▌       | 21.2M/81.8M [00:07<00:28, 2.25MB/s]

.. parsed-literal::

     26%|██▌       | 21.5M/81.8M [00:07<00:28, 2.24MB/s]

.. parsed-literal::

     27%|██▋       | 21.7M/81.8M [00:07<00:28, 2.24MB/s]

.. parsed-literal::

     27%|██▋       | 21.9M/81.8M [00:07<00:28, 2.24MB/s]

.. parsed-literal::

     27%|██▋       | 22.1M/81.8M [00:07<00:27, 2.25MB/s]

.. parsed-literal::

     27%|██▋       | 22.3M/81.8M [00:07<00:27, 2.24MB/s]

.. parsed-literal::

     28%|██▊       | 22.6M/81.8M [00:08<00:27, 2.26MB/s]

.. parsed-literal::

     28%|██▊       | 22.8M/81.8M [00:08<00:28, 2.20MB/s]

.. parsed-literal::

     28%|██▊       | 23.0M/81.8M [00:08<00:27, 2.24MB/s]

.. parsed-literal::

     28%|██▊       | 23.2M/81.8M [00:08<00:26, 2.28MB/s]

.. parsed-literal::

     29%|██▊       | 23.5M/81.8M [00:08<00:26, 2.27MB/s]

.. parsed-literal::

     29%|██▉       | 23.7M/81.8M [00:08<00:26, 2.29MB/s]

.. parsed-literal::

     29%|██▉       | 23.9M/81.8M [00:08<00:26, 2.33MB/s]

.. parsed-literal::

     30%|██▉       | 24.2M/81.8M [00:08<00:25, 2.34MB/s]

.. parsed-literal::

     30%|██▉       | 24.4M/81.8M [00:08<00:25, 2.36MB/s]

.. parsed-literal::

     30%|███       | 24.6M/81.8M [00:09<00:25, 2.38MB/s]

.. parsed-literal::

     30%|███       | 24.9M/81.8M [00:09<00:25, 2.38MB/s]

.. parsed-literal::

     31%|███       | 25.1M/81.8M [00:09<00:24, 2.39MB/s]

.. parsed-literal::

     31%|███       | 25.3M/81.8M [00:09<00:31, 1.86MB/s]

.. parsed-literal::

     31%|███       | 25.5M/81.8M [00:09<00:30, 1.91MB/s]

.. parsed-literal::

     31%|███▏      | 25.7M/81.8M [00:09<00:33, 1.76MB/s]

.. parsed-literal::

     32%|███▏      | 25.9M/81.8M [00:09<00:33, 1.74MB/s]

.. parsed-literal::

     32%|███▏      | 26.1M/81.8M [00:09<00:33, 1.76MB/s]

.. parsed-literal::

     32%|███▏      | 26.3M/81.8M [00:09<00:33, 1.76MB/s]

.. parsed-literal::

     32%|███▏      | 26.5M/81.8M [00:10<00:39, 1.48MB/s]

.. parsed-literal::

     33%|███▎      | 26.7M/81.8M [00:10<00:33, 1.73MB/s]

.. parsed-literal::

     33%|███▎      | 26.9M/81.8M [00:10<00:35, 1.62MB/s]

.. parsed-literal::

     33%|███▎      | 27.0M/81.8M [00:10<00:37, 1.52MB/s]

.. parsed-literal::

     33%|███▎      | 27.2M/81.8M [00:10<00:43, 1.32MB/s]

.. parsed-literal::

     33%|███▎      | 27.3M/81.8M [00:10<00:46, 1.22MB/s]

.. parsed-literal::

     34%|███▎      | 27.5M/81.8M [00:10<00:49, 1.15MB/s]

.. parsed-literal::

     34%|███▎      | 27.6M/81.8M [00:11<00:48, 1.16MB/s]

.. parsed-literal::

     34%|███▍      | 27.7M/81.8M [00:11<00:52, 1.07MB/s]

.. parsed-literal::

     34%|███▍      | 27.8M/81.8M [00:11<00:51, 1.11MB/s]

.. parsed-literal::

     34%|███▍      | 27.9M/81.8M [00:11<00:50, 1.11MB/s]

.. parsed-literal::

     34%|███▍      | 28.0M/81.8M [00:11<00:50, 1.11MB/s]

.. parsed-literal::

     34%|███▍      | 28.2M/81.8M [00:11<00:50, 1.12MB/s]

.. parsed-literal::

     35%|███▍      | 28.3M/81.8M [00:11<00:49, 1.13MB/s]

.. parsed-literal::

     35%|███▍      | 28.4M/81.8M [00:11<00:49, 1.13MB/s]

.. parsed-literal::

     35%|███▍      | 28.5M/81.8M [00:11<00:48, 1.15MB/s]

.. parsed-literal::

     35%|███▍      | 28.6M/81.8M [00:12<00:47, 1.16MB/s]

.. parsed-literal::

     35%|███▌      | 28.7M/81.8M [00:12<00:49, 1.13MB/s]

.. parsed-literal::

     35%|███▌      | 28.8M/81.8M [00:12<00:49, 1.13MB/s]

.. parsed-literal::

     35%|███▌      | 29.0M/81.8M [00:12<01:02, 893kB/s] 

.. parsed-literal::

     36%|███▌      | 29.2M/81.8M [00:12<00:48, 1.13MB/s]

.. parsed-literal::

     36%|███▌      | 29.3M/81.8M [00:12<00:52, 1.05MB/s]

.. parsed-literal::

     36%|███▌      | 29.4M/81.8M [00:12<00:55, 999kB/s] 

.. parsed-literal::

     36%|███▌      | 29.5M/81.8M [00:12<00:54, 998kB/s]

.. parsed-literal::

     36%|███▌      | 29.6M/81.8M [00:13<00:55, 988kB/s]

.. parsed-literal::

     36%|███▋      | 29.7M/81.8M [00:13<00:55, 979kB/s]

.. parsed-literal::

     36%|███▋      | 29.8M/81.8M [00:13<00:57, 952kB/s]

.. parsed-literal::

     37%|███▋      | 29.9M/81.8M [00:13<00:56, 971kB/s]

.. parsed-literal::

     37%|███▋      | 30.0M/81.8M [00:13<01:10, 771kB/s]

.. parsed-literal::

     37%|███▋      | 30.1M/81.8M [00:13<01:02, 874kB/s]

.. parsed-literal::

     37%|███▋      | 30.2M/81.8M [00:13<01:05, 824kB/s]

.. parsed-literal::

     37%|███▋      | 30.3M/81.8M [00:14<01:15, 720kB/s]

.. parsed-literal::

     37%|███▋      | 30.4M/81.8M [00:14<01:15, 714kB/s]

.. parsed-literal::

     37%|███▋      | 30.5M/81.8M [00:14<01:14, 720kB/s]

.. parsed-literal::

     37%|███▋      | 30.5M/81.8M [00:14<01:21, 658kB/s]

.. parsed-literal::

     37%|███▋      | 30.6M/81.8M [00:14<01:21, 657kB/s]

.. parsed-literal::

     38%|███▊      | 30.7M/81.8M [00:14<01:25, 628kB/s]

.. parsed-literal::

     38%|███▊      | 30.8M/81.8M [00:14<01:27, 615kB/s]

.. parsed-literal::

     38%|███▊      | 30.8M/81.8M [00:14<01:26, 621kB/s]

.. parsed-literal::

     38%|███▊      | 30.9M/81.8M [00:15<01:27, 612kB/s]

.. parsed-literal::

     38%|███▊      | 31.0M/81.8M [00:15<01:24, 627kB/s]

.. parsed-literal::

     38%|███▊      | 31.0M/81.8M [00:15<01:24, 628kB/s]

.. parsed-literal::

     38%|███▊      | 31.1M/81.8M [00:15<01:24, 627kB/s]

.. parsed-literal::

     38%|███▊      | 31.1M/81.8M [00:15<01:24, 631kB/s]

.. parsed-literal::

     38%|███▊      | 31.2M/81.8M [00:15<01:25, 623kB/s]

.. parsed-literal::

     38%|███▊      | 31.3M/81.8M [00:15<01:24, 629kB/s]

.. parsed-literal::

     38%|███▊      | 31.3M/81.8M [00:15<01:21, 645kB/s]

.. parsed-literal::

     38%|███▊      | 31.4M/81.8M [00:15<01:20, 653kB/s]

.. parsed-literal::

     38%|███▊      | 31.5M/81.8M [00:16<01:22, 638kB/s]

.. parsed-literal::

     39%|███▊      | 31.5M/81.8M [00:16<01:31, 578kB/s]

.. parsed-literal::

     39%|███▊      | 31.6M/81.8M [00:16<01:34, 555kB/s]

.. parsed-literal::

     39%|███▊      | 31.7M/81.8M [00:16<01:34, 553kB/s]

.. parsed-literal::

     39%|███▉      | 31.7M/81.8M [00:16<01:36, 543kB/s]

.. parsed-literal::

     39%|███▉      | 31.8M/81.8M [00:16<01:59, 439kB/s]

.. parsed-literal::

     39%|███▉      | 31.9M/81.8M [00:16<01:28, 591kB/s]

.. parsed-literal::

     39%|███▉      | 32.0M/81.8M [00:16<01:29, 582kB/s]

.. parsed-literal::

     39%|███▉      | 32.0M/81.8M [00:17<01:28, 593kB/s]

.. parsed-literal::

     39%|███▉      | 32.1M/81.8M [00:17<01:26, 605kB/s]

.. parsed-literal::

     39%|███▉      | 32.1M/81.8M [00:17<01:25, 609kB/s]

.. parsed-literal::

     39%|███▉      | 32.2M/81.8M [00:17<01:23, 622kB/s]

.. parsed-literal::

     39%|███▉      | 32.3M/81.8M [00:17<01:22, 627kB/s]

.. parsed-literal::

     40%|███▉      | 32.3M/81.8M [00:17<01:20, 642kB/s]

.. parsed-literal::

     40%|███▉      | 32.4M/81.8M [00:17<01:20, 641kB/s]

.. parsed-literal::

     40%|███▉      | 32.5M/81.8M [00:17<01:20, 640kB/s]

.. parsed-literal::

     40%|███▉      | 32.5M/81.8M [00:17<01:22, 622kB/s]

.. parsed-literal::

     40%|███▉      | 32.6M/81.8M [00:18<01:16, 670kB/s]

.. parsed-literal::

     40%|███▉      | 32.7M/81.8M [00:18<01:15, 681kB/s]

.. parsed-literal::

     40%|████      | 32.8M/81.8M [00:18<01:15, 683kB/s]

.. parsed-literal::

     40%|████      | 32.8M/81.8M [00:18<01:16, 670kB/s]

.. parsed-literal::

     40%|████      | 32.9M/81.8M [00:18<01:17, 663kB/s]

.. parsed-literal::

     40%|████      | 33.0M/81.8M [00:18<01:15, 679kB/s]

.. parsed-literal::

     40%|████      | 33.1M/81.8M [00:18<01:14, 683kB/s]

.. parsed-literal::

     41%|████      | 33.1M/81.8M [00:18<01:13, 692kB/s]

.. parsed-literal::

     41%|████      | 33.2M/81.8M [00:18<01:15, 673kB/s]

.. parsed-literal::

     41%|████      | 33.3M/81.8M [00:19<01:14, 681kB/s]

.. parsed-literal::

     41%|████      | 33.4M/81.8M [00:19<01:12, 696kB/s]

.. parsed-literal::

     41%|████      | 33.4M/81.8M [00:19<01:11, 707kB/s]

.. parsed-literal::

     41%|████      | 33.5M/81.8M [00:19<01:11, 707kB/s]

.. parsed-literal::

     41%|████      | 33.6M/81.8M [00:19<01:10, 715kB/s]

.. parsed-literal::

     41%|████      | 33.7M/81.8M [00:19<01:09, 731kB/s]

.. parsed-literal::

     41%|████▏     | 33.8M/81.8M [00:19<01:09, 727kB/s]

.. parsed-literal::

     41%|████▏     | 33.8M/81.8M [00:19<01:07, 740kB/s]

.. parsed-literal::

     41%|████▏     | 33.9M/81.8M [00:19<01:07, 743kB/s]

.. parsed-literal::

     42%|████▏     | 34.0M/81.8M [00:20<01:07, 745kB/s]

.. parsed-literal::

     42%|████▏     | 34.1M/81.8M [00:20<01:04, 774kB/s]

.. parsed-literal::

     42%|████▏     | 34.2M/81.8M [00:20<01:04, 775kB/s]

.. parsed-literal::

     42%|████▏     | 34.2M/81.8M [00:20<01:03, 786kB/s]

.. parsed-literal::

     42%|████▏     | 34.3M/81.8M [00:20<01:11, 701kB/s]

.. parsed-literal::

     42%|████▏     | 34.4M/81.8M [00:20<01:08, 730kB/s]

.. parsed-literal::

     42%|████▏     | 34.5M/81.8M [00:20<01:11, 696kB/s]

.. parsed-literal::

     42%|████▏     | 34.6M/81.8M [00:20<01:10, 701kB/s]

.. parsed-literal::

     42%|████▏     | 34.6M/81.8M [00:20<01:15, 659kB/s]

.. parsed-literal::

     42%|████▏     | 34.7M/81.8M [00:21<01:13, 669kB/s]

.. parsed-literal::

     43%|████▎     | 34.8M/81.8M [00:21<01:15, 655kB/s]

.. parsed-literal::

     43%|████▎     | 34.8M/81.8M [00:21<01:13, 671kB/s]

.. parsed-literal::

     43%|████▎     | 34.9M/81.8M [00:21<01:12, 678kB/s]

.. parsed-literal::

     43%|████▎     | 35.0M/81.8M [00:21<01:10, 697kB/s]

.. parsed-literal::

     43%|████▎     | 35.1M/81.8M [00:21<01:08, 715kB/s]

.. parsed-literal::

     43%|████▎     | 35.1M/81.8M [00:21<01:08, 709kB/s]

.. parsed-literal::

     43%|████▎     | 35.2M/81.8M [00:21<01:06, 735kB/s]

.. parsed-literal::

     43%|████▎     | 35.3M/81.8M [00:21<01:05, 749kB/s]

.. parsed-literal::

     43%|████▎     | 35.4M/81.8M [00:22<01:05, 741kB/s]

.. parsed-literal::

     43%|████▎     | 35.5M/81.8M [00:22<01:05, 747kB/s]

.. parsed-literal::

     43%|████▎     | 35.5M/81.8M [00:22<01:03, 764kB/s]

.. parsed-literal::

     44%|████▎     | 35.6M/81.8M [00:22<01:02, 780kB/s]

.. parsed-literal::

     44%|████▎     | 35.7M/81.8M [00:22<01:01, 784kB/s]

.. parsed-literal::

     44%|████▎     | 35.8M/81.8M [00:22<01:01, 790kB/s]

.. parsed-literal::

     44%|████▍     | 35.9M/81.8M [00:22<01:02, 772kB/s]

.. parsed-literal::

     44%|████▍     | 35.9M/81.8M [00:22<01:07, 710kB/s]

.. parsed-literal::

     44%|████▍     | 36.0M/81.8M [00:22<00:59, 806kB/s]

.. parsed-literal::

     44%|████▍     | 36.1M/81.8M [00:23<00:58, 820kB/s]

.. parsed-literal::

     44%|████▍     | 36.2M/81.8M [00:23<00:59, 806kB/s]

.. parsed-literal::

     44%|████▍     | 36.3M/81.8M [00:23<00:59, 807kB/s]

.. parsed-literal::

     44%|████▍     | 36.4M/81.8M [00:23<01:00, 782kB/s]

.. parsed-literal::

     45%|████▍     | 36.5M/81.8M [00:23<00:58, 813kB/s]

.. parsed-literal::

     45%|████▍     | 36.6M/81.8M [00:23<00:57, 824kB/s]

.. parsed-literal::

     45%|████▍     | 36.7M/81.8M [00:23<00:57, 824kB/s]

.. parsed-literal::

     45%|████▍     | 36.8M/81.8M [00:23<00:58, 806kB/s]

.. parsed-literal::

     45%|████▌     | 36.8M/81.8M [00:23<00:57, 821kB/s]

.. parsed-literal::

     45%|████▌     | 36.9M/81.8M [00:24<00:56, 827kB/s]

.. parsed-literal::

     45%|████▌     | 37.0M/81.8M [00:24<00:57, 810kB/s]

.. parsed-literal::

     45%|████▌     | 37.1M/81.8M [00:24<00:57, 820kB/s]

.. parsed-literal::

     46%|████▌     | 37.2M/81.8M [00:24<00:55, 849kB/s]

.. parsed-literal::

     46%|████▌     | 37.3M/81.8M [00:24<00:54, 857kB/s]

.. parsed-literal::

     46%|████▌     | 37.4M/81.8M [00:24<00:53, 865kB/s]

.. parsed-literal::

     46%|████▌     | 37.5M/81.8M [00:24<00:54, 860kB/s]

.. parsed-literal::

     46%|████▌     | 37.6M/81.8M [00:24<00:52, 890kB/s]

.. parsed-literal::

     46%|████▌     | 37.7M/81.8M [00:24<00:53, 870kB/s]

.. parsed-literal::

     46%|████▌     | 37.8M/81.8M [00:25<00:51, 891kB/s]

.. parsed-literal::

     46%|████▋     | 37.9M/81.8M [00:25<00:50, 904kB/s]

.. parsed-literal::

     46%|████▋     | 38.0M/81.8M [00:25<00:49, 923kB/s]

.. parsed-literal::

     47%|████▋     | 38.0M/81.8M [00:25<00:49, 918kB/s]

.. parsed-literal::

     47%|████▋     | 38.1M/81.8M [00:25<00:47, 960kB/s]

.. parsed-literal::

     47%|████▋     | 38.2M/81.8M [00:25<00:48, 947kB/s]

.. parsed-literal::

     47%|████▋     | 38.3M/81.8M [00:25<00:48, 943kB/s]

.. parsed-literal::

     47%|████▋     | 38.5M/81.8M [00:25<00:47, 956kB/s]

.. parsed-literal::

     47%|████▋     | 38.6M/81.8M [00:25<00:46, 975kB/s]

.. parsed-literal::

     47%|████▋     | 38.7M/81.8M [00:26<00:45, 987kB/s]

.. parsed-literal::

     47%|████▋     | 38.8M/81.8M [00:26<00:48, 930kB/s]

.. parsed-literal::

     48%|████▊     | 38.9M/81.8M [00:26<00:49, 917kB/s]

.. parsed-literal::

     48%|████▊     | 39.0M/81.8M [00:26<00:53, 841kB/s]

.. parsed-literal::

     48%|████▊     | 39.0M/81.8M [00:26<00:54, 818kB/s]

.. parsed-literal::

     48%|████▊     | 39.1M/81.8M [00:26<00:55, 805kB/s]

.. parsed-literal::

     48%|████▊     | 39.2M/81.8M [00:26<00:55, 806kB/s]

.. parsed-literal::

     48%|████▊     | 39.3M/81.8M [00:26<00:54, 814kB/s]

.. parsed-literal::

     48%|████▊     | 39.4M/81.8M [00:27<00:52, 840kB/s]

.. parsed-literal::

     48%|████▊     | 39.5M/81.8M [00:27<00:53, 833kB/s]

.. parsed-literal::

     48%|████▊     | 39.6M/81.8M [00:27<00:51, 856kB/s]

.. parsed-literal::

     49%|████▊     | 39.7M/81.8M [00:27<00:51, 853kB/s]

.. parsed-literal::

     49%|████▊     | 39.8M/81.8M [00:27<00:49, 886kB/s]

.. parsed-literal::

     49%|████▉     | 39.9M/81.8M [00:27<00:48, 908kB/s]

.. parsed-literal::

     49%|████▉     | 40.0M/81.8M [00:27<00:48, 905kB/s]

.. parsed-literal::

     49%|████▉     | 40.1M/81.8M [00:27<00:59, 733kB/s]

.. parsed-literal::

     49%|████▉     | 40.3M/81.8M [00:28<00:42, 1.02MB/s]

.. parsed-literal::

     49%|████▉     | 40.4M/81.8M [00:28<00:43, 995kB/s] 

.. parsed-literal::

     49%|████▉     | 40.5M/81.8M [00:28<00:48, 893kB/s]

.. parsed-literal::

     50%|████▉     | 40.6M/81.8M [00:28<00:43, 999kB/s]

.. parsed-literal::

     50%|████▉     | 40.7M/81.8M [00:28<00:47, 898kB/s]

.. parsed-literal::

     50%|████▉     | 40.8M/81.8M [00:28<00:50, 850kB/s]

.. parsed-literal::

     50%|████▉     | 40.9M/81.8M [00:28<00:52, 812kB/s]

.. parsed-literal::

     50%|█████     | 41.0M/81.8M [00:28<00:53, 801kB/s]

.. parsed-literal::

     50%|█████     | 41.1M/81.8M [00:29<00:53, 803kB/s]

.. parsed-literal::

     50%|█████     | 41.1M/81.8M [00:29<00:53, 798kB/s]

.. parsed-literal::

     50%|█████     | 41.2M/81.8M [00:29<00:54, 784kB/s]

.. parsed-literal::

     50%|█████     | 41.3M/81.8M [00:29<00:53, 788kB/s]

.. parsed-literal::

     51%|█████     | 41.4M/81.8M [00:29<00:53, 793kB/s]

.. parsed-literal::

     51%|█████     | 41.5M/81.8M [00:29<00:52, 799kB/s]

.. parsed-literal::

     51%|█████     | 41.5M/81.8M [00:29<00:51, 817kB/s]

.. parsed-literal::

     51%|█████     | 41.6M/81.8M [00:29<00:51, 818kB/s]

.. parsed-literal::

     51%|█████     | 41.7M/81.8M [00:29<00:52, 796kB/s]

.. parsed-literal::

     51%|█████     | 41.8M/81.8M [00:29<00:51, 819kB/s]

.. parsed-literal::

     51%|█████     | 41.9M/81.8M [00:30<00:50, 834kB/s]

.. parsed-literal::

     51%|█████▏    | 42.0M/81.8M [00:30<00:49, 836kB/s]

.. parsed-literal::

     51%|█████▏    | 42.1M/81.8M [00:30<00:48, 851kB/s]

.. parsed-literal::

     52%|█████▏    | 42.2M/81.8M [00:30<00:51, 813kB/s]

.. parsed-literal::

     52%|█████▏    | 42.2M/81.8M [00:30<00:50, 829kB/s]

.. parsed-literal::

     52%|█████▏    | 42.3M/81.8M [00:30<00:49, 840kB/s]

.. parsed-literal::

     52%|█████▏    | 42.4M/81.8M [00:30<00:49, 841kB/s]

.. parsed-literal::

     52%|█████▏    | 42.5M/81.8M [00:30<00:47, 861kB/s]

.. parsed-literal::

     52%|█████▏    | 42.6M/81.8M [00:30<00:46, 888kB/s]

.. parsed-literal::

     52%|█████▏    | 42.7M/81.8M [00:31<00:45, 904kB/s]

.. parsed-literal::

     52%|█████▏    | 42.8M/81.8M [00:31<00:48, 842kB/s]

.. parsed-literal::

     52%|█████▏    | 42.9M/81.8M [00:31<00:44, 906kB/s]

.. parsed-literal::

     53%|█████▎    | 43.0M/81.8M [00:31<00:44, 914kB/s]

.. parsed-literal::

     53%|█████▎    | 43.1M/81.8M [00:31<00:43, 932kB/s]

.. parsed-literal::

     53%|█████▎    | 43.2M/81.8M [00:31<00:43, 922kB/s]

.. parsed-literal::

     53%|█████▎    | 43.3M/81.8M [00:31<00:42, 950kB/s]

.. parsed-literal::

     53%|█████▎    | 43.4M/81.8M [00:31<00:42, 937kB/s]

.. parsed-literal::

     53%|█████▎    | 43.5M/81.8M [00:31<00:43, 914kB/s]

.. parsed-literal::

     53%|█████▎    | 43.6M/81.8M [00:32<00:41, 958kB/s]

.. parsed-literal::

     53%|█████▎    | 43.7M/81.8M [00:32<00:40, 989kB/s]

.. parsed-literal::

     54%|█████▎    | 43.8M/81.8M [00:32<00:40, 993kB/s]

.. parsed-literal::

     54%|█████▎    | 43.9M/81.8M [00:32<00:39, 995kB/s]

.. parsed-literal::

     54%|█████▍    | 44.0M/81.8M [00:32<00:39, 1.01MB/s]

.. parsed-literal::

     54%|█████▍    | 44.2M/81.8M [00:32<00:38, 1.02MB/s]

.. parsed-literal::

     54%|█████▍    | 44.3M/81.8M [00:32<00:38, 1.02MB/s]

.. parsed-literal::

     54%|█████▍    | 44.4M/81.8M [00:32<00:38, 1.03MB/s]

.. parsed-literal::

     54%|█████▍    | 44.5M/81.8M [00:32<00:37, 1.04MB/s]

.. parsed-literal::

     55%|█████▍    | 44.6M/81.8M [00:33<00:36, 1.06MB/s]

.. parsed-literal::

     55%|█████▍    | 44.7M/81.8M [00:33<00:37, 1.05MB/s]

.. parsed-literal::

     55%|█████▍    | 44.8M/81.8M [00:33<00:36, 1.05MB/s]

.. parsed-literal::

     55%|█████▍    | 44.9M/81.8M [00:33<00:41, 938kB/s] 

.. parsed-literal::

     55%|█████▌    | 45.0M/81.8M [00:33<00:38, 1.00MB/s]

.. parsed-literal::

     55%|█████▌    | 45.1M/81.8M [00:33<00:41, 930kB/s] 

.. parsed-literal::

     55%|█████▌    | 45.2M/81.8M [00:33<00:42, 905kB/s]

.. parsed-literal::

     55%|█████▌    | 45.3M/81.8M [00:33<00:42, 904kB/s]

.. parsed-literal::

     56%|█████▌    | 45.4M/81.8M [00:34<00:42, 892kB/s]

.. parsed-literal::

     56%|█████▌    | 45.5M/81.8M [00:34<00:41, 907kB/s]

.. parsed-literal::

     56%|█████▌    | 45.6M/81.8M [00:34<00:44, 852kB/s]

.. parsed-literal::

     56%|█████▌    | 45.7M/81.8M [00:34<00:45, 837kB/s]

.. parsed-literal::

     56%|█████▌    | 45.8M/81.8M [00:34<00:49, 767kB/s]

.. parsed-literal::

     56%|█████▌    | 45.9M/81.8M [00:34<00:51, 735kB/s]

.. parsed-literal::

     56%|█████▌    | 45.9M/81.8M [00:34<00:51, 724kB/s]

.. parsed-literal::

     56%|█████▋    | 46.0M/81.8M [00:34<00:55, 675kB/s]

.. parsed-literal::

     56%|█████▋    | 46.1M/81.8M [00:35<00:49, 750kB/s]

.. parsed-literal::

     56%|█████▋    | 46.2M/81.8M [00:35<00:50, 743kB/s]

.. parsed-literal::

     57%|█████▋    | 46.3M/81.8M [00:35<00:50, 744kB/s]

.. parsed-literal::

     57%|█████▋    | 46.3M/81.8M [00:35<00:49, 749kB/s]

.. parsed-literal::

     57%|█████▋    | 46.4M/81.8M [00:35<00:49, 752kB/s]

.. parsed-literal::

     57%|█████▋    | 46.5M/81.8M [00:35<00:48, 766kB/s]

.. parsed-literal::

     57%|█████▋    | 46.6M/81.8M [00:35<00:48, 764kB/s]

.. parsed-literal::

     57%|█████▋    | 46.7M/81.8M [00:35<00:48, 763kB/s]

.. parsed-literal::

     57%|█████▋    | 46.7M/81.8M [00:35<00:48, 761kB/s]

.. parsed-literal::

     57%|█████▋    | 46.8M/81.8M [00:35<00:47, 772kB/s]

.. parsed-literal::

     57%|█████▋    | 46.9M/81.8M [00:36<00:47, 770kB/s]

.. parsed-literal::

     57%|█████▋    | 47.0M/81.8M [00:36<00:46, 793kB/s]

.. parsed-literal::

     58%|█████▊    | 47.1M/81.8M [00:36<00:45, 799kB/s]

.. parsed-literal::

     58%|█████▊    | 47.2M/81.8M [00:36<00:46, 779kB/s]

.. parsed-literal::

     58%|█████▊    | 47.2M/81.8M [00:36<00:46, 784kB/s]

.. parsed-literal::

     58%|█████▊    | 47.3M/81.8M [00:36<00:43, 838kB/s]

.. parsed-literal::

     58%|█████▊    | 47.4M/81.8M [00:36<00:43, 831kB/s]

.. parsed-literal::

     58%|█████▊    | 47.5M/81.8M [00:36<00:43, 819kB/s]

.. parsed-literal::

     58%|█████▊    | 47.6M/81.8M [00:37<00:54, 659kB/s]

.. parsed-literal::

     58%|█████▊    | 47.7M/81.8M [00:37<00:43, 821kB/s]

.. parsed-literal::

     58%|█████▊    | 47.8M/81.8M [00:37<00:46, 768kB/s]

.. parsed-literal::

     59%|█████▊    | 47.9M/81.8M [00:37<00:47, 746kB/s]

.. parsed-literal::

     59%|█████▊    | 48.0M/81.8M [00:37<00:49, 721kB/s]

.. parsed-literal::

     59%|█████▉    | 48.1M/81.8M [00:37<00:50, 699kB/s]

.. parsed-literal::

     59%|█████▉    | 48.1M/81.8M [00:37<00:51, 680kB/s]

.. parsed-literal::

     59%|█████▉    | 48.2M/81.8M [00:37<00:50, 704kB/s]

.. parsed-literal::

     59%|█████▉    | 48.3M/81.8M [00:38<00:48, 719kB/s]

.. parsed-literal::

     59%|█████▉    | 48.4M/81.8M [00:38<00:53, 654kB/s]

.. parsed-literal::

     59%|█████▉    | 48.4M/81.8M [00:38<00:52, 664kB/s]

.. parsed-literal::

     59%|█████▉    | 48.5M/81.8M [00:38<00:53, 651kB/s]

.. parsed-literal::

     59%|█████▉    | 48.6M/81.8M [00:38<00:58, 595kB/s]

.. parsed-literal::

     59%|█████▉    | 48.6M/81.8M [00:38<00:58, 598kB/s]

.. parsed-literal::

     60%|█████▉    | 48.7M/81.8M [00:38<00:58, 594kB/s]

.. parsed-literal::

     60%|█████▉    | 48.8M/81.8M [00:38<01:05, 526kB/s]

.. parsed-literal::

     60%|█████▉    | 48.8M/81.8M [00:39<01:09, 497kB/s]

.. parsed-literal::

     60%|█████▉    | 48.9M/81.8M [00:39<01:13, 472kB/s]

.. parsed-literal::

     60%|█████▉    | 48.9M/81.8M [00:39<01:23, 412kB/s]

.. parsed-literal::

     60%|█████▉    | 49.0M/81.8M [00:39<01:22, 416kB/s]

.. parsed-literal::

     60%|█████▉    | 49.0M/81.8M [00:39<01:36, 356kB/s]

.. parsed-literal::

     60%|█████▉    | 49.0M/81.8M [00:39<01:39, 345kB/s]

.. parsed-literal::

     60%|██████    | 49.1M/81.8M [00:39<01:40, 342kB/s]

.. parsed-literal::

     60%|██████    | 49.1M/81.8M [00:40<01:37, 352kB/s]

.. parsed-literal::

     60%|██████    | 49.2M/81.8M [00:40<01:40, 341kB/s]

.. parsed-literal::

     60%|██████    | 49.2M/81.8M [00:40<01:36, 354kB/s]

.. parsed-literal::

     60%|██████    | 49.3M/81.8M [00:40<01:31, 373kB/s]

.. parsed-literal::

     60%|██████    | 49.3M/81.8M [00:40<01:34, 359kB/s]

.. parsed-literal::

     60%|██████    | 49.4M/81.8M [00:40<01:28, 382kB/s]

.. parsed-literal::

     60%|██████    | 49.4M/81.8M [00:40<01:26, 393kB/s]

.. parsed-literal::

     60%|██████    | 49.5M/81.8M [00:40<01:24, 403kB/s]

.. parsed-literal::

     61%|██████    | 49.5M/81.8M [00:41<01:23, 407kB/s]

.. parsed-literal::

     61%|██████    | 49.6M/81.8M [00:41<01:20, 417kB/s]

.. parsed-literal::

     61%|██████    | 49.6M/81.8M [00:41<01:20, 421kB/s]

.. parsed-literal::

     61%|██████    | 49.7M/81.8M [00:41<01:17, 436kB/s]

.. parsed-literal::

     61%|██████    | 49.7M/81.8M [00:41<01:14, 450kB/s]

.. parsed-literal::

     61%|██████    | 49.8M/81.8M [00:41<01:13, 458kB/s]

.. parsed-literal::

     61%|██████    | 49.8M/81.8M [00:41<01:08, 491kB/s]

.. parsed-literal::

     61%|██████    | 49.9M/81.8M [00:41<01:08, 490kB/s]

.. parsed-literal::

     61%|██████    | 49.9M/81.8M [00:41<01:06, 502kB/s]

.. parsed-literal::

     61%|██████    | 50.0M/81.8M [00:42<01:09, 477kB/s]

.. parsed-literal::

     61%|██████    | 50.0M/81.8M [00:42<01:07, 494kB/s]

.. parsed-literal::

     61%|██████    | 50.1M/81.8M [00:42<01:05, 508kB/s]

.. parsed-literal::

     61%|██████▏   | 50.2M/81.8M [00:42<01:03, 523kB/s]

.. parsed-literal::

     61%|██████▏   | 50.2M/81.8M [00:42<01:11, 463kB/s]

.. parsed-literal::

     61%|██████▏   | 50.3M/81.8M [00:42<01:07, 488kB/s]

.. parsed-literal::

     62%|██████▏   | 50.3M/81.8M [00:42<01:08, 482kB/s]

.. parsed-literal::

     62%|██████▏   | 50.4M/81.8M [00:43<01:14, 443kB/s]

.. parsed-literal::

     62%|██████▏   | 50.4M/81.8M [00:43<01:15, 438kB/s]

.. parsed-literal::

     62%|██████▏   | 50.5M/81.8M [00:43<01:09, 471kB/s]

.. parsed-literal::

     62%|██████▏   | 50.5M/81.8M [00:43<01:11, 462kB/s]

.. parsed-literal::

     62%|██████▏   | 50.6M/81.8M [00:43<01:11, 459kB/s]

.. parsed-literal::

     62%|██████▏   | 50.6M/81.8M [00:43<01:09, 470kB/s]

.. parsed-literal::

     62%|██████▏   | 50.7M/81.8M [00:43<01:07, 481kB/s]

.. parsed-literal::

     62%|██████▏   | 50.8M/81.8M [00:43<01:04, 501kB/s]

.. parsed-literal::

     62%|██████▏   | 50.8M/81.8M [00:43<01:05, 498kB/s]

.. parsed-literal::

     62%|██████▏   | 50.9M/81.8M [00:44<01:05, 494kB/s]

.. parsed-literal::

     62%|██████▏   | 51.0M/81.8M [00:44<01:01, 529kB/s]

.. parsed-literal::

     62%|██████▏   | 51.0M/81.8M [00:44<01:01, 523kB/s]

.. parsed-literal::

     62%|██████▏   | 51.1M/81.8M [00:44<00:59, 537kB/s]

.. parsed-literal::

     63%|██████▎   | 51.1M/81.8M [00:44<00:58, 545kB/s]

.. parsed-literal::

     63%|██████▎   | 51.2M/81.8M [00:44<00:58, 546kB/s]

.. parsed-literal::

     63%|██████▎   | 51.3M/81.8M [00:44<00:59, 540kB/s]

.. parsed-literal::

     63%|██████▎   | 51.3M/81.8M [00:44<00:57, 553kB/s]

.. parsed-literal::

     63%|██████▎   | 51.4M/81.8M [00:45<00:57, 550kB/s]

.. parsed-literal::

     63%|██████▎   | 51.5M/81.8M [00:45<00:56, 568kB/s]

.. parsed-literal::

     63%|██████▎   | 51.5M/81.8M [00:45<00:56, 567kB/s]

.. parsed-literal::

     63%|██████▎   | 51.6M/81.8M [00:45<00:54, 579kB/s]

.. parsed-literal::

     63%|██████▎   | 51.6M/81.8M [00:45<00:54, 579kB/s]

.. parsed-literal::

     63%|██████▎   | 51.7M/81.8M [00:45<00:55, 567kB/s]

.. parsed-literal::

     63%|██████▎   | 51.8M/81.8M [00:45<00:52, 604kB/s]

.. parsed-literal::

     63%|██████▎   | 51.8M/81.8M [00:45<00:52, 601kB/s]

.. parsed-literal::

     63%|██████▎   | 51.9M/81.8M [00:45<00:51, 605kB/s]

.. parsed-literal::

     64%|██████▎   | 52.0M/81.8M [00:46<00:51, 605kB/s]

.. parsed-literal::

     64%|██████▎   | 52.1M/81.8M [00:46<00:48, 643kB/s]

.. parsed-literal::

     64%|██████▍   | 52.1M/81.8M [00:46<00:48, 647kB/s]

.. parsed-literal::

     64%|██████▍   | 52.2M/81.8M [00:46<00:47, 648kB/s]

.. parsed-literal::

     64%|██████▍   | 52.3M/81.8M [00:46<00:47, 647kB/s]

.. parsed-literal::

     64%|██████▍   | 52.4M/81.8M [00:46<00:46, 665kB/s]

.. parsed-literal::

     64%|██████▍   | 52.4M/81.8M [00:46<00:44, 687kB/s]

.. parsed-literal::

     64%|██████▍   | 52.5M/81.8M [00:46<00:43, 698kB/s]

.. parsed-literal::

     64%|██████▍   | 52.6M/81.8M [00:47<00:43, 697kB/s]

.. parsed-literal::

     64%|██████▍   | 52.7M/81.8M [00:47<00:42, 721kB/s]

.. parsed-literal::

     64%|██████▍   | 52.8M/81.8M [00:47<00:42, 711kB/s]

.. parsed-literal::

     65%|██████▍   | 52.8M/81.8M [00:47<00:55, 550kB/s]

.. parsed-literal::

     65%|██████▍   | 53.0M/81.8M [00:47<00:37, 810kB/s]

.. parsed-literal::

     65%|██████▍   | 53.1M/81.8M [00:47<00:45, 667kB/s]

.. parsed-literal::

     65%|██████▌   | 53.2M/81.8M [00:47<00:39, 767kB/s]

.. parsed-literal::

     65%|██████▌   | 53.3M/81.8M [00:48<00:42, 706kB/s]

.. parsed-literal::

     65%|██████▌   | 53.4M/81.8M [00:48<00:43, 681kB/s]

.. parsed-literal::

     65%|██████▌   | 53.4M/81.8M [00:48<00:45, 659kB/s]

.. parsed-literal::

     65%|██████▌   | 53.5M/81.8M [00:48<00:46, 643kB/s]

.. parsed-literal::

     65%|██████▌   | 53.6M/81.8M [00:48<00:45, 656kB/s]

.. parsed-literal::

     66%|██████▌   | 53.6M/81.8M [00:48<00:45, 656kB/s]

.. parsed-literal::

     66%|██████▌   | 53.7M/81.8M [00:48<00:43, 676kB/s]

.. parsed-literal::

     66%|██████▌   | 53.8M/81.8M [00:48<00:44, 662kB/s]

.. parsed-literal::

     66%|██████▌   | 53.9M/81.8M [00:48<00:42, 684kB/s]

.. parsed-literal::

     66%|██████▌   | 53.9M/81.8M [00:49<00:43, 667kB/s]

.. parsed-literal::

     66%|██████▌   | 54.0M/81.8M [00:49<00:42, 692kB/s]

.. parsed-literal::

     66%|██████▌   | 54.1M/81.8M [00:49<00:38, 745kB/s]

.. parsed-literal::

     66%|██████▌   | 54.2M/81.8M [00:49<00:46, 623kB/s]

.. parsed-literal::

     66%|██████▋   | 54.3M/81.8M [00:49<00:37, 760kB/s]

.. parsed-literal::

     66%|██████▋   | 54.4M/81.8M [00:49<00:37, 773kB/s]

.. parsed-literal::

     67%|██████▋   | 54.5M/81.8M [00:49<00:36, 778kB/s]

.. parsed-literal::

     67%|██████▋   | 54.5M/81.8M [00:49<00:36, 780kB/s]

.. parsed-literal::

     67%|██████▋   | 54.6M/81.8M [00:49<00:36, 772kB/s]

.. parsed-literal::

     67%|██████▋   | 54.7M/81.8M [00:50<00:36, 771kB/s]

.. parsed-literal::

     67%|██████▋   | 54.8M/81.8M [00:50<00:35, 794kB/s]

.. parsed-literal::

     67%|██████▋   | 54.9M/81.8M [00:50<00:35, 788kB/s]

.. parsed-literal::

     67%|██████▋   | 54.9M/81.8M [00:50<00:36, 771kB/s]

.. parsed-literal::

     67%|██████▋   | 55.0M/81.8M [00:50<00:37, 754kB/s]

.. parsed-literal::

     67%|██████▋   | 55.1M/81.8M [00:50<00:36, 766kB/s]

.. parsed-literal::

     67%|██████▋   | 55.2M/81.8M [00:50<00:37, 752kB/s]

.. parsed-literal::

     68%|██████▊   | 55.2M/81.8M [00:50<00:36, 763kB/s]

.. parsed-literal::

     68%|██████▊   | 55.3M/81.8M [00:50<00:35, 773kB/s]

.. parsed-literal::

     68%|██████▊   | 55.4M/81.8M [00:51<00:34, 797kB/s]

.. parsed-literal::

     68%|██████▊   | 55.5M/81.8M [00:51<00:34, 795kB/s]

.. parsed-literal::

     68%|██████▊   | 55.6M/81.8M [00:51<00:34, 793kB/s]

.. parsed-literal::

     68%|██████▊   | 55.7M/81.8M [00:51<00:34, 788kB/s]

.. parsed-literal::

     68%|██████▊   | 55.8M/81.8M [00:51<00:33, 809kB/s]

.. parsed-literal::

     68%|██████▊   | 55.9M/81.8M [00:51<00:32, 834kB/s]

.. parsed-literal::

     68%|██████▊   | 56.0M/81.8M [00:51<00:32, 827kB/s]

.. parsed-literal::

     69%|██████▊   | 56.0M/81.8M [00:51<00:32, 834kB/s]

.. parsed-literal::

     69%|██████▊   | 56.1M/81.8M [00:51<00:31, 849kB/s]

.. parsed-literal::

     69%|██████▉   | 56.2M/81.8M [00:52<00:31, 845kB/s]

.. parsed-literal::

     69%|██████▉   | 56.3M/81.8M [00:52<00:30, 881kB/s]

.. parsed-literal::

     69%|██████▉   | 56.4M/81.8M [00:52<00:29, 893kB/s]

.. parsed-literal::

     69%|██████▉   | 56.5M/81.8M [00:52<00:28, 916kB/s]

.. parsed-literal::

     69%|██████▉   | 56.6M/81.8M [00:52<00:28, 925kB/s]

.. parsed-literal::

     69%|██████▉   | 56.7M/81.8M [00:52<00:29, 903kB/s]

.. parsed-literal::

     69%|██████▉   | 56.8M/81.8M [00:52<00:28, 921kB/s]

.. parsed-literal::

     70%|██████▉   | 56.9M/81.8M [00:52<00:28, 908kB/s]

.. parsed-literal::

     70%|██████▉   | 57.0M/81.8M [00:52<00:28, 921kB/s]

.. parsed-literal::

     70%|██████▉   | 57.1M/81.8M [00:53<00:27, 931kB/s]

.. parsed-literal::

     70%|██████▉   | 57.2M/81.8M [00:53<00:27, 943kB/s]

.. parsed-literal::

     70%|███████   | 57.3M/81.8M [00:53<00:27, 928kB/s]

.. parsed-literal::

     70%|███████   | 57.4M/81.8M [00:53<00:26, 978kB/s]

.. parsed-literal::

     70%|███████   | 57.5M/81.8M [00:53<00:26, 968kB/s]

.. parsed-literal::

     70%|███████   | 57.6M/81.8M [00:53<00:31, 812kB/s]

.. parsed-literal::

     71%|███████   | 57.7M/81.8M [00:53<00:26, 954kB/s]

.. parsed-literal::

     71%|███████   | 57.8M/81.8M [00:53<00:27, 911kB/s]

.. parsed-literal::

     71%|███████   | 57.9M/81.8M [00:54<00:28, 868kB/s]

.. parsed-literal::

     71%|███████   | 58.0M/81.8M [00:54<00:30, 810kB/s]

.. parsed-literal::

     71%|███████   | 58.1M/81.8M [00:54<00:30, 818kB/s]

.. parsed-literal::

     71%|███████   | 58.2M/81.8M [00:54<00:29, 831kB/s]

.. parsed-literal::

     71%|███████▏  | 58.3M/81.8M [00:54<00:29, 828kB/s]

.. parsed-literal::

     71%|███████▏  | 58.4M/81.8M [00:54<00:29, 845kB/s]

.. parsed-literal::

     71%|███████▏  | 58.5M/81.8M [00:54<00:28, 844kB/s]

.. parsed-literal::

     72%|███████▏  | 58.6M/81.8M [00:54<00:28, 869kB/s]

.. parsed-literal::

     72%|███████▏  | 58.7M/81.8M [00:54<00:27, 877kB/s]

.. parsed-literal::

     72%|███████▏  | 58.8M/81.8M [00:55<00:27, 893kB/s]

.. parsed-literal::

     72%|███████▏  | 58.8M/81.8M [00:55<00:26, 912kB/s]

.. parsed-literal::

     72%|███████▏  | 58.9M/81.8M [00:55<00:25, 955kB/s]

.. parsed-literal::

     72%|███████▏  | 59.0M/81.8M [00:55<00:24, 956kB/s]

.. parsed-literal::

     72%|███████▏  | 59.1M/81.8M [00:55<00:24, 962kB/s]

.. parsed-literal::

     72%|███████▏  | 59.2M/81.8M [00:55<00:24, 962kB/s]

.. parsed-literal::

     73%|███████▎  | 59.3M/81.8M [00:55<00:24, 944kB/s]

.. parsed-literal::

     73%|███████▎  | 59.4M/81.8M [00:55<00:24, 953kB/s]

.. parsed-literal::

     73%|███████▎  | 59.5M/81.8M [00:55<00:24, 963kB/s]

.. parsed-literal::

     73%|███████▎  | 59.7M/81.8M [00:56<00:23, 979kB/s]

.. parsed-literal::

     73%|███████▎  | 59.8M/81.8M [00:56<00:23, 1.00MB/s]

.. parsed-literal::

     73%|███████▎  | 59.9M/81.8M [00:56<00:23, 993kB/s] 

.. parsed-literal::

     73%|███████▎  | 60.0M/81.8M [00:56<00:22, 1.01MB/s]

.. parsed-literal::

     73%|███████▎  | 60.1M/81.8M [00:56<00:22, 1.02MB/s]

.. parsed-literal::

     74%|███████▎  | 60.2M/81.8M [00:56<00:23, 944kB/s] 

.. parsed-literal::

     74%|███████▎  | 60.3M/81.8M [00:56<00:22, 1.01MB/s]

.. parsed-literal::

     74%|███████▍  | 60.4M/81.8M [00:56<00:22, 1.01MB/s]

.. parsed-literal::

     74%|███████▍  | 60.5M/81.8M [00:56<00:22, 1.01MB/s]

.. parsed-literal::

     74%|███████▍  | 60.6M/81.8M [00:57<00:21, 1.01MB/s]

.. parsed-literal::

     74%|███████▍  | 60.8M/81.8M [00:57<00:21, 1.01MB/s]

.. parsed-literal::

     74%|███████▍  | 60.9M/81.8M [00:57<00:21, 1.02MB/s]

.. parsed-literal::

     75%|███████▍  | 61.0M/81.8M [00:57<00:21, 1.02MB/s]

.. parsed-literal::

     75%|███████▍  | 61.1M/81.8M [00:57<00:21, 1.00MB/s]

.. parsed-literal::

     75%|███████▍  | 61.2M/81.8M [00:57<00:21, 1.02MB/s]

.. parsed-literal::

     75%|███████▍  | 61.3M/81.8M [00:57<00:21, 988kB/s] 

.. parsed-literal::

     75%|███████▌  | 61.4M/81.8M [00:57<00:22, 957kB/s]

.. parsed-literal::

     75%|███████▌  | 61.5M/81.8M [00:57<00:21, 989kB/s]

.. parsed-literal::

     75%|███████▌  | 61.6M/81.8M [00:58<00:20, 1.01MB/s]

.. parsed-literal::

     75%|███████▌  | 61.7M/81.8M [00:58<00:20, 1.02MB/s]

.. parsed-literal::

     76%|███████▌  | 61.8M/81.8M [00:58<00:19, 1.05MB/s]

.. parsed-literal::

     76%|███████▌  | 62.0M/81.8M [00:58<00:19, 1.05MB/s]

.. parsed-literal::

     76%|███████▌  | 62.1M/81.8M [00:58<00:19, 1.06MB/s]

.. parsed-literal::

     76%|███████▌  | 62.2M/81.8M [00:58<00:19, 1.07MB/s]

.. parsed-literal::

     76%|███████▌  | 62.3M/81.8M [00:58<00:19, 1.06MB/s]

.. parsed-literal::

     76%|███████▋  | 62.4M/81.8M [00:58<00:18, 1.09MB/s]

.. parsed-literal::

     76%|███████▋  | 62.5M/81.8M [00:58<00:18, 1.09MB/s]

.. parsed-literal::

     77%|███████▋  | 62.6M/81.8M [00:59<00:18, 1.10MB/s]

.. parsed-literal::

     77%|███████▋  | 62.8M/81.8M [00:59<00:17, 1.13MB/s]

.. parsed-literal::

     77%|███████▋  | 62.9M/81.8M [00:59<00:18, 1.09MB/s]

.. parsed-literal::

     77%|███████▋  | 63.0M/81.8M [00:59<00:17, 1.14MB/s]

.. parsed-literal::

     77%|███████▋  | 63.1M/81.8M [00:59<00:21, 894kB/s] 

.. parsed-literal::

     77%|███████▋  | 63.3M/81.8M [00:59<00:15, 1.23MB/s]

.. parsed-literal::

     78%|███████▊  | 63.4M/81.8M [00:59<00:15, 1.24MB/s]

.. parsed-literal::

     78%|███████▊  | 63.6M/81.8M [00:59<00:15, 1.20MB/s]

.. parsed-literal::

     78%|███████▊  | 63.7M/81.8M [01:00<00:15, 1.21MB/s]

.. parsed-literal::

     78%|███████▊  | 63.8M/81.8M [01:00<00:15, 1.21MB/s]

.. parsed-literal::

     78%|███████▊  | 64.0M/81.8M [01:00<00:15, 1.20MB/s]

.. parsed-literal::

     78%|███████▊  | 64.1M/81.8M [01:00<00:15, 1.22MB/s]

.. parsed-literal::

     78%|███████▊  | 64.2M/81.8M [01:00<00:14, 1.23MB/s]

.. parsed-literal::

     79%|███████▊  | 64.3M/81.8M [01:00<00:14, 1.24MB/s]

.. parsed-literal::

     79%|███████▉  | 64.5M/81.8M [01:00<00:14, 1.25MB/s]

.. parsed-literal::

     79%|███████▉  | 64.6M/81.8M [01:00<00:14, 1.25MB/s]

.. parsed-literal::

     79%|███████▉  | 64.7M/81.8M [01:00<00:14, 1.25MB/s]

.. parsed-literal::

     79%|███████▉  | 64.8M/81.8M [01:00<00:14, 1.25MB/s]

.. parsed-literal::

     79%|███████▉  | 65.0M/81.8M [01:01<00:14, 1.24MB/s]

.. parsed-literal::

     80%|███████▉  | 65.1M/81.8M [01:01<00:15, 1.13MB/s]

.. parsed-literal::

     80%|███████▉  | 65.2M/81.8M [01:01<00:17, 1.01MB/s]

.. parsed-literal::

     80%|███████▉  | 65.3M/81.8M [01:01<00:15, 1.09MB/s]

.. parsed-literal::

     80%|████████  | 65.4M/81.8M [01:01<00:18, 926kB/s] 

.. parsed-literal::

     80%|████████  | 65.5M/81.8M [01:01<00:19, 867kB/s]

.. parsed-literal::

     80%|████████  | 65.6M/81.8M [01:01<00:20, 820kB/s]

.. parsed-literal::

     80%|████████  | 65.7M/81.8M [01:02<00:21, 776kB/s]

.. parsed-literal::

     80%|████████  | 65.8M/81.8M [01:02<00:22, 747kB/s]

.. parsed-literal::

     81%|████████  | 65.9M/81.8M [01:02<00:21, 762kB/s]

.. parsed-literal::

     81%|████████  | 66.0M/81.8M [01:02<00:22, 748kB/s]

.. parsed-literal::

     81%|████████  | 66.1M/81.8M [01:02<00:21, 777kB/s]

.. parsed-literal::

     81%|████████  | 66.1M/81.8M [01:02<00:21, 769kB/s]

.. parsed-literal::

     81%|████████  | 66.2M/81.8M [01:02<00:21, 772kB/s]

.. parsed-literal::

     81%|████████  | 66.3M/81.8M [01:02<00:20, 779kB/s]

.. parsed-literal::

     81%|████████  | 66.4M/81.8M [01:02<00:21, 759kB/s]

.. parsed-literal::

     81%|████████▏ | 66.5M/81.8M [01:03<00:20, 773kB/s]

.. parsed-literal::

     81%|████████▏ | 66.5M/81.8M [01:03<00:20, 773kB/s]

.. parsed-literal::

     81%|████████▏ | 66.6M/81.8M [01:03<00:20, 778kB/s]

.. parsed-literal::

     82%|████████▏ | 66.7M/81.8M [01:03<00:20, 785kB/s]

.. parsed-literal::

     82%|████████▏ | 66.8M/81.8M [01:03<00:19, 794kB/s]

.. parsed-literal::

     82%|████████▏ | 66.9M/81.8M [01:03<00:19, 801kB/s]

.. parsed-literal::

     82%|████████▏ | 67.0M/81.8M [01:03<00:19, 807kB/s]

.. parsed-literal::

     82%|████████▏ | 67.0M/81.8M [01:03<00:19, 788kB/s]

.. parsed-literal::

     82%|████████▏ | 67.1M/81.8M [01:03<00:18, 828kB/s]

.. parsed-literal::

     82%|████████▏ | 67.2M/81.8M [01:04<00:18, 834kB/s]

.. parsed-literal::

     82%|████████▏ | 67.3M/81.8M [01:04<00:18, 840kB/s]

.. parsed-literal::

     82%|████████▏ | 67.4M/81.8M [01:04<00:17, 857kB/s]

.. parsed-literal::

     83%|████████▎ | 67.5M/81.8M [01:04<00:17, 874kB/s]

.. parsed-literal::

     83%|████████▎ | 67.6M/81.8M [01:04<00:16, 885kB/s]

.. parsed-literal::

     83%|████████▎ | 67.7M/81.8M [01:04<00:16, 894kB/s]

.. parsed-literal::

     83%|████████▎ | 67.8M/81.8M [01:04<00:16, 880kB/s]

.. parsed-literal::

     83%|████████▎ | 67.9M/81.8M [01:04<00:16, 867kB/s]

.. parsed-literal::

     83%|████████▎ | 68.0M/81.8M [01:04<00:15, 911kB/s]

.. parsed-literal::

     83%|████████▎ | 68.1M/81.8M [01:05<00:16, 897kB/s]

.. parsed-literal::

     83%|████████▎ | 68.2M/81.8M [01:05<00:15, 920kB/s]

.. parsed-literal::

     83%|████████▎ | 68.3M/81.8M [01:05<00:14, 958kB/s]

.. parsed-literal::

     84%|████████▎ | 68.4M/81.8M [01:05<00:14, 951kB/s]

.. parsed-literal::

     84%|████████▎ | 68.5M/81.8M [01:05<00:14, 956kB/s]

.. parsed-literal::

     84%|████████▍ | 68.6M/81.8M [01:05<00:14, 954kB/s]

.. parsed-literal::

     84%|████████▍ | 68.7M/81.8M [01:05<00:14, 973kB/s]

.. parsed-literal::

     84%|████████▍ | 68.8M/81.8M [01:05<00:15, 863kB/s]

.. parsed-literal::

     84%|████████▍ | 68.9M/81.8M [01:05<00:14, 913kB/s]

.. parsed-literal::

     84%|████████▍ | 69.0M/81.8M [01:06<00:15, 857kB/s]

.. parsed-literal::

     84%|████████▍ | 69.1M/81.8M [01:06<00:16, 823kB/s]

.. parsed-literal::

     85%|████████▍ | 69.2M/81.8M [01:06<00:15, 834kB/s]

.. parsed-literal::

     85%|████████▍ | 69.2M/81.8M [01:06<00:15, 837kB/s]

.. parsed-literal::

     85%|████████▍ | 69.3M/81.8M [01:06<00:16, 801kB/s]

.. parsed-literal::

     85%|████████▍ | 69.4M/81.8M [01:06<00:15, 815kB/s]

.. parsed-literal::

     85%|████████▍ | 69.5M/81.8M [01:06<00:15, 845kB/s]

.. parsed-literal::

     85%|████████▌ | 69.6M/81.8M [01:06<00:15, 809kB/s]

.. parsed-literal::

     85%|████████▌ | 69.7M/81.8M [01:07<00:14, 867kB/s]

.. parsed-literal::

     85%|████████▌ | 69.8M/81.8M [01:07<00:14, 887kB/s]

.. parsed-literal::

     85%|████████▌ | 69.9M/81.8M [01:07<00:13, 897kB/s]

.. parsed-literal::

     86%|████████▌ | 70.0M/81.8M [01:07<00:13, 917kB/s]

.. parsed-literal::

     86%|████████▌ | 70.1M/81.8M [01:07<00:13, 935kB/s]

.. parsed-literal::

     86%|████████▌ | 70.2M/81.8M [01:07<00:12, 946kB/s]

.. parsed-literal::

     86%|████████▌ | 70.3M/81.8M [01:07<00:12, 955kB/s]

.. parsed-literal::

     86%|████████▌ | 70.4M/81.8M [01:07<00:12, 933kB/s]

.. parsed-literal::

     86%|████████▌ | 70.5M/81.8M [01:07<00:12, 946kB/s]

.. parsed-literal::

     86%|████████▋ | 70.6M/81.8M [01:07<00:12, 970kB/s]

.. parsed-literal::

     86%|████████▋ | 70.7M/81.8M [01:08<00:12, 965kB/s]

.. parsed-literal::

     87%|████████▋ | 70.8M/81.8M [01:08<00:11, 982kB/s]

.. parsed-literal::

     87%|████████▋ | 70.9M/81.8M [01:08<00:11, 1.00MB/s]

.. parsed-literal::

     87%|████████▋ | 71.0M/81.8M [01:08<00:11, 1.02MB/s]

.. parsed-literal::

     87%|████████▋ | 71.1M/81.8M [01:08<00:11, 991kB/s] 

.. parsed-literal::

     87%|████████▋ | 71.2M/81.8M [01:08<00:11, 1.00MB/s]

.. parsed-literal::

     87%|████████▋ | 71.3M/81.8M [01:08<00:10, 1.01MB/s]

.. parsed-literal::

     87%|████████▋ | 71.4M/81.8M [01:08<00:11, 970kB/s] 

.. parsed-literal::

     87%|████████▋ | 71.5M/81.8M [01:08<00:10, 1.01MB/s]

.. parsed-literal::

     88%|████████▊ | 71.7M/81.8M [01:09<00:10, 1.01MB/s]

.. parsed-literal::

     88%|████████▊ | 71.8M/81.8M [01:09<00:10, 1.02MB/s]

.. parsed-literal::

     88%|████████▊ | 71.9M/81.8M [01:09<00:10, 1.00MB/s]

.. parsed-literal::

     88%|████████▊ | 72.0M/81.8M [01:09<00:10, 1.01MB/s]

.. parsed-literal::

     88%|████████▊ | 72.1M/81.8M [01:09<00:10, 1.00MB/s]

.. parsed-literal::

     88%|████████▊ | 72.2M/81.8M [01:09<00:09, 1.01MB/s]

.. parsed-literal::

     88%|████████▊ | 72.3M/81.8M [01:09<00:10, 988kB/s] 

.. parsed-literal::

     89%|████████▊ | 72.4M/81.8M [01:09<00:09, 1.03MB/s]

.. parsed-literal::

     89%|████████▊ | 72.5M/81.8M [01:09<00:09, 1.02MB/s]

.. parsed-literal::

     89%|████████▉ | 72.6M/81.8M [01:10<00:09, 1.00MB/s]

.. parsed-literal::

     89%|████████▉ | 72.7M/81.8M [01:10<00:09, 1.02MB/s]

.. parsed-literal::

     89%|████████▉ | 72.8M/81.8M [01:10<00:09, 1.03MB/s]

.. parsed-literal::

     89%|████████▉ | 72.9M/81.8M [01:10<00:08, 1.04MB/s]

.. parsed-literal::

     89%|████████▉ | 73.1M/81.8M [01:10<00:08, 1.04MB/s]

.. parsed-literal::

     89%|████████▉ | 73.2M/81.8M [01:10<00:08, 1.06MB/s]

.. parsed-literal::

     90%|████████▉ | 73.3M/81.8M [01:10<00:08, 1.06MB/s]

.. parsed-literal::

     90%|████████▉ | 73.4M/81.8M [01:10<00:08, 1.03MB/s]

.. parsed-literal::

     90%|████████▉ | 73.5M/81.8M [01:10<00:08, 1.06MB/s]

.. parsed-literal::

     90%|█████████ | 73.6M/81.8M [01:11<00:07, 1.11MB/s]

.. parsed-literal::

     90%|█████████ | 73.7M/81.8M [01:11<00:07, 1.12MB/s]

.. parsed-literal::

     90%|█████████ | 73.8M/81.8M [01:11<00:07, 1.13MB/s]

.. parsed-literal::

     90%|█████████ | 74.0M/81.8M [01:11<00:07, 1.13MB/s]

.. parsed-literal::

     91%|█████████ | 74.1M/81.8M [01:11<00:07, 1.13MB/s]

.. parsed-literal::

     91%|█████████ | 74.2M/81.8M [01:11<00:07, 1.13MB/s]

.. parsed-literal::

     91%|█████████ | 74.3M/81.8M [01:11<00:06, 1.15MB/s]

.. parsed-literal::

     91%|█████████ | 74.4M/81.8M [01:11<00:06, 1.15MB/s]

.. parsed-literal::

     91%|█████████ | 74.5M/81.8M [01:11<00:07, 996kB/s] 

.. parsed-literal::

     91%|█████████▏| 74.7M/81.8M [01:12<00:06, 1.22MB/s]

.. parsed-literal::

     92%|█████████▏| 74.9M/81.8M [01:12<00:05, 1.22MB/s]

.. parsed-literal::

     92%|█████████▏| 75.0M/81.8M [01:12<00:06, 1.18MB/s]

.. parsed-literal::

     92%|█████████▏| 75.1M/81.8M [01:12<00:05, 1.18MB/s]

.. parsed-literal::

     92%|█████████▏| 75.2M/81.8M [01:12<00:05, 1.16MB/s]

.. parsed-literal::

     92%|█████████▏| 75.3M/81.8M [01:12<00:05, 1.19MB/s]

.. parsed-literal::

     92%|█████████▏| 75.5M/81.8M [01:12<00:05, 1.26MB/s]

.. parsed-literal::

     92%|█████████▏| 75.6M/81.8M [01:12<00:05, 1.27MB/s]

.. parsed-literal::

     93%|█████████▎| 75.8M/81.8M [01:12<00:04, 1.28MB/s]

.. parsed-literal::

     93%|█████████▎| 75.9M/81.8M [01:13<00:04, 1.30MB/s]

.. parsed-literal::

     93%|█████████▎| 76.0M/81.8M [01:13<00:04, 1.29MB/s]

.. parsed-literal::

     93%|█████████▎| 76.1M/81.8M [01:13<00:04, 1.25MB/s]

.. parsed-literal::

     93%|█████████▎| 76.3M/81.8M [01:13<00:04, 1.30MB/s]

.. parsed-literal::

     93%|█████████▎| 76.4M/81.8M [01:13<00:04, 1.32MB/s]

.. parsed-literal::

     94%|█████████▎| 76.5M/81.8M [01:13<00:04, 1.32MB/s]

.. parsed-literal::

     94%|█████████▍| 76.7M/81.8M [01:13<00:04, 1.09MB/s]

.. parsed-literal::

     94%|█████████▍| 76.9M/81.8M [01:13<00:04, 1.29MB/s]

.. parsed-literal::

     94%|█████████▍| 77.0M/81.8M [01:13<00:03, 1.27MB/s]

.. parsed-literal::

     94%|█████████▍| 77.1M/81.8M [01:14<00:04, 1.19MB/s]

.. parsed-literal::

     94%|█████████▍| 77.2M/81.8M [01:14<00:04, 1.16MB/s]

.. parsed-literal::

     95%|█████████▍| 77.4M/81.8M [01:14<00:04, 1.15MB/s]

.. parsed-literal::

     95%|█████████▍| 77.5M/81.8M [01:14<00:03, 1.14MB/s]

.. parsed-literal::

     95%|█████████▍| 77.6M/81.8M [01:14<00:03, 1.16MB/s]

.. parsed-literal::

     95%|█████████▌| 77.7M/81.8M [01:14<00:03, 1.16MB/s]

.. parsed-literal::

     95%|█████████▌| 77.9M/81.8M [01:14<00:03, 1.18MB/s]

.. parsed-literal::

     95%|█████████▌| 78.0M/81.8M [01:14<00:03, 1.21MB/s]

.. parsed-literal::

     96%|█████████▌| 78.1M/81.8M [01:15<00:03, 1.20MB/s]

.. parsed-literal::

     96%|█████████▌| 78.2M/81.8M [01:15<00:03, 1.23MB/s]

.. parsed-literal::

     96%|█████████▌| 78.4M/81.8M [01:15<00:02, 1.27MB/s]

.. parsed-literal::

     96%|█████████▌| 78.5M/81.8M [01:15<00:02, 1.28MB/s]

.. parsed-literal::

     96%|█████████▌| 78.6M/81.8M [01:15<00:02, 1.27MB/s]

.. parsed-literal::

     96%|█████████▋| 78.8M/81.8M [01:15<00:02, 1.30MB/s]

.. parsed-literal::

     96%|█████████▋| 78.9M/81.8M [01:15<00:02, 1.29MB/s]

.. parsed-literal::

     97%|█████████▋| 79.0M/81.8M [01:15<00:02, 1.32MB/s]

.. parsed-literal::

     97%|█████████▋| 79.2M/81.8M [01:15<00:02, 1.36MB/s]

.. parsed-literal::

     97%|█████████▋| 79.3M/81.8M [01:15<00:01, 1.37MB/s]

.. parsed-literal::

     97%|█████████▋| 79.5M/81.8M [01:16<00:01, 1.36MB/s]

.. parsed-literal::

     97%|█████████▋| 79.6M/81.8M [01:16<00:01, 1.38MB/s]

.. parsed-literal::

     98%|█████████▊| 79.8M/81.8M [01:16<00:01, 1.38MB/s]

.. parsed-literal::

     98%|█████████▊| 79.9M/81.8M [01:16<00:01, 1.18MB/s]

.. parsed-literal::

     98%|█████████▊| 80.0M/81.8M [01:16<00:01, 1.29MB/s]

.. parsed-literal::

     98%|█████████▊| 80.2M/81.8M [01:16<00:01, 1.22MB/s]

.. parsed-literal::

     98%|█████████▊| 80.3M/81.8M [01:16<00:01, 1.11MB/s]

.. parsed-literal::

     98%|█████████▊| 80.4M/81.8M [01:16<00:01, 1.08MB/s]

.. parsed-literal::

     98%|█████████▊| 80.5M/81.8M [01:17<00:01, 1.08MB/s]

.. parsed-literal::

     99%|█████████▊| 80.7M/81.8M [01:17<00:01, 1.09MB/s]

.. parsed-literal::

     99%|█████████▊| 80.8M/81.8M [01:17<00:00, 1.09MB/s]

.. parsed-literal::

     99%|█████████▉| 80.9M/81.8M [01:17<00:00, 1.03MB/s]

.. parsed-literal::

     99%|█████████▉| 81.0M/81.8M [01:17<00:00, 1.10MB/s]

.. parsed-literal::

     99%|█████████▉| 81.1M/81.8M [01:17<00:00, 933kB/s] 

.. parsed-literal::

     99%|█████████▉| 81.3M/81.8M [01:17<00:00, 1.20MB/s]

.. parsed-literal::

    100%|█████████▉| 81.4M/81.8M [01:17<00:00, 1.22MB/s]

.. parsed-literal::

    100%|█████████▉| 81.6M/81.8M [01:17<00:00, 1.20MB/s]

.. parsed-literal::

    100%|█████████▉| 81.7M/81.8M [01:18<00:00, 1.21MB/s]

.. parsed-literal::

    100%|██████████| 81.8M/81.8M [01:18<00:00, 1.10MB/s]

.. parsed-literal::

    


Cleaning up the model directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

From the verbose of the previous step it is obvious that
```torch.hub.load`` <https://pytorch.org/docs/stable/hub.html#torch.hub.load>`__
downloads a lot of unnecessary files. We shall move remove the
unnecessary directories and files which were created during the download
process.

.. code:: ipython3

    # Remove unnecessary directories and files and suppress errors(if any)
    rmtree(path=str(MODEL_DIR / 'intel-isl_MiDaS_master'), ignore_errors=True)
    rmtree(path=str(MODEL_DIR / 'rwightman_gen-efficientnet-pytorch_master'), ignore_errors=True)
    rmtree(path=str(MODEL_DIR / 'checkpoints'), ignore_errors=True)
    
    # Check for the existence of the trusted list file and then remove
    list_file = Path(MODEL_DIR / 'trusted_list')
    if list_file.is_file():
        list_file.unlink()

Transformation of models
~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Each of the models need an appropriate transformation which can be
invoked by the ``get_model_transforms`` function. It needs only the
``depth_predictor`` parameter and ``NSAMPLES`` defined above to work.
The reason being that the ``ScaleMapLearner`` and the depth estimation
model are always in direct correspondence with each other.

.. code:: ipython3

    # Define important custom types
    type_transform_compose = torchvision.transforms.transforms.Compose
    type_compiled_model = ov.CompiledModel

.. code:: ipython3

    def get_model_transforms(depth_predictor: str, nsamples: int) -> Tuple[type_transform_compose, type_transform_compose]:
        """
        Construct the transformation of the depth prediction model and the 
        associated ScaleMapLearner model
    
        :param: depth_predictor: Any depth estimation model amongst the ones given at https://github.com/isl-org/VI-Depth#setup
        :param: nsamples: The no. of density levels for the depth map
        :returns: The transformed models as the resut of torchvision.transforms.Compose operations
        """    
        model_transforms = transforms.get_transforms(depth_predictor, "void", str(nsamples))
        return model_transforms['depth_model'], model_transforms['sml_model']    

.. code:: ipython3

    # Obtain transforms of both the models here
    depth_model_transform, scale_map_learner_transform = get_model_transforms(depth_predictor='midas_small',
                                                                              nsamples=NSAMPLES)

Dummy input creation
^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

Dummy inputs help during conversion. Although ``ov.convert_model``
accepts any dummy input for a single pass through the model and thereby
enabling model conversion, the pre-processing required for the actual
inputs later at inference using compiled models would be substantial. So
we have decided that even dummy inputs should go through the proper
transformation process so that the reader gets the idea of a
*transformed image* being compiled by a *transformed model*.

Also note down the width and height of the image which would be used
multiple times later. Do note that this is constant throughout the
dataset

.. code:: ipython3

    IMAGE_H, IMAGE_W = 480, 640
    
    # Although you can always verify the same by uncommenting and running
    # the following lines
    # img = cv2.imread('data/image/dummy_img.png')
    # print(img.shape)

.. code:: ipython3

    # Base directory in which data would be stored as a pathlib.Path variable
    DATA_DIR = Path('data')
    
    # Create the data directory tree adjacent to the notebook and suppress errors if the directory already exists
    # Create a directory each for the images and their corresponding depth maps
    DATA_DIR.mkdir(exist_ok=True)
    Path(DATA_DIR / 'image').mkdir(exist_ok=True)
    Path(DATA_DIR / 'sparse_depth').mkdir(exist_ok=True)
    
    # Download the dummy image and its depth scale (take a note of the image hashes for possible later use)
    # On the fly download is being done to avoid unnecessary memory/data load during testing and 
    # creation of PRs
    download_file('https://user-images.githubusercontent.com/22426058/254174385-161b9f0e-5991-4308-ba89-d81bc02bcb7c.png', filename='dummy_img.png', directory=Path(DATA_DIR / 'image'), silent=True)
    download_file('https://user-images.githubusercontent.com/22426058/254174398-8c71c59f-0adf-43c6-ad13-c04431e02349.png', filename='dummy_depth.png', directory=Path(DATA_DIR / 'sparse_depth'), silent=True)
    
    # Load the dummy image and its depth scale
    dummy_input = data_loader.load_input_image('data/image/dummy_img.png')
    dummy_depth = data_loader.load_sparse_depth('data/sparse_depth/dummy_depth.png')



.. parsed-literal::

    data/image/dummy_img.png:   0%|          | 0.00/328k [00:00<?, ?B/s]



.. parsed-literal::

    data/sparse_depth/dummy_depth.png:   0%|          | 0.00/765 [00:00<?, ?B/s]


.. code:: ipython3

    def transform_image_for_depth(input_image: np.ndarray, depth_model_transform: np.ndarray, device: torch.device = 'cpu') -> torch.Tensor:
        """
        Transform the input_image for processing by a PyTorch depth estimation model
    
        :param: input_image: The input image obtained as a result of data_loader.load_input_image
        :param: depth_model_transform: The transformed depth model
        :param: device: The device on which the image would be allocated
        :returns: The transformed image suitable to be used as an input to the depth estimation model
        """
        input_height, input_width = np.shape(input_image)[:2]
            
        sample = {'image' : input_image}
        sample = depth_model_transform(sample)
        im = sample['image'].to(device)
        return im.unsqueeze(0)    

.. code:: ipython3

    # Transform the dummy input image for the depth model
    transformed_dummy_image = transform_image_for_depth(input_image=dummy_input, depth_model_transform=depth_model_transform)

Conversion of depth model to OpenVINO IR format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

Starting from 2023.0.0 release, OpenVINO supports PyTorch model via
conversion to OpenVINO Intermediate Representation format (IR). To have
a depth estimation model in the OpenVINO™ IR format and then compile it,
we shall follow the following steps:

1. Use the ``depth_model`` callable to our advantage from the *Loading
   models and checkpoints* stage.
2. Convert PyTorch model to OpenVINO model using OpenVINO Model
   conversion API and the transformed dummy input created earlier.
3. Use the ``ov.save_model`` function from OpenVINO to serialize
   OpenVINO ``.xml`` and ``.bin`` files for next compilation skipping
   conversion step Alternatively serialization procedure may be avoided
   and compiled model may be obtained by directly using OpenVINO’s
   ``compile_model`` function.

.. code:: ipython3

    # Evaluate the model to switch some operations from training mode to inference.
    depth_model.eval()
    
    
    # Check PyTorch model work with dummy input
    _ = depth_model(transformed_dummy_image)
    
    # convert model to OpenVINO IR
    ov_model = ov.convert_model(depth_model, example_input=(transformed_dummy_image, ))
    
    # save model for next usage
    ov.save_model(ov_model, 'depth_model.xml')

Select inference device
'''''''''''''''''''''''

`back to top ⬆️ <#Table-of-contents:>`__

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



Compilation of depth model
''''''''''''''''''''''''''

`back to top ⬆️ <#Table-of-contents:>`__

Now we can go ahead and compile our depth model.

.. code:: ipython3

    # Initialize OpenVINO Runtime.
    compiled_depth_model = core.compile_model(model=ov_model, device_name=device.value)

.. code:: ipython3

    def run_depth_model(input_image_h: int, input_image_w: int, 
                        transformed_image: torch.Tensor, compiled_depth_model: type_compiled_model) -> np.ndarray:
        
        """
        Run the compiled_depth_model on the transformed_image of dimensions 
        input_image_w x input_image_h 
    
        :param: input_image_h: The height of the input image 
        :param: input_image_w: The width of the input image 
        :param: transformed_image: The transformed image suitable to be used as an input to the depth estimation model
        :returns:
                 depth_pred: The depth prediction on the image as an np.ndarray type 
        
        """
        
        # Obtain the last output layer separately
        output_layer_depth_model = compiled_depth_model.output(0)    
        
        with torch.no_grad():
            # Perform computation like a standard OpenVINO compiled model
            depth_pred = torch.from_numpy(compiled_depth_model([transformed_image])[output_layer_depth_model])
            depth_pred = (
                torch.nn.functional.interpolate(
                    depth_pred.unsqueeze(1),
                    size=(input_image_h, input_image_w),
                    mode='bicubic',
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        
        return depth_pred    

.. code:: ipython3

    # Run the compiled depth model using the dummy input 
    # It will be used to compute the metrics associated with the ScaleMapLearner model
    # and hence obtain a compiled version of the same later
    depth_pred_dummy = run_depth_model(input_image_h=IMAGE_H, input_image_w=IMAGE_W,
                                       transformed_image=transformed_dummy_image, compiled_depth_model=compiled_depth_model)

Computation of scale and shift parameters
'''''''''''''''''''''''''''''''''''''''''

`back to top ⬆️ <#Table-of-contents:>`__

Computation of these parameters required the depth estimation model
output from the previous step. These are the regression based parameters
the ScaleMapLearner model deals with. An utility function for the
purpose has already been created.

.. code:: ipython3

    def compute_global_scale_and_shift(input_sparse_depth: np.ndarray, validity_map: Optional[np.ndarray], 
                                       depth_pred: np.ndarray,
                                       min_pred: float = 0.1, max_pred: float = 8.0,
                                       min_depth: float = 0.2, max_depth: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:    
        
        """
        Compute the global scale and shift alignment required for SML model to work on
        with the input_sparse_depth map being provided and the depth estimation output depth_pred
        being provided with an optional validity_map
    
        :param: input_sparse_depth: The depth map of the input image 
        :param: validity_map: An optional depth map associated with the original input image 
        :param: depth_pred: The depth estimate obtained after running the depth model on the input image
        :param: min_pred: Lower bound for predicted depth values 
        :param: max_pred: Upper bound for predicted depth values 
        :param: min_depth: Min valid depth when evaluating
        :param: max_depth: Max valid depth when evaluating
        :returns:
                 int_depth: The depth estimate for the SML regression model
                 int_scales: The scale to be used for the SML regression model
        
        """
        
        input_sparse_depth_valid = (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)
        if validity_map is not None:
            input_sparse_depth_valid *= validity_map.astype(np.bool)
    
        input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
        input_sparse_depth[~input_sparse_depth_valid] = np.inf  # set invalid depth
        input_sparse_depth = 1.0 / input_sparse_depth    
        
        # global scale and shift alignment
        GlobalAlignment = LeastSquaresEstimator(
            estimate=depth_pred,
            target=input_sparse_depth,
            valid=input_sparse_depth_valid
        )
        GlobalAlignment.compute_scale_and_shift()
        GlobalAlignment.apply_scale_and_shift()
        GlobalAlignment.clamp_min_max(clamp_min=min_pred, clamp_max=max_pred)
        int_depth = GlobalAlignment.output.astype(np.float32)    
    
        # interpolation of scale map
        assert (np.sum(input_sparse_depth_valid) >= 3), 'not enough valid sparse points'    
        ScaleMapInterpolator = Interpolator2D(
            pred_inv=int_depth,
            sparse_depth_inv=input_sparse_depth,
            valid=input_sparse_depth_valid,
        )
        ScaleMapInterpolator.generate_interpolated_scale_map(
            interpolate_method='linear', 
            fill_corners=False
        )
        
        int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
        int_scales = utils.normalize_unit_range(int_scales)
        
        return int_depth, int_scales    

.. code:: ipython3

    # Call the function on the dummy depth map we loaded in the dummy_depth variable
    # with all default settings and store in appropriate variables
    d_depth, d_scales = compute_global_scale_and_shift(input_sparse_depth=dummy_depth, validity_map=None, depth_pred=depth_pred_dummy)

.. code:: ipython3

    def transform_image_for_depth_scale(input_image: np.ndarray, scale_map_learner_transform: type_transform_compose, 
                                        int_depth: np.ndarray, int_scales: np.ndarray, 
                                        device: torch.device = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform the input_image for processing by a PyTorch SML model
    
        :param: input_image: The input image obtained as a result of data_loader.load_input_image
        :param: scale_map_learner_transform: The transformed SML model
        :param: int_depth: The depth estimate for the SML regression model
        :param: int_scales: he scale to be used for the SML regression model
        :param: device: The device on which the image would be allocated
        :returns: The transformed tensor inputs suitable to be used with an SML model
        """
        
        sample = {'image' : input_image, 'int_depth' : int_depth, 'int_scales' : int_scales, 'int_depth_no_tf' : int_depth}
        sample = scale_map_learner_transform(sample)
        x = torch.cat([sample['int_depth'], sample['int_scales']], 0)
        x = x.to(device)
        d = sample['int_depth_no_tf'].to(device)
        
        return x.unsqueeze(0), d.unsqueeze(0)    

.. code:: ipython3

    # Transform the dummy input image for the ScaleMapLearner model
    # Note that this will lead to a tuple as an output. Both the elements
    # which is fed to ScaleMapLearner during the conversion process to onxx
    transformed_dummy_image_scale = transform_image_for_depth_scale(input_image=dummy_input,
                                                                    scale_map_learner_transform=scale_map_learner_transform,
                                                                    int_depth=d_depth, int_scales=d_scales)

Conversion of Scale Map Learner model to OpenVINO IR format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

The OpenVINO™ toolkit provides direct method of converting PyTorch
models to the intermediate representation format. To have the associated
ScaleMapLearner in the OpenVINO™ IR format and then compile it, we shall
follow the following steps:

1. Load the model in memory via instantiating the
   ``modules.midas.midas_net_custom.MidasNet_small_videpth`` class and
   passing the downloaded checkpoint earlier as an argument.
2. Convert PyTorch model to OpenVINO model using OpenVINO Model
   conversion API and the transformed dummy input created earlier.
3. Use the ``ov.save_model`` function from OpenVINO to serialize
   OpenVINO ``.xml`` and ``.bin`` files for next compilation skipping
   conversion step Alternatively serialization procedure may be avoided
   and compiled model may be obtained by directly using OpenVINO’s
   ``compile_model`` function.

If the name of the ``.ckpt`` file is too much to handle, here is the
common format of all checkpoint files from the model releases.

   -  sml_model.dpredictor.<DEPTH_PREDICTOR>.nsamples.<NSAMPLES>.ckpt
   -  Replace <DEPTH_PREDICTOR> and <NSAMPLES> with the depth estimation
      model name and the no. of levels of depth density the SML model
      has been trained on
   -  E.g. sml_model.dpredictor.dpt_hybrid.nsamples.500.ckpt will be the
      file name corresponding to the SML model based on the dpt_hybrid
      depth predictor and has been trained on 500 points of the density
      level on the depth map

.. code:: ipython3

    # Run with the same min_pred and max_pred arguments which were used to compute
    # global scale and shift alignment
    scale_map_learner = MidasNet_small_videpth(path=str(MODEL_DIR / 'sml_model.dpredictor.midas_small.nsamples.1500.ckpt'),
                                               min_pred=0.1, max_pred=8.0)


.. parsed-literal::

    Loading weights:  model/sml_model.dpredictor.midas_small.nsamples.1500.ckpt


.. parsed-literal::

    Downloading: "https://github.com/rwightman/gen-efficientnet-pytorch/zipball/master" to model/master.zip


.. parsed-literal::

    2024-03-14 00:01:17.089818: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-14 00:01:17.124320: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-14 00:01:17.685982: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. code:: ipython3

    # As usual, since the MidasNet_small_videpthc class internally downloads a repo again from torch hub
    # we shall clean the same since the model callable is now available to us
    # Remove unnecessary directories and files and suppress errors(if any)
    rmtree(path=str(MODEL_DIR / 'rwightman_gen-efficientnet-pytorch_master'), ignore_errors=True)
    
    # Check for the existence of the trusted list file and then remove
    list_file = Path(MODEL_DIR / 'trusted_list')
    if list_file.is_file():
        list_file.unlink()

.. code:: ipython3

    # Evaluate the model to switch some operations from training mode to inference.
    scale_map_learner.eval()
    
    # Store the tuple of dummy variables into separate variables for easier reference
    x_dummy, d_dummy = transformed_dummy_image_scale
    
    # Check that PyTorch model works with dummy input
    _ = scale_map_learner(x_dummy, d_dummy)
    
    # Convert model to OpenVINO IR
    scale_map_learner = ov.convert_model(scale_map_learner, example_input=(x_dummy, d_dummy))
    
    # Save model on disk for next usage
    ov.save_model(scale_map_learner, "scale_map_learner.xml")


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


Select inference device
'''''''''''''''''''''''

`back to top ⬆️ <#Table-of-contents:>`__

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Compilation of the ScaleMapLearner(SML) model
'''''''''''''''''''''''''''''''''''''''''''''

`back to top ⬆️ <#Table-of-contents:>`__

Now we can go ahead and compile our SML model.

.. code:: ipython3

    # In the situation where you are unaware of the correct device to compile your
    # model in, just set device_name='AUTO' and let OpenVINO decide for you
    compiled_scale_map_learner = core.compile_model(model=scale_map_learner, device_name=device.value)

.. code:: ipython3

    def run_depth_scale_model(input_image_h: int, input_image_w: int, 
                              transformed_image_for_depth_scale: Tuple[torch.Tensor, torch.Tensor],
                              compiled_scale_map_learner: type_compiled_model) -> np.ndarray:
        
        """
        Run the compiled_scale_map_learner on the transformed image of dimensions 
        input_image_w x input_image_h suitable to be used on such a model
    
        :param: input_image_h: The height of the input image 
        :param: input_image_w: The width of the input image 
        :param: transformed_image_for_depth_scale: The transformed image inputs suitable to be used as an input to the SML model
        :returns:
                 sml_pred: The regression based prediction of the SML model 
        
        """
        
        # Obtain the last output layer separately
        output_layer_scale_map_learner = compiled_scale_map_learner.output(0)
        x_transform, d_transform = transformed_image_for_depth_scale
        
        with torch.no_grad():
            # Perform computation like a standard OpenVINO compiled model
            sml_pred = torch.from_numpy(compiled_scale_map_learner([x_transform, d_transform])[output_layer_scale_map_learner])
            sml_pred = (
                torch.nn.functional.interpolate(
                    sml_pred,
                    size=(input_image_h, input_image_w),
                    mode='bicubic',
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            
        return sml_pred

.. code:: ipython3

    # Run the compiled SML model using the set of dummy inputs 
    sml_pred_dummy = run_depth_scale_model(input_image_h=IMAGE_H, input_image_w=IMAGE_W,
                                           transformed_image_for_depth_scale=transformed_dummy_image_scale,
                                           compiled_scale_map_learner=compiled_scale_map_learner)

Storing and visualizing dummy results obtained
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    # Base directory in which outputs would be stored as a pathlib.Path variable
    OUTPUT_DIR = Path('output')
    
    # Create the output directory adjacent to the notebook and suppress errors if the directory already exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Utility functions are directly available in modules.midas.utils
    # Provide path names without any extension and let the write_depth
    # function provide them for you. Take note of the arguments.
    utils.write_depth(path=str(OUTPUT_DIR / 'dummy_input'), depth=d_depth, bits=2)
    utils.write_depth(path=str(OUTPUT_DIR / 'dummy_input_sml'), depth=sml_pred_dummy, bits=2)

.. code:: ipython3

    plt.figure()
    
    img_dummy_in = mpimg.imread('data/image/dummy_img.png')
    img_dummy_out = mpimg.imread(OUTPUT_DIR / 'dummy_input.png')
    img_dummy_sml_out = mpimg.imread(OUTPUT_DIR / 'dummy_input_sml.png')
    
    f, axes = plt.subplots(1, 3)
    plt.subplots_adjust(right=2.0)
    
    axes[0].imshow(img_dummy_in)
    axes[1].imshow(img_dummy_out)
    axes[2].imshow(img_dummy_sml_out)
    
    axes[0].set_title('dummy input')
    axes[1].set_title('depth prediction on dummy input')
    axes[2].set_title('SML on depth estimate')




.. parsed-literal::

    Text(0.5, 1.0, 'SML on depth estimate')




.. parsed-literal::

    <Figure size 640x480 with 0 Axes>



.. image:: 246-depth-estimation-videpth-with-output_files/246-depth-estimation-videpth-with-output_48_2.png


Running inference on a test image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Now role of both the dummy inputs i.e. the dummy image as well as its
associated depth map is now over. Since we have access to the compiled
models now, we can load the *one* image available to us for pure
inferencing purposes and run all the above steps one by one till
plotting of the depth map.

If you haven’t noticed already the data directory of this tutorial has
been arranged as follows. This allows us to comply to these
`rules <https://github.com/pronoym99/openvino_notebooks/blob/main/CONTRIBUTING.md#file-structure>`__.

.. code:: bash

      data
      ├── image                   
      │   ├── dummy_img.png       # RGB images
      │   └── <timestamp>.png
      └── sparse_depth            
          ├── dummy_img.png       # sparse metric depth maps
          └── <timestamp>.png     # as 16b PNG files

At the same time, the depth storage method `used in the VOID
dataset <https://github.com/alexklwong/void-dataset/blob/master/src/data_utils.py>`__
is assumed.

If you are thinking of the file name format of the image for inference,
here is the reasoning.

The dataset was collected using the Intel `RealSense D435i
camera <https://realsense.intel.com/depth-camera>`__, which was
configured to produce synchronized accelerometer and gyroscope
measurements at 400 Hz, along with synchronized VGA-size (640 x 480) RGB
and depth streams at 30 Hz. The depth frames are acquired using active
stereo and is aligned to the RGB frame using the sensor factory
calibration. The frequency of sensor and depth stream input run at
certain fixed frequencies and hence time stamping every frame captured
is beneficial for maintaining structure as well as for debugging
purposes later.

*The image for inference and it sparse depth map is taken from the
compressed dataset
present*\ `here <https://drive.google.com/uc?id=1bbN46kR_hcH3GG8-jGRqAI433uddYrnc>`__

.. code:: ipython3

    # As before download the sample images for inference and take note of the image hashes if you 
    # want to use them later
    download_file('https://user-images.githubusercontent.com/22426058/254174393-fc6dcc5f-f677-4618-b2ef-22e8e5cb1ebe.png', filename='1552097950.2672.png', directory=Path(DATA_DIR / 'image'), silent=True)
    download_file('https://user-images.githubusercontent.com/22426058/254174379-5d00b66b-57b4-4e96-91e9-36ef15ec5a0a.png', filename='1552097950.2672.png', directory=Path(DATA_DIR / 'sparse_depth'), silent=True)
    
    # Load the image and its depth scale  
    img_input = data_loader.load_input_image('data/image/1552097950.2672.png')
    img_depth_input = data_loader.load_sparse_depth('data/sparse_depth/1552097950.2672.png')
    
    # Transform the input image for the depth model
    transformed_image = transform_image_for_depth(input_image=img_input, depth_model_transform=depth_model_transform)
    
    # Run the depth model on the transformed input
    depth_pred = run_depth_model(input_image_h=IMAGE_H, input_image_w=IMAGE_W,
                                 transformed_image=transformed_image, compiled_depth_model=compiled_depth_model)
    
    
    # Call the function on the sparse depth map
    # with all default settings and store in appropriate variables
    int_depth, int_scales = compute_global_scale_and_shift(input_sparse_depth=img_depth_input, validity_map=None, depth_pred=depth_pred)
    
    # Transform the input image for the ScaleMapLearner model
    transformed_image_scale = transform_image_for_depth_scale(input_image=img_input,
                                                              scale_map_learner_transform=scale_map_learner_transform,
                                                              int_depth=int_depth, int_scales=int_scales)
    
    # Run the SML model using the set of inputs 
    sml_pred = run_depth_scale_model(input_image_h=IMAGE_H, input_image_w=IMAGE_W,
                                     transformed_image_for_depth_scale=transformed_image_scale,
                                     compiled_scale_map_learner=compiled_scale_map_learner)



.. parsed-literal::

    data/image/1552097950.2672.png:   0%|          | 0.00/371k [00:00<?, ?B/s]



.. parsed-literal::

    data/sparse_depth/1552097950.2672.png:   0%|          | 0.00/3.07k [00:00<?, ?B/s]


Store and visualize Inference results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    # Store the depth and SML predictions
    utils.write_depth(path=str(OUTPUT_DIR / '1552097950.2672'), depth=int_depth, bits=2)
    utils.write_depth(path=str(OUTPUT_DIR / '1552097950.2672_sml'), depth=sml_pred, bits=2)
    
    
    # Display result
    plt.figure()
    
    img_in = mpimg.imread('data/image/1552097950.2672.png')
    img_out = mpimg.imread(OUTPUT_DIR / '1552097950.2672.png')
    img_sml_out = mpimg.imread(OUTPUT_DIR / '1552097950.2672_sml.png')
    
    f, axes = plt.subplots(1, 3)
    plt.subplots_adjust(right=2.0)
    
    axes[0].imshow(img_in)
    axes[1].imshow(img_out)
    axes[2].imshow(img_sml_out)
    
    axes[0].set_title('Input image')
    axes[1].set_title('Depth prediction on input')
    axes[2].set_title('SML on depth estimate')




.. parsed-literal::

    Text(0.5, 1.0, 'SML on depth estimate')




.. parsed-literal::

    <Figure size 640x480 with 0 Axes>



.. image:: 246-depth-estimation-videpth-with-output_files/246-depth-estimation-videpth-with-output_53_2.png


Cleaning up the data directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

We will *follow suit* for the directory in which we downloaded images
and depth maps from another repo. We shall move remove the unnecessary
directories and files which were created during the download process.

.. code:: ipython3

    # Remove the data directory and suppress errors(if any)
    rmtree(path=str(DATA_DIR), ignore_errors=True)

Concluding notes
~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

   1. The code for this tutorial is adapted from the `VI-Depth
      repository <https://github.com/isl-org/VI-Depth>`__.
   2. Users may choose to download the original and raw datasets from
      the `VOID
      dataset <https://github.com/alexklwong/void-dataset/>`__.
   3. The `isl-org/VI-Depth <https://github.com/isl-org/VI-Depth>`__
      works on a slightly older version of released model assets from
      its `MiDaS sibling
      repository <https://github.com/isl-org/MiDaS>`__. However, the new
      releases beginning from
      `v3.1 <https://github.com/isl-org/MiDaS/releases/tag/v3_1>`__
      directly have OpenVINO™ ``.xml`` and ``.bin`` model files as their
      assets thereby rendering the **major pre-processing and model
      compilation step irrelevant**.
