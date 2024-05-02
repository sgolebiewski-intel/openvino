Image-to-Video synthesis with AnimateAnyone and OpenVINO
========================================================

|image0|

`AnimateAnyone <https://arxiv.org/pdf/2311.17117.pdf>`__ tackles the
task of generating animation sequences from a single character image. It
builds upon diffusion models pre-trained on vast character image
datasets.

The core of AnimateAnyone is a diffusion model pre-trained on a massive
dataset of character images. This model learns the underlying character
representation and distribution, allowing for realistic and diverse
character animation. To capture the specific details and characteristics
of the input character image, AnimateAnyone incorporates a ReferenceNet
module. This module acts like an attention mechanism, focusing on the
input image and guiding the animation process to stay consistent with
the original character’s appearance. AnimateAnyone enables control over
the character’s pose during animation. This might involve using
techniques like parametric pose embedding or direct pose vector input,
allowing for the creation of various character actions and movements. To
ensure smooth transitions and temporal coherence throughout the
animation sequence, AnimateAnyone incorporates temporal modeling
techniques. This may involve recurrent architectures like LSTMs or
transformers that capture the temporal dependencies between video
frames.

Overall, AnimateAnyone combines a powerful pre-trained diffusion model
with a character-specific attention mechanism (ReferenceNet), pose
guidance, and temporal modeling to achieve controllable, high-fidelity
character animation from a single image.

Learn more in `GitHub
repo <https://github.com/MooreThreads/Moore-AnimateAnyone>`__ and
`paper <https://arxiv.org/pdf/2311.17117.pdf>`__.

.. container:: alert alert-warning

   ::

      <p style="font-size:1.25em"><b>! WARNING !</b></p>
      <p>
          This tutorial requires at least <b>96 GB</b> of RAM for model conversion and <b>40 GB</b> for inference. Changing the values of <code>HEIGHT</code>, <code>WIDTH</code> and <code>VIDEO_LENGTH</code> variables will change the memory consumption but will also affect accuracy.
      </p>

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Prepare base model <#prepare-base-model>`__
-  `Prepare image encoder <#prepare-image-encoder>`__
-  `Download weights <#download-weights>`__
-  `Initialize models <#initialize-models>`__
-  `Load pretrained weights <#load-pretrained-weights>`__
-  `Convert model to OpenVINO IR <#convert-model-to-openvino-ir>`__

   -  `VAE <#vae>`__
   -  `Reference UNet <#reference-unet>`__
   -  `Denoising UNet <#denoising-unet>`__
   -  `Pose Guider <#pose-guider>`__
   -  `Image Encoder <#image-encoder>`__

-  `Inference <#inference>`__
-  `Video post-processing <#video-post-processing>`__
-  `Interactive inference <#interactive-inference>`__

.. |image0| image:: ./animate-anyone.gif

Prerequisites
-------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    from pathlib import Path
    import requests
    
    
    REPO_PATH = Path("Moore-AnimateAnyone")
    if not REPO_PATH.exists():
        !git clone -q "https://github.com/itrushkin/Moore-AnimateAnyone.git"
    %pip install -q "torch>=2.1" torchvision einops omegaconf "diffusers<=0.24" transformers av accelerate "openvino>=2024.0" "nncf>=2.9.0" "gradio>=4.19" --extra-index-url "https://download.pytorch.org/whl/cpu"
    import sys
    
    sys.path.insert(0, str(REPO_PATH.resolve()))
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)
    %load_ext skip_kernel_extension


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Note that we clone a fork of original repo with tweaked forward methods.

.. code:: ipython3

    MODEL_DIR = Path("models")
    VAE_ENCODER_PATH = MODEL_DIR / "vae_encoder.xml"
    VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"
    REFERENCE_UNET_PATH = MODEL_DIR / "reference_unet.xml"
    DENOISING_UNET_PATH = MODEL_DIR / "denoising_unet.xml"
    POSE_GUIDER_PATH = MODEL_DIR / "pose_guider.xml"
    IMAGE_ENCODER_PATH = MODEL_DIR / "image_encoder.xml"
    
    WIDTH = 448
    HEIGHT = 512
    VIDEO_LENGTH = 24
    
    SHOULD_CONVERT = not all(
        p.exists()
        for p in [
            VAE_ENCODER_PATH,
            VAE_DECODER_PATH,
            REFERENCE_UNET_PATH,
            DENOISING_UNET_PATH,
            POSE_GUIDER_PATH,
            IMAGE_ENCODER_PATH,
        ]
    )

.. code:: ipython3

    from datetime import datetime
    from typing import Optional, Union, List, Callable
    import math
    
    from PIL import Image
    import openvino as ov
    from torchvision import transforms
    from einops import repeat
    from tqdm.auto import tqdm
    from einops import rearrange
    from omegaconf import OmegaConf
    from diffusers import DDIMScheduler
    from diffusers.image_processor import VaeImageProcessor
    from transformers import CLIPImageProcessor
    import torch
    import gradio as gr
    import ipywidgets as widgets
    import numpy as np
    
    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
    from src.utils.util import get_fps, read_frames
    from src.utils.util import save_videos_grid
    from src.pipelines.context import get_context_scheduler


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-671/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-671/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-671/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(


.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    from pathlib import PurePosixPath
    import gc
    import warnings
    
    from typing import Dict, Any
    from diffusers import AutoencoderKL
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import CLIPVisionModelWithProjection
    import nncf
    
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.models.pose_guider import PoseGuider


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino


Prepare base model
------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    local_dir = Path("./pretrained_weights/stable-diffusion-v1-5")
    local_dir.mkdir(parents=True, exist_ok=True)
    for hub_file in ["unet/config.json", "unet/diffusion_pytorch_model.bin"]:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            continue
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            subfolder=PurePosixPath(saved_path.parent.name),
            filename=PurePosixPath(saved_path.name),
            local_dir=local_dir,
        )

Prepare image encoder
---------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    local_dir = Path("./pretrained_weights")
    local_dir.mkdir(parents=True, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(saved_path.parent.name),
            filename=PurePosixPath(saved_path.name),
            local_dir=local_dir,
        )

Download weights
----------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse", local_dir="./pretrained_weights/sd-vae-ft-mse"
    )
    snapshot_download(
        repo_id="patrolli/AnimateAnyone",
        local_dir="./pretrained_weights",
    )



.. parsed-literal::

    Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]


.. code:: ipython3

    config = OmegaConf.load("Moore-AnimateAnyone/configs/prompts/animation.yaml")
    infer_config = OmegaConf.load("Moore-AnimateAnyone/" + config.inference_config)

Initialize models
-----------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
    reference_unet = UNet2DConditionModel.from_pretrained(config.pretrained_base_model_path, subfolder="unet")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    )
    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256))
    image_enc = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path)
    
    
    NUM_CHANNELS_LATENTS = denoising_unet.config.in_channels


.. parsed-literal::

    Some weights of the model checkpoint were not used when initializing UNet2DConditionModel: 
     ['conv_norm_out.weight, conv_norm_out.bias, conv_out.weight, conv_out.bias']


Load pretrained weights
-----------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

Convert model to OpenVINO IR
----------------------------

`back to top ⬆️ <#table-of-contents>`__ The pose sequence is initially
encoded using Pose Guider and fused with multi-frame noise, followed by
the Denoising UNet conducting the denoising process for video
generation. The computational block of the Denoising UNet consists of
Spatial-Attention, Cross-Attention, and Temporal-Attention, as
illustrated in the dashed box on the right. The integration of reference
image involves two aspects. Firstly, detailed features are extracted
through ReferenceNet and utilized for Spatial-Attention. Secondly,
semantic features are extracted through the CLIP image encoder for
Cross-Attention. Temporal-Attention operates in the temporal dimension.
Finally, the VAE decoder decodes the result into a video clip.

|image0|

The pipeline contains 6 PyTorch modules: - VAE encoder - VAE decoder -
Image encoder - Reference UNet - Denoising UNet - Pose Guider

For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__. Weight
compression aims to reduce the memory footprint of a model. models,
which require extensive memory to store the weights during inference,
can benefit from weight compression in the following ways:

-  enabling the inference of exceptionally large models that cannot be
   accommodated in the memory of the device;

-  improving the inference performance of the models by reducing the
   latency of the memory access when computing the operations with
   weights, for example, Linear layers.

`Neural Network Compression Framework
(NNCF) <https://github.com/openvinotoolkit/nncf>`__ provides 4-bit /
8-bit mixed weight quantization as a compression method. The main
difference between weights compression and full model quantization
(post-training quantization) is that activations remain floating-point
in the case of weights compression which leads to a better accuracy. In
addition, weight compression is data-free and does not require a
calibration dataset, making it easy to use.

``nncf.compress_weights`` function can be used for performing weights
compression. The function accepts an OpenVINO model and other
compression parameters.

More details about weights compression can be found in `OpenVINO
documentation <https://docs.openvino.ai/2023.3/weight_compression.html>`__.

.. |image0| image:: https://humanaigc.github.io/animate-anyone/static/images/f2_img.png

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    def cleanup_torchscript_cache():
        """
        Helper for removing cached model representation
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    warnings.simplefilter("ignore", torch.jit.TracerWarning)

VAE
~~~

`back to top ⬆️ <#table-of-contents>`__

The VAE model has two parts, an encoder and a decoder. The encoder is
used to convert the image into a low dimensional latent representation,
which will serve as the input to the U-Net model. The decoder,
conversely, transforms the latent representation back into an image.

During latent diffusion training, the encoder is used to get the latent
representations (latents) of the images for the forward diffusion
process, which applies more and more noise at each step. During
inference, the denoised latents generated by the reverse diffusion
process are converted back into images using the VAE decoder.

As the encoder and the decoder are used independently in different parts
of the pipeline, it will be better to convert them to separate models.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not VAE_ENCODER_PATH.exists():
        class VaeEncoder(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
        
            def forward(self, x):
                return self.vae.encode(x).latent_dist.mean
        vae.eval()
        with torch.no_grad():
            vae_encoder = ov.convert_model(VaeEncoder(vae), example_input=torch.zeros(1,3,512,448))
        vae_encoder = nncf.compress_weights(vae_encoder)
        ov.save_model(vae_encoder, VAE_ENCODER_PATH)
        del vae_encoder
        cleanup_torchscript_cache()


.. parsed-literal::

    WARNING:nncf:NNCF provides best results with torch==2.2.*, while current torch version is 2.3.0+cpu. If you encounter issues, consider switching to torch==2.2.*
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (32 / 32)              │ 100% (32 / 32)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not VAE_DECODER_PATH.exists():
        class VaeDecoder(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
        
            def forward(self, z):
                return self.vae.decode(z).sample
        vae.eval()
        with torch.no_grad():
            vae_decoder = ov.convert_model(VaeDecoder(vae), example_input=torch.zeros(1,4,HEIGHT//8,WIDTH//8))
        vae_decoder = nncf.compress_weights(vae_decoder)
        ov.save_model(vae_decoder, VAE_DECODER_PATH)
        del vae_decoder
        cleanup_torchscript_cache()
    del vae
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (40 / 40)              │ 100% (40 / 40)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Reference UNet
~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Pipeline extracts reference attention features from all transformer
blocks inside Reference UNet model. We call the original forward pass to
obtain shapes of the outputs as they will be used in the next pipeline
step.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not REFERENCE_UNET_PATH.exists():
        class ReferenceUNetWrapper(torch.nn.Module):
            def __init__(self, reference_unet):
                super().__init__()
                self.reference_unet = reference_unet
            
            def forward(self, sample, timestep, encoder_hidden_states):
                return self.reference_unet(sample, timestep, encoder_hidden_states, return_dict=False)[1]
                
        sample = torch.zeros(2, 4, HEIGHT // 8, WIDTH // 8)
        timestep = torch.tensor(0)
        encoder_hidden_states = torch.zeros(2, 1, 768)
        reference_unet.eval()
        with torch.no_grad():
            wrapper =  ReferenceUNetWrapper(reference_unet)
            example_input = (sample, timestep, encoder_hidden_states)
            ref_features_shapes = {k: v.shape for k, v in wrapper(*example_input).items()}
            ov_reference_unet = ov.convert_model(
                wrapper,
                example_input=example_input,
            )
        ov_reference_unet = nncf.compress_weights(ov_reference_unet)
        ov.save_model(ov_reference_unet, REFERENCE_UNET_PATH)
        del ov_reference_unet
        del wrapper
        cleanup_torchscript_cache()
    del reference_unet
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (270 / 270)            │ 100% (270 / 270)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Denoising UNet
~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Denoising UNet is the main part of all diffusion pipelines. This model
consumes the majority of memory, so we need to reduce its size as much
as possible.

Here we make all shapes static meaning that the size of the video will
be constant.

Also, we use the ``ref_features`` input with the same tensor shapes as
output of `Reference UNet <#reference-unet>`__ model on the previous
step.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not DENOISING_UNET_PATH.exists():
        class DenoisingUNetWrapper(torch.nn.Module):
            def __init__(self, denoising_unet):
                super().__init__()
                self.denoising_unet = denoising_unet
            
            def forward(
                self,
                sample,
                timestep,
                encoder_hidden_states,
                pose_cond_fea,
                ref_features
            ):
                return self.denoising_unet(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    ref_features,
                    pose_cond_fea=pose_cond_fea,
                    return_dict=False)
    
        example_input = {
            "sample": torch.zeros(2, 4, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8),
            "timestep": torch.tensor(999),
            "encoder_hidden_states": torch.zeros(2,1,768),
            "pose_cond_fea": torch.zeros(2, 320, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8),
            "ref_features": {k: torch.zeros(shape) for k, shape in ref_features_shapes.items()}
        }
        
        denoising_unet.eval()
        with torch.no_grad():
            ov_denoising_unet = ov.convert_model(
                DenoisingUNetWrapper(denoising_unet),
                example_input=tuple(example_input.values())
            )
        ov_denoising_unet.inputs[0].get_node().set_partial_shape(ov.PartialShape((2, 4, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8)))
        ov_denoising_unet.inputs[2].get_node().set_partial_shape(ov.PartialShape((2, 1, 768)))
        ov_denoising_unet.inputs[3].get_node().set_partial_shape(ov.PartialShape((2, 320, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8)))
        for ov_input, shape in zip(ov_denoising_unet.inputs[4:], ref_features_shapes.values()):
            ov_input.get_node().set_partial_shape(ov.PartialShape(shape))
            ov_input.get_node().set_element_type(ov.Type.f32)
        ov_denoising_unet.validate_nodes_and_infer_types()
        ov_denoising_unet = nncf.compress_weights(ov_denoising_unet)
        ov.save_model(ov_denoising_unet, DENOISING_UNET_PATH)
        del ov_denoising_unet
        cleanup_torchscript_cache()
    del denoising_unet
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (534 / 534)            │ 100% (534 / 534)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Pose Guider
~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

To ensure pose controllability, a lightweight pose guider is devised to
efficiently integrate pose control signals into the denoising process.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not POSE_GUIDER_PATH.exists():
        pose_guider.eval()
        with torch.no_grad():
            ov_pose_guider = ov.convert_model(pose_guider, example_input=torch.zeros(1, 3, VIDEO_LENGTH, HEIGHT, WIDTH))
        ov_pose_guider = nncf.compress_weights(ov_pose_guider)
        ov.save_model(ov_pose_guider, POSE_GUIDER_PATH)
        del ov_pose_guider
        cleanup_torchscript_cache()
    del pose_guider
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (8 / 8)                │ 100% (8 / 8)                           │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Image Encoder
~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

Pipeline uses CLIP image encoder to generate encoder hidden states
required for both reference and denoising UNets.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not IMAGE_ENCODER_PATH.exists():
        image_enc.eval()
        with torch.no_grad():
            ov_image_encoder = ov.convert_model(image_enc, example_input=torch.zeros(1, 3, 224, 224), input=(1, 3, 224, 224))
        ov_image_encoder = nncf.compress_weights(ov_image_encoder)
        ov.save_model(ov_image_encoder, IMAGE_ENCODER_PATH)
        del ov_image_encoder
        cleanup_torchscript_cache()
    del image_enc
    gc.collect()


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-671/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (146 / 146)            │ 100% (146 / 146)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Inference
---------

`back to top ⬆️ <#table-of-contents>`__

We inherit from the original pipeline modifying the calls to our models
to match OpenVINO format.

.. code:: ipython3

    core = ov.Core()

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#table-of-contents>`__

For starting work, please select inference device from dropdown list.

.. code:: ipython3

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    class OVPose2VideoPipeline(Pose2VideoPipeline):
        def __init__(
            self,
            vae_encoder_path=VAE_ENCODER_PATH,
            vae_decoder_path=VAE_DECODER_PATH,
            image_encoder_path=IMAGE_ENCODER_PATH,
            reference_unet_path=REFERENCE_UNET_PATH,
            denoising_unet_path=DENOISING_UNET_PATH,
            pose_guider_path=POSE_GUIDER_PATH,
            device=device.value,
        ):
            self.vae_encoder = core.compile_model(vae_encoder_path, device)
            self.vae_decoder = core.compile_model(vae_decoder_path, device)
            self.image_encoder = core.compile_model(image_encoder_path, device)
            self.reference_unet = core.compile_model(reference_unet_path, device)
            self.denoising_unet = core.compile_model(denoising_unet_path, device)
            self.pose_guider = core.compile_model(pose_guider_path, device)
            self.scheduler = DDIMScheduler(**OmegaConf.to_container(infer_config.noise_scheduler_kwargs))
    
            self.vae_scale_factor = 8
            self.clip_image_processor = CLIPImageProcessor()
            self.ref_image_processor = VaeImageProcessor(do_convert_rgb=True)
            self.cond_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=False)
    
        def decode_latents(self, latents):
            video_length = latents.shape[2]
            latents = 1 / 0.18215 * latents
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            # video = self.vae.decode(latents).sample
            video = []
            for frame_idx in tqdm(range(latents.shape[0])):
                video.append(torch.from_numpy(self.vae_decoder(latents[frame_idx : frame_idx + 1])[0]))
            video = torch.cat(video)
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
            video = (video / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            video = video.cpu().float().numpy()
            return video
    
        def __call__(
            self,
            ref_image,
            pose_images,
            width,
            height,
            video_length,
            num_inference_steps=30,
            guidance_scale=3.5,
            num_images_per_prompt=1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "tensor",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            context_schedule="uniform",
            context_frames=24,
            context_stride=1,
            context_overlap=4,
            context_batch_size=1,
            interpolation_factor=1,
            **kwargs,
        ):
            do_classifier_free_guidance = guidance_scale > 1.0
    
            # Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
    
            batch_size = 1
    
            # Prepare clip image embeds
            clip_image = self.clip_image_processor.preprocess(ref_image.resize((224, 224)), return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image)["image_embeds"]
            clip_image_embeds = torch.from_numpy(clip_image_embeds)
            encoder_hidden_states = clip_image_embeds.unsqueeze(1)
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
    
            if do_classifier_free_guidance:
                encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)
    
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                4,
                width,
                height,
                video_length,
                clip_image_embeds.dtype,
                torch.device("cpu"),
                generator,
            )
    
            # Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    
            # Prepare ref image latents
            ref_image_tensor = self.ref_image_processor.preprocess(ref_image, height=height, width=width)  # (bs, c, width, height)
            ref_image_latents = self.vae_encoder(ref_image_tensor)[0]
            ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
            ref_image_latents = torch.from_numpy(ref_image_latents)
    
            # Prepare a list of pose condition images
            pose_cond_tensor_list = []
            for pose_image in pose_images:
                pose_cond_tensor = self.cond_image_processor.preprocess(pose_image, height=height, width=width)
                pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
                pose_cond_tensor_list.append(pose_cond_tensor)
            pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=2)  # (bs, c, t, h, w)
            pose_fea = self.pose_guider(pose_cond_tensor)[0]
            pose_fea = torch.from_numpy(pose_fea)
    
            context_scheduler = get_context_scheduler(context_schedule)
    
            # denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    noise_pred = torch.zeros(
                        (
                            latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                            *latents.shape[1:],
                        ),
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                    counter = torch.zeros(
                        (1, 1, latents.shape[2], 1, 1),
                        device=latents.device,
                        dtype=latents.dtype,
                    )
    
                    # 1. Forward reference image
                    if i == 0:
                        ref_features = self.reference_unet(
                            (
                                ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                                torch.zeros_like(t),
                                # t,
                                encoder_hidden_states,
                            )
                        ).values()
    
                    context_queue = list(
                        context_scheduler(
                            0,
                            num_inference_steps,
                            latents.shape[2],
                            context_frames,
                            context_stride,
                            0,
                        )
                    )
                    num_context_batches = math.ceil(len(context_queue) / context_batch_size)
    
                    context_queue = list(
                        context_scheduler(
                            0,
                            num_inference_steps,
                            latents.shape[2],
                            context_frames,
                            context_stride,
                            context_overlap,
                        )
                    )
    
                    num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                    global_context = []
                    for i in range(num_context_batches):
                        global_context.append(context_queue[i * context_batch_size : (i + 1) * context_batch_size])
    
                    for context in global_context:
                        # 3.1 expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents[:, :, c] for c in context]).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        b, c, f, h, w = latent_model_input.shape
                        latent_pose_input = torch.cat([pose_fea[:, :, c] for c in context]).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
    
                        pred = self.denoising_unet(
                            (
                                latent_model_input,
                                t,
                                encoder_hidden_states[:b],
                                latent_pose_input,
                                *ref_features,
                            )
                        )[0]
    
                        for j, c in enumerate(context):
                            noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                            counter[:, :, c] = counter[:, :, c] + 1
    
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)
    
            if interpolation_factor > 0:
                latents = self.interpolate_latents(latents, interpolation_factor, latents.device)
            # Post-processing
            images = self.decode_latents(latents)  # (b, c, f, h, w)
    
            # Convert to tensor
            if output_type == "tensor":
                images = torch.from_numpy(images)
    
            return images

.. code:: ipython3

    pipe = OVPose2VideoPipeline()

.. code:: ipython3

    pose_images = read_frames("Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4")
    src_fps = get_fps("Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4")
    ref_image = Image.open("Moore-AnimateAnyone/configs/inference/ref_images/anyone-5.png").convert("RGB")
    pose_list = []
    for pose_image_pil in pose_images[:VIDEO_LENGTH]:
        pose_list.append(pose_image_pil)

.. code:: ipython3

    video = pipe(
        ref_image,
        pose_list,
        width=WIDTH,
        height=HEIGHT,
        video_length=VIDEO_LENGTH,
    )



.. parsed-literal::

      0%|          | 0/30 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/24 [00:00<?, ?it/s]


Video post-processing
---------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    new_h, new_w = video.shape[-2:]
    pose_transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
    pose_tensor_list = []
    for pose_image_pil in pose_images[:VIDEO_LENGTH]:
        pose_tensor_list.append(pose_transform(pose_image_pil))
    
    ref_image_tensor = pose_transform(ref_image)  # (c, h, w)
    ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
    ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=VIDEO_LENGTH)
    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
    pose_tensor = pose_tensor.transpose(0, 1)
    pose_tensor = pose_tensor.unsqueeze(0)
    video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
    
    save_dir = Path("./output")
    save_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    out_path = save_dir / f"{date_str}T{time_str}.mp4"
    save_videos_grid(
        video,
        str(out_path),
        n_rows=3,
        fps=src_fps,
    )

.. code:: ipython3

    from IPython.display import Video
    
    Video(out_path, embed=True)




.. raw:: html

    <video controls  >
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABFBxtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAaAZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VHXZB7WUcpIog829TuVootu4+xXoa+iVvI39vs8OeoqyyfZAubkL1wZJKUa7B+6GTv11m1IZLoDjuATS+wRFhcqaD5/k2aWyTlJDNAI5BHl0+sns/Xe3WPgNVk5Z0cZO34XcdY42NkHYdsIJQHg4Ieg56MrKE5a/lQ8OUnrcLkZMLX3yPAMK7hyB7ifIuCiLr97fFPUbQQOEfB2s0B714qxsbwpO1W79ensen3kLp+AAAeDoKPNm8j9nZicQG/68qKPa09KqwLJBMiPaxRpzRc9RVU2A6CdPStS7X5HlfivcjpQ7F5v6mKvagMzTvAdcekTTinV9hMvE4dbqBorO/9b5AxOOIdbRmSwBsMs1+KkEDUazh2i5He3L3zqzmB2eAQEeUX8zA0g2hwJC0A7itHkqpfsO9Jl7LNBww4XrlpqNIrtMPz1fyYajFmQHxUZYNPYWV6kGQLLFUEOtfQ76loVN6vNCma+DGCsoTdclQYrfIaOrhEm3FdUGuKBk2rHuhDf71UfNniUSckjIjrJ8C93KGHEjcRHjcIeISOIKrt+4D1uLgkaDPXqcHFO9HDEcyiMgO+8SAEc9T+zlB6iJTn5JCfhUUJ/reTly0x0+dtOWThJcIybNOh7xwA0c4/T2mybw7Vu/ah5DjRO8qalrrp09u5jlCgye9APKin1LACgXuTNu8jAQGUgemJAaWsWURTIqPrWcmVbHyiZt5vKrgCatbvau+Vg9/uO97paFFMZQU7hdmEzjQ7QtBwdCMblTsR0PKeON+hR7ih9biFVihk+qOSnLlGRgAbgGgX/GaGjLuptL0DvWlLE8HweI94QzA3GPeaDFbDL7nvZp37y+zH0SjBkvEq6OZ73PqX04v4NDs9DFx/zbonyAkMn3CvGh5TOd1sDzKt2L40ze43bN4qcm0NxM7E4g7uAAAJTeyhg9xB+c8wcr33XWEgamMKPLQxmV4P0S+Xh7qAntFEUXGRtuI/SO3goupPud8O1U3zQsr/kKcv5xlcqGOSyr8CybQLeC8nlzCkyhy7PBrBPmNo+eIsMoI/qtQEbtHoEbfWJ9jd3eHCOHOQKwGPt9XprwbTB8oSqDUg51JaCFqZsQFnlZq3dn5JSCnooF9btnfgk3wmPfRMdD4JhjGg1hc/TslQrkUICRRpxVk0tv9vFQDBXAX+cuFFoTsentBI3Sic9qRRygILaYMEZUud6c5qM4S09Emq2RgFKAjwTTwE0HKVi9SD+CwFHfqdTiwxwVLcABwc4jor37XROPHMbFpAYNTCh76EOW5xAmH7Y8HnXtix8HViZWVt8sgh8u86Q97eoveD9NsOVWONkitAcl6OFD/E9KSIJVJmf8jETdDi+Ze346JQKmrvB+i9MJTX6PEQ/2pF60deTP83EoShETEvAftQIUkSIv+rEcGtbxM/1SE/NcQmJbYunfyKtW6+R87nqJNojDd3v86rPEBnV7d2TrvD6FlHrENvUVy7O2Y7Tsy7qtH8TiAa0YWexXLLm/us8bgjUN8cCCluYl7Cc5ja2TSbgZcG+zCuW9bnAdPwj9yU0vvwCIGu1W+kH4KbaWqWtnj/jw6oODT2+7q84sEPhdIzsRW9h9ltTOP8AWiUJ7jiuM+hjgTc16qfzRlSybxhn/9m+RSpzSGS/bgvgp1W9kb8XddiguJZs0HJgfEw20BrMfTKP+9g0sC808KIcuxukyN7+2GwPzcKRrzou0dSBqZfHwO5JXaAjevvofwRgXhR9myWjfvy+xBWmnw13P0DMUE9OqtA6Fbz2MbXDWgpIp2Jhp+MWn4+f1ZhEIit4Em2/PDCLV//fMz2zpkicStsR/gW/CVSwtVFTFor8IS60zNntEZte6jlwjgumKnfzvCFCfq/rfZ7XII+wi53AmjJFbvbzA6XVaFrmgAAAwpxs30/zzfO6eSq9I4shTmZAttXAPdNP8edRdn0mCSafl9UXoXgBifSVTbAtPKp2mSk1jsPxkZihnWM5JKt1u9v8suHJuur+MMGmbKFLHya2n5Cqfx8yKbzFmoUt886X3NMSxfpwmmvrxD2KpRu7a8jsjFcJ5y+rAC5ERxLtiIf+dF+atHEUNuDgS+mFDMP7+h00AAAm4ZQCqiIQAc//RaeJPR9/+mopk2uoAmIETYEDrtY7YwgZScxjrkBb2imJABPTzsaVAhCwUESMzsL06p4+zWyDNVQByy0hp+jzCW5/WxK1wDbCAaHCly3aq67YSPAnO1PdJMRWVhC4IDqxAsfLRVH0jNdtOgkcie2vb2C5ED+6otkZo3ow7TArSBqO1PMJd82YLh8ClnGc17AK3Mj/CcXE3/DrLUPe9Q0x/1Zbde/4lf7LgGJ+JLBEU3AhLBGHOX44BeX8NMi3LzoDx7xtuUz+CA/L6zPD6mXqYyZjCYu9FJQuACDlpr6fZQ1569V/2CNPf4IebUFkCJXKWa4dBkSPXmeMWt8x5yHGTg6lUKThjmnd7L4Jtli1+RMrhDV2r4YKJL3010OygwDfHnz8PRg3FECgJzEBXtcFcf5n8y37cE/F72tUQ+lrakEpI0xNFjxGGNdzMHKK+Mryn8+dM9p7IItH9RsyAyiRAXU3hzZihgdf00ScAgdfBqbgUZyz4WExURmuFt6+r/8XAp9QCHZEw4AmrcG3QIMWsv8Yepe4kjd4z/YFHTyhZFXGr6wshhaL1Fr4Q9lU+KssNJMLafHLbMe78pq846OMwl9hS2LjOZik8YY7C3zdejQSjBUP9z3LQBUE7PIP3WAAf7IuMiFP88tHheEc7N/J7IG0yLiynOO8zEexpNmWiMdzyhas/ZeTMxXquNgQJ2jCQP014HxiLMu9TzQHdEy8Wpyuku5XObpq5pVdNxjPoKTeiFgAAtJ8DgNz/pJK1uKqLmQ/PcO4kh6FSPeDTztVWOgN2fMDysVdV7R/nwFWgev93KhVBk+yIm4yc8VDaOVPn6Onlv3HV6OHhx6p/IxCliYa6UsfqeHyLGDef9MyiyaN1f9KsCln84un3ON9Z2ikysQT/mY9rFmdf9jynD/vuwqso56/AJTFbOr3BkEY1fcEvQty7xMGr02Of3tUe7ASC/SaMCvQQ23zc1ACIFVJJoTF3/UOQ/wGUZ/BUxl9X3rivnKreRC6AI1fHydqzHi5xmFVX6xEkWRp4r3rPaN0yUs21NqfqZ9U18UlEIGTaVgITiT1D2wFR9AZhMda5H67ni1Z4wnXr6M9QSencFl4d6mMub2RdxKAhwvfxmmAW52oqgInJxn7tqq13c6QV2ME4ZSIE43FGVuQpR+HjWpN6JQdw7zfnttC8mxyO+yEKH5xJPOLajl76LUGX7UF7qTooYke3DlKSuLLFLGW++wbpIYAuz+nnIf85MHgzh9Usu8ljGVLwfaAM3YixNo182kdHYyjJTgGCoFlnZ2ceg3CZuQGTo1Nbf9RJe7M12RJoDC+GFPtWB+cIQJMi62ON69ChRhtfLoz3NiA+g2XK/vzlK2966L5DYZ81WXldia2Vmd2weCVV0RLfs2jkg3cwGjYtiTv1jKjx4DdCHUGw/BBWyNTRalGR5XVKWltM8v3ltKrhpCx9RI6u79VpUPNCHl4cGeEYYlA+U/pAdXKrBfdjQGq1a6E2gIpNeeeLibrlYj3cNtqrnQYW6lYnIluqwNh/u62fe/kP7aJwUwCZ+FVaZvCR3nB11L4Sc6k/ltjiQ31qh+YwJhJwCHoWeSkHxyeZRHCvl9VZcC3vW0k9XaX6I5DUELR8oQdBCEOvMo4/YCKeFZ3UiJ/yGEEzKgKSttfgQC2s7kB8ISkwXmu3K2elspm82ltNi/hRTEQIltQLaB1g06thvK6Cb36+rZQqRnk9kZmZB62tZ+/u0d9WPn9EC+5eclgXzLKsnd01xSLX1kZPATaTIYBufzaa3u0kfXWUbHHMkGyGXNIIL1ObmrnV+YAlUyBNdq0T64q3bTxmR32neEsFX88juqYNv1Io0jAsqaG5n6WAqiCLRORX228ZoujnjL7pzTYB/7MxZCmKMxGn8OjMr8PU0b74ZadZb1rz0g/ZFfegaL39bNo/PLMlBoyaCPakzUzVO+CEuz3/4KYT+8KwbrsKzHzN1zY46EKgbC+1YGkid+hG+U7Gj7tsR+6t+XsPDhPZfcNUBBRf18yISgRoi+yw+StBfKXG/vjhKkxVOklAkPpBzqK+HB7C6nDySeTh9nUg6G1Pcx7BURPkwZEZi53eKw0C1SLCa3Sqy2Df4LMBeTmnDEpxIVggsY7Xt302u07P3l127N5ImC7oOAaDwgqOabNQoDOQyw6UXiGCBVgdwn5O/JCkLCXi/08TUYT08q5nNik7m31F2HAKGlPCfmDUOxRH6DMThrsx+3dbyxV7vbVDJU9kobvMZ2vdZtwULE3yUgB6ZB8jVYWxoLNWgYaQ9gb3rcdLLHx3mrQxsd6x5vxmvPV2MQCbQaQUvZF64wS8+a6a+5d1K1kV6db6QKQHloJrjSN7RGLDzfUUm3PDg71mSu71EV7WTGzgDUwcPYPXHa3zKRRiQFW8q3TADeqR4NQakLvNJWUjriuEbU/wBpEjhWoXqBcI9YEemNPZzNP0trypVHd1ArFQzZ+DCYyYadaYEIQ41GW64HvWQ2OOCSCMElggQwupJs/+/ganQgBOqq284QiTk0KWaasY/NV3zaTTs9KZSdxd04r7o/sZUSM6PnZF0szv1YojZVpvT9P85EOP94zYNYhJeIFIUmJXsUyoqvFxTap+MH9EI6QsRRdn8ADLWOv7Gf12dy5UICLkF/d3mL/135mz2npD2KyM4GdItPCHg6hqDbSce0Uh9lSJ+5lAOfUsn2mAoMvMJ/J5OBEOSogZl8jFwjNi+a/X7+omSCzwMuWInkSApslWgpCP/xTCThGt8qHEXXcrBYCf4TYkdHW9vRplG/C+WljLAPPtFkFAAe3AVGKShTmAVeh1FhA9e8+fMrKMRD2wm/4LYEzCAzF/WjMQbDHxUieMKZTKcA1gipFYGE5ImGqa1V6XrDb4J+cmfZQAFl4pLeWmEIRt7TMnxqF6GnvWaBb+Vg6QYaZVJytnh7uJHVFjIi275I3XAB+qADPoUG4DIExC41+NJ6BH0BcmDOjf7ji27jhMmiWEfYH//PqC4SyIZ0HK/a2RXNH7tPVeen/Iy/Ny0t7e4d5+spc53f4d1Ympm1UmdPN5Y+vmADKSZ9nLwB065jvyDoVgl1imeQ/daUaArhaIhfzNB8jCA2xeDGrRtqNxzz2Dwd5eCBQiAZtpZNQ5uj6sgpYcO428DY7c8MAMO9zjOZ3SzGd1l3+yCOi4WjjCXfJBzp/D3dJnL+yGE0iodtGA2mEzOGmGDZyAf2WH4VhTKxP2eInBVDRwo5AYUv5P1HakR5hc8uN4TLnJr53pCB6KnP5YD9Zs51bivjYICxLsAF7HIwAACvdlAFUiIQAKP9aaKWsUL8ZXp0QoAJyrg+bJVWPuRaMzGSri5kPx0nJ+VktHKd6be6dkeVwuXlk4wQ4YY9SsTBSXswIXY4dJKsTrlXTeIy2CtO+c53fVwbBqYW5Z8Vh39F4GZ4iJaxHYlINSXEAkolw3RklRvUDKuDHosUbrXp08ELNAcXDHO1KbUXn/e2S0Hk6CLrbw/ohrXim8prtoulL6qFA4Nq55MHX72E970qQcHoH+7h2H0xHwS5tDNYp8BEvWIhcMX11BHY9/9TmdpEJJtXtiMSSnLZ0jWzQH4h52/H0qIv1YS3gSPyztCae41HFdKRY3oDH0ZCcmiROIQ01YGd4hzG4iEn+Wrb8nMOsafBA5gCpisUFn6e8Ndqk74Wbi9Ag3/Qh7/uqq+c6S7CtutrvPcCa6TP6lqiW88oeSpnzn+omrD04X/eXaScc/qvYLvfuCll0FzrG8wjudwDvN+CxpKz71swxkoAF6cHVx/yyeML7o93tmOL4fOONcpZtIAlCMmkB6TOHYzPQlAz/hY1DpsEpGuwzjVMjXP7Wjr9oi5zNmoy6Uf8JvqdsTY7teq2PA1KOcGMPJuZScUDmg7AZdbzx+7r/FmnFsJspPLbSKdD1JphSejBabFmN2DnFaDAVr88Qy2UzOCVnLJukKEiuFyQITL/a7SCavUxg7iK93d4nGKMdPDp8dj8fXuyRFgo0wrRzX5EaqPZZi7S60fBohB9Y09BL4AFYPan/Y2eeK75hQftxF1yCAGPU8p/cWbfafw+3WEXaJBGou7qZbiaHS2+a0x8jqV+Yu3HkdtsNq2/A6JinDECVaQ0UKBYsEJL0LO1/RGRbeyCU9g8UpjaIN66LCZ/TL3PWlJ/JK/xat98JMm4RDF+NY1zcrjBuMHsDxF5eGa0aFYKhb5j2Ff13CyK4Ae7sbZL18txG8oou6OxBBaDNg15tq6pUnIAHaV0mC9w5Ve7wF4yBN36kWwb76fQCriTZ79a/rGePeZCjgERtSFq56VXjEK2pJWgzdWhoNFsGMIy55DNPWd5bG8uj6mn8dhM/0Ck7a/+mI3yX7X3vBn8psTwDmyJpcaTU2KU6axeiEgn4Pcf7kmW5zH7DOvJYXM3i98rDaoBexwkdF2uMO5QF7YLKAkUNtY6KOK5kIsfXHkOBQ+Dfs3adyVMfMGccyTWHVyvQ7sEPH9FetcMZH8cY8Jlvs5NmnUhYAAqNiNtM8fdfZ3CVLYIu+ki4HfsXZeYsFKqC3zR6cH/8ksePBzGMd8hbXgJvTb8gMOAzOCaSBf8ufflh38H3QMdtXXZnjXlI0318IuILm3s6vKtj6BbCMVLKCT/QkrNdbqJ0wpU0wyH/0Rh+db32hVD71d7cjo1lKR5BSlTM4ROoqGgfVRN/CD/nC6zjTrVJTQ7giXDeiMtw8yI+tQlgyPf2jQiasGtljFXU1TvcJ8xAAB5vjHU6HLoYESr6sOZ4V6oFjOm1rKmVI2ljuB6LQA66YGogcH7VhXDI5y6bC5oBzkTQA85XWfg+DFDpDAhQ3tdzAIpmYbAWJvQBNGPKfpGqn+BSGMYzdNxzdgktFaxdGxJ0Y3Nkmo19oJZyWcZMbmb3MvppvVLumCqY09Wa4M9CZxrZb7BVoyY0csdXJqryn9fkVXe7NhXRd/eOaSEh/3DWPujrg1raACpud9LQbWLqXq0XjldgQYXZUkv66I0WlGITPUgZ5veMdaHDfk5zouE5YD1v4+fI9GBCzAlbLrzdWUkFz8sAfj5291acGdCE2QBu+ZrFODwUmVL+vvZY9ZEqMiYEdR1xklOdQHf+T09EjGa+7eqKCZiD09dZuga5znmIBfUvop1Xp5qRFCB8FrnEIvB3V/dC4AvQhRycg76URuH84mk1SRBrzc56yeR+hcNRzTQHd7G391iIfL+dm/t/TY/56+TAVjlRzfedBBQ8ii8FsN85D5+EIx59IerYqXnnCcUPBCrIo7lqXhGO9dfxbYeW1OTmDzIk4lXytmN2351fDNJVEs3NydQuPAzvQYhgi3eKmwzS16avu8Uyt5yCRXIKgLshjlvIPzILYH1oyd43bp5GcqnVRRAo0ev3beI010oIVqY7E+hzid3UZzjt3HqvafAwEeYU2+R1fdpMHUeuF1k9Onl1D2sNAjZiX3gaK7CPkNT7GpDaL2j7u7go7xCJ38YczoYqjcrEhHqbAbCmpr7JBz9Lnwqibiq126O59D+fv7uqoH/q/C9WLCdrokPVzl4CF1Ln9U3JkXGiA8B3yEYtcWfhoBnFC5ujs3k6ZtIJa/0DzMaxcB1KqKqZSWkjrSvALAArhjz0MttsJ6qkCqaU1T8HYPeyHM0u5pvAphjSh22glBxp8cpahfJg5OldV+8lRKhnvnlY4IN2rwenglBad283GgMSNyiLZsMjBZW2eGqzx7lyy5ErR1m/xcf/Z+pmzid8n951HuWdqXQWgZIcxD43KspXBTaoq4zrhJ8bHq+pqIOzaBnhRZSR38D5SVz5Zrls/zvOJCWWIDauuXDDOLFeaPb4uujJF1JmyowFczevlWSXwYuQDg0u/Gg6GW9SVYI/cvIBCTcGT6CAdbOe6dq2na/qk9rX01Q70tPejiwx6pgAvqPWP5N4cY8KPrktqqdGucGb5QSrrPpv830gb2UkdJ/5Z8sa1h3dynDzDOJ5TQlHjX29rFj4ByhY3gV1zsU9LL+7LvBDNLmIKzx5m7Zy95raou2YuR4PJo/liW23duIWtHxjv7FKCLmHatwkolyUf9h4YaChYbuHOKPEfRIWMeIXAoaMYsN4YJIZjfBNg8ky6OHuxgQMpRtdvFC9fACEu+zL1aYgK7nr+Bt4Y8Da0SAgq9yJ/kpT4iOD3M5F1QHLIHFXZRiQdlbM59HkYOn9LBaUGi8om39KXco32sdcCH6sh0LIy4c3oIya6u/JYVjHo1gPKWU6Zpyf49XZWmd1tY6jzOOnWz4lm5auKQ22t/MrxVxhW24YCWcsVuk97A4efGmnv3Eouo4G6WCREZvnJFTGiyN8Un6A3sqLw47YqPshjF1Y6uJfh8fXsrA+2QSTyyi/FmB6P7y4GVAiS71cbO24Gqn3fN1oglfmSoCKmPnH7bl+3DTKLrc9PPiqURvn5a65iRCn++fvGpCRd+9aBpeHgE+7YHu8TG+ojRVhJiyZ1gNJT2wX+IHoMPCVQlpZNV0uakpRRfNkwX7N4n0yKoZAX85yu0IrYaKI22c4AbZ8J/u6FGFsKVdkVBnfNRnBUZyVvJh4rAMXgtxEq5s/qJ3Z5ZMyV/JifdU9RwIFu4cnMG1Nd8U922dfkdyq9hp/6zmcF4muBZIuQ2jbr93b458kJXD4KuwMXFImoP0bwVyzr33H4FmYcXwiY2oL7nBh/xjVk2PSC3nXGIwKjfUf8nzB0n1W7ZX5EVwkuZmTumcwWpeJ1nU49ixy8C4fPVQocrZIDiYDWLzphzR48pZ5W4XnVRvm3aQqxzWV55Ihj0jlvPsAQz9wSKPCB43CNS+qo0870540zHpKSa6yhZsOCUZlnY0AK/4PktyPUI283timOqz21GH2gjHUiOCZtF6gxpT53kAtIq7XWCRDoe0Sj15bWPZ4h7j/aplTglP5mcgqROMO12Hx6FkP6TgYiz68w9yiPBluSGal/ahRW1HjLbROpqqJ6y4gKCZZFvFp+fGRdR7JtTclE5pNuYfuWSbBHPH+FfUv9ftfe7UaOg/8LmktrhYyxaekB4QAACahlAH+iIQAKP9aaKWsUL8ZXmpgcACGOqwl7r5okQfQ+D0kmihhn35duf08YCyoxfGHmzuLc8bUDrPtMdT1053y53z14BvOqlApmTqvh/7owPfvq3ho0BetIPtDJ7fGLL+gl+KabkhHnBt64+Duwbg1VY0KaMTDo1mC91/E29yoHIZf+IwMbN3y4W+Ikd6oL+UsQZnVOK23xyAtLVa3+tMiGJT0qHnNRSzpkhF3cljTUMMgKY/CvwF32gjJmyyJt2B/J9XwnEuhRMXfJvd+AOhz1yQTkfF9TB7dghwuyh5pxfaZ+J6m1jvAYyCUoze+CuFsGFPinGntJqtyD0oYTULPUdsDZFDBtYrARQwoup7gAJICO2Hk83mlbTCYPTixa9kQ0O82FKisWu/nFdHIwsV/ZCiAEfEDNhEPuMTahB/NJmeJlorRXi67ZtGp7dsiP188PGNeF4QarL/NVwm5pfjb/AP5Ku2s4c553O1+ugboIGRu8FQ5VJHhfGVWUghLN8Eg6oXDr3FkYvjCRsr5+FkFZIFEfrvquLXTBnxD5DKpLPz/K1fK8/0BMAZjqDGtG+rZD8kUEPemMD1ihlH8KhlWebQZeaZA+uF4ZrdLZ8eTQR/XltPl4FVd9ZOgDz5/s8AhvwXWXjCBqI+p3IZBS58euLxiEm9G+mWftCiAlRYSBQjiUAKYJwY7WzigRqthSXHG7Uyp+0YvWwv+G2nCyslyEuT1s4c0xh8rY4U+OeRJKieG25AJ+8rXdT3pUFMrIGWaVGBWNbo6ldROdAOopF/ObAUDSVzMF8HFEMe8yS/SDVO0LFYwLFJFNhjswtrQAmVfeqe+mXKfB0VV2Fq9dxNi7IzJfMCpMrgwlNfQIIxDqNHTqrdW4TRwi5qOHDd/k+padUK+bF5ukpoy6t89R6DxOe8AVWRfFsCpAAL/1r+OfFgxRrv4g1CP3I4O6NP+qhkAp2jSjqQGAG+fNDTnBPKp9IhNurRwvE5pO/7YveKWnEaWXPOO/iEPDOxSHCzZj4bz50vX6pNBQ56g1QEtRyCkrl6rYj5QwIWWASN48sMM6W0U4G9bK4RWMke5shlYSHnxMQYj2Fx2DWeMbScFR71k9xsIf0rmGayGgQQqwqeRvmmtHzwsqNVO0hjrQ7obOIg1ILqjtvxWL4wYff8ePS8IMqezr3HwmQ1PbIuoE86RwHDk3rAl8XsRKOxg6wbTQVmTRHgb0P7NyxnsQOPVH8dNMV9pah2kettJLPTtyWcuouYq1kpQPZV0VoGK0tNj/IAbyT9fsI4GbF6zTHKDVKjDXouO2XH45AhIetRR4G/C1p01XgB4gwtRjWgzjVhgC9zcA9O4/+JNqW0qYi8Ao7jBGy9cd7OwcRU+wHc6qkczryS/UUt+SdCzwbTiGfQlL7hTt90xx1qOHf2vPhNArZk5UUVFOgIoQVtvGNXF1dwPAbYL68g5WpUvD8GmJ6+Z1LbxryJHtqNnTc6ZC/saHgLp5Dcl0rcJSh/EcQ2ApqB3xvv733oMsjLbUOqhASlwuhjBxbkOBfae7DsMFaE4zIRfpdkRtVmd0sOjLgPAkoHBXuRO3+Fj/+OL3hT97DcDeIQGlcHkrJ1gF3LpXR0AOC8JhTpaYE8zuVsCZmlsLmMyHWbksBQ6f/VrSigjH1p3yMAoYzMXf8uGD+prEScuN2BW7KmQl97QTEPdLnmI9SHnVLlaKTxevIuXepZ/QTPxgclVgTT8cgx+Sb+HN0sGXmUPkE1RJrTiopi8mWBdCJ09F7O6i27BuAKMKlDCxnRkOc35qQqslfzQNGACI3wQpcj5kZReedShOKTkFmor9dOXVJbi0JRPYSmyFK5HQnoBeACJ1gu5mtxcWaWqiJFuwY9Q+A4qMgYNLRuQvHBqDdP+0KxM5kYodhiINBrl0ps4KolDMRBIeJMvQw4ckz/yyyQDF/K//YGu2bSunm9QsRNWTuYr4OR+ROXowrQ5kEGKLO+rs7qFh9V4yH7iRuaKFP/CrCf5qj7iPD3B80/Jvlk5w7N3QeiL2iOej9pi/U7CWqtp1vqv0fGcQ0Cc3AvUpwflc4aUZKh915TMdDWtGrnI/gGsStm4n2zoZ794ukaHe3nghINyQVSar/vZtmyEuc/hvJQwuNZYNaI70sKBEwDS2zGuabDbh12K2QOmq78vBgOfcl/XMiGCF+hJrzwXRyNv7zAQ1YJkXRqqVgDolhWcqc/ogjB0eFYVEStxhJLENHcbP4XxPnhjmj6IaeBj27UfcrMU0gXnKsU5QPH6poCj21nU0GugNvZPuXpTbxvOVinoOqbMU7JS6XftU3iFeEHxQDqyZIbmjm8LTF147RlL4dYDE79MJ9dO16JPc+Kkp/m9xZIGy9s2cUdyXFyFn+frqZXxqL0e3C0k/PKY3Wg29SZ+bmecdCE61YXYomCUinQcsGtmMX3k3cREM4ApsKyufzwlPbTnVTV0D84b+SFXF6YeedXoLu2zoqbFEIdV4Os0lTG5XW2SkLojgi5PJdUpx4CLad/BhkIDvEuSvwxWOriynb9Rqgk98nguW02tIpPj1ECmWEQBiBEpUWuBvabUk9FzyhNKU2Q1gmb4W2foNH/XPegiUMoz3wnJPSpZCNwHuOYxxk06hxdZ1YanG5PsCt4GnfyUi7qWs8CFDEscD2TLRFmozZr08+mX7zQko3nHVOayNa6WmRrXI5RnqixRwffnxafTch/Tfwq7bu5tpBw4OwGJNyovlgzE3bUQaw/HlkqbqXdQ138OioOoHscOigHuWiPffKgEwwtp+859uNKE7X0tK7AgEv0QzbAje3XSrK+uPg1N/ECQFGdCk9J53+Wk5SEQKvJBFi1Dln6g0RmE/9zWCWGQCa/4GQaMIaWaE5CLpOw8AvmWp31ESKapaXA8fqZAbsgP1h2Y0JJQlUxMo5d6Fe9eTYMkASmsafGG/5eEHrRKpLHADMfA4SqVAi8snrjUfIJ3j9cJ8xXXwgvxjM9H6dh8Wuj2bWCn3VRWLGMVfpcTohdOa7W+zRkVDXzNcM1YJBFKWS/1l8xTdZ6TVZQOFFclRtsj+8mRpyKMVcBAn6TR/YwYtOnrXetdoop2dsIZuB/M0/IwO0NJXGHS0BMaFYIGzHys3NDoM9HB1hiEkbRWmpK7ATMVsPsdrEt8BqaC+TD1V7orteuRumeGmt4ZwKfjrAYm3wEp1u6mX2zY1vK9vDgF7PJFkywjD61rH4MfQtdWQoEMMpkTpbywqcvptVgcYIx4kS/opB9k2lX6YlURRjLEL9REfyhtAZUZv4SP26qEAAAbuZQAtMIhAAo/WmilrFC/GV5ZDrOAL/5HAOXvFgox+7f3H4nHrbLL9osbucQpukWEZJzIRxh+LTTqNuxgVMJjaaNNurDUnDdxhrEqHmqZnhFmeuyCs+2NIOdIJTIMdQZEpYua1+aBMV+Ohhca1MKD3hdlBLoFySHwmHfHq9YDh+GKjGRNtYs/LSPefQbID2UW4aVMwyQvwQ7LG3coEhTRBU34W4NjpFidHvo0v/PZzvLJHC2AQP2rHctWCrzOGhYDWbvMKKs39l6rbFIJ1jWw+tN6htgM/WVx+kIUbC6Y/V6XheKbABARuASKrlWzHmw+hILv28BzpEG2tr1o06uMlTRNlMxr8ltVyCxoepigeu3i5hJE9XVFioc1VpLxGvtMgDzYkQ9jM9uPHrMXaYedzZyGYSwxtaoAJWJOu8LWj8q2lwoh8kd3hyZ0qloFmL+x3PJ8yTtINgkjoZjjVCrqN0q+PoPqOiFJHVKKxCMdU0yX0nISIuv1y8lAjxaIxyGpjiulubISsvACnot7hyWDuxuxXKIavyCaPWL8UaYnPwWolNidDJ4YY3QA+Ti2qkl1Xx4JZ5IOr7esk/8kFGbWl939CiYTHFla2/ePfBUHnuKDU//dZ7In9lCG9kNSIMt45r4/vxBcNEqBgeB27tUjxRpX6jLpol7obA9Onc61mXyK2nb6k+bRwsfVsi42FY7ZEeUrEKoZAOn10aKegbik6Ea4dnOjuVsR1ukIl/hWgrcbDb6DY75aRjJ3Vp++DEIp4wV6rBdTxlIKYskYZKXO7WDZNT1PUu/BSYlXg2rU9Fb8Nb18J+q9OhOQ4oOI+UyU524nLf1ylkFp+nBVRxnKmtLqDV027k5x374BVudkZpqMdGVi3JiBtnrwID1EKMItAeaN0DYiF2wqt36zy4W4Ky1VMKAYxmn5+cOvvzpJFlA/OBb9mzXurzWLLfppAbjFxDDipJQHq57hk2M+sjTv+JAg/kscAotnJAw2zx0EVmK8ljChF3uBkv4Pi879MWt2z42gZdWcjTuejKcSXX4YbLG1l5gy3JQ40Q6pYV9ZL0YuWBDOsSs8FeXBTg0WqIyRDLIfVGtwMPFEeP2IyjhbEX45vQ2mkYQKhdmy4k24xyLn/QYbNgO7S8Td48+V6r/zTxAYHbl2rMZnGkSEPZ3HzcOZzZ/UnfLcSnUyWfaCocK+MHJ5oXhDU2ZBPYFVdxiSr2LrELaTVZ4WGdE0pW2yqVU9dygwVo4aZ3W2GqmLuvc5Z3cKBdIUDqhtnWPSv28wiTNNHdcmvKwIHM6j2bHpAXVxptooD5+PuCHq8ccN4RQ1+f4X82smpYXkN7/uBOxiRvUK+/zuXH+BjrqFGaXdxBad1E0oSd3OngU4qu/H+Sbn9luzmpWTWvzdydKQZV6AdAJ+PmNRKhubJa75Djx/tNiho1RohXtwdhGb3Qt956dfoJcIoopldauOi9fvOyoHFqhLw912YLFp9lbL8ki8LCnGcRM0w1vG27HQzfZCAS0SO+oww9p1ECiGEbPtGRtVGcyF9bwr7j3YxX+o/WVYkvtKuQ65ghe5ZsSb0fR/1tXdezwO1w6p20Mrva0izSaTh/XZNbSzHJt2WH+Cl5twENJlZxrQpqgUIOa2hQFSJoEslmbloDLD18LhltCIBI/GjzvfxmMvDQoyATMdnRfpB+/RLkOFIp6FVLirFoG1onpX5D5dga8FsmFR9m1S3wHPBXKqDE3KUdalDaZlp0DuY1/S3pz5Pw/VgThrP5eZynxinFFaZhKq2f94lln1DSJu6xuU+mTbGxHuHYMYR3m4R4jGn4Bg67SOkf3TJT5Mm7JP/P/nQWEX1h/rhjgzTmH+bXOizWg9j+YZcmc4s/4j0ueL/YKf10CFW7dHnwIqr41H81Uvbg0F7vxwqHTcEbAB/+w5aQ+C6cELNDbsYDFFhbRGYjYG27H3fL+HeZ2rOFgLhQuz12Rmot8LajyYGyWx44Thu4s65eI3/RtAqFP9nl42fowmXD8lcj5DqgNuEA1cJHYHx1bxjS200kFlziN5f1Bp+Iu2DZNgKSeHKBlc0kjXtwxNDnx1zR8TZDCKKHMkOQTRP+XURGr/ODz2lzUupBkW5Ev3wD11beinexIrRgyHjf4GSuFJQo0hKwhAEpuHf47VGpAGuQAwtYtqblfDH2GgaUmnPLVrlTra6Jw3nrO9QT/Vei/qKjfQt9/0FV1lPurpxJMlwXZRWHtaYmW+4Ckm8ABZSbTDB8VOdD+o8XJ/B4E+ATnXtlziHbBPKnNO3dMRNai5GFH2+Ql+4J8U2IdXbKoAe7LYaY5Otc3CNIgt0nKPebSy/xsu5witrIx+RewKpJkgzWt6mjSzHgQAABOVlADfQiEACj9aaKWsUL8ZXp0QoAMenMSR/t3N5m4VUlpjkc16WQ9aXQ7eUtubC0yC5AyHQwFCA/W+fWBc6RSvcmmIcyVeApNfHl9D+EsJC59DPbAms6trxoOZOQUwMrPRUdQYhUZc8LQKYLZoNu+JvJc0tj05rayHwFzTMq4RhonMx+nH/yez346QzOvPs+flrH20pveKJk/qjN8tNMcr4pab+eVfr/g5sdtwYiDcQzD97M0pT/xVWDVNBpW6rwemfrq1j5sjlRYeULCdAGDBg7nAcyK1D8ZOvzl2U8j4KLgpu4TI8/K2cGepzAAP7+5Vd6jpgOkWktxXuHGgc3aZdHedp1nBIpRB2e4H4udcmvLZidrmfQ+pqDzQ/gMh2KilSWBncsOTgR6Q3KtZLGtlX9HrYaowsNgSLTVSHtr00THRT2qzv++9BnHJEnxd7u+03MHK3YAG8FY45w4ag4tj4S3wL73g1RWrprcDfRrDZEJyhwlNAcu074SNtY4FSQepmKm3t35quoZHs9Uid1W8F254Te0nj2VTLRxk51RttImyZlc6jGxUGmFhBKBk4Ru3zNyNIeqahN06oW63493tSqASjO4p83lCdS3hFK88JkH0f+7bSd7V2ufJ/Bv4iUa/7K5KnUweC83c8NCu9KQJWG5AyVyap7zmNuJakHq4AzSDRslyC5URZashzYjImuueypREi0IxL/TS94c4qkXLiYSWefi1D+m/N5fjBz8w3xTkQknUvAGzvfownhV1wIU3lkzM5iGXAeX/zYzm4Ojb0DzHwkpGE6Y7btXBxdWYlWV9Q43CW5R9yUkUrLq5rbKdB7fRGvpX0hnEIO2gk9QnH/to3VDTET7SmIJ7C+aK7qA6Wz7dx2hF43SWp+o0AWeg/0eNay10rJ2nCals2T9B0xSPUl6IZv27j8lB7/Y+OoYwhkVTMWciFUcue2pG+oo04gz7fJMMPg19/LWvhVbTcqWaCkUWttJsJ1rfzH+CvUoa5rSLMpdNqViEGufPyX4LhO0k5wetlxZG90oTiHC58PGECtSR7gyzbH4wHcKDI0hFtPXZZUPhxyIUCGPSoEs9IPCJjrcy0nOotU7Wd6jA0jPzQFaVfaxbvMRYXfX575Sxcw2gSJSu4/XJcoMoifkwuRNATqXN3u9WxZU+t587mplztJ9ld4vypWz8GN/Mab873igdriDh5441kg2Sok9/WmiTkbB43A3KK3rFLtN6kVAvYqr+eEA079Jxq7ENJ/63aI6OvbxbLOKixwyRQoV5CZg/hRzBSTYyY12CwRHr/Or7JGpxMSIl1pa5ZO/J2Jf7+M7joYRg5A1xARS9PTDDLN0VU6wLOus4oeF/Jd3kGGgClrn2Q1p4H3z8mo7v3369pNAjXQDkb256BI7yxRMb510Z2NNMk3QtyAirFE2KOHnpIELIiXCvFiujXD1opW79OV6pmow0rlPnDxlZPvSot8TQ4kf/IXW1JymWf5cP/ejrFH5RGxkcCOX0FG+AzPJUEQxrXsFxccywkt6QHLXjnb/oPIja4E7JnFdRJF01WhYTDd9jKiKa48/tDbg2MNNORO/l4GI/Lne3JGJYbXG49An8fcrvaMqJTHwK5L+KJHfstm72pP8qAzCIX2T+7gloM6LjryTBvOzXlPrdRPpHxQKDkUQAABlJlABCcIhABj9C54lBmL/5NA2LHZ4AELNGYW1ITK9cxHO5ia7lbvhsP4OBeOx8CXGUp0P9pCi8yKTyB4GiVcxOThAYaU6WIZG5dj4oySWgbulV3vc7Gjepmi3r0tB905SYGgc+UJJfZGVBo6a9FQz8sAMR3Rn2NsBpSawT8pakNOeBfZThcdtnqPZBPrJO+YmslmvlNSjlRfsVt9mcI/zX1d3g3lWMkVMXlIZli8eRGdORnky7/gf5pPdgNnqylKrFT6e5+/exKEZBCoSZCct8pG9eXmxeo0YxA3d769MqWJF7wkwrpgNCBcP3CpwYMkBzdpn4kErKwpl+zKLoCTsufVM+6usg1BurZK3hmAAANBG1s1lP+rgKgcLGa+P0LhiPxWoY7DvNFU4riceVCL7QkCeM8EH6xieLpsNlkp4OV0lMtnTTQBZMxeQnFNiEtyqte4A2+ifsqkC7YRB6XC8/4EAZLt3wP/CMILXhSwZ4MwqIGKdXTsvJ7ifIcrCUQ2PolXGbr7HBMXkFO11+BNc3GdMVGYBQfv2BcCBR8SR8irdmtz4Gia8ycnN7NlAJPeLrFu1n6cU+gikAHSWbEpMaQgxcHtXNx3JJNohvxaHp6a9sDkkt6BXYHaE4uaQ5bq/oYe9JVhHE1rU8HWLEhzyQDf58cG/kSwW24SbQo+LPVT0DsbcXR/Na5F7cti53bYw6yzKwkeXbHdUfXpOQvRVCSQz4H6SFKoI9mIEiDWNR895XuDxOaakXQR0mgyc74QeAmqZ+OqOsTEILcKsae9hqyVHuF1ZDKAhe2ETM2+pBvJU9SlUV6+tzkYaHI5FQfmNwgBDfxWuNWe43ZUshKCAwz26IJLbTgse4a6sVhBra7mSStx/wPbSMSc8tpx/sg8vTRtzpwYSX5sTknm8J7Fg7aBoZhXPu8xuqW8+Cx/19t8UHIwAAAAwH/mmFnRDWfpwEx8tkmAtUv6YMqim3TSMo5ed+qVcL/3GdaPyG7MXEe1ceh0D8wnSNwu03/uRcjo/jH57JP92q1zAZxei0TewySFD7kvxPwpWOtMLx8w3rmKQB3NmORnbSrH0B7faaZmO0mrfC3OVqQJgPWYdRTEazOvfoNFpVA/larWNuA3epLYmUiKf+8Xny5KLw6Z77PNfb75/wFSnNqPh/rl0v/Sqs1L9/Cctlf0FX6a1PPJZ642E28OmDGVPOJfIYmhmmJlXEwndAG7nl2cn7+KqlX+irDp/oLLRQ18TGKXWkvrCCJ0K6o89yRNHvAjfKvLmFSX9eEzxaHHOahLz4Uo/ZiJzGjb5BEZ4YWRAkxm82FuO2XkoctsWXkMeRhkrWKThg3XFk9+jMtjcQpUy4SPxx/M48wuRhXmoNbEbVM4AE8obLw/URI/PlEI+7rDtUZa8dPWVbd8joPlMv4J+PE6QP8GofA4p27qjgRDWcc8zTpNnn6SUogAyMt6kHX1uuPaTVXDAS12j9UDObz2eDUuCFP+513vN+gPUVYk6XBv8O5NZK2q8MbHmVEOGpmVOsj5FknFY6iOYgFZv35W67T1OZVwj5ZuxeOufM06q2S8FS2dBJu3dQvVEDWNtPBkfxFymuuNxs/IEnFDsGnZnbpB4hKRfetLkZj0l9PiVT/n4R8UkHKaWjYezNcYIsN9KTrEvU5tivpFt+ajYt6kcC0am69N6T43L70NUboYF03HcSMWVYejogbejmESYT6bVKwDnCoZoal3WRAbayd6GuBfpAiTTKmVmGCBvqLRGU9E278NTUDtmAbd+xOR5Wgx/hKC5X0u0FxBbk6zukVJvvy6EXlJJ+8TFkYjMRSJlB6FkTL/d6efpQhHhXY9jLiUJYMRN7aCi4xSCSGgVIP7/UY8NfwP+bWAxO0+0JmmDLjHYIA7S2k3EHm8esaqdJv06TsQANIOXdXySwAhYqwfPfxgf6To/fOIedllXQA5g3NWn/q0+icYDTYSatX22SHsQ9jQcBsC1QF6e/Q3N1cta0kpCyxA1kftAvJYfsQ526bYLwSZgkQOWlHJ3jV+QN9Y3DaevTrK4olXn6dfl8VwZ3wZmmp2neTVGHlITD7/mf6JoLM8Fa8aJXmgGZB//5HVF5OiWa8khYVZDo3GAAr3lmKq9oO1CRlsoLHL/jgAzi6DfH58r/n0NjhAAADr2UAE0QiEAHP0WniT0ff+zEr2S67gCZxVmStIth5uJ3xsavW1t94UPKHvMAtsmg6lQ4OncIO/V+VbJN+K0A5o+0YSgUKfsrp2kPMr/VBP/1R4T0WU/jBA6b8inewrz9RzjjtCF4EkHpQ7KFVAIJMhfMc7HqLlAhgFMYzfV3TQjKEPNaFyTvMTFIJwSWhvzuo7dSv1kiAmr/QqYnP3Ja48eZM49yvdtMic3+8AQ0s9eT1H1wIhH2S5A15UVoRi1z9PatF4QKnVza6yNBou5l+3ZbMHY0tTY6PSSq7/2tgI0FSz6zPn6zzQvs2sgaVNUmfVA3dWkT8Jfgm5jJ5IShcVZfzMiwq727we3XmyEy5bwpPy+MktC7E1rYPkwfw2ssT69iC/ayxKQ1y4IiELX2OiEOWgeKeoOo1Z4lEV03oCT5iK3JDVRhO2q6LHSkZTxUuHYn1gqAAG0HPJpL/5ZF2QdkYX//64+CAA+CPvFe86jIPSBzP2uSehQDfvb6yTZdVVxyF/0pHSFrrl/53ONvEGrvM7MQFR5GM4x5/pmEO4e5I0gPL/kZmmF/UwXwbEi37AnTJUm6BgYuM4s0OQXhP3XAeD1VkmJ9pKxCIlJUX6lXETcFwzyemDOiY9Y8Sw7/gPsrPp2efMMkhRq4s5dma5tuYSCFLYGr79trxh+HaEPSOTEj/fu2KNp4tjNY0U03uWTVUb7ANJdekRulqaWKnS55zZPZ/GOdC5Ro7olj+uUmHXIZIWW/L5rsTm0mXAKY5O3j9pLfUD1pfZTsUwfC8FfXaORRCxN1shW/RBqM7dm1mb7jmblrXuPic6c/GMEf3Xx/nsCf0SfKWHmA5ovnpyuKD6CUuR0BZ7XYAPWzGzUmotvKHOiQ+dqheeehXE8XMfcZaT7br9xKrK1X8WAkqj4ewA0xExAXfdwrq9DSyA89hhVaNTxNuExJZ2bBs5iOPhvHBuYu0uB1yklNiPMGrDrglEE9N4FyCoiY7Ywl5dZ0pf7Ffc38ScGruPCSbQdCcl9WEyvp/Xz3mNkXVXfGbVyYz+SulxjodUCLhyUoEDxjyHnwVpMphM4AAd/0GoEU+gxuTettk4g9aThJ4JKpoEm2pZuyirfsneXtXi9dE8SJZ27MhrleNY1RK9nmMa32tXAZ3SFRd9ClJSopTK8fUqsF9C27R+weQDNqJay0AAAMACYUay4NbN09+fqH1wmJrp30AKBqyeG9Y2fGAWN0q4MpqTNgBhnFl/jnfkXF6RLsAAAEbQZokbEOPGvSKMAAAAwACWqf6dP6A1pav/eaMUMOEm998D8o8Lq+myT+wMjyr94FalPhTXFunllB5/0cY85VMc4xwPru9nYwElkEJ7jkbgA7vPzVMsl9SK9I/V7l+RMcHLw1d4sRfmXZHxxzvnifBhmNNgkwbM+81CdgD4SJajjVuEuT/5+gLU3KGtK2NfU7rJVTizmcWX4NMRvDGOMMwvzG+Gb2bIXgv+01v2bHqBagLTfrfwxjyfROb8gWb6NQVNp4uPBheTOTToETqEWrSVX80R2i1Dj2sZK5Uc2t29IAXWwmnXEH3x4lRVJisvtb0Jz+hbDTf+aciUEJUjNXgIHqR9+Ev5HwwwTwXhN/TCQDqWdhHyRVOpJ0+hAAAAP5BAKqaJGxDjxO+J+VFCT7vH3F1bL6h8yA9iVbfcpESd0H1E4NCf3Kjm9UCo3xdxR99ZoUSAqbuJoOwLhgssGnE5rhRPf5rNFp1TM2GKQFRDpkEK13DpnMJU+/0KUmCVvlT42Rp4rYQUY8c7HsaePAjFRtyoLJit01MrRXaxb7lWV8Ughnr4x0uKkgaHTcsmZtxK3Ok1x1GMWIp+m3uR0FDP9w32TyPJGT4P7kFMu3odo9KK39gE6TTiQh+3JsgJtpyWbODxUpqEx312gIopeBrdX5ohOwr8llPeinRhSC/zK8IqfV9wcxlxdYJxSeyKTMr6iP0+KhZCOjy8NJUgAAAALFBAFUmiRsQRP/X2/gA4u7oAaDhOV4h+wRGkywvVvlDIZoQFnihIqs3T6Rk8cMVVlhxBTuexQ2x6/lR+DJkAA1d2bpd90iqN5/PmoMJCTze03U8P8Ym74UdL5MXGvdz8Ehl3+nhwKBUb5GO1bxBNOnThwDcsRbtFf98lrwC0HNn4NK5auCfnWYkbgSouB91gXX0wpgYKSvR3jMXJ/hVom2eb066mKCSHgw7LKKfcxoWWhgAAAEAQQB/pokbEET/IQ6p2A/hoLqpHi3Veq2IwAR9vKlzMkPqNQDsl0be5ukAvE7p7H5z1cnRtkZEdNA6P/UCaio//BH6NDIrx1G5ro40TxgExX6IyaZ8oaPh2ymjQlJD0gZ8qHYH5hmqK4Tm7+P5wQngPzC8BtRkeCcdoypaBiXSUIwaCZY93fapFCdHdNfesy0YKNUbVfqcl837n167rjc4feWvYHhn2XfWvXH+Zi7hfmcRgyIxtY9u+I92e84s82G65nkVQHcZS67J1DUlfVr2HcRLYGDMTsJbO6zfrbBcWV7ZMYj1z7lGxx1bRZ4g+j3YLM1t26YpcVp9WTGk+Gn+gAAAANxBAC0xokbEET8jfUv7Bb0HDgja57UmLfx792VhCRBomFuNUILMxxfgBH+ingKxDAzdnJ5hLyk32G/Tjz3O5RKvg1l6FAiqOOYAy+4yZTvBpwNA0kduwG+5hgA3MrpHhrEv0zOeo87JnG736wD9qc4MFMN003kC290stFn53yRtOC16seXoOrUilG/RN5z5p9OgO/vkhhoQUwt/gwUCwqeurGpKTR19ISRtH0SFNYbLZyPIBhrfQaBKkq8UyIWgW9vHoTJTe9imO2vre/PycAZg2nKkfXRePkXUOVS4AAAAvUEAN9GiRsQRP9fb+AA1Z405m4ADs5KyCeO1AEJNVAICdfus2rHfssTXeH0wzVGks9aHGK2+Lp6gEgzFNMopJeNIsFLJLXmrbaokqB0IEt1qVPADVjic3bAqrHvGA5pj7eIufVdkRI50g+B7bUjAM4LcbUT3SrkRhhF+eTQNz2v6EKWiqOaO7Iiu301PYxtqTEfQEsTaY90W0V2v2M4PF8Gx2g1NUinVbwQ/Nxjn+xjJ1sqjStB1YvddjQZfQAAAAHhBABCcaJGxDj8T7k18R7l/gCZ19H3CsCkvAaD3RkiJZfgAGCJmoj8lFDDY4xDwM+aGCRE0FC1Nys6YXNCpEgnM5ESioQpHpC25Kos60rqyyCSc71OH5PMhyEgAHESwtTcQMvzLIrI1lDXQYqV6nv5Fc0Ec/yDqiZgAAABkQQATRGiRsQ4/Gwg0AAADAKFAPHUeaa8fng9Vm+5huUuzg9ZOWK2GCkHNgekfujM9hJlI7kIPO14IXIqkOy2qZanWSovx3i065lVvWsJcYtBOxegUzDPYkhT0/NlFW11MP4+t+AAAAGpBnkJ4gof/KQqXhifIGkwEyT8dP2ibZ6kUEhTTkJSqwl7+2Pof3IaPvTQa3EZthxmi+8kZtMx5cAyUG+7QF8MN1918bceSeHn/xIwEtu2Y8KkzlLFZLqSumPWwjO8g6WbtdWKiklhyXpxhAAAAgkEAqp5CeIKH/1Q5xIAsOjBfgKt0pDyB0zU1NDr68K6ZCpSSWWLf9zXASSf0HQ+5nDXStENZtZYmWYNN2geClBFY1tcm/7YcSMkGdEjZYMAlC8St9THzRTdaY7P+yzRKLq+7Ick+BvuABgZYETEv2ZAbm2It1OYtV+W3iMEQoyMvB0EAAABSQQBVJ5CeIKH/bOHz/WMiunLSmwFk3hNw3DqV4HuDyae2VDhLxTiH+Mof1LPllIxTzLFsNiECuzqxB+TX/upXjfNvp4Lmg4ObEvIczMbwWmJYgQAAAJpBAH+nkJ4gof9a96YVklQ4A5lEXp1GxjDhQreG+YB0qP+anhaldBGRhQnvr2jjb1h2rZJbpnLEUZvsv6UG2VUOtp3IeInq4kq3ZVxGrMpV0xJPMYSMrkDy1OQxw+qjpocSihZAsqvAohm55bIzao+zBAooXdSS70IbHQOYgH+hLZGkr2mu2ApGzvrPaF6X/5TlJL6jQZkTs10pAAAAeEEALTHkJ4gof15IcrB9mDkBBGOOr0PAm+aPexdjJg/vApguCZTqjDXOkH2k/C2UnD1pUGxNR7LcaAZrkqIbDhujPU2QUxxQgpPz3P329qm5efZuvB6Ay5NCRN3ctmfXGvLzJgVX0kdGit6W0YYBMljdQpxVVrT/xQAAAGZBADfR5CeIKH/tuzNPs+pEjfAuyUFX+rn1f3YzTy3N5L3VASSsH8VY9k5emeMUIX7QFaDo/Vp9sqF4HbbwmWtDUjKK8BmgCL9FyCz81hYWEaFHLkWIHzpFTVRJjut5XJ5YJAgBMQMAAAA2QQAQnHkJ4gofbc4UTO1xmOI5GczflZDQ+rz6k2xm8YsapGSXKforrOay3yqwrUOTWFomHRLfAAAAJ0EAE0R5CeIKH255DAGq4NhqkMxSiB7oZj66qdBvHE6QcxPmqkYHQwAAAEkBnmF0QWP/KtCWozrSmTXLQlrCHpO0FhKLIPHTkQiG7s8KJdXzMWIMKxA0jkNawHnGPE7W2AKb/Yi1ZISuocSpQKaamLbTq/1UAAAATwEAqp5hdEFj/1i0EKvCnMQNdN+okrbFEAUOlScXhTPLT2wAW2SOzvueRj4Nn9dn3xjvSedPQXEigVocG/Pkv7GXCo4b3aZDoqlMJD+XU2AAAABRAQBVJ5hdEFj/XyEgq/L6B1Lo4b+MUxUHZdk+gxr8zpLDt/u8QSQgGQomfQZnBVRxGeOdOB9Opd7kAiH/ZezNm6Eh1iLmXi4fH6PiSBNpaXXwAAAAbQEAf6eYXRBY/1z0ah6q2EWBaFi704LrBNVQ+SkNoqQ3Bmh3Mfc4BzZE8vioJz8puMfvvqrXS5ix1ZeyXxiOnJm6iLISrlfZHKV7hgRdh/LhgznqNgE7sd9y5DCGSqI1Avkkaqy2zHLZimNUoaIAAABgAQAtMeYXRBY/YYuwfYOSSTXDaaGfjDHb/lOtnu/Vi1V9PXMxM/lVdnDYLHzHJsSHHI9cCGzhJXZ1iFrmLcUpySmjbS0KDj61uw+zDLnbP+QtjBZ++M1NKyA70hF5Q/2wAAAASQEAN9HmF0QWP1ivJHmRZH+7Pu5tbuyXKGhIFhnLZonPqbN1Iw8M71z5rDvW3+Ib5gnIVA1x193y041SEyXRUf5DRBt9mIRXefwAAAAvAQAQnHmF0QWPU7XK28agzt+H8nksaIHSqTJozbjRSfFsHENOC47oSyzCO2/oXoAAAAAdAQATRHmF0QWPcdlItrDnuvMnKgImxV8ofY8AF3AAAABIAZ5jakFj/ytXvmme0NoeJEYPG/yGyVnitPrg/m48Co9xmTa1hONEEnu75EYo1h3DE0SCUR2lVjrwE8yAw5tALu/DUcHH912RAAAAVQEAqp5jakFj/1geMq4zUmE30Ja7I0DlA0ntQlJYPtuRBX8jPbLiAjm143/K6Q0vYi3jR8GRhuBFG8aoh5fKoka+pMyv2/g/sU7PaZAP62wg4iogW1kAAABPAQBVJ5jakFj/XMy7cYGZqk5FzvCwTf0KUF/nrdnAsu5t1fCTkjH1n9IVIj8UtQLBxOo2EfJjnA52Q+vzvzx6Vsnh20C0wN5bD0ReWQrRYQAAAGQBAH+nmNqQWP9fE1P7Pueg4le5RZ/G3rU81dBLS/KPeUYd330f3zvTu1uY3L2nB8JvonFRgfDw2fOagtHCmb/nNzyqTE9eyYidcYTOc8+8Y6FsjhM14TqdDbZ7uT+WIlk6dbuLAAAARwEALTHmNqQWP2FVoY7awfTvAE5OrWfcZUmqAn5xOM5q2b1bB9LWkaVYzdSErrpmOVGrc1KlRzlpdwsSojPdVNIrizRisGWlAAAAKQEAN9HmNqQWP1hTecaJn0Nc/7NFIZs6C6a5heoz+oblB225/8cwPeThAAAAKwEAEJx5jakFj1gvEhSqeCwU9x/KHbOQ2IO8FldaoYk8kZOrAv9lxPe1uUEAAAAgAQATRHmNqQWPcx08ZeVoCyrdo45z/hPAr56N9n+gOGEAAAHkQZpoSahBaJlMCFH/DGdlaAAAAwAACPGsXdoCFbpKt6CWWX3X1ZrfM9F8lprIsuTCcI1wFHsJZVsTfDYRieB4ciZZ0RcuPDslSua6navi34PHbP/uOu8QnHFef0CWyqSElSUKZtk8yDQZa7ukGrUqitnaOtLFsbPhmLHXaTnXhLKKd8phy/sU16BC2myGSFeA1PRCfhKQM4uupmTYQeMOit4cD2dHecZOnJ9+gTXHwuXpXXAwfIjf6XRj+FEARVaf6Ioob5JQoGxDl2klnLwtl0iaMuUFoSAF2H1DZS89vFTd5N8ZaYmREqjgj8To1TF04xVjktqQeDLzfWaPEyy9mxkvRZeccWIhlh3gH1qW3ZRG9Otg2eiyX4Cdmxrg4sPz+ILLXw0rS+OTLiS808tcESlFajpdcva9Nq0xUSZ2T++0LMVhLb6HHU3fR7SkwjB6pDFENZ+Q+p2VeD4Wq6bf4VCG+cKYn1DvY/voKCyfyAfuSqLcM6pyAM8jrhO0gNP64ZnCuCdIt87KLdM0/I84aXgJlYegQpmcN8fYyDbHfm7qlVwVMAZ4bJ7ddn5kKgbV6QAMKjIId9rFkSey1LlfgFhafjG4Y6bV0Kaw/35LnqFd7gL+4KEfZTkltBJXqgpQYe/jCQAAAb5BAKqaaEmoQWiZTAhZ/wv+6JuaYL3HCGdvraVz4NWbILUdbjUIRU3bsCQD1QzL9WNiBQ1Bz1/urBZgAAADAOKuoxNAF5MEEzTqnhMPgnuzz3XvuGoqz+y/FAg/Vnc9fQ197SiEsf0hht+eUa+qPT68PArRM7GyB8H0zyhZKyrVh4bxOkIQIXOi6sPue98kk0gCfqT0n+x7ap3hQYQWMccOTz5TYVa5ra0bQqTVVnCsAp3Im8uOEKvbFGbFTUD0s55WUkF4bPq+gJnnujr+F6pwxsRhIYnyIQEn1zuP9wLuq+YiZiddqbSDmnEaqBRcGS5PxTJWDwhkWWJBYT8TK7tdu6mHjtWCWlVT1q4xEn+N9/VVFhhyzSi4senWGQIXlGg+uGgFbxLsbZu3LuWv4Q37rakYNJEfiLeONW+M8YDhtbumcxgMagzJSBrN4uZqtchoyGWJ8+PMg/7Yiz2hL1iiE2T2H5UoNb64l5VlM2NLv9dm8GYgqgH9QyMHGENcqOdflGCY6vPPMJB9NQLk89Dkm1gGdZYHo+bgPke9OWB1Yws6tokRkh40HrQLi/uW57w9p/mawfuo4xtxHApegQAAAVFBAFUmmhJqEFomUwIafxJel0MsGX9JiDVHQEi+g5ZGOBnQ4SmUdFhWop1AAEILm5plmY8spZLh03+d75Ou5Ps1kdajWhvlgAbiUBrVqucsD4cOen3ylYxQAW5U7U/F/zbLBoY9zEcRFXi9KOh7qoaWZaRgpfwgyWJmiU2KkC9XIG5jGtq69ax1tqj+PsxyH+q2TE+pS5xum8xnLKUy39Kd1G86/U/iM+qhKmzmpjgmm0CkFvIZwWlM9ixAbtE4fbbnr6CLF9OpODjDzMRS7qC4S7JC7xnO1QftXsoQN52jiFWfeutWuDz66sLO+QBHlGHoq2a3aDwinWKASjN/f90/2NptP3SjPXtZvRAPVsTONMwNYaJPEkjx3FVmTKqLg2B9XQqnLHYlLnpqrO6umHLcZk7cRz9shhWuEYBR6tniwk4gBOVmKh6OBoaNzvEsx9NdAAACl0EAf6aaEmoQWiZTAhp/El6XN89cVUp29KV6rG8IB9/oMedHHmnP60bcLLo14YfDLx/wBRXIFI1m5iIcGOKBLz4dgLsAaiy4vjqlgK1rR3lSN4/xSkdcq30qNqFmmfgfzYCgjyUq8c5KsKlfLaPMqoC47vRC/yoak5BhuGAA3T7LbPyKnD7qhFF8nFi6i6Sy0aTMKid2SPSwLEe6TYz4P6HH2LPKDAhvxluBvESW5vBbIxv/LsijJohhHk4jMKzH4MwF9iJqyqbAwZea2l8ArdphuvviQx7qQVnfLKKwdpuUsqqwBp28YGC9yZp4OwfczW+S7Vg7y4cDHECgO7Kei8NBT5sG8J/nmBIzG+71LV7RA7sh06cEdAqm5pXY74vsyni0+WDJPjpZyM+WznnOSHw7/bxs3bpfSioWE65rto/kGGgzaNuRggx+HbreUOkjsWE7l45QN/fKUeu3ydeg4xAsLEoztisNcuEFltPNtZIS/bacH3qJJ5K8JvQ+tyP8s1Raz8SCBsvd4Qr2xarPcRxvrKHbkc2J4JNgh+nHTEVn+xg0ArkiPfuMiyIZWK1l6PI23iNxMVYYrYp7EwV2a4fDa5shPTbSrl7abTIF24vDoH+OPIu6X2uPfdLY8UxIySoSyzpY41Ciur+RPs0BP+SX/BLHqGYdhqmuXCk8PVbLGv64+KARgxMTNNF/i9kpF9+1mRHwd22kxFSk4Ym+k4cYNGQPbX+/hjx5kI1cUjCOVOiw21BeKOTbL+2WzEp7xL62RnHcCUp+fqEIKOKRYA5XCCYWfe/ZLyTR8pIUe/MK5hBH9MxDd5gla05bvovl0msW4S2pMViJubAAcXz2lLqcLZByeOz28O1vkHRepQxqA7vp4MkfwQAAAVlBAC0xpoSahBaJlMCGn5g4rJKqHzF1BmkawxwbzAbdx5jzMADNrBii4xluryIlTp3nhpcA4ixJIeBB1OwiOE7oJptnWR3JZkSSptv0sP6SoEYq6K46Hb8g8K0T/055zut+kuNwPJo+JXoPDEnleYZglLECSuddlkOe2/krloTsRB3IR0qJdaNhMCAfINl0NnNLn0+u/TjmLCACbDdk1GUstSg2aeWylvdmGNHGeDoABUjaATOmj23SdRuPFvnnrwmVKiCCP7C9e5er+0UIv3CSRnxXDsBbuevJkTB6OdMVkoXSYLuksFENFFhf0hP0NUGAC3i2Rwt+acsLdTlfvHHMPK/jUPkAhkuaT+2AF8v8iUd5vARyiyKhhV4atJaXWWcOUsOsAg56Jx1HU2Pm+XAvNXpeFWdMjsz9oXdBVrpmtV6skUYtaeXBxKD/HBrRbqr6ju5wkHqM5d0AAAD7QQA30aaEmoQWiZTAhp8ZfLvFcEAAA+GrStvfQ4Zqb7YwACd0T/swoAABdck6G1YCHkK5OBZhkG3WNknYf8Q4j3eJ5LcVvxnqkHNytC6a1Nt4AKxL7RRKnrnvjw0OHiDYMINw9sl5wywiHmX+X958O1EJ61UrBFrmI6nOA5opYUpfMbq5vO2HCJcMJN9OG+AdQOnTAxSu5MAMmiqNHQH3vx8y0DrdQswCkHevcLNEI8mzPgGMrcECkWCQ8j1EJ+H484vzXT1z8kWv9IcirVbzv4N59PGietosve6uN23UjibIe6hHBaiD4v7xpt0qfFNOBuE8dfwiUWqrhcEAAADRQQAQnGmhJqEFomUwIUf/DuuYE3AAAH34daaR4n+pRlQASZklAjFi97LEgu9fqxDx92BhRb+pW+DZTlHL9HAwMssJpm5d/fK6DCYtVJaNZzq++FqnBE4coNlp5uh+ZNr2Y9UtAkzHeTmHFi9KSvuJPkkvTpmVPcPmcJKSZ616/6IDGl8ctJfQxbX7PQBNTNliVSy3t5+rwFtt2LhNz9COAegVKjX+WUhMxGI0Ztgd+PyK3FZfJLZ3SlQ2GUGHPXr8YSCUQQqEeavKIWUy6rd/3eEAAACUQQATRGmhJqEFomUwIUf/LHHcNuRpQL4y3AAAAwAABI1Vbh36JXy1PKvj85a76Tgz2F9U8+aeDD0M7QyPiuMsBGOkVo5FJ9GTEs/oY66tNGg1PcJDXXK+ZVB5YvtBXMHPVkBr3/f8zxYIszs9VSA5t1zlPxbRDsm6NU1cWrgAwrQ80Z+Ss5JSe0Ct2trQnNInYhoaYQAAAJlBnoZFESwSPyeBnr4uKZ4DItLRBiWzwayxVgCqQOjHVAQra64GTZbRjF4QA0vnSXvyvIOArrsDSF86anNpdqToLcPea09PiHktfjc1SzegPAIcdtofwqU4caANNt4hGDPl1FloL5O/pPSHQzcW4hDXph7EU6GmRyiArOnRo6u0l226dQKTV2O1DN3Q9l3rrFzk97FzeLU9fiEAAACRQQCqnoZFESwSP1I2eYEQLInIMv0HGE05ookt9S8nqNqCbNKM9NEIjyhgZvDv7FKfvHD6GXDc9dEn5p3Jjtagi/HREbe3B5Qcmujl0XbgIdx8+mvOwHO/U3EI/m7c2JxXVXKeXwj9Zak+CYSosSULA9FTWNAK7roEBUm7+6A6JLITQEeDZDJPmz2yasHJAa325QAAAJdBAFUnoZFESwSPVfRHqDvUhIjlD6GRcC+zY/SrPBorKPPkkQ5gx+mJZtwLHsiXBRUTl+Un7/8kVWJSpn+wUW9xdIrAHrgR0sAvrimzgsGhmw+CG+fKvTLIk5HdjvNnvlEMJk5bxx/qSGZ1uohcdoKFpBXEkJ4YuEelO3eQuGXuLLKIQfZmdnlJ50YhI7U7jkZU+URxYWXtAAABDUEAf6ehkURLBI9ZCm6aT6JePzogGiSWsNC8P+bAmqyWPflx1BvBNCduwRER74xdGTYSRbeloMcifLfA8/ZbBqN6D45awFnxZxY9hWLEZGOJ6GMm1fk7SMJrM6vAqxhxjA1gSfwe9szLUfsW/s/dRGqLZ+VCnz/0Izzonh3q5WnNUoV3CXvMJrri7SL6v3et/16i0qmXdr1o1riYGOX9N5ZuCGspJSyshsudQEtp9opFAiy9/KfCXCuy7m6EFmf29mpewKqHUExECIQuaqBgjnWufIKDx56Vb8xtmorcvtK4qktRskSxhaO/+zs6Z00LxqIdwJknvvaRCz2h+Iy6yHXpG3ZShBreTOwkG1CpAAAAmEEALTHoZFESwSP/WOpVpx9ny8Ql3tXIObyDRhHQ95gCDHTszbfZSbDVfG1RcThW6mdUjOJmBaZ16Qmw6cNy1+Y3FxrfUMlTJry+ggayieKprTv7qKSZVeUf0qVVOJXZFQ4FKLtJdQL3jtDbnbA9emhCj1abWFCmye6lIvFBxRobsnfO9N3o1UoW27Vs9b+pkSfdeBrOmk6JAAAAdUEAN9HoZFESwSP/qVNLNXwBTiyiuVPDfUoso/SbxSR3byEoLgdUj/6k3b6eBFDSVGY5aDfnsEaZTLosqUlPMW2DG0eAxOo3HQXKFQXG5+i5c4nix0OQRph3Q9NxPussJHO/YJ3df0nqMTwsHGb3CTKrdoMjuwAAAFJBABCcehkURLBI/1FeBLhWBOQXuwS4fKp6RQj+qkuXLNTFp4ZT5EnvQAw0CwGVguAMp0CrsnHlf6mpiq6h8JI/FkC/vQ2DgztURVDDdA1dEJ2BAAAAS0EAE0R6GRREsEj/alEe6Q0OE5nzEORYH6ArWf6Q9Z/lS9lD6QH+Jb9qyUe055HPOpo+vnZYN10ncXhTTSK4Kb4VONtE4sG26Gow2QAAADUBnqV0QVP/Ka/76OO8CB5Uc5zNhdkvetIn527tvyEaxafG6CQK99s1gYSzk/u3p2a5V40wIQAAAFMBAKqepXRBU/9VsRbA9Md4ht3h52QZBo5rlaeTrruPj9DfumZ3EuOA5PX7g8e4HiORadBCNUwZ8/Xs6tCYXaL7rAXhvl+7NXObLlXheT6xdxNBKQAAAFYBAFUnqV0QVP9auB9siFum1y43WUuxTdP1cdo3DQRNxnPS9201Qge9swoirI8SzNVSCABlXibluY9k1PqvsaMz+gGg+VcYHEXpwADHjBjbZ8pgqbYQHwAAAHABAH+nqV0QVP9c+B97+3VDLpxgrKch8oqPdtBAvHFuwloCPM21hAJ+eovB1JOalfsHIxbGxsyHkXM4LECB6zkfrMHFg8O5HhE7WJD8DdfsYVUGjBPFBWv8lDGJr5e0vkBBolZk0LxThzBG29gCrTajAAAASwEALTHqV0QVP19UEpBWI6ujOj0fSdjAXliYAbMLBQDqXYEeZG+xmtc7C2FsP9LCK8R4iIXRz+ZJneLKxLIGWRgXcCJarLJSzlAqQQAAAEIBADfR6ldEFT9WKEWRGpS4xwAb9NsmGMoyVOdVCBbtdJjaTbDcI1KJ6YqTHm1FwoTyx1EzWVpHSJ5E1KETw69r3dEAAAAtAQAQnHqV0QVPVilwjKZMBg4wjC4A2nI3sYW3i1U+M6Tl3g1JSpJ2fBm5xxixAAAAGwEAE0R6ldEFT2+IroXLbaIh7HtyNqEtQ4GLYQAAAFcBnqdqQTP/KDL0boqKsoSm1BteCzpLpcyzMAlTbogqGf5cIgJ19JIVyPulNUyO9K+PP9WElja3e7eomkOjbaUCaTv1GVsxU2GCP943nNRfXwmANJUSVmgAAABSAQCqnqdqQTP/UqHmvmIZUc7sIpu2U+rz09w1CAEKViFcaW4LItWh7E53K19KkIu5s7XVG3SKuQd9rvJx+0FITOwc/dbP+iuFzMORypCPN9tPAgAAAGEBAFUnqdqQTP9XF0yCBhn0LfWB0H2RRj6+FbYcb/OFNX6d/rzFHqJzAD2Vg2ROjoMLDJ2reOCM9prN89Y7o5DNIFKzCQFckh6K7bdu+uHC3UAh3+pgCkL7zpGLvjK/im5AAAAAjAEAf6ep2pBM/1lVanDFGZF0SC9gBjLiXleeBVBywIykrs/D/cfDiAHWZA2OpLPNAO+cr9343VQkkk+6e7niAAFmE9AvfzyvfFfdiEJ2wiqi972jhxsAdIHl0BKTmRzewpwGMiuzt10g1UpG/vPIF14dNuDnx46/555fhqNt+M8qbt1wq8+BT/huecSKAAAAcAEALTHqdqQTP1Vcel8ZxxV8Pg8uJvaMpuKxSUWtwNl1BO+fm0NzUIscpo9JsUCKdJJi2yMzhI4RDBaispizyJHfSiRchugH8I6Lz33qWrWnIBt3TRLgqFYbgniknFnzEJg259V6pAu3cLknIIYAOiwAAABRAQA30ep2pBM/UqRZLWBKpP9re2eO+TGpPrkU/mhYEOJDGiV/k1Joh2pODpJ0YRVRD4SnLhVF8EUrf56E9R8lwQQ7cIKUm3FbOumzpb0NO03AAAAAPgEAEJx6nakEz1Kjx2wTwRizxBhHYPSvrjI0ZgghkpjV200430M+G+ZdMDYc1+q+Waw01NIKnE9mbMC/ps3gAAAAOQEAE0R6nakEzy3ey9H3rxmeQz1hWo3iFV8S2JPqU49F3RlsaJwRQk+O4AR7JED3w+lfav1Ntey1IAAAAw5BmqxJqEFsmUwI5/8BptcDJa/hdi0DTaCfuC08SGmrvmCXbrb8f+8BFDwo9Zv9ExkAMrSDczz/ar8l6QvqDqFwiThYGuQ8UdF3WD3l4ohFfe0CYwWkj5UXs5OiVCUE5HAuP9iWAcXsCvtv214lP1Q+ovAAAuENbIm/8e4PQK0zYW6zvPRN7T1/jVeFY71fYynYen1Vm7Wd878w0CJiXO/3os33HjPDNhZT4Zqc736ilGQ9DSL0r2dHRsXM9xO9rDEVermRWCvUCL3P5Vy2lIZ7xbqMDUCwkf4vN25MByMYP4Ko/kM1eYCl2qoPjCt4j6aSe3gXnCT6GBR/51XUgnkFi+N4rghXbMZK+TezIQJlNFtULFR2D5lr+JG+zxG5+F5W9UHrCAngZtHbMsVktcqJEPoVQVbZO6yG16i1ZEU9mprHqeXPJaQh2rFqRGoVTxV4kF+Js1j+p/NSJrSkzecB4ERS1xCCapxUl25bw4/qcF6VvZz7W6GOWhjLm4mGBD9H3X0ASlKg1cPVd32BX5u4XnqgQ/yVqdugugGR/qCOoQ61MYPg8w70eXOJCBjAqHKpLahzIOrLSB8CIpTUb/u051qGzO/Xn3a8bvmAeSM1yjAfOsvQIOz3LAYW34TDMIxDInMs5oLlIsYwUbIRWrH8RBlkODWYpd1HIZ0clDWAi1/rtguDHB9cifwlTCg4jwJnEK7CSB6mCXMZXycrRbD0phz0SZ3cP6JfMYnGcBgU88VDgQ8svFn/8NL0Wb89IwzabhntQE7zxFWhejTY3Y5KhWagtnS46Y7YM0O6wFHzk1XrD5MuLk+LIWNioiDMOqn8/pBFblw+4wSh7ZZF9gPmukp+jTfUO+JSyp20Y8rQfbufPSvmle/5VPMr+fV4Z7k+vj5ziFiV3dXVR9/Xej3dQsnC7AFzaYsOPMTTXQ3EFLoffjA5sWLQBEo82Oj6Oc6hgYSd2T++CU4DF91A9sucK/yOteWcBh7VFS+3G1JPIZ77dPuRXBCig0XuZ1Krf4xl3EddC/T4ABBQcR5IgAAAAnZBAKqarEmoQWyZTAjn/6Y0QahAABuXizN56NZ+FRQia0uQz/WZgcawwwm9s1S3nL3I1YtbMByaAwyyzuKFDYu445wKhAi0rxW0UweJxoBPLm1ZfnY5NL21Jg/7G/AMcU+B5VzzojKHuwN93u3NUMxkwp/XgMthAEYW4gC7AGzzO7mN2zz0HgjKo4h2gFcMS0ZSh7bBWvUp67mId9Y6u2PNc4YQ2b88qK/g6rFt4ju7B1q5MhU/Wij2mV4bz2jqWqJQMmbE+dsYJNmAY6EEhDQ1UxPluPINbOattsANWKgZA6PLX4ZTeVJldzEjjjyd1vPDE35R3u6rZli8t85e/XVduGKVq+rOagX8lOd101r7vcg5vKPfQAP1u3kDktPRMmlnYMN/i0Bk/cMVKzDK6pA2EdNUXiiFYSS0qUUjD54xZDrla9e1ZpybkDpc+AeyiQyeKfLj8WN/EbB2Vo41nR7gfAlTqSQJZjwYUNN2UBVj93XTwP5tpVCUquLjc2z3czJ1MtEI4j8DvDMgRPxHiCs/3pGDCDN+exZPeXnFI52ndHl27gtMZPiGPSOUzy+TLFofZBpGuVQVhrAyTpb7jML6tKgTqAZ3spz1YER0eb3HXjhLJTEVRDvQ0GIQhLrztXAowO36RotPPmUbbGx+yC8B6XXKTE7E9zjyLgIRGbEkbq5t+3BOc2A6QfDtqckd1n13EvePN7cFZsYWib0Iv/FB9Hb2TMTw2iiSbYKpH6KEwjOYdUaCbNjgKPERBInBiF35fbO0PhzKEf/BwsXgU76DlFEBdCca96CDD+qck5oe30JSbqLhV2Yqya2tcTEadU0eoFZuIXAAAAISQQBVJqsSahBbJlMCEn+XKJEd9o5vs1BuK/er/+eC/TrzvpYE0BFMS+elZq9FZF8V0NsARoo74iwVIHVkxikB/sG2eIFhF7/hBFP3NPvh2lX/TkikwRzK0PJT0z377fC6FnVsqXIEZK2h/2QOn3eGk/fLtITCXimDstYof+y/ehQyamXhmj2qVJe3dsBi3kIGRsAu2MiuV05u6D/MONtpFYINUbn+8pwgvV9wohd6t0VNE5621LzwX27yRiDQ2UMbPIvtnZK6+YBh+DNtcl2q/W50l00RXCzOBpQlSCWAAhp6HVHwfiiZZoYr7r7hxHshmER6EzdS55YWjITwxmON5wGW4r9Y8jF9db+xnQeaQOCApuWv8st5ZPBI+QPWOgQ8RonkMGH8rtzf3izr3h7bKkV0l2smNKMQs70Fw9yCyKYv8riccBD02QwckCqTje3Cxku2CED3KEkP0x9ouYNPvt8jZ6AgliXPo1wO35X5TNJYtDeSRwHiHBeKEX8oO8iqfUUsnPL7Al5YEvqPzDHbldeUqGimZm9aCbjnfL9HjT+RHNoK286/CI6gtk4yODkCA2TTC3us4RxtwmIq9pZC5w/jBLN+O/19ZDnTeAl0PdXk4cagLUuXuqe4ms1H1zFIiQkb6Z1cwgtw8OCYwzaf3dt0ZRPAh3mVGHUT3PcLiEGCqdzJPfUk7RDhzWN3NuadPmoAAAOLQQB/pqsSahBbJlMCEn+X7wkV2FpOhKAAY1wp+jyM6H7Pt5Q8vJCpeHTgWQ1CbiPBhHm0ktbigPevBg7X6Tta1zsZ0W2Bo3ChmjMPi8F8KY52Ha6VvTC6wqVZClE2AWTYwJZjC+9wwnVXEliccxlYmyubLAcL/eTWwelLCYrktxKYBf4iFRr8b0eaid7mZS61Wyq9fkUrwAaNawb95idtMe47Ai8MbX/uAoA6mOQ+AgqhtilofW4Hsr8ha+lAuLjWkqB4fRllDC4okBHodgPk8/UWvkhyGIneHLBGVG3VE/lXAnE19pm4P0RvHTckVJWPVU+lgDhjxdTO91QjIH6XumQOTeTzf3UCu90O25H/y+6X/dwS11Cm27NF5QP2UrEDV5lVvOiQxpeU8RZA54kGkWDJqai+KHXc3oItcYZsLIy2vhoHCfxDfVudqIC4SA99IwCz05gaWlD8PC8uKZCim8A2+Hny5QSv3TdH3UHv+5l79e+kudBH2SejrhCpNzN7N9eqNUASG6UTq5EVe1ynWkWt7r93rOuteYZZzDiC6lTBg5FWU6jzOCTnG3Eg2MppiNjkAShIJ17TuMy4HCD48j6J7U0r+3HeMIUyeZXsJsx4+CtyNk2BNduqt6vyy/pdc0nxXrcSfqmn0lQfTDq+/IZMpWGtIMksxZox1cya3ft82ckTvlGGZFqUSM99ueIlrRG5f92rQF0eXbGC9jjryV2ut9EbnHGn+hsLTpEJUES+/P6znkEKBc3zWk6Xk+LsNfiYKMMFc+6pRRbQ8+UjbodLdTS/v5kDT9lsJBVmq/Bvhd8mGomLp4BZrhR58U9T/59IMCHssWULN9qTkrDcjhU8YnTRbe+2JGyqo9NaZqrHLUSRWDeu3h2DrjvF+slGnN/7dH6nmhI84t0eha+VXhTuVyBsUZmQsgfzvtqduFxpqpyyHTSTNoxI6aYPkXOUhbMp8VVavn6n4GEdH5g22ucerM7hdvb2IJ9yJEFGx6nykIIEFvuCUBJfAxICtQpw6v3UZPCTdZKBg8Pazs4Idd0/3l89CW0j/wXwPsez48AhJ9YeHicq/LsCqJLHplwE2KlDhQQGE6CjN0EiDc6l2srWJRVbVhRcCT0eDJIvxg0REMVPVEdeUz65NTa64o5ZcbdnUBv0zcNmzfGbCfEyP1TOWMXCsk8Q/wWAPpdtvkFNdU/6VlTBj/8rgAAAAaNBAC0xqsSahBbJlMCEnwbmElgMg0IdetAD4d0zuz8ez2bLXd+V+w1J7/z0B9IPf2jOzcMq8cvaIPWRD0NyoGYGszmBiKfq8TNk9oD+jyajVGt0BBFFsjNum3Q/614WB/X9caDNe56g68bxHMXzztHBKX94U3sbUnfV53J+iU5PrfLVlT0jrpSOSP273JDp3Gt1qyEgOOshZgUW6H4d3PrmWuuNiEX9W7x/lIagQQXZpstzDGtmzDa/cYeQ0OdUxY7Nm73/9j0tQ4NBgMe8coCGYs64z9bPEpj1vfbLdzxaOv9X7uPB72v8IFVQzdajz5LdbnSACzG/dt8d0r0CitaR4GwlwKdBYjnSQtbgTHHWM1fMl30RRpICmRmCVQmVdMZmxjm8iw7Gmh82esBCLsyPmxsfg9bLLnzg5JKhG/mBoTRaUDdT3otDP5nKhCrKNDd8YlXTCxpOJ6h9Y/6nlXr+qidiS4BXkex4rCNJ5388SajAQPg1YIbqJa2VMLyhtYjox00zcB7NjRJcrxJgtVz+KpqU48wylnjdjnT2ix80JbGgiAAAAblBADfRqsSahBbJlMCEnzG0oTJQASI7qDJ2y17pGh+jIBLA1eMsUAGqvoS3vfofYdJBLnr/k9jx0v3GQwEwtoLmy1x8J22YNrQqVMe+RBv2EkK+0AAyWExtpHDMJo2enc7hDg7aMMNfazvXyXDvDnkEMw+7cISjsuMqjrW8QdkdoEcrx3CfssOTCZeLcpKZz7gMKtQOhyUnvL8Eq2C86FNasEWG8BIefr1MZyXw9fMEKSRxVa/S676fj/Mbf9/1u49pk0W9vfdakS89rOdpqmbpOkSTjGdsEMs6HQHql3a8OfippNInFK5D0Xajb5tl+coW7haiy3h0u4SID+XTXxefE/rn8T2I0Sv78hbD4lB2G3Py9fRR2BJnrwDtGCLV0dtJPHwgcYGx5NmMV3hpmA5EP3uhKDYIQoH9npIwXZLvc+s9qK8AU8tOlurZNBSMsaHCIEuAtK7wugBW3GpuERkL9lzHawdQhKmpd2IaZ9oyIzwRuwA+4dvbUWW29NX0H4KFJNoTHTzbWGVa0JDF9EoldoMHBI7HfTbGHqbEf3axWt/nMqcxloTgYKjcReSZihfX3Z0JIGZauH0AAAFxQQAQnGqxJqEFsmUwI59Gsg1FqI9duzv9WB0TvAAAc3PlhGmrQoyKTvhgXaqussiO+3VyevppKmF94qlBIjEhNw2KwI0b0V43tRSfzplGEw/rExHdq/sK6jCNS5zsAM6kt/lzuNC4+BH1Bl9/9SOUMAWoNOHJPvm/55Jdp3NMcZueEJHVKjtG2pTx0jeFX6ZYCUsdyxMJEIjq57KnmJZYuulzWhi9PG1J2kRyw8jrnFbGSekCG/H5/aHQjgNVHNhZGeA05UkLUb/jf7zh2bw7+LiiNb1IOhKE8AUJjwe0dqoxZuFnFd332BHvXTiZH90es+Y9+OH2nTTaxcdGXx8n3v3Ft1DmcxpCan1JUqsCgde3vwfOXLR+n4E3HbIbA+0KfSvERFZ7xioEUEiyyIPragWXIh5l/eDA7XQYOjONC2MeQrG92gq7cemr7JL6aG5MqsMtAR5hoyLQ/s63UsCCneiy1qlvI2cGpAf1Uq7m4/PAAAAA8kEAE0RqsSahBbJlMCOfMK+yuIpIAAAGVZu9hTpHGwLN3xJ2PVUq6aoo3U4r+JBEtv9ePLK6Z8tD9p2mV71wLRQZ3Q8LXHswB5ErlPa6epHd+Qm4cKg+9GMJLg9iUGff7GJq5g3xLvOjZGMu+iQyfFH4HqnDR8A4NfQ67jtk6wb44di//VVqvisiGWYQOINe7uJTmdu0Cgu6L3jgYySztpje8YbfGmEK0AJEv5MCl2rVOIKSfVV7bmXYze0R1Yv+KmaVrR9NEfUVMZQj2SQPP6Je91hRyISdZ2EXeVXbRXz4D5xDRCAjMRoMMgLuI8BFiIhQAAABLkGeykUVLDj/I1yxWvw5lPh/DgWXESsT7MZWepkrwYN+lHh8laKTx/7y/byjHzpcv0AJgj8q4jzqtv6cIo6p7/LFUZzq9f77B+gJMuC7rvcF12dAY4DgTmlE0p+C1vdNdoTfFKE63O2OD3HWFs31JDdkUMd21G8eodN9PYf4NT/mwmnplTQyEB1pDx0jc5nHB1E6AHN68PxR4WBBgHTez1nHhgqbOXGZALy5FrZgwct35Ypstmwhrxf4QFCErAPTT8PgGgsXcykcN3I++85sd6quTAZ0NPSIYLaMYwAFdHYYbNk7QjYpogTf0GOjNL64N8nXyyT+ywe1Jyxc7BVEJfj0TJmFcfEidCZl4z0BjuCAc+UKSmRnH03VEAdWqXbXSEcD3ODB6DzaUhw/EiDxAAAA6UEAqp7KRRUsOP/nnaD15aKpg431y10aSvTJSidYKRJ8HFUvK34KAhllrB49nqwStuITsSmLpdOfIK4Uc4jELGjS/KWrpVrTrkn6q+frYAeFFiQmlDwn7byWBQ/BsAsZBp8zIFfsx7qjm2/Axg8CDi+36iZLnRPF8Va8cq3czaGgyz4jNO2DLnWNaI95onVrmjkLghPemsFMhqKEOZ6SMxoXAUoET4M72VxMos3p4B3GjwbORFxjgS2x7db2fQ3uNsS1+sFTTqQ5yDXBK/m2oEO1J/I09ipau52z7JDuzajHf740Ze0rc9RBAAAA+UEAVSeykUVLDj9MeJ84RkEtASrVTvNSS1jck2oy1xrjomkSnxX/z7r772B04S9k71XPqWXBKhiraRV3srulg2Frh3sChLGcAhE5iilLxnyHSWyAdqdp/VjrJ06pOf2lNWJC5jlCS9vos/XM9/J9A3vmOKENAiRzdarlT/tg6GiSv1qrwo/VeHtpJoOiWPDl1GTXtAOTgv9JjK3JCkrWseD3xr7XcJlDgLAwL/qswsWyFRgzL2pxUGeyCnLMdr+rs5ffMiYHwzv4+HZ1bYWZKXlV1erNGRC2n75tVCRwuM/rXu0PFGKxJl18lny3Qvg6WEoSEGSAKPK/QQAAAgRBAH+nspFFSw4/bJYLoAZSYVRQSF1Zn5sGqezu7PVrcOMTbiRejz7cbLUTsIp+argFBzp9O82rbYexQu+Vfyuu86qOac+tYJpn4H4zpyGOh0Y46JOmPQ6ziFDHe/6CPqLrveUfBRNbSv26zoPALDR/bXQt/TsTzh9w7NplbOZVSipi9GmnPpAnYdLC183j9WucYeNdiW9lBM7v7queLi9WarkKrrv63IGZWkp+40nBEujpIiWKVw7VHqOgSttX0LyxCl+Gl+FRtFPr+RMHFDiISpgm1VpouiArqfXvKs81S56JlFLQFSmKVTOWSzkZ49XggJ/4TJDk99xwZrSMVokFLKhUrAyY9CI69IGxtYt7Zs4xaiJtvZdvtSRiMPdnJQv8cN85MC+7ZOGNhnpHY08YHvC9W/22Fje17wzkRjXSpjwKaF7JZAeI+Y06gesLd7DFDl0qWgKmLi8VXSKqBHTSevzvV00hT5GbDk2S9s6nT5fK20ug9kt5akgwXgNt3N0zd4kD/n6qBZSTz7drzJbUADnzLil0B/H/sHnAaoDmav/5B62SUCpJGDF+YEjRKQ1CmFJSat1GurtZU7MgI+V6x1c2zdXMLUZaBRt4XVvwPp/0a0D/WUZrD0DPaeqpurvV4oqMyRmX/zq7d/O5dz+tFNe7W0UYRM7Zjztx7/xhnzTYG3EAAADbQQAtMeykUVLDj+F1hv9ZkAdUTwX7/xLCfYfNVY8ra8QPvT6MGZbLhaqxgKYHx8mkqU7dY+d2inB78VvHqXDf65H/sAJLM7gTQREWHXHznrh/XASJYvSxntGVYtKy0rQhrKM4O/5zhLKj3NShH0fzoPK6gsVVVAUP1KhcB7YxCyxIdierEKvCGQWvAhPZGqUBW8sw9UwUU0NdMLSyOo8PZxOk1BkStk2M62qNKW/iEHyCeinKI0pwtRVrSHPM/YTkRXvm3yIA3xENafvFqLfCf5ZMLt6MudAgqVGBAAAApUEAN9HspFFSw4+mCcaCKWiQ1Pjd4qe4XYumdaoJvWaMv8QYAwlZwVqe+mlvGSG+NtIswSd/1fVybtSTYqPuJZYtaHbmCeBSbGkhvFOzGV3p5MhvHSK9EJ5Q+ix2d1e043B22nKmQoMBnWThoK+SLaXuhzfhHLeDj2rOsiROR8ETYNh2VCRU98qQbFPKKxzPgHkmGW6XWPpWhSrymvws5xQ95e91/QAAAH5BABCceykUVLDj/2oWRA54C3TBG4mjfBwJi7PxMXrDiDkPPbka1Tu3XrJvWTMja6A8I0pBxZFySCUeYFAvxR+y3HW+2H1wDzs9lnAdN255/uJlTS2ox/qKPRe/jhm8FQVfRG25MZRfWVs6vNyvn4IK3p/D8naskylKlEkDq0EAAABhQQATRHspFFSw4/+xjPJ/wDJN3BByLikricmKYPkGstbcR6DoNUr5yLbboqzxnbro16Xqs6+TBX6xyH+vmsST0fpDCYJ+IVBxRB8y6qissT5ZyzFKB7VvnA7NQLxoXUg2gQAAAHMBnul0QRP/JsKRR+ryfBL38hGy17+Z1/ImlQu9JSynAHE/8aHwFxjv9mkRRPmTW9ySuZ4boEh6sMEuw8S4v0vGlJ2Y/wUOKiAP97PaGd0vwxRjVOLT4SubtHgm8tqdclYRrjOJ2njrWd1baK1t3/bTFRpMAAAARQEAqp7pdEET/09BuaqQp+fyacNng3kf5tTm+wAQauQDZjcxdhnm+3jgAyMqUJ2c0c8WAjcDyQ+4ZVONlw61VOJ9bUDyWAAAAIYBAFUnul0QRP9TlpPZIlB0W0MC/FyYmMvogkwagAJjGJv4Fs8kw0SPThWLG103UjQYUCdXyJgU38OMKm8hoSdosBuE0BgmEpmzBcAgz88TWOliGaRjt2rO5DUXQ5byxw5XaFd4rorAfCy4hDtrurMsoJ+uGnL/Q7WoEl67REaKaZ9aeN9IQAAAAKwBAH+nul0QRP9T4q0viKV+DqXpshF72pnDhlXBaqX4vO1GFQlpS7nNKTiC02pIXJmHSS8prqbFK1KAYABjkpi+sqy5g+Wve56wSLc/KMpKGfmHlH6AwFZA2OMxmFk7v5XXvsdj8sT/3sHV/ffVqSdHOOHH/HqoMaM0istWaGyoZx/hRc6bPNNhY11O52bUiRTOO2ABo/1KJHs8qsmGr3WCso1u2a4zdUH/y5yAAAAAawEALTHul0QRP1H1zoxdZyL0TTa00abxgbxcl/FDj2AWTyUSg8uYF8pO/fzgxh3ix0H2HRWbTRhEjSa1qmAO8xIaqxPWn0GY1cu44EV4jhtO6pF9ogVzB7/Q4S+jZIaToVpcQQe+4c2Lp0GAAAAAWAEAN9Hul0QRP6UToWDj8/wWj49VWC4W1PxGCoP1rIcYUeipZYGrLVKkQifdK3DetKXcpVaOnJ00t0AbKcUICbJoTf9oyuA/IiiBpjdwKV62ns1m/AyYr7QAAAA/AQAQnHul0QRPTz2t0VyVYVmxB0NWINnEv3TeQGM97DZjcilmRv5yWtjxNcDQ4a7lig7qJwb4KTGWoa5vO63SAAAAKQEAE0R7pdEET7qQ0Pni5hAA/ol3SOKLhyqzGfDFv3dpYsf0J6fs6W1AAAAA+gGe62pDzyTXAyDJeD1zNai3TvERjJMH180oqfSH9REUKblu/7/8cpQfX1hiKzdveYdrj+MW/f7Ur0aUp3BjXK/iMrAkiN5wFMfnaaWEEkemh9UweyrhMf3mgrYh168qFh/58AOpGirh1MYn8Xf/2+hoBbH6ZA6uSgS7iF1ap5dwOri8vyZqF34f8reEG+juz63aJYYtlglSadqD+cZO12XgZf8QwwQuJ5SZL/F8BNVo1uWoND0gK007hu+17lxyFrus0jA82EKeD2om2nk10q77lkR5rdQKwdXTwXQoLkrAhik3tYyXnzAFEa4quEwvrlJ/8WhXfxqsO5AAAACmAQCqnutqQ89LgPrmoNZLb6vyWXo4FQ4hcPUHR/KCbVoPVPf6QWrGCdY5Eal0ykFKPnRik+mfmoQNdkHchfhvZTyzQWCQHtt9wGHnllznn+i/BSNhxBFBZOoUW4SW+fxIDnrZ+SsBY1vhvXpsE9rj9aZRVX/p01ZY9JSC0dMmckl+f5kGTre6xHKo5qIAhIG5BeI7d3X17VgNTsRwZpG/vCu76h9RgAAAAKgBAFUnutqQ8/9NtJymJxbMpJnM3vj8NlumWZHTtotxGfWAtLo+2LcRvXUFz19Tp2whjeEy04FZ+dL9dlTBmmKQPg61ALk7csyGN2OADZHykDrXNcwRweaqo4LFokwtcu2ppU4u4qzLYQF+09qa0RXG5QTPSkvHBa5GejuMJKZqRz3oWS5v5kRhSck7p0ZwIYIfjPwrgL3fKKLvMjWQ5DEK2HDbs5H2oIAAAAD6AQB/p7rakPP/T6O0am2cIeHoTu4x0It0GNAB4ndYTvplSbyTgRFKbUr1rsw8l0pfpcM/n78xdPDZ9CmdZ05Dw1QHqn2WrUVPVb/WO8BylrPS5VvaCI9GfDtklstC1FNYHwvmekSpg75wtFCkcRsfGood2nufHgm3mhQiPOCAeWPtlhXaQE+gYK1ea6ZAKugukWSGdP6QjBluDu0H4vWqnBQ3l2prYX/CMGlphe43TbtnCHzchI9aY0zqAFqUCJM0W0/l9NrQ23GSxWOINJMDsibcgsAKkYi2xbuYVYg331idq/Oi/5sXZZ04nLQQ9gYA3j41At/uURPT0AAAAH4BAC0x7rakPP9NskessPZG5lCB/enxfZJjDYnNTmkiDKCe+724qpxAH2/+iqfLkBe2dsIDFNpS8crvEIfZg6KjRSP5Ej/MFB683PoWy0ldcAD5vBNafkUs7GJgwlyL13MYo+DkznKVCfsLffR4oI7PnZntctZDiVPYUiBLrmgAAABzAQA30e62pDz/oBZnEYLqxvAoDAoJlk75zZ4k6GUicEAjBkKdnk7uZwSMxO1alRe6opMoracX549aNV0LloJXIPOfQM2CdNmZx1KEHF1vn1kUm8bm1OrZHfb/SpjOhqkVtV9mNH3OOhdIRg2flDo+6VhkcAAAAG4BABCce62pDz9KwkZZ0KPOqnCGSrOd7nj8n0idgRlLdDxupEqePPoxbLsvh/tsK8MTt768kQI5wBw8pZDR9MH0YMjk5jFJr/ITX0DOyDqjDad+Qo2CwBjo9Qlag55qulwgt3zuXrq5Fhyf/kroQAAAAGYBABNEe62pDz8qLe0i+Cz9JkiMTY9nC8XFJ0GDVsZ/uawzCzTGrAwuZAALv+XmLhp6AHLOAf31rr+sAVVxtZxp6zMh9Ryd+wGsST5Ltf48JTWJO9zi6VPu5N/dd0nfL9kZsZdib0AAAAS7QZrwSahBbJlMCMf/AWBjfAIUpzZvB3iIIOH5Jt3eb4rJicMpihZwWKTyHiZaV5npA/nYKMd/FJtLPGnemMxnzOFcuc9DubZoXBy0yfhfLlPG6R+4vvgA/jHnqnLCiZCO7JHsnGhub/GW08nJVblrX9hlfHoGx9uPB6kUiwCjFmJJjcLFEjlPFFT7nhYl1bDnlulBh8gCqvc6DHvQF/oeTGa6XhfFkanfklZ0qoV/Uba6AA8rg9ESQd85qLhr7mDr6KoKsqw1bNzfvDRbk/tvsQn9xOu1+4vQyY4TyczyWMKaAbEPbCU/ObXZOC0sU1I3PlNtHsVnTHqShUu1pMeIwIBMz1wsHocr6jrAl2XxUB+PNTk4iLsIHpVsu1L4GKqvn5Qym9IHVaF0mxw4QIPRb7HLHpZx5zMWaq1vup0jihbTAKEMNd3aVlDYkhxfah5iCuf/Tm3fISxjpHmAHvy/ZHTTR3ZBHN9EFHR3umyty5dDLjw/XbGtz5Svmp5mkuFjFNLrXY7pYPcDkBiG3s2mYM0/DJ6IWRDPa1SDM/6PZ7N49uakpjUbktho68KIyjBbfvJj9jDPSOFIxUjYacRno++vOrvpB2DNeBFtSwpmtUFanEa0ZBliD3eq1CfubJIWwweUYldrTR3D7IOdFdkkJ3Nic3Ka/imHaadU/KE4XZFtOA5JQ8SrbSyb9eldh8d9CN3kBy/XNYEx0crKxWgCzEExJ+1JMiydJDpwwB7wzfiRnmyRqPxPIz49bScss1l31nJaK92QGacPgBROXUIX6jfDL6qo3vsTWXPu/wOsYC+UOKmQHF8ZVmHzu6q1SOSbqqTh2N175L3u7wjzWhRxKPtNwmsaCr8txiFqzXATU3DMNQWwny+te/I1H1nexEsqhOglT0q1QM9E15CR97u620JqYsqnBm0QDZSRLIBh8/MkA12c4wvzajozIhfdPyNrv/tCZRyd8Mmg4RgfRpFrA/ygcM9qgYpvTJB39nsSxX9rWjDPYWXN/KeZLLxsaQwRgiNUWOx2AFyGzetWBC2q9GzFM0jtfUzqVOoCoG6C63QtOXXFY38JCSHb96b0wMdvSvTcFtswbWlCA52Ql+Gh23kQqPfkK2MUbEWupc3mIkdCHV17nFYphoPnpoWny+jhywcpEme41cauRiTXVQ7w2waFmHzrc5SwZfrwL6UhUkYV89OKCvdua/biKTjdPtu1bPDw6QosvYbuNHztZ7KUyTiLUqdwbxluaOccSYTOCpt4trDL7YtQFJYhvUjHqEmbPgcFRElAtn3EIM1U9HrzR4d5O3Rvo1zFZ3IaNed3wcKOWBMiiSAuqPTyhf6sf0Zlgu2ATtB1fCi9hdXNC+vH6he/Jysytcn3tUWIbRHakNibODXPRTCvOtTkfZVbyOrYhsRaYHv8vemY3uBY/qRrjQ12wxWLYybqHuTrRrk+g9E18VHKhTPFkp3XGAzbHVRb4X4nip+K7hjDccqout3LTwjWyrsvze+hWJTqUQng1i/qc+bsj5s/jUTB6ih/GjBcepHxLKLPNw8aCDZt/3TVASWwq706xxsDiH6dWsWmynR82RAWTMLQJwvrDg+PcaSOwPTWayuhBr9drsEAAAN+QQCqmvBJqEFsmUwIx/8NjpTdWyGpXa6UIvQlYNgFB40/J3IMSm4U9bpbHYfXwa2+vaee8FX/e9FaksN33JE2eVizpF+2OAFUFMgAABMMeL9ox2GaI3fMG2JZzR8P36G/G9AmKq/kQsx9iANRKu9ZZ9NYSqqmVKgodLMT7ZBtS14BX1rj66VVF0qiPTV8/d1WeKpxeVBEPGd87/egblb3r02vks5K0yKhK9cHd3QD/lduFotQApj3ZiinoXvL5x87O9cl6Nxqe16h5UMITdGcuRkL1D5oOly4I1lL8U4Wrh28zETopjs61Iu7bg/vBzpORbsmmT8AVQMxWjc6WSQP+Ybv89H4cwJOy3EzI9kfb6XgTvx+4AmPsEufPdnIFGfuU+CixH7DArIDsZ5TKARMEBdPUPcCstOSAhMhoVBDJWYuFrgFGuH2mgWv43/jXQnoRzxGQ8p4i9jKWu6POalgaegXylcbfVerxiiy9FeZ1I8ZKrtYQoe0lCLTm0kEc77VO2Z6+NA4lYcIbEPB7CpaEv2sS2WoGf2ZfcKxXHNlX+IXKTE78v8QSFbI37rsu/h1hHWMEzq0TLPxQcw4OLz20WwvMBeFj5x3a3RtYCeCzIbL66mtmQGjX1oSkqINYubA02rSfFicdbCd90mTtWrEfRL6g3TNrfSv66BjQRQ6i5j1Chh/4dvra+MdT4Jw3r6szA1S0N6D1Lm6pqR0+HY7r+7wM4nwEoY6okKc5i2R+doQpA9jT4p9TOrPasGkpWUKxZ/BcVgs4SnV5tpGPdu9VbZ2lv/B+CYm+aXNKNMdlDWK1rsGJMhc6+27LIiAMShTrKr1G6UzYi2fzlE1Dc0wj7fOUqiAs/xL8czrl/SZtQWie2n9AvqQL8O+QIBg17eir8rEo/bYX7z3ThwU7zRWc9IFBOpRpHDhZocGdNqJhJwPmn2EQHsHDpVsNuY0B1Q2wf4g4KKUaFvabe/UNrgZFgkqMly/aqjm87sEI52e7f+3Ivt+ZpuGwCvSX6SttreFLkOCKVuKTcytxhfUMUN9YpcwwqSFONPB/4Xo+1pfqYV+2n0iTsw0w8pjrGSP6DUk5Mk5p/6TnZ+55sLSw/rX7QXWYOb6VMGxb0u/n6usULd4XfNtTZUVe9CxWxjYrHfgF24OLL3xcQsGfSsoFo7QM2DMo+nzQQ+3Wa7BXEnhAAADdkEAVSa8EmoQWyZTAjn/A5AalRVlucMMgv+CehungWbPP/SldevnI8fjfY2GNrpyaC+C4eAkZZQL2V0ePlk7sFyTIiLkP//w0X4WT1t9Zx+dYVglLm0pmjOgsiSvnmyEDr5FMkRAGM6IjdC0P1vXJrHHOYeBV7d00xxq/DIYIasn8FswyukwoZgrTItPq8JzUT+RoQoWfbFHhwJIT2W9uFyKpGZHFj0yu4dSeMcsvTi0O1+2J645KVhjT2WuPOkIynWcRytZK4MxJ0jeEPokhiS4mb/V/hO6SGHcfWnuUk6t/tj4Pe64m2NVdae3rqC0+EUQ5k9CuGrYldv1lHtDvwCyze7Fz46PX9URSsZ7xjYA1QtNwNbNDZEUa3UA2VyhUDWbs+f4Bt3kZLOvtfYU4/hSB1PIWvxeGS55GurCZJCWhKoqxKfAbGlpEKz0xIuC512Sa2iALB3+p5k0/s87y/ZAGIl3iOjTJVXNenLQ5EU1p7PvBOkMFfuBZtuvJwRVu79VwyD7bpTK8rfVbTP5nTYczRgJ02YfWiTpqgAZBp4M3IBKQ4C9HVHonMYCm/bePy4G1L0lxtyoHS0mR84zSyFI1ayG1SzP8iis+qeSJgU5mdyDvBJHOHr5FgmHejQaPptXEyI1U/+OdbKsjsXQ+xdzOhX+pCjgZWbR4Ab2f1bt31JRy9eCCPmf1d+gKwU0PbOKExTLsbEEaH6CYUJyXEAuRLINLHDzsDpIAwc3Uys8KW/eM0OZvBSnxuW/oXlI3HcFoKd7PufW+bmRNpqqWWG33hQ5efRUfrJiKPqxiUMUkES8g4EaK8s8X2RO9LZIThMo7NxJeNWInm5rDmAcbUt4l5otfzWn8emBgfuev2VHYgY2w03VjOUmYQ5XrnwcydZ3lOsFW9ca4lLXG1mfifG8NqaY7aJrxd/YBcJTU59DugCy9LOU5/NXtXtzgvl6ls+XaKx0uCMjscwzTGkGNNtgyBBL65LQ6KYJ5XUE1VwsjhqECHnEcwq/RXXmQRDWKBT2M6LdQhlVBy55/RF5G13GLwabejNsw4n7JyEQZgCl1/ozOk4039BUkjHfqq8DXJVKF1DOie5oU8jKhQwgRoplW5jZEqrRoR+wKqp168VwbBCpoV3UxgkorNtjMhM3GDFy4rA6JGN6uPMJpMoyG1G2q7MAG4EAAAYbQQB/prwSahBbJlMCOf8cGL1jtEKBWye6oH7ss7HLHXl7/YtIAGIanA8W1NUi5hbAsyt283YsXcbhTi3VJVxy+CgfWAfpnWTqjldAEyxF97FVRKP6x20egmfU2aAAY8yiYkZa+2DRt0v4b5/BFX5HrCIqhFARPSZm5BjMt/PvOrSgxb241ZQCO2hpSNLlXWKQpdoBRH8qPF+S9WyBaN+cvCGcN1lxnejSE93DxUXqSDTOFzr5UXav77xy9nzpdwGGb0+z/ABJxvJe+2YnCXZwHsYlgd9oBxxpnQvHVrUx94TKWLycrWM0uhFtEd7lnpNSdSjCmMErPcJ74s6WKc2VPWLMQuFIkOV5WyGSeE5qsLfEnIGpSw7HdWUdyNPCnYgT5K+Z5PIYbSC9kWbCNVwQohcl+PZk79S0A429+M5FpbNA+ZXfCQ91YRGWIE0ds84Kz1rU72AuEQQ0fhqepwGMbKjo/MrnhICbOWkejckMX+d1OGJTTi7MUJ5hMsnw4Kgv7awn98IM0tgcXYZbJuSRNcDppcOFuXrKJLvlOOmq9KZiABjR7FOkim2t15mr4vbhNWNBySykwz0ugaFQLbX/WX6uVVf9czXLZ9hMo4AI68rLPnl3MTiVYBnnVtsIEMw5d4fNoQdxZgxZujsYa2GQM0r0jCd0Hlrq1KwE5NQ0LwrlHwyw2nNOYJz/L1fn+UQPnUbLVBHn5Yh0bQE5FDFmpLqsEMnXL18SW2Szpj0snNM/Vj+uJxO1BiZFYpPYYXGrSRNDPrHxCItwLCT2dN5kMFvBIvIXw34+pObnITjBBhexHCjLBWBU5VfOl5r0lm/JiBiyYBLobz5ytYubiIUpsUdDOL817IuXeCneNtkFnabQSLrhG5xzBcdEgdjdU0VxfU53V+zUEBM34prfer1xH/3lTUmssQwclMDfDbsIjWHuxLa/81lZn4Pfo13PAwZDNXx1BqASLIGBo25Vv7GCpQ1fxpa0a6t/KS4K5qqJGcrsSFamgV79PpWY6iETjdoiSGeZ13hvQoxYFvnLW1tK4gAbcP6U02J4wYlAo+KKJaIkwyoDn3o/47kXoCSjKBErJ82ZVidxsPZx9r/8HDXtlUmHo7711V0NZHtKH4ZAKWeCX+YFus8f7eT+tVtK4g18hMoRBpQHPmVCaB6FF3hQWt2K6LkK2jjVl/FRfcM9b6FtaVLhDc73+c63Ht7+nqm6IG1maebIYgppI/pwPPL6gKi3jlOaQ8D5Xrx2U2QU0vQPs5oZo4nTzQ10BLPt1LNsvQBOAJd2mNTkKlN1XtmCfUxMCme+ZWa1BDEqsyLdA/ScwLkeTxkTpJwj5FPamBPWV0b1yJHvMEvvf/du0EeeSsIXjfuFAeDrjT9xWMTmyvRLE0nU9X8TtLK0iFmf2UI7dHjKs18PU8viJrHKr/Rukp15HF4bRXtJMvmHLo0/Q7IRKcCGHANCOhahsyMeECkvToNDnIH+OggtZ0dzCiZujSxs2TCKXTGPc/XTOoBlhytgjD53tNGbhcipbqkQ+X7FX3NIph2Mywqh9D8dsabpOWJwjt9e9RZEgTKgB72wfwnJNKp+82FxXrItJeQ5Lz10s38sbxpfCPj01SEag8XSQxpCv7+TfpOQ6YLYDETAFj21Jjs9cf6FJK9LvjSJL4bH97m9o7JtWNMKpSzob7gS//J63B5p5aAHs74+dExFkLxsZMX60+ennOvtWwFmTGpTHuDS1vfXPesJL8v7qDZgopc+FsFeSH0TBvzj8i5pPLER6XN/obQwamjrRvLjK11uSJzVeCPGdMxOXc2Be0JQ1lhfxgKNvFuE2zt3KzgDpT14la7PS18KptKznIfx8K+DCw3tWQTO9TCen1MN3ZwM0IoqTEV8IvsLGQzUnk97gJ9CQi5/Q5Vnyzq4b6r+g/a00vG32tn+KuDjew5w7ebqzLUERPv0iNWytEOpq8CuOAG62iVgiOZQFgeBFOlFwo/pPepZP9FAggU5DKgA/XBHbb+zy5O+pvWRGkcuhr/l+qHFk8U9fMryDQv2Vyu6G09epEaxrly2pkgHHJnMnzA71utldptJJDi5xOahAAACMkEALTGvBJqEFsmUwI5/A6BgkAEdHAQKQsZ6H1/9QdV3xmbKhYsPH0vgRlyD2XWO34VZgAAHYTQZ8pFeFL+lFz0zoHbFkV+2HLCGo8e/MeXK0RFi0pRaQQrXK322UkgGKtDmPGDDiRTIAlbtSwJy9u5lxlECCl4MC5kqLaiKLa4CrZ6bs0uTfO3McfDyn93/wlJ9D+ndTjriuDSWyaWTVlGWY+bRbbrSfoewYHU0tQFQNqlwrsTjTny9kmjhtViV+19tUPkB7QFkLJY0CZzA+EJ/0WEsU2BzJqKxGCcmo0PPk5hPruoapGDYw82PeaLnpPwwl0PDgG+OLf4xVGXRlFNP3LHnmqXhJBiCr09O4X4gHiq0SS1+J3PALVgOK8TbbEyrb9dY6DPukqIxF+GoJ1N0UNAjDFZFji3QrkoSQ6O8HQ0IptQ90kTOTKEU8p1S8ijLOH/AB/QA4CDZa29CkXN+3qaYGD9EpielqEtlWKdgZVx0x2PSLe9OpJ5stoK+Y+444tIWUemFZU5YZ/m6nusWoNf94x+Ukl3mVBQmcWRiZBqaO57re4vGGKnRYMr+ze9pfP9Y55mbZN28jl531fJRQXCEbiY2ElJBW7gm9ZGLf4kQrVldi9kZFgoJLQ9FKKluMiYhSGYSGmHfFbE8Qv7G54w38812AIy8RV0mAfqF1OudxWM2UKmDNctPk7tY8rIEbAxd/wy6jwFQQ8thaoWTUEcaZKYMqRlWA+t8o2Dv+2EAAAKGQQA30a8EmoQWyZTAjn8JRK4OXjoyfGAAAOY0OMW6N964+uMc8WbxsP/B0h/kU9Sh+1XNsUDZdDlTookRAMM/xS87QNNW1PkG1Aazm5jfvrkNnEwS9P5zo9Zg20Xv79cbdF0XbZMu2vkAADlkQ6ToK4X1IYvxPIbxQzdddgLGgQP0mzapZRyCdTd3a4hW37bGv23Nfhp3CUVhyvTAkLyNmCGEOtTA4NN/Om7iakFQULPi1UT4QciVRCBKpTHES10QMsfpLiDuQ8cIfiUC5dTh8douYxWvkzRdyxld3Pg5wz8OpVdyY/LQuM28HFviY4/07CLxcDowZqyGiYhGCpKImMbrale1Khriu1wxPKHSeZZVRJ90l0fMzvj4NQf6LITv9theDKzB9OgvNWKnqiNwjh2sH2zHTfOeMwA4RPWOFsWw8seKARuds3jqxx99oZd0O5OVz0Rls3IgVzP5aLhvFUB46BmWqV/zOMcNxuIXhoZiaWv6wXPXXXfPat6gi6ztpeB8VuQb0+54EjVHKWIcMwbcYG1tWavdx+CWDCTmlkAtuqlnkcLoLnxDVTburSTRaPAUCUMeS4fdUE2+gievqOscXnFcQfYBC1wPftFsL5lfyRB9uJWZfrYt/3CmCYw7wQwHYNF7GOz17J+fdyFt4RI5hD3+KqIytKqxGuzRHmkDDDjo53WZxJhLYZYK3uiMukAfv5TJU23JO8a0vpwh5r82tbsX7dy3zD+Xbq0nJw12o10UP+sKbUyQLbiPbBJ499/KmP5mOHEQjoNneLJsXZWduF7Snft3Fpan40L+BL/VR7N4MxSU9Mi4jz5YR3U62Bl9XvLLCf+/ZXFNAvc60YAkwkGRwQAAAk9BABCca8EmoQWyZTAjHzec257jMAZiMWvOSsgjLza8x14l/Pl6n7ZRYMy16I2ulHVQpWJvZEddZVKWtgNzajRr08sGipnOCgaRY0AahqqNj8MmpNtWOUrGGsfBML3VjgXsMqqeKyar7XapqtTzVPH47zsMefje3qVemPky+nupdx8XzwlTDF2bbUNKIW81gN0oAlNSxHoD+E8qjqKlH+O4rugqR+CbJ87/ONCqWLwuG5gexj3jEabtI2BrpRPZ7D6ozpxeDa/2I7CxFhcU51dGtUALftkMgtl+mJ7dB4+Kc5eeGwVJMx5TF52Fp/H+AHYSwOvL06r5CGi4fXFd0Kl7rJhY0LG8Rrn6cMNedGH/EM2Ih41NtEse2NNs9KBWIw3kf2CeE1HG2PwgdRnOLDjhttEYla2PlszPrczYN3uPaLunYDf3WeR3aQVTv/XkHH0IDNRU/6pq4f6307YtxxKM+YsGu++G+kAlSJSb2/U9PghX9EaHvlxHrAyTQIrs+SuEQVTasoS5CTBhKF8F3rCuIhIemPUzgR6mQjtlyKeSiol5vu6acXDdFZnStj8jGw0S660AbdZqtjZGS1afimeH/sJappf5bUytV04RfsywdzEuzWZiKc+JIuqxTWTCNsMSrCohG6ilznLeyr8FqGhhK/kjL37ijcJBB3dbQI4RqF0xKLq9ufKQhpi67/q0O0Zel9wcVhujkMBofE1047U2MxpTSGLZwxTald3ZpBwOiipyvT2x3HannvTeX+7V+/mK9CnHPuAkEhJu0cxt17kAAAGVQQATRGvBJqEFsmUwIx83ni/P5zZ7IS+Ad2hhu0JgAAADABLjyFDLm2kFpIy8qK+Ql4bKZYSzXh8BuG75CoE8cGCQ2E5izWaXC4uAvB5N07EgnAf2Q2waBWsDlt935khSaA0toQZKDkybFXMCYS0DAeD3OS6BoKygBpNxqtUovBkRt1y53TqGi2jShiJVJfHUjjOaxXqvrAJ/z6yeBnV61OPdzrxV2Ek4k4OR/heQGQve6W+qr+17Ke2I3hzmIc7pQqEXgi1MCjZVniqDZeueTmbAH2s2L9CTKxLdaVuko3wKctXRKkH9U4/E4kxx+vTrAdtAV5xTAGUI5v6IKYHLmvhGGPbS2EZuiZ/323f1I6hjzjbuXvd+Wd92mcJyKl+0QOhEtyaEuLCel5ZlLDPO8oRB1zjq1LU4KRcVOvVkWDTjyL5P9E6k3QsMnsy6aUr0xVpIqxnnjCg/4qVCMQcKXjlPKxuMYQM5Lxid7sr7d0XT23tEckaEVWJrFtWi2pyPbBEpRR876CvJotNvhHJtZRDbgdCRAAACXEGfDkUVLCz/Hh5G2+UY0BoQ9Paec9x6rA8SVljPvg4Aovbzk2635mbW0PLhg6m00FQIj9rd8xcbP27GfAXag9g5xvLSwlutW/FVilUpLis/1wAlXZYS0/w4bLwtmYihFNV8y6TH2bnOFltn/KjSA8R5JEaxPOfTaHBugvO9a+Vd/D9n6PHmbQOs7PB+25W9YeZl2Ne9HH8efVY/xs1U1qj1sk2mCVxBglWjUOfTR1+AJIoS5U8fMHGZcxuKUzkgeKET867euXHayueNvVH500ODdYgp11PniC+aN0rnOnuErd8Ctp6yzBMi0vImCVYTYl6aVV50BjwoYF2pZLHlN4pzkGEcONNnK85USmRPwV4rCOokF2yhkdvyu0CQbevZbl+AZrJPWMfUnrPlIgAkeGilrx7meHaawkeDBd26G4i/GlNS/rRl4YGPfgvQSKmyHsykYsqc21oDihMcD0YzONRsROQaZWkRWDjpkBztc5nYgdPPhcBr0ZYLcNS2960CH+X6oT3pPYURYWblDOKqO78O6nQgRPKhCsDkPOXlJmdQJlIQ/0sbK7wG7h0py2lT8M6by4ZGfcQFv7jfC6ede2oxtwyhrfyXmpT0FGuwJF8sv2SsKnkknwmjuZ+NcAo7LDfCnCx5z7Ta7mIcrUKcV4yvIAPdnF71x6QGL4UYoxxzFOqAvZHiKMo0NF6FzzJ6kz+qBB1gp5RjjHGAiMU/E4vLGPprhdYYXGFtm5mgeHbFbQrS+8feew+IvTlMQ7AQhcChlBAld6AldLps3YWc0f87VPnqhToa5yMdiUEAAAGpQQCqnw5FFSws/z2WugjOlioPMuJe/7E4mT/LUej+BdMw83sQEkP5ADOLPlAK6ZhRKChOnJV85NepJ4PgeudAHRPGTGZCIvYkWvTyY8JDx5MKp9BfIBPASAvKj/vlDhM+Kg90hWEbcxmCozDGP6puHLT1dhMLfC5e5l5IJDkkBh8/o1Can6fqrqTCULgT9AbPfG9nLx1n+KI1SMBUiXJDaVY8u2zwGfL5AkFrvoxOVtNrIZlSfavLFwsV4e7twGaAxxunr2R/prcQEh5vXrcJ5cbOOFtlRkKS51nCYnCWQl+/OHViyTTOShHm7JefJGilxIguJ8bUSRV/4+WI1djCCz/fC/e45JyjWQ4NuBbh1kmdyShNFTmnfeb0NFBwKZQLaC/Wisa8tr9YI/irduSGOABWtNTJR6Djx72gVMShHHKMXIYlz7FdEjCrrDzs2Skdzq5ySA7m3FCdZiq+mgbaUaeLN1s3QRl4isVF1kYHJ5KdZpoFGHNxNP3k8qDcZi5f62uWFxn2D3U7sr0RnvGNcDetAoay1efjcpgue2n87DTDFFee0t6aKPMAAAHXQQBVJ8ORRUsKPz27zWfC+P1gZaZXz8WAnNs4XBfFVIYmP7Auv+BDYmOFwAHPu3OWMOGRXN9CbxUY+9xBJPyTg4PJppGxBN99gJkGS5FclKECXvuIeAK2um2iWu3+fXW6/J1YSnPXg5qZvnglBuxpcSeoraaKlRZYSj+m6Dd1bS0WtGt/FgymoL7Tj3j+kl+GSygY4r8wdNd1FBHS7oanBUhLb3eDA/eDWYlI1lrWyMCO6FT/dJpJyDoiZG4OcrzFacZX7f9Hcz3A9c3bizvfLQILfKyGVuZ+KKxkC+iWMUVKdgbFLqmEt0xt+c07abml/csA6IZb/HqGBUW0jqz/WpZacDEjSzrEi59FW3IWrxOJOE9fiDL/Kwog3EqULCTYwZJe0ooDsiTjtoTj1nVXEtmuhFP1Nc8gwz9R8UQt0v93vgbqGQR0Eu9XP6XR218l13WWsn1E+w7LTohDeEjZiPgX85Zwug25x6YZOmEVWQJQg/W0GtvpQJ5Mi2Y9J6e5eD/h836uuUkIFPMNBOsCxQchqzBQ6qnQ5IBRWxqTScCju6g4DNWrXdMoPgSZz+gescwHp9Z8TDue1JK0CVY877WL5C1sUv730m/inxVEggQolGNTCnNJAAAEGEEAf6fDkUVLCj9hDxQBY4wukPTn+BUh7bXuxfZKENTwuaNOrL0yZo+sWA615R5s5whPk/RkJPiRTtG5VE1Vs1r++oylVSIP3KWdZe9kdvhv1hnEakahvSfnCxYw0jvKZkJpQRszEKeEA01iexDGGhwcJRHoaf3mbuLl13bRVvhaCQfxI2nZVmDtxDMwEa+TE0olpZP835KE9xpPHNUq36G3xnBYGnYsPtYMmJxENeqA2UhFXW5BK8WM87/4iKdMhc2FTRszfh6Vg4DZySCT9JLjKwWspLhVHzzfk+gMi0UDNiieh5dWMU9xyb/ptodILonGxonOB276GKSx1AIN999KpAPkbgza+OnLaSgwjvO6m1zuuST2w5b50OzWDJojGKT8RqzVXIKRXg6gcYIAU//MI9JtIuI7p7+P+3SbPJevAJ2yupq4n8Vc1IWAn1DZApw6/Botjuqq4kgYw4W1vbHurWuwVsg3Qgo6SQjWWL2iDnZ4UPjbsDgw095WzQm0DaXRMvctm41VCBzcMAZhbTmdpp9I832j3biMBH7m6mxZIakKYkd3kHhuVKKMyisDe/WQ/SL0kmNt4uQVfwDSQRhJ/Cz2M0zqPR4AygJbnN32rJhAWwvs98BEtzkI7c6b05XaDhgJfbbZNI9mMScB2XRQJ+GjUmZevUI1g9e1nUidnB4Y/F2WZtvQa8sJGaE0nwJn+8rHuZaFRo2LIoelJwYRu0GzCAu1hG0Ps7BOhMpyVvCc8ZRfAwtl/V/YICJGmAjA8gMrAFUHVbFKn4h8spYSS58pAEX9WtNS96e31hmm2ylfjvdMP1KL6iNs6Z/VXHhR6J1/8P/VXuxaAj0KI/AVi2tHupnVtG1QGpVg6hQhx6O4xi5jVc739Q2gqDgQQL+czS+V7HKJavCMNWjDBPxWWlByJXMr7BgkCaISfRVL2dGL4FgGxThOuC+9iW/Wj8TbCMzEUsfioUKwE6dII1oBmT+21dR6Qgt+0NQmDY8plWV/TQf6rn5ZLW+5MQbe3N4QZDQEhYgag14mKJ9/OzTtWhTbAoIXL0AvAqSVXqCh0BnQTcl7G/96kicCFUJwYnbCYVctW6fkrLg6s5FHth8OEYmHpQ78OCLoN9VtwBB+SMxvSAzJg8GGWpK+H+9bv2tcoXww1n+d4meanbq2Mn6FYnGHzhp5rTCCbd8z14kRkh9fxUTZFW6SIakEeokj18XDyZ+7mTmJr6O66TD4DeqHVm5TwvSBQRl3RBtGlCpQJfV44BsNYaeemGEJ/5qpZTtEkXt0cTIM37QG5VZQr/5T8RMEpiOgxRD10FwUPqxCsfe0FPoEX9KRcB5UhGGzWQI5OAxXapRgwTj3rxSlBZEwI5mf62qEbtxoXycfE2lFYLiVycQlvAUAAAEmQQAtMfDkUVLCj9iWy5Hba9lxmDbSDxCMSaOXf19cq+QiFGpgIZOe6+Uf5v5nxe55WUZDR6u+E67jI9/4TgdPOfntcV76hTr+SsqfYCyN+KoiYCdKtR3r0hTYLXi9yLlZW3L0MbZeutHHbGfjDhgSdMeypdwHwfZjkSsDd5+903IGb8aA8mC3uQ/A9u7phIrWBHN17ZSa/Jfxl2ErQcykpC/N145d/ct7JMkWDk2pj3Rq4hUjvj8EIWI4wtkdYGNNKEM2tyQkoMryVqG2cHc8+EnaP6kOkuO+aL1S1zRy9UJu/2MuDvkaCgi7FyyotyUzlyoIuIsnND8OAW/Hdqu+N4roPnhA54VniyeodXWhlZztsshTxpPiZCehyDHbD1eI8uKbt44PAAAA6EEAN9Hw5FFSwo+HeyV+IyKUtlPFZ7Ql4U1E7WDPbBhClYhogIjk+DsuT9PiAmDZUJoFX5C58DmykvRhG3Sci2P7S+fXGsvrDu2fh8saQOxjsDEuPepXyU7CJKbFlNFVV+dIKxtTzksBjyOOgT2rvE/MX0JGlcdqiZhdaZHRiMKHONTPDYvI1OTRp3gPzs6EeiiMXk9RfpimoLFfLt6FwJrweCQjPX1q3h49JEFKc5P5dA2lzosdKoG6hHSi40HjMV+KQwDX+kA63OWo4HQvBtDKdfOURqr/YOlfTtwUzFVrXwUQriYWU0EAAADzQQAQnHw5FFSwo/89u6xm22oae59PhO/E38UsKALgEhVWM41QBH46ZMIisOj9K2mDaXuGJtCGiGYzZvOYvtFNHy5Iirl+CsxgGEf6vShhRALTWth+iMYIQdzNufftp3aswy/VS5QPCUkiH9WN5+5/XDa2ZpyqI6FEbkM+Xe0zDNDU/b821CDhtqWGft4frwa1fJC1MiVnFbfDZlRKjhSJYnVZtDnm3r0Nvls2ir0bT+3liQ2Skiy0FC4cWYUqNXJbwgTyv02DodwaSgsvaWorNAeKCLVSYgq4JIfxL/vQzkPgJyCia2QbI/yPk4jsLJHzc+DBAAAAh0EAE0R8ORRUsLP/bbyn4QALAsuVgHBm5RJ9bCPtHMyaOkfWDwR7A+9ebZzb8UOmXrQaEMdwf7cFLXRWEm0aXY2/Ckqfj9UBm2mPbtJSHyH3jvHBEf3OobigVK7qs9mKiWUhF6PZ55jG0TQQ8gN01kUMTATsVBV9y/8CLsTYD3hLl/L5KEDFdQAAAWsBny10Q48j2/bo2ueFdPgzsvBVK6PSI0vQD9mpqmRe988OsjbDTZQEjm+eQBwazK6jGUuFvoALQbB4qOSNr77yRXENktZEeV+mZYOAO7yrP9Pt7jV41L0mbzB9lxt+UfceUOCbhNTgKfFrGWOi+0Ju0Tu5eJc9KRtMleOY/2cUn7vonDWHS39B/PvaXVPSojEOVdCskpKIBThljh7otT20vqulP1UaOYRVGRWyoULIFwkAVLhaTJYtmHlEe7TRq56OgG0NgAvAlS4nZDtzv3+m3lguOo8tKLzlDOcctqvkI64qjX9xpFZzm8aScncBmIHswrHrA+eXXVFZlZyqUPcWQ4DCxI5vRfRaJldlpPrWYfrenXOqH2Q7ZV3cJzBi9LtafJ7PetVJAoSlQyfA6xZSwYU3dJrqI2Pw/4Duj4a4Yuf5/aa0Rcn2xxYctVLmkMFdVK/FJSCqaI9rygM7kk2nej09MjZ9T0CB6rEAAAFFAQCqny10Q09EAzOkYlBLAvblugYH+JlN8k9cU4/zNuMje7h2yuRQf3Ig/34yN83hB46SwGCshy8YUiF3zibr8VsI7WybM+yF6k10AfEbsB5tIc+LMEawyUJU1nQ14HZ5+tWkTiTSachJyffazUyFY3riSuJj05g6wiN+xZhddvd9AqlstCX36bnTkXMCoM6i8na9Z38ydrbZLqZNcfRvTdwvbzz7J0dgEB60F07uAiIgaM3yGWKI99lwPMsItq/kz8QmLcgGvxpD7IxiRkUxQso1CEQd0KC+i9BlxvcgFajfd4jb9ymdxDye88cHuHPKedt6TaeVXcbVxYR9SQbVUqmvi1QNkpf/lWD6iUCJeXJjnGRum3e9X++d/hYqvFNgFFOYE1LnswsggLaVpu5dEINY/II0eYl6ZKqjvocKTizrkIg5AwAAASQBAFUny10Q0/9H82U1EoOo5bfvjb4stjLzoTvABy7l5KPnqxQiqYk8KOeWsTCzC0ZWvwqQFPi1qaiee7ilrUT9qjHlcFolKCFU3miS/YHUkgr5fsV/rUfY6LuGFeTDSzSXVPzRRqo0KlMtsa10UMw0L4okbYlrJPXfkMoUmERfN9wvBkwHPnZdKX7LV5Sd4sZOR22rUCk2iDVmxk62eD9dIIPuMRI6QY77BSIZZxt/x7W4znCRlGBAxw5zLc6Kd7ebnbr5J0YkLLLHMbX+HDZIGJAW+HLj0hDmOrPaDZxrsHwAipwWWLaA9VjF+/2+QHXIGVi8ljffQBQ9hAfhmYuFw9Gs26fFbrwZ+qgAgvHLkFz2YhIV9Xw44R7aswPONhKR8mBBAAAB7gEAf6fLXRDT/0sh1ghzgcUJ83+msAL1rsXquZw1Kcpu3ZLoK1goBY8rPP4qMmKJzmsbonfLeBDRn5S8qhmAAgBT7VQQHwJV/LuMyltpJ13kgXDnvDUFP1tOhLJfLFrmz2xrl+PZwvFezi4sGman7Cw9wBa5WTWtw0n/OQiuQYzI8ni1RUDzcRO/SwGsGRdalxAa6ykUJ1cthTtqQZlwrhyKmYnmdjibFTzxLHuOEJmydHzbJQUdIWP7Pm7Or7tRfQd0Qrq5Ma/n0PkU56Cbf2C45N+atfwNfi6ZOzckAaOwgaVi4js0QWdDn6u02q1UdMQJPtfBY/Zej7A/BwtsyKYXPmgpe4XOHeXlF1zPWzm+5STw3ezXf8nw6RJpChT21s7kwTFH8K2V2gp0FSW0+PX8BNbjtqdJKScy2oa3K8CdRAwNQ4GfLL7tonw+9Eei7b9JaH/ikjh4ZMueW/FyCXk4cP1182+8UwQROWDYTd76CvaKKw3VwtKB8lLyxHtIJdn8HVBFwx7aORSAUMX8kFRsVHC7rXg2aFjLwUvycfDP1cBGezJSWVUl1gu0fmXCyk+OUXzHVbYI8ZGMEtfUEdUH0w2K8c7WABeCbZwbGADia1cK9UfK8BfXPChBjotST8A95Rjmm7bBICihzkshAAAAzwEALTHy10Q0/0XxPt1rrMkVYAXy7g3Ju6yZLOQXjpWFmQPN/o9Qy+kBvAG11I6Busn88eE6pua1gOSmVeU7kohgFeQi5Ge0j9+OyisnvsyTXbmrCBNqMwAd3wM+IMMi2Le/WIFQqn5kzFFmHlWBVNXEtP2Gop8FJheQ2xytibHR9kbwyBSmdSPcm/Fnmlxcq3ajInfs0GEX1jQfOb6+Em46Besrl/mltST8CrrgC9+DNbczrZ5DQbzR2s/bXTXlmtm1eRsLLaioLcWqjDiwtwAAANABADfR8tdENP+Sj+GCnC+OWb4ZD5MXgo09yeAEf5ks/itjQli6aHq2Nof6JvAADboZoGawcP0WcI4A8lC2YyohLoAQ/S6q2eyOGkp1tFFbvePkOHg8HX/+qEhjmBtAbdZ33DUxjj2A4djvhgAXMly1/AYwO8loS1hJnsJypqxq50UzcIgGMyO+IEx826TmX05Pr+xYcgF9rdBUsp5Q+6CASGrT+9KXLcAM3pA+1+T60Fke2dM2RchOCQjLFIMyDcu/DW5DdxiJhPkOUcrXUQ2BAAAAlQEAEJx8tdENP0ZtSK7AA8plz/AqBv/BFQZdtM78uaYrM/YV1w4UjeJJ5kPJvcUC6Jo7Hl1AHR8WyY+2WKgll32Q4U4iqVOrxWM1+AbkwqZfIulrU3BIvQc4FsD/L/qvMG8fEGyYimni1uBj/2OqBGjRUAGGBlnz/Gq4TgmMHb8jz20v9HnOb0LXqJvWEBuZ0bi0f2TBAAAAYQEAE0R8tdENP1pHIQdlyk+/k8eAPF32ULwGn9MynWdbSFSW8lOnKyBZkm1ANEiKv0j18jX+mSIp6CVvNdq6KcAB0V2kRNiC1CxDBVbpTJyiwqIS9LAaGFFPib3V6hbgGXEAAAEtAZ8vakNPIRqaYTaV04z2UNWmtRbZQQG5a/xveY/oLMXcD4Ve8LQ1Dj++jZ34IwJ6WK3pB6kRz0JfDkWwZLqoWUdOCGwTStfRpll/zDJrVL3FiB8Boes0UZFLTo5j7ea0qaZCh6+zgvDzHNQUFSAiZPi3rlRxmvhzWEL9BpBAehpllvUImGl1JBsgE/HNS8ZX6ZBGiAf3FZ/dQwt2SetLgbrJhXfkyvxVrYQrqJMHlPwvgkotFRpDF7PCGqwzCIL0qVkF8l1d5ep6X0YEroUg8FzMJePrJ8genZ17yBXDoQkGk0B8lW32QatpJT7Kfw+pjxeEPHM7zfnaBgWTIaP4Hnn4f1I44GU30X4VlXmMyJcGBPhnNWGk95kkHA0qFY+UUmbxDUEWohGlRl10tAAAAN4BAKqfL2pDD0Cs/FjgkbCo/JKjZ3wiJsH0UAx4+Ybnf9LUHy2xyx8rRXDA0DSZXwNCGrK4q2XU+Rgckxs62qTEb7PxKJmPFUMEhkoUjjhlgTe4UH2KxkvNo7jX8t7j84ra/S8jFqCgcWqyLCJTIbeetM4/ccS01HbZwW0IjqxfKKRLE2/rPL8J1V65vklqolvIzxq+sWjijEQx3X9Xzz0y2zsXdztMY/dMlB7rgNa0ZGFExoWYI57jCmw7zHFREHwiNejjXKH7kNPcYVShC6V+NUS/Nol7m6P8N1Mwf8wAAADiAQBVJ8vakMP/RQhSV1QImzdz8/uehVVSzoAkVxH126zBYM8/QCaXODJwJoXHEVNjmk6bh0Kauuoy0CjWSM933z4OFamvQqnbiArwAcCdxSNd8PDZaliB7lvmZU/ypW7GoH4iSFVKHRX95qHn662E2AowIqdsW/voff8NX3kiY6FKsOZtfbEf+N5SSEjB15nP4YoPd6jZI7h1qLXQk4ENvcOVsX13TpujuBMdR/lJuzC6xtpsnCWcApZ7vi1vDbuZVS70EOmhabTccqM0lN4rBGpEojq8M9a4FTRwuaBeaBMv9AAAAhMBAH+ny9qQw/9Exn8EHTp7BYs08SaZ8LQc2CXPzSvMNo3itu3ZMYqhTJiRea0c5Ezxj1mP5apP6dN8VEsJOWP8D9xjb7ib0+sPPdbPIpwFyc4N924MSMy9tF4CdemVBrf33J8YB19yq924or6hfa/qVNaPlx4o/IpgQ7WWe+2pDKv6Ia0PyAbuHOgdTCpl1fqD3g6aQXNrQlegSylbnoQvBqFixi6SwuzIGare8rzOAfV/kEk4KK+sxqCLGPfOqjla6Qz/dVmC0REKolxfXgKscKOlpiXSn3mZYIocBJZatljrLipiOf4NGpmh22N/Hyp6DFSvC7jQYbdoM+LQI8nCGxGISWqg2xC3JWolUSTyP+5QlAV9rVaxpDtcMEfH+bboO5VLmy/vzhNnToxHqkK1LQVpjM8AEjMZeBArBKPosGGUhjVqvlsHVoG5PTACV5hdQ9ft6umQBMUahNeMmvc/C8vXlvdYV2i950AzsqFDsR+qyhDE+R7Un9OBufJ7npS1TQjYfzMe5Y8KUPqd7UYC3QEVqTj6uY1V9XgbjM7jSfjwO1ZUqXreqGTl4X7wsXHV32E698PLgmFYcSwyr6UtuA41f1QRumoqAsMgG9z7eBdKp/XVLlS9oIdMoSdgpbxpmKBT/Y6ANgfsF+G2Sr4yi9eFqoNdm4LtRyiVY0WNuSo0zhk9/8k6UCMqto12UaiL3WgAAACoAQAtMfL2pDD/RLeaA5EUDPD86tjW5y4DWR1Bk8HS3dmdQImY2CP1vPj+xPZLlp76COW5vHSmoWmmWMa95Vih4zqWQFcvpAMTV3tAyXWQJkBMXi7B/vpYGgHhnVCVK4V7xmczTF4vgivaWxHw8caQAwmgwbdiRD4y0oNIKj4sCJVXsuJck61pAA/9odjQT9AsyZNA651D0h1lKMzb2RJoJ2DTbBtwTK5pAAAAmgEAN9Hy9qQw/5ECNp+IBjOdkDi6veAL1XXceFaUJRUWQc9MEyArx5DkpCLEw43viFKIr53iuIza+iAUJFWS/T7VGTcEKzSsBW6NSd0GcrPcz1HiWKzKSOynvJF9c7QEv7USdyNhInHNU+pVm4VBl5SWfWf2sBMhiFlD7suEGX8BTCInYIwmTIgzKQIl1vw4ykRWHFOje1NLKcwAAAC1AQAQnHy9qQw/RKtWW5xEuTyxpMivuWEGZsc+gJ4kQeSdOV3fDiilMea/cnYvo30sJH8JmSQBYLkmiXFUOG67ZCyY4KIVmi7Kydt1yL3oDuGqdjtlk5m+jZ/6eoq+NYsIORd8jhN6FaySkGBwzpS564xdZX+SRnFGEZ+wDw/T3cv/Id7DcctifUrPyJ2tbOO0tlp4+ka4d1Xp2Vy2I1QscnAfyD75ztNTiq0RjMsk3oqClwkuPAAAAHkBABNEfL2pDD8iYEuSs/0by7hl2HRACdq9hqBcvEG0FUchk+XxUv9WVHia+RokAIr+QbYjx3YozqLTwg6+WzVFvbQwYEVz7hPfBlZr9bXEwN2IP/7WHIQ1LFoAk4cFmAASrfQgyWYTqD0NJi3IKCIl9ONC3bPgY2JaAAAD+EGbNEmoQWyZTAjn/wGnnefW2ad+TECmKkbyR5kqPlWC902bi2+6iGEtiWusvfWbstGWe9wuuPjsA5mToewsp0gVZXWop0FDimpCiZY0UPvJPaxY9Zqh2ptfbuvYgidUqfPlwjnfHMwJvI9/qwN+4AAWXk63QtEPVWGXs5HNVydCWuESyaFUE17Kd82XmEfQjHGJbNepiKPv0gNvSyzY+87viBKPlUHoNAtlwNUVh2d+fgG3kqrpScuXKnDrglUoqyUipNSpOdWP5mb5VywQ/Sld5YaDc0AMbjGQ9YWHlxdIuiZqqDEEPS8E+bU48lZZ4P0LTBy+GJPP9kyxQkiECfGOWarjr+PoBj+/KaAuCg6x+nva393M4eK2hVmpic4yQxxqBRAcD7NCCehkNsRWrB/T1PyFDmyc1t/eXaCt6f9a6AmLTmPyy7/IdkLAXG8Y9MTKTbB6O9KJ32KLw5Rw7bw58Qzz8I9fXZ/JPNgUM9nHst46TF1t3ULGR4vCjzoCrPprXZ6Obw0s2jc7YJ2Ci3qfHtwXhZmqSgbjWogxso1Kh5pgWwuVEapHw45QywJDjQOMe6QHcCnUQj/fSTqcs4DfvtKFN8mTK4js0wOcVyTmQaz7PGFOYlp4+DOk5VH7VyVih4elq+c2kXk4I2m49nLtYR7uHkT0R4NyZTbz6x2U0luN5SXeItxpivgOjhRKsxpV0F6sB8Q9AZKCuGBQU6tRp4cdZHswjtJo6kezAXIoPhBl9/SMZDKVDOGbXVnB3JbojJms2h9C2hL4eF74jS/0mftAQKlVcPGU5lX1QWatB7nS+ISM8LjnsvKbDQ100hbmk6bLa1rM4iE8n/THYIuyHSYeXytzZQIqxccE/qdgfNKx0sbwLx7fhBJ+yK02u8eiDF/gNjDbBDFXJBPNhY/FgzE2Bxry0MbaeJqEc1MvKZ9hgljo2j8NFsKDGQrUh7BBQfTPLq2QIKRj7f4VaT1tTE4UH/5aghXwNFBrtq4qgPcFFHmllkLSSsuRD1MdRo5euXzgmNBOusHJE0F3jI4FuIIPT84DPfJqqH78R92XZpG7+OeAMEbESYCGHzKaJb2yj6bgoYKgsQROt7TrIgZXA/eMQ25Vip0IANqJGbZTHGyCV7h0hB7MD4am65SvLUOl+Lo/vOfVnSq7YoS4CrXyTRKTPjWT6hmhdlKJaDqFnDobZqgfBCegZ9da0HjAYgZWZOCUZcSlu4d+i5xbQf5uEVynkGal/6q8ra3Gf2VIg4jkjee339tLt75TVacN6AdySnaWjwMipWfVvL2FUd6P0C5c5gHeI4J7uBClXtHE+ePXOR5+PmjgaKV1/oCITcgb4Hh9wJl4AAADgUEAqps0SahBbJlMCOf/A2o9071Puztd8VW0g2RXDtgMdBlW9txptBpXS/rSTtsrYhoPrB3FHIhXsWObJmGM2zLwOHdfnGvnRE0PoJmCcLIE327Jqse3eljzy6rj5SW4AAADALDazg81KmHITmO9eUJZ4Uz9k0YDfVYeddl7TuihwtUHuHoOGkaAnI88Dt3KvTxa/Ta4gw+CqY/UTiBT6jWSx0JQe6R0sV2YM7OCgw1X7faFV53g3+p8FSUA+pURqp/34dC81GqWIW0u12m1FbFniaLddZ3A5OjUbFDUiM8XDmGSgfLMsbYrLGqglMOj2CGYd9gjBxu6g9/mNz7fNxmoxYvxPm6R2PLlDwXJGX6rS7avbUcOfYgFWrlAMHVYJgufnLFDwJ8loo8zbXIguN8X7noRIEekNRRZZZ72pw/E+7stBB97ZVl0H+ks5I6BQp6x64hBO9JN+lArNSX6JWS+eP7frzwoRv+7Ia5yqfhiWgxvELiBaQB249btfZqOe8NzjgB78SvxNJ3po6MmURwlnPOxRoVsaZqGo77Sdv4G4icinzGjrtXhjyqdfSFUarKs59kaqEeQFpA1ioGAyh/xTOGjWNrAM2TDxYn7vHWy+Hx8ckQd19O0vRytmnVs2wOi+KzGOpT0tDyTYu5nEitCkOXqLr9kUoEgXZmgmkTbAcG1jZ4vKaQWpKGH/eDWRJnxR2DlxoDnO4fX9t5vRdx6zbSu1Cfh869haGlcSkDWmCOPLLIN0IUiJo27PDUZZv6n+qX4l5z+hdgDX/0ea7KPgziSWwIc2+XGa4qXDUaywkxMbu4MUiP/nJRsCTrLEuCH5t2bDZ5ejpQn15Et53EY78F3TapCtxrlNaapqj7Qe0LJc4I6BFvbxKy0EHm3Fyd7l+nitYjAZrRLoRNiRyF36zZ+k4tt6HCZsxjtHMSdjqDIcEqUzdLLmecS0yINFqrEOKwHv9CqVgiEnaf9xOc8VlGwc9b9BbsGqkYWM08V/nIUfS0BuVxDTpz7hNRI0dnFRmQ6vyRjMNXzT4GwkRslqgd6jfeEhGnz7q4o15JEdAsTgLiE92tOgOBwSRG3QVTCcYhZVW6d3L1qi/7MJuhq0GYlSYMvcUkuAMZrRtDBT2dK8qp2XmT1vtZHxeE8WgqExNx1Nd7vbP0A6PZIQ2+zKQvwNhGvCao4BsrlMWBNgAAAA19BAFUmzRJqEFsmUwIQfwSHKnucniifU7bbZVsCo65h/RFK7Kpu75NJO0rafXS6c9yP/iEZyjEJtWbeHFiU6uQVqb0TvxdhKSE2vqyNU7xDPRJ3utn9A6N0acotrXPW4LbwoEwDw88CH/MkJsA+QVroHOOJ5hnpP/DM/L3KAGEwRnxW71gX3tKrC99Nao3bKBj4a6wl/FoQw06RArlD6uec1n40njG9GpdVsbJ0u6eiIIN9Qdxc21tiCM0veiuUvfPKL22o3Yuz3Fn2u16KiOJf5KNNVJOOpjk1gjGB5ax7lF2ZZ7m+mpkcfUeHfoWmw9me1pSGsG8M9bXIi1DU4FBriflj2CM5bMluCFrVKU+xSxkKg9bwY/bAwBJHyO1pLr1PLNU9Y/1PuEbdgz5kzZWRQkZPXRstPwXk9DCmkZjVE0QJNRb1HVP/oYP5S1Rx7X4WiOCHWMKucjHN8gokquAL8fJAb/sV+p6gWqtCcLWyIGqXaRNh4KijwAN+Km39VWt9cUbm2PJwuTEpTZkHz/SrVibHlo6hbhcW4d/nQOcr/TbQVzsyDcRd+Cqt5JVXZBN7X7bV/972v+bFw1Zcpj6KJl5V6Cs1swgxsFSyJM63goxaHPp+URYpLrNZhrXPO9nb2kqVvt0hXP0wwxXBvqOeLgRH6zF0a9xNaVocydkCgzIGxPHTv4W3bfyU0GtUZA0Y5wtBEvWPjE59Eci2xBVUWPjP9ucNyZ95/86Vcta0cX9QJD9x7gBSblgWWMlVEvGyZC+73gX0VACj1/On5hgTfRZpJHcNask7oKt6ff5NlbzdPI/3LDXoYa2W3yxXKFidxkyT0R1pWyK8C+q+Hw41TRdHIBKTsxvSnv9IpkR59eIJon2Jj7rCUsjleik7IXJ7IUxP1gK5IV6Gvty8ILpEK6LkWzwMCOuDskJ/iCSCID1pXJGLM1eeiFEd+apgDUqMsn49525aalEqhQ1b6FgXFBY38zFFqKWga2vflhYptTeVUArZwHsT2ECMO0ln7SWxhQOcgdt5PEU27RRCXKHewQjo7N4gOH1tctkUzU/EyH1a1KEr7Oyje6SjAfJGx9KJoY8YwJBSP/QdtwolhRur2h4ZDveKR1sMbcVl5RVsHpocb1yaLKplIZNPnKZ4uwAABTpBAH+mzRJqEFsmUwIQfwt+GLn/XkcFCvst1TnQEGifgaFFY4EexljrULeYgiZykxJix/x+Z09It9KGQ8C/OKaryWNC9Xd6zF4BU6O5xLTTAGlG1xv2nAL9emiwHaOr7Ea093/UhMtizfOm4VDAY0njT2YyUIrNVpACAKay0QKbMsKviAw/p0vEGKetst3p670DuoILfWKoM6RuPzjIc1Z60EQ9LgoCUadz6BBpqHumiWXmlDBax96ptbWXOij0gWHh8JeDLznX/eb1A9OO9nFFPx5Y8OA/Us6txn3PDF6sQ6/nCALv+TihmudcnxaB7yaIBZ1/GopqoP+/Jt4aMpvTW3eGaYqBMxEH+zeIluP+yc+R/RIw4i2lxOxUy4Bkqwhbf/AvdUnnWlHHjsQVwooFDDVzEjRH6s+frxGknqJzR7GiBI0SpqfF/bIzlcLFzy3cg2u6fIXqUM9iRLlqDi6f/ZpXf6k2ZKGZN2c59hHE8kTD/tebaP89MttiG3A5I1s6j1iX3iQT3VydJwqHTXfqpjEwm6kv8l7/ae6avRXj74i1k+QJHaBrcFbFDJ7H6/5T1hR3z+YtfLL8QVDAMmef8N8LTV6N1cnVZvLHaO4JKFRIWYiomZXpia6cInb0IekdZK062f4rB43OX0uwwEIO+8lhk3PdBHmvvz8S6dzEtjUdyyY/YvDb56iqnFlyaVkCsE0RRC4ZMSC5SNoOsTgVhQ9Xc9/jkjbTEqX5b6wJKFaM4flOewarjnW3GBzal4xqKdMt1DBEnXuGUFOqYLENlhA4h3Y8QLMTnLg+JzIrF8nkyMS21DVAyVuIIbMrD+Kv6+W2IgXIEK8ug4D6n3q2aem7DzEPfTmbvu8a5HwpDGppb37Zc48KF6yUX8Xlcfq8JgoRsWHE9P1UB8KxvGQsppWuiO1Qq/wrNSvlZdVRPfmGFRlDOsn9A/RP6r2wa+KR7xXwnk8+4PcA0jwptcnRSwMd5i5m9lKBJsvV4DiHG0eP8a5Vek9NROOfB1/wYnO38XWbgf0VMHB88WC27My8MbYoyA7T7qNYy1oAcaNg7PbDduKhqBNR/mtprJ73HV3NVZR0w0NN3qLW7l0N8ZzcXx/WeT36kENtSxU6uT2IWPc4xTFnAoWXat2ynqkOTbWSrtJuhTb9Xu1Utw44M0PL6diBRoWCIzpBCCgtLqgY+HfvoP3APgtW+kBfjokluLV0VP3JZylDygrQSwZQMMwX4yTE/NtYmWwt3QX2sr0qKxVdu/f9PyBpZASI1lkWHrVWXoeK21RAMRQdABpI5+ppVCshPNMtxjc2XBU10MZNYOhLJtuf+nrQjoBwbnqBjLdeFTvn0dyRpGGzS/Ki/gdiITO7wqnFtC1vZLmT9zUo54VnaLBR7avoG93UD39nYmkb0YJwxPAZZSlCAekyLfPuAz0IuiNlSqVbBsfsfJu78gf3JyvZAGjTqFpuDlYMvnf0b1/PbL0tUtyr3+6Aa6Dzi1uNNOq3RYX0c9U2RNBd6Ekpdm7/F/fCIr4Mz9F8+mcFYA0APYmdd+pGRhoXGWx2JRPOjCMRGDYrsp8drnMHAQJlX168OC/Ri3RxEOXsWv2zFloRdiTs8n1DZ+xuJfF1CPn0mYeR/9S04R/nPZ+SillDf1Ma4zwShaaUKmrOzEP3h4LR20rq7YBvdRmIeJpP516kGI8XPnCp55vjfMsLdqzW6noYtCakzQshUIusVZ3q6DOQhkGdEoiyRCVAGL9s4kzOXcr3xluyJ8gwliunSNw2/65fXCtk2cAAAAHbQQAtMbNEmoQWyZTAhB8ErKYgJWmOUO3CJnxmVF8L6gNoXJgAAASJBRDTMdxV7AKR2bqJaoEB+pq+l4UOIfpkb09kBHxD8XOovnuwTEw2OgTtE2r37Jv310TAc1uP0UlTCUNewlkTOKPYzQMipFoyEdjLC4qtY6+dk2it9xN28R/HshC7ZtZUofqDpR/Hk/CIdRdXNKE8rXPYl3I7Ly3+2GtUgf2G92L5QA6CGF0AxBdMoWpZuRDSOkoWDx0wsc/TOcZWOTb7HMM1SCz3bi+Fzoi5xlekiCXV5qrnMSrIxz2t5LNMyebeSZScrxxl/yx16Ib86odXvokK5TYE45Mk7u4oPg9qbE1ZizgVEq7Fvkt3evD+YY4c6UEt9lfcf5u/8yJvMZblu+11dY7OWkn+DcDkVRAPFqsY8am9D7ooPRcukIkJmiEhJiqRtblq+3P9TdFT9LRaGASvpPlaGjpS7mpefXyRryDkEUBJ5visPM0QtPyH8rAKRHKuluBF/UVByHnQwnIh0e/UBWQomhpcmajVRgz/vlbtTATMau1eP2hI6LEAW5q373E3IFsuzHIJE8su+F0j9v9+5YbgV5JznPDUML4ZjDNyAbFk4sJIhPICf2IKOG6/Wi0qWAAAAjJBADfRs0SahBbJlMCEHwScENc8XJB60Ao6ApXHHBFUjkQ//9kDI7ZAuNQ4kXFdlVUAAAMAOEgF6EyG9nRqw97fyvS9GpGCcBA1DQKb4octXXFMB4SXvd1K+qvv/BOC2Fz/D9ZCaRdcjlS/2pZFvPPf5idKlxLIDscS1gNkm/FypTJ4HsSUP41Z2geNewM4UIzqEqQNnL/X1pbMaLSpZ1g8R6WYcciUH0131pUkr/YWVvwoiTq8eSLTX/E2N9waW2WK3rf1g0mvDFSLYwYjp1G0muSFAKEdE9xA5usYGqBevXRKwri32QJe7qaSM2Pv2V5Qx+5hv7nUT6ekuKFCo/DAJlqy+YUbFd+vmdy/QbdEk0FOGop4k+ilfnUHN/cLvFrLzNIApAOEAqSFUwrEIY558PwtOQUmDZ91oHqbHQpDjkAwM0kurHroMaEPsB27fPzdfliZ2hOE5blhUIjHdgtCib5QHu28gjPXAoB1YD7H3B0Byv2yvi3GIN0MO/6oqKrqSf9vLtV1BZ57im6jL/QEv6F/6MO6/DiWOqmED95NPF7chhGB8JU+QVWq3RFapD85mc34WeMbA96J80JZ99jiuEwaDZxejFvV4aMFMeaUjx8FndbL7G1hp53Px97mZKSp3omuzwRG+jHgKPxbMKgY7TIUWowVUMgystLwm008DJSJVwBDHTF57yGf4UggOl+e225conSFr4rlDVlYxstkIu2Gl+R1O2GVt4xEVxJQQwvAAAACaUEAEJxs0SahBbJlMCOfA9vGLQL3NZNmvmO5v4PjJuZqZGjavdTavBIlqi7aF9y0m7t8WKbLmQ8LuxkXeCqDmrazxZO592wdCjAAAAMBDHGv27N9zdrA5BDKmvN67INorONrkokRzd2Toxh2yPEnaHWf1Y83gIsaAatXa7SHQwo7xS7xsYMJlIZEsbqkbzlVD67BPKeDbL74SM09JqbqVRgsi66ndTnPdpO+UM6MnfOvq4Xu/wC7DnyeoeLf3BwoyS2bwcpDgU7/yRBv7kEyZGIc4CT5Iy/IWiOlQ14FeBqw/vtF5HUVaumppIGl0tTZu5cGVlLIRiHuyP2MbvbikjmvT1CttUf5lBA6ht3r3rt3vDhkyPhgvp1kZgFlWgoiwOw9XIOOYzRUmZyo8kPJ2mjKRg/99kHpS3pV1vhGFJCHSwTK+X8jAvzWYOlq52Ldr4L3B97G38hTaPWFGe0ZWlQYHuO63gtWsn32IZQs6MgwkqCpVhKRbDB4b4n5gSVwlJgc8GLxsXrwrJPKKTTtpE9uBMOI7D+250tpCfKMXyH/CaT5XefpEbYScvJMfxbHAev6Wm0EFtiXF0+95QJMf+x3rTGeOWOsmHbg+NRwgOClkxYJlKgxGd52ga7fh9YI7Q8TUOHynqKIxIc4FYKEeb1HMYFQ1gCjrTYGvBT/AUkPvxNOLFUruxxbNKmAHGyiqP51sbc+QrRV7TXijoAmoEXNnx1aVya/piuSzvv3l5TymBDbZHY4ncgTWdkuGseGMXa8Aqf4DqVk82EMDR4NmMmBVRvyMXIBy37xd/KldtmTUmIrW8lbaHXfAAABJUEAE0Rs0SahBbJlMCOfAiPUuaK4EKDrBpTXzQfQEHh/y8iZ0VXzw8x93LvMY4SFHK/hTpNh1W+ZAyOXUNFk0Eb/l8LuyK4tT9dd07hFvf5ELvHSj+0kSXB8mDsEuM1MIr3mZV3aHBYmSewX6jwCyv4f8JXtLAZNMnz/KlQFVt/oM0mXGh2gs2/1ljKwoG4JOuNFjukcCQnXBSKbJp99MpG5tVk6vp9wlTk6Ts9FdJ/EVLMDF6bBVxqb/z1zADyYaPEkgSQh6G2Puk1gCBUugyVMXc0eIhsXePoKPJzcBxbr2iGqnDSzMnioOtW15HG2fCezEhS/RzsZXNzgr5QPLlZTMphxUEnBdSmTHbtCpdAOrZ3yGbabPV/nE2AOYbKDXi/lhGBAAAACq0GfUkUVLCj/HRxbWZjI14UaLxkVuc5Rejf6KDlQh3vUCvgJMbYsf+SoOoRiYpsDaJBhHaZGymkRe6mb3wWSfdue7MbClpTCtV6yIHcBLITB4EZpyJRkLKzHliAJdL+MgF8EHuYej/xs5pwoZW8jedrSgfidpEarzcf1mEgQ7Jkx7BTrDEF0vfUhpU8A8EoanhuM8t1gdCWM8fmPRuTkt7+MWwgCf9u24dDpJ/3Jtq2R/4C4uYUUT3nSWzQ76qMqVoK9HGhAYkCQR+pgllSGfgxCJ34cSdzW6LsSIu8HHBaK/EE3JdJsWJq22iTOvEtrAjSlyKTpIKms3aLaxLbPT7KFFnmY4SMvXXFYEquDFZBHwNvuuQbFSQeB8akPHAy6PHqYTfN6gENkrXN5VGQoJVwkPi6sbtEhrV+9UonTO236dZVg0iVHr+OMDFEdGTKPL5RhHKS63TXmre9y3j0AUKAAix9eawg22Wfhat6aF/lFXA46F4OIQWzBQhSe4wp0UjPHttlh5b7QTBaczhJ4b7aesps78MVLAMWfcXRk+L3CEXiZAI9z0Y+xGRFtzbkTQi0IZDNOB6krYkmp0EDn3p6N4kUtj8cUfBOCO5NVzrRMgxtzffKyUYxy4lEGvEqkXs86PKMtKqUetKjZCkTSOFtaeCb2j+jwzuShlpSD/UdjXwKhdaaOYjJ1naGOp+PAnbsRe5dahsqdbFOZIm9GwyKvFhqD/BejS3AIwJC1rkxnXrJwif+hOqwy9dQnWwR/Vt/ouf+V9LYd3OasYB5XMPrgJ8sZ1nvGmVJcRirLQ2Vcp3k9QIhZhAfv79iKmZOfvOD513kkx8o8Qd43fLT3XsxhmsSns7YZK013NrFDBghqNBIxol5YKqCBtMoOCImyYkTKG4V2H5CP5tGBAAAB3UEAqp9SRRUsJP85SnmgxEXlxbSIQSWOMSJSjPO5n2Huy+wjh7RFHnJEG+ry1/O0Fembv5EVqlV3vpGrzJ/+NQ7rrze4XMAMXi1CbTWhzoJHubTws62EVx8S7DA6IKmoA5qmMgMfPeAGrimuAjhFCjjuyEKv+jQhhA5FWXG+qSmq3Ieg75BXg2XnjamGBFbcGDmNkKcpbmIujtZLd68ZP4WaFTKBEfVmqH97DY9wX09uAu/tYJQvhm3CfI9C4dX7rnkFe2kbSakPc/BxvDHaOKmlLmfIzoBCGwrkvLTNnJWLFESqRHcBa/hxGe8ai25Gu6ZvlqDZjAs0V9pipVDRm0D27s2+mQrbbd5reSKthCLDGcvw4slOoHyuQF/iK0qZ1t5m8MkjRl+ZT5TKLeKiAfVm/v3kC6Di8DcAqLgs1yLp9fYMgyuhnoALe+JzynxL+g2Mtbv2l4KKP6RTNIoEHR6Ltbxt2RFAoAp9p2rOHFR6I/1MHhHWOaWMwK/AH/oNEivATMWPG1CCYB19PU6vIe05+qwc1pqbI6Pb7fgE85nvxoJj+g1z3tyn7292MBroU3e37keZjPjVvcCwaPTDdznGGJ8J+4uFYpQZgooeHd7dyvx/FywheCOncASxPwAAAbpBAFUn1JFFSwk/PBsoBoTa1+oP0KGMMgtnmYot7YNZPv2uuzC350kPkZjtQ1fKatXWhOdX7DKBJPdh8yp1R9k5KbgJxGFCXcp+O8uScHUgHf0F0P2qYf9/NOBS9fiY28qy1MRvvKSwIPAbnrKv+YIpKa3J5Tlz93aLRMcydyoaBKnue1zVOyYbRC2UVZ1Q4vKa1GtoLQ5RBs8h8+Wxsqt+dUPJwzBtJhcrUMcEq9QZj5ejZGz9zQhS3xkMz+5hNDx/cl4CV61OKb0hgynhnmpINXguCJQObwoUk9NWEmTAjkM2hnlKzmKQfoFV2ZaAQMLJ00Q+sAKQfbnofUvtkEh1wifed+58BrEh6fS9MEQzCzLPppQiUCV9VjR29CMutE009PZI1lkmnZe2MMWdbrnsoq77LUTT7lFUv3qQrz8qVblvShwZ7kazvkDpsQqsauqyXOx3F8ZygMAZtZUAaicFrEU16k2GWf6n4/nwV7oL7kujsCBkCraLxo/xcFtOJQK+txofObochyc+6Tcjj8rhTPV4NFvQAaXNiuCFbNAW8wNABejaH6nh6xMutdotZ8YfYMSuNQQXeNgtAAADy0EAf6fUkUVLCT9c9jUFTQYSPZ/7gmzUSTxQo+HXqjEtm4nq70uMQL31rgQ7GpYLSres+kRdc+1hQ+jbb3c2F0gOwsvzEAzPnyNmXzuUDITrsYVT7G/hRZdoYZtljwBIjC2FmnBU7A6endl6o3uX/g+dTzMkfwyXEwVcE02HhuXEBooTRb7e+1sG8Y2zUqJrDt8+gm6FmeV7kw1JDqU7wM1IDFMSK4fo9iycwkZSQFG1F1MjNuB8oOb9S7XDzHjyOxFuNLfrSUjjeR7g4qVrnZwsdgyK/m00J0qK3mSA7QwOC1JNsiLFQjABVPboSSMFjuyRPNy0WYJgm2DFYMv9xu+boqP7CWbyhQyT2l7HYquwhP3+zKmsSlgRBiZ2bnial17a6OyMwHNaozTUGYrxiTLyonAWoJZJ9MnzxYFTt6CODt4f6QO6n8T1d4HX6TEM1iCpyEazdH+FGySGCXvR8cZbKL8Mu71mBf5d/ZvleBMoBgYAaAb+aE6Gb5Dz8fNiJziQ1gUS0QYLd7nUgOO8B32900pgu3QT9c/DnBc5KICxo2S8j+Jyyrx9/02GVaYD8q7iKyEu2ZBjSrSYwa4pf0+DtfepRn5FqJ/rhCa9w9Te9FF7dSt173KSo6INjvSeErwPcc4F4JlTsSBt41THyj2/btp+NeHiWoPa8c9amGD3Sq0qRp4wcgEbMT1YQiFgIipqyw+zaT/Ft5JkBNwAFgt+jXmJrIjGARpH6vHW3hZUHbxBMJcAVZ3upKEobfIdCIK1fYiFEfWemsizCmmKgJHVW7Nn8sb1TDRmndsWgOCnXuX4A1RqAAjAeiO8KkhT4lwuwrhTsWLTWqlpeGZEU6NeFYtIKbA6d/+uVpFLosp9LbfvwvFitlAJng3k7WqKABbgNfh5TsJ35AkiBr5tMhzR5w7/co5LcX6OPXR4fmUIkgdMy6A2351Y3px8Xrhccp77M2zSZI6UmC68GsUHHO+uIK29wFU53pkJDhxxc18hheJSaw1B3s7GWsoHUz+ogg2YM5dF80B3jbxB8f9O6f2/OStbzJ3tCeUna73evgbw3LVuKpY1gGhsA+3UXfZNaOkaSdGZxIOSi1tcP6C11Hb9FGjuloQuLydq5QWOXAERpMkn5ei/29ZfD0ipYPGhVUszAgl7G7uv1YGZvmz5w8F4OcqSOlLWkwhZsZMCUgN+hMU5HSTUE/a5rm5ugsgKyU5tEAu6ZOE1h0gG8Q66f15sK7wl6JF61FU52/jUx17N21BSQliGdp0T5fEL9yp6v+lGJuTHjvpaVVDDAAAA6EEALTH1JFFSwk/XtQv6llMtnzhgVSW9R52xAOli9nz4Isx5O1O6gTAm/GsXRw4BM5FF2W+xu/DaoH4fRpriNt6lzRrSUEY3RRKetYCRsew7gSHgXx5Gr7u6lrr1GnpFRkRXKvUtfEOYA9JFndWZDtAqfVrSxum5g++z7gh5fZq4iT/Ooj5cb4bgyG4aZkHXsK1hN1Bs5te6Fz2K8p7xO+6hZ/Phq8A/gpckYT0XpUhyBuay23yZIPhEcRj1aiX8OoByFix+gPWm9AU8YRX27eOSP6J/xAd6BGA/P8ou4AwXlEokRomWrhEAAAESQQA30fUkUVLCT4OiwwtXVS4+F79UExcCeDGfgXJUjDyAk/fIsIemuwRNxQ6mRDVl45XXriGNnBeKQM9/w0zKy5DUC1eESJ/6LVVI4dsBaOZzWPJHAZykaaz/cPk2CmT8XI/uOQBHyiuQ7Ri4h/Xddl+7DwtCzMiiviUryfyeaV0uMv7BWwUwus36PVpo0faJzKgqJEg5frB+JYf3C+yWAG1UniTbhRTFQwwv3DWplWsDg4HUZGKfllpfwcFi/tIpMQ5UOLre6VnZngKEp6SnrqcI+FOs0ZbvRTJpdQCNcIYUnDP6iibtmZsJBmdy7O/O8jkyKePUT9iSOi94sDELroNzqs9KQPV9fFoSfHXhRR7Q1QAAATFBABCcfUkUVLCT/4obrcj9fhwB1m9kU8z4HgCue6O+7TD+Siw41zF9Zk2+mCI+LgWmjVpTPOGtOw5D3v6sK01xthpf423/tlUQhw9D7FzPztRP2TMKnlO8qqfsv9hBxjB6lsjdGXKm6NkhKPce+D1ghf+x6oADtlAC8sDUq9ihEZVwlPRLMZGsh8cgNVnrDPxB+v8ezYpD+jF+FjDa/DiPxWQlXbieIMIecXm8pHUjjP5HSyw4+laYcKW9hFupD4GyO4d/yTsCWIQ2izZR1YsbphnTOlTiZxvsu8dfdJyAA2lZH5JrGLA4BRkLQiZvBcbEO1Rm3ET8XjMHMNdENaXLesKVIgTcZBsnpnS9BnSadXPeszqwMCg2cIO3wCU2P8t0dSRtJMZN8j0X4VwbfhNAyQAAALVBABNEfUkUVLCT/yC3wAC8yqsJc4yO/sLeLPP7sCl0pZoVBUil/s4guJniiz8HMMhRZ3G1uwwbEEsDcBIILt3t1gTj0JQYf2DzJSAMfeJ2QnvaODxSzmSluTfA27aGilnoa4IGK3iw67ZghjPhc+Vn0RE2dtnnjS/wHehuaCFQ69WnZwlwF7PuZXJfnmCfVPBrGpahSG29X2boFkjJZWgrmmfvj3ssMM0vH+N6Nv183QZ1eDLhAAABHQGfcXRDDx+B0oSz0FEZwZ4ujP1Na7HIisaX7ryi5h+IYWJ3zeuXufF7apyXY4WXNx0wAjmzwfHzVWmhRcpEgNQMR5BTeBaX6fP1a3zh/zyw3phw/79vCqEtoJBLBKZPWbcfGkVhnpTcSxW7Q32XYyRzXgASqN7PPeXK/4WMxOLEecJzLvZKNj/bIbGi15sgTrUE+pltei6N+jgHVGgwR9bhveZN2CQOsf1YWr+q1ayavBnXnO0twtTd4i+ejDE7v1/t4MNMs1Jj51LUgac5zKVr7GfMzN3H+5kOVqg3VS6m3cnef8cHkCn0Y1ZWj2K1vKLiYEUbVfMjIiS5g1+35cm8NXM7p6o9hl0uk6Lh/gnRfWPF5gPYmIPdynv7+AAAAPQBAKqfcXRCzz4bxK98+hdyrbVyy2gAcE0k9hWFT2kb21gN8LcPR+eULUBe3b1Bw7rTWdsSvh4ySK3tEvqAEjb3+WODFN2LOqxitcBVfjbCemwTVWAXIKPp4KxtUhEsu9IY3jeBVJ1c2/bHe+3OUc8VUzO/kN4bH31ePDFTS0cQTChQhzRrPh+DCLUgU8ELrDe0U3qWAUiCL2Q/PZDWig1vRzpYKFJMkzGUpZD1RMdZj5zlaL7QELMXtt59u4rYzjkAFIvBbXk19P4odBnbea4c+VU33JEXKOUE6ReyEuQ+EAMjF0BDm7Y1jJu2Lb5XI4NVH/QgAAABKgEAVSfcXRCz/0BO3aSzb5H+N8AYF3iJhSna2Vm6F2N7lAsRKT/6QjG9AnvDspWqV77wQ5g2bbtlPieRiYCYj06+DsJl4CJH0tA5Q2zp1+URNO0cTz/37jCimnFQiwSQEx+f9j9/Z5c/OB8bC1RdfrI5muXCZ8+idl2eTHQ/tyOXluXk/80a2BFM6a8QPC8Ti0Bd5xInHnjo04FdCagGhJG/bwnZ6o9MQOwg7RqnZPpDDq1RcomiDhFBFGhxiuNnwsnrFFW5XQ4q93r3cK5pjPHn98Ns2+f0icWVeLIqUXCnPHHXtX+OWo1ZY5GCnLZldyABPwivapywRxE1Ds4hHujB4h6JiPhEsqoHs4OlBEBFPbi1we1nhmK5sofUlmUpQa4WaDwhXeUR8awAAAIcAQB/p9xdELP/QTl7xmqV+xgAOz0Bg/+BalJ6CB3nMr6hFkPLILHpwb41LsgFcs6caISJ15iKNtflAO7HgmfPs/CneBGJrSssGyrfN3Qo557rXbwcE8NKQYkM+yXF672g1YImdIcPA7w6lZOLF+I9weFeh7mieWBuPNBGHZTrgaox2vSWHwoBvwU6pSFk26guJ+kkeHUKCUDuqTblfk3K+o093KXkcgE0YGVrpsgz4mSipVVYhWDZFazHoc6yH1hLyGMh6dskVetJbS07opesFGAQeyT1o2Vzd0n2v7wa0doeP8u4uGZwjR9OvMf8PgrsSJG/yTcM7EqHtyWLwgqCKb1c3VrnmQsiJ+kwPeMD4LOKkN7EMhgsN3+bA4dMSmapaLHbo5ii8FDeBho0Z2Ce9wBcYFNBAzGo4Yrm36b+GvQZjwNxsAO8yaR6z1OPMspOp0T0vug7RxWVU5ZXt9Qt/nT7ADNV4WNTErqWgZS/ngvX0fOTrkkf50FWIAhMya7sn7qbNE67eZ5TZdR/wQdUEjRTnywkshV13w3a0iZfNq+j1RsBAP/ZwXT8i/rw2Pyad67gik292FY8XZpWCcQMadFAjftzuzG83hA83iM9yHeQqxvDG2dtJLVCYoCbXt09hptXDkSLUgEzdxzPsJkL/7fMTHgl3cjUS8ST8jcCcn2Ci96lGIhNgS0Fz9hQmmlkSuRd4oyHSxqsLerwAAAAngEALTH3F0Qs/0D+lwu0fCfPnd3oUaAAAOiRc7DeJ1B5vesLJN2tYUe9QHsUGCwwYUR2T4Na9+TLudmMnUXUfrJA8MWQZfD3D2E20pZFDLbkyoisqjoohxe6vdCUChBpoZNbUYtRb6R1BX8GWZhc5ZM8nGi5/JgnmFFbyyfub39gnxAY7obnFTUfSTQuCpN+SUzAIgSAZmyFAZ96Mc2tAAAAqAEAN9H3F0Qs/4rhztgUTuXvBFkyUqsa6mtK01XFYg8dOyVViDWtWEVLZcS0xCBLtm6CjO0k05CtUnMCXlgqel2JA+DNXVQf3Xfv09qkgTVKsrV4tTjxZymwRGpW//kBK7ZjsBpP8192uTaMWJZYo6nfK17+QC1fkXJ4ZPPVp1x+2y3+fUoUObzX39QVQ8rRAV1icVA7Q0EuaDw90TFO0u5+0qdHmovUagAAALMBABCcfcXRCz9BekFc6CsEngAlebCT4+dNUCJquWYpQOfq0KJnolUHwalUC8jgDL66adh88OBFgJI45dDXhqpp28TNwQIX3rajIUxbUBaY4NBS8/oAi7YQARsCrRBfynfzsP6EN0hCIOy+/h5l41S1uI1GQTQG/1koQezgaZUCLlDZelw7KjpVTyi1X5tykRZS6dw66l35E87gpjsanphPEdT+91BAN5AzyT7xR+j8KcvsoAAAAG0BABNEfcXRCz8hUD9MTsUAvP4BVXuJwf+vXlroREmWqfwEGGpRcuH9FrJ9O+HjYdwQMV9pAwZY5jCuKnr/fFq9w35KjFJlp399O9mCgmQZHP8rZwieGA17dPs7KOmkkB15Y5vqmesj1z9MpgSsAAAB0AGfc2pDDx/XBJCrYKHOSO8yrTwvS9QL+r8ovzWbc8zuZhOvCEtAYVNQdlgrjgJS1nKWDzD2lO0+T8Y6uaxEpXraCAccowexu6N6cGbYUXGBNwXYCwiudS+T0n42biS5e0iOdkALnKJmoxDHhNiqljMmwvCp19luVXftLO85tui+SQEpLUgZ4q+uLX3QAXAs4qGa5lnymfwFFm8CdtqScQjjsi/jM8qNDnnboGnvDuhtO0ogwON5rYdlMuVJq4xOPSCYEJhykVVZilX+X1Xaqgd558MJlnfGY59S7b89AXuRR+uBgRQdAmy/1nclBKnI23bTlV28jvHr7hcWtDKeZGbYBVLV/zxAWKd3IWwf3tWbxMx4yEiuCzaRiRnc6KfH8NiS1nsNeq68CHZyyGH/4lshesF860V3gUbnvcmiZEZZX1cF8iYktGtp0ET99cTTBdQMMVvwYWhyfYjv0/wLMQWyYyb09mZEOcQof/emUKhBb6BLKniH6mDDI49FneGKEiAqQFll72xunGZGgKOpyHVftvRuxafPvEqQWtxQYOb85VrnXRTtvkbzl1ZTD4WhKLdFvFx3tj3GXYrMYXIvfFb4gK9WPvb+f6szZ9UPvqXgAAABAwEAqp9zakLPPjMR1ZvnEUtjZrWIQCqsJo1bx66LtD8blnX9l+xDL9heUjKW938jJjU9sTGhw98mp0pyoon0TM2h0pmOH/26iJQdFrAPcSMCyRIpbQjj6y9mjzcl1WvyVB75uLUoEswOW+oHB4RdXO4+KsLxkKSX0EK47HgS1Hxv+huqV45WM66mUcHhPwOCVjUhAEQDm4KoEG+1JfyOajXMQK1FA82EKSWkhV9Fn8u3CltAlOsbNHolfhzy3c34Fiqe7JtMsPgw3WwbipXdqhElSxPw00rSLNBwqfH4LmaRml9GySIYxeH6t75y+HN9diUja8JvTRCBl6P/B2ubJQVW0cAAAAECAQBVJ9zakLP/QSS0+Zy+WsVKkxVVPO3n3BAF7pAuRGnUXkyOdkUnyA+cc1fOeRArqxoyu5qUO7lgUkuiWSeyQyXeMmnFKUAcq3gvHMDpqQSJfj3HJDmPFo97SV3bTauQ5D0yqq7PlzNEir8PJ6lO2DQNi4I35tq8huSrzZ8X9Jn273M6fAO4KxPaeS4vgwqKbdQ6wPlNtjVLZt8Sxm15D632YX7bV8E+HBK5DpZmjg1jiOEUnHavOOD0q4s47mbpOpri3vADyHpZ2df/BidFueXEaf3LPY22VAQdng0pc9bPwC/BnTU5N96q1bv6BbfRg4bvI9nzJ1C3onrav+6S1X8UAAAB1AEAf6fc2pCz/0Mwa01M4ZRpn/PEq29k8tRoiqxbKlBptO3PhzLzV2QkA64hPhx3RPBdim+iXDS6sDpyTJ2rb7XnooZoJoIqFGCDGyZQ6jL+3VSF6MinKmtJAZbwxUBdQOAFzscn8KK5/gqIisQ9L+x5VALNDLAWZNBU+9s/aLK+sJfr5Nlh6/Rf2QA9X/aIPRbIDn1xuwg3ZxtV6H3U8iMPJsnW6DxHVfhmpRhDWNkxKXD+mzLdSOkCMVGcwrD7Q6nIqi/koyTjOO2E31GUIqh1F3Uep36HRUoTlaXA6zMds0dxFjgBshTFMtOtzhegwXt/D7X+Y/NlgO4dCkKRHwFMz/2qbvtGXzEsllpxfk7tndngI6lox+D713ajWjIzyxThY5eoWQSvPAHIO/mayUpi+aTcNnW2Bstu6J8U/mdvi+BrtQS3GbRxo0uhzJl0Iqfu9e5hx+ttNKiAmUL25um3c5wB5EP1QEYfxc2NcTKbn9nk2UVw9LbICf+hMrIYZo63FdEzwvw9T8UvsmSQaj831ukI/lLZCTGpLkqFqVobt49tWn2zgeA23PhpE/vYvx5a2ExSVQHvOaMBRa6f5n2pbPFjtJJH7DIm/g0m1S2PgiJTyAAAAMEBAC0x9zakLP9BMHloBbkJfHAUOeos6niOzRApimnaSYWk4C0usIU2Z8GhotuxQLwm5qF7IgoPq4riBvzpCfCMTIYXE/PGGQk2VLrhHiY3gHcXvGE6jFPkEB03Lj15m6aoL1SHnIG14xBnxrup8RWrQEzstrqB6qU+yI044Os2yvDS5USPdObdPbxcYNle4FiwQQWvPA/gMLVLsFKlXNqQgQVZAakCysId/Ol3hLoVQKmexrKkXwCA3AnIEv7akRiAAAAAkwEAN9H3NqQs/42fbi79ppkZr+VJOorSTaSTFWNOFMZTbetsxhU46MCDQovCCLwVYvVRSTIFIbr19ID4QllRwDifMAKj/xOHi0ShzXY/QGEy6AcaOc9kLUknC1GoWGHz7t0nRc8DiAx4dLHN5gzIaGR5WJ6L5vmxGtYO1bLY9qp6D/JZw9femeLZqtSrbJqKSzM2sAAAAKkBABCcfc2pCz9DJ3DYMD5H5Jb9nPuxyVvqKWVYBBhsw0R64SKo/sPQCGM2aPE8rLSDQD+iPRA/+ozEsNNnqeT99ZE3iChlhIIhuttqdJTcBig2lCXZi7ziaNUX4cQafa3wzZMggx//5XG9bwVnrc0T9jEMBk4jDFz8reuXOTbJj1v4qx+ZWPhHc9HcNtz/CjorZjo2BnkmKZWhQPDCwq65IlU+ctc+xTPgAAAAegEAE0R9zakLPySY6/lLM2UkEUd5x5s6SQO/J1DXi4Yd35qxBQdJA/Jm5OY2/Adbma8aW/BFWsfx4nOEGnxgEDkGusVMZCwU7/ikHnTK3u/eTjRbVTn2v3xN+XqQXs4F3gvBLp+SZ0/iL80xUJNWHpXfHVYFWYriB9X8AAAC8EGbd0moQWyZTAhZ/wWt7BjZTWjo/ayFNzL7aigmz4cweV2enFEnbfsGTSvxrMf7Ar9v7Q1nt+CUWJZnKhNJLojRitL5sl0GaGfapbRtJM3YgNr0Td6EkIwcO9B1iliwAABzChEboMPrsc+IMzNG5KW3xDq8qQtgSYnhPwjaTDbdZjgEFjn6T3rHG2r7hkKWlW/052SHuEJ852OxYFyeaAHD2QBqn/piXZ6KiNT/KmiT96lO4OJqxV3EY5U30KQv23Jt0ZBI6/IZZokpOQPih+vzGXELAPntj5Po5wjJd5vSsQxsY/g1Csd93KGk4u5RXCDbF06e5HrBJQJkRECs1ME1ONQig0mVcRXXPj/e9jGNVdwelHxZzRr26hjx7pbn1K1Dnt931Slqv+y3LJzUHrueaQO6p/2I0xmvSMHopQDngKyq5JPx3hvy1MofSj7UwDI1TCcGDQg9QEsbdW0AksuEeQtMgqDQzq7HJW9IxwF/hIbF+u2JA9Qwpxgx52GNor9uWpDzjziVZs7Uc/aW1iBK77mdmNs4wPVwz/cq8WPX5adr0GlNtLXner1Uy5AmoVLUOPR9OH6hu8nPZ7uGLzJc6ZzYJnfH2XEBX4jnz81Xhm8A+pW8soD5IfZ8eTxWZDoLxJIUZtpE8HOW2Aj/B8K46X4xxcH2QF4ERXjibkmDhSZVNk62mFQX4R0kumSs4U/zoy++96vXOPW0uUNU6idkQiB0Z4TCe0m6Z6tGEK3zyuFaIz+MlDSudPs1+veTmAuopAdKguYflXMc7upuDa1XK0/BqcnU8HF10t5pIlQzDZB2e31bvUhmfH+EI2sEZJLsftx/N98mB8l7WbJBG1rk29MmOBoRruB3+m57m4ZAwUNhhz7zUOt5CyHDt+pyt1vQADNEuz9VB7u/uuN44ZWshq04JffrSXIa5SNPmO/Bb2weBx0k+NUMePgDtdeeFQLUX00CjCeO7QxPS8E73uwJdfXyhxKDdGLoNm/Pw3GBAAACR0EAqpt3SahBbJlMCFH/CHtQOO/KNS5lD+dpj6V8GJykOZwx8lFkf54z3Azi3K4u+TpeQA5vfbB1kuvm8IGoBaNstoAABj1UwwReURPvSlRtXh/DeXjQjrTLOb4B2XFgF7hkKu9DpNTqM+xvmOdkxXQIKQ15bxIlwJ6JcxtdgvTAzNpEeH9yiHWEztWgLIS9ux22bhkkrG8sMmzcLarrNgWm2Bu0whvS93jVY4YnIwcoIYATzPQTCp6/z33rV5vb8OhR6hyY285Y7xilHO5IIJC/n7QE84NA+OiUMScuMFOV2E1C51CtyR5r48dzsAxp7kBvSmqvquOafyjzLmPAnnOikKUJpBwy/ecKFLn+bvmhFUIVYtZ+Vdm35GbJOfYEUChBellNH9Hxsq2hqyFkSbgNOGUzIS3lQn27gfsZ+CnrbtpvkOM7uKGHh+WbSehteV8CoTGS65hWQZzcXM9T30PKv7Wo9E1oaipTX9xTQwM9AyvWnenwCDdhtYl5b6tZyxa+8t10IHHjgwDH4k/yDPFUhZhO5f7HUOGZcLjzk61W3QUTmmgfTtQ0GWr3SMpOosZ2VPsiSKKWrhRUi1HyiynqYWcoe9WKW1MET60Fh55setdHVqxyci0b8pnm8SjltuTBOXK+RbmcT2y6hTtsv7kLUjlEGpEeA6pZNbB2rbiIx0IagF4mKbK0aMtLPT09Y5YeFx/35YHfAV1AO9vqojMleCCpnEMq0elNFf6rfGf20nb2FRumzSZyDkaCfWflovSqXYZbduEAAAGwQQBVJt3SahBbJlMCFH8I57y9cXlGDIP58HzFuH8Tk4N/YMOV6yJv89jaJg8atkUAAAofkfb3QMu3NuAPevinoZ6vvYkorISaFPZTnPbWI+IU2Ojs75InNfljcz4GqwgbtBC8envJYGd77zpHed+XN7BLtcJ/1+YRI8J69pZVWO+WiuzqdZxutTbuCdBtTD+L8y7bact+2vxfP0i7FnaUs1/ouGJXrsZIkkZyn7OH0xORjxBD2c1VOqCtj73HE9z5Y4ei3ONJcbMviGK+7RKQH6VMwFP6snjdpTC6Y6je3+MbZ6SouYfykWYDu8KS2HxjKHwcAVTtKP+UmM+awkLXnMCLn9RHw4YUjCEE6Mz/Sivcp1PF/yJanzrYN81GvqKNKg4oS+k6AH7YeLsrvWlSM+VZLL13N5lJHGb3ht9DPyXPIhE+td1ynjR1jY0WDNq5cfiTWclASMTf3374CP/0HFclbmsHFyyNx2KcTbz1xCKkJJku+vKxfkAYDBqrVTNQ6HiDBVkTG+Vp9/ZxarYbUTM1TvP85TKzugB2XhICp10A33QFz5goph0yEcSdFtNNAAAE7UEAf6bd0moQWyZTAhR/CVPt674jmjiZC9vV8mBgKMbERQbjETYDwL2UEmvFFSAb+L9BUQiAaJfJj/q5skewxxbK+BM7f1S6Zfk88TDnVHNwt5QkANYoHzZ3bmebO4KKOJGQc7KOQGUxPaS7/brByUZQ7LmE8OS7B/vUlFQxQAcmqRJZX3GlVSer+Kj4QZqiGv/sXybjOAHIISL8X/WsG67ngch6k6SX4UyJgFSznuQla6PZ/wq47tFlYMm8iDhbljohGIVhSx2yY+NQfDaViLEXXzDFXFfF8xpDwPdY6rf+0iK9PIVomcrP804DdQpQdyoAkeAAA2ywq0Z2FLu11EqpzCewdSQxW3rS6QUaFSjWQxv2X0s3L5irSG9KILNrTZ+OXQiJ7RnEPR5Zk/s8CeNrst4exwvWaasYXy6j70LhtTSBHsq7ksOJ9RdaLrmPfzFf36JoJFyYi5SlfIAd8GIqZJJ+0Yvz2WPW746aVjqyCvogKtpygAiRiXvKi6GDpCaab+ODuriE/myvcvVeNaoxvhJUx1Odf4YKXNPT/sUF0AXu+jDXMLBvLGV9jHcWT67917Cj4j3rGhZDs2NJzmMqm6grvHqngrXQlhMOWWDZcFNLuraheRuodCokI7JoKlxnhYwZjfVC1GrDHtD3pgt1trI2r244QpqUn/SQY9m9hoffBRifwAeWu7v45bnfwDNEbCk++4psGSY5ys+8ckAt+YncOLAZdPUE1FOM/UPNuhE94DrlXiHFOxg6uaeitUizd8XbOAD3heSomdG76TDaNX5Z0+sEp/SSXh9Y9RF4tDe97Fv1yPHNjPIDzzdFl8itTNIFiKEbpbbz+L8fNfm4LbONxElW+rY1t8/qqPMyEuDzcnkzNWUkkegrIMg2nMBAsi1EKbDezeWI9KOlYOmfmLpVzv8bsbbFUTqzC2bNyNHn+vwp+MnDDYcre2UfEcsqSPJMlpGII4ASmKC9HIlbOscLKoZ0D/2z2MXNuDVFg7nkdPotrEHmS1lNI0WKiOC6NjcscDJ4sH9jmFhNZmLCPfl/WeVV2oYftdHatNaW/3AxX5ggld47W4ugvPyDQ9Jes69ef/+yo302AqesBTLJ/WTrWDdLCvZTq8wYpDXRlsRt2Gl6o50PZooCu2wz1yobDCmDhkTAARbl8obS52FokzImCM4touSbapHhpml7japwGligMBPWZ0LF44UynoU5kqlriR6Jn6BADiu2Fhuok67Lm+L4F9n3uzTbkpbLFIPiFO2hmNWSS0i2l9OcaR3A1s0bAfNJYJ4Tb3PpCXqFLvQANwm5dUeTE47eb5boxHSY+61/fkEtPJOI8H/p5qN0QLBNmBVD2Hu3ddPPJqP/rO1crVmwfBcqaNMDnVoEHkZ3U1CHDmR4CJve4+2IBda+qZo2YJP3Ho3APxztZi4ifqyFLiAJ4CCTpAAC30g5yyge4Q6FLWXGYVNKigHCPBG1wT5PXyko+GJ+QL6UXkUh2IOG3qku7dtNZ2LtbAl+ca/+/hg0ubMj2CH+w2j+2OmjlFoHYXwlDSe91j5oYklAJkK/4qhZw78WyTKwd8WnVfug4JxqJQdvUVUCKF/KRSzC7GGw/2FeZr8WvvglgtOF05mwa7xqrgXwFwANJGEGl3zOVTm+cZbY++BrCw9KSaMNneMpkGx3VFasVMEAAAH/QQAtMbd0moQWyZTAhR8Jp5ISjGVB/CCznP3JmM7M4n7/zAUf1DAJr8EMoojuabscm1v2ey9oRZV87TN1lbbkL8043qV5kx22Dd5xWwNcTt+CtbmFwM/uWlS2eXB55dejHRPiVvyMGWKRcaoXnuuRfhflbVv+lFzj0hJgSd2/0hj1BdEjEvHGDOMYZChARVYL6UE4q7n18Xo0IUjCeKZrwNHlfD8ECnE7Yi52gmHC9TQ8yLQ7/degVUleb9kW5aYssfHdFyvEqqL2nYAV5Xl3IfETnpHRaSDLll/+N3dshgYZbRN/+11WgJr6t//sJeLYaFmn8+FK3QBtRG4V4avgwCgG63Ob/NunLOGMloId8OX68vFIXQbPRH24BR2ZqxaYvk0ZHVBdp+MbA1I9ePpEvnDzvWpTfaLEncmnzfoy4iDIbeYGMd9ZDqDEVL6bkkFr9DWFM2+8ic//PIy/kq/o8qTH5QkXHPohiaLIWWM2nia6S9KlRWDdcOb0Ec+8SgYE0RNfFkmV6j7WkdS8zKW7sOewo7g83Lj9pEMJT90GWQYXVDQx7SsRaVcI0SwafDuktNG5nHIyM7oXQ1QAgK3SeFVKnOYn7y1Wv070nzVU9VgY0xK3HXwcqvdw1zFTv8S6fEL5AiE4X1OD1E5PrtrCMxbF/yXK+RF0kxHPUGOSgQAAAX5BADfRt3SahBbJlMCFHwkmt4F/c/9ut+LrXqg22A1OvotRkxpndppuAAADAmbzqOqj8MGyMVxz5Oc3IEDcEC79JA52JDF9K5ccAgDHPWcZS9lJfdKd1fjygWYwCqHQlbP6Rduf+L3noeuSqaAkDfZjxh14r0fyzsiHZq+nqyUKcZGoyMn7n4+0zkOxRYR1Qk5TaucKnG6nPR/3FrEFye+z3WdnsYKJcSpnO+ppnFCvbrCYjM1LX2ZXiXi03+FlWts4XnIOT0Bn3aF/brVUouZ9wBeQ4sJ0dHAokpqZNRE5LSY2gXAlc0VOhGakGyPAmCVmsJLbyHq+OMZpHdinPifgEYI0mJCoLhvnYIfH1gO6FUomtjft9G9iHuUzspAdF2So7gsCCg2erskok3/fqF7iCoUxba3AkizNBMVtBpMQZk0B/r+WS/BQaV4QUkHFZhXk/7W9yCdafMXH13bsMqR2gcAKf5X8jRqQYiYPS0CYOFc9yQ9gKKX4FK8gFcyLAAABnkEAEJxt3SahBbJlMCFH/wlL4ISUdKptS5zClabLuTcngvAABQmL0xMW2XCru22lAimKEH8GPA+PS4NScmEMgA/5vrkjLUgcbhO0aPgw7JpPmBURACMcnrnp24O8Rm78HaSSC+q3REfWCsBevWkT/Y8SqOWHvjmfweOtQOBNmv5Xqb6pQ/VyNA3t0gvlAZZeE5qj5bz02MvIMcGKqCZL/IRrbRkXgiBtx9ae0d/wkkr7wkfSrF6b17Q2ZOeJiUE7h5GdlSYkXeEPCT6fKVxK4KeNRhZzg4c+FN7QOSmvjJA31kJmLkO/vneT7KLKZ9J8d0jxn6Z2FryKn2RnB2WunaKpxY2I4Q81k54J36ly1ZYUaDnm19ETc+UTbmsjRsgwWSCgvLcc6oxhY4gDm1udEoPLmJuzik64AIV5aXkQsHOzn0RaThbxGgddUUoxmT/72zT8adCQqwVgiS/pgoDXZH4yYMobyZ1gwyKhd/MHP1LjN5WdT6LLxD8Exk+ymywsg9CZWBfHfP7EMtKEOjw06zFO90v+GZob4ao2+ZZWeQAAANNBABNEbd0moQWyZTAhR/8FWC99fgAOp1BKTr8F5K5/PSabnaU6GN/mQr8XaD4sHk2k+u02gulFv5K/2udRCWcIXt1KXZn8dSWXPUSLVIGCEM3e+ng57ME9lR4B7casTqihs2tdycJDWD4m6k1znlZEYLC/bwv3RmfQKA8CLCQtGRchNKrKcCxYeKxWvaXUPFYWNa9Lj6riWy2r+OO5Z1KDe7gX5gV8CYpifVvHxhhbRnAFVoRvBjdQy+NXdFBQjG4pquedGMOfhhf1qIEJOAGQBJ2hAAACW0GflUUVLCj/HRzqn0VHf5lha7ACWOSfCyjEZokcBLfN3TC4Nz7uxD8Th4PuVk/BRJIi48Aiakcus4jNEsnUhqQji/eKqm7muxx5Yg/txrPEDMFTBS0U3mjwALRDKk/Vaoc/UokwAxb23Ili+jA+a/TPzBwltQKBZx3AHawAQEVoRWa6uXyBv9SKo2U2nKpGPwDFnJc434sDVEJCohF0hU1NXuAy5Bty3EW2gqDI2qE+iOg2BHDAWUAvome4WdeY0ozkJQAJWBCN1MuJD6++HHoQzRz/7ze7mBRXGNyvZirDaB/k4Jcc9Yi+nksivxZmH7Z6BQR+Ui/tKLEzXYZq0QMLnxV37LfcswCBPIExxclar4oqfcI+Rm+/NEU3ogJBq11FRLfGkaDuLG+SwQzCighKmnLJlwUIT/NuFG9UslFl+JFLZZ5KZ5ut9pCUW2BFm2RBpj1i3NEgUK+wvKY6X9P4Nu2ImhDwVX2MhoerFgLumWbwqovtizD8dz8eSwfcnF4xPUrnPEAxpkgAyiUuIn/4k7wfgxItB+7n5LbqZHWvtWMZIuSY3jWucqUZId1iC1xY+qmpdVc30+J7m/iLWSBHOEEzASJdhSHMsUhEGGb2SmPSCNJp+ywkGiAiVuc1p8JnebTCzwebybsFDM2W/4hprWQREDDv9H4osjKpGgFGpz3EUiA3GyjbMLE8NTMmwjwbLlmuw/SrWL1ND4ClG9mVGk6622yj8gLBL/piPUS/nc7Ofe0vT05dw3wCDEM7OebuowzUOV6ORjkHnUHj1ry2uv90ghGrC1Zu4AAAAX1BAKqflUUVLCj/PGmoK7CDoFet/XhEOe16qKIz5eRxTWMELGDqOK4UlNOqB08ttqhsbNuTWv2M4zLILwM4L9KQ60RZh0EDmwePtoQkg/DbzBZRhGJl/p7ihxjqsIS4hUJRGvlpK7JKnEoEfp7RDU1mEkhLCDMkSESerQBDeBly4PExD0mdhnH3/Q9kI42BFWNvQfVlX1ScsLuHypl5AF8xahTIzXQMdT5y54Z7/B4grB3Z+0Tb9sZepa+dJ7lHusuTo6QoQDDtyB4W0FM07u7YglgXIGGMrBaX8Bmf+W+QZmvgN35NIpqY738/ktGCuMHc9sxdnCWRz6mP7L5n0069XBPg5UjWyu6ryhyucQG6Nb6ELPA+PAk/RZ6rSmdmpmvlocYNJOMna4fz255ug0LFI0ddhb/yfVHkRigtV9rjc2crwLzFg020jr3le+5XZMf/SlRd2AXKfgdnoyJEjpcACurPL32NmaMOmADoE9MEbED47vTBW7Lm+kMUIsgAAAD4QQBVJ+VRRUsJPzwJuIlcoWUNdWzYFYEPeAA7QgAgf1IbqZPMKSh5skZmb7UxthtAvJbqtqvKTQogjAB6scG0RLbTBEowVfAUM68s1Fch1UYR2//EEEH9CxQMfUgf6ja3PXGrPFx2/HGYdQvKL6ez3SrPBbvOkNi+N/wzbvfwc+0auT6j1Mjr+nGduQTJL9uoqXtGlY7ttvdF4VRgP4PuMrKgkanf4fBvFtp1eXLeG9a/2bHrzXlQWO+wi1xUWEoshIHhhsFQ4WVw908+w0weEX+ZQUasPLB7iEvJnlLiLrFetOrF3wakqiBpkw7Js4xWrYaCJUV28/AAAAINQQB/p+VRRUsJPz56bpbHHaYgxGE1lfrwicJ3xAbnQuekkDtqKHSu/8IHVtbk0YZ+b2E84h0b6Q3IEHQO1cq243zkGE6hYt3ZSPK5wslckXY3McfGKNduXW/3L8ktl3Uz4jwzQdjP9xgYAh3SR/dASUITEDsne0+xc8QcPUazgl57/efe0OJYM85Bl5zRJrP+v7xcqImAJPmDLQwID9xRbgOMhRYK3Z10MiibFhK0pkY/UykBVevNIB2UvA8ddMB8WGoLGThGvGNczRAEherpaYp1uLnZJxHlTYEFhp3JAZsK15Weiim0cynq7XRhQB/SBqYb/VmKBGlRd+bzMOov3DAjCnCG9+dBBs6vaBZCzKmo1D9OcaWt3p2BdGMNTNMmSM7sTAMcMXKk1LB8/ai6njTaZblmJqNQ+U7Ga5Dw0MaBcuRwS3Gj7e9YCKdAjv1Y4462cpskuZmQACWkZLGceYt9Qzi+0YdKzU/nUNwg7hUS8bo4kzsI4SW65+wfI9dyHIuFCQ60wYtSkZZMZeEk+rZvpvNZ+ovcLETAt/sYlnw6WZ5Nk51vQazIRGqbYUrUmxhpCsLr4j0Bcz/sRdhn3D67vJhDg6AvDOZOdcsbzFaSozwEHXgy2iWQTTNEMzRSQhfwHbxmu2otrobXoCUmi3uwwQQYZ8I2fe4aj+ZFfLFPzkzmuSrHT0wSRniAAAAAw0EALTH5VFFSwk/XtQwIb+dfJpdDme2ncqDxK1MbQtpu1tXDUjndaMAgtdPhhmYMNDSKFlSisJNM8QTr9gO758xsLomaJAsfbpth4bErrdZH0/zfRuAWUYRcqRZJEOOzzGsOgqBXoUpn5uxSAq5lWomBa2S1ya/Ht23qbtBOdfpJk20tF0+82nEBjgekRX5GvsMw37vcagN6z5DGA/T9u2gNe3B4yOggYOs4zt5ry8kK0ygq6wkVC4lMiAy/WPBSNfPJgAAAAN9BADfR+VRRUsJPg5rdeXGmm6y/qNZ5la1UAOXOd0wixggrF7A9m+954qAXJJiwEZ32Iod6mewOXEmuUCRMRE/STQ8biKazbCFwmcDS7CPzvt7Kf5XqnvCHOcea985x2928Di9CY2iHXWOYT8XRxRYtDZ4K02cChB38TWn9UZNRWe93jlwlpBAIG1RTs3nFy7LMAgKHcpCOaNs8aLM6V2i3cHyOFxVoMcnhVa/NO/QrbgZhi8KTuz/dR+n8oI03qAreBQeA2etqO+3LxWKn561ZpaK88iHi5Qj5rCcI6BlwAAAA1kEAEJx+VRRUsJP/Wv8VI2whX5VRnO41aYWXBW5fdPGJuhfGaLNX1yXs8C+owtjtX0zDi19RIXciGre5PRzdwfxvLFArAp3oTGzcXYPdWPvvIIUvX9WZVJ1diz/wYjzhWYDF7Tb6JBk7jQxbl+oOYbsfQEST5/iXJhaWqelm5wEyqoW9NTky6bWdI0cdnE6vLDUEnTSZVjBCjtvue/6Ak2Yra2hzbbRXJMZ/Rrh3j1EC+66XRyR+BJb0MJgr6nKqNl4FmDLiaYIRMNbTpDLoewhkbOxExSAAAACcQQATRH5VFFSwo/8lOK+/F6AB/HNGPdhxx8mQBs8C+njPquKofHzhcF9qO7TI3MmOXSg0UbpWL9js1Sn9sFa1JFYQFZdQx5WgyWsuC3mplp2OwN9iI9N6XmG+CIqiy+H5sI8kBLT46+7n8FDb9/bMA3hn4MVw/Wc6GS5Q/csMvYbtbB6/WmzU7slfBFApmfxq+gzdtijoDrKAxvy9AAABswGftmpCzx549bFpHs7PpFV5oV+zFwg5cJ2laJp/iKs1XH5L21mQQu3ZN+rB+adlznO4ZGSaAC1vYjqjIvx2dPnAm0glBxZbmfMuMVyuPNgQBWPQl8HkLSqIEk3a1tdv+J3BoZa35o8B7B4wpI2AWUUE0JyA8WCl2y0WrsFxpJDTCpcUoBdXH93SwD58QXAfPUd9adgI/KdSN39i35XONcR1KJQJsJHjwzWywEVW1R8x2BPrAMC24Hec/2ijW1sQGiArFKT15WK9ZK3okjB0RTW80rZpeFaLihijYQ3fIH00JS0J6wj0rVxDOx4cCv2eZmNGZc0MDdUlIodsqKPIAOqkbVh9YTOcAOlEfw61mrHgjqT3e92g10pyALKXeRZ0BX87KemNrKrcAYOXp++RAZh0XQYPuKnJH9B9Neh6TQdPnybVSr4pVicuJ0EZApSmQ9yY1yuPMTsIc6EAE/qla6mj2JLI0ZGc2V7eDHjiasCLgWfLmxfVHyT0AselLaabgO21ZsIz0DjUVoa+SBt/XDwTew7gMkkhr0OS71oCuE8MtrD7qHntNqRVt+z0kBRwMgAb1wAAAQ0BAKqftmpCjzqO/FPKAG+Uq4a78QP2ncUuM/WPRFK0RaY/v4pAZR9+s07B7ToQJ3OcgybchxMjzf35iTSAPHVg68CZGxDUreun+UbWDrrp0Bcf862emlql4avQpDD0zadVVxQ2tI+L+TE5EQey8ktQkTo5QXkgtlZgBK+SLu71wcVgrI3XyoFU8l0O0bpU2ZsQeKEtilsK4qucnQ8glVFL62SNZft9nmRdeH/NS7wOXsYsiKAQZPEzkhcbkKNuBkO9nB14DOdnsEjDhwWLz/Dh+FxDBCDkVL9D+PLzKjKUQTqvNuMJbvVFQYwTZHj8BDdPdxytFxvB3CM51HAUzXtjhbvinDdsqxz3YmvyqQAAAL0BAFUn7ZqQo/8+Mj0bDFR3n6cy/MwUSc2X4PKZFG6onrw6kyG58Ln1kc8j5bNYEuNBy288KChqWagNnIdMACLAN1I2EGEQDSBL0TxQcD5LPGjwQGFD63ZNOEGgW/Nt+7ehYqCHpRRRkN6QTagLj2yi0JxoxQp39sf9gtQg+rXZZ0Jp2F5zjJPKfB0yIfNGkVI7vgKljmWDKX/0Bn8Za0ox8XjZ1G3+GubLqurNNKIfyke0QeXZkydKcSllUQMAAAHTAQB/p+2akKP/QWNE0f3ZLrwg5oVuswnT/UwR11i3jhegyiWGNZq/mCR8i1aW28sDsb30u9Vv2+PQH0M1LcrLjEYq8l7YG6n8LYl7NQJYz29JYQxFL833J0Vr68djtjFuyXKDeT7XsChwDLvWRBQku1/ZSGxZ28opFv8s6FGtV+L1Pl0gwMcVf6+dWCEnwdlKqnwm4hrSNfKFl9prY94ji+JRoRwNVqj56zrLjXpJeA/iL2tFPjYLbyvLqJjBx73IvzEY3405J5HT2kYB/kp/EDun4RwMwh5ZFPQwA7HSt7Dx9caxdHYzRek3webRQdNdYtGkI/ID3Svnv9fxlQYvUdMZo8k4Va8T6OMC7+UqE3ubFk4OCL/DPWdH+t7k8YEAgmDVNxNlvLIfwHUiRz1H6pluJW7Jhu/4HrUXUiDl7ph3M9IHQR5mN+CsTxEJaixlrLyxjBXwasyXLSK4D881QMWJTAKMivl7WqR2VYUuciyHYcoe3BEqhwG8r/SvJ23n6TCuyiLJOOVyJzLwvfVCiNKkNhXU3RSaZjWSJjB/dmsuseNqsL8PyeDRmROXHIs8oPvSS6xf6hRWU4nX7uvsnxEMjhWQTg4Egs0ep9TdC4ZD0TEAAAEfAQAtMftmpCj/QeqtKUX7Fr2FevFK/LnADk8dX/bQdvw457jvxDk2uDVB3uas4ZXVnwxu49JTTbsxRNd4DocBsZ4eenKNpflC3rP+i1tRALVXGBMmASxzBula0V1Q9QGgSg85ZvDxsmLAisYOvgEx7fg1WqlM9rrZ2WAnti4bMdl2QM9hIDzWPgS5CB67I1vYQrne5yPKs3jEecS8Nn6t+USLRm1D4Ku1h37P3d0meuH2HXZzRimZalY9dWtKTqZ/4Rl9rgLAgxdS3EIypwfLzqLzRWsVJ0P6Gu+jRY3pjuPpx3XpEDI4HZoXu3PK6Kv1dPhl+UuLyI80IKLY5Pf3rEtgTGRbbJNbLEaC2B1CyjFOc1fjHVt7q7DFZWcR9KEAAACqAQA30ftmpCj/iQrS07B2q+yx52Zyxwxd2upys8JFyC+IKQkNxmqVee+MH6kuwqKRbhEloGasHP4SEhsihh2qdNh5fEgbLgOv9Gzt0OaCHbBv7nTbVdZrgpzgs7E+E/eMFcWsuqaw/OW5yhsIL+IinD6L+Js8SilPT2FyTdUc2O8P2gYo59ZyjAMcTS/gei/Ls9WaKxD9ouskz8jxkHvxzhpW9wGzZDui84EAAACCAQAQnH7ZqQo/PuD83n9J9PSHsUexjK1S2aX4kY9TogpDLF5uYvykyblWhBOpv2fQjFsuGuhcyCTdUVxrpwQZh0GFXs+Ps2KU3nGBXpcBJVR0znpUT17a3pC/iGgPYUQr3/5AoKDQLnguGgrCYhb4hbfjhzA6DuCQTgxKWvlQsBf9pQAAAHsBABNEftmpCj8lv9ex1X0xaesLnwaSlD+RWlA5nL0TnvAIEsPgHCu6/DJnteLAuYVoj+NzaFoO4taTndmSsJCMqEX6M7TyXmEH3uImCsAQ5evFRGBogkDP4rz62zRPF8zPD1SAACHMmvP11RGsD3m3KAmZ0w0LRvSyHpEAAARDbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAAyAAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAA210cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAAyAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAABUgAAAIEAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAMgAAAEAAABAAAAAALlbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAA8AAAAMABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAACkG1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAlBzdGJsAAAAsHN0c2QAAAAAAAAAAQAAAKBhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAABUgCBABIAAAASAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGP//AAAANmF2Y0MBZAAf/+EAGmdkAB+s2UBVBD5Z4QAAAwABAAADADwPGDGWAQAFaOvssiz9+PgAAAAAFGJ0cnQAAAAAAA+gAAAKyMgAAAAYc3R0cwAAAAAAAAABAAAAGAAAAgAAAAAUc3RzcwAAAAAAAAABAAAAAQAAAMhjdHRzAAAAAAAAABcAAAABAAAEAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAAYAAAAAQAAAHRzdHN6AAAAAAAAAAAAAAAYAAA9iAAABl8AAAMzAAACawAAAisAAAtjAAAEmAAAAkMAAALuAAARAAAAB5MAAAM1AAAFJwAAGoYAAA2cAAAIdwAAB5AAABfNAAAODQAAB90AAAhAAAAR4gAAChEAAAg2AAAAFHN0Y28AAAAAAAAAAQAAADAAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAATGF2ZjYwLjE2LjEwMA==" type="video/mp4">
     Your browser does not support the video tag.
     </video>



Interactive inference
---------------------

`back to top ⬆️ <#table-of-contents>`__

.. code:: ipython3

    def generate(
        img,
        pose_vid,
        seed,
        guidance_scale,
        num_inference_steps,
        _=gr.Progress(track_tqdm=True),
    ):
        generator = torch.Generator().manual_seed(seed)
        pose_list = read_frames(pose_vid)[:VIDEO_LENGTH]
        video = pipe(
            img,
            pose_list,
            width=WIDTH,
            height=HEIGHT,
            video_length=VIDEO_LENGTH,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        new_h, new_w = video.shape[-2:]
        pose_transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
        pose_tensor_list = []
        for pose_image_pil in pose_list:
            pose_tensor_list.append(pose_transform(pose_image_pil))
    
        ref_image_tensor = pose_transform(img)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=VIDEO_LENGTH)
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)
        video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
    
        save_dir = Path("./output/gradio")
        save_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        out_path = save_dir / f"{date_str}T{time_str}.mp4"
        save_videos_grid(
            video,
            str(out_path),
            n_rows=3,
            fps=12,
        )
        return out_path
    
    
    demo = gr.Interface(
        generate,
        [
            gr.Image(label="Reference Image", type="pil"),
            gr.Video(label="Pose video"),
            gr.Slider(
                label="Seed",
                value=42,
                minimum=np.iinfo(np.int32).min,
                maximum=np.iinfo(np.int32).max,
            ),
            gr.Slider(label="Guidance scale", value=3.5, minimum=1.1, maximum=10),
            gr.Slider(label="Number of inference steps", value=30, minimum=15, maximum=100),
        ],
        "video",
        examples=[
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-2.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4",
            ],
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-10.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-1_kps.mp4",
            ],
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-11.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-1_kps.mp4",
            ],
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-3.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4",
            ],
            [
                "Moore-AnimateAnyone/configs/inference/ref_images/anyone-5.png",
                "Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4",
            ],
        ],
        allow_flagging="never",
    )
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/"


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.







