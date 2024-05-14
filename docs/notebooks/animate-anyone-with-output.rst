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

-  `Prerequisites <#Prerequisites>`__
-  `Prepare base model <#Prepare-base-model>`__
-  `Prepare image encoder <#Prepare-image-encoder>`__
-  `Download weights <#Download-weights>`__
-  `Initialize models <#Initialize-models>`__
-  `Load pretrained weights <#Load-pretrained-weights>`__
-  `Convert model to OpenVINO IR <#Convert-model-to-OpenVINO-IR>`__

   -  `VAE <#VAE>`__
   -  `Reference UNet <#Reference-UNet>`__
   -  `Denoising UNet <#Denoising-UNet>`__
   -  `Pose Guider <#Pose-Guider>`__
   -  `Image Encoder <#Image-Encoder>`__

-  `Inference <#Inference>`__
-  `Video post-processing <#Video-post-processing>`__
-  `Interactive inference <#Interactive-inference>`__

.. |image0| image:: ./animate-anyone.gif

Prerequisites
-------------

`back to top ⬆️ <#Table-of-contents:>`__

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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-679/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-679/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-679/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
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

`back to top ⬆️ <#Table-of-contents:>`__

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



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/3.44G [00:00<?, ?B/s]


Prepare image encoder
---------------------

`back to top ⬆️ <#Table-of-contents:>`__

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



.. parsed-literal::

    image_encoder/config.json:   0%|          | 0.00/703 [00:00<?, ?B/s]



.. parsed-literal::

    pytorch_model.bin:   0%|          | 0.00/1.22G [00:00<?, ?B/s]


Download weights
----------------

`back to top ⬆️ <#Table-of-contents:>`__

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

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



.. parsed-literal::

    config.json:   0%|          | 0.00/547 [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.safetensors:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



.. parsed-literal::

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]



.. parsed-literal::

    reference_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]


.. code:: ipython3

    config = OmegaConf.load("Moore-AnimateAnyone/configs/prompts/animation.yaml")
    infer_config = OmegaConf.load("Moore-AnimateAnyone/" + config.inference_config)

Initialize models
-----------------

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__ The pose sequence is initially
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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

Denoising UNet is the main part of all diffusion pipelines. This model
consumes the majority of memory, so we need to reduce its size as much
as possible.

Here we make all shapes static meaning that the size of the video will
be constant.

Also, we use the ``ref_features`` input with the same tensor shapes as
output of `Reference UNet <#Reference-UNet>`__ model on the previous
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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-679/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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

`back to top ⬆️ <#Table-of-contents:>`__

We inherit from the original pipeline modifying the calls to our models
to match OpenVINO format.

.. code:: ipython3

    core = ov.Core()

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABFO9tZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAbDZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VHVyOeUXlJDlWeictY3pOae7n4aVTLK/rPFRox5zvh4tw2+vGHyL7fJJ5ah4GqTU73IuSNB89oSZ/+jM4yCZl6bDIkVaMzFHH7tyjnjPnPkpQZduUirUEXT9EMl/3+7uy8rS/OdiQoxXvEaEqJbfv8JRecAiWKHw0pi7YmwZGL6EH3bVlE4TFkYrDTYjQl562K4CWEAJwfqEm6zYe+S55c5+YQSCkkti4oGfb+1cJ7bkUEFMHI+CecAgh83rKvsM5i6iqK2AAA0CAZhzE/Q1s5I/5nzpwVi74w+8eC+xDyU2Nn/X0xaQTlC8xQlwNTmh0Z4/rTgp3MmIEGn8Mn+zygU+JSA9JVJvILw2p2l/q/T1UWsEJ8H6yAslG0M59RbrymWXAVAJNnjPfkrAZumXycRGU28Rn1AJ9WycPEoqjikPCwHa/mxB0zTvq8b1FBwj3nCvMlnJoij76EepQsmGdOqlAQCiXjbuXpzwZjlUAMjKK7Q4IKymX/6XdaXX4Q9w5uwe18usoY5W1SNGsJJn53ILlh5wTjfrJczZQ812goaEYtV9mO5v5DAw0V6nC8N3otdARBQQvHcpoh4/UNskaSy8OaNqlmd6YsT3GNnySqnBynmdc2FIJWuZ3aNpGmCRWnBZuxsxMmb1VTb09NUZvojDCiRCcxfopovfltHV4qOcYW4hVtKQw+jiDRfrxrcz/sbLxAl/rCLz+vJVGszGNsnjW5W03jn/h2dYN0VzTsLWXKtAyCJnrSrNRehU6i8QenOUb+GNQ8xAHcEjS7r+O5zggd2ZEZaKQ2+7rmwvrt/ZB+9Lx0PKf2ZWoMeON0giza4ypY71mHMBwn5PR6PYHnmGrnBS7yqRumsb9D4Pg8R7xIYRjfrRGmw/ieA0sLPBD693ER0px+hiVI7cHzo2rZyti8LG9DOXDeO5hYbxBk+4V4vKLL7G+vqRfiR3WmDI8PgtPAfqy0yRD+ZGpSAAB7LELwwBgEahoHAPKncXph1qvoQ0azzTWI3LMmALjbq8abPRmWFhWYuuYa3IjVoAUkFAxEBqbr2Z946U3kx7Gu/4d6sXF5iGKEKliPRB9M/vNtF63y5jFC/Wq20iD8Utuay5hVY82emkdLiiTx0wgj2jB38Cmi/L9vhW3xJiSft36Ciw3TZkZRWmWCCJREiRtgwY51og5ijSq6nN/4URzX8FdG5EWn7M+InO29TWGwlTSdC45EfuM4tCijr2/D6+hobLgABh/JOZODqhiRdrlGQLJcf5PR47aEthiAa/83D21oI54zO0BWLEnlDSnLLZmTvhnNhQYTNrH6QY9SHR/BX/cow8OFJmArLl5Zwuqb+g4ejP81C7rKDob69cYpSSfucbi2v9qShHSoqGkmbe5e0jK9Bi48CmSQjK3+T/Ua2lpjoI1cF13RoNNmz/LP9RrLxHM5JmkNLKgAXyb6nDr0GPTVhuu+MU+/P1x7r62N+YvP6Qm1kjzlaUrAVqH0OHv+IOuIDpCZ12Mup1Jcdqmg62S66vWML242jL0ajLMvW0YT8S7yrCQAXnH7eluAGxSvcSwfxxs/ZEitPuxwfGAzNiiloBNtJlUujALgKvkATUGpWN819witcLTyjJb+o/4CBHvILomZKQlkaYzM5qCaHVHkWpAPkZdnQp12JxQl8IC7ueBABeE3YKmMGOsSl9pNuyzWjEErpKhiVjRb8sLb+cm55kmNflKytB8Zn0anKAJw3f2p9lNu6L5yG8+EaGc10jmNZpcNFyJ5POpvGXcNsm+OLqyuLNh2so0deIxu4AYZ5F3IaS/TiwbhBPGEZi61OeLFaMY4gXA+VWA3brUUHaqU/4+QAkQzT2J1ZzOOHs9hB+n/ZIfJ1hXBSqzggpw6C0/0j527ZIVn7/a/8JMuxUqZk4c5FnfwVTSYP6Xbyha8kfzZt6tWiaCqMTdmTdnl34VrpnLNiYTdM35iLHLVVqpcGZZ7EBQAAL0iN1Fg0Sv9SlNu4IsMIl2vzRvvpPuXm2pr3FbxnNeis5CKfDbWXKwlrEgMLM1oXWmVLJfsER/qPe0SUJxkrvqADtqCL7CtSLvM5mn618JRHcS5u7iLmxOgZgyM1d04u6OzCFgQFr3x0/5/Vr7FCT1CspJf4IL0GXHqdwTTw8PV1E1lRDnmeQn8az8cXVeoKGe7GFXQMaMM7P7u3V7k8kSBXG45EIHQnbAAAJuWUAqoiEAHP/0WniT0ff/pqKZNrqAJiBE2BA67WO2MIGUnMY65AW9opiQAT087GlQIQsFBEjM7C9OqePs1sgzVUAcstIafo8wluf1sStcA2wgGhwpct2quu2EjwJztT3STEVlYQuCA6sQLHy0VR9IzXbToJHIntr29guRA/uqLZGaN6MO0wK0gajtTzCXfNmC4fApZxnNewCtzI/wnFxN/w6y1D3vUNMf9WW3Xv+JX+y4BifiSwRFNwISwRhzl+OAXl/DTIty86A8e8bblM/ggPy+szw+pl6mMmYwmLvRSULgAg5aa+n2UNeevVf9gjT3+CHm1BZAiVylmuHQZEj15njFrfMechxk4OpVCk4Y5p3ey+CbZYtfkTK4Q1dq+GCiS99NdDsoMA3x58/D0YNxRAoCcxAV7XBXH+Z/Mt+3BPxe9rVEPpa2pBKSNMTRY8RhjXczByivjK8p/PnTPaeyCLR/UbMgMokQF1N4c2YoYHX9NEnAIHXwam4FGcs+FhMVEZrhbevq//FwKfUAh2RMOAJq3Bt0CDFrL/GHqXuJI3eM/2BR08oWRVxq+sLIYWi9Ra+EPZVPirLDSTC2nxy2zHu/KavOOjjMJfYUti4zmYpPGGOwt83Xo0EowVD/c9y0AVBOzyD91gAH+yLjIhT/PLR4XhHOzfyeyBtMi4spzjvMxHsaTZlojHc8oWrP2XkzMV6rjYECdowkD9NeB8YizLvU80BL2udWr65+TPRc8R1Yu2b1gAHiI1BRRjW1IAiccDRQZP+AtuibGPeNNTpSNGFl3CTD14TY0B3BNTkIxvrclPlzAP3wGs4+HTtYuP2o69I0gQaFs+ME1E7kpat7gnNm2TgJFGn4f3gYPJVWV2MYVrcWB7nqD75ONb7W+Wrjvdm8MrW3ITwOecUzEWETCtOug/RpIl3MXS70VMHrcztq7zNxwapCQuHBAk/vWhM1/UfHOxByBM3v+MvbaeQltj5Cz0GBGLZW2qXi4+Cb/q2LK8Y4aN5Jvu+xxcY94EaDvL8ps2IKT13sr1c870HhBR2+UhpQkw6yB/bCgK+HLEg0y04vyWjWXZ6o7GFCYkoXaRs5pexsdTV1Cq9SYGqC0a2UgpdoZfrtWxATbkgc0evRB6tRokHTeMR+6nWC2Mok7xgjxHBSk9WM/dtVWu7ngWiTMrGIAfe7Vrl2PMNcAjLSwJgqTqz94m659RQE+n2RFDj3AmlNO4Yh5SeQrz8iNQLuihiUr9ry1HBLQNF08of3i7fcunmJlckQZt9YntlCVJmqm/x6fg+0AZ2LMhDjfbmk3g7JLmgWb1W9DHVlbortar9gfRwdtN/87C/Oyf4RQse2WWGgZ8NH8OqfTW4NiahUjIAwNLI/el/kgGPIXv6CwoVr3WD4TGomNVLBLPkNvqohz/5n3VHFvFwTFk2cEwfzQniAs3xci+NAbyqzC61/P3YkY8N7YJZmZeWQWmTlHW6le/pwYCgBRuvoxoLZ54Fp9MeWzXFXFZkJl/IifcpWlMjfw6f47iS/lXH5xDpSAL6gR5bWGB7Z/hl0CSk9TaC8qF9GiX3tSMsJECOFEWN3S4OOg9kEp6tEvhedAuH87gNUxhbwpjUyIlWae1NGznyZlb1ZIMRxVZzkPqGRO2WEwN0BK6gLdZCNbFxWCh/lFbeZRc5HGFPNwltNAOSvHmTAM8mGBrI+dG2qqxMCmTLthXa2XPn1ypWeNet5W/AeB916eIoGv933Upmg0pLfMg8D68ziNhfWFedPUAT0GqyHY0ZmlCTAj1fmxMyf4VntlAQuy/YkK7lxOT7Sx3j+xQVtuwQ1bE5AlP0cfCGDVnHx8/6Eqc3yCqaRLBtBViU5lB6plgXTqpuUJMrV6iFxUWyDyJeA9ambHPEabHISZhfJnjujxJi7wtPzU3fFcJbNVzXPOZtYP0tbE/0kfSPOon17N37h9aaS/ykNJ/FusMAk+R5WOG0lyfFrQvd98QkLOLOwPFMIAGDGckFgCr+FnM0W3Fso62/AKqPgMI2L+yGVzrG1GOE2symyV49RQY3zaXgWBrRLyr3vh2pEQCXiMa+aeu8Ox5xroga1UUo+7a2AN9LG8Txhw/gtkMyVHz1XvxhVn3qj7iEBL8LlsJWNJHFCjPKPyiPXKK+Ds9lztCb4mEvgUQaUqD8xPyXt7SXsJv2EsEDB6LUtP2h1BaARwAx/IOQee32PPjBRh1CXhfROE+3MY4/1zRvGV8Y6Y3tsJpYGiAzDu2EPKdJozCYePiw43bJZANOM+EWPX/Sir2ZFzzZSkueOiRyV/xPYhTZznHXw3Se/T7PV9MUV/TX4faaARNMwrmKGqQxkjxtdPpL0GJH2+7QJANh4hzMgmYF90C7egxfD2r2WWkO9IIXHlFBBSp76TXetNXPvnCMiUPON/l7r49GnEOxZMe5x6wwO3Ow6j38s+dPH9dkQepjmdeTbCk40nhJLWOV3tEz7C6Br08WXgM9M1Sc20L1yj97wYYamHK2khXF6fgCZY5otsN9ne1vnzoj8BWlya2ZSD5YqCRgXbWa3D6zfIze6x4Ds9EuyGv/vLPfAPncewkEV+9HbgjW/JE1nggkUvyxot0FRc2tSMdJZA8mXzsI5P+7MYt9CtCrC0mYs9VMFky2xaE2aoD6ZF2F/u+H8GAgE+80cngydsKcBV0qGyIeaSLa08mPZP1n29DcqvI9giYnbwoE+UzQJhxjBvgZoktdhUDCnTwHer8p30EbW+BCWB7p7e3PZQhvi2AP+U3gem8E+z/Jt1fnrtantOB5gBGFcCZkfMTrfwEH4Q23yhuVcvL9MECfwAAq5cF5/+y8t0zrHswqTEsubzgH2qYibFzogwGvGAl4Zy8+6H2ePp0R4L4VGbbBzOVlMZxGYbs8MmPUD2ZcsTlw1iyACMCGqr3GM/+mDr4baqLo9auOo66GmYgmb67tSpXBeCSbiICOLe9GLK18G02m4v0Ngtr/6G3QnbaV7q9rplqG1mmSPGSIEntzyCDbfLFIOS97B7S0Fy4EH/j5am3DhrtZbJVCYMwvo8W7rATbcEEh1KDYPIORSeMcwF2o4aaI5Fsn4wxGnONpoLzK+210gTHoEunWdM/obFz1Q8LFkX+HnAl78eJzGuaU4j8RsVSw6J0NAmQ4VUA5Q48G2XupCA6tgXFfStJ444CLyj9WJvsXiOPclo4C2+n1VhQb1trzQ17u158HOdP+anD0dZYsfgL8/jBgKjzF9wUw1oDRCJviT28ya/J6aNdwUXuw31xLYZajrprt4HTC06jNLBYN8Kd/x2rtPDkxS9Xfr2XjDaDAHID5AAAK/mUAVSIhAAo/1popaxQvxlenRCgAnKuD5slVY+5FozMZKuLmQ/HScn5WS0cp3pt7p2R5XC5eWTjBDhhj1KxMFJezAhdjh0kqxOuVdN4jLYK075znd9XBsGphblnxWHf0XgZniIlrEdiUg1JcQCSiXDdGSVG9QMq4MeixRutenTwQs0BxcMc7UptRef97ZLQeToIutvD+iGteKbymu2i6UvqoUDg2rnkwdfvYT3vSpBwegf7uHYfTEfBLm0M1inwES9YiFwxfXUEdj3/1OZ2kQkm1e2IxJKctnSNbNAfiHnb8fSoi/VhLeBI/LO0Jp7jUcV0pFjegMfRkJyaJE4hDTVgZ3iHMbiISf5atvycw6xp8EDmAKmKxQWfp7w12qTvhZuL0CDf9CHv+6qr5zpLsK262u89wJrpM/qWqJbzyh5KmfOf6iasPThf95dpJxz+q9gu9+4KWXQXOsbzCO53AO834LGkrPvWzDGSgAXpwdXH/LJ4wvuj3e2Y4vh8441ylm0gCUIyaQHpM4djM9CUDP+FjUOmwSka7DONUyNc/taOv2iLnM2ajLpR/wm+p2xNju16rY8DUo5wYw8m5lJxQOaDsBl1vPH7uv8WacWwmyk8ttIp0PUmmFJ6MFpsWY3YOcVoMBWvzxDLZTM4JWcsm6QoSK4XJAhMv9rtIJq9S65ueW+TsnHRpTXGCxgBvd8N8OAPhHNfkRqo9lmPbN40HlmU7PlL9mtAA/EMhhv3R0rtLmQjywVkIjfDQyjLN+8iIdRl7btf+s043SNR/Vfsk1xOrwC2YP/hqTct8HFBRwG1q3ShlnaufYMAmIBnldoRKEhvhah3j+2fKXn3p1cEEp9ePbF/GfsWcgWBwmDEjLe/SHc4SJayXc4emr0exWDJljrqKLk4B1Z8INqiEoKDFPYq/LFLdnuVE1uNRLID+p/w97KxLk7XVURq+nLUAfGTFtL8dCTyLDUvJb0jGFXBPQKriY8R2VXyRbag6lKZPGxEWgBQ9eT7uQr0yXH7ZomR4tado6CMuWK9bNZT6+rDg5jbjizED7eMg8FamP//vT/7uWzA7AkJWhCpOD4wQobxpNTd8xnPRcZS+QGGEbOOZmn/lDt0A7dVFlI5JUmhaYn7EPKsb7yeVpgRzyUcDhc62PnyFZUx2AvGoHPyraUapI40HaoJsnpURVCcbnHLYN72yfFisXgqBAEo+I/tKsZ1kqE8e/6ZtSeC91CQoendepJw4J/g7WgI9EJYA4zWKaupu6c636IvTq5/IFp1C+7x/7VXscXvLuGQ7iqfDAzpVjMHC/coPqfnjsfwGVR//Z7DT3Z9HP/OqFZ8mMCdXJJxNJWBAalIo47DjWk/JbtkZYfcJHKt2OKY21Z7pkvG2SfRlQIYuaBGxLiQkFN2k8svq1wN3sbIb8r+/t1dgElYL49PMLbsbB8AYUlcAAbAlsYqvLuMmilUkLR/yyl39GDAVFqbaanvrHmWx18XiZrkDrXPq6Q5gr2spHzO04It1jmzTOZKJLMmafDaxv7YQRmbgsyYb8qU8Oz3N2j9O0UxX/MRhl3mRr3poaPJeDw6OUjF+hc3WevxinVcR0WOO7SgmH9ZfFAvORITxmSyDEdpOpVWD1ANK8IdBhY3x8OK/K8iEv8ICRQVG6oZpBLkntwV6hdx/IxFkCgD/qxhNOkTRmnaTQnt54SlIBE7SyewgRTQ1LAKk1H501OBloF/czNQNawPIVGFphpriUqTyglUSP0hgYhl0GFR2+ghrJauB8XxuGHYsYome31fTia0dY+emChvG6YgX4b+Roavrei0I6lu8pXdpL8QmNiYxvOiQ7H8Yrpa0w3fRb3BlegQC1qzSc1kpHLeK6vtXLgmuR608BG/aKiLkU2IpYLzNQhMrEX/xc4PUnJw2IKcbqBTws83+b+SAHXj3YupKhWL0u0Rd7qPS23nzKL6mmUe++VDdqlNkCO77nwJsa+rlyXAd3d987/luECOT7Z2+9WY9lu7jKSj4CTvRK/74GOp4+4mcEw0PDX3e7t+YB6lTLFunhbJS3JTAOFNyBdipnHQ6wnCGCSurvqOmmDRNTTpW+avED/NT72CtftptklIBmcz2A7+exxbruBlPLBWmVYLxlQ+9bZnra35v7YO7/vGteXBBYTuJHfVMZar3il5rpU2IyTXLAddWZVHBTQtmKzGD7yuGg7GJiwwb7uAlqbO4EiwhX+ffsfYM9+MwomIwu469PYU3/K4aoV26hIqAoY6MCaRNHRRC0WPPg+RsQaXgdxezIZLz3fORq6OggmWJ4fK37GKVY0M/qnsZVD4gdl7Q/TwOjV8HARlq5ivw5+HT4oLSdGX2MyuZKYu4INO7wabQTYdGYDys2L+JVzddPdoKxv34SoIUcNLnoDraYXGklggQ6xkAxv8/9hk8L9buhVd7cHddPxq2suUwrCCaiYY9ZV2+4fX4uYwwjXjeQOKBy3UPDqnyNyBX5l3HcJqInyYian6djNe8QCJFT92pwadMFNqtky1fscGYelNWiNNPsnIV6Jj9ll9vjCkcn5b2Damg+gOm7FgtXeikSfM3xSaR4W+BtIcbd4zcQ7VPJ+6SNNTlGUQZCYbJEU7FPBCdFYggZl1exWOe1jKWnEIjJqrwXElTkfclXWlQ3G9scSh1UJfoFWiCzKuCSQBpIsfbTe7U1zmNGScvoCpbLNR18f9O2fZC2kYK7h7h5ewH1v2JSkn+dH7fd15TXDFXkEFNfD3BmavRtfC9qJVaNd8hy8XyetR/qoYYSd01dU+4gV9W+AkLSczWTn8hBfk1A8/stn4bteKR8tzrFiQASa/qn+mCfaBQdaPNsxkJKzi91HSs1BaniGF5LYFZ2Lvpn74mEEoK8MRNzYkcYUYgk7XBeKXLaj+/pvGxAU1DEAVbRziLyxz2BbVM/QfWQDA0lcQUNci3307s/16Xc2Ad50+isLpl1j/yEl3Toxfps4jA5UzDe5OjrbttWZ6RxEwR6fDwVmKsOxI9KF7LKtr4r/0ZLvtcS9OdVSX6BUVHrVchXt1qC20dheS/2pJv4txRPKR6NvgEmba7RL+xzFn/vA3GzU8CldKUwYHMNBPWX5vXLpQXR8/y1P3kWlh4z8OLBoq4morKpTp6x2OXAON7TL60liMRlF6Q3h5egeEutIvlRfdKYh++JC97gnqVyY6BYNK4uoJqmST5rwIdu0fwXKQeZYOAqoCS0q09+5VTxhsuAQLA/NRozqSh3M4MJvqJp34OZsDdUx//kS6cuXcuGscn04ANtGIEBL4Lmwn0Lsj3fyzZK747XRhuFvfxbaJrIvlM3jF9qvMHK0XvkSI3kh06UKiUf0dv61s5TWhqoizdM/gsa0K6JMPPMKH1fWdDc78yu6BXrxTDbufem1JYqPC6W9pwjKUEb94BK2d3viY/YI6kfkRWGiuxIYBBhwq60056xOo8MhMZLLzU8Zq3sfmN4WmAbqW5wHVL+Xr8ghW8PnTpkv6Hd6JoOXF7Ge22kGLZ+cvpK8RrTa9tkvbeRYPT8UWDr/LaNMYyKYgWsIiFYDnDb1GIr6O3w5k/vVJQSXOGPYmfB/BrcKYs2kFp5kDZY9xwlFGm/uRLOax4c+CKtH++eaD3JmOivmMTcPPZ45yF16532gMdRAR9bc7LBhVPmgTrq6db3skk8FgrZ9l3aXS6rWl2RqPsYI9aRKVzjb56xyKn7ub9LMiqXn8c2HnDrNJbyv5lnf0SBcWJ2fiHCU2MpJQJPQAACdFlAH+iIQAKP9aaKWsUL8ZXmpgcACGOqwl7r5okQfQ+D0kmihhn35duf08YCyoxfGHmzuLc8bUDrPtMdT1053y53z14BvOqlApmTqvh/7owPfvq3ho0BetIPtDJ7fGLL+gl+KabkhHnBt64+Duwbg1VY0KaMTDo1mC91/E29yoHIZf+IwMbN3y4W+Ikd6oL+UsQZnVOK23xyAtLVa3+tMiGJT0qHnNRSzpkhF3cljTUMMgKY/CvwF32gjJmyyJt2B/J9XwnEuhRMXfJvd+AOhz1yQTkfF9TB7dghwuyh5pxfaZ+J6m1jvAYyCUoze+CuFsGFPinGntJqtyD0oYTULPUdsDZFDBtYrARQwoup7gAJICO2Hk83mlbTCYPTixa9kQ0O82FKisWu/nFdHIwsV/ZCiAEfEDNhEPuMTahB/NJmeJlorRXi67ZtGp7dsiP188PGNeF4QarL/NVwm5pfjb/AP5Ku2s4c553O1+ugboIGRu8FQ5VJHhfGVWUghLN8Eg6oXDr3FkYvjCRsr5+FkFZIFEfrvquLXTBnxD5DKpLPz/K1fK8/0BMAZjqDGtG+rZD8kUEPemMD1ihlH8KhlWebQZeaZA+uF4ZrdLZ8eSkMkiMnKXLp5wuztv0AgCx/yvKLxSzbLxhA1EfVem7i7qiMf9gIAY7kExSbmwAEYwRyHW4fAEvFjcO14uqET3q8Jy3oaqZ1G4RIaCSD+LirMxL/BVhIlrKdrab0fbj/cU5Vedyydfi/7HRO/crkrEmpyqhjKwjx4CShqjlXX+wY3OihYYZ5TBYgb86eyC5p686KokdyAftsw3jpY3IQbx4OQEa74WZGkWwJTdjTXvhdNSG00aZfF4/TkA9qR5i1ncR1r9oEK8nC+G7GuAtxdfcQYLg7ExawAwJMOCRIXUy6s6jC3y615yEKnvzN7PGaleoiU4G4OyZo7YdI6K5k317pNaRtU7Ocemei5OJ/WBDxjJz0omm2V4M7Xy+EX/+F0UfgSYPak0LZqpMj7HnrBmGgltuGxN1CRX8LoO4XinCrgVXUpGus9ZMLjIOO5j4bz6gHG2f2mJr33WWS/nbnSk2ZM+TcNbrqVNAjEFcWwB5vvy9u7cx+q649pZMIlwcB2vzxHsLjsGs9KyJJ6tqZutHbwdOMc6Zlr5rN+TM9SADFPAOVSuV8mX8SaiySnComicUJ3NJibf1gdf/oFUKMJaeuM/Z17aN8mQ6q2tAgFwUI8Zbzhw9vrXSgr3FMbSZoYjmt4gYaovAxSlkvWENDXt0PXGUnZUbr6rMgu1JtEljoL5oDMD2rkI1SmN5ObMaatpuSfhDU6v9ONFhzn4KltgSCBxdlx+N7y66/Jkqi1UpBEV2VJ1TUwHmEzlRpHl/i5D+1NE4NgWqdo5aPirctY5LmZdm2oGVx9IH4FuVwJFHTOw/vlAvOYTQKhdAUUflqNSdzeTbez5rBQVOian4gx05+cAHCIl+seRFPl38FUa/L6X6ejbonBWzP4xlOs9GOvR/2wmz+7FLKb720b7ONGjskiwIyzluJlRheYV8+gS3Cj6cazqbZWFlJ5n5Q/C8QS86EPFwMSt0Rz/FXxFqBTACOmw9EBY3ZGX7uOuxRM/Y0r6h3BMzKBeFFGvphW3lkd8sJoaIopw/0UrCgClbmQCPU1CLYpTeNeinRBsjUdDGv2DHTAp9/XGkBH2vIRX0koSHYp14oSGNEgXv+Xqic5krpKZg/Ej2oamPoMImbN4EVanTqXYgcoDG7sOYCSjU8uvH5Y3uqcq8N1UB2OmJh3MoV0s49toVMeNkjg93ZWK62racC6N/n/p3vtW1ms9jkNQqt+jhZ+xRG0PFmFBvwBQp7TJU02pWMlyRNYQGnKFZHxXRgw9VmmKk47u5sw1n8qyzkPOtr/3BwPWBe0p3Ft0N5ALu2ZL/vUOGYBkDBpaNyF44MAbp/2hWRwi/vE1T+TpF/WqIzfq7tgIZTPkFbNf0MNfxE+65AAfUgF34mE/F9OOaWEF6zZrJl2SnoI1FzikO13bnHm/9fieF3KGxKyfSgJp/f8HbWn5l8Ch6sFxXWGG2FxAmmrwhvEbqRge1MGnBtd9wHkRLnbTu5sxMtmaQWRXPM5JWAA/7e+E23VsL3NxkJAFynOEhBaWJWsc7y1qoDvm9156YDHlDsdGpuFoTdTnVd1Dr+REN860gWnM6f2fuJrpxZAKWZhxtfqKhuHR9CEUV271A31tPJKSe/JwEW1ja0wg1BGXVuNJTouJdj4nSnqvJMMrHMQElTIHEStH9RnRI1XjlpZKFLgyMtqLsgGYx5uxlaxaQ2BM/a+YCzVPgXGW/RADXF6Y28bw4sc9gA9GwBqGcbuIufpxxeYTIRAu1Ly3KfMeGFkXsRsMlvfvo0cX0PUAIE6EMw2mgWe0/LAmDgYy6Jk1Z2pkrZRfyRZQ/SsQNBaZnHyhMH0gpyIQcvE4MQrsj6NrvOlgKBvuHHACF7AtWJE3Nryrb5C/u4jP8cabYLCuzAVHneoCfbQSP/N4CnZk4NVRxQ+g2/80RAtp0sp4bUPloUlNNqytylKtKUC69kX7U6VR7DNvqkC/LIuJ6Wsh60UAeaGG1SyGCDr365zynD5TzA98U6l1QKlBKgqYIzNn3FDMCWitJrbuNXOlwV/3tp9JvmMzmqwUriW5Q5cIlkBsQvFSB1aR8+wE1BbJeaFGnZIWMs7h0BmW8v+L9Xi5+C0nMHlQrEKCUVcT8Brp5GOWlRNt8E7pGC9LfVpdpBYEqamj8qy/rJWAu1b1XOfQqmiZsfwkOGAkvC63rngqVZV+njqVsllxouzeQPi18nOiLa6d+N+28F7veOOlp4Y4Ii4VZvAYD5ugeS/xflaQ6zf3AxEnMdg/q4Aaq9ZiWKf8y6mIdFD7ZM0H/rqCxrHX4Um0WRg5eV3fQRi4WjNbHdNDwLobXRYX4ixJuurNLviR1MXz383Su+/Dt51Fg+syPKBL93mOIQ5crxQmkcmd/+4ujrmtNkegNHGtH+iS4y9i25wg+jPbH4LgJXqBmCTNa4DLzdjYyjiixd3BEEsAKGpV5JWFScKbLSkdzvG/jSUXuvWG/201OJOivnB6EvpNLWVHcUMY3SuddnQ5vqWtGff5DRIiO96r0QGjd+9G634Gbqtehx2UhZxWd3jlQPSaRUYIdSOVnYL7QcaAOEvyyj8p5GibME57QL0riKQF7k9XLoe8mrueteJdUA0XF7Bartf+TcG0XcHPa4FZdcX1LaqE5/sCyXhJfdscibSahxAmW2NUQL0FvH5GwN/0tatO7lm6dTudq5YcdL4eUVRc9JyBHk6CShRynPf6mIV01jsCqYms38rdPZURW4QAABz5lAC0wiEACj9aaKWsUL8ZXlkOs4Av/kcA5e8WCjH7t/cficetssv2ixu5xCm6RYRknMhHGH4tNOo27GBUwmNpo026sNScN3GGsSoeapmeEWZ67IKz7Y0g50glMgx1BkSli5rX5oExX46GFxrUwoPeF2UEugXJIfCYd8er1gOH4YqMZE21iz8tI959BsgPZRbhpUzDJC/BDssbdygSFNEFTfhbg2OkWJ0e+jS/89nO8skcLYBA/asdy1YKvM4aFgNZu8woqzf2XqtsUgnWNbD603qG2Az9ZXH6QhRsLpj9XpeF4psAEBG4BIquVbMebD6Egu/bwHOkQba2vWjTq4yVNE2UzGvyW1XILGh6mKB67eLmEkT1dUWKhzVWkvEa+0yAPNiRD2Mz248esxdph53NnIZhLDG1qgAlYk67wtaPyraXCiHyR3eHJnSqWgWYv7Hc8nzJO0g2CSOhmONUKuo3Sr4+g+o6IUkdUorEIx1TTJfSchIi6/XLyUCPFojHIamOK6W5shKy8AKei3uHJYO7G7Fcohq/IJo9YvxRpic/BaiU2J0MnhhjdAD5OLaqSXVfHglnkg6vt6yT/yQUZtaX3f0KJhMcWVrb9498FQee4oNT/91nsif2UIb2Q1HA4ulRXyjahAGmoPa/Y+jCmk//F9au7lDFs22EpLig3CWV38xgWcG5B77XDAhxCOarRJxFaKV8vIMwKbX9BH62BqwRz6R7I2rBgXDie+oeuEEN0V12VjHMsqv4CuBXlL3Uz4k7W0h2fOcvWwJh7CSJBiybzEYfQuHYccMu14eAfWhF7qps7JLIcTwsWDrhVrgsWpMVsXieT0Fn8Vhd44A+ua+qfMfmKlrXKwKHHFnlr2Pbdc+Kt/fVM2EYN1O7fiKu7crXDiEFFzdqwSclk/SXaU7jwxXnrZQTj3rha44uU/XzF1OpXANkFjvPtCzGSM193JG+gA27cKPDN0kbQZPJl3WCST+iW9ePPFy1qqIJfUDbQDHZkichOZBuOUrnk8z0CiBUtDnN3Wh3jRbnhiwPCAJ5Zcyi+6H+2vG2hzDCo/aNx/x3UnsIsMsgpRrWKqOMRsAwblJGYp4m+VBLoDG4jdElsqPEir8ZmtxLGRCbNSsrVcUB/1c9e6UIVzlpJYJsnrVenioB4fTxz30p4FwrZdu66hY1EptvoH5ndO50XdtqAn4tBRzNpPVx3KfopY8j7gThmNkn2Y34A3y69DH14W9v/5EqGKkz6UapW2C+qk9lZaIXJ+Z9KoFrhly0cZ1uaSU85KyZa+BD1XMDUwC+cl3In+YQvAZsiue1eE5S1E/2lEldP1TzgpSBoU2iWR56KW+CrNlgashxrkf0cB2l/3b29XJ3I/ApzWHUIycy2rqPX3pGMJk72MhjHx1GE0V3cpqRActJaAQeIQOASfpwvZP+hkWF6HPhogav+W9iWWAhEZfbWnuhSyDqG0Dn6JBIewsmPKczfrK7hQtUQjg0yLC9yARVSd57IgUCE3ghiSrKQRUYqZ1PiQpAs0cFoOjg5AdBmSGn7lHxJ+jdnjAG8/r/2FUquItAIAfrAiHJd0rhD0npgS/mkvmqLvCQnVAGYSUI1yBRg99JiD1kLhIBbQibg+Pa1huEbFbUnkm24kGBHB1TTBo0zPz67X8zQ89FKiZWZz8qjTCt9+PCZZONymTVPRF3S4z6WeZlMojeeP+5O/YXctvuX2dKRfFDEbirC9vbVC7lOtn203w+Ta5Vxb5BIzEWFhb8a2gMRqN2aYnIYAzPhQw7nJ4HJB6CnH8y8cV/MucteFz/6LCMHslfTCU2Y0Eamc9kkpg/YjaZSxPGbzH/2yfSaLzkO71uPeKqUXqanAiFw1zDdJF4qviXYUg9Uc4Z36upaXhHA/Ymx+Ftolk0Xty4lIhQMfhGdQUHebbKcWpIuNCi4bAgchi3077i9ZgFh4ZoIN2LX9nSmJXE4jywckltJxoeW+pO1nJ5lbjEJQYYxH5taYg+T0dBTeeylBt3cANcvQvvfMS7RwNsHmoy5ziSdV4txMzjY7O8x2aKjksA6mFCZ+ToqrgEiH3sjHul/9NI/RRLJqlkKzjAnenAFNo7yEJTQ23JxQAaTHVgkYIFDLf7/kZKtYeEn0bFoEQ+9ojEtIayVp75ayCIDBk6boxNAQAJ7ezwcP3V9iO64gDUG2dxeItdsvttHRO9B19/ABimfyMRXofiWOtirzQPD67YMzz6+J3PD1dY9nuJPW5nPtG90udogAHnY1/viGjvoioVezqA30zKSA3XHfMZzt6AbGKw3Nwv8YI4BwMUsytI5Gi4Sq7MUvjqCWPxtHHhK+PoOzoa3h+Xs3wogq0aaZS5mbfSWX0TH1BL6BqQW5QPUEcg+eXhZziYwQ/UyNV1eS2Uoap+54x+KYrUQeTh6e24Awi1eS84g11bnlOt0s+PnLwyjsK9DsHr8HoiQkM7iHgMmjVz02JOgPDUAAAUBZQA30IhAAo/WmilrFC/GV6dEKADHpzEkf7dzeZuFVJaY5HNelkPWl0O3lLbmwtMguQMh0MBQgP1vn1gXOkUr3JpiHMlXgKTXx5fQ/hLCQufQz2wJrOra8aDmTkFMDKz0VHUGIVGXPC0CmC2aDbvibyXNLY9Oa2sh8Bc0zKuEYaJzMfpx/8ns9+OkMzrz7Pn5ax9tKb3iiZP6ozfLTTHK+KWm/nlX6/4ObHbcGIg3EMw/ezNKU/8VVg1TQaVuq8Hpn66tY+bI5UWHlCwnQBgwYO5wHMitQ/GTr85dlPI+Ci4KbuEyPPytnBnqcwAD+/uVXeo6YDpFpLcV7hxoHN2mXR3nadZwSKUQdnuB+LnXJry2Yna5n0Pqag80P4DIdiopUlgZ3LDk4EekNyrWSxrZV/R62GqMLDYEi01Uh7a9NEx0U9qs7/vvQZxyRJ8YtCNwUjYeABvBWOycOGoOZZ2Ma5yqN3ttIxMF4+X32rd8zIbyGt7rLzVKkdGMQEYC4Y1GjVNT1thuVp80FVNZu7TCP9Msd3WN47hFsqaNdB6lG4yRjw+rvzL5jgTAfmeIO93Y6wxzFCyTUh+9PQU0XJmgJh91eqt7cLbMrORE0diTjeKMDHiXVOO8FRwsV9ZH2YM1wiHndZ6FqAdeSsceLvB87j8w8bcS4AW9XvN+/HEmx67vR2hXxTXvB72mvYJhD+zYdVYqwbJPcxIuXEjEYT/xbf6GBX9NsCgoz595vA+BhT5pzvXSLeq7HQIKbyyZl4nTVYGH1qX8jmw+agrd9oQBwKD9uvzO5q4VkhU06yVcVbVDzwaXNY1PnYpDdF1aH8ELI4qX+U1Cd4l+g/aZMHdOtNvNdR1Z/sqQ/MZ2hhhGkmmsSAJvdB6ruYzuyjctOAas3MibuM4N9eLAe5X13D2ZmhIO1kQbAo44z+7E1zPKEKf+rLGWqISWEK75ciHwr2Uc6O4g53kXTSlV0IuzMlucMwANDkPbAlrR8PBIbYQtXd5/np/QMW933C/aheZ8LTq0dLCQjEnu618XuFugHfiMjRBPk1u5gQZvOdt6wh9sJkamLGOYmAXW8jGBwOBcn2aicA9F7pbZKAarewc7qODvYG5IXPEHRhH6Aw9oEKk+NLi+oV2lG9zmlOlbFJ5xqHXaWCZW2UBOpdKWqPOT20WCEtYkghVv0U2XpTJV4AFXqgNJX6dupWC2QwEve41tZCKIqv15yoac4SSZEMp7chb566M1Xx+PFoKXwCUmrPo4FPuLon6es0Iz6sPAzcRu7tKAxDMCsFVfISltkrCLzAgDDllg8GXeNvIYerPbfsrgz/LMMvGf8s/XtrwODS8xLw8L+MrAgmxExOBvslZba+10ETJcPxHf9dNxLU9eat0PgQo/qhbkIHKMyJoA7unSxnw0FCnvSotRIM+4AJIrFFfBfFIaFttpWsCoTE2A8lIJZ73QH/1fHYxPJ0OWFtaUPsFlGtrLv6+jES5KSFe37HW/DIm2dFA+DpLT8vQI+H8vOYGAxBggXC+fXNhZLUThJXwv3Wjg6kt4s1+2U3bTBeqVSuHIHBONfO8wGNzn27f8xIetCCkfaWi0dpgOBrMIq40Ws6CCzqW4tJCeO6zGCrAUzcqTEVrBMlAYOjAcr5lf1zlOUdtZLMxxfanY8esfvpkkKy3dfvRBs/1b10yt/vWBrCc7Tr4IUHvlEUvI2msdEYeLAAAGz2UAEJwiEAGP0LniUGYv/k0DYsdngAQs0ZhbUhMr1zEc7mJruVu+Gw/g4F47HwJcZSnQ/2kKLzIpPIHgaJVzE5OEBhpTpYhkbl2PijJJaBu6VXe9zsaN6maLevS0H3TlJgaBz5Qkl9kZUGjpr0VDPywAxHdGfY2wGlJrBPylqQ054F9lOFx22eo9kE+sk75iayWa+U1KOVF+xW32Zwj/NfV3eDeVYyRUxeUhmWLx5EZ05GeTLv+B/mk92A2erKUqsVPp7n797EoRkEKhJkJy3ykb15ebF6jRjEDd3vr0ypYkXvCTCumA0IFw/cKnBgyQHN2mfiQSsrCmX7MougJOy59Uz7q6yDUG6tkreGYAAA0EbWzWU/6uAqBwsZr4/QtxMT5udmzN8W8i18BW2SY40Nk0mMgg/WMTxdNhsslPB2Cjlls2oVEk3KIK/AyUA6A31oc4AKqHd/ecp3IrSpfgnsDrxmyY2Dk5f0QJ9s9sUVDNS7z5FE45dbILB2tXHctxLoBI0ZXLHwT1ABMsmxRprRzgkfobpqU9pjDGYWsWCHU0Ou+FVDnu2gwh/0/6viQ/UT+L5J9wMSk4ZvcPPAYf9fAA7cK5bIqgbtyjO3zNjlPdukvDQh62e6qUU6zguzPyXPYXvQ0PeD1NqZlYfB50BYLwRCxP32P4e4EQ2LomkHlNAQTe99jp7lVlsR1r/93mcY+0t3q+/0B570WAPeup68ndWI0jr9Ld+fnIPfjbjR2cfMvU6udhI6pBCajv8jY+790D/tcZwXzK0UR6qZezhzvY8U9Dww5CUkqPlF6zoVW6052gz3Qjz0TqOtnMpYPxTKD5TrH3L3V0SeOpBJF+l3jmAyancrjsjW8+uZnGT0IFW8cFaRfPRa0RkDNf9Ps0EGWsVyFgDfENvj6/p8czuLVTkFA0bJxWiZfgHDz4bSDrD6g0Qf2eWNXnVc6aAAHvDbUvTe5NddpXOHUDZBRlVbXZcTttlR2Qpi2XJ94vFmd24oIlN1C30HQBEoGIm1h8k8s4S8xcidIpngkKe+2NWH8maU8vghsLEvQvwOxal6Ydl+9h9CpeiBilShZ774EY7Z0R3Y42LjRJbSMI3mS9uIaO/6wDwS3Vv+XM2NttmJoWDlyzfa4p80wDxaztgw+S8lXKGHMsr7yf/pYrMDWNT1KOcTc6/PNhqKX0x8sPPAiD74mo9ixDE3QipE9ksoABnUEDdU4HcYtSj1tDQpkktwVnVivR0ybAhzynGZH2fHBTpQBMbSHgglw5hH8H2MLOFInc7lGFED6MtoleVm7KkC2EhwFRZ+AcoZYD/s35NjWMc5l1+cuud4RKMXrTx7NkqzdFu+B8wd8mx3oq3gl/9+LGnSWn8I1P4B4djIBEjcbkoqjR6PBIMbpHhnczab5my0lOgD+JG/qZiI+BNnepnJFxBKaQ3hWpJt6qTUFBLv5f4dJ+EEHc4v8MimidkaNZ0Va7byOV4EYeu2YtVAcb1MSNxc8+4qBHPOukEXytBpm4V6mMaxqEgpOkw5QOJrvdjLttOz3ahniYqngZgi49ZRP+eOvpVOZOzplSZqO2PEfa7qCD2Cl/cnpKscP9xFktyseyfknFOIMFR3WF1ZCi0JzhfmZqrX3SVeZEeCtjWLs8AuRGNodCuyQWmWRzQ4vbnAKJu/gcgNcalvfuqOch6XVwoqhGMltTa/ss5e3kwKPjgn3L0K7ih1fuFejIVC8V/gRTQymTvGG3R4cfOjeQ2AWadNYyXhJ2aUEnFcSOrzXGS2rW6ZX+/ptMesUHz0aDGcVmv4q5AvAD4E1n/foQo436wBKULxhofSrndeOLRDYhxrqxi72yNta1RWU8Gg6/xsv8/T+E+sH4CJzdHizx/r/fj4mO6D1z5lkW3opRg8wOi2uDK+m8q97gY129Hg2JhAs3ySRHkx9cxrA7zg99IlarVNmzeOkk8xj9czYGnZX/Jg7GPMhST9dX2ojCzeKVscOyY+slhTWy8j/oqTR1PCEtS3bRDkoplMfUC+EFYZ3Eb2PGmWp3uamTMqX9Ul5/08V1xX+NzBVUM41EAt+hCnvYT26QBLpg7S+ML1aDWXEB98p3bxAPKYm3czR+GDc4GEMjVLx1upx/qhvdD+50pdfn/DskindRbWosOrb2ngeOTXYJsmt150KOCSAtxxSeSAkDgZrq/v7LC9CvbzefC7b3q6hcPIz7zcvHN/8h89FDuQnERYMgVAWv0RxO2e3OR4UvTUIS+9hkloWrXiC95CMRXeblGdbRyc6z8FWaqvg7TCFVYjuXAEFTaJK5AiO2SWdhq1+BFbDlfsLJXgmlwQAAA79lABNEIhABz9Fp4k9H3/sxK9kuu4AmcVZkrSLYebid8bGr1tbfeFDyh7zALbJoOpUODp3CDv1flWyTfitAOaPtGEoFCn7K6dpDzK/1QT/9UeE9FlP4wQOm/Ip3sK8/Uc447QheBJB6UOyhVQCCTIXzHOx6i5QIYBTGM31d00IyhDzWhck7zExSCcElob87qO3Ur9ZIgJq/0KmJz9yWuPHmTOPcr3bTInN/vAENLPXk9R9cCIR9kuQNeVFaEYtc/T2rReECp1c2usjQaLuZft2WzB2NLU2Oj0kqu/9rYCNBUs+sz5+s80L7NrIGlTVJn1QN3VpE/CX4JuYyeSEoXFWX8zIsKu9u8Ht15shMuW8KT8vjJLQuxNa2D5MH8NrLE+vYgv2ssSkNcuCIhC19johDloHinqDqNWeJRFdN6Ak+YityQ1UYTtquix0pGU8VLh2J9YKgABtBzyaS/+WRdkHZGF//+uPggAPgj7xXvOoyD0gcz9lknof7NtQKtEogw0lRv3dpSnxpJnHhOCkk2VHpQYSeux2T8Q5Di40xBJfJ07kFwHUDXxayU9tioYpixPrccX5iu3aSsuHubT3CMTrg8ktpmJEHm1AUMw1N/OzznobE0bwWB5a1lv312xxcWf2WYfMuYYMZsA1u8r72uKvsmZk223YDyCX4Et6IcxX2cgpnX2Ra12rFofjLKMdDfLma6WmQcshO5ux/Djji5gQhYhOQ0RlCqgnGe0sTZESgqP7LE9+fjR0gqIQN3oURVcZnFEZO/mQg3J+wAvZNbiD+VpoJK0BCJJdIjQhbXc/+gQ/01fSeLCuETczCvL5v+OL4ahfbf4tkEikf90lagXCSmfF2QF962E0OOW3gWk/48/p4jvyGODY9IhlsK9Mj64xc2WRBzzHdsP4mtAcTUJlkz7S7w/AgDhj0cXIBV8AJmJlcd8UczSfct+Ut1rSmFxSBYHhnH/GDco+qWesRSRbRrW9lnZ/uJjeJHsvE/REsnnXC0ZtWyXi9V0frHdtOUlwqrL/Qi8Rlmqhv1wJPHmbWc2FeXfS6GlMtFTNER+xnNdGxdbsuG/l5pbAXLspdOju3IAeNuBR0Eo2SnWYx640YSiCZ0VW6in95hdeMbIDgf3UtT4+ndsCIjbTYJ1Bq1cnvpX9WKyY05z1OTSNKn1hN8kH+7w+bJRMdn09Uu7nRnijLs10ZA1vWCaAAADMchNFV4qCRZ9QICIdiY/in5gASwv2NIsWwGIErquJ3GEIVAEpyIbkn6zdarsQID6nOTwAAAPNBmiRsQ48a9IowAAADAAJb9GpcxyBuQYFaZYWNiBwcHnKa2nQbg2RYsdiQu+kHAUuurYErr75FsuYmR8aXmmgFbkTVL7QaAr6lxfvFVqmWV72KYAJE3GSr7GlY4B1IBmRBcZ3IXG9WCsnxr4wkQZwlUOnDFBZo/KqlyaSAuWG9Y93IDF/rcLGdUgibxrtL+RU0Orl4eYxMcPOwkSqygQ5Nv/bUi4O1A7VnldW4Zs7veEg9YoGRBjXyIWQb9ftaHQDUdzET6es4FN4bfrf3RskwJRcmItteho1EGQWdK1mS6X37pgzhPOIaGyjb/T5tTmHiWYAAAAEGQQCqmiRsQ48TviflRQk+7x9xdWy+ofMgPYlW33KREndB9RODQon5hWImgKjfF3FG6V8uRjpniz4rJUCG4M7pAcCzJJ1l0wu0yAqxPi1vS28W2yr7JtBpWpIuPJen87FNAhkCHX1wACOvtozhIMnoeN4O6a2lqa1e9UTussVe+65hS3/qyWue2Ld1IXzH8suxc4BRubCZBVXBQ4y10EkP1xUx0D63pai0NvkvhHTgeetDGr8CMmiX3CkizYVabKvLzJ8TYz2T/+lTgqJiRl3PQPMNMzqmEnboRU5grJAER1KoUO4qJJZvY2XfKwNj7gdeaXpLyzIOea8e7US33kIVFqVe5qVjgAAAAMVBAFUmiRsQRP/X2/gA4u7oAaDhOV4h+wRGkywvVvtr4C+QACijmUOrjB2kYOXe6TwQU1m+gs6LlT3MW/SskeZkS1cmTrPch5ybaQAqAim9CfXirCgOQiaYKyGilOGl4I+sZChX9V416QotOY5QU9gcnu53CGJ4qz9lxtgjv9hwuttzT93CnCCkS1FSSv6jkdov2CC0ftsvHdJeyjMGBZxY8ktOxD6DiS6FNqN+sWvFM9+gaht3SuzlfLu19vVJ1Ef6PjgTUAAAAOVBAH+miRsQRP8hDqnYD+GguqkeKhLm6lUBItQe8A0kTBYe2jXLhEjAAPF2zO2F35CGZp67FILgYr0ZFL9vf60wyOjLVvMfZXmwCPWRoXNNT9kK2SpROU/XkWSOHlXAKX/QDv832ctqDW8qPjxekHqYqKGYzTiId4w89AhEyvmgP02ulS1+0j0LWzjY9G6gR00JwGykJnzCXT3NWpC0gf8vUzYHKenWLXiHMHLaLgiY38FZfS5XFnENUUFJguiccAPuEEhR35SYQS6KksFInDPv5X2gBiWg/8XjmH8ONlKSJHXUUx6AAAAAgUEALTGiRsQRPyN9S/sFvQcOCNrntSYt/Hv3ZWEJEGiYW41QgszHF+AEf6KeArjbUaAFNDtsp1RKXBL5QixNJtvACd23z5q46e5bnYWo3xwGEexcju4nwOMBduSL6O4ibn2Fojh1YoxTObl7HiD5+IVZcuiOtml9F+BXJqZzbDyQQAAAAIZBADfRokbEET/X2/gANWeNOZuAA7OSsgnjtQBCTGOAgHwzNewI5c2IARDb3BXfpGLkl8/r/nJh05WY7OxrHvla9mAAsUCQUHCXzirSStiYkByWzi3ESuJbiLNOHSzSobq9zDhWD6E6uhBri8v6ycI4e+tVtn4WX+DHYXmSVHQph/4zsyH06AAAAHtBABCcaJGxDj8T7k18vbFYlARZE0pYSqlte64Abx3IqCoi1JzIqEhOgAoUM7BIZLLtY1OVrTPQiHW3DFS8nvGo/w2AQ5IDve6kdLlZPPgo0tXFqfjEdsNiphGTraDgXhm6hMGKZX/g2L8C1t0E3Xuzn+rtoIccjojgkYAAAAByQQATRGiRsQ4/Gwg0AAADAKFAPHVbgL0gKYn/uRdQG87+G8d9J0gKb8exZghwDeMHNsn1U+DiaQRSc1bXXD08+QSyGabRK/A1ZwBZjF5Jg7HzrWjLUxN8m5HAFf2mjBGc2Gr+n3ITrUdNqZNqFXDXctXLAAAAckGeQniCh/8pqiFQUEfPFQBOv0jr5FP0S0XhNd4mSVhWoFVZO632RZQz2hZW8hAH29/C6VzYnCxEjKNMiuYH+0hr+VhuNFZVV7r2rMRajgHXaJq/k+/XF+D9I5ULUb3ROnO43QRu1NG9NjNetRzh7N1/UQAAAGVBAKqeQniCh/9UOcSALDowX4CrdKItW6SB1Te8Vu/bnGCEKBcK+1pBMdTy3poayVFFAYcSGJtoSdi5lWvnwMgRu0ako/TXeH04Ns+UdSZehAL5gGwkB3JojMEaGlIOopFA+o9PSwAAAGVBAFUnkJ4gof9s4fP9YyK6ctKbAUPkzIJU1UsNRB+RtPEKjd7W7DMZ6qrvnHbzk/PJug0qyd6nxI5y0rpcxyMlCjZb9muLU1cHADdlWlIiXHYwc3nHSj8d7bDS8lCYw+ZDq2kKHQAAAIBBAH+nkJ4gof9a96YVklQ4AZISBBbh5C1Eu180r7Wzprbu8E1rlr6wbkAryhCm5GOeDa3opRsChWJ2BfBIOVuokjjmcuGIdDg+jEBhlS/XmPbjbmn0875R8DXso9GP4hTr9QhN5Bj+PitwT4ElGA8b6W2eXrWbiZNJoqxq6q5DuQAAAGBBAC0x5CeIKH9eSHKwfZg5AQRjjq9DwJvmj3sXy1lYRRmNR2jNL2mcnqaCWcmmL+qC+N+Q9kmIoZIHGbAE7dBYKMYbndYi8BN/Pzv6g6RG22GiYkwlZCoj8sNWxS3i+skAAAA0QQA30eQniCh/7bszT7PqRI3rrVePUm7W8W9lN7wfqxUVyTTZ1hOxgRIe+6cEJysmLZR2vwAAADxBABCceQniCh9tzhQ3nRGA8F9DkH9S/lZp++sE76C7PJA8WZ0BDv6tb3vQZl/DhNFlb1qk7Rm5TdNyDlkAAAAqQQATRHkJ4gofbnkMAatyMP4KsFUv52LAoIJFo/l+EjVbxZVG94N+nkZBAAAAQQGeYXRBY/8rQpVRvi+TQDNjtZLg5HaGuieG2vULDw74LyJcoTCeHbn2L/SnzCWliX7msYvhqCG+SK8lH/6Z0WzQAAAATgEAqp5hdEFj/1i0EKvCnMQNdN/LnPBj5CkGBThbsxtKd6N7ozys/ckCBvwXO0NC6ZQjhHaLiHySVKYDIuyxe2HWuQSeKV3B9eFoH7gNyAAAAEoBAFUnmF0QWP9fISCr8vWZUrVNtqL3vAqChhHLGfyRl+2yDs+DVE+42vcx/X8De4WASuy9znwZ/3npqTXO4H5iQOctEQoGBmg44AAAAFgBAH+nmF0QWP9c9GoeqthFXf8QEXTN7hoZrxDv8FrtRT4uHJlUR6AaocxbIk1zJTsa7Es54lKsSBa90M6kXt8QciEftpWTpzRFPF6gG+8xCoQDL1UdSEg7AAAAMQEALTHmF0QWP2GLsH2Dkkk1ln5fBd2sWw4GWEdBwkO7Wz6BL5EjzVMwVlDf1IHa0+gAAAAeAQA30eYXRBY/WK8keZBNUCHpYtNbDvQ0v91pGp6EAAAAJwEAEJx5hdEFj1O1zz2ptuL7f8+ztI/sPJAt/AjV0SK5MZi2SHYTgAAAAB0BABNEeYXRBY9x2Uiio6zCYNmpIqfdmXcFv3fbMAAAADoBnmNqQWP/K1puUQgKKCu4itTtfoSIHXUyDUMzsW79Uw+4ZP69U57QBqzWIGcGJyE/iyLGq3EvGOKpAAAATwEAqp5jakFj/1gebieasC/6L/n6T+5qfI97p3ALUQ/i5pbFDRaxzLrK00gOeH7sf98v9PAIMAIc9QzPKxGGE9c1AxAQJ7RGWkIkAtoT18kAAABUAQBVJ5jakFj/XMy7dB4BcRt0CL6XjLbpAJi3ujm9Q6GFZZkj/qPeTyAs554XGZsOoxJ6xNncmdFxtwRpoamz1jyZ1+H11plZz6tCdvEfmHj+ikFlAAAAZQEAf6eY2pBY/18TU/s+6DfhO2wxwUwNspysynHhDk1z4kHDAKu1tuOd5eI3QZj0YKqLF6/aIZLXrg4TVBsBMSNOPP+LYOR8G84vu9rjISZLBpEcqMVdqWvmMUTQ1WiCWbLH8iW9AAAAQwEALTHmNqQWP2FVoasj9/hQ9dwLwNm8qbkNSHL66yo/TLVEan2lCPZSJccQKBY6twgrcmOUTbphwMmvLmSKuUqmH8EAAAAvAQA30eY2pBY/WFN5xomfOw5ccfh+U80xoQj1PWmXH6Dc49kwDi86QCWE/7mh4qEAAAAqAQAQnHmNqQWPWC24LduoED01mqxP+LL44/3i6QzbO1ps1S/UuRm7LAe9AAAAIwEAE0R5jakFj3Mc7ByZw8HuaIgBk3UsN0bff/b9gt7nq7ZBAAACCkGaaEmoQWiZTAhR/wxnZWgAAAMAAAj18+iNpSJ90uJ+L/9nmEY6NzVltNMsLAjAH9Haw//G2iUZPxTjWVaJQrRriXAGpIITjIu1/IaXEdAW9a9PQJZDG1azSJZ8/EzgQwlCh5zsVOoNM7+8hyMxc1fUB8LarSvdfxsJKE7oE8Cpmb1p3MpYAGzJELaplnUPFkyVhP8oYametmst9aRXlyFtTejgaS/FgDewh8igp2NnM7PvySSnERkNA0p8bUodp9WoLJiMYpe86981IsYWnQCSKggskAucAwYlm5Pegt5FALFNdggcO0sg3SODs1xvlpnYZII4ZWkpzinKvOZv7xn9zZPw/oJIRgGfiVHIuY7hxV2KwCDQFvi24WYR+dcN20YVKs8bvNS2n1nOh26folwK6pRkPbgMTD4pIy4MyoxiVsJY4ykulAbRDWhqS7ESn+/O6uxArpDRhlOggSI7sc4Crbz490wEE48b8FELxDCdIK+L/7SdNwtDAYwfpr7pXvafPI3qVtKntwd/IMI1d5bpu+vcH1aolqyJFK/dCrK3xEU4TkV8VTEgjHSt4J/6glchvO+oJUXCp22wADGpfVZgVRjaA2YbT8cfNVxFFZBrJV5056iF4VyCaP1qhzGAXKAQEx9rdOL6SXeN9T6JhqcGYfZuseXZ3MI+bIh25REzuc38QhgaqXQFFQAAAcNBAKqaaEmoQWiZTAhZ/wv+6JuaYL3HCGdvraVz4NWbILUdbgPKUnrWWJXKUJ3v6uB6dmwUpTJWvOAy7WIco9gxpnKYAAADAA4ssASEhsCup831uuVs3pW6ldpmsZ0l577j8vWT3qnKzZiGw97q2PujIjurGotWPE/E08ICMTlBUaq+HTq9NIaQqtQ6XqcgAHayYkqAAFz5aemXYzvsVmD0zwKE7MDvbYB911G50fDanoQFmZH8v6mDlgyBeuiSAArXvsZxJFExW99EaP6frl4gKGiSlVHKvDomKpvJ6FPJ8bHnCvcpbbUOSo8jfMJJY0RkJjO09R4wLiD8xzSerA6KsSlRp3ohpILf+fFpYBnBKNU+SY2XhvbXHGA67aSohYrWCn5+0j0gbD1vpmxec4/8uGQgL5/ws3OFuj8Dw84nngTSm3v85/gmOl0VupG8gSMHgw/johSbVlEUTJUcsobCJisT3MkyXHwLMKLMw7RyuURQynFg0KxSf4JDdwGrOy7yKgZY9SmAfJUbWnzgSVaSimvGgXeDYc8tzhspGmWFVAaxlztc+8AFKR1OoSsA3g6JehgUyvLon9mYstxirzQ/V1jXAAABfEEAVSaaEmoQWiZTAhp/El6XQywZf0mINUdASL6DlkY4GdBef7CK/49wAAAOwIzYy2gRb3L/Ei8LhWkxE89Zoo+QDfYly/a6nOdBwaAdw6910eUAJlHB4bzCkmsO2GgR35WwdtcfqTemoctSXq16hLEfz47UAKsRbQ6vvQF7q0I9PaiaKF3FFfS9eT2PzM7tqhFi0ztiXBdmzltJAH8lBQ/SrJ0yztA2558WiaNAdaI86dGMUs0sI+qyJTb3vTvTYpmPUvjIVuX0qGpgso2FCRFrTXRUQa/+SjshJ+6wwJtmY93Twt5OVuq94YYRKhPiQ7uK3h1qqKTjA081U80XVEavO3QV6vvDWnyL0z+qmAie1JPXO47jo1B67U6zDF27FWxS8zohIlF4SxtdJ2YKVIXDmY79Cl8mLSIBkGy+rWppJo+obQwde80UKavdBuMbuFcFoobv7L3SUfNRcWODPJmmaLrp9/B27DN4WiD65hGRHqfufeldkAsBOiS5AAACOUEAf6aaEmoQWiZTAhp/El6XN89cVUp29KV6rG8CxXQLIpxVb520Rhf82pma7F75YswAEZHX5oViGRTdA0hO5Wk+/i7MWvCqsa/WA9LEJWAGQeZNysv1GfC9E7J2gbNlzXfIT6EDxR5eEyGsoTyT7s0wXSJQP/7litrjDK11ozOdNhU6ADLZBPu17xfpPh2GiQklhF1jaePbxDzziCfWk+mbRCNaMEfr/j/mqLCgE9YrssfwRF8tV/Prk9sSXkOX/vE+m6AUGyrhKOi3DepUUhI069qE+JM8Y/dQRIUyorIP/T6XRkFaeM6dVzDjTZ/wc2yDFI4xsIqi38QlGNS7vteF5t5e9+EidAHboN+KrCJtl5+2H/DK4lk6iIgQzjFiYlmV0zYgdXjzu9Tcsv3utMabflEclxi/8bh8x6V91pETxNH4xIXdR7MKAgGhY2sf0jAExu/gp0+59ZOrNPn9NZ8ZC7euLgvCirH1XkMiJN71g9Iy4PQ3Pf2V38Sri0UGnLOFcMnm8GS6tgytPtt/JxeCWldMNJLtjDlWASXAEI92xnZ5rV66sWK2AkUYMjIVfr2ncY0T8SeXB4R7z//lAFW9zdAvQUanTgWfyu4JqCpPB3Ocasxi+kjggnf24Q1op+DC1JOAEyc7GdA6XvnMPSmkCaTspQ+HGFg2h4bojTDyIYmYw3zydf3Ii9Eodhel7MmFavkESJlbk8fHKQ69Vqo097uTpMtccziLXLjL524pp2Z4d4hyJ9PhAAABDUEALTGmhJqEFomUwIafmDiskqofMXUGaRoQCcTJOpOBMyoHD9mqAIr/rC0T0+2+eEJARjQvyRAnJedYz5YPlWlp3OxnmMgBtB0TWbbMWfIaDGTKbKYHXto8Y+bcS2pHbvmc77ADSA5syIN7zCIXX7b3ePWsI66VHm59hwlKfL4d/EcM8MDYi5GVqIxkQVrrNCE6SwBOOkmjtFw1MDoVqmy7zi9r+MaUxkcO8pWfKfLTbqQxv/BUS+iSqZr3s66obWZnRfR46aPGGcJZZ1ATZBYXT9oHCDHRipDlDXffNJeCYCFIkAN6nYZZ6CRe/tbasWmmncnJXS+ZDXUZDkfrqe5gsztXfNyvJUmRCUfBAAAAyEEAN9GmhJqEFomUwIafGXy7xXBAAAPhq0rb30OGam+2MAAndE/7MKAAAXXLCMqbeJdBPWtEU6DBJEd05vNzgwBQ2yCn1t1XAD16OUjF869YjbcpCCmuwli60IvIS10NYdYgZfDRLq/A8RxUIUjacVgePpJJPpmK3Co9e57JWkNXoK2gSfNXHeapliFRgIef3gdNdMhg3gDuxhTb84ApRBnNyxhIWpF1pOHuXJyew/NhprpRN+dU8uwr1g4Wc5enhGsLVHYn4kFRAAAA0kEAEJxpoSahBaJlMCFH/w7rmBNwAABvOc134NXABQVcRmu93WN+nu6noUta/NxfArpFUHf5RzkGvQNEJYFdwjGQYi+ZAGB8ySrU2p/UDwNaDEL41M8XduJkcEy5WQphj+jMn+k9NgXPfrMCrGZtLNr7xQ6sRKIJdPAPBMlN38F5w72gEGfhWTgJYircHahg4sXhsio3awaG3E+HSc6GAhE0T0jCfQRdurdUwnY9gTlle9/6YkNz6n0Qo2i6sBnyJttGmJ4FLAfz+goYCpLq6opU4QAAANNBABNEaaEmoQWiZTAhR/8scdw25GlAvjLcAAADAAAEjVusWy+6E0uqpKfd8f2wLOR81bb8Jfv2g1SE2lUALC1zJEAYlJb4Ibn9XdrYNHPaLun1x7bo1hycRi88hgF6sAYs+CpPMWdbPeFOj3yWSLiK5qkfW4TVifCOl3qBapSJZsdt+Km+kz2b3xp3PdaXwvJyBmSHbukOHmQ2qVSnqhmnLnQxxOKew5ZRBkOLdLBjyIXKQn+E2DwlJwjzpwityYdfWDXOS5jdepFXI7OlAASfWUGNAAAAjEGehkURLBI/KLgPfnCIwJdP7CJk77bQ3B9rRfqP3b3Yexfmj4OUkz7n9nMrHa1a97J04+50kqzLgMrPblIxxPeMOOZ5oWZe1g8vGnQGI7F9gGNbIiJikMg4b9zxG/oiguG+u+8kfVowBlbQto3MM3g8FSJrB7GZjOgOwc/x+viF/VNFdcmqW4fzCDppAAAAmEEAqp6GRREsEj9SNnmBECyg2a38RnzRoYS7fwVEySzPwCzRkqGxEYvYnWe7+88d8GjChPaB4lAutc3Omkpckkag4cvgrHqmTbvC7cX0GYIy3OvnKGIaY6KvnBIB9irmvSwJVeSOouLYs3zzCF4Rrdc1NDT9p4iw3l90XTcNoRxdLTkUjZGZzS8KZCdpIFfYHYM5gYEu9VEhAAAAoEEAVSehkURLBI9V9EeoO9SEiOV9vZEpGTv3sIMbEOAiVUbD4M7jivusPiCiS0Izzw/OOFP3guSJwb8BbfyjnqrXoI4MM6NeQ/RLHxpM61o/muu9qBjMAwI5v/ZDxAV9yGOJtBJiQInJGtZpEOmI3vGXvTu5sPJFd2KRQ4/AvGTYM1kvZsEmcxJa8qLj8h+gFu5QEZ0LjI4R++2kFwOjkYEAAADsQQB/p6GRREsEj1kGCJwQKwrNRrgACTqYmeDsYn9UYAdSFZEsQVz/NUosB9RDwv5qaSmA1txmEg3k8wk9UWKnIcBfkQRw9y8HHostQkb4CBkviXt8aiwuxvHm3HvvjQH7HE+NHhKSokq203EHDu04z23Zq2vqO8/2g++Rnhn4Jo9YUKNYDSAx59U+P1yV6FhrDNgqZ7QsBnyusZRbPhh3y0RhCmqIvm0F599B/4plnsbeqU1KrBGp8mTNiGu1r5kMDpD9fwrAix5xbb/O3m/Yydp/H0oAseBpqXHTZp3PEfXH6MNpVvuEKIF72n8AAAB9QQAtMehkURLBI/9Y6lWnH2WfR5+WG4+srY4odr87Z7DC+C2Vhnjef/ueFzDDaRyVNRjJ12dkfEDwBJBElafXsPEV/EEH+5BWFBF+/NsEvgKDDRpsD3RIpM0LHMTJOGZagfko/sAdIj0cwjSlfJLEOOuuaW78vwuMLYk3rdEAAAByQQA30ehkURLBI/+pU0s1fAFOLKK5VXvYcLXQpSP5e0pM9TzCJQhxOw8lvSNq0XM8AWon/9N0EWHL4MRgfI11pi54aG/aP2DfZpLQXNdQ1kEKE30yYoGfMGsKw9U7fblQtvNZfqzDa+mCRPD4JV3/ir1BAAAARkEAEJx6GRREsEj/UV4E4HkeeYwrsvpU4DVO5NQUxzrF02y7exG7LjNT5JIPTvHOM3LNCy/t9FNvQCr50Dd7VGfFdhzgj5MAAAA3QQATRHoZFESwSP9qURpD7bitF4YNPhIVCcpvcmnUMIuTXausT2MYzResQL3l1kz071CTiKiH1wAAADcBnqV0QVP/KdbLzDaMmWuZmrw2IeuhG5T1FYz5MvlLINwAXDCgvcKMn6KHpgjdJvyPigd7b+2BAAAAQAEAqp6ldEFT/1W/xYmipHdxPmltO26z316oSMvkoA4GThOASkHnk4wOF3y6IMg/g0t7m2nqsIOYbzkKOIXwd2EAAABVAQBVJ6ldEFT/WrgfbIhZ3nSE/PD1WkV8R4VbOrlZNqmrXZUuu/rGzKWftJKtuivfAxBhg/1Lq5cZCd2jSSbTyZYFL7GYMBgJsltCML+6gYh/EnnrDwAAAGEBAH+nqV0QVP9c+B9sRjw7CHclZbtM5IzV5O1IYVAJy85yjxUBg6xbr0cXJ6oMYinSq82I07mvDfz/9WxElQIdeKz8Bc2pZ8i2sQ5NFRWvhzL7bWbkgrX6jkfKlV/zWU2RAAAARgEALTHqV0QVP19Tv0qzVq1g3xkUeuhbi1ZNpP0SkgDgDRDou5u5giEP1Xp6lo1H2Fk2sJjRbMP6+XqX7iEqppSYN0r+bDkAAAA9AQA30epXRBU/VihFkRppKfRhRTiHzTms7FbiExZH9iLzpmI9s2K1TjzOqPD7x3+CcAm1XpiinNbZ79KSwQAAADkBABCcepXRBU9WKCmJEK0fEneEES3kmtZIFJfxCQFXiIHB5M8GzovJgsdlQLZKCYYhjez/ajzVxekAAAAgAQATRHqV0QVPb4lEymVPt33zaQFDINMTENbpv9BbORsAAAA9AZ6nakEz/yf5k7UlKX7WZlyFhaHPoz/Y9/iNpq25soAHUMstkKgVNszRPAXx3nlUQpsrOawSeHlcMDzhmAAAAEUBAKqep2pBM/9SobWvDRMKvE489nQ51eZtx7y6Dv2Nc5BPeLR17cKC/mgdMSTxgDMEvTdBfall7yzrEn/ASyO8LUcNamAAAABzAQBVJ6nakEz/VxW4fPj4xLzSegiRz1sYoqkNC0SvUjWJoZBtak7HipbxowWzwWAFCxXdicBWwILU53OI830Gh51+rR2kG0hstCrvrioLxB4FJZx5UccWi203Ll82VnqpArfsAANmnoxoBEAkQ/eOnz4H+AAAAKkBAH+nqdqQTP9ZWwMzF/vf/vA9ydsyO1zaUUUkAnMtH5Xpw58FzTXeUNIBNSfvUH93siznxPDTav18bpX9VUIvJu+9nDIkJzgzGykR01sAHUgq8lUCQ2gKhaZmTtyq2SgtAZPGZy6J1Fw8zXom9L28YvMlz1coQa4JHgiFtmOuNQXx3iBN/4NVNAuJEr0XDf7xSXr7SrxOpVRe99xUKRuNttiMlHFsMAzAAAAAYQEALTHqdqQTP1Vcel8lNZb/HE4tS8vXIE80K13lYHGJE2sdT3xFge9AKqKrr6nQX55MJMQwwVaJ1TBgpXTUKBy+BsioxiEhb5C0Mbe+usJzivnyNXTDebNKCXMh/9+j0dAAAAA6AQA30ep2pBM/UqRZLV0r06Aw+EoC/8h6LKE/j78NRjMY73zSSvrxKHhM2OxQn7QhbSMe03o1FrgZQgAAADUBABCcep2pBM9SpFjC/gIsvpYhqsiOe351pGMJGm9xWyqzhq/bPbd1wId/2rnxBB4EV6Z4SwAAACsBABNEep2pBM8pYRgD//MvSwvSK0V5ZCFh74bEE9M87Fhg8kZ0uGCiGjogAAADjEGarEmoQWyZTAjn/wGlNtcyRlziRgE/BXCyPi6Lg0wuY5y+Qac4sUIUg8ESHjAkOvmPzZHpRE40u/Ries7wIfSciO5AhbVwQt8BYMzNkr9VHCeu5J9Q18rjNyLTgfx4c/AKTCwm8YdiQoxP3F6zsQoecPjdIzIOg3frMz7pbkj20XG5rGHBhrBwfABMXmV+NbOtcRiTbKUA6fejAI3XNiFM8xqh7ASkjcCpHroIiXRTfYz5uzQnn+CWOkrGFji6MKddwqlN9pWXKICtf5skO75WIwDQ9vgB9pphMTe7l8CbAfcjbdGttZljVIO/7/z6WwqOCwXgThI+b1d+4bo3rCSUzuL0ZK5ed0FztQX8lwoDGvUVa5oPB33QyH9vM6jY/blcFL/4+Yb7GpodlelhqUptxstNq5wFfQ9+IX+GcnDNiu9CKApJ3LEFzxlld3ErN18DIaDTJ5WLvirDJLeEpDKsoMRpgpFisHhOrbS39JNa/SaC1EuoOkWWFDuTdZuiR+Of80Ddq8FwMegJzwoZ5yX9SSUEBOEEI9y0mISlRrrBZsTcR3INQXsIeYywuGXdpnhZHnJcrh9WrxQdREJzIH7IhWSzvlNZpJr1WEgfQNHLZcFdPEv/Jx76sKVks4BM4ZO/WXsW+FdejhuUi2BVK/t5eg+T9vdkqUfTHO6e79ZJJXBYFviD1oTFfRQgEWUOBTibEMmbbH+DrMsS4Yg6H5EKP9YYGqKBWBBLWujkbK96+zkmlnOqSPNNiGuDy3ivnbNM3wdt+mAlwwvgJCi3/3r5vU0NETRDyw83JqBUog/EHd/Uq5m/+TUP/ezpVRCPjPVrze3HtcVHCLIkECvHXUMCAz7eUYcpQZhm6NEqKvNhLvs3ZUlLpOfCcUa3DokY/+IB/4KsOdfvd79ot3Ymu/ID+Kq4QQQRF2w2SRbdEYnmqIaWPWPBJgtMW+apwHUMWbRsPnz+YOhKhSTmXSasUKyV2jR+56hWJL3vIF+fs96CQsfGcnZlFprJSt0yBH5iTigHgEORvKRPr+pEtk3lkdHhuvoySOMvZ6JyUw/XEdV8RqSZD9URLkSHHNewoHNgjbBUMjC+WMxjZ2MniTHOUDRDzjRhnOHrR8bXY9NDRbA2dyoMKoOepA5r0vq/MmR+SRzCV6ii5OoiMK879HdhgqYpZOqQAesIf+ojW8DCxYtOrn3hJb5CuhuRIb2AAAACyUEAqpqsSahBbJlMCOf/pjRBqEAAG5eLMUIWO8x1Dg4hnf53Zs3RKLJrTNUt47J3wCm+4PzZm8srr/QIueEfLbhMRCa1fyxdAy8OcAACa2u0pgCewP6nVA01BPB+Vhqh+BVzO7PN8/LYwiNyGl7dbbKTBde1r3UjErnY5eMTD+w/DOq3lCNOpL82CqgrA0V1fFZ8pLUrwCz3MHs8nk89pyKTbUDy6crPHOALZQ6nZWYjizINH6UQT30SoFo0Rhl08J+93K43hZolBrlQBZxT1f4ngKQLQ7hMYKg5hsfBSyOu7hkfBM+COfCKrny073mpI/41UCRRr3NASCaNCjFQxiaBydwOc0FHIkUvCjmBPQxEvWWRi3yUD009FhdIUqyRxNi9szba3t37FBCEzuYwegv+rKtpDa8kKacja/2ilqSw5wBxtH1NJ2dgWiFQ9xQodu6Z9ez62NOSROPFe1giRtqszSd+ZlxbW+qrV/HF+QptWZIt8ZivVWriob97RF3+TVmWLKwAkSeiV1LVOUt8iprPhVZAUvMpqHRWsy+4ne3RRuK7rdVkwOw0CckvcoEQJyV5HfPWRd3lNgXN0zp7yd/+EGOZnVqZj/zzv8to5vpyCXVSY4RiUwRW3QXtgVBDf2F2AyG4rqABnEW7EYZIOfQZ2voF9MSEvo81lzu78Q/QVxdLabMWvqZIu9wqfBxdpb5vVTrhJwId/Jgvf9A8jg5ide9r+BRqIC/OQWbiZdEH6kAlOG5joT3Dej+XY2w0xRw8bYbw1GRBhd3+nvPO02tyNp/wv1MgsacVmhawbqv69fNkVOT1v3VVWMUwVemfceKIDtWLklpzx0VDjvGdqJvQ5tRGi5p0jcJu9P84X1IHWS647mefJcruxUBa6mpqcZ40bYZNrOSUl7BnuTNjYfgeEittDj0tNxzBsi0e55jCbeqoxRkNMQxQAAACT0EAVSarEmoQWyZTAhJ/lyiRHfaOb7NQ8r4IYOxv/MBsdq4TFIlokn7KxJPfYZBpTYMKkDViDhbxSIOVUAAd9f4WwAKJINv05EnNDLwdY9U3RF6UzVQoBghL5ynAWWpjJuucFz5z4bl+ZrABYfXE5rHHEzMaT0LMdc3eTnlKtYWIoU4FCWKDH0t9RN1oP6hfJFZMzJfcgpgixG5s89sMT8izsYXWu7ZgQXcznvjA48Pn4aKTtwSsuYGFUtTYeuNT8bn2flSJZavdpCZTw9PTqTbYR8mooo3TUZRIb7jn++jqpsZLNgqRhifmGp0ieiqjXdrSwlKV2pwA7jYBOkHmdEhRP8r2bnAC3l/2mDu13vfvCAVgL5J4KPzFXvBW0H9ji6hHhVMAmwBh0bJoWyF+VuSQWHYOOZYOx467imkS4TgL6VKlBWriZPinqOzOvNLRVXhbb91NnZyaVWN8SB1UyMsO5vy8kQAQMOHy1yyrOm/Isqxp1k4LqHyfUynX80x6KjMXApc0Atc9kufFAK9o7hmP5k4yGFP9CKa+y4dxq/cFcPTywNNTfd+7AkL5qQmkDcVRmlmeNsytuPJa+kMasiLcV0yXVB5jdaIc9SX/koQnRGlHIWURLV2jZ5sfQnhceqZ+USYfIENX6EPsWFC3T501/fQD1lYl2+SErymS0l1viy5bR1ONLptAQAQZvUl+kwktU0bBrUa3tFmOMa9Md3QaFIbYjkVxJyvDLgJCqdPk55xqXtWxchDZ7P9c1EaXYongh3sSW82WHBO42nzTEQAAA9FBAH+mqxJqEFsmUwISf5fvCRXYWk6EoABjXCn6PIzofs+3lDy8kLLwLjb86RxbvwSSxbh2lSGNJ1GOEJ8F8IALHslz6kE23Ct8pSoXjl2XoSHiRULig859Fw8ifYoAqIlufGoWZiQIKv3/G/HUhJi1zYDyy7F99Y+hqhNqErPXdAYHvXRsKK7yY1Iqy23bImHGZIT0+hlDAriP/tptONB8Qswei6Kw3frdFR62OTMoJNyjNElA5eooGTWk3ZJiI4OiuX/BN1CVIgr/DJB2ZbioEl1EglbAFxDxAgNnsDtmRR/fj+ZD+PW+WKtovEM3iv67Ptm3BgHLcAlyIMRQJFTt+JgaemEWX+aJUS+q7/0SLHVP75N8OkmecKZvoQH4ZEbEbLMWSPRE818TO2iggCemDI4yR+PHHR9pH5ttF30n3sevSgsMnD5YJNuwyWssXoy0D+2QiD+/FjgXe7pnHp4oZ07q4Tfcoy7J40Av8b7g/pk5dfotA1KqdaWDFe64/2+FlyqfXMIghLb+CgBMNhaSYUTfGHc8ctUd/+5l5wA2ifqvUJ/flQEl/kaly/cB3IGGfbpX8Ca0kjbcr4hELz/m2ESI8o2SB3fBuU8uxGRflcpAtyKj27KfHLDun2NWEq20c7SHjf+WYGEBlrXbDF0o9L8lXlZm6n2DMWjh8htSC9oN+L8DD5cvq63xLR1t43rA1cH+htt4u4A2qh5/upwcaDT0JXQCDq9m0w57HZR/ghZvo9apaa2K6Y/hZyWuG1QlJxUnxVP7depVLpqyC1ipdMUOU5ZwiiSM+VDvcmzoC6PlHpKmq9nLNKaxURPAEvy/yEymt1AdcWBPcukC1oYV5SuoVJvtcZl2cWrY0SPbirwdio1/2sIvNrgLd0vK+wrHVjjlOf7VoaxZSQd4nonXpCtj51jEN6/s+aNOG4aWn2vMCQoMn7AmPXAA1qZjIMpB7sARrLwyJVC1qK1WZfB1qhVGz9ZXK9eZI7DjDl5bdcXlzUh94tKv1rtEe1RJOH3yMHo/epwPTuP5LDSV5MDI3YOcbzFLV45+4QpyHaitTqk4ilYXzOsMx7B09fKGVJAalbeOC7fe1lItUpIJ85BlOeF3Iqrvv/fMplUX0Sl/L7PTan+J2H0pP78V/gw5F7MuKZblbL9ekTNZA+jfqX2tgcLRsCfvwT4SSeQRC2/FPgym4Bb88JqztXtbGtHd7+IHQEt0QSPieYoRG7d2rECRUa+a9rHU3qmPVbRGPFREZYNSt/usEZ3H5MgbnsLZlgughvJS676dIYp7gJN/W7xQQAAAAXlBAC0xqsSahBbJlMCEnwbrVSjgnhr+oCbPcxHkkKskFSeBhRFuqBK8Aq4yEjDC0gOMC913zkESzvUjf5TVatXr6pNcDPS2iAswqg/jiZQC1FQ8ePU44Vjtv4FeEfpx7x0S1jka0SD+FiC3NvWqQA2ox+CpXgaWfF0375+lSkx83dpuGVXpvE7e5a0blJw2zNOy83L/8FYcYWwEVH9scyy9VtbTpFgfBcs0k7BOL4hz7DfYrH7voyHOjCjnA8EbFnhieKkd5ycYGFf2b1XB/tziUAooGq5rN5FIPcktkS152Fi2423nQd1838w70dkyLmmCpTvzhqa27ZNeNr1CGe18a9rfR1P3PpdeFYnT4GcnbQbsBCK/5XMOIQkJuxSDkvEOp5S6rsCbzRmkc4/3hJ697gWlGKYf1bGiES0uB6ApVRwkWl/TeLvqA6Ygve65f8EdDF2wlsGSQQeukAK2/Rzt2iveh/5K5Hh+4AAiNJArP47Bj45VcLhHYAAAAUZBADfRqsSahBbJlMCEnzG0oTJQASI7qDJ2y17pGh+k1Lt0APkhOkkNfczb5Qd9+jxLO+oQAHxJ+pypeALo+IzvPsIWCWyAE4DGIsoAwvUOM86iMz4N29ZK4P2GoFsJQjaOpVSGsPdwyPVq5/LMd/auqmf/X1u1O7/NGd0v2a8yu2JgLn8V2BEhiO1tfKKsOmJYdtAx02cvl1bN+rV/uZbiz3UK4JooZPxCKgfLYfpAeDTlSNBLp+pZpOFTPF6fm5T+nnYyGds4mJ/NGz+gH1OPJiT9T6GWXFFUjDH/9pjnLi8dPeJ8FZwCzemXzM8vYjhtfac+Cw0SSp/CtuhvFIHgZ3qFQPahLu7Bi/Of0rFofNgrOI6v9IpgAWYQE4j8c9wIMQrQzktHRNxim4QJXQ1fPzD34Ie66LJL9AvmwyVSGBQo3+d6SAAAAZdBABCcarEmoQWyZTAjn0ayDUWoj127O9lWWhKrQoPqgAAfFhMJHqfq7FMTcXSYAWKRrECAAKVidPtu39rsqxGGEul1O3DP4X/FZpGguu+DOvPB49UtsxV0crYyjuer5zu/4tsL5XCkQlJkprhrQ6U95dAnAabFLIKT6enkXCi+eILCbahqDO6aIvmGB5l9xMg6t1TIJqZxJXl5h6mrzi3cRYjA0PjfB34FmWPJIitVVVRJPpDhIRNwnZsZJlz5E0XZ1/Me+QeLcHngA7MoUIJR4yiesL4oV/PYJXBKpDBGp80FoUgawZrZVi6qO8gF+w6+dcQtrEmA6+yrVOAv/YAe0pZSpD0W5ZXBuW5o6MKhpbUOnee1VC2Mm4FhmEr9XQfOq6i4k1CtWFb5FC/g+5gwsnUPkkOygRmOo4jNqJp9G4S5m7TunFufeTpvJ11H1kyrNAj9YhfpbQnaacVxJrKbdIO+220q1dL5XT+VEfp6RAY2qatv5X8WsXDtUxKrojxthWiG+TSvA3YJfYZn8M+BxKpQhnYsYAAAAPhBABNEarEmoQWyZTAjnzCvsriKSAAABlWbvaxrLNAIKvqiNGKS/AwWt/64zC69So7ZUurY50r1LHR2ux40ym7S+dV9bHWXPzY3yYDz6tFGZ8r8jhuvfsg2mwYtetJEmmmXjZRXbpg+LRFo8Cklw+iKcMjvXBh3vyoiT402fnydsYpU6xVwInIMovhGWp45FBV3ewd4LvuF2tNf3nzsoEsHg0WWxeuF4hIDnao0W1aprHkl4HbmezFuWlYSc3QUHyW8V5wXWCVRlo+DXYrgpk4wHsnaspfeNhqpWlbhHXNQneMMgL56AjVJEm2ONMq2568k8zn6x67wgAAAAQxBnspFFSw4/yN3KwtuXhCPPvbuBJsT4f+6rCw2Kd4o/v13l3/vhgu6qseex9EXvBLLgRzP6pA5d6BXe+DBP7ksRlk57lB2pYsIRFKPEKSnXIFmI84Xci9IY14dimKiUFzKhbt+oXMR/Z+SnoccrlbpKpJsUaOkW+x8/A2R5EfJ7D+JqTB7Z0QVSUcqRYOWFTFgl6d0jQDJTWl1rFDpZtIvIqSllFuncyEBJPbWRLItVAeewaPnU70EmsG0lrTVgOV1lNYyUjciJgnNPev585GXU6rYFtAx5Jr2oaEmYgQab3nIhdnqoitLGVs5Abf3g9amOd1qGR5aZmYPajs/FdqXXk9JiuCsFSY9lh1BAAABD0EAqp7KRRUsOP/nnaD134U74+ibaiaNru2MZvcQDEPQXK9qi9IYSHDjlaPFs+eMON/0XFbKzrdEj6N9Yi7+rbImELOwl2YcyOsnHWsuwQb8grToasBTVz1281+u1j7hDvMZqRXcMauas5CE+tVsslDXST2clO/ugFqo9kby5Bt5tZxJ0q+nuF50ee1tXJEZg8APCtp1HKGySBp1q2/t4d3a6ql6pCzmkufGwDU7FAwpMmCwMVrZjFdOpNUd4hWIq3XnJ4siWIdZCs8YfdeJQhNsvgJhYSYBaENcXUNyizEzHF3CnAw3GACEsdncZvuXM/JR3snpZKQ7z0XZnCPff9fQrKu4l37e1LrdUPlfOCEAAAD2QQBVJ7KRRUsOP0x4oAl0A+LBMl+eR2Kqj1SCD6ujRkxNmfBPY2f9mvNvzegBhGcsYWC4psYFuFNXYUsKM14nVa8WfUn+aY2vrFXJHRGUnfLpE4eML38qLi5F1Z6zXZSkUCNzPj4dKYJbS+eOiM4zl6cPnKb67VbSCeDeTyyO6WwuL03sEmRTmJW1A5FJsMiEP0XGP3AeldUtrMwSQq4HTQUCyErtgoyk+DOKJbF2jQQd9BMcgIbVuzPsM8Jz1PF0mI0F20uvJsnq+srFNQizxXhdWqf4dcwio01Hb9s0QsjpxG2sYSI0kvNIkmsDeFxH2oYW26YRAAACNUEAf6eykUVLDj9slgtN6WqAHr+wyhfxf/ceLXGKHwv7lGCsOZL2NV9l57cO9eJdOtjDlW6s3TBx3tDdoj2/djkL2e3B5iJ3bKtQDqtJSf9DglBrhSigDVCBNI0VjvbeKGrr56baqjg1WxdIOQd+a5DWKLuFA1f+y8zQDBBzooPy73EV90fuoD9ardiEs7eSZcp/npcq4fcPfSycUqLzkc6byExGC/Ygz1KaSmrMtOeZetU+IHmD5Q3/6M32I5adWqNnZulRwWvTkDJzUmfpVCvPmFMo4W+ugXYII11oJ3AA94tUOZQzfa9ufvLBOqMVABBwj3Vbx3CwqYoYlyegIoSvLs69/uyhHe0/7brhOs2QA+sEThNl85vn5apL40BASQOpgZUcQEVIC1aoIIlU0HBfn1V5Ym09SJW+JGR3I3eW/A6QzDmjCV2FAm8rCJmhCDuiNX1Z0U/mo5UplwY8AutI5SRNBrPkmT5n8TTlBWrhrCpvbD1UKPjnZuuD7DYovkJKrf+rIOA+4N71LncXCvfSg6X78fuKP/WlTSkIeWPpKO5wukrq6d/QlpqKv7NECK7ZRVhggWw86MyCVfKbOq//96f0HHWRnyDusFBz1CnqHBrbUw5kfErOQoOVfj0QrK1bNoPjVQjA1BMxy0EiY93wmbgl3downjfJMW55lZ9CytKvy9UtMggFx0jzTlI6OTpiGpttzF+iPftasbQTq3jlblBqJcQmjWyPCEO6gU8XjRbXd0EAAACtQQAtMeykUVLDj+F1hv9ZgxcOa64dB8fqUe0gDxkx5W14gM1FuWogc1irk/qqwLjxCHYSLwHE8hzm0l9BfvZKBQE2cjoW4NiIk1GER4Vuh/DjbrSb0t2kkoxBxZpqwZJl1Mt6XeuFyNOOB1zuGz3HIrc6r9vQR0POwKlV8+cnkRlsoK2fvy0xWD3pEAadpJSCSJc1uDlVovaA5Yc46mt2qZNI9Me888XC699HT5EAAAB3QQA30eykUVLDj6YJy2QaTKVHCkVuSN4mE1erd8/uIAR8L0XdmqcEB4MbGQ5cPW7N6zsCRbikfzP8DFUpSGzVceJQUVGlmFJgXruvB8FDf2yUIYvMfM7OfE6YxOC2+SgtCPqNuTnho6AZldyVur5YWwOrtSeSX+MAAACDQQAQnHspFFSw4/9qFkQKaLiHi4/YxLfbs9lJrU1Ibiy9zzlbtsJazyb3rbf+FrrzknIFKi5zD6u10Nh0ma84/kCynJmjPMktYBJCY6Itlx1ZldpfGDk3hLZ0CX74HFcqQENCB7Pka5n7fO7sZ6H5MqT6O4W7MfY8aSbTEG8UbMPlnVEAAABOQQATRHspFFSw4/+xjQW6+jQ+u7tWQurxmc22VsXBBJwnqDvevAg+EadVh93rf3NCqxiZRqv0ZdUAxH036EuRjWViOs6k4DlPl2bxIouBAAAAQwGe6XRBE/8m0Bp1xpKN1ACGJi+C7sM5JterQXoRqc8+KnjW5v2TP0Ib4BJ5AGfXEspX2xuTsFcmh8CTphYctfrUSUgAAABwAQCqnul0QRP/TxCv71tj1rJnxtDuPQ/J8go7P5vcl7+mcnv/pKwSpA/zisgrjK42OoxgtMFNyuYBMy06CtlYI1kobCsDfmaBjh9HoRgsHyJF2rtso11H5tL70FJbRqDjdzGOH6UccMWJJJ0ktg+oaQAAAH8BAFUnul0QRP9TlpL43s6tOlZD1AMtCBBV19b2oNWfZ+rG8IvIDP9qtoWZZAyEYShNVycTTHfBdWw/UEa2tGseMwFPkO7GLpvsqruptphZUKn8YXZ+ZPOyRA7UGtxygZ1XRP5J9NKFk+LL0CC3iEox3k5/mxtjR5dn6+xAQo3AAAAAvwEAf6e6XRBE/1PirS9+DzoDUL1gTqc3Djbn4C7hwclXVmADynOABrR9Jh999fr/nFwfWDV9rqG+QNxz7M2PscMy0soEO54mP8e7VbzB6GwHqCDdkKDVav3i/k/pHioxgRxUcS3fRTTVw1RAOcLWddhqTIFJhObC025TCMgRpTVpkI/c1rt9FhUisfoTVMuc9al2Of/xZC42HBjevGv2iYEfpHv0QU57AnwDTVHimo15xwXwqIXg6RkITfVCeHzAAAAAcwEALTHul0QRP1Gmqann3ZflOz+78taF1J+0BPQTJ3vg/TYpdVuaPwCxhHriAyHJ1s8PAKlLlt0Ah3ekS43+s4krA5aou8unZg2gSbbwBfpUBkyWMPX5Iz9A3Z/Z1y/zc4puBYNSS9DVHctVDa6fW8Kk2EAAAABNAQA30e6XRBE/UWQTcRFRsoLCiWEJUazib0zdiR/KbgqHbJMY+mTfDliBoS39w34DrrNvH53zyPoCV94tph50+Y7mEqkSH/hBI7M5SYwAAAA4AQAQnHul0QRPcdSyniaJLALOJFP4sgYaC8sAVKoz6DWdsyP2FPCDxuV5zJUtOAkPmLyWXlFeGVAAAAAiAQATRHul0QRPuo/WfqCo/OhU0QWjuIyi0YIoLXGeLuAp4AAAAK0BnutqQ88k2WHRxjmFPauaA9lECUvTAtnuQdglWmij0y5Zc5LX+T2pIIHyk2nOldqTl3fMAXQFhce/PSurKKt33Z32PPoJv4mkz/eodjWsg+fFQtgfqu3ufhy2KPSW8RPnWF9pc08ATvGC4eLuleBVMUin29WqqYuvG2ELm7bl4OMSE5XbE4k+XRNfa+0neYZ75ffC172S+P/Krum8t0NgBkJ7PJ8L+T0XlU3YhgAAAJgBAKqe62pDz0uAgI65bkeAAa/5kVTn2qkuiNVGdeg1XoFBzADlwd+c3WydKKquDxTHTPnzlzuBUnsveFzUMbiAXXSXy3of1CxdFm5pmzFxOUIt+cfrOpz2aghM9c1EyqTOxwKuCiwBVCf9IVUkcnig4G4ChCoibAZFR8ULMzNn0ZF/fCycdsNb0VRX3nMh4z9HLfF5QzdM4AAAAKEBAFUnutqQ8/9NWPeZV6sKhC5NLR8tLQfO45j411ChzuWsU/grp+nJT7/aySPfv5Qxlt+B1PXD2xSvy0/oiK4KI5P6fhX64GA/bwa6sOwG+K+4oMpY8TIJINLqAK+6p2Jl8L3/zLn/gzI+3ZvDXiszrjMHilJkgeJxIwSWv4dCWDpcZGpE7vSrAi0tNffIIsUJcDq4e+4fQimkMVDcNaRfwAAAAW4BAH+nutqQ8/9O2u3EDi26sFLdjldaXyjZHai9Fr9q2+qm+AABhPNcGy4SExPIdMy8/Xyoxsz4l/ONkMRDKWgFTzN4BqIrb627/cLeAkC9dn88TWM8B/aTeZ6ICb8hIkVf8l6osuqbuYVi9cF/wrNKUQYf1E9ndH2mkSYfN2FXhqRvfHIEAy7X0BL5WoS4evNCSoc8TV7FONAVTLnoHUApsvkgg5z1ikhDi+oCLzxoBwmmMfUFS73FjOBhSiR3VMbZ03S/GdTpIV1dEtvbHQng0mFNH5R5TB64IrlhsE7OHHzVXlaqwvARAuVfFZ7cGPTFy8cThUTtNjeTPgWvCIAWAz2c3Z4i+Lm1YrwPYlFNxatHXeKJV42YSxxBomZvyfrxiahZ4FNtcvnceV4n/ucuOhLTwGV0tRY7ZnMMHJcCSxLxDCy+MujElp8tQOY4Ry8IjPXKNzkNn1nE6dTPME6JmuRqY22rOtsS3ONdcQEAAABLAQAtMe62pDz/TW4tef0sAq+cM5jslJdzGeeyoJBNj1yVxEymtpgo48seux1KQRNiu9eyfhCFfKG10QA+qShEy6Q75Hzz/H/+EoAKAAAATgEAN9HutqQ8/02Qpm4H3WwpsQ63rb9mbZLpmK0a/Mt9CDxBHheAB/koFNirdKjIzWLZClgzx+DY8Yn156Psl6dfkGShEKUPAzm3rvMXgAAAAGsBABCce62pDz9NPfm3lgkLV77gWV8TFqdMfL+d48Ad3qRyZoyU2lUI4UFkBALV9vRuh+XserzHfe77jYFR5UtqHpPQ4w8i86kH70m0pLL8zak2RxRhsl4rdvFBe7Ak10/72j/Ji0khWIN7NwAAAEkBABNEe62pDz8lXx9RmzBn5EelobssvkV7HWAhe3tN0++eloT7DVrCq05wot7se16kXb0AOXu3rMpc1fV8rHcwnpCp3TDsgBXwAAAEfUGa8EmoQWyZTAjH/wFh+Ap6+y9hmHPaepAAIfsbdGRAdVtjNZ68TnqVmy+YmXKYaqznbNLAMFOvMj5h6v7wrR5LevzcD07Djco+wDY9JGJxgdoc1ZiWIiHYujcQmIJl5hhQ4AXES2SO9izQ7F+Caky3GzPzuorOtK+eGZLJ4+PeAeO4lve0iQ0qoqn4XWF6/E3Y7an1RR8ASdSJzQjRdpJ2PzjA6C+YVb7xMpKIFLKPh42IKEicXdW40KYpPgX3Umr7M16HqwWi/1eseQ6z+MJVSVs/B2zdbgkELAYfu5dg6TPeR/WMiQIGHr87eFjRHeFUg+vie7SttoMGCe3pHNvIG8TiOjBKOSUZxsIfMw8UilFI35Lu0YqwX0GZ8dZwRiPlukGG4cXHbSLE03RpsPh1RiXm7v/IJ0nflw6wfbAGUuPyBt26QDnBtyF2NNIdNBVYvRpjXffoQY0iiJF7KMegx0/eWKE1H7pIO1k8wPtXCzGMkhAteife7WshisKHC7LjI9xGZjX3DKto9Rc/ED54cAYYyKECW/DJgoZtKZj5I+/neDyGA0vqoVjGTpiuD/gcBG5j5yw1oCNAz16LZzG+wKkLrUwoA0EQI13nqCGU+HCB9jSjGDxhTtNtdrJ7vGoNCndMYNNvVLVdk9xGg70o35JoNimhaSPYta1r7fuGqH0T0dQkuUA8GvR1BUc7G76OO0vQvw/3AuSrvTPSD/uj/jgDIEnAbFg/Cj3PCgG1KFMvjf/LqXoSKR2xZtck6aC3JwrIrSLzw9UpdDtYIXSWHyPt2voFgnEf5OIuX1a9bsuiXin+/81wCN7U5kg8p6iO0rU0bxjbqMmN7xGgdbObQtBissyuvb1DssfI1IhBdB5uf7GSvMLS+xBQu5Fy4nXYbdVbJi46EwkhG+pMuMZG8Q62SCrZ5SWd+gmwo3mQYZ/1dE4BYvs8yYcRKR+/aExCJxDi3uRmMv2D5xUddYpBzbrTnn52McSV/y47fD7pauSBWR9Z9oqqJ5A6vnIxUAmG+7axlqoEoST4bseia7Ucq1zva+nSU4rg67lPgEfJAvQ0sulQjVHZhIbSufDj0ckJRqnmrmN0hTwS1+Ov8bQNgjdnx/4NsiufImE8mzMEKZm1MbMwAAcTwMmzfgglIm0CIwP1CuNzHQV2H9agFyI2e3/ovBJLmxV8wCAh3e+Wm7sNwkfbDGVlgNoydKJL4PEcxOeXssFjOWtKlpMTy/ptlU0/R39GYqJqSzYYxW1ShBUwv2YqEl6N2hu6Rnj/X3oclkkGDZA0tTgqQ8ddTOBPmgMZw/E4c/BmTTHlKPys5XvGhUtkv2eTC1FMbXKLFP6XQHN4AgkHF3aEi2WkkTBNHN2Vl49cPN++RScrv8kZxKkLpX15yns7U3t5xhZLj1ooJelepua7g03yDVQZFqb4vRlgUSUgSe7Lq8uwvc2/pvTqR6aEdAx/u7Ibj6yYzVpZD0UerVSEcCIJ2pgk5J5vKIwNepBkCnHCgzJaVEOjtBufggprtEjKHI70YQAAA2hBAKqa8EmoQWyZTAjH/wLpq1Y5JBknchXqEZZ4AluCB6f2WYKjuJ+Ws8wKXiRa+wMCVHqlDMfERxVmz4hThsuWtEQOwKt5p7vfKXKTiAOBkyeAAAG8jweRqEA82B2txhuEnzS88D0W3O866Z8yY35ZpdBj8Bqbd6c1ZJeddpzSezQGcFWrVC6ouYOx44LxV6sqRfiVES3RBUxXSLOyxN5JPULHKP+lvLJoyH0e0LKdDPZauUcqFnifcxxqpv94uy4dwDNtTSbP8lOlc2uFhbOcBwXbou9x4YkilMMWt0wuilybJllHiWJYkoUJ74pPtalt5nGZM5lys37l8LZmdrzS2R4YIPi+vyZHAfQiiBgBcJUtBfuZxu+6phuoZl1Jz/Y/btoX49D4r8EmIEFe5qKF2FibNNIynt6ieQkA2IA50yu9b7XYyTVx9IhYXAdj9kyJeQMcEdaSGCav4tPD/7uKLCkF9vlgxV4ovVzlNFN+HuPN2V10Vzvz0E8Cnh5u66le0gDAjWqNDyy6u727bii3DYDu/1Obu/heItiwq2F88gI+Iyz2ifE1xH0m5RUf5BbYTbrNyuufQTpev8vFoevfL1ynX4bC6Rq+VfGzinBuTlF8JXx2IBwdv1GUqwupLMPssoOHpADUGEq6qXotpy2Y+gyxo4mlLmnfgyy4c5a6AUZBf7IC538NSQJXzzFOA2WOW6MzMM2gbQM2HZz1PLbbe4efToYCup/EmuQ+TwbJA97qI9HujZxW9oB+D1QuCD1qynCFOnAKocRS0uhwxn3qti4bsOt1LNYgt1MKLHcn5BwycbrXpTxTK5sEfEQAkimAWdj4uprMPzOxkA+h/HiAa7R+jLhBNhM+VAZTaxhGCJU+37SoEfGnk1ryvAF8udTxXTdlzT7EEZ+GHatK+v0rkSj2q8/4nAZtFUY5rs2V1drMXeiVVkDMQ06WVp+1rlZDgEZPVd+jEDXE/hQn3MJ1aUlLldI7BMuc9ZGSe6UBFyf/aUipy6XvWJnrj0xSqd56DmUnaJxnxjcEG7SXsQUwe1kruM+oCPB+9WN+CPiB4D/OgbryAAAm7pZ4LksmhkXhPXRiE2M85V5ALcObuCU9ZRc6B4UfdzxKDtmbbtrhaGH095EXYzt0l7FlUzTkd5SNA1Pz2AwDxwAAA1NBAFUmvBJqEFsmUwI5/wO5ix5J07P89x8D2bZZMNwvIOXeWs1004II7Rs4n2CcLgHc7Dg95JZ8HJYChwPBAIDZGDgNw6m6UbKqz//oGTwrn1tyEtY3x/nGDj+yqnsLGnjsjU/gcnslf3T/I8zdHLTXQrdeaWebBaocM/C/BxRu7b0Lh66slOq8KiEJ8TCuiuBQiQ9sMHyhecJGBTYmXhg0wQ9+sm2k5j/eiREvECftPKn0YE8vnNwCc76gk0vG1riaih493kZujN+F/AgTyPcXQfhF7N9yFWe2CmCn0uA1Fy8GT8rWqSRLGMuiSUmYXhcaQQpUJrz81H6g0plZMnkegR9hmbTAxEokcCxMbKyrCj6YcJwZgd93hUswza2c0ByvquXwIOvOM9YdTtonzXMmIagl3i6Nklarj5O9+WRzD5Bo0RvYk/WwyxSWLDm+2E91x6GVa1813DaFm3azRGhnWuwPQF6+V0T4Bp5jZbFDXYscwFA0JR62kGdD+5esYcxmYGBCDwuIYh5fjAlzftMW7ifn3+SEXjdeMnSYbHYz6YBhd42ju/O/nxMtmaQf+7Ksq4/c7zK1ySferVcDsbPmcW0HxE0g9UWGvaUAA4Xfo1AuMMboJGesj/AW27xhXPKu85oJDC7iD0+9j81j0yeHHOf7zr+7dgtPdfjOm65DTcR6cHfstPVHo/KDtEtdFwEgowZpMEgSNkspnqkFxz8tMpIhRDYWybAvgZqpXDSXvNtZNFf4U0Z/L2UDOWZcmhrIfV9URP6PTkAGemMr47nnE7HB3SUO92zJsFUVMy5b/yDboxHo7C4fofLxNWCdQ927AtyQp9zqD3bBe2ea3Kw6APhGH3CYCyZguZViz6UoFc5ZO+G2cxTAkF9A0LfSoJ+6iD1MMpMYv18w0FcvjXo0o1L6nuddM9opU0Cwbp11W+zyo5h2b/qp+0Zez/e8Un6WWMknA37VyimCj4WSulgyyCNGMWLmFYs/o2Zc+g99CJHbreHJTCJ0WDK5MxyauWEiHP03gLHwteK3M2g5UjZPxminLf0xGnx3k7LzfggqkzcRE1j5g2tkBGbUslwgYiPo4CeYIjp8HpQf9sBxsRlop5XkCSPj2s22aF0Mwx8fhC1XrQAABgxBAH+mvBJqEFsmUwI5/xwYvWO0QoFbJ7qgfujykBrE4AXFucaAH2anwBav8dneHC1l++VufL7XI4ms8oSFWLOFb1BBS8e8echd56yy6lHFLWBguW68pTFonw2K14JpAgln43UGL7YJUhTeDZR9WGRAKnbkI5hJEbZPqIfAvv8zDr3FFH22m/ItYTOUtqSKiKQNvrtMKNQh05zLPZR2FSrSRl9PtNQ/K83bYoBnWG9MdqjVSyOY343lsK0xENwqNvSzYTWL83Z3qFzdscYre0Xk3rdIY4JWhOtF8xXuuwYiyGTQn6J2pUB7q1mYlQHdup/9/hOxMp+wh7KhzlLv0Nb8nX9k/bFD4NiFE3/XT79+8W+xn5Lvv/7aZdS2posDziCdaO/5+3CDkhNBFDyBjLihWxeyVJtmC3QVaBitKuPcUzpsPCOAkxqa47UOkGF9K4WkTGTCVb1G9JAlutvtLf0Y2c8oqg66GMbk0Ns8JIrijLTgRsvAt+yMe+a3RKZIWJvopngDSOaMrys5GonIjQT6Vbfz87dW9SsHBwAfu5x99VwR+vVKrcfXsvH6AXy8EAdbXgtB9m4GmKlfDH4k1vkBuZEGTDzBFDFSmWYPrsC7X1+YHpN2O6ccRfepzuMd/NHvjVFNvm460atiSLvZFrwBZUoWZuQFxON66v/w6xi8PxBWmfzd4/RVJwHfcdrRsO5lxHx28mKCB+ehgXrvrUJYmK7civzwABRsKcGFqfAVzvd13BzN0CCT7BW0Cu7KVcicBWRWHHYWnlTo8SZeCiWTQ+C8qOZs4OIboEgXB8cz4GyHDNULs1u8M8vS4uCcjUzE93dK4dMHdWZrPXrhLscg5GJcW8u0LXZkaj244RDteH9isfH5nSJOHcXVSKCAGarlXOYm82GRTkHgvHHNAulQT+uNJbUj/yJc7dCMn0UdnAsMlm/l7gazELaKS/ZhQpfBlpo0kxuHlOZkK/jaI9jrsz/bai2yg6MlzSZd8CwZX0YOlXcAFfH4xUXbpQ7wLCDbtnyw0oIgvKZ7HkzB8Gw9OkDjSALR1twC0k0SgUYLuNv78W2da4SmarVO54aC8MTAYHabp5JgzoIaTuNX1oPIjQALEhgalS5X2R0KZFrn1gdp+y2KgYa1LT/G8YVz1X8sa+BclYyJkzLoo0MaY3EanX34A8KmhmjBoJAVV6JatG8UcQj+EuhRj7ohRPEy4QJdz7RvOWZPy8V+zOTdaa6lcdVtdjKSPtJmOj9zbcXaKKHSaFVUsJKeFVuaNBQi0L6tmVLlN+EwbuiIJMttJnncf8V2DOIBzuDdjPwQ8TyB6Vx97J1cwjUle/CoKQqgRoK5ECYcbyvzvMYn91F20gnQlvW4Dple4K6dZC+8KvT8cnD5nNhQllZKe/uyANZ4L9T9U4L/YPP3PJNqW7w8PzlTx9Fu/dpnouNi+/hj+2NZGE0YSbETjGQkec+qdvRJ34I1Ag3srepyzGcotkBnhYOh9BEXGl03ItKr+al/4eSLZFC90MXCvkN8pvatfWKH3+bPrI1aDC/TQN+qWxeFJAhCjXIYiCGNkd3jhxiDzkwuj1LD8GeJxX3wTpTK8r3a8wMoDsDL4APTOU/MJoIuf/4u1rBwtpSBjF//C+L791nhEfLWe4GIb0W7LcfoLYYXJ7V+LOH46Y+bK8TuXPfodJpZCl0AcKsnlj/juPnEg2QddDZ/lq81q6S/wEqPYnPsT9MJUxnN7B8zk2ORFSS5wMYaOdpm4cXZ90ipcNIoUyb46tWq62R6oMOndHk0O+TSJEHJ2phEeZ/di7yDfUEyFI4HUpPnx20EPWF4CsCLUBJYpQxKKoORFmihvczfGVC7XGDWewSrZt3IGnWxpHIzMvk6N9NSMtGGC0nlUg6bCI5SOSK0AdWnTotjv8jyrtG/02K254MysDqW1eLnKyukNffFD+Pf/cF6+J54jn/PUIDdqj+muSIWyIE3tlNioVfcjr5Om80haYeLV6OHQTY3oAC3GsheYKYe3QtPZFxtMZDlXWuB6cg+ewkvLPM15GaLykp+iRfNqe41Y6sWgHkAAAJfQQAtMa8EmoQWyZTAjn8DoGCQAR0cBBn91k/w+v/qDqu+MzOvuT74WHjyPoaxwYLGgxC/3Q37BCZjobO84WNM639pzxBG+/U6sUns0qJ2LR4vlQuEbwdoGQKHx7q5Yawz4qW209uuYrTcVwGn8SnCqvwDkoqRPN4j989eq+rxxlU0h0sYznY7cWjYzYqCK9oTCRYam0Kg4dWSvi0UyikYYx7mYkOz4K92SUSB45EGg1KtVMrmVjZ2nkjzq1ZEUlLfRX95uqnKz/fqivp+gQqb6GU7TSxqWyTv9XlgRSzRFc+wp+3dKPJHrnRhKUN4mGKV8oAdQDHv1J3Tkfx8RgWxk6gDgn2EZcYgOHSGZsz1fYLKGA2lCwO+37YfdfLKiVDCWJpMAjuf4WcL+tEN9C0BR9HiNwcpLOIdSF5Q5TQSCkrz9CvYnZhelQsW63b/2wdahAEbOThvqHRYYS4vjXyldpQ4OWtsXFl5OI/tRbtsaW4dMl3sYp2N5xc7xocHdYnHWo6Qeqc7526F7WS6PuZGMRsLi4geqMuhh7lX1sw0a6izcEp2LxAcamzRCNRCnhDUS0Nl3EMd4fz8UnA2t7j7nPWARpf7Gu1wZVpW3DpeaE4MqBKfbjXDMvkXCwucmGxTPlUVlmN8TGDoIblMklH7cGYh/5A9L9rIC670rB7dD5KvBsyr82aC5EV3OddI+SQ8JmA9KZ7rmDNwODbLRdZD6cmng3uObzcsUlMeXxLHkrU6NNCh1G1MWNZxCWS5ZjXUfeKvo7zoNlfMmQxsWwC4MXOA5ABSb/SLdvu0tr/9gQAAAldBADfRrwSahBbJlMCOfwlErg5aXcbrPfF8Lhx9cY54s3jYf+DpD/Ip6lD9qubYoGy6HKnRRETUK2zuKMsynatqfINqA1nNzG/fXIbOJgl6fznR6zBtovf36mChMZ7JwGVB/OAAABPlU1uhXQZBjsua+2lssyg/s9YixPM55x4PHV7yabcYfCk5OF+IbDISFKZKeTkKWgCIvhnIG225kRudKswTIUSVXW0mgHYOP+kHX2NQmxVDD/M7VimgQDzI7pi6cqmhhkpWwjbYjrs/OeR+JpSYAZAr0Vg5wy98OM/uBfjijVOLL8CyVBnxTvfz+18NQID1u/HdwS01a+tMsL+UixkLdZE5bvqWJY2a5/Jy7oKT3m6KjAzStVkWoottoxeiS3iiTS1c/AtAIfL+XrDiaKag6vZidwpfOAOOjKjp13r8dMRQ9XtAcAQDZav1v+vQnAXl8AVL32FwZkEEnSfN90j2llx4R2NKHQYZTt7JyPnvbpFpk70cUM6Dp33kc5Ezr59C6Uew8yLxyA2VSh1rfJm0Xoo0KHiZ/EUzoVKlDkUIVWsoXLuwzeWlSWzu1NvgJvbkQvGpBoB4CqHv3SzgXjfE8iJmkk1cmXDOQwWZRWxUPprk/7wRsk85vAaraPkrlOX+dCDE6sBFyp/NvwBdq9pKCyeVmVp6fEncp+1U8IfI5ywE/JZadvkfKZwZEr/UDXl1tUnc3Rk4/scgDcBI2osJJ7B0KO8uS2jbvCHo7Md0U/TylcA/+lUsLmjtpNTTDeH5RxNIRolQTB+6j1fZ449mrnH5gQAAApVBABCca8EmoQWyZTAinyzol3jy8AZjRtoeGao3Rwahc3mNAk0DW6UgckNRNIt7h7Fb0y1j8WqfveQi8O+GmlCeE+sQw5m5dGBWbkAlRVuaTm9V+qAAAAMBa/z3qQg8JXKDA5doJHvxRMOUKGkJ4lfMrBO9MPI/wJvYn99ZvA+ND8KIJober+cmb0npla/ASnp0NaQGgZ2A5LiORwAZT9kDHtwYWQZYtbc/+KNSATlBqDd4grgKX3qPnr+hLi7MBVuIgMGZlA/Yi/S3P82gXK5ToJNmXr6xEsbee4EG3hCfTK6/QHuUR3l9+7q29qni73ODc8Lh5jftPpmR4O9tcmGbpWQGz9lLpY5j9vUZzY45bMeUuENb16lzns3OawgcKv0JMa2DTgYZeYwathlDzOHs4EEF5pJEAYAYVMXNRJ6ubO6LuAm8sz2DGN+NdQ3/TkC0Otk+q3lMXtH7aJ8tOZkxQGIpp/G86dFrD8Be/Bimn9+mYfyK9QfTH+hjzX1tlnvAjFrQak5oaTELS1+Z3QiHEr5oDaaMiQL6RVp3MgSLBPtYgwrucu1NY4k3ZwEUnyPAkIlBzQ28a1JDNecnUuihUsBg5OQiLYKsIn69cgRkjb7MrRQZRagj8AX2NO3Jy4N4lKOtFFkdPXdTt2p/AQ+JeZeclYxg3CX58q8zgumBVXh6x/H+jcADG68wv67MDToF3jwQdMnPCunskoRxQsei5hXjs3F0spG+k9qjfInReCn7+Hb+G4hFHD9BxaMVKw5/PFK9ODQNX3OKgvjRfKuwknufBgSNgwYTmm6Ghb8+pYo/5oL13rtYqN40nbGmOLz5qCUgVrjRh8T1GBp6N0Ajj9y8LMg7Uq2vmPk5AtynCkx2SQE9AAABYUEAE0RrwSahBbJlMCMfN54vz+c2eyEvgHdoYbtCYAAAAwAS48hPSmws8GSwaPh9vA3UnBgxcdVKT833DvuyH7kmfVtF5gsGSUeqarU91mqZ0O3aDjSfWMT5mSmZRT4f4sVze8UlFUOQ21M38fadxv7FlkpY2zRywxzWgvGWuASIkzgSBQdFrCYp4SkZaKGqaiBQz8qeTkLaEn9Fh2+EZ+8xrN1acXyWOUiZh4YuPbK7sdCfpddlf41EEK2D2Cb7ZII5cdA0U25bW/8z1JVqlLezCqShP7k7dIv7KHJ7xWR0VGeoPsxIW+HJwxkYfusyRyKCXSBjjLkcPsZ9gm/kaQaZnt7lvKDDMi0VHj9xGeI3M5xKHs3Sht5lTXy+cxggczwqiZSvMRxQMl99Pwz4r/aZP+qJiS6izNBmvRXBa6Mn+7KjhZNlxRaqYHDR0ANRhGXfgNbbJ3xYiGkDRYI8NE3TAAAB+UGfDkUVLCz/HnXRhDcf6irsoKGNyrTsvswc4WWe7+HtPHxrYmkW7mQpiVVxr/WNmzzU1Z3PYUNXgiQutawsy6s7ZCCQmYc/6n7VZ91jNhChxjhoob//Tod+roB2MxIqPWGvIEodcIRfsZ7whvyo4JxDX58BsB5NWR5WiKhyfGRtxiZ7lAMHrkINtOgWtuu7fxqdpr/qC9UjAZq3sJY35q/rL++N9CscW/uELFCl9jpsg4788L08jPLUOIwlg6TpP2TsCnJIQ0rTBsMnRyVkB0ZXEbb8mVQpC3bbqR6X0YtZqy5D3mPGXshgD7AG0X9I31d3dgEO+mxAhm7H4U6o7YkvILcLbplOVybqVvQ0AE7/vOYYE5ngebqfnFJaXSujqY9Oh9nY2rg7VNc1ENzBfIYWa3fNIWomywx0xALy9+MKBuyvsllZvDKZZh+/INC7DrxBvqpS3AHl+uszaVdoxWXbwzCXBJPRaRg5KpoAS1TuDahCthBKUvvneIB+LhWqNpbpLLL8wEJpMQnTGe64/VeTarh0xFecVfxQg2aWCyC66GC6uQeAaITqHsxZv8WCcKafWN2XPOjz098jlUdkFZUQ1phEH7pjYIKBQdqnjBaIhHP8BRQz4xxFEPTIC9G99iU9a7ejzSRC1JQhRMv9JoEITN2gF7GuNfkAAAG1QQCqnw5FFSws/z2WugjOlioPMy2V/JO2jJ/lEPR/Ats1LMe9BcnJm41wTHuM5ecV2rl5nHo7ttGRopm70/Ne+WEhdTqSvxlRqOApBBn+2yklYcFJFC7muyGpDWO0MUSy9UScHUVJZa58MGOwwwZ+h8cIfCbTWr0IMs0N+Hx9B/Mi/Qc7tlyoinYMtPt4ue0aB8sWadfe0QwZJoPoOUbBCqbprTnGCWVGFMHiqLe7qDFDMpblwD0XZ8keSqGocglS8R9Y4AwwgoClNF99Ah2u9Q8Kz/TyVFTcULIojkXc8sc84T9ubaN9oIbJVpo9u61f/oC6+D+RW0luTbOBkj1FbQ5UpC7CXFKvGZPBoPw5H1DKZdtmZV3U7oPyjTmQee2xKf+Ge9vKJCk68w7KQmonc/k2rpfJP0zdbaMPHnF7LPekEE16eHWri1jHQ4ZG6D5KbBp9b0plZ5KgfKetZSl0hWvIdDmURxuqqHCBpxR6Txe+2nhjbtHrKJ+3JPj3p3WfFUTEyW2NGjMoLNyB/LIIGBTN+gjZkjyWPVldQaAlnOtIWdMRXB2sldIJAN2z7iWx9+qNjcEAAAHpQQBVJ8ORRUsKPz00L5g0tAQRHU7x+e0AoxtK0zHK7X7qjjPks/hAc3sSwZT0TLgxBJEYQsB2t/SMmlAGZU42jQWqswEQlwQv0RNtBltE0qsh2dp2pYoRkEv4eNW69JQ2hY/pebtwy3MFyjK4HtArHRpwUKOaNtlAic9gSMlVE8SxC5oNZRn7HqmSGeDPkHCTPBfgK6CEfn/Jz8LOB+xytjUsvHZwX3UE4mIiA2hdgBs54C2FLjU/bdApwrRTmSCzYDT8ijqXVoUqJHBqE28FWE8bLunbNeZQ86xY8no8WMO6+/ZfZlrrnjL1Xqm6tjVix3Q0Yilb6+BxvsFiYI4eoFCPN0eo1g/IxMVixP2h08+a8VqUxzfPveIp6xNuU0pLgV1eo9yjryPgoS3qz38y0P6VOogEkEB0CBzGEV2+OSCFnf1rvKn09obvNpAE7qLTptzl9WfpvhTpYPfXj2LMI/FvsVbJ8PjAv+JGxuWK4roQ4bMd44qmtJFDZh0zPwoI6vmRvG9rvm8u788y7hnKRWS2BGapDYaTj3kOYXQ0iUjRVWe8aUwls3l+Aee396rt1lY9SdQDvsTUBP+bd0laFpv9V5YPU4w1LVcQnvpK0DDDBJ49cVrIcZD+NKHD4Gxr9fgzAHVMBI5hAAAEYUEAf6fDkUVLCj9hDxQBY4wukQ4KWULmpq3qy+qFAViRHh6RKz5g/kEZpfSfoX16z05nUIK2I7cM88LK3//CBkfZ9iNeh+0TOsQsjajKb7IuYrawcSUR7mKK6Ads24FnRbTb+D8hgzIkBnk8TUGBc2BznkkMetkJDRuVvnGUlk5LRlNwaZVN4mcfB+mNlYOAR1wzhOo7/nzjzqGNMOBD6y6MGeYme8Q0bltxOc6750QnwfbLdZvKcnXJuYWLQK3pXZOv/i9se2UbKfF7VHl+hUF+LvDh234SkS+7AwiSaz58w2SrEtghGYJoOV5htCwdPKVVeuX72ym6TvFhzzr9pYnkitKBf0YWB85QyvBfJHOJYzbPH/Vli/h2LhzOh3+qftCR1joZz9pqPxWfEWRGpzPHL8uhu3Jls3+GDNBvAp27vBXIVNOSreY+EsCU8YHlzzWRshpLNHm6deQdmrvggC/cOba9LRAuHkxwDlm9RBnqPGIV5h2UPTDbNymyacinnsw9ax8zGT7La1DeqEu77ZXpjA7R7yuuEyLnEH7hqge3jazudGYYWbGumuDEim0nwYDkhSTEqLLp8/Dm9BR1FXZgYuWN77v7iQ0hVPtTPFujSrcyrCjmIoMfKAKv/yCOwlJgIkqcRNb1y9XugFduUpWJOVCC2CVSnxBPXKW2FPS96lCGjhsl2JzuGv87jy+LUmqmG31gZ5kEWgujwGA6vYxeQHFZNzZfwwVS8EugHm1oKtVPslG+dQLA05RqulTd6wN8n+v/y+XmCCjKw9OmQewxiXZ0EJPHjK4mHlC2QX/hcWVOJIULhH8lRh9Dm+EgygNr5iTcrX4hfumbc2MK4tE9Yhp22qGNGDQAl5i9zdYNbjZ9Bty37Cry8AzNn8O/ca4bNTPPkxcKn54iOtuVNijU+jncdb/QG4AKy3N4ys3DFSFJB7j3prdQL/Bz8DwyNNr/QU0b8PsQo9tx8epewHKizArxrX4O4FqNm/+iXBHpl78dZSeBZUM2hAMt/yrXkf15JyMCakpiTE5CWRBdgROknTqgAhcmPmM2BsbKGe1MSG+oB+35xKIymGXn/UMd9ypAxX0aasry9Jji9aFdMbryd1b3dQZl/LG3GJBDS7M83aD0D0gH1aUWNOW0aI1SS98QK0lfzGepwkWmHRhnQjuq5qtXU1l+UAKp6vE3aiJqiGQ0YAOWP2fSXPJBbDGcwyNCzcsYoNSqfJZ0MSA6IUplzoo9uE1C4DiLTlCXmxiJL8IpYZCLO4Ptd9wUnZvqknibCLHmwZP0IXxcIezC8K3YGxTnstLyfaOoIAxxrT5zdd/u21go/FPj5X1XP0rJM9g4P8J/Ypp7UXaYq7fxClgyamLU+MeSWHVoWFPxJS/vAzPeixaweZ9QR3pXlxf6opn6woNxko5PU+2aSAi9mUhanteN6VFn7lzROn4IQ4ckwdKNX2GeEdf3xJPs/18qYj5TBxvf8jxpqK1MhdMgH6AxAAAA20EALTHw5FFSwo/YlsuR22vZcSJTMFXaPbJYOjjUI9CtJBWTvt+puO/4k+xIF8XYfdPWbLVp6B3s2ek8aVBVL5oiMvrtXKl/2hR8jgIf7I5doBsL2MmzvcdloTaWJzvq0MvFlIK6rPk8ZM4WrjABC6orzA26UA5xUGEFicxZEBkgNJcBcnFLmc2ge+zI8Ly3LtLhnrXUJHZbe8pT4Ek1ygSgR5qrs6coz/AWXKeMgRgxX+vGm/R0yjn4hnFCGM0EXIP+loIgqcZW3EKAo3Aq/o4kTV5pAs3BeP2ovQAAAO1BADfR8ORRUsKPh4OTg0l5Yr/aSwojLZw78Zy2NQJLPGqLUKPc+aOoaNhcDiBJ3I3ZecJZbNgMRChmPm8kOjpxooh5AFoXkd1J9tLvah3u/jagtlVF5HSGGUo08l3v61RMrxL5R7H4kBGtI7+6RlY9roZ0KWCnbyw/i3cOwrIR2jHEHE86LMWtA7jFlm5F2w8T/YObe1aEb+aVb9rZbiVU3nn/tlqVTjZUdXG14vUhfH1tbxmuM1Lit3CCKjHW86JZgRwAY6T84Ba3+ZF7/8ZMJu95B+NgLqUKkoOl0L8HuHoLWvaFeaIjo7jq5kEAAAEgQQAQnHw5FFSwo/+ZdEcxwvCFaCD5x4gR60AIF6N7JJYk+avGkqyUXE2AY9uleh2Pslw9VGacVTbx5oVGG0x7fwbY31SNIsuwyxucToLQv/G/WaTIVKyMxIRpteDIWTGwZTsa2IVuGT7oEUE3KdlrLwa/2HMy4KMRBXNwBX/GEu/mJ2qtwk2VdD9Qvt8hHQTPwVo7jmQLvXOtO0cSW1Ss7qTwN6XEHXCbIaZyTKa52LrTePpyMwd7M8nhrjLvZE/fQAsegVDEMMRQ8h5YfRJtD6OHJtG2dLjDYYnPSrNHgofDDr7xuO0bcqXCFAwEbQFSfAn84R0sU8P3ZSCVcRafh97bABUWkIWdnDXbB6SQkBSUesvXLVTqCCHCwsgIsQLxAAAAnUEAE0R8ORRUsKP/aW+nRqAHagTSzbIZXX3xFOau8AQvMggUTZ/N1x6d2j0PEHn3VgDLZ34NBLnUUcStYbRLxvPybrKSb3UcDm9lwcyNsthroFqeI+TVAdFZFMhkkIM/COCuec4n5EDiLwwUP/eMU+FsKdn4pVs7882XMGgh3pFMQJONN0Mka+r3i2nqSZn+qqDSw2fV1moy1EU8BO0AAAFWAZ8tdEOPItmjpM8mfRmga2HitLq6gEb7I4DuY/hcKNlwnn89xCzrkgPlFzN2AjXQ8dY5bM+S0mW8Qnu/CNJ1ZY7tgtkLTFW6Khl+ytIv6eOhxwBx8g7UEuV0AjUUOJbUy0A2kyHqN9/P2EJXIs8aG6f0W9HVcfabXSKpqFEsTc1S75S2XFKkFxPFoNg2Ul/ia3DEFeaKjFyi1fscJCxdSyfNqNbYjhGI+VdapvBHoqz17qHZdtlveAEwH1fLDRIuG+hNaE8q3rzqZM/OvDgeFw2+NNMvKwUMPNW9NXh/eMLrBH+S1DzRrUQ6yRIGOKNF4zVqq9B9GDWymeH8j4Cr6BsL/6f4lw38vkKFi5i1Zwm/VBvVSrOh1FvcZM8gVjbUke8d1Ks/y01F//Ya2EZ/Y623j1bFwimhFbb5lm2R/Mk6GI9U3Swi11+gYJkiYBBWFkbvwBJBAAABKgEAqp8tdENPRAMzpO4syNBZoUlfrbdNT8d9OTEG0nTdqpermB30Zo2iNTAAm2W4i9mqx8MUMB81Z0E9Bzd7fbAjcu7zNkrX6zTG/StJ7X9fhyS3rahY0dGtt/Sp/hH5cA248h2UJe/XXvGTJi1rN9q8/9IJ4XHztmmfowDafbMp9fmBW3Uf3wEtU8Czr4sg4jLfbXFVYIGGnDqZZVfY7Bmgtadb+3qMv6J4UIoXO+wz6n/I7v7fU0yxXfv4EocRCuKL0+G/qzX9wpvYLG3TMjhb4PCJjf5nTfv2a4CIvogwzUXFmLDweE5YPSW2oMktenWn8amWRqVWxXQDS6AASiwUGit8Okx5F4G7BCF8GlyKHeLx3mxb+XIrfdwKcCYplg0gfWfbWBbQKWEAAAEWAQBVJ8tdENP/R7KofZm+eq9+kfmRiRbwAwFhApuzFrTivS/lOCCZFW6VFM/Nz/ZT6Tdsec4vExNMB6nDPdFPItMifABg/eQD43a2CkF32PWtNzKHQK3BHxUL0CXXd08S9HstmXeKXIdQKm144qAvAPl8mpcKrOtAio4qoL8TB/tdOMhSHJ5DBLrixpJxZuH/Q06aKtJSOFKUtX+/pIpj1jPdtbhxum7GujbHIHM9eCYc0HZoKADQYLLjPPOF3ivWw+gnx3tTrSt3kWE6Jgp3+ah/gs6ZBeqhVfLVjIlXGqWdUmFmVja97gvp3wRbcsAZUPNWxZaS4q+o0jJaWgPxGAwHtdT9z+4htFlUKgqE8XDhK7l3oEEAAAJEAQB/p8tdENP/SyHV/uWe8x/N366N6mla0+iyMTnaOC3fVYTStLxepe09f+GTSynrHIL4RfucPgbfyFJ45r8O6VISNYDotRoGbcaFm9iDyLSJBVoXQijzHIUoGFfl2kAXyrFNDe/pFzbtBSVnBhJKYF2gVmTIIBiEsDfYz3r3uL5krbzvO8G6eDDGK95X9y1ulaXRIjpXdUyhOuRIDTQUEvbcKSbhUVDmcJZvY+2GfkzTNX8TQ2lny+xZbkd3tvMB1UK/fsN6BG15vyLTb/1N1T69SJlwSUEcRSA/2LuWuft2SIgqiqX6Y4xim7BN2MfQs+2aeP8f613BXbLFa3XsB2J/Uo668OYFcldKoyZmyVi3EvqHCopxT15wfxZRvUAZC9IIWZ/FV2aSvhsd6R7Y6lPZVqtQxb/xGHTMOkfrBjZ4wePmq4crTZtPTqrSxKlydoNmX9VSDvZalf80KKXFQLqaer3Z1xGW0X93X873e0UIVSuYrh7b7v5hPnjgkPr16NyP1R1EbVUVDRRMNW9ckr9nDRQXnxHa40h/e9IQGrmD8VxNVqnOuVqEvNDEOBZFSZLUdEVbeNGi/8K0CuRvktKHjsAs4aFNSteg5YieRlJmHY+8C6VopuO6OH/wZu4zqKHLsgrAY1180pHEM6IYd8dWPuYlZAZylanVkaz+QMR4bV2+EXu6/s8kv86ODeelHfpxNOW1Y1W4QFp4G+gz7G8z+o4k+4pCV1Vskjm7YVxWgjPkE8RSbCR9LNtE7QixkKeWywAAAJcBAC0x8tdENP9F8T7dcOf2h2gSLV1/eCmk3sieaU2alpXN2kAhUAPJXWEHiTTp8vR49owVeq/4BEcz1PAxkCM5yhjpcbUba74dFIzGR+gBbw7acV++nqJYwnojKBfezmarlmnHIqerr6KqiFDFSDbH2D49gofY4e+whGDDbq3nPAkTtQVZLwGmJc/VDoWPCCMXsBl40pshAAAAqAEAN9Hy10Q0/5KP4YKcL45Z3mHP7Bu2YvmcTzcsJAjPiT8rOv3jygTm9f/F/KOmyWZVuHiDnFMppWZ76/UZsVIiMlIt78oG6RBBPUrhZ/3wnYivz3+PUuwSVlkn831IIumUhbJvOjVhXnOy/Fb1u6bC8yiWetrE4X+Lhmi06j/4pIRRJ7B16p0eLxnWznqIKXLTlurMOMu9sZTozcLzwLGLUniZEaPAgQAAALoBABCcfLXRDT+kkV6aIZa6fj8YZ0lCkDWdaxIiYRAV3dwQ0vT1AEL6cXR0CxTx0UP4o30QoQWC4F48WY2xpYEJ3dFBAyvhWa0ImK75SEFoYWCk5ZW4ESMGC2qhi7XUAeHoeRmdIrTWTX9+xzGKIe+qyI4kTc8u4T7/JWhaqX5a2NFqo8JWJa7ihfj3giqALIpI4kVDa4cUMU9UL7KUu2R+JCJfitJv5G8RedEu1h2j86mmo8jxZVpdvKEAAACCAQATRHy10Q0/WkXjx8hDE5DSf96/uPoe5L1UFELTgRNf5yZHmZe5SJ/jbW2q0BlyQioUkKoGr4nfFtLgNsoOKH3DbFDuITMOn0Vr2c111pk4OlxmJXlfpWQcAX0D+Ef9vBgp/0JslASsZw+5TZHbXE7yhyOV/oFnOzVba0EQLwABbQAAAQYBny9qQw8fv0yyV9StXXQNkSoq5pJeudgpKUyX5aHLPD9gghjy2Dts7o4IMkJQVAi9QzYYbXH5OncxekvKX7qPO3P86tMd50Zc5We41PudJM3iPIgcTcvJLbJJsYmu1bqeQSLvVkdnyt7pSf0D/4TLHPJix9eKuBWPLqu9cmjdgtvqseRK74tOsCXZqz7Lc3eJn7iUPebBYmVJATggadHAHDTTsvPqE8w48c8NNZAiHhMmfth6pKPKuJW8JtbkiL0IcLEwA+yt7RZ61K39cZP5kVjtaV8rXaQ4HiTkfZEBo4oj4laUqipktpK2glrYixHdgWNfFLBGN63FXZDJXRmRBeQvKBQgAAAA8gEAqp8vakMPQKz8WOCwRUWazgxGR3ryGHBlTnaXUvEdmA9U31+8TfImaps2pFMTdDN3zUFEy72GpGbIr4oF5nNQ23vlR439AFYYud1B9dOg8hV0WZc42HX2UY9PakSTVFlm14V97A1XfI778Cj8a652ho9rIFjytDr2CQaM7ra9EuG/CmUTl3Phi3LQ978iEL5fJaTkigKJqHJ1Z5YH8Wf+Gt8NOXnDAd4L1ExQN3JXsUxXxwnmSH0UL4J//bMI6SkrEiIcjxlMseyzlGTz5DHNgXZokRo3zr+mav/4QxhSmK7cdDz7afmAoT6tPgcY34iAAAAA1QEAVSfL2pDD/0Prgrq1yMQFQSvUGjuXib1ReyDQ3cYW+6K8NAkAQO3HEX6k9/0JJYx2BEuWpIgL162ipi30B6CnWlQryA9qOlTm2PyKb3Cy8XuTtcMK7GEQ8zn93uqRBmoJElt5mKvA9fNzhkdl55dq5OZSiUYnPE8D3WKS5Ca8z64mbZG74tKWiUTkE0YNo/nU7PJ/GR40nulC2F6ndQPqdtlYcLaSAVnvDG0uFwUNzDUYhiLel147aK4ZU/ynM36+eLyRpmU0k/NLWTu90jzcUX6mgAAAAgcBAH+ny9qQw/9EzCPcYPYUo3hAGEVWANUous60meWthrLNY8PlsxsS9jfpBictu3RNqnsd3dHVEQNVl7gQLf7qF8ilvMYJPP/yX5WGe5OOvRK+pQ0+N0Ad6ucjPZoT82ZKo8o8LfvPCR0c1FzWeLiCXXvS84q8tyX5tmSS8/ItMl+7gCAF14GxGk+hWPjVJg+86Lx3QT+DcezywAlfKr2QPE8n9ofYzaljkvNgXxq3Mn1Ks7i4DpvwhYxuTe9uQsllBZ5WRVEE6gK0P57ctwoYcQd9Ah3HCQMfEhouljYZD//bkcLSseLD76jmd25x6t/kZO2/EHKCOtcJTn7HBTnwjiPHQmaBLl9FTi34e5kEhMG7COBo/IfGKSQgk/TirHLNNLF4eGVVAKvUPYIh9RCN/DIzFbn037TJ5Ns+x9dYDVR1JrOeEaKU/M+fLmifGQbaUGG6yF2I1Rym1i5B5kc/gYKJeVWxF2hkx5+owmFVCFvHl5y0NCzE0uCJ1G0pNEGUqO0L1/eF3kadXdDYbnGta2tZ3tAkJHThvONTUPaI4Hc0JkhexK1SH+5EXa8kdyd27XvGGqyv4DRpCJhcPBiNNZqDS5YEz7Qj++pGdjkkuadaDw/hKXALngjXJDft7h2c8/MXbXWg+P0zdGpgMDiUq0ya0VFxASQAqYQRcysWkNdk4mO/gOYAAACsAQAtMfL2pDD/RLWXwKG7bWy0Hp1OT+Im+D1JvkRm/TQGvdUthVllIJPBRjP7X6gt7h2ysrks1VnmRF017G8ZQafOnf9dhXWfEkT8IuKrlL7DfIbqyl30D9nIWBo98Dl/ZHisO+z4q9ZB1jQ8RJYeTYF95jqQiZnJEJCw/eKywbGdH+dTXyL1wyWqwuVag+MovYWe6Mw5SPWYQQa+0r3fwwsowX/GzlLW2SuxngAAAKkBADfR8vakMP+RAjaDT5Hdo7GqubGyp5iAhIWhN0jwjKwTMIAefi4d9yK2CSDa65kIw7nA56drVB9Q19huWgxods8LjA8/HzAoP3JfuunR+K/GLobsYu2p2hIGFXC7CpHUYsaxZAZK3wNPD3WsBc3noO17M3I2VQONmsbytrQnHWh8eK5xFbiQHxvrukoi5SY6SfAPR6lHYu5Dx/uIlC4tjVcV4y7UKXbAAAAAxwEAEJx8vakMP0Skw1odwLx2SDneamUwS2qbqQPOY/YHu27owzDjOOmge1ANXsHWVi0/4uC8b65AIgaZ5++58ic8LtEMvaVhCnUxJqpFrqR1QS9URDoRe8rJR/Sq/oqKdZqsjZqm5MUhnS9cq8yZd7IWHPVcZUwfq2KoHSahHKWAc+WtAiPywOCqhmS+PmN7h00jwZVQFVVq9IXsOEpqzYUfpkHsotbQxjxe19PjyhJb1VqWvyZHTEAuNRdqDt2uP1B2K6DyTZAAAAB6AQATRHy9qQw/JZW2Jf2qJRwqEr/uhgzxrAwnlCJmX0jAh3cSe59MQJzx1sjZAlnYciiBGLkjBnsakNxZc40VXoh+3nVQ/lkQZGBqA33O5mem8BPE3h2Vscf9RgClRVco5D7q0ItzxjA0vskrGqxwYCQyjujOpwEsJ+AAAAQNQZs0SahBbJlMCOf/Aa7Z4mOiY514wpuXNsemsZHhe1c0Dd4uhYJFntBUT9c8ATlYdxqRaPGXfE8jDhc6husMrAvLhXbRUIrgXzZJn7mVbvu6nr98bukYcvnyQFpXQRNMCDGzxxvjUNNJLsvhCY58hzLVnbqrSMZvZqZLF5VZo3uDMTZsn4qk62Un4NSPMXzzZ67QBd0XYcncBuSZaExBlYYCjYP2BD0ZXsP2sHJpBUTmoLXtTCqj9x0UgvT6yYgZ8/jq3adQSDo0SWbNVIL819dXAMkJdYjyRmLtwEEIp6GYJBIt+B4SqN9DvsirzmK4kD8FQ8YrSO5Wi8GfgxY7MxLvheAicliHxmw0po45l8oQvgUqBI6Kf9em7kCtLS8GRQ7o/fSZIz0FdaGplvvaq65yZYjdDGpMC3phZZQ25YnVZwGEN3tOfdq/chfBLS2//+vxCcU2W9ohIdfJKPG5uNSnwJyHi8FEVjtsIpw29Mug7gENhiOIVSla5VzJcrspLR51bR+jr+hEwxpA8i8BcOAyPA83QKIv2/wTZe/1PnVXOVbOj4dqisaxUr2bjN5v6Bgg+RNFk3KnbwD0UOWWrLd8itn1joAnIV21VTkQnS6Duuaw0jKzApdYlotUJAVOkr6tPUxBRw10EupyVoIC5UMbsIVwkgC8DyOT2lYMMdcR1tue55/10z/4Fg9WOtKHdjcGUrH3fpnI0NVn2s/CTSiPJxEtGrYiNAfxawpSMZyGouFxL33UBVhHrnhkmIKu/1e4UPXA8Gbfr9uUqKMVrLebYtfy39zCANoYuDtf5B3iukWtnPhfJKbEscfjd9ViJzZKvcjXlcYqGK+od6DXSexNhBYJfO4VnEI7qSXuesXlHyYiVo6uHIAxjcysJv8HNoBFZlhYskyFZfJw8Efwb69ZlIF5ldyYtDaSPGzZDAILqHMl4/d9WUBGPcZJ6dULis9bOXqQNYsGJhQ92wn6JU07fWtDk2QU7sUNbPSdXL44R95DpCVgsyMaPSrcEREKG/FRp78bqmeSezmKlHmHHA3vVpaObjVyMPyUO6VXepg+lBz1oAXRDQzwmTBYV+7jwlJJyR4C0p/o6UW2UegvkNv6pKr/tU9wtVWgsdfpL1HEb5SFsHEbJ0jrlnveFrkmU2p1PtA3J51k37afYczU8fjw0hdD5ff8iGuX3IHmvV/Veh3pZA3prCe63Cc+VSzjVi89/WfeVPIQ1QU5TbqRYnBfokpgimTPQCiz5t6iEEMqaYgKUJIxBDWCKTwCeCvEHnk3mdsT1RtMxk6icQIyhNRA5FEsQfKR+cWGa8T2d6VlntMjAzmyyIr0L8jHukJ9g7oi3w5HXlDfebuQVsMaoXATozLzqKRLPVdFb4AAAAPOQQCqmzRJqEFsmUwI5/8Daj3TvUks2QVRQNQcU9xb0ZV770SC3xgT26bbUnpkERFPx8DO+eFgOFVwWrIfCosLGcPNSXmao0g7zCSjgVj2S/ilXnapWLtwO1zDGQmzrUBzSCAAAG6OJPFv6mkAS8TeVDy/qEvCVzpZ+XHE3FCsIBNexKyd65ySuh6iSow6NvaXAlyjT8022mZF31FNGCywbJ1KP9tuIvlx2fVvYGJihpt3WeU7//X76VhkLJ0kn4dWPmN5CFkc6wluX5sqhASzUTXIsTpiT4g4idIBidBtuZFDQ8txUhYM0i2Xuco6CqNg9LcDFb2HIv7pEL1hBSlvdg69lDBUlz363rkI7XVl6XsS4T27wRy8GTsz0L+pBFoZOgD0Xm2Qm634LuBC8OUDMtNwHz+iiHtTlAWD4+40iXd1PfNTwZBQAN8Su/zV7emClxZiJKbe5Bmbi+1y8S/Zshs9g7iBSeiUgh6h2r8cz5eiWJ3D4dRmUu3pT6tGPqxMM/1luEY/8+iBol8glh7+FBcpvFdUAIAcLgnFHHnanmz+AB/LKss2o3A9KX9Yuv8vf9tG/h7+BgUOWR/xu2OOgK7BX2pHKrfoRXPmu/kZxK4SatfEz/4peT2GzH6Ev7dkqiG3R+kSEXCx6kwr6S2RQ1KfCf+pUmYNjviRBGYNWrA/MeI0GSCLf0Wc5dDwgrxx6ZdLswlizYxpXtlyIaPjRy8XkIgBh1gHCzdHs7E7Bb1n3gtS3Tq/+zA7LkzFPkd8YRYlhK70Zztzm8zPFql8yr7g8MLcUO2sBKd0OqQUVsRebFd0765NghuXXItnx2Hz1oVPEPI1QFTWyV6OMbpY2HDfikc8/c0WPCsHdJEToB5fb/HRN/U5nLMjee2YwFn+V/L5pHwsXYBVF6tiSYxDau/AcCwa3anhXiVm5epAAkvS06q6lJpP2WuhckWMv4C2KKiI3M+37RLF3+HhwcsihvyoJI8G5PbbP7PXw0mTUzyA2zZvwAa+CIPYtWuM7DBdxyhp5zHSDYYoCFshSBp3yVTzZcc55pzcqfQqp831n8QdC36qx5MErrqG4d/jwTxCXzG+wZGKyGvWTSKEflxch4f6rGsxdSjy5zA+rwepAgesW8uZ4MWyWENZqSIADaLEcjLrbTseKTZP9FX1IUGPHh6mBtmwLh5f+3NRtiXuOQ6OdaZjOTLOubw19FznnRThg4LqiNx2maJuimnNCUfblV6IHedo8MAJsC3nlAs06tdZ1pLpgnW7+bs3ZbqdssSAeTexOXYEKaSbOZkFrVAAAAOLQQBVJs0SahBbJlMCEH8EipPK/sw9rkuNdphTSPU6R+xA5ZbHWIymG8BqD39eQVlzWDFfdNHSg6srts/zGLGrMXES4zxsp4S7pQvswIKZ4KLmpXjUJ8LJ4fPhdPgEhKNAAABAdYbhc8igNeZ7V2itCfx9uSfRddj+S7koEwn95DaGHUg7uHt+trBARP/z3CsOXdt+luyoiW94vuS6vqFvECNeqQ8tXS0lHkyObUB7UrVIharFQ4Y7y0ed29LMym36n/5W6SO8t6BzhqdVDdpa/ib/PW127IfoEzs4Uj72tSvZ8NkMS2mAGG2C3jSvprGTa162/Xg/EUDILkCtPeuXDlLuNvENLS4XxUaZlSdSrqNX5HnvoliFCDaOoqjKszmRo1Qb2s72fyT/u43a44yU8e2NGfBFL3yw3lMhFgpPhz6P82UlcDdJ5NVgxUKcVR8D65G1dPhVlIdafNwOg+9sIacLmTeJZWz//yp53FPlOQQ4P0jQmOkwpWOSBdTomWSsG99SQwFKkD2ZDEnF60ZpdR81+FkbTpSXxvrkY3IF2rbxTCo8XX38HoaW7Y/QJvM2Cl6QpTg109//HkdjvtOMs9g1wOg8O2iGhFsfvXq5Q1gTbgMm4ThjxpAMDLGQ9j7mVDT3OS0yZyDpAghWgsGfa0smJiV5/s9E/ztjxzXjNajUM0D6UgjUqV2operBBA0OqND2be7eQGuOqdMl+Kd15qTVTFBGcWquiCHFL9oxIBLgpD1buFlKFa2BeS38bWYgy88X0rE96vscAM/gQJI+AsFc3yOuMzwv6RZtXJrdwZor2NcN8RABIxk8Wvwz08aJTxNdneFrvjY4YG07GJw79a8LAeS0QFXu0hWeEeRm+e3sehJFLK353M8DgB9v3jr+tlaBz3XJI3HBTCL6eY6pnYrSWdW0cakR7HOq0F3lkJOxetepNW1n3b+xsiz59iu3Kw6dNIH4T80UQ/dUUFI/1s+89+XAd7lA/baXTp+LsSAhzvN5z8l2X2Bjkok+FTlItVx0UqHCuGJJ7uqqJLAxK6/c9TCJ3bG8tA4rVOpMPdp93QpGSFihTtO6x1VCKD6VxFvLPBEi5WPNsd5jE0yb3uYyt5GhXEflQsrFalTAWqRN3mDjUiOHf6sGXSYJRNs4uCCa8k5Y9d4bxEcf+/AafW5bN5pt8RE8pHO1RiDivHTPJQ3ssMoOIGxYQAAABXFBAH+mzRJqEFsmUwIQfw27YdMU5gcl9v+shRZwlxMInZOoI1UrKIOlB6gkdIdHADxTgjE5lD2Qh60e62x9YDpPckLd37J/w6rX6y63Br+AKofHFJ89S4wUa2XfcY+T7AKNcxeoCAhBWnQqGRESA0rll8NNwZVN1Kri9hSb533OTix681oVgAAAAwBSMENVgkXiHbE/5UM1/x0B1NiuMdOmXdGyWsACK6dUgZAVIZc+1CBhoYl5gjucNpNlYSM787U/tUDGcBgyMW3yuEXe6B9WMVUBy8jFOrnPjBdgjXyIm0T9y3PqLKGfwdAYqTF9YtNkm54xJQRg0KZiWDcRJt+n7x9jDBYVnQ3MTleVZL600sQadKmKNFH0mAWI7EJ/S3QT6Oji0W5onX7qw04ghwActczGEYLu+aNyGs+HXgqYxSKxNCAxf7V1+AsRmDDhTgX1TFENToUkZflyad3P/WgNNUIfAqBkF4AGfj0g/vW65sKqQBJaVGV4WOUqdET6I8R6em6/BOnDD2dPwZUmFSajOMv+POCEUodqnOhcEb5rnfeRnvLr/d2wzH1y6rlYKZ/61Az3ViWCDbVIU58Zj8fZ5eoyY5Dic/CkDfae6bWJhXwIvMkyosbZXBGQOc89iJB31oVLxwsZ1iNxW691NkCItDWRXAzR+gHSyJe/c5/DJLUNPypxE9DaeYWtvokjRUsVP6/esIsLpFA++fnxadZSkR4i4ydkteEOpe3jOD/KWlMlieruMRX4ks+dJ8u30hLxqK0A61nvMhC2m+FNREmDIUalBjdrNBEqhS8FCe9QDAfQmk6Y2h4/Pe6851fxSbauQB69FDS4z8wnjMNqPzcxnJ8K10BqPb6ET1hO/lXijqP1Ns4344jeXFACvo23A0Vze5MrOqqy0s8oPKQZ3d/ayQkX2G2V1U96dxoMAU4CLyQooxP6qSIkXZ4EZJuE7jU+CPu3cOfUxjdiFjaW8hc4itGyx8LgfKT7FZ0anDbAvuIl4ZszNXfKEEsuarV/45n8OtusssW3uhhbZyBZNKVzOKPMVZPNwWQw5bH7dNYL9q+X52u91lZtaUYcwQflllFRDnkKisk5Fwshgakcs0UUejC6Fk2QL6IAdHb7zOR2FhUIeeurUhmXuPJr1vmPsRKsYZJe7QCk4Hq2MEkG48dvYwraz0rXg7sAi12pJIb5bey1v3HjCWkvDoima2GM/E0xK4FkZKxMLuLf2ttc/jl0e5QAeYuxENSNFEzQb16NouixsZeO/XokfB7g6oQI9jRLY4p7kmTdO+33fXrTUuy4p11MxdlgrFjYbrQ8q61WSNJpLZiK7iUQQCEeiV56JXdrhIdcjpNBCIYSrSX3cSCOhqkuCRtdxOJFrzHqJmAuo+MbMHPO0MVYODuAG0rihyfmeSgqK/vtwtTTen+f5KbNvX0xTeWiELIF+2q/gMVWsZ3iQ4J0C1l+4lvD4sDP7XXkIuvXYRv+AIr4dR+x4UiQBXLsl122/3zyhX9ldaEDgjOGeN96YOhyoMoSqPnOoc0Hcl/1BdWZRemB1yvRq3rvAkX6wXQF/wo9J5X2TEqHF/5G0VZtzq9TE8ARtqiMZl0KAdLLBvTY/Fp2+QjGz0DlFgj1gtjDnw1yZ/CxYZRSebGalv3HxbOOJiK+GFyH5VwbHT4RRlGFW/cfnUA7gJOo/6t0PlPw15ArO7dxT+sLUAX9ei4aMOcLVCocXDpMPeAXJILHFjpXPa0EaX+fP1bVSBEX0JpfLo17Ndlbxlkv84vSAhwitGFPaLIzKkekjAch60JB6fa8BtQJUaizpRR7vkEdvQrfeNkhu0eI7ut80I1+denM4QhiUAZMCrJ2OqWTAAABw0EALTGzRJqEFsmUwIQfBKymICVpjMVciYUTtJoO+yZ231Bl+nwZlWxQLZNwCJCqkvnMDisA94BtW2Vhjz/TDY7Q3HFZD7HH8RmIXk9DFljWgfgWBJ0RgeAaw7sc57II6kIts3fYItV47szieYqCiSb/agFrrDsD0wvaNO1kqqvdXmJrlN+0LJC86JqoRSEkw5StU2LSbThYy9sWjYt1mm2rtQeFFsMo0hFmNllBGB/IIEjKSjuuQP5jxviFsElh+/EBgID4Loi4v6otRxRxWWDqhBa3XgBcIfM7APsrtvkJ5yiRVRJCFGx6DQtGtKUqPkue6kItYbNDrDNMRcLrjI5Uz26vuf1Xp1IG3Q34rS2i2AJl9I2JmkoT79LDn+CIjL8Bli8LESvm+4WdUT/ovwKYwjDNiyhVjTCzgo2DOrcujKCaVWIstDDPyCUjnBjRd4g2t6KTgho0BEa06c38yIQ13c0V6X3urro3HBlDkLTjIIgAeaBHjN+c08nzRzRlT2YQh6KVxTlsigpiMYwwsXaAdOSDYRz9DirsJqlLw4EjDn3eb0x9TYb9sUrvmtk5T0QiAmeuEeB6zzjAf5M8imjk73gAAAIsQQA30bNEmoQWyZTAhB8EnBDXPFyQetAKOgH6b4l0zkYEXpo6fijo92CXbbJ8qrPsAAAH3bJUOHeYGcMzX9492lUPaUTq2XJecKnkghp2krohwazM2ghaaLwlpQ0y57WQ+tsDd0NIlBvzFif0lUnI/t92O9+NFDFyutZUF9nKPa4RRFHppJXld6EGsQEP1S9ZIbz4QhABiR+6ksuMkmqyHhIgi8vUTv6uPOJHVlg3kdmRnrmtWPGNif+72Ve/PlcT64jpuhrk2KNIZpd4hq0R3QXZwhmX1q+zTj7chigrg083TaCEfPo33jIIZWjuG3IWy9rkNN5ymnNx/vQpXDY4siIcSkrv4BDQ3KMIfVwrx4GVIjrvlKEw6Y1vdQu4+8g8p78hcvnfawkHY4wiHUrL1ymGfW/K+9lWXSrpqjJ8C/FlwppP6njjaDd4eYP6YB2Lo3dfjotGkIH2S06oU0C6EsV/UWtYILc2pqv8JD5I1AE4mCoA5RUVplil3wNiLstqJZwS00hU5AWEmRqOxPAJt3ynDD5xzFBileorVF6ZyY/wci55G0gL/aYA3U77PZ8h4K1TLmKSlQSCo65LsNVyznQuklwgQ5r8q7oQFcxV8mVjh3tzzEFk/HRaI7XR4c5mJYbwkuN5oO7B64FLR9GoaYdXoTbxpzsDASngw4/6njYb0EJo+S5jMJoINhwCIDhusnoJtQcv8q9gZOgIJBG1tQlZ2ZuCwuKaaKN5aAAAAlFBABCcbNEmoQWyZTAjnwbhLQAdwpT8gOREaJGV3G5x8f8ehaGFzgRFgOPXGiMY44H4P0dSSIg1aGgxoZbRIAAAAwHGNFTQDmYnlC/IR8vOb+7CjZScpqs2LA7/YPAYcRrpw6vbrHW2XF0tUEkDbLOaUD4vS4SZlVIpoIeH1mFnTUrhYAWj8+XRjbiB7hi4K9zihUspv6BPnE1CxAdH4duXXOgMuFjMxY9PuHrW74ayId4rJrzHSC1kMjCjjnDPVvu49kT4zEF03FdSX4xTw8/jvTMdAYaLD/Ou2DWm4FQmF8SL6sEprPX4OaaZMy5rx2/hGFaQQGSQ3bxp6fnxUyUPkmGXxfB95gjk986uk8cWTuRF+04He/7oQPLzcfRtYmwnlAr2CYm57MzXK86Z3etZGVT12XYP0HCLFTVe/KiYExu3FR9LViCP85lH/AfCxMwyTdntEpJTX7u99Pd/aVna/p/Dh0Sw/ZfJC9VvwyL5TXOl4KvVBTBQkIE6dnK07S3wnKF/bxPom+fGx0MMnJvHhEcRCK46+kWkcGg5yRh83fURCudgrVrvHIknteBY3ck+dlDvQe3diW46F0LkP3PKsRjMOcVrafBHWRvusb3qrtk2rVPq2oo06uIC4FljZHisaV0EA3RXNtnAJOxosgAIfwk8m6d/a9/+s1W/uERaV1u9A0hIe2jqeUrchYFGU1Egv0HZxfQzIKrijQqa9W6JPeCbIXq+xf3XfEmLXX2ab66UvOk+g1xSfm0XYEmts8eF5q1vw4Zfkro1SuqWx2o/oAAAASFBABNEbNEmoQWyZTAjnwIlUbQDAJIYfeH8kBEwqC1jHb8clDIrnOsgDAvy1T/86MnDbPwwBUwzr2x4OhXuUyNHXIs28i2d1rbSub1kAqAsdGdNjZaxQL85Ls6BWpyKM2DujNupcXFq5pGmOoMUoFFWhuz/o2QX0xtS0unUnIGFadbWZMIfsT3QClbxIehbcSK7r8zg8kd5SrRfVkQ9HiAOU1neJVLXGgl+cQULp8BCZhGLyj6OTwcRyRJxn7ObXDfXR0eHRxOyWc0ht2Fc+GwLRIs0P+g+3MZeiMOdE5nEGTS8OSe3zgQ5pY4VV7pSHUekSwSvy4ClBX9x8OPmGvY5DO0w/VM5NyPsUUWumausRY2JbFWX2R8/q0iDGgFodXBIAAACikGfUkUVLCj/HZCs6hEV33DQAWFjctvGPs4mVWUYZRgkdc+VdZOmZBNrooiHuO78VEGn+MOw+1z6P+GCTLSvZGYiA1HNws6uUDe2QTTOsKAOXgmR1QyITz0aEDFU4xnKHkTCOHuPmOYjhPO0aFP1KaMOktlaWrWU1ITVEb251jcTXV35giPeHuZ4pz3sLw9F5BcHhzodTf2nvCG5Pt9VUnxugbiY099PBn0AmC5HwL12hp/DwzVzI9zW+mlaw/0EqhB7r7rCdYRd7b+lSYKOJAZ5BccKZVsMXiE8MXD9M8WaYOphLFspUJsPPdjltIWRMu45VMa33RtK3v+H2fz5aow1qvNJYIHMvyNI2u91tiN3CghZVJ873EajVa+wz1a9GRZyagezBr2UDBrRqRYCag9Kqg8WyDqvE9o9SOnFbNNwAPO4Z7uS6LbPZfmBvFbtUAp6gJVIBJ7jkjJpEs27DN/2fcnS7JZcw1hl1+DLbZXyhAhPp7m638NWXN+L/+4OnXqdacqjBBx05jZWODDbqUKSJ9J7qZsR5rtRx+UAjDqoy7FrugiLX8A6pCzMu6inq0087HXW/xwhEMQgKeOOwl5LsqOuQOgZ9mdQiaNgNO4JnBOokyxJxMJDmX97wLj0E/3gloeI5GeDC9RMI2al+gP3dJniDn2Gnrv4L1rnaDQPBus4sYtun3cyBQKLmUGjLMpy5T5Ade+kX4TcQgopDXNkwZICbWVPmzA9AVK/D4iRhjG9Ih1m8dIclP/Qi5c6HY5YfdMufumshiwf6SPJiknxe+0RSzjxGH2/PPjJwQVFko8hujnyKSU71YtYx9hM07Po22tA1JRoL3UnDaghmfqfn9kMGQsPXd1BAAAB10EAqp9SRRUsJP86W74ZoMTrgy/WgF/rv667a8siLW7I1cbmG5QAwOnPZUd6VghavmERMjYjVujQtoMZI6VxjszTFXyPN1eUbEVWTFAvkRQ69DapsRXZJIaz2itZsP5vdd+/MNLULGPghMiAADtC1nwmZbGnC/2qzN90IiVeeswRxTmuZ8vhlmYRmAYct+jhhN83h3Ggi0+k8Ef8RS8WB7Y4BI43yJRoSBcynMBW8Sdu7fkTM4BFiJlOgVRfHgxHqs5D/MlJSFqkMAAxfN2FI/ixHL8H9qUPOG9S6pFrNXM3C8jJdTQRL+LmGtuBmm5hp8el77E6U1jAwSzQhlfMEAk5+7RpmmRmhb2P2jS7g7YjwXzL/spxosHsRBURQqqqeKEOXpgY5rbqL2K5m6BD4TxACVKnFeddKzQ0d0dK7KZNdiz3WbDfIzRt2TPs7raNiER4wcGYEy9zSoRHSwkxfLvrSMyKRdVIreIQ6GYqpW+3lilHthDrqQrs9rYyHOkcYE8uewUF4Ro0VMnixTXoMBvF5pJ8R9KKGB4lV0e/9eU+OMOIRAU7Gj13IdQszwH1Avo2s3XIu5iPKQt0FvdVOMpIwmHbzPGxJyxLTXUOUCV4/EK3tcxfkQAAAk5BAFUn1JFFSwk/O4EtRg4ICNRuHnDP/zn2Q9S2wDcwoulA9O6/ofNUePDyjeXFLc1zHeNu0B2P+Ne5U9rrPyJS9MoM510dUKLXPz6FrABlFafxVvoPBfitHsgbL5cZZQ4RKufShEufE83viXl7V162wtS9aMBhmxDocVi5OmgfXUN/VqIl/AVtVyr9K7u58lUuG8F6A39pYFUkjn32IB2dXbgll0tE4j8IimHa87nbPsPwh72siaxSVPs2eUI38g1Xlq6O2cXfHrHuSYR4r75srXDFk/BiBz/IGdklbDguw1sf92B/wNm91qsHZrrQci8FZFLRG0fOY5J2pOursqy4teIMPvw3a7ki6H/OTiLs4oijxb0/WomfhRK5nmGNSm+OCV+ZKHC3ks09w39ZyzZpCoAzNyMl8D3nQ8o0ADorD44njAK6dlo9CyPe51socn8pJy58WN1pKlWunitBiuWYrCrv5X5Krgq8mXItTfGtxUjW6vkLTFRbPf3x7lFkh3NrOeApppFVAFLBfFit9s9tMHu0HhcucZlZnomxKV4xc66RYmklwIakeJpR9iYjNXWF044dc/oniFH5aLbKcf9KUMgWk4HrENclXPpuSA+6ZU0+b09N5/QrW2nguECypuw9G2jO5F+Igm3KLbjNjcFh8sgX7hjY1V5N8na7TqodCMCgGoT0PMNLnISRNSXMHVzbkqWYJE334mzaOTXO0pRCeuMHHUqEH9Q9oVwd4ojrKddbBasLnXbxcck63VDi0Tla7EPMAJtrrc+WB2ajjQAABGpBAH+n1JFFSwk/XPY1Cc+krAjdwMrkW7nQy78hGkcLn26NeusQbl+oi3XXoi1Osu3lnaSfVrpqJ3NlYfrR8+UCsB1KcolpVbni2NRhJphrByAoA+i04yP06l3nbNwbuZjEgDPDWgKmBNv1nJcbMdjuKfhbj+4UDaTNfae2LTJT7+IzrnISRb+pT+2HoHxiqGxJXR5fBw+yUtvrlRzHsrIGibgBAj54fUcgKmuIVGi6W+uTYxif43XuPV3tMb1M75OoegMTC723LNVojIw8nw0Vvx5PFLtHHnqYhslxA1ZZWrgSik+CdwVYraRCE53DXlmY1DuANCSJFlmKLXAFxDqYO+QwKTNrhji+x4KJWy05kTV0aIn2g5KPu3W+qOkfxk3E6SUK4po0163j3FKmuZhdyc+25K+Ds1sOIxbFcJ59D2NdFvz1MD1yTcuqH0EH9u+fRxItv5k/H1qbeRJWbNfutC8Z7epjSqlH7EKXBuvdjl30ZvR4wQPYYSmpJm1o6g/VLWtWR6FC6VzNlgZziBPHuh34VugiAkGkSiURVH9yePiTYupfHRSjXBzlz8h7OgtJ+rObYVCh3PhGWUxgUEVCOwEjS+3nJ4+DerGkZpLo1w5XW4SSoCEbdunnNgksXvFqHOZQUBTxL7J99Zd+ROB/GBentfsryDo/iVTmlAFnvn5SH9/F3V72eVaHHRwlT8378dZov6xoKy86wjnpiNNgFuW3mlNhKUxDLexwgKEPjyHZy8Hixu3iaO98CO94/OFhr0FD3I1w2ZuA/MUVdbDCbEE0wlnkz+ImRUZ6TdDPWi+HHMPxLBchHl2e9kudmBx597oU1N3zZzVUPpBsUesOwR7EEG6Qu5wJr4BmiLiUzUeXHtagql7K/HUKL8APozcEyKXfHjXI7F4TInMT6PebX4jod632tn4I5Uw49DmNgld1fIeA+dp7ojySN7KYrcNXMvYB000BrX3fnncHfpVMjAhFm2fdUDOkQdKxw7zytPLwtcNhTEtaJeDfHsIMA6KPpJUNjEVEgo2NBd1Lka0oWAi/YXa9vrXMF8nhMCQW0eNRfF2Kyhq0gwsxvyO0mVWkHAOYA6XHGfaxit+uGj88sBYOEAS4umX7SYHmRajrN9e4UvboEeU0B0kL6wKnqIunbppcqlLxI3ojcTY2b4t3BJu1Pj+iKdPL6EixXyfbhOFeWldi9Hl6gl3OUFAb3lwKLoUbCcHUmK7DJ7kcjWC0AXuSUECUIULql0Q7GHu/ANtpSNawg6CVuWdkvU8NiKzTy5H0t5xk2IGCSdNwJ5BP0zHjiEbtySWkUGsdCvEzqTnV9XLAJaOrp0eRRtl5u6L5EPg1aYj6iYmP7tfXnT+HnQjqvP2Yj8/7PeRh5k0qckNYAR/0EvLZm2tVqHGj5Ho2qFJsYmXSlm3yiegv/5qR0r+qLGXyBru2t1kq7Si37uoBmNtre8GoKuWmpm581bsK4NHF5A1LoZvG/quLSpNh5lL0/y6ykLrAgQAAASxBAC0x9SRRUsJP17UL+pZTLZ84VzF1QfnMMTrPiADQSj8orf0m/GJbuG1KxwiQtDI5IvYt/hvFuOS2CZrV8/p7OCiGUgayhxg8bOosRSRFHTxMEN9grZDBkAYE2P24lgQYdiWvKY+10OwB+9npsrEDIvdaBb5vYB4U4/N85erfZRj+p3PzxR9zxarNbkea5SfKcYPWxte+nep3BtNQvLI8BhHIdXpgBEim4V/8FFd8VmoNwARO5tpnzcuSOaQxh8WkodkQ9eF3AUfuuOdNST08Lylqs4AlptW+rgkCPS0/xKgRaSrCiPTVSULvns9SDyli+/qdlrYeZtxpc/3kAA77gVs+pR+YQdWRIXHbYvteSbdNtcZ/I8QMYL/gKux3Bb/Y0/iGjGXBJIwY/rkAAAEzQQA30fUkUVLCT4Oix7QZSG+7i3SBDC/kI/M05f0nmu9+dt9DsVHIHa8XpP1uQteWr99gxbVn4+DyEak9Zv7pXeAOX6Zm96LyTjLsrXxcuv9SuUFSOnz4U8pbWY5r3GDsqqdD4GDrKO9XBdMysxdKt9EAfU0seWvXdT03XWbN7aGOCfOzujDpi/kixpRlgO3fTkjLDDWEgRht6RzmuM5MOT3VHNmnNfBepGNz+Qc7hAG2ahAkKgmmts1/i3fwVdhr2BHV9TVDS4R7pT6SeGI1BQIMzE45xzsAIPNZjGNrhHWxiAtfMj7YqEjPV0K3B+DWGiBmsAWK4WhspfN/AcI5AXXyAMV55FRbmorxIzXugx3rqGfFVtZ+5N7iiesFoj5XhUi51kP5ZrVUcUrdUv3SUS8XgQAAAU1BABCcfUkUVLCT/4obrcj9NtahiSwhlXRtTG2dMLOXWGCwRyG2Aqi1Wx7kM3Ee0gkHGeZZtBx97eev59nRD9gZ46PTL2ZvUvgt+uBrVH0VNnxI/MM1rEAojNJdJPXJr1CJhr6xNEvyUAxvwwSFIr+iJQrcizVqsjwPhVpv6WYCcpK680ZwCpbPBSAVxPRny54qu11rujopAbOJq6hQ8N48PJm6OppARNC7Cz5Ybt/3DooQQ2BOr2tIaYrj5UsPafWdOG1l6chzb5y8/b8FeK8KGsKRxoC+7PQZDlouOLTAxpSk7Mtt/UdH5KFYcifyma9JyonvX56I+A/vwjBvOWGU+8tJ/qOV3FmnvOngnAEubuwuM45gwg7q1AGl6eS7SjfljHkDzepWHfxHV2e7SIjmkO9qt4895YCuS/cjRKQbIueLKcWRnVWgOQxwrtcAAAC1QQATRH1JFFSwk/8kE1vrwxwwbb6+VTiziz+boJEkgXvb0doNVCW22WN9REREciHtD63xBCRNAGYapAaaOEDLorpV969cJUSi7/DdJhdbpzyD2UFgt7E9faCnWaA/2DQTHsFo44wFN06+30gsC1hcixj96Wxd8lp8QLTdR2IPsmR8TQFbWjMUhrzGw0ALwPMAeeGK/zJgqS470Y0GXGvaljXa5SjdH54TY4larjY6VGxtswtEoQAAAPsBn3F0Qw8f2oDZYjg5eGEn4GzfEEizcDD9W7dMTDOgKkzuJ44cCZvXCvk0/0G4AAtkWf9pXMwXgIbrc+fLx4IJH/1W7TfuDlx8tuXv75U5wMkXv79J16D3MuG2DeXm5rIZ/enBuJM2kHTbxMtEi5RSE6wuJnduE7k//1VF5aWo1P0lz2Z2FPOivbBueFbUXrZuDuLRMyI7z5H643IT/7VMJxGOzeoOMiPXl0Znep5EauBzWSsl/T/2UxIMosAw5aroLloE0/4+mFmsHyVTv4tnQrFsNunM+ND/eSJNxIZBnjO2nvzDLxUBkzP+urZsfdRJyHNEWSfITB90QAAAATABAKqfcXRCzz4aopYs+kwt0WTip/HomADrUHHxiQsQHN6n+h5bfSzRnXsnDKSpFNIL0rjIFWZzRy7aGrZdX/UCEkRyukqs9FZUsaUSlZFA8rYVT3bqSBFct5tK7ISngc8Mi+Aattrga6WWGvHHKAVOjB1yiLVG8ZHVqpjhhJDrgkC8CpfZckEGhskUP6quheT9/ubEy6OV+oTBY4hAukbT6vClMNyy1tDLzyA/NAimo3kCLIBmtb5FBWv1bBFyzUM8r3NK+hQmv76EdHc9dv5rS3TEC4wn2nbLDLgDrLbuGrrve0uvSAF8wOKXqDWMiSLj4tjfcUG6XdPyCrNEiWPSrrRmXLVfcErzqdIlzIhFcCZ9esg3iQ/CAaOnc76XdBsDZqij7bSK59zN4kiYvviAAAABTAEAVSfcXRCz/z9U0coYDBnEgDAxmuE24wDjq8CzG7kArjgqqMNNvK6yewJeFF/RvMFxz6bMx0Vc4ICgp1TL0os4uMht/f1wK7wTAD11VYU95q+eGA+1hvMnrjAEnWXDuaGAOOWpH0vX59YIn1mWM6LBI4ubIB+4UTh2mxqAFjkxpTrFFY3xRTpovddkzCIaxyqJWZlFByuhoznheSBUK7qBPO+hDI7aEjOUIzMX10fxTI2KR4XJllc2TzG3vMGHLteb8qHETy7O/B3tu4EJX1DXs1T0u29XQ7Uo3fsTd6QOd3O1u2cwtgdr6deHlKT7OcapyGJ4ktiTQdS3tJ4u32qcK/Z9OJ44lj1kSoHabKaWu7hffG/12LLRdZdoCvvE00VWEqFdVFRiYeS5KfTtH3SQZLWjlvIR3+O9k44RFucRxGZNG34533gAOT1AAAACRgEAf6fcXRCz/0Eyvt7bdGjODW4IcmyqIHl/Q+ulpdJcQ404Wane1GkqYfxCDV4EhBVwD/YoLtfNA7WqDkhvuQ4RH6v4TEAh2h/wZaAJyUZioX4dQXIb3TlWT+04Iu97jV/YqwWFC2F/5oSQoHw7F1+r/X9HPs6e3vDtGJrccnTiC//pmH7esuNAXtDJzRT/eQeBGINlafSpLRcJQaK8y4dQXEUtoY/N4kydVY1XpFhmwJDjexJUjBFG7j9cGLzkRQkc5obeViEJ7Fpjb+xMCa3ZMRNbYey9Y7tGm4fcoLQ6oUqUMzU3cNgICxFYW4phVntJb6wcdG0lzyBTP//ZlR1Eu8vm/ESQZ2F3lUVInmTyfqvqW9GNQxwH7V3VFzd2jX9706EVZe7D9uZwYhKheuP18djAt7bi+0B3SdeiNkIz3oiDhF48sOm9EN6LYrC4JvbgvLSg0YrWVxxyqwZU2++sjrNwMMpwP2M83SmiOXT/4yUvenFey0Xnhs/2g42ZwBlZ5hNatuvdnmN6RFT8XMcChFAo/X7dUTD4Uel4asNy5fKfJsN+9/k7BNRmP4c/lJTRIOIJet2l0t52VrsSoK0CxUevQrPaVOFQ1m4Ac5V2x2vULDCN18cacWvi1BIFH0ONMoPaSGGv/mTSeMMVc9gN/AUmHDii451QGNswaWTfN2I06Q2E2oRDLEvxOSt393ssEUSuvpK2MMQ2tuIQn3vm16/AhAcp1M9oGtxH4n/6XstJ22pVufr9+EJ9zenXKagixscbUgAAANYBAC0x9xdELP9AmZfqZBT1eeRCR7ZNuleWDZIA3fWd/WHIVl1WRHKbIyRiR0j0hc1NgJL5vxA8Geq4ya0IIiUOug1kOU/RjH6ePXm5N4u7FkpA+DWOh04Uj6KxAy1ekbvOBniYcLQPvPmB/BPDMXDTPvkJ/NR4xT3O8+4/YKWcfGj7HWbU5zgle5NBchDogWjzWt+1eKt3aft3/Mg/BMDAgxlyNUJRs4RKpAyHNWg7ea340nUwj19ZCQSoSvvwEtkibN+swhDU6+aHe+HFDUlAZ2TVDcXAAAAA9AEAN9H3F0Qs/4rhztgo7k9XHcGcpNs+jvd4Z9u5wdetN9JnCQ9WPDjItNEclv2vpTrzEqVlqc0UDqnWBfNrS2KdK4iHrgc2E0JjXgCoXrsChMkO5ztN4ReXN23pwBxOBiMKs8ooQlNkdVz6NOsDDPyIddkzTTJcP4lGhGZD33d4ilg5DTlE73LomMu3xEStylIbV6rtMtba3jWoHTur4cSJsPjOeXugwjFl51SSEfmiTwDNp7jPN0qELqT5kzvppY70ZNkNdZNL2qu2NBeY3DDzXyTYTVMH4LcU79gddsZ8J3+HxoGn+4rZwDQuUeTiO4D7HyAAAADdAQAQnH3F0Qs/QXo8/2spumkKJDs5REBctBIDeDkWZK8mHocF/gH9Xf98m+vOPWW0U2KlUTA2EB4Gps+/9f936MV1kA3gGAOb4YYqjrglsyXbuuNKpekB/iNBVTlo/O7qgNLwfmncUebLvmTwI8AKlmoRSyB/KobNqM9Uy6HVOCFBmgPVYKpU6XwLfSZ6eGEbDZjm/Vnuurl9S613JonzAk6bi/h/DRRZwVEaq/+pd6gBhWiG1OKRduhMdVNO47CXzUBCGQ36wMvf2j3J2gj+64fUpUuQU0fT4f93Z/AAAACqAQATRH3F0Qs/KcPdHVLIkAPTM7dO+u15uJfefnTlXO7TvLj1DeK9iJsn1i9Qb0/JZ3l/5J8EUP+tDZCzxiwo+eQJfpn6E1Rio1gAGed85Q+IinaeWKQwcl9eFFO/xex5rNpfDLdbdG/ZX3StcODgE3mbPaPipXmOgFqwOhHgUV3P28qsdeo7mTw7dBbayKDHiGbM0I+Z3WlrM/XRBpR+45qg1oRUQ7obAvoAAAHjAZ9zakLPHqliDlCv6G9lovCdR6ZmtbgUl7xyIoQcNXLNc+GM/4uSfBGNVIymqYQwlg3CwhgSSQBOtF/0/L5UyvgoIJKvV5OAtHZ4BAeXRDHsr+PuPLKC9QZ15ey06CCjbOlJVBIw4WRKHgyLtFeriwkXNzKbzMoxShVx9JnusSadTV2S7v36CAHDdsPHjBEXG6SWcmEbGfFDtQPzIDi9g+J871Idi+4khR7g/Vwkynkq6nDckl7xMUfZSlI8Sce5TaflxQgxgbM/if0IULpI983t6Ai0bkK1PcnIIaQLU8Lb71Uvi6GS7ajcwZkVdA7lYGj8BZrVUkpwlzvpASiPOkzd8n8JGd9JppQhD9+lIE6BwjWBgR3aT7/b48C1mthP+YyNeAcvFx0ipLJzLz7J5/gHzEoSkJ9XBOooOHu0CpRwGW2bW/z01WstbOQkfmZ8ghk9P6P0MJ8qrqzgY/JgZuqBxpzBYWfeQwKarZvZDq9PWrQ9+tPtfB6IYYe4gjpqNJg2DwsIMXa/SbIRGZwbNtXpEq68SqAC5624X+a1tBcMx5L8QfEwpyS8EMVbee+cgoLJHph6O9XJIn0oNOCRLvkSGtE2m6C2DXbuyfMO3QML3iIJCBwZidHYRCTY/1Y/rRMrAAABBQEAqp9zakLPP0UkDMt2L0N35ksPTLpxhb0eWv7zSULbIc3Dnya6hYJTh6QvRIWzsPaXaxPW9nLBGmZoRk4C6x62TsGz1FNSp5djMBIJ5wPAPoqAjppTpa37uXDRY4vsC53rN3QgZHWA1EyG7+p77WMJ2YHOlWhzdfzq6zcVkoR3Vm1qPuXEGKWAQAnfQCa1eg17MGOVvTfEA4NKNMZ20MHlllPT0VVZvAxBZ8B7isYw8Bo32OfwOSrtC1zses7HWU3TQwiRz3Qe//Poe3I41uNd5075SEhFPszxrj/TfOq0Q3oWhyG8PIASwvyA6c7xmF6oAfb5bUG24dsLCFhoYJTJTdJGIgAAAPEBAFUn3NqQs/9BMrXYs8XpCYDTKdCW38RVIMap/UiT0uDER1fkMsjLYVdPzKst0yXx4XqzEgvs75KhsW2SjWYAFn7GuEOHYjwH1lPemKLn7kvfs0IjQ8ss14wSC4iT0uDFv/Frqqb7nid++3xBs7uS/Otzjh7dVrwTzAG2CcCd5eIdoqMFdXgtUvuODxOxzH8C1YW9wTSKe+rFwmzl20iFR5h/IhwImz4vYu3sr0kNdP1FriSE3yFLXbwB/iT4XxMKpSegh3xR8CZ8czt6Su6G+5rJjT8ntuLgxT6gnCKmrk+7vncWs5Okd+oA+u3IEvOAAAACBQEAf6fc2pCz/0OCAeANvzN2emLMeWnm1Jn3dnKzcgskkrrvVEZqereUn1kHlbPofTTC2fhldivVbtUqBn6IsffuoUtE0fArwpIpbil2VWJAAv1tQfoVzV5njBnLm+Ywc07ddLvXc9EmCWBpmdvvdzSYuNjz5N4KfEEq5pJy0x6sshkzZqN8YZFUcZDfSp/EPp9m1ShkaeMQdLdlxKq9PsZ5hs6GQpsR4q+7b9edjH4LGAXmB6XtTS0H6WXUxfKV7bAP8/W4On6F13tJLKzcuIYrdNxic9vQm1X5tOjnbZBxAtGCeNH0l8Ew37jc3CDeiZF52vxEtzoQ9Pqr6P8G9593HPPTO/fbPrG3Z9R1zOazTUzkW/K/scZgcY5MqnOgi85OF2DOmzy2XqYd4iNMBGdznmZSrc3s5zPdxLIm71XIOLBHm722sb1h8HwE0TsMtngZiu9vMj7BrtL99ZGuWisSsa1zgHI2F1XdpNxQkVkfLB839EmR0aRUAqbp1TpDK5jadfYCgxy0xLGLRxAwEVOvYB/SldOqyC+x3e7PlTdvt+fssE5BXZxQLo0JzkNbQm9PQjEv0t/dZO4WpeTyvXJpy2LoQpKTqg/KmMLY7+y1tS/cclFuLT4iT41KFrZsjxEHXMFV8UIQgnk4ruYV5flyjHwmywOHJxFM7VXiLyzU++o4+uAAAADkAQAtMfc2pCz/QTC78F5nImi7GYHKCp3yXC2s7KC0/VQcWnZ9CSFSooWhpLA3qi+yNxRc+iFO0hcXn5009bk1kID8tF/yvqQm7pPI8nqdtNdzt2uePBBqe6ZPVaVvw3gLiLLLobWxhGig86AdFB0XnowOJqEnE+mcg1GdvRPRobGG+8/cFAi1cVvfBg/CwiRvyjhQ844fJZURG9dXxZa/7uxBSaLCIpGi+iAbVhA1k6tFxkD34B024NarX1+l/u3B66wJGDFOUv2bjOKOz/8FPXR4mwUW+x7FxBJgSbTe1If6iIJLAAAA+gEAN9H3NqQs/42fbgnvW5oUoT0sPhONmmgcgALWLPJ0eh2qJTTIekZV4n/7ntAtcaWoQNM0uLzQdzEnT8/GUivOiGSxS7QlUC7yhhaiL1hHcKHwGRHZsXEviVQ1OgKmTSqWniYS+F3pnXvZAXD84myp3Hch/u4YwL79fzo1KXckMIwlut6XTrcY4VudhILvWCL7Yj1pI7rDg4iGRcAGVaMTtZD+kRXTCU7GhW+IxhrX4EEQog1UHyJyRrz4Ls+eIM9mU0G/4MpVrMqFto7cyrvNd75v2pqGUTrL9mRnSSu+bx86tY2mGh7DHbLUXxkoz+IB1lY+HMgqjzAAAADVAQAQnH3NqQs/Qydw2PqInZduADAXlGQrdcBoAxTrB7a2Yf5gI+RXtUpbtnwYtikOJOzSCQtNHCH1SjVkj6PMXZYgfUKIKMUGuj8VDRaAhNkwyVKKuYbWfU6/IjyG5P2jspX8AJ11hi1fT4+vZgiP7GWwp/Cfv/141a2TCgLXKCznv/0DuGrVCQGO6AmUUTmOSV24rJ/PVPuSqx+ogVEZ/Mm+wWm5MBIgDjjTYvuqZeLXSgt37RKgHS9gQh6w/jJzSNg9CfIwmRLoMHod4KzpWboZ1zaAAAAAdQEAE0R9zakLPydo+1Qs+1N9SHZxXDVeG3iSl2IBSksY3SYGS/99aNd60D9T0qk+2Veisk6AHlBI+sL0um9LBEETVx11WuTlbKqA93MAebNBf93Rm9LKwOeZknmM0WqQCclYCxZX+6zEpnQ1/7X7/md8kk4DagAAArVBm3dJqEFsmUwIWf8Gnx75ygHn3Ae+r3nqsG/BeBJ9y20IBQliK94noZDhGe/zXxYe0LgVsovmLfaaBm4kQI8VOyYZyxK/tRifaiiTWTvBlNb8gCsEaC+loYtMDoMgzYoAIc71JnBvqJRReUwFR/PM1nM25l+ajkmP/FdqFO18tx79ZFgmzfGMzvZK2zucrR3xf8Uw/roUJ05YX+HaPMeFqlBfto8Xdr58cfA8DuGjgp+TzDRxqFtU4J7PPoo2RltgsySA9srnbelF4TBvZeuKya0DfI5cwrxWybKLzMXOWtA6eeO9WkajWuMxAVrRt3nYecveTpM1Ts5eLVHcpgXrD55D9O80OfFuwaaOREDvywfH2Q69NmAvIcePMPUHyH3gj2xvjX95PovXiwTvjMrk4eD/EYSkkf5iOe1GWWOX/hy4sqELeiLzZ3H7UjxkMKDXVNIijHlkXzdC9kg0QMIyu2talNgpxj+wXvOAjwKufpMaEJySbppP1Oiih8Oxptr7H/J62FrrZXt1Q8xAfLio20cFBPnlMhv68Rq/v2KQiPWmhQfLRJ/kQ1xrczC7omjkZ4cKXy0Xp0INSGaxS25iTATuUX4mDP7YZRrLkDcT9VToJdzUbwiIMXHPMdoGMpGVpfq0ueTwzxM2zAekwCBV2/ag0fo8VNQg1qGvv6hlObiDarmmnPBtT/3vh4zVXQlrHCA3ynUu/yPlhnF3QMjW2JwUDtc9yXX8J9QS7YeWS9kuAl8RMrWKsmfzYuWov0uiWP0x044BiwU4afnMU8m1VWGNWoOVulmsgkRHcQ8x5NqvdYg5baofM91P0geST3FVAi4tVbxbk42RXkHWJi68mA4KjkBN8oMCzQkpvmZlqcnDWk7G1L1NuDQ+WVmG51MOyBY2LqwW080JZP+MqPgX9qZlpCEAAAIQQQCqm3dJqEFsmUwIUf8Ie1A478ov+Q0eno68jtx3agDj6tl/5WMNml7ftrFwC0NvLcoBbSGDbW5ANnOEeQAAAwD5EmGSh6d+kosYyfiQHg7okaCwpp9DXx6ceIk5U7mw/bVAtYLYyL1GSHcPRdqBFIAzGP2DsBrHo6fUG2Ic/6gaPREgyHIVYZzgYf1X17ljORhraGKu/9QEnfGeQXqD3gErVDpvs2iSfaA58PkroKAR3LTxyI7vBEQBTtUtQhbJpzszUq0QBk1CyxIqnBqahkhHZT1CcBjnwy+4i2s9lYAt+LF0zxmJ94alpmLBksI37N3vEzkhPFLP+IFR5e2bCChlNjpPeIDHuQztGfeFl0Jeq5VXtTG9wemw/r5uf44aLUzzjl/mq86kWhndgVNMjc3c/rvb8k5P47Hb4AaI6iUUvP5JtvwVNTa+sAuneidyHtkkXFXii9zHqTQ7ehuPq2a8nSz2aJivOn4c5IPWCLAWixdjrcXm37WVEAn/D6gPHFcy9eedScoUxVagnZyzw6ra9MRXun44W4RtdLaANJZNFkXOpv1uPdv0Fx71zBzeqGcRjhC/S4V/R5xeGs3t/s7+owEmjfpj0uRQiY0+fS6oD/+TNUNXkzZtfRQc36heE1fX+UeuEVueC+DdCB6oWUMt8fyhtKsXgfaQ+l1Vc5PcVETstxD7nw4Ce+pVldthAAABuEEAVSbd0moQWyZTAhR/COeTWc9x8WK03gGbymzfsV0jlCTrtry0TfznXpmgc6ewcAAAEHm2zBC1gWrZ22TgViJk3SD51RYwlA2wjrgJWlS8lbmkAgBidWo0GcKIZsGNi2M5Vap9C0Tm2c7DhxodiXBJddFQVbxzJSfUgQc8va5TMB4IJ9oT9KYFoqrwrWOL5zjUVPedtF7IqG1dQCxcNIfl/sZetIgHehTiIphuyXgGtE/Id913dFmvJYsyD0tSejufRrI4vNfFoaVo7HDjTJX96ytX6ou7MbO9IGJJX6Vz+kIEuAdg0gQgluEBYL4kl8g4Yv1Bz6wH1vAeZpMy7RhFW46QYr1uVLIu8tAswxOsBPeKGSaJYBeWX4cG/YeEuxbNTaA2koJAPtwmpdQwJED2pUDmJXW+VikVq0RyWxn39mxwchmjo1lTjxM3OG5JaNfKRNQQKT9NKJvDcFJbTqPtxBI8nH9StXO751ummLcU44zoZh+6V/ytNBZ6vQPTpxihDijJ44z33IAOWUX4GBEY8B4Yme71YLAxrNEbqgRmIUDXR8ceFK4pt8IJ/xmh7jfmDcLqVVMZAAAE3kEAf6bd0moQWyZTAhR/CUvMqrk/OqYoEQxq0JTSHpXG7ckNTiqRu2yPLoi9+C5vpeKB0nJ0cWeaN/Rqk//lqNVU13/hekAAABwiRlS96ADmKpdYf8ngzCMlYR1FM5jj2M9sS7AjBS07H5RK/esISmukRbDrEF7EydabAQtepI/FMRa88Oa6aMS//ptCkKOjpraUsilCbJ2bBhKIt48JshKcwfUo5by6UFVyIDpFSDsEWg4OUio3AM+91X/zQLer7WIIFwdEf5TNS+WqQ8bAb5cQ5sK+baGHjCWtbYXnAkOIyv5TuWFZ2QmOy4W7Yz3jkoE2E8v/csZr21USzzpUzu3mA5uLtwl5uewKKiUy6FKA4i9r0F/YWMlbj4hVKYmizmm/5HD/0O17iBGgQBUIZ/fj0ylG1fVnjH3k8Ci7u2gzmmEfV7oAWT+XV9kGNu6GEXauYrlxtslymj7N8e1aOxSc8NDRaF+snQxzVn8zdhILmKLDyGJgI15se2jeBjaFBSSxKiLPB5tLaFjLsP0bNR/TtgOjYtEtLRzTtkTBiVO9Gct3/dErvIhFEfjCbVdMmyLC9StzvEhJ5Dn2mPfwJE/vh7+GnJNqo3SpK7KCx9ahTKzgWx2qQ9R2K+7OECBgN1XKFokXDHd7v+bq8mHh6ZTu5dsOAaVqvmMtAXiCHAfpIp6yWFLZtUQfKfvJmu7R93KBU1spJtJUfzFEYksjXBCs8C301MmM3HHOAlKFnOCcc/1VvSO9wfWIUoPpo+yjdx/caHdgEiuC1aORCkpd2qEYw6YaZln4hPmrojZQrvNBCew2lpJDYD7QYAKspCifLc9tF19Wrbqgh3hoVYqW9IKoxXhGYweCAmIyZx9Lk8GnTI/ri/P/DMIXkHnLBEzpAbXpiZyNsXwv/uIjuBXa3t/dKRyDCpUSQ18P9C/PglPkdtl+sC7EWkKxqEys2pQYMP7GPpwuEgpSmqtSzZJFsEmp+zr66wBmA6cNlQ51Pgv5JP0uDf7CNifTBshBxVYg/BSEmRvjtfioE8wHuSHaSiPDgvcyj+jn0zECniAEKjSrVTAoLLt5qrdG3q1ci2PFy/08P5veqsqZTiAjpJV60aVG4qUv287Xl55QjKfFL9f5o259AdPifTu7bhIeyaAfwpCle9jNUaH4h0spSaT8ftfsu7MPnfaJoa25cY8H6JpUFrfd/qOiujc2AlfPlDDcjvjLF1HhJMdWz0XCLoFesP3wbK5WvF3+aBZdBLzIqLqDayxjjUGgPtrKv74GrRgGuWGCudaPl7rrTIBFO6RbH+DCKcPx4+WuPAR/przYdeKqWpIqe1pnQx/g/MojYcQ/R6UBe4e3oMoupG0EfGlPUoVMl7GTMY0iM782zU5x9JLnUBlipK4OeWDz+6MFN+FKMNzL64t8tyH9IpZrlvy4SVXab/vUbC2v2BkOpCUCqon53dOqKn/+HaAikaB+rwgd4QKbjdllPDgk5JjqpW4BzKdl48Dv5RcRofPT3ybIZtALUYfplEGd9DpVVWUUQMJDDZFCSkvVJ+E6YxxeEpjkb/vmTWqrqZvArPWhRG1PGssH1L1jwJEF3DKrz6ZPNrrcMfZwi4IohqE4A7MCXC8V/WOg0SqTWuZ+EctD/y17maYk/XXY3rMQLhOFwBthVusAAAF9QQAtMbd0moQWyZTAhR8Jp5IVRFt3+TlhL/8SpX9ZfYbX/jCyy8ya/yRsW3sx3NKvTQ739D6ebidBnO7FlBDwv4OiXApaS6eO482dbYgG3sA7M78h+rmGh90ZU8lI5i9KlAyU1gURBwJNGQU150ca20txY9f+E7ZRP98kgL5QM9iJ+VudUJuc+YcGbNEgBcJV5AwEGz2n4vpm6+oN4w7YildI7DN+IgREz4s2p+fOwr65TfRNZwWcYmygb2ftqM5A6Y12PzMZJIv0KaNB21NCw/Gf8bXqeCjZ4Bqj5WngA80zpKzN+eBKNdhtXuVPgguFoipu6wM/diO7nHEkZs00E/lyPomaPU6+iLILXWtk3IzU/f/B5gMoXEFJAAKSxjOHoLxzPQP7vGimOCHMduey4NuKvXTwjOsuXaOWdZyrLuObxUYlagGuJrABzjGPThZi5sWVg4bqtk2+WITF3/HLXSLfirRsNoVyWJkVUNHY27Pqv7KYkN62QvqyJ0m/AAABR0EAN9G3dJqEFsmUwIUfCSa3gX9z/2634uteqDeobeBTlYI0BPAd/u9vAAAa+FxeowcU259+Jv6g8Rt2RdzoBTu4uICg13N7Zy/dGrC1X2vPHREygTW7itnOzSOuysP3weMnP4Qs3NiF4snOXM03YOVRIB8695FTn1ma+SqvEqqwSXLP8Q2Fs/9IX21AKO/brjQ/20L4rTZmJLUyzVKGctwna5nDovdHHCXAvFj1nQ9sn5EiBlCGJr4FSD1QiJSioTmHRo9048d0HmcUDcoanaQUtiIphup5OVKSCYiIUOtlhP15d0O50dkyUpUJFH+aUjVTkhU+OEIh44zFriLMF9ypRLbobC3cp0SyHfDE1OpCh1drDOQU8AaCZ9Z+6MMlJTcShg6bciqQkCfpPCxGRaVXY/fyrkUQOV3h+jg4Q+Vb+4en2eQ6GQAAAV1BABCcbd0moQWyZTAhR/8JS+jEssDY8QCuym5AAAtTNiInnfgM94+ilqle+08lu6quiNpsAs1/4GTWV7Hws2MqO5uk82NQ1Xef3Y34pMRdtm004kg4VNtbVdYU1X/tkZLeMpj+JyLiQWCgWoTPwwjMCJDKJQkBI9krU6dyTRitQE2n2bLjQ/+TtgWiI80n2MKHfxPwwMShxNzfQ7mQn6zoDA+/ItXfCZkFDKtN7vjrtYCYl48+Xj2EEolGyZ1Ol7qObWxYYvslt7Iu2abUxSLwBRlgPkq8aRqIBK61tC5PYrcNizbsVHM9P4GQaGP0YLV+AlEswXtsI2V3W0L0zw/6RxpNP6ST114deHm3VSjHYCO790wBHXRgQVhw72zoJ3RAZFcaKqXYSVH4sAGOUuqXEoXp/KDjPWeyqTctZcRo8EMSlMb3e2m8cZW8lVxKJR/lXJ3eF7/uYN7uImkPAAAAu0EAE0Rt3SahBbJlMCFH/wV3NOAMaLpjJgyj7VLFRnIc4CKhCew5E4T6hGlggudTbSSgs3i1efs/q8bqM5SvCu3M+H+11zzP7weKb+eD+Nd7+3FG/yfkLC3gEMip/dwBMs3aBEFHOYKpiNdLImR+P8d4i3CgRjmuCfeEXz4Ej2/1n02xsTiAGovFtLiyXmd5LLccicGCoKsd2JCOHfGFSKOsoJZ8KbYixP5pwrvdBqhPxGQ5n1H70IpXYR0AAAGuQZ+VRRUsKP8hNePobz9jcEWsyxTpoF4dpoAuMxNB/0PgNwrKFYeSjP4AGwmgtHr8PF3kdWW6PMfUmUrpIGCtcPMETzJdkTeL4xpEWxiDDepmv0OvjcJCEFY2k/yXBJBDRF23nqKZ4JvJvosONBDOZNruDEaNru3HSMyeM9hnM/SWb5jwnd4I11Vg+KwwmmXOzt506AfLVIjdJOq82g9t/ccw/7NCMvLxe1FpHevfoBAKynfuKW2+1Dsi1IyDWQCWQY+Fui+D7zkWzv/l2kXZRkB9BH41Ajsb4Qf45JGCJtayuSMVWCfOY4Zy8IYO4ChMWcWKG+woNcTuRMY6jx8MVXbUtHaPAvl7LOpd/rTQAiPIHIw42zMEpTEOA81MJW+ThMirs1I0YGMxXYb1a2inmQwEySvRptHMvd3oGRxV2Oadrk+vxeae7bbqtaN3fNrvL1eqqWFkEgAa1ZQWHkgy6CXrriATcSjf1VClk07/GspD0CP4nmCN8m20yOEmniyPTZU2Cwt1vhZy/+ILY84SgqGR3PLnfVtlZqNwpPJEfbkXcUauLSRLkfGoXfGvBQAAAR1BAKqflUUVLCT/OqmTcvtwcentFfmji8pW4xslt45wakxxtHQO3IAnQ1zSoNP+UZ1l06NVyS7/Ot1hagvMowXtWkr/cFkCNzOt9Rjstqqs+dcCc7d2t2CFdxYOFtjx4GB6yKLqkx+tLOTafz/HscL7jPRuu2cuJ892t8jNmgaHXmdcxZIvbPAf2OSfkoRo+ttldPwQ2n+ZyyslgTd3HWAEd5aLe7uXZrj2ysZ3V6nqgdAHO/puU3fAERHHp5pE5ufFOU9ndt/YYBa5p80IOpOqDYj/ViFmll+iCeRrYauHsUElGfehMmmnF7U0eySup4pdNizdIwbKG6ipBHyuSLIBYxRH2JQ3cS/SNKTi8Hy6le7ePjEImSD3K3t860AAAAFaQQBVJ+VRRUsJPzwJk/hOe7kU82ZFha6e9l9XJVVAztAPucsPpoPPOpwJ4ugEuDKX8pzOdG4B3yjo2MbumVFa7X9MFPOZn3eyxtJVVXJ2FCKndYlEaCRvE+i3TrMs5SayrLncBxtm42PqUkECBsWlkron+p+v1IbInG7Gvqz1M/uplnQ5srTLrf1z/FojsIC6a17nrR3MQl5GSkjOGZ1cQ/Tre370F620vb3vUxHgPtMxeO0HPhbPxs51SEhxVur49ugeB6DNTdX1AKy2sYM3vL3sQwLr890pclYNsgyb4PLheaa9x5BFld16eCKcbt+0Mw7mhaRvjjAO/N34p+cDxKNDRTllyad3MtgEeJeYtLghiYCn0rjkqIZ/lLj8FADioMzk8B7TKVC3eQ3MJ3Yi43D6fFr+ctYjLq4hIf5vNt5KF2rU1qq3M3xfEzCHrNiBKWcgOLtVwky4JAAAApxBAH+n5VFFSwk/PnpxbXdrhOgZbJTa6AZYzEgrpChcss7ml3ysjWs15LN8MxL4dMExWXXIXctHm6o96jJOAjGm46l/hEOjif7vnbzWa8BPCgnjKXcr8v3bKHc+VETPi6MYrQCxSAB47NRiFL13eL0eqkIAnDtFRaDKVPvyhpveKf3iPxU6MN4YAWOrYubKp7aUNxk2K/Gi7g8qneZlqP5u3YoPpx8piYxuzV89dPvhVcrIBi1I8oe3S/1PFgvyadetv0/08gsey50x6PBkiI7y971NpdGUfd5SfUscjSUmaENNPb8GwUQEkRBPx0eN7DKx18+3HyL06KfHH7Y85pQ/ERn45yBY2pE+Xet9GZAWSIJMEJQzC5DjqLsHApIgJxxDV+oqKOhCFZ14bkbqyke7Cr56WUG5OyRlIh6uiRrLGs/hHlV8GRhUAJmOM8YSurSOQye44oHAbmFH1Nv51UURoO+e7W7K0pVojvvO9RslW095jQzENPtdZaXR8cJiKrHHGsSPqQq0RFGxAPT3vlZCqyXBuQws39/51Jklqcmz3a90KgBX7Uf7AYgwfXFCep0i9ifw3b+vsg5Lh3fcZOkfXssxHSkHQnUEVv7Os4Yv1/2YwdAeW20Ahzi5bxqm7QD1cNKyexJLfHS7WcE/16DQ1zvc7AYvqDy+mJeVa+C5VxVZRINR42saFi0bkIuLx946KUFzDJZqDUjINNRdhj693DoO8YwAEuEfQyuQcWGiO3LYibiNvHSSBidv0TEvd3vNdMnpNOrC5g9y+nlS59G65onCEClo+LHdNdy+qLo0yb1tovKTVhR9TWBqPm2+tGYqFZnVSJtdqcBW9eQ97nkBs9uTr9xSk/csxKkdYgZvq5w94MfWApcyNskN8AAAAL5BAC0x+VRRUsJP17UMCG/kCn7srmkVnfk8Zy58lfmrS8Wxgfe3enLg9XHdjW0ZbqmJ8kZl6SuRrGlMv6CUJxRFI3yUCjTcm//5HK5i9+QmKeoNK20HXRRK6qwWoqbVy75DGkozBZRg1HgtZufUwWBcpnwns3x9DHNdBgqv446x+D/E0z7MwvgMSjp4q5RaTHUZJqzTzNI5nAZtYlu8vuGC9zVCniaDyMxuGoAgLcy2DpQRZ8UgWALecqOfwGngAAAA0UEAN9H5VFFSwk+Dose0GUgpWv8Bed9K48wAys+AQ+cl7z04ycj63aXfjlv1HZEmthmWzhcnE+QrL4hfr+Wc6pxgplV9pmmUQIgmMo9JBArmyKZDt01uPBBr5wQBj5bjjWKLHwg8srwVlC8Nrv+L0p3OhVG0VK7b2FGBhglH8qk3co61vbV+cXThWm38Kc4gMTy7VZsiZ0YnKUKUqDZxzaFqnUtstGV+uasMhVoJcnkhN9PyvFIXmFwwFQVuFrcV5oeuGgpTuSRcJ+UfCvl5D/3YAAAAvkEAEJx+VRRUsJP/Wv8VjwGRvpUJeaSUXTyjy7OhdlP0OLH0avSkh/grQM0ddj5rQMG+gBxOXiAAZYJzSs7jWuzA/db74Idgo20E8ySdJ5Q1qMKnnkcYgvYzde9JnCtQkN6LxqGMOUjk4R7kXQhR4r8bBB8OS1scsOu4/NU0olRPLXOX/A7+XnX8PZDYr8JIGkud1qhvRlE5pel4/6TxotLNT1y0wbd1vUIxY8deYFL0tMgm7/yuwna4Z3zMHIAAAACsQQATRH5VFFSwk/8kJX7du8ojmhLfg9dQVbmDB4Htfrpag0BO+zsR4wVTflWsAR3LUmOCZBbm53CnbcmRvCKtO0zLppfHP3a4Gcn2jlDAWfmDGFSxE5cfViVple5fxNj0meGxhoBAa48nAEhiH2NY3l1qeXxfXLDP0vaKZum0dVucnGuoi4q/8SY4Fq6RQruO3r6G3Vl8KCuXhGcd5rF1lBeP26ROaKj2WyUDPAAAAboBn7ZqQs8fGNTU/UNWjP8lUcJYGHrimcLoRTOhBx/EvcfwHX71uQm361/BjKRb3+UNe0JyphWAOrMsP1DABX7bDDWjVvDbTCf4pRAMXKUvcfaBmjTZxIhcG/ATxG36UgQDM2o8j0/Vn52EZv4gXYanHEDKNYGKnqZlzK+orx7Wg+Db7+1dy0luUK8JDsCdRvmJ1b9UxoKxhGqJaANToGkxMd5jL/lmme9ibPpNgqPsxt8n3TPyHEadbI7tK3am25bXrGcx2WNKRCaX2+Pp/vHzhdItIcX7bi0mBJVqc3PbGcXz/reQ8xPw3Few2byZKIFwnazWYOzB+7bVADTX8dkLvn1pjKa02VJDnlvW7+xwiGc3SIWkaP1YiM6KePCeRqxrFjMKwz9B96EoTfNgKyTJiS83xNYdA39/DrX25keQSC8ZH2eJaOpPXqEi/RsxcjHlH+e/wy34XY2tcXg65uPabdkacaegvJNBaBeNXBVVi2PL5lobIBQz313GN8yzXzUgedAwnIi9/K9yJQmcvzxfvtjUsGHnnbbO6vWjsBl2KEOTHZyJ1IivDji4vXVQC1jSQeqBcPAhvW5BAAABOgEAqp+2akKPPFpeByfsvOyzzHCTcyCscVOQtk25ZYr3BaAFftqk8axs89uB0cYlx2iO0cKkrDAgCnbgcwB6uHnY1emIaMoRRoaOL6AUCh38s4GwTAzZKlbZoWT5xrj8/X5tBYJih+iDXAQhwTaMBFggENPY9tiW567K52e0oezQ0/x69INEf6HXUaMRKk/gal4OrXU8MOF7P1OD7IP79hpg1uWbVn6dHBmkt4esNSdZNnEx4fHY+8vzrp44QdsWzXatIDLHjFHl52R0sXqyrqiRn4boayWwe5xC2Ts5pf+FqVh2Ym9p0vQ89JHud/3whkY/dtVEFLjZhK+/lx0pCf1TQireoYlKM+Ayh4XW+LxrKjHJtfG0Jd2t5ny8obbRtxFSqNM7GTqjgqvH2sx4wmbQ2taJla44wBxFAAAAxgEAVSftmpCj/z1EndeI6n8oF05/RH0WcJeH3+qNlEPbxbt1XUabf1HUxb4uHFv5Zh4Aiv6yV694Ef51A3Qv5wCFYbezheiJ7dACEO77FAMLDcmo304NMmGty9oVNOcgDUha8mwqnh7Gw+YG8QDAsn/uj47ICkwZx9kBTmvWyyv4DIcGIAcTu+sfNQcVe6wsg6ikWo1VITnkINhsokKWOt2C0HVY8/YF2NVtGMSRRA636S4quGkdlK0Bmcvb7Q93FPkEw9tmcQAAAZEBAH+n7ZqQo/9BXujtrKjWym+1DrwUSSI2N683+oowwhPIYBr07AI06wmH0JP1PaYNQFoo8dObptGdhqzLaBOcpiaC4Pafu2MtN1Y+UeOlQKdmSaYr81PiZkuPIlCJkPKAnLSQny5f8KVirdxH1P2s+3b9pLka4DhMPsPghtvuRE92c0OhhMEeFmKptDU2j9R+7cKwdM+kx8n6jXUCZ38X5aO/+4cvpD9YZlNbEBbnI2H9+MKmygjG+lnSOOKdY9js4rtW1cng488KvW0UPhGvnWpLjSvkzx//8jtc5qmCxZnMRrIpPWj8aK23GgSoxIgcmJ/Ajo/+03xYBF0Oxlkf+Ermv7QJnggzjD5WfDcT8IlXbyrzDT8hy+SX6o8am+jj23lp38HmzqUdDTeAWMwHGEvrOYTnFc0Rc3ZSEtEbXIG1pjEJYLGoqCeeS+qxJSyGstruR3ENFxujKnwmrpBF53zLgVp0RcEu3Uy5hPmQJ1JaFGisJ+yXKmIp5PVA4IOqC5qZZ0OOJqCJuLSuGIPM5QAAANoBAC0x+2akKP9BRJU/LH+wr14pX5YEo5+R1I30ezHzPh61yK2TpeZ/0nS+gwXXC9LGl+ELkNAbnq+ZeAO55jJt6ooGPei3Cz3uvGNIDY/Z8GhKIGO5v+WYzvJgnd4LHPtXUNJlpJxPRk5DZ0sidGMCT2zJ55/DkpaSBZuyrzKdrDA6VlGOZvxVwl7mNjRZl5635kVU66djsM+BHXWFNZHLCXAPwzITKPRnF3hSilEYgFMApFm/H4pyDBjng5ZY1it4TDX5HrGTePiNpNpryHq+VM9EDoVWKwzj+wAAAJoBADfR+2akKP+JCtLTmUPGfsyRSFOP9O6Gb4qgRNBWnb+noJGAzrCE7dcXfJTu7BhhsL3NTGel494N/wFHNOFARwqM/zP6kROx9jgduocTaiLzWNpTO6b0BGOqONL1DEIfB2UgEibckJXQGrhgnGfWORjVxJHrdEtpjlU1DTdMTtkr01mGEM7weTNZyTRHIiwjAVys9dLbjIhhAAAAlgEAEJx+2akKPz7xyg63AGFgnfwBYqTkLflJnx/iGq0Tx4V5Fx4FrKhBX7RPpym7765s53CDP3dPffapIqJq2fI4rJnmhYzwcGkm+1aoQ2ZVvh+B2tp2cK8004I888+LNa+1Dkxid8YPuHT4+yYKjsNCnYPc4u9I0yNq6ad2zRNFoFC6IJpEvSgcKLlvWnauXrAK0ZdgHQAAAGQBABNEftmpCj8kJH9owgfUlpNM4NNTmOtV8F9ObNMLgBLIAZp0lyiHw1+9wgbex3Ci8x/WzVXggyVpoxBrVp2xHj8PibuB8gQB1xjHn/h69S6gZjvz8tGSNSF8nza3vRSm87flAAAEQ21vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAMgAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAANtdHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAMgAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAVIAAACBAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAADIAAABAAAAQAAAAAC5W1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAPAAAADAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAApBtaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAJQc3RibAAAALBzdHNkAAAAAAAAAAEAAACgYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAVIAgQASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADZhdmNDAWQAH//hABpnZAAfrNlAVQQ+WeEAAAMAAQAAAwA8DxgxlgEABWjr7LIs/fj4AAAAABRidHJ0AAAAAAAPoAAACtEGAAAAGHN0dHMAAAAAAAAAAQAAABgAAAIAAAAAFHN0c3MAAAAAAAAAAQAAAAEAAADIY3R0cwAAAAAAAAAXAAAAAQAABAAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAABxzdHNjAAAAAAAAAAEAAAABAAAAGAAAAAEAAAB0c3RzegAAAAAAAAAAAAAAGAAAPvUAAAW3AAAC1gAAAeQAAAIhAAALHAAABDwAAAIpAAACuQAAEeMAAAdbAAADKwAABMEAABoQAAANnQAACHUAAAeKAAAYWAAAD5oAAAkuAAAJJgAAEFcAAAnaAAAH2QAAABRzdGNvAAAAAAAAAAEAAAAwAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY2MC4xNi4xMDA=" type="video/mp4">
     Your browser does not support the video tag.
     </video>



Interactive inference
---------------------

`back to top ⬆️ <#Table-of-contents:>`__

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



.. raw:: html

    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

