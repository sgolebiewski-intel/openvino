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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-690/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-690/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-690/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
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

    diffusion_pytorch_model.safetensors:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



.. parsed-literal::

    config.json:   0%|          | 0.00/547 [00:00<?, ?B/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



.. parsed-literal::

    reference_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



.. parsed-literal::

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]


.. code:: ipython3

    config = OmegaConf.load("Moore-AnimateAnyone/configs/prompts/animation.yaml")
    infer_config = OmegaConf.load("Moore-AnimateAnyone/" + config.inference_config)

Initialize models
-----------------



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

 The pose sequence is initially
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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-690/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4481: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
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



We inherit from the original pipeline modifying the calls to our models
to match OpenVINO format.

.. code:: ipython3

    core = ov.Core()

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



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
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABHlFtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAbDZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0gzk4RTtcfdazWBe1iIZEIyYjsPUnQAaQlJGZnNVdO1HN6m0WUIgqmmm9SLIIHWvhNVrDQrahMWufxDlw6rVLfr0K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" type="video/mp4">
     Your browser does not support the video tag.
     </video>



Interactive inference
---------------------



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







