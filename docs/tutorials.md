# Interactive Tutorials (Python) {#tutorials}

@sphinxdirective

.. _notebook tutorials:

.. meta::
   :description: Run Python tutorials on Jupyter notebooks to learn how to use OpenVINO™ toolkit for optimized
                 deep learning inference.


.. toctree::
   :maxdepth: 2
   :caption: Notebooks
   :hidden:

   notebooks_installation


This collection of Python tutorials are written for running on Jupyter notebooks.
The tutorials provide an introduction to the OpenVINO™ toolkit and explain how to
use the Python API and tools for optimized deep learning inference. You can run the
code one section at a time to see how to integrate your application with OpenVINO
libraries.

Notebooks with |binder logo| and |colab logo| buttons can be run without installing anything.
Once you have found the tutorial of your interest, just click the button next to
its name and the Jupyter notebook will start it in a new tab of a browser.

.. note::

   `Binder <https://mybinder.org/>`__ and `Google Colab <https://colab.research.google.com/>`__
   are free online services with limited resources. For the best performance
   and more control, you should run the notebooks locally. Follow the
   :doc:`Installation Guide <notebooks_installation>` in order to get information
   on how to run and manage the notebooks on your machine.


More examples along with additonal details regarding OpenVINO Notebooks are available in
OpenVINO™ Notebooks `Github Repository. <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md>`__

The Jupyter notebooks are categorized into following classes:

-  `Recommended Model Demos <#recommended-model-demos>`__
-  `First steps with OpenVINO <#first-steps-with-openvino>`__
-  `Convert & Optimize <#convert-optimize>`__
-  `Model Training <#model-training>`__
-  `Live Demos <#live-demos>`__



.. tab-set::

   .. tab-item:: Recommended Model Demos
      :sync: recommended-demos

      Demos that demonstrate inference on a particular model. The following tutorials are guaranteed to provide a great experience with inference in OpenVINO:

      .. showcase::
         :title: 251-tiny-sd-image-generation
         :img: https://user-images.githubusercontent.com/29454499/260904650-274fc2f9-24d2-46a3-ac3d-d660ec3c9a19.png

         Image Generation with Tiny-SD and OpenVINO™.

      .. showcase::
         :title: 250-music-generation
         :img: https://user-images.githubusercontent.com/76463150/260439306-81c81c8d-1f9c-41d0-b881-9491766def8e.png

         Controllable Music Generation with MusicGen and OpenVINO™.

      .. showcase::
         :title: 248-stable-diffusion-xl
         :img: https://user-images.githubusercontent.com/29454499/258651862-28b63016-c5ff-4263-9da8-73ca31100165.jpeg

         Image generation with Stable Diffusion XL and OpenVINO™.

      .. showcase::
         :title: 240-dolly-2-instruction-following
         :img: https://github-production-user-asset-6210df.s3.amazonaws.com/29454499/237160118-e881f4a4-fcc8-427a-afe1-7dd80aebd66e.png

         Instruction following using Databricks Dolly 2.0 and OpenVINO™.

      .. dropdown:: Explore more Model Demos here.
         :class-container: ovnotebooks

         .. showcase::
            :title: 239-image-bind-convert
            :img: https://user-images.githubusercontent.com/29454499/240364108-39868933-d221-41e6-9b2e-dac1b14ef32f.png

            Binding multimodal data, using ImageBind and OpenVINO™.

         .. showcase::
            :title: 238-deep-floyd-if
            :img: https://user-images.githubusercontent.com/29454499/241643886-dfcf3c48-8d50-4730-ae28-a21595d9504f.png

            Text-to-image generation with DeepFloyd IF and OpenVINO™.

         .. showcase::
            :title: 237-segment-anything
            :img: https://user-images.githubusercontent.com/29454499/231468849-1cd11e68-21e2-44ed-8088-b792ef50c32d.png

            Prompt based object segmentation mask generation, using Segment Anything and OpenVINO™.

         .. showcase::
            :title: 236-stable-diffusion-v2-infinite-zoom
            :img: https://user-images.githubusercontent.com/29454499/228882108-25c1f65d-4c23-4e1d-8ba4-f6164280a3e3.gif

            Text-to-image generation and Infinite Zoom with Stable Diffusion v2 and OpenVINO™.

         .. showcase::
            :title: 235-controlnet-stable-diffusion
            :img: https://user-images.githubusercontent.com/29454499/224541412-9d13443e-0e42-43f2-8210-aa31820c5b44.png

            A text-to-image generation with ControlNet Conditioning and OpenVINO™.

         .. showcase::
            :title: 233-blip-visual-language-processing
            :img: https://user-images.githubusercontent.com/29454499/221933762-4ff32ecb-5e5d-4484-80e1-e9396cb3c511.png

            Visual question answering and image captioning using BLIP and OpenVINO™.

         .. showcase::
            :title: 231-instruct-pix2pix-image-editing
            :img: https://user-images.githubusercontent.com/29454499/219943222-d46a2e2d-d348-4259-8431-37cf14727eda.png

            Image editing with InstructPix2Pix.

         .. showcase::
            :title: 230-yolov8-optimization
            :img: https://user-images.githubusercontent.com/29454499/212105105-f61c8aab-c1ff-40af-a33f-d0ed1fccc72e.png

            Optimize YOLOv8, using NNCF PTQ API.

         .. showcase::
            :title: 228-clip-zero-shot-convert
            :img: https://camo.githubusercontent.com/8beb0eedc6a3bcafc397399d55a7e7da4184c1c799e6351a07a7c4aef534ffc4/         68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f32393435343439392f3230373737333438312d64373763616366382d366364632d343736352d613331622d6131363639343736643632302e706e67

            Perform Zero-shot image classification with CLIP and OpenVINO.

         .. showcase::
            :title: 227-whisper-subtitles-generation
            :img: https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png

            Generate subtitles for video with OpenAI Whisper and OpenVINO.

         .. showcase::
            :title: 205-vision-background-removal
            :img: https://user-images.githubusercontent.com/15709723/125184237-f4b6cd00-e1d0-11eb-8e3b-d92c9a728372.png

            Remove and replace the background in an image using salient object detection.

         .. showcase::
            :title: 209-handwritten-ocrn
            :img: https://user-images.githubusercontent.com/36741649/132660640-da2211ec-c389-450e-8980-32a75ed14abb.png

            OCR for handwritten simplified Chinese and Japanese.

         .. showcase::
            :title: 211-speech-to-text
            :img: https://user-images.githubusercontent.com/36741649/140987347-279de058-55d7-4772-b013-0f2b12deaa61.png

            Run inference on speech-to-text recognition model.

         .. showcase::
            :title: 215-image-inpainting
            :img: https://user-images.githubusercontent.com/4547501/167121084-ec58fbdb-b269-4de2-9d4c-253c5b95de1e.png

            Fill missing pixels with image in-painting.

         .. showcase::
            :title: 218-vehicle-detection-and-recognition
            :img: https://user-images.githubusercontent.com/47499836/163544861-fa2ad64b-77df-4c16-b065-79183e8ed964.png

            Use pre-trained models to detect and recognize vehicles and their attributes with OpenVINO.

         .. showcase::
            :title: 201-vision-monodepth
            :img: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif

            Monocular depth estimation with images and video.

         .. showcase::
            :title: 202-vision-superresolution-image
            :img: https://user-images.githubusercontent.com/36741649/170005347-e4409f9e-ec34-416b-afdf-a9d8185929ca.jpg

            Upscale raw images with a super resolution model.

         .. showcase::
            :title: 202-vision-superresolution-video
            :img: https://user-images.githubusercontent.com/15709723/127269258-a8e2c03e-731e-4317-b5b2-ed2ee767ff5e.gif

            Turn 360p into 1080p video using a super resolution model.

         .. showcase::
            :title: 203-meter-reader
            :img: https://user-images.githubusercontent.com/91237924/166135627-194405b0-6c25-4fd8-9ad1-83fb3a00a081.jpg

            PaddlePaddle pre-trained models to read industrial meter’s value.

         .. showcase::
            :title: 204-segmenter-semantic-segmentation
            :img: https://user-images.githubusercontent.com/61357777/223854308-d1ac4a39-cc0c-4618-9e4f-d9d4d8b991e8.jpg

            Semantic segmentation with OpenVINO™ using Segmenter.

         .. showcase::
            :title: 206-vision-paddlegan-anime
            :img: https://user-images.githubusercontent.com/15709723/127788059-1f069ae1-8705-4972-b50e-6314a6f36632.jpeg

            Turn an image into anime using a GAN.

         .. showcase::
            :title: 207-vision-paddlegan-superresolution
            :img: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg

            Upscale small images with superresolution using a PaddleGAN model.

         .. showcase::
            :title: 208-optical-character-recognition
            :img: https://user-images.githubusercontent.com/36741649/129315292-a37266dc-dfb2-4749-bca5-2ac9c1e93d64.jpg

            Annotate text on images using text recognition resnet.

         .. showcase::
            :title: 212-pyannote-speaker-diarization
            :img: https://user-images.githubusercontent.com/29454499/218432101-0bd0c424-e1d8-46af-ba1d-ee29ed6d1229.png

            Run inference on speaker diarization pipeline.

         .. showcase::
            :title: 210-slowfast-video-recognition
            :img: https://github.com/facebookresearch/SlowFast/raw/main/demo/ava_demo.gif

            Video Recognition using SlowFast and OpenVINO™

         .. showcase::
            :title: 218-vehicle-detection-and-recognition
            :img: https://user-images.githubusercontent.com/47499836/163544861-fa2ad64b-77df-4c16-b065-79183e8ed964.png

            Use pre-trained models to detect and recognize vehicles and their attributes with OpenVINO.

         .. showcase::
            :title: 213-question-answering
            :img: https://user-images.githubusercontent.com/4547501/152571639-ace628b2-e3d2-433e-8c28-9a5546d76a86.gif

            Answer your questions basing on a context.

         .. showcase::
            :title: 214-grammar-correction
            :img: _static/images/openvino_notebooks_thumbnail.png

            Grammatical error correction with OpenVINO.

         .. showcase::
            :title: 216-attention-center
            :img: _static/images/openvino_notebooks_thumbnail.png

            The attention center model with OpenVINO™

         .. showcase::
            :title: 217-vision-deblur
            :img: https://user-images.githubusercontent.com/41332813/158430181-05d07f42-cdb8-4b7a-b7dc-e7f7d9391877.png

            Deblur images with DeblurGAN-v2.

         .. showcase::
            :title: 219-knowledge-graphs-conve
            :img: _static/images/openvino_notebooks_thumbnail.png

            Optimize the knowledge graph embeddings model (ConvE) with OpenVINO.

         .. showcase::
            :title: 220-cross-lingual-books-alignment
            :img: https://user-images.githubusercontent.com/51917466/254583163-3bb85143-627b-4f02-b628-7bef37823520.png

            Cross-lingual Books Alignment With Transformers and OpenVINO™

         .. showcase::
            :title: 221-machine-translation
            :img: _static/images/openvino_notebooks_thumbnail.png

            Real-time translation from English to German.

         .. showcase::
            :title: 222-vision-image-colorization
            :img: https://user-images.githubusercontent.com/18904157/166343139-c6568e50-b856-4066-baef-5cdbd4e8bc18.png

            Use pre-trained models to colorize black & white images using OpenVINO.

         .. showcase::
            :title: 223-text-prediction
            :img: https://user-images.githubusercontent.com/91228207/185105225-0f996b0b-0a3b-4486-872d-364ac6fab68b.png

            Use pre-trained models to perform text prediction on an input sequence.

         .. showcase::
            :title: 224-3D-segmentation-point-clouds
            :img: https://user-images.githubusercontent.com/91237924/185752178-3882902c-907b-4614-b0e6-ea1de08bf3ef.png

            Process point cloud data and run 3D Part Segmentation with OpenVINO.

         .. showcase::
            :title: 225-stable-diffusion-text-to-image
            :img: https://user-images.githubusercontent.com/15709723/200945747-1c584e5c-b3f2-4e43-b1c1-e35fd6edc2c3.png

            Text-to-image generation with Stable Diffusion method.

         .. showcase::
            :title: 226-yolov7-optimization
            :img: https://raw.githubusercontent.com/WongKinYiu/yolov7/main/figure/horses_prediction.jpg

            Optimize YOLOv7, using NNCF PTQ API.

         .. showcase::
            :title: 227-whisper-subtitles-generation
            :img: https://user-images.githubusercontent.com/29454499/204548693-1304ef33-c790-490d-8a8b-d5766acb6254.png

            Generate subtitles for video with OpenAI Whisper and OpenVINO.

         .. showcase::
            :title: 228-clip-zero-shot-convert
            :img: https://camo.githubusercontent.com/8beb0eedc6a3bcafc397399d55a7e7da4184c1c799e6351a07a7c4aef534ffc4/      68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f32393435343439392f3230373737333438312d64373763616366382d366364632d343736352d613331622d6131363639343736643632302e706e67

            Zero-shot Image Classification with OpenAI CLIP and OpenVINO™.

         .. showcase::
            :title: 228-clip-zero-shot-quantize
            :img: https://user-images.githubusercontent.com/29454499/207795060-437b42f9-e801-4332-a91f-cc26471e5ba2.png

            Post-Training Quantization of OpenAI CLIP model with NNCF.

         .. showcase::
            :title: 229-distilbert-sequence-classification
            :img: https://user-images.githubusercontent.com/95271966/206130638-d9847414-357a-4c79-9ca7-76f4ae5a6d7f.png

            Sequence classification with OpenVINO.

         .. showcase::
            :title: 232-clip-language-saliency-map
            :img: https://user-images.githubusercontent.com/29454499/218967961-9858efd5-fff2-4eb0-bde9-60852f4b31cb.JPG

            Language-visual saliency with CLIP and OpenVINO™.

         .. showcase::
            :title: 234-encodec-audio-compression
            :img: https://github.com/facebookresearch/encodec/raw/main/thumbnail.png

            Audio compression with EnCodec and OpenVINO™.

         .. showcase::
            :title: 235-controlnet-stable-diffusion
            :img: https://user-images.githubusercontent.com/29454499/224541412-9d13443e-0e42-43f2-8210-aa31820c5b44.png

            A text-to-image generation with ControlNet Conditioning and OpenVINO™.

         .. showcase::
            :title: 236-stable-diffusion-v2-infinite-zoom
            :img: https://user-images.githubusercontent.com/29454499/228882108-25c1f65d-4c23-4e1d-8ba4-f6164280a3e3.gif

            Text-to-image generation and Infinite Zoom with Stable Diffusion v2 and OpenVINO™.

         .. showcase::
            :title: 236-stable-diffusion-v2-optimum-demo-comparison
            :img: https://user-images.githubusercontent.com/1720147/229231281-065641fd-53ea-4940-8c52-b1eebfbaa7fa.png

            Stable Diffusion v2.1 using Optimum-Intel OpenVINO and multiple Intel Hardware

         .. showcase::
            :title: 236-stable-diffusion-v2-optimum-demo
            :img: https://user-images.githubusercontent.com/1720147/229231281-065641fd-53ea-4940-8c52-b1eebfbaa7fa.png

            Stable Diffusion v2.1 using Optimum-Intel OpenVINO.

         .. showcase::
            :title: 236-stable-diffusion-v2-text-to-image-demo
            :img: https://user-images.githubusercontent.com/1720147/229231281-065641fd-53ea-4940-8c52-b1eebfbaa7fa.png

            Stable Diffusion Text-to-Image Demo.

         .. showcase::
            :title: 236-stable-diffusion-v2-text-to-image
            :img: https://user-images.githubusercontent.com/29454499/228882108-25c1f65d-4c23-4e1d-8ba4-f6164280a3e3.gif

            Text-to-image generation with Stable Diffusion v2 and OpenVINO™.

         .. showcase::
            :title: 241-riffusion-text-to-music
            :img: https://user-images.githubusercontent.com/29454499/244291912-bbc6e08c-c0a9-41fe-bc2d-5f89a0d2463b.png

            Text-to-Music generation using Riffusion and OpenVINO™.

         .. showcase::
            :title: 242-freevc-voice-conversion
            :img: https://user-images.githubusercontent.com/47499836/163544861-fa2ad64b-77df-4c16-b065-79183e8ed964.png

            High-Quality Text-Free One-Shot Voice Conversion with FreeVC and OpenVINO™

         .. showcase::
            :title: 243-tflite-selfie-segmentation
            :img: https://user-images.githubusercontent.com/29454499/251085926-14045ebc-273b-4ccb-b04f-82a3f7811b87.gif

            Selfie Segmentation using TFLite and OpenVINO™.

         .. showcase::
            :title: 244-named-entity-recognition
            :img: _static/images/openvino_notebooks_thumbnail.png

            Named entity recognition with OpenVINO™.

         .. showcase::
            :title: 245-typo-detector
            :img: https://user-images.githubusercontent.com/80534358/224564463-ee686386-f846-4b2b-91af-7163586014b7.png

            English Typo Detection in sentences with OpenVINO™.

         .. showcase::
            :title: 246-depth-estimation-videpth
            :img: https://raw.githubusercontent.com/alexklwong/void-dataset/master/figures/void_samples.png

            Monocular Visual-Inertial Depth Estimation with OpenVINO™.

         .. showcase::
            :title: 247-code-language-id
            :img: _static/images/openvino_notebooks_thumbnail.png

            Identify the programming language used in an arbitrary code snippet.

         .. showcase::
            :title: 249-oneformer-segmentation
            :img: https://camo.githubusercontent.com/f46c3642d3266e9d56d8ea8a943e67825597de3ff51698703ea2ddcb1086e541/         68747470733a2f2f6769746875622d70726f64756374696f6e2d757365722d61737365742d3632313064662e73332e616d617a6f6e6177732e636f6d2f37363136313235362f3235383634303731332d66383031626430392d653932372d346162642d616132662d3939393064653463616638642e676966

            Universal segmentation with OneFormer and OpenVINO™.

         .. showcase::
            :title: 252-fastcomposer-image-generation
            :img: _static/images/openvino_notebooks_thumbnail.png

            Image generation with FastComposer and OpenVINO™.

         .. showcase::
            :title: 253-zeroscope-text2video
            :img: https://user-images.githubusercontent.com/76161256/261102399-500956d5-4aac-4710-a77c-4df34bcda3be.gif

            Text-to video synthesis with ZeroScope and OpenVINO™.

   .. tab-item:: First steps with OpenVINO
      :sync: first-steps-openvino

      Brief tutorials that demonstrate how to use Python API for inference in OpenVINO.

      .. showcase::
         :title: 001-hello-world
         :img: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg

         Classify an image with OpenVINO.

      .. showcase::
         :title: 002-openvino-api
         :img: _static/images/openvino_notebooks_thumbnail.png

         Learn the OpenVINO Python API.

      .. showcase::
         :title: 003-hello-segmentation
         :img: https://user-images.githubusercontent.com/15709723/128290691-e2eb875c-775e-4f4d-a2f4-15134044b4bb.png

         Semantic segmentation with OpenVINO.

      .. showcase::
         :title: 004-hello-detection
         :img: https://user-images.githubusercontent.com/36741649/128489910-316aec49-4892-46f1-9e3c-b9d3646ef278.jpg

         Text detection with OpenVINO.

   .. tab-item:: Convert & Optimize
      :sync: convert-optimize-tutorials


      Tutorials that explain how to optimize and quantize models with OpenVINO tools.


      .. showcase::
         :title: 101-tensorflow-classification-to-openvino
         :img: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg

         Convert TensorFlow models to OpenVINO IR.

      .. showcase::
         :title: 102-pytorch-to-openvino
         :img: _static/images/openvino_notebooks_thumbnail.png

         Convert PyTorch models to OpenVINO IR.

      .. showcase::
         :title: 103-paddle-onnx-to-openvino
         :img: _static/images/openvino_notebooks_thumbnail.png

         Convert PaddlePaddle models to OpenVINO IR.

      .. showcase::
         :title: 121-convert-to-openvino
         :img: _static/images/openvino_notebooks_thumbnail.png

         Learn OpenVINO model conversion API.

      .. dropdown:: Explore more Convert & Optimize notebooks.
         :class-container: ovnotebooks

         .. showcase::
            :title: 102-pytorch-onnx-to-openvino
            :img: _static/images/openvino_notebooks_thumbnail.png

            Convert PyTorch models to OpenVINO IR.

         .. showcase::
            :title: 104-model-tools
            :img: _static/images/openvino_notebooks_thumbnail.png

            Download, convert and benchmark models from Open Model Zoo.

         .. showcase::
            :title: 105-language-quantize-bert
            :img: _static/images/openvino_notebooks_thumbnail.png

            Optimize and quantize a pre-trained BERT model.

         .. showcase::
            :title: 106-auto-device
            :img: _static/images/openvino_notebooks_thumbnail.png

            Demonstrates how to use AUTO Device.

         .. showcase::
            :title: 107-speech-recognition-quantization-data2vec
            :img: _static/images/openvino_notebooks_thumbnail.png

            Optimize and quantize a pre-trained Data2Vec speech model.

         .. showcase::
            :title: 107-speech-recognition-quantization-wav2vec2
            :img: _static/images/openvino_notebooks_thumbnail.png

            Optimize and quantize a pre-trained Wav2Vec2 speech model.

         .. showcase::
            :title: 108-gpu-device
            :img: _static/images/openvino_notebooks_thumbnail.png

            Working with GPUs in OpenVINO™

         .. showcase::
            :title: 109-latency-tricks
            :img: _static/images/openvino_notebooks_thumbnail.png

            Performance tricks for latency mode in OpenVINO™.

         .. showcase::
            :title: 109-throughput-tricks
            :img: _static/images/openvino_notebooks_thumbnail.png

            Performance tricks for throughput mode in OpenVINO™.

         .. showcase::
            :title: 110-ct-scan-live-inference
            :img: _static/images/openvino_notebooks_thumbnail.png

            Live inference of a kidney segmentation model and benchmark CT-scan data with OpenVINO.

         .. showcase::
            :title: 110-ct-segmentation-quantize-nncf
            :img: _static/images/openvino_notebooks_thumbnail.png

            Quantize a kidney segmentation model and show live inference.

         .. showcase::
            :title: 111-yolov5-quantization-migration
            :img: _static/images/openvino_notebooks_thumbnail.png

            Migrate YOLOv5 POT API based quantization pipeline on Neural Network Compression Framework (NNCF).

         .. showcase::
            :title: 112-pytorch-post-training-quantization-nncf
            :img: _static/images/openvino_notebooks_thumbnail.png

            Use Neural Network Compression Framework (NNCF) to quantize PyTorch model in post-training mode (without model fine-tuning).

         .. showcase::
            :title: 113-image-classification-quantization
            :img: _static/images/openvino_notebooks_thumbnail.png

            Quantize MobileNet image classification.

         .. showcase::
            :title: 115-async-api
            :img: _static/images/openvino_notebooks_thumbnail.png

            Use asynchronous execution to improve data pipelining.

         .. showcase::
            :title: 116-sparsity-optimization
            :img: _static/images/openvino_notebooks_thumbnail.png

            Improve performance of sparse Transformer models.

         .. showcase::
            :title: 117-model-server
            :img: _static/images/openvino_notebooks_thumbnail.png

            Improve performance of sparse Transformer models.

         .. showcase::
            :title: 118-optimize-preprocessing
            :img: _static/images/openvino_notebooks_thumbnail.png

            Improve performance of image preprocessing step.

         .. showcase::
            :title: 119-tflite-to-openvino
            :img: _static/images/openvino_notebooks_thumbnail.png

            Convert TensorFlow Lite models to OpenVINO IR.

         .. showcase::
            :title: 120-tensorflow-object-detection-to-openvino
            :img: _static/images/openvino_notebooks_thumbnail.png

            Convert TensorFlow Object Detection models to OpenVINO IR.

         .. showcase::
            :title: 122-speech-recognition-quantization-wav2vec2
            :img: _static/images/openvino_notebooks_thumbnail.png

            Quantize Speech Recognition Models with accuracy control using NNCF PTQ API.

         .. showcase::
            :title: 122-yolov8-quantization-with-accuracy-control
            :img: _static/images/openvino_notebooks_thumbnail.png

            Convert and Optimize YOLOv8 with OpenVINO™.

   .. tab-item:: Model Training
      :sync: model-training-tutorials

      Tutorials that include code to train neural networks.

      .. showcase::
         :title: 301-tensorflow-training-openvino
         :img: https://user-images.githubusercontent.com/15709723/127779607-8fa34947-1c35-4260-8d04-981c41a2a2cc.png

         Train a flower classification model from TensorFlow, then convert to OpenVINO IR.

      .. showcase::
         :title: 301-tensorflow-training-openvino-nncf
         :img: _static/images/openvino_notebooks_thumbnail.png

         Use Neural Network Compression Framework (NNCF) to quantize model from TensorFlow

      .. showcase::
         :title: 302-pytorch-quantization-aware-training
         :img: _static/images/openvino_notebooks_thumbnail.png

         Use Neural Network Compression Framework (NNCF) to quantize PyTorch model.

      .. showcase::
         :title: 305-tensorflow-quantization-aware-training
         :img: _static/images/openvino_notebooks_thumbnail.png

         Use Neural Network Compression Framework (NNCF) to quantize TensorFlow model.

   .. tab-item:: Live Demos
      :sync: live-demos-tutorials

      Live inference demos that run on a webcam or video files.

      .. showcase::
         :title: 401-object-detection-webcam
         :img: https://user-images.githubusercontent.com/4547501/141471665-82b28c86-cf64-4bfe-98b3-c314658f2d96.gif

         Object detection with a webcam or video file.

      .. showcase::
         :title: 402-pose-estimation-webcam
         :img: https://user-images.githubusercontent.com/4547501/138267961-41d754e7-59db-49f6-b700-63c3a636fad7.gif

         Human pose estimation with a webcam or video file.

      .. showcase::
         :title: 403-action-recognition-webcam
         :img: https://user-images.githubusercontent.com/10940214/151552326-642d6e49-f5a0-4fc1-bf14-ae3f457e1fec.gif

         Human action recognition with a webcam or video file.

      .. showcase::
         :title: 404-style-transfer-webcam
         :img: https://user-images.githubusercontent.com/109281183/203772234-f17a0875-b068-43ef-9e77-403462fde1f5.gif

         Style transfer with a webcam or video file.

      .. dropdown:: Explore more Live Demos here.
         :class-container: ovnotebooks

         .. showcase::
            :title: 405-paddle-ocr-webcam
            :img: https://raw.githubusercontent.com/yoyowz/classification/master/images/paddleocr.gif

            OCR with a webcam or video file.

         .. showcase::
            :title: 406-3D-pose-estimation-webcam
            :img: https://user-images.githubusercontent.com/42672437/183292131-576cc05a-a724-472c-8dc9-f6bc092190bf.gif

            3D display of human pose estimation with a webcam or video file.

         .. showcase::
            :title: 407-person-tracking-webcam
            :img: https://user-images.githubusercontent.com/91237924/210479548-b70dbbaa-5948-4e49-b48e-6cb6613226da.gif

            Person tracking with a webcam or video file.



.. note::
   If there are any issues while running the notebooks, refer to the **Troubleshooting** and **FAQ** sections in the :doc:`Installation Guide <notebooks_installation>` or start a GitHub
   `discussion <https://github.com/openvinotoolkit/openvino_notebooks/discussions>`__.


Additional Resources
######################

* `OpenVINO™ Notebooks - Github Repository <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/README.md>`_
* `Binder documentation <https://mybinder.readthedocs.io/en/latest/>`_
* `Google Colab <https://colab.research.google.com/>`__


@endsphinxdirective
