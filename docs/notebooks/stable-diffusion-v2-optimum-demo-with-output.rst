StableDiffusionv2.1usingOptimum-IntelOpenVINOandmultipleIntelHardware
==============================================================================

Thisnotebookwillprovideyouawaytoseedifferentprecisionmodels
performingindifferenthardware.Thisnotebookwasdoneforshowing
casetheuseofOptimum-Intel-OpenVINOanditisnotoptimizedfor
runningmultipletimes.

|image0|

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`ShowingInfoAvailableDevices<#showing-info-available-devices>`__
-`ConfigureInferencePipeline<#configure-inference-pipeline>`__
-`Usingfullprecisionmodelinchoicedevicewith
OVStableDiffusionPipeline<#using-full-precision-model-in-choice-device-with-ovstablediffusionpipeline>`__

..|image0|image::https://github.com/openvinotoolkit/openvino_notebooks/assets/10940214/1858dae4-72fd-401e-b055-66d503d82446

OptimumIntelistheinterfacebetweentheTransformersandDiffusers
librariesandthedifferenttoolsandlibrariesprovidedbyIntelto
accelerateend-to-endpipelinesonIntelarchitectures.Moredetailsin
this
`repository<https://github.com/huggingface/optimum-intel#openvino>`__.

``Note:Wesuggestyoutocreateadifferentenvironmentandrunthefollowinginstallationcommandthere.``

..code::ipython3

%pipinstall-q"optimum-intel[openvino,diffusers]@git+https://github.com/huggingface/optimum-intel.git""ipywidgets""transformers>=4.33.0""torch>=2.1"--extra-index-urlhttps://download.pytorch.org/whl/cpu

StableDiffusionpipelineshouldbrings6elementstogether,atext
encodermodelwithatokenizer,aUNetmodelwithandscheduler,andan
AutoencoderwithDecoderandEncodermodels.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/10940214/e166f225-1220-44aa-a987-84471e03947d
:alt:image

image

Thebasemodelusedforthisexampleisthe
stabilityai/stable-diffusion-2-1-base.Thismodelwasconvertedto
OpenVINOformat,foracceleratedinferenceonCPUorIntelGPUwith
OpenVINO’sintegrationintoOptimum.

..code::ipython3

importwarnings

warnings.filterwarnings("ignore")

ShowingInfoAvailableDevices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The``available_devices``propertyshowstheavailabledevicesinyour
system.The“FULL_DEVICE_NAME”optionto``ie.get_property()``showsthe
nameofthedevice.CheckwhatistheIDnameforthediscreteGPU,if
youhaveintegratedGPU(iGPU)anddiscreteGPU(dGPU),itwillshow
``device_name="GPU.0"``foriGPUand``device_name="GPU.1"``fordGPU.
IfyoujusthaveeitheraniGPUordGPUthatwillbeassignedto
``"GPU"``

..code::ipython3

importopenvinoasov

core=ov.Core()
devices=core.available_devices

fordeviceindevices:
device_name=core.get_property(device,"FULL_DEVICE_NAME")
print(f"{device}:{device_name}")


..parsed-literal::

CPU:Intel(R)Core(TM)Ultra7155H
GPU:Intel(R)Arc(TM)Graphics(iGPU)
NPU:Intel(R)AIBoost


ConfigureInferencePipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO.

..code::ipython3

importipywidgetsaswidgets

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','GPU','NPU','AUTO'),value='GPU')



Usingfullprecisionmodelinchoicedevicewith``OVStableDiffusionPipeline``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromoptimum.intel.openvinoimportOVStableDiffusionPipeline

#downloadthepre-convertedSDv2.1modelfromHuggingFaceHub
name="helenai/stabilityai-stable-diffusion-2-1-base-ov"
ov_pipe=OVStableDiffusionPipeline.from_pretrained(name,compile=False)
ov_pipe.reshape(batch_size=1,height=512,width=512,num_images_per_prompt=1)
ov_pipe.to(device.value)
ov_pipe.compile()

..code::ipython3

importgc

#Generateanimage.
prompt="redcarinsnowyforest,epicvista,beautifullandscape,4k,8k"
output_ov=ov_pipe(prompt,num_inference_steps=17,output_type="pil").images[0]
output_ov.save("image.png")
output_ov



..parsed-literal::

0%||0/18[00:00<?,?it/s]




..image::stable-diffusion-v2-optimum-demo-with-output_files/stable-diffusion-v2-optimum-demo-with-output_11_1.png



..code::ipython3

delov_pipe
gc.collect()
