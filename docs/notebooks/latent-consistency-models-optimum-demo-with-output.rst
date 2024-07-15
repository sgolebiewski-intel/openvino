LatentConsistencyModelusingOptimum-IntelOpenVINO
=====================================================

ThisnotebookprovidesinstructionshowtorunLatentConsistencyModel
(LCM).ItallowstosetupstandardHuggingFacediffuserspipelineand
OptimumIntelpipelineoptimizedforIntelhardwareincludingCPUand
GPU.RunninginferenceonCPUandGPUitiseasytocompareperformance
andtimerequiredtogenerateanimageforprovidedprompt.Thenotebook
canbealsousedonotherIntelhardwarewithminimalorno
modifications.

|image0|

OptimumIntelisaninterfacefromHuggingFacebetweenbothdiffusers
andtransformerslibrariesandvarioustoolsprovidedbyIntelto
acceleratepipelinesonIntelhardware.Itallowstoperform
quantizationofthemodelshostedonHuggingFace.Inthisnotebook
OpenVINOisusedforAI-inferenceaccelerationasabackendforOptimum
Intel!

FormoredetailspleaserefertoOptimumIntelrepository
https://github.com/huggingface/optimum-intel

LCMsarethenextgenerationofgenerativemodelsafterLatentDiffusion
Models(LDMs).Theyareproposedtoovercometheslowiterativesampling
processofLatentDiffusionModels(LDMs),enablingfastinferencewith
minimalsteps(from2to4)onanypre-trainedLDMs(e.g. Stable
Diffusion).ToreadmoreaboutLCMpleasereferto
https://latent-consistency-models.github.io/

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Fullprecisionmodelonthe
CPU<#using-full-precision-model-in-cpu-with-latentconsistencymodelpipeline>`__
-`RunninginferenceusingOptimumIntel
OVLatentConsistencyModelPipeline<#running-inference-using-optimum-intel-ovlatentconsistencymodelpipeline>`__

..|image0|image::https://github.com/openvinotoolkit/openvino_notebooks/assets/10940214/1858dae4-72fd-401e-b055-66d503d82446

Prerequisites
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Installrequiredpackages

..code::ipython3

%pipinstall-q"openvino>=2023.3.0"
%pipinstall-q"onnx>=1.11.0"
%pipinstall-q"optimum-intel[diffusers]@git+https://github.com/huggingface/optimum-intel.git""ipywidgets""torch>=2.1""transformers>=4.33.0"--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


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

Note:FormoredetailsaboutGPUwithOpenVINOvisitthis
`link<https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>`__.
IfyouhavebeenfacinganyissueinUbuntu20.04orWindows11read
this
`blog<https://blog.openvino.ai/blog-posts/install-gpu-drivers-windows-ubuntu>`__.

..code::ipython3

importopenvinoasov

core=ov.Core()
devices=core.available_devices

fordeviceindevices:
device_name=core.get_property(device,"FULL_DEVICE_NAME")
print(f"{device}:{device_name}")


..parsed-literal::

CPU:Intel(R)Core(TM)i9-10920XCPU@3.50GHz


UsingfullprecisionmodelinCPUwith``LatentConsistencyModelPipeline``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

StandardpipelinefortheLatentConsistencyModel(LCM)fromDiffusers
libraryisusedhere.Formoreinformationpleasereferto
https://huggingface.co/docs/diffusers/en/api/pipelines/latent_consistency_models

..code::ipython3

fromdiffusersimportLatentConsistencyModelPipeline
importgc

pipeline=LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")


..parsed-literal::

2024-07-1301:00:03.964344:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:00:04.000289:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:00:04.671482:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



..parsed-literal::

Loadingpipelinecomponents...:0%||0/7[00:00<?,?it/s]


..code::ipython3

prompt="Acutesquirrelintheforest,portrait,8k"

image=pipeline(prompt=prompt,num_inference_steps=4,guidance_scale=8.0,height=512,width=512).images[0]
image.save("image_standard_pipeline.png")
image



..parsed-literal::

0%||0/4[00:00<?,?it/s]




..image::latent-consistency-models-optimum-demo-with-output_files/latent-consistency-models-optimum-demo-with-output_8_1.png



..code::ipython3

delpipeline
gc.collect();

Selectinferencedevicefortext-to-imagegeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..code::ipython3

importipywidgetsaswidgets

core=ov.Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



RunninginferenceusingOptimumIntel``OVLatentConsistencyModelPipeline``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

AcceleratinginferenceofLCMusingIntelOptimumwithOpenVINObackend.
Formoreinformationpleasereferto
https://huggingface.co/docs/optimum/intel/inference#latent-consistency-models.
ThepretrainedmodelinthisnotebookisavailableonHuggingFacein
FP32precisionandincaseifCPUisselectedasadevice,then
inferencerunswithfullprecision.ForGPUacceleratedAI-inferenceis
supportedforFP16datatypeandFP32precisionforGPUmayproducehigh
memoryfootprintandlatency.Therefore,defaultprecisionforGPUin
OpenVINOisFP16.OpenVINOGPUPluginconvertsFP32toFP16onthefly
andthereisnoneedtodoitmanually

..code::ipython3

fromoptimum.intel.openvinoimportOVLatentConsistencyModelPipeline
frompathlibimportPath

ifnotPath("./openvino_ir").exists():
ov_pipeline=OVLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",height=512,width=512,export=True,compile=False)
ov_pipeline.save_pretrained("./openvino_ir")
else:
ov_pipeline=OVLatentConsistencyModelPipeline.from_pretrained("./openvino_ir",export=False,compile=False)

ov_pipeline.reshape(batch_size=1,height=512,width=512,num_images_per_prompt=1)


..parsed-literal::

Frameworknotspecified.Usingpttoexportthemodel.
Keywordarguments{'subfolder':'','token':None,'trust_remote_code':False}arenotexpectedbyStableDiffusionPipelineandwillbeignored.



..parsed-literal::

Loadingpipelinecomponents...:0%||0/7[00:00<?,?it/s]


..parsed-literal::

UsingframeworkPyTorch:2.3.1+cpu


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
UsingframeworkPyTorch:2.3.1+cpu
UsingframeworkPyTorch:2.3.1+cpu
UsingframeworkPyTorch:2.3.1+cpu




..parsed-literal::

OVLatentConsistencyModelPipeline{
"_class_name":"OVLatentConsistencyModelPipeline",
"_diffusers_version":"0.24.0",
"feature_extractor":[
"transformers",
"CLIPImageProcessor"
],
"requires_safety_checker":true,
"safety_checker":[
"stable_diffusion",
"StableDiffusionSafetyChecker"
],
"scheduler":[
"diffusers",
"LCMScheduler"
],
"text_encoder":[
"optimum",
"OVModelTextEncoder"
],
"text_encoder_2":[
null,
null
],
"tokenizer":[
"transformers",
"CLIPTokenizer"
],
"unet":[
"optimum",
"OVModelUnet"
],
"vae_decoder":[
"optimum",
"OVModelVaeDecoder"
],
"vae_encoder":[
"optimum",
"OVModelVaeEncoder"
]
}



..code::ipython3

ov_pipeline.to(device.value)
ov_pipeline.compile()


..parsed-literal::

Compilingthevae_decodertoCPU...
CompilingtheunettoCPU...
Compilingthetext_encodertoCPU...
Compilingthevae_encodertoCPU...


..code::ipython3

prompt="Acutesquirrelintheforest,portrait,8k"

image_ov=ov_pipeline(prompt=prompt,num_inference_steps=4,guidance_scale=8.0,height=512,width=512).images[0]
image_ov.save("image_opt.png")
image_ov



..parsed-literal::

0%||0/4[00:00<?,?it/s]




..image::latent-consistency-models-optimum-demo-with-output_files/latent-consistency-models-optimum-demo-with-output_15_1.png



..code::ipython3

delov_pipeline
gc.collect();
