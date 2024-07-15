ImageGenerationwithStableDiffusionandIP-Adapter
=====================================================

`IP-Adapter<https://hf.co/papers/2308.06721>`__isaneffectiveand
lightweightadapterthataddsimagepromptingcapabilitiestoa
diffusionmodel.Thisadapterworksbydecouplingthecross-attention
layersoftheimageandtextfeatures.Alltheothermodelcomponents
arefrozenandonlytheembeddedimagefeaturesintheUNetaretrained.
Asaresult,IP-Adapterfilesaretypicallyonly~100MBs.
|ip-adapter-pipe.png|

Inthistutorial,wewillconsiderhowtoconvertandrunStable
DiffusionpipelinewithloadingIP-Adapter.Wewilluse
`stable-diffusion-v1.5<https://huggingface.co/runwayml/stable-diffusion-v1-5>`__
asbasemodelandapplyofficial
`IP-Adapter<https://huggingface.co/h94/IP-Adapter>`__weights.Alsofor
speedupgenerationprocesswewilluse
`LCM-LoRA<https://huggingface.co/latent-consistency/lcm-lora-sdv1-5>`__

..|ip-adapter-pipe.png|image::https://huggingface.co/h94/IP-Adapter/resolve/main/fig1.png

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`PrepareDiffuserspipeline<#prepare-diffusers-pipeline>`__
-`ConvertPyTorchmodels<#convert-pytorch-models>`__

-`ImageEncoder<#image-encoder>`__
-`U-net<#u-net>`__
-`VAEEncoderandDecoder<#vae-encoder-and-decoder>`__
-`TextEncoder<#text-encoder>`__

-`PrepareOpenVINOinference
pipeline<#prepare-openvino-inference-pipeline>`__
-`Runmodelinference<#run-model-inference>`__

-`Selectinferencedevice<#select-inference-device>`__
-`Generationimagevariation<#generation-image-variation>`__
-`Generationconditionedbyimageand
text<#generation-conditioned-by-image-and-text>`__
-`Generationimageblending<#generation-image-blending>`__

-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-q"torch>=2.1"transformersacceleratediffusers"openvino>=2023.3.0""gradio>=4.19"opencv-python"peft==0.6.2"--extra-index-urlhttps://download.pytorch.org/whl/cpu

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


PrepareDiffuserspipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

Firstofall,weshouldcollectallcomponentsofourpipelinetogether.
ToworkwithStableDiffusion,wewilluseHuggingFace
`Diffusers<https://github.com/huggingface/diffusers>`__library.To
experimentwithStableDiffusionmodels,Diffusersexposesthe
`StableDiffusionPipeline<https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation>`__
similartothe`otherDiffusers
pipelines<https://huggingface.co/docs/diffusers/api/pipelines/overview>`__.
Additionally,thepipelinesupportsloadadaptersthatextendStable
Diffusionfunctionalitysuchas`Low-RankAdaptation
(LoRA)<https://huggingface.co/papers/2106.09685>`__,
`PEFT<https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference>`__,
`IP-Adapter<https://ip-adapter.github.io/>`__,and`Textual
Inversion<https://textual-inversion.github.io/>`__.Youcanfindmore
informationaboutsupportedadaptersin`diffusers
documentation<https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters>`__.

Inthistutorial,wewillfocusonip-adapter.IP-Adaptercanbe
integratedintodiffusionpipelineusing``load_ip_adapter``method.
IP-Adapterallowsyoutousebothimageandtexttoconditiontheimage
generationprocess.Foradjustingthetextpromptandimageprompt
conditionratio,wecanuse``set_ip_adapter_scale()``method.Ifyou
onlyusetheimageprompt,youshouldsetthescaleto1.0.Youcan
lowerthescaletogetmoregenerationdiversity,butit’llbeless
alignedwiththeprompt.scale=0.5canachievegoodresultswhenyouuse
bothtextandimageprompts.

Asdiscussedbefore,wewillalsouseLCMLoRAforspeedinggeneration
process.YoucanfindmoreinformationaboutLCMLoRAinthis
`notebook<latent-consistency-models-image-generation-with-output.html>`__.
ForapplyingLCMLoRA,weshoulduse``load_lora_weights``method.
Additionally,LCMrequiresusingLCMSchedulerforefficientgeneration.

..code::ipython3

frompathlibimportPath
fromdiffusersimportAutoPipelineForText2Image
fromtransformersimportCLIPVisionModelWithProjection
fromdiffusers.utilsimportload_image
fromdiffusersimportLCMScheduler


stable_diffusion_id="runwayml/stable-diffusion-v1-5"
ip_adapter_id="h94/IP-Adapter"
ip_adapter_weight_name="ip-adapter_sd15.bin"
lcm_lora_id="latent-consistency/lcm-lora-sdv1-5"
models_dir=Path("model")

load_original_pipeline=notall(
[
(models_dir/model_name).exists()
formodel_namein[
"text_encoder.xml",
"image_encoder.xml",
"unet.xml",
"vae_decoder.xml",
"vae_encoder.xml",
]
]
)


defget_pipeline_components(
stable_diffusion_id,
ip_adapter_id,
ip_adapter_weight_name,
lcm_lora_id,
ip_adapter_scale=0.6,
):
image_encoder=CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter",subfolder="models/image_encoder")
pipeline=AutoPipelineForText2Image.from_pretrained(stable_diffusion_id,image_encoder=image_encoder)
pipeline.load_lora_weights(lcm_lora_id)
pipeline.fuse_lora()
pipeline.load_ip_adapter(ip_adapter_id,subfolder="models",weight_name=ip_adapter_weight_name)
pipeline.set_ip_adapter_scale(0.6)
scheduler=LCMScheduler.from_pretrained(stable_diffusion_id,subfolder="scheduler")
return(
pipeline.tokenizer,
pipeline.feature_extractor,
scheduler,
pipeline.text_encoder,
pipeline.image_encoder,
pipeline.unet,
pipeline.vae,
)


ifload_original_pipeline:
(
tokenizer,
feature_extractor,
scheduler,
text_encoder,
image_encoder,
unet,
vae,
)=get_pipeline_components(stable_diffusion_id,ip_adapter_id,ip_adapter_weight_name,lcm_lora_id)
scheduler.save_pretrained(models_dir/"scheduler")
else:
tokenizer,feature_extractor,scheduler,text_encoder,image_encoder,unet,vae=(
None,
None,
None,
None,
None,
None,
None,
)


..parsed-literal::

2024-07-1303:58:25.762354:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1303:58:25.801185:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1303:58:26.402848:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/vq_model.py:20:FutureWarning:`VQEncoderOutput`isdeprecatedandwillberemovedinversion0.31.Importing`VQEncoderOutput`from`diffusers.models.vq_model`isdeprecatedandthiswillberemovedinafutureversion.Pleaseuse`fromdiffusers.models.autoencoders.vq_modelimportVQEncoderOutput`,instead.
deprecate("VQEncoderOutput","0.31",deprecation_message)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/vq_model.py:25:FutureWarning:`VQModel`isdeprecatedandwillberemovedinversion0.31.Importing`VQModel`from`diffusers.models.vq_model`isdeprecatedandthiswillberemovedinafutureversion.Pleaseuse`fromdiffusers.models.autoencoders.vq_modelimportVQModel`,instead.
deprecate("VQModel","0.31",deprecation_message)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1132:FutureWarning:`resume_download`isdeprecatedandwillberemovedinversion1.0.0.Downloadsalwaysresumewhenpossible.Ifyouwanttoforceanewdownload,use`force_download=True`.
warnings.warn(



..parsed-literal::

Loadingpipelinecomponents...:0%||0/7[00:00<?,?it/s]


..parsed-literal::

TheinstalledversionofbitsandbyteswascompiledwithoutGPUsupport.8-bitoptimizers,8-bitmultiplication,andGPUquantizationareunavailable.


ConvertPyTorchmodels
----------------------

`backtotop⬆️<#table-of-contents>`__

Startingfrom2023.0release,OpenVINOsupportsPyTorchmodelsdirectly
viaModelConversionAPI.``ov.convert_model``functionacceptsinstance
ofPyTorchmodelandexampleinputsfortracingandreturnsobjectof
``ov.Model``class,readytouseorsaveondiskusing``ov.save_model``
function.

Thepipelineconsistsoffourimportantparts:

-ImageEncodertocreateimageconditionforIP-Adapter.
-TextEncodertocreateconditiontogenerateanimagefromatext
prompt.
-U-Netforstep-by-stepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)fordecodinglatentspacetoimage.

Letusconverteachpart:

ImageEncoder
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

IP-Adapterreliesonanimageencodertogeneratetheimagefeatures.
Usually
`CLIPVisionModelWithProjection<https://huggingface.co/docs/transformers/main/en/model_doc/clip#transformers.CLIPVisionModelWithProjection>`__
isusedasImageEncoder.Forpreprocessinginputimage,ImageEncoder
uses``CLIPImageProcessor``namedfeatureextractorinpipeline.The
imageencoderacceptresizedandnormalizedimageprocessedbyfeature
extractorasinputandreturnsimageembeddings.

..code::ipython3

importopenvinoasov
importtorch
importgc


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


IMAGE_ENCODER_PATH=models_dir/"image_encoder.xml"

ifnotIMAGE_ENCODER_PATH.exists():
withtorch.no_grad():
ov_model=ov.convert_model(
image_encoder,
example_input=torch.zeros((1,3,224,224)),
input=[-1,3,224,224],
)
ov.save_model(ov_model,IMAGE_ENCODER_PATH)
feature_extractor.save_pretrained(models_dir/"feature_extractor")
delov_model
cleanup_torchscript_cache()

delimage_encoder
delfeature_extractor

gc.collect();


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:279:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_weights.size()!=(bsz*self.num_heads,tgt_len,src_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:319:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_output.size()!=(bsz*self.num_heads,tgt_len,self.head_dim):


..parsed-literal::

['pixel_values']


U-net
~~~~~

`backtotop⬆️<#table-of-contents>`__

U-Netmodelgraduallydenoiseslatentimagerepresentationguidedby
textencoderhiddenstate.

Generally,U-NetmodelconversionprocessremainthesamelikeinStable
Diffusion,expectadditionalinputthatacceptimageembeddings
generatedbyImageEncoder.InStableDiffusionpipeline,thisdata
providedintomodelusingdictionary``added_cond_kwargs``andkey
``image_embeds``insideit.AfterOpenVINOconversion,thisinputwill
bedecomposedfromdictionary.Insomecases,suchdecompositionmay
leadtoloosinginformationaboutinputshapeanddatatype.Wecan
restoreitmanuallyasdemonstratedinthecodebellow.

U-Netmodelinputs:

-``sample``-latentimagesamplefrompreviousstep.Generation
processhasnotbeenstartedyet,soyouwilluserandomnoise.
-``timestep``-currentschedulerstep.
-``encoder_hidden_state``-hiddenstateoftextencoder.
-``image_embeds``-hiddenstateofimageencoder.

Modelpredictsthe``sample``stateforthenextstep.

..code::ipython3

UNET_PATH=models_dir/"unet.xml"


ifnotUNET_PATH.exists():
inputs={
"sample":torch.randn((2,4,64,64)),
"timestep":torch.tensor(1),
"encoder_hidden_states":torch.randn((2,77,768)),
"added_cond_kwargs":{"image_embeds":torch.ones((2,1024))},
}

withtorch.no_grad():
ov_model=ov.convert_model(unet,example_input=inputs)
#dictionarywithadded_cond_kwargswillbedecomposedduringconversion
#insomecasesdecompositionmayleadtolosingdatatypeandshapeinformation
#Weneedtorecoveritmanuallyaftertheconversion
ov_model.inputs[-1].get_node().set_element_type(ov.Type.f32)
ov_model.validate_nodes_and_infer_types()
ov.save_model(ov_model,UNET_PATH)
delov_model
cleanup_torchscript_cache()

delunet

gc.collect();


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/unets/unet_2d_condition.py:1103:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifdim%default_overall_up_factor!=0:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/embeddings.py:1257:FutureWarning:Youhavepassedatensoras`image_embeds`.Thisisdeprecatedandwillberemovedinafuturerelease.Pleasemakesuretoupdateyourscripttopass`image_embeds`asalistoftensorstosupressthiswarning.
deprecate("image_embedsnotalist","1.0.0",deprecation_message,standard_warn=False)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:136:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:145:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:146:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
asserthidden_states.shape[1]==self.channels
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:162:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifhidden_states.shape[0]>=64:


..parsed-literal::

['sample','timestep','encoder_hidden_states','added_cond_kwargs']


VAEEncoderandDecoder
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheVAEmodelhastwoparts,anencoderandadecoder.Theencoderis
usedtoconverttheimageintoalowdimensionallatentrepresentation,
whichwillserveastheinputtotheU-Netmodel.Thedecoder,
conversely,transformsthelatentrepresentationbackintoanimage.

Duringlatentdiffusiontraining,theencoderisusedtogetthelatent
representations(latents)oftheimagesfortheforwarddiffusion
process,whichappliesmoreandmorenoiseateachstep.During
inference,thedenoisedlatentsgeneratedbythereversediffusion
processareconvertedbackintoimagesusingtheVAEdecoder.Whenyou
runinferenceforText-to-Image,thereisnoinitialimageasastarting
point.Youcanskipthisstepanddirectlygenerateinitialrandom
noise.VAEencoderisusedinImage-to-Imagegenerationpipelinesfor
creatinginitiallatentstatebasedoninputimage.Themaindifference
betweenIP-AdapterencodedimageandVAEencodedimagethatthefirstis
usedasadditionintoinputpromptmakingconnectionbetweentextand
imageduringconditioning,whilethesecondusedasUnetsample
initializationanddoesnotgiveguaranteepreservingsomeattributesof
initialimage.Itisstillcanbeusefultousebothip-adapterandVAE
imageinpipeline,wecandiscussitininferenceexamples.

..code::ipython3

VAE_DECODER_PATH=models_dir/"vae_decoder.xml"
VAE_ENCODER_PATH=models_dir/"vae_encoder.xml"

ifnotVAE_DECODER_PATH.exists():

classVAEDecoderWrapper(torch.nn.Module):
def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,latents):
returnself.vae.decode(latents)

vae_decoder=VAEDecoderWrapper(vae)
withtorch.no_grad():
ov_model=ov.convert_model(vae_decoder,example_input=torch.ones([1,4,64,64]))
ov.save_model(ov_model,VAE_DECODER_PATH)
delov_model
cleanup_torchscript_cache()
delvae_decoder

ifnotVAE_ENCODER_PATH.exists():

classVAEEncoderWrapper(torch.nn.Module):
def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,image):
returnself.vae.encode(x=image)["latent_dist"].sample()

vae_encoder=VAEEncoderWrapper(vae)
vae_encoder.eval()
image=torch.zeros((1,3,512,512))
withtorch.no_grad():
ov_model=ov.convert_model(vae_encoder,example_input=image)
ov.save_model(ov_model,VAE_ENCODER_PATH)
delov_model
cleanup_torchscript_cache()

delvae
gc.collect();


..parsed-literal::

['latents']


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:1116:TracerWarning:Tracehadnondeterministicnodes.Didyouforgetcall.eval()onyourmodel?Nodes:
	%2494:Float(1,4,64,64,strides=[16384,4096,64,1],requires_grad=0,device=cpu)=aten::randn(%2488,%2489,%2490,%2491,%2492,%2493)#/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/torch_utils.py:81:0
Thismaycauseerrorsintracechecking.Todisabletracechecking,passcheck_trace=Falsetotorch.jit.trace()
_check_trace(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:1116:TracerWarning:Outputnr1.ofthetracedfunctiondoesnotmatchthecorrespondingoutputofthePythonfunction.Detailederror:
Tensor-likesarenotclose!

Mismatchedelements:10409/16384(63.5%)
Greatestabsolutedifference:0.0016331672668457031atindex(0,2,63,63)(upto1e-05allowed)
Greatestrelativedifference:0.0036048741534714223atindex(0,3,63,59)(upto1e-05allowed)
_check_trace(


..parsed-literal::

['image']


TextEncoder
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thetext-encoderisresponsiblefortransformingtheinputprompt,for
example,“aphotoofanastronautridingahorse”intoanembedding
spacethatcanbeunderstoodbytheU-Net.Itisusuallyasimple
transformer-basedencoderthatmapsasequenceofinputtokenstoa
sequenceoflatenttextembeddings.

Theinputofthetextencoderistensor``input_ids``,whichcontains
indexesoftokensfromtextprocessedbythetokenizerandpaddedtothe
maximumlengthacceptedbythemodel.Modeloutputsaretwotensors:
``last_hidden_state``-hiddenstatefromthelastMultiHeadAttention
layerinthemodeland``pooler_out``-pooledoutputforwholemodel
hiddenstates.

..code::ipython3

TEXT_ENCODER_PATH=models_dir/"text_encoder.xml"

ifnotTEXT_ENCODER_PATH.exists():
withtorch.no_grad():
ov_model=ov.convert_model(
text_encoder,
example_input=torch.ones([1,77],dtype=torch.long),
input=[
(1,77),
],
)
ov.save_model(ov_model,TEXT_ENCODER_PATH)
delov_model
cleanup_torchscript_cache()
tokenizer.save_pretrained(models_dir/"tokenizer")

deltext_encoder
deltokenizer


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:86:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifinput_shape[-1]>1orself.sliding_windowisnotNone:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:162:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifpast_key_values_length>0:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:287:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifcausal_attention_mask.size()!=(bsz,1,tgt_len,src_len):


..parsed-literal::

['input_ids']


PrepareOpenVINOinferencepipeline
-----------------------------------

`backtotop⬆️<#table-of-contents>`__

Asshownondiagrambelow,theonlydifferencebetweenoriginalStable
DiffusionpipelineandIP-AdapterStableDiffusionpipelineonlyin
additionalconditioningbyimageprocessedviaImageEncoder.
|pipeline.png|

Thestablediffusionmodelwithip-adaptertakesalatentimage
representation,atextpromptistransformedtotextembeddingsviaCLIP
textencoderandip-adapterimageistransformedtoimageembeddingsvia
CLIPImageEncoder.Next,theU-Netiteratively*denoises*therandom
latentimagerepresentationswhilebeingconditionedonthetextand
imageembeddings.TheoutputoftheU-Net,beingthenoiseresidual,is
usedtocomputeadenoisedlatentimagerepresentationviaascheduler
algorithm.

The*denoising*processisrepeatedgivennumberoftimes(bydefault4
takingintoaccountthatweuseLCM)tostep-by-stepretrievebetter
latentimagerepresentations.Whencomplete,thelatentimage
representationisdecodedbythedecoderpartofthevariationalauto
encoder(VAE).

..|pipeline.png|image::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1afc2ca6-e7ea-4c9e-a2d3-1173346dd9d6

..code::ipython3

importinspect
fromtypingimportList,Optional,Union,Dict,Tuple
importnumpyasnp

importPIL
importcv2
importtorch

fromtransformersimportCLIPTokenizer,CLIPImageProcessor
fromdiffusersimportDiffusionPipeline
fromdiffusers.pipelines.stable_diffusion.pipeline_outputimport(
StableDiffusionPipelineOutput,
)
fromdiffusers.schedulersimportDDIMScheduler,LMSDiscreteScheduler,PNDMScheduler


defscale_fit_to_window(dst_width:int,dst_height:int,image_width:int,image_height:int):
"""
Preprocessinghelperfunctionforcalculatingimagesizeforresizewithpeservingoriginalaspectratio
andfittingimagetospecificwindowsize

Parameters:
dst_width(int):destinationwindowwidth
dst_height(int):destinationwindowheight
image_width(int):sourceimagewidth
image_height(int):sourceimageheight
Returns:
result_width(int):calculatedwidthforresize
result_height(int):calculatedheightforresize
"""
im_scale=min(dst_height/image_height,dst_width/image_width)
returnint(im_scale*image_width),int(im_scale*image_height)


defrandn_tensor(
shape:Union[Tuple,List],
generator:Optional[Union[List["torch.Generator"],"torch.Generator"]]=None,
dtype:Optional["torch.dtype"]=None,
):
"""Ahelperfunctiontocreaterandomtensorsonthedesired`device`withthedesired`dtype`.When
passingalistofgenerators,youcanseedeachbatchsizeindividually.

"""
batch_size=shape[0]
rand_device=torch.device("cpu")

#makesuregeneratorlistoflength1istreatedlikeanon-list
ifisinstance(generator,list)andlen(generator)==1:
generator=generator[0]

ifisinstance(generator,list):
shape=(1,)+shape[1:]
latents=[torch.randn(shape,generator=generator[i],device=rand_device,dtype=dtype)foriinrange(batch_size)]
latents=torch.cat(latents,dim=0)
else:
latents=torch.randn(shape,generator=generator,device=rand_device,dtype=dtype)

returnlatents


defpreprocess(image:PIL.Image.Image,height,width):
"""
Imagepreprocessingfunction.TakesimageinPIL.Imageformat,resizesittokeepaspectrationandfitstomodelinputwindow512x512,
thenconvertsittonp.ndarrayandaddspaddingwithzerosonrightorbottomsideofimage(dependsfromaspectratio),afterthat
convertsdatatofloat32datatypeandchangerangeofvaluesfrom[0,255]to[-1,1],finally,convertsdatalayoutfromplanarNHWCtoNCHW.
Thefunctionreturnspreprocessedinputtensorandpaddingsize,whichcanbeusedinpostprocessing.

Parameters:
image(PIL.Image.Image):inputimage
Returns:
image(np.ndarray):preprocessedimagetensor
meta(Dict):dictionarywithpreprocessingmetadatainfo
"""
src_width,src_height=image.size
dst_width,dst_height=scale_fit_to_window(height,width,src_width,src_height)
image=np.array(image.resize((dst_width,dst_height),resample=PIL.Image.Resampling.LANCZOS))[None,:]
pad_width=width-dst_width
pad_height=height-dst_height
pad=((0,0),(0,pad_height),(0,pad_width),(0,0))
image=np.pad(image,pad,mode="constant")
image=image.astype(np.float32)/255.0
image=2.0*image-1.0
image=image.transpose(0,3,1,2)
returnimage,{"padding":pad,"src_width":src_width,"src_height":src_height}


classOVStableDiffusionPipeline(DiffusionPipeline):
def__init__(
self,
vae_decoder:ov.Model,
text_encoder:ov.Model,
tokenizer:CLIPTokenizer,
unet:ov.Model,
scheduler:Union[DDIMScheduler,PNDMScheduler,LMSDiscreteScheduler],
image_encoder:ov.Model,
feature_extractor:CLIPImageProcessor,
vae_encoder:ov.Model,
):
"""
Pipelinefortext-to-imagegenerationusingStableDiffusionandIP-AdapterwithOpenVINO
Parameters:
vae_decoder(ov.Model):
VariationalAuto-Encoder(VAE)Modeltodecodeimagestoandfromlatentrepresentations.
text_encoder(ov.Model):CLIPImageProcessor
Frozentext-encoder.StableDiffusionusesthetextportionof
[CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel),specifically
theclip-vit-large-patch14(https://huggingface.co/openai/clip-vit-large-patch14)variant.
tokenizer(CLIPTokenizer):
TokenizerofclassCLIPTokenizer(https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
unet(ov.Model):ConditionalU-Netarchitecturetodenoisetheencodedimagelatents.
scheduler(SchedulerMixin):
Aschedulertobeusedincombinationwithunettodenoisetheencodedimagelatents
image_encoder(ov.Model):
IP-Adapterimageencoderforembeddinginputimageasinputpromptforgeneration
feature_extractor:
"""
super().__init__()
self.scheduler=scheduler
self.vae_decoder=vae_decoder
self.image_encoder=image_encoder
self.text_encoder=text_encoder
self.unet=unet
self.height=512
self.width=512
self.vae_scale_factor=8
self.tokenizer=tokenizer
self.vae_encoder=vae_encoder
self.feature_extractor=feature_extractor

def__call__(
self,
prompt:Union[str,List[str]],
ip_adapter_image:PIL.Image.Image,
image:PIL.Image.Image=None,
num_inference_steps:Optional[int]=4,
negative_prompt:Union[str,List[str]]=None,
guidance_scale:Optional[float]=0.5,
eta:Optional[float]=0.0,
output_type:Optional[str]="pil",
height:Optional[int]=None,
width:Optional[int]=None,
generator:Optional[Union[torch.Generator,List[torch.Generator]]]=None,
latents:Optional[torch.FloatTensor]=None,
strength:float=1.0,
**kwargs,
):
"""
Functioninvokedwhencallingthepipelineforgeneration.
Parameters:
prompt(strorList[str]):
Thepromptorpromptstoguidetheimagegeneration.
image(PIL.Image.Image,*optional*,None):
Intinalimageforgeneration.
num_inference_steps(int,*optional*,defaultsto50):
Thenumberofdenoisingsteps.Moredenoisingstepsusuallyleadtoahigherqualityimageatthe
expenseofslowerinference.
negative_prompt(strorList[str]):https://user-images.githubusercontent.com/29454499/258651862-28b63016-c5ff-4263-9da8-73ca31100165.jpeg
Thenegativepromptorpromptstoguidetheimagegeneration.
guidance_scale(float,*optional*,defaultsto7.5):
GuidancescaleasdefinedinClassifier-FreeDiffusionGuidance(https://arxiv.org/abs/2207.12598).
guidance_scaleisdefinedas`w`ofequation2.
Higherguidancescaleencouragestogenerateimagesthatarecloselylinkedtothetextprompt,
usuallyattheexpenseoflowerimagequality.
eta(float,*optional*,defaultsto0.0):
Correspondstoparametereta(η)intheDDIMpaper:https://arxiv.org/abs/2010.02502.Onlyappliesto
[DDIMScheduler],willbeignoredforothers.
output_type(`str`,*optional*,defaultsto"pil"):
Theoutputformatofthegenerateimage.Choosebetween
[PIL](https://pillow.readthedocs.io/en/stable/):PIL.Image.Imageornp.array.
height(int,*optional*,512):
Generatedimageheight
width(int,*optional*,512):
Generatedimagewidth
generator(`torch.Generator`or`List[torch.Generator]`,*optional*):
A[`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)tomake
generationdeterministic.
latents(`torch.FloatTensor`,*optional*):
Pre-generatednoisylatentssampledfromaGaussiandistribution,tobeusedasinputsforimage
generation.Canbeusedtotweakthesamegenerationwithdifferentprompts.Ifnotprovided,alatents
tensorisgeneratedbysamplingusingthesuppliedrandom`generator`.
Returns:
Dictionarywithkeys:
sample-thelastgeneratedimagePIL.Image.Imageornp.arrayhttps://huggingface.co/latent-consistency/lcm-lora-sdv1-5
iterations-*optional*(ifgif=True)imagesforalldiffusionsteps,ListofPIL.Image.Imageornp.array.
"""
do_classifier_free_guidance=guidance_scale>1.0
#getprompttextembeddings
text_embeddings=self._encode_prompt(
prompt,
do_classifier_free_guidance=do_classifier_free_guidance,
negative_prompt=negative_prompt,
)
#getip-adapterimageembeddings
image_embeds,negative_image_embeds=self.encode_image(ip_adapter_image)
ifdo_classifier_free_guidance:
image_embeds=np.concatenate([negative_image_embeds,image_embeds])

#settimesteps
accepts_offset="offset"inset(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
extra_set_kwargs={}
ifaccepts_offset:
extra_set_kwargs["offset"]=1

self.scheduler.set_timesteps(num_inference_steps,**extra_set_kwargs)
timesteps,num_inference_steps=self.get_timesteps(num_inference_steps,strength)
latent_timestep=timesteps[:1]

#gettheinitialrandomnoiseunlesstheusersuppliedit
latents,meta=self.prepare_latents(
1,
4,
heightorself.height,
widthorself.width,
generator=generator,
latents=latents,
image=image,
latent_timestep=latent_timestep,
)

#prepareextrakwargsfortheschedulerstep,sincenotallschedulershavethesamesignature
#eta(η)isonlyusedwiththeDDIMScheduler,itwillbeignoredforotherschedulers.
#etacorrespondstoηinDDIMpaper:https://arxiv.org/abs/2010.02502
#andshouldbebetween[0,1]
accepts_eta="eta"inset(inspect.signature(self.scheduler.step).parameters.keys())
extra_step_kwargs={}
ifaccepts_eta:
extra_step_kwargs["eta"]=eta

fori,tinenumerate(self.progress_bar(timesteps)):
#expandthelatentsifyouaredoingclassifierfreeguidance
latent_model_input=np.concatenate([latents]*2)ifdo_classifier_free_guidanceelselatents
latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)

#predictthenoiseresidual
noise_pred=self.unet([latent_model_input,t,text_embeddings,image_embeds])[0]
#performguidance
ifdo_classifier_free_guidance:
noise_pred_uncond,noise_pred_text=noise_pred[0],noise_pred[1]
noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

#computethepreviousnoisysamplex_t->x_t-1
latents=self.scheduler.step(
torch.from_numpy(noise_pred),
t,
torch.from_numpy(latents),
**extra_step_kwargs,
)["prev_sample"].numpy()

#scaleanddecodetheimagelatentswithvae
image=self.vae_decoder(latents*(1/0.18215))[0]

image=self.postprocess_image(image,meta,output_type)
returnStableDiffusionPipelineOutput(images=image,nsfw_content_detected=False)

def_encode_prompt(
self,
prompt:Union[str,List[str]],
num_images_per_prompt:int=1,
do_classifier_free_guidance:bool=True,
negative_prompt:Union[str,List[str]]=None,
):
"""
Encodesthepromptintotextencoderhiddenstates.

Parameters:
prompt(strorlist(str)):prompttobeencoded
num_images_per_prompt(int):numberofimagesthatshouldbegeneratedperprompt
do_classifier_free_guidance(bool):whethertouseclassifierfreeguidanceornot
negative_prompt(strorlist(str)):negativeprompttobeencoded.
Returns:
text_embeddings(np.ndarray):textencoderhiddenstates
"""
batch_size=len(prompt)ifisinstance(prompt,list)else1

#tokenizeinputprompts
text_inputs=self.tokenizer(
prompt,
padding="max_length",
max_length=self.tokenizer.model_max_length,
truncation=True,
return_tensors="np",
)
text_input_ids=text_inputs.input_ids

text_embeddings=self.text_encoder(text_input_ids)[0]

#duplicatetextembeddingsforeachgenerationperprompt
ifnum_images_per_prompt!=1:
bs_embed,seq_len,_=text_embeddings.shape
text_embeddings=np.tile(text_embeddings,(1,num_images_per_prompt,1))
text_embeddings=np.reshape(text_embeddings,(bs_embed*num_images_per_prompt,seq_len,-1))

#getunconditionalembeddingsforclassifierfreeguidance
ifdo_classifier_free_guidance:
uncond_tokens:List[str]
max_length=text_input_ids.shape[-1]
ifnegative_promptisNone:
uncond_tokens=[""]*batch_size
elifisinstance(negative_prompt,str):
uncond_tokens=[negative_prompt]
else:
uncond_tokens=negative_prompt
uncond_input=self.tokenizer(
uncond_tokens,
padding="max_length",
max_length=max_length,
truncation=True,
return_tensors="np",
)

uncond_embeddings=self.text_encoder(uncond_input.input_ids)[0]

#duplicateunconditionalembeddingsforeachgenerationperprompt,usingmpsfriendlymethod
seq_len=uncond_embeddings.shape[1]
uncond_embeddings=np.tile(uncond_embeddings,(1,num_images_per_prompt,1))
uncond_embeddings=np.reshape(uncond_embeddings,(batch_size*num_images_per_prompt,seq_len,-1))

#Forclassifier-freeguidance,weneedtodotwoforwardpasses.
#Hereweconcatenatetheunconditionalandtextembeddingsintoasinglebatch
#toavoiddoingtwoforwardpasses
text_embeddings=np.concatenate([uncond_embeddings,text_embeddings])

returntext_embeddings

defprepare_latents(
self,
batch_size,
num_channels_latents,
height,
width,
dtype=torch.float32,
generator=None,
latents=None,
image=None,
latent_timestep=None,
):
shape=(
batch_size,
num_channels_latents,
height//self.vae_scale_factor,
width//self.vae_scale_factor,
)
ifisinstance(generator,list)andlen(generator)!=batch_size:
raiseValueError(
f"Youhavepassedalistofgeneratorsoflength{len(generator)},butrequestedaneffectivebatch"
f"sizeof{batch_size}.Makesurethebatchsizematchesthelengthofthegenerators."
)

iflatentsisNone:
latents=randn_tensor(shape,generator=generator,dtype=dtype)

ifimageisNone:
#scaletheinitialnoisebythestandarddeviationrequiredbythescheduler
latents=latents*self.scheduler.init_noise_sigma
returnlatents.numpy(),{}
input_image,meta=preprocess(image,height,width)
image_latents=self.vae_encoder(input_image)[0]
image_latents=image_latents*0.18215
latents=self.scheduler.add_noise(torch.from_numpy(image_latents),latents,latent_timestep).numpy()
returnlatents,meta

defpostprocess_image(self,image:np.ndarray,meta:Dict,output_type:str="pil"):
"""
Postprocessingfordecodedimage.TakesgeneratedimagedecodedbyVAEdecoder,unpadittoinitialimagesize(ifrequired),
normalizeandconvertto[0,255]pixelsrange.Optionally,convertsitfromnp.ndarraytoPIL.Imageformat

Parameters:
image(np.ndarray):
Generatedimage
meta(Dict):
Metadataobtainedonthelatentspreparingstepcanbeempty
output_type(str,*optional*,pil):
Outputformatforresult,canbepilornumpy
Returns:
image(Listofnp.ndarrayorPIL.Image.Image):
Post-processedimages
"""
if"padding"inmeta:
pad=meta["padding"]
(_,end_h),(_,end_w)=pad[1:3]
h,w=image.shape[2:]
unpad_h=h-end_h
unpad_w=w-end_w
image=image[:,:,:unpad_h,:unpad_w]
image=np.clip(image/2+0.5,0,1)
image=np.transpose(image,(0,2,3,1))
#9.ConverttoPIL
ifoutput_type=="pil":
image=self.numpy_to_pil(image)
if"src_height"inmeta:
orig_height,orig_width=meta["src_height"],meta["src_width"]
image=[img.resize((orig_width,orig_height),PIL.Image.Resampling.LANCZOS)forimginimage]
else:
if"src_height"inmeta:
orig_height,orig_width=meta["src_height"],meta["src_width"]
image=[cv2.resize(img,(orig_width,orig_width))forimginimage]
returnimage

defencode_image(self,image,num_images_per_prompt=1):
ifnotisinstance(image,torch.Tensor):
image=self.feature_extractor(image,return_tensors="pt").pixel_values

image_embeds=self.image_encoder(image)[0]
ifnum_images_per_prompt>1:
image_embeds=image_embeds.repeat_interleave(num_images_per_prompt,dim=0)

uncond_image_embeds=np.zeros(image_embeds.shape)
returnimage_embeds,uncond_image_embeds

defget_timesteps(self,num_inference_steps:int,strength:float):
"""
Helperfunctionforgettingschedulertimestepsforgeneration
Incaseofimage-to-imagegeneration,itupdatesnumberofstepsaccordingtostrength

Parameters:
num_inference_steps(int):
numberofinferencestepsforgeneration
strength(float):
valuebetween0.0and1.0,thatcontrolstheamountofnoisethatisaddedtotheinputimage.
Valuesthatapproach1.0allowforlotsofvariationsbutwillalsoproduceimagesthatarenotsemanticallyconsistentwiththeinput.
"""
#gettheoriginaltimestepusinginit_timestep
init_timestep=min(int(num_inference_steps*strength),num_inference_steps)

t_start=max(num_inference_steps-init_timestep,0)
timesteps=self.scheduler.timesteps[t_start:]

returntimesteps,num_inference_steps-t_start

Runmodelinference
-------------------

`backtotop⬆️<#table-of-contents>`__

Nowlet’sconfigureourpipelineandtakealookongenerationresults.

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Selectinferencedevicefromdropdownlist.

..code::ipython3

core=ov.Core()

importipywidgetsaswidgets

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



..code::ipython3

fromtransformersimportAutoTokenizer

ov_config={"INFERENCE_PRECISION_HINT":"f32"}ifdevice.value!="CPU"else{}
vae_decoder=core.compile_model(VAE_DECODER_PATH,device.value,ov_config)
vae_encoder=core.compile_model(VAE_ENCODER_PATH,device.value,ov_config)
text_encoder=core.compile_model(TEXT_ENCODER_PATH,device.value)
image_encoder=core.compile_model(IMAGE_ENCODER_PATH,device.value)
unet=core.compile_model(UNET_PATH,device.value)

scheduler=LCMScheduler.from_pretrained(models_dir/"scheduler")
tokenizer=AutoTokenizer.from_pretrained(models_dir/"tokenizer")
feature_extractor=CLIPImageProcessor.from_pretrained(models_dir/"feature_extractor")

ov_pipe=OVStableDiffusionPipeline(
vae_decoder,
text_encoder,
tokenizer,
unet,
scheduler,
image_encoder,
feature_extractor,
vae_encoder,
)


..parsed-literal::

Theconfigattributes{'skip_prk_steps':True}werepassedtoLCMScheduler,butarenotexpectedandwillbeignored.Pleaseverifyyourscheduler_config.jsonconfigurationfile.


Generationimagevariation
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Ifwestayinputtextpromptemptyandprovideonlyip-adapterimage,we
cangetvariationofthesameimage.

..code::ipython3

importmatplotlib.pyplotasplt


defvisualize_results(images,titles):
"""
Helperfunctionforresultsvisualization

Parameters:
orig_img(PIL.Image.Image):originalimage
processed_img(PIL.Image.Image):processedimageafterediting
img1_title(str):titlefortheimageontheleft
img2_title(str):titlefortheimageontheright
Returns:
fig(matplotlib.pyplot.Figure):matplotlibgeneratedfigurecontainsdrawingresult
"""
im_w,im_h=images[0].size
is_horizontal=im_h<=im_w
figsize=(10,15*len(images))ifis_horizontalelse(15*len(images),10)
fig,axs=plt.subplots(
1ifis_horizontalelselen(images),
len(images)ifis_horizontalelse1,
figsize=figsize,
sharex="all",
sharey="all",
)
fig.patch.set_facecolor("white")
list_axes=list(axs.flat)
forainlist_axes:
a.set_xticklabels([])
a.set_yticklabels([])
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.grid(False)
forimage,title,axinzip(images,titles,list_axes):
ax.imshow(np.array(image))
ax.set_title(title,fontsize=20)
fig.subplots_adjust(wspace=0.0ifis_horizontalelse0.01,hspace=0.01ifis_horizontalelse0.0)
fig.tight_layout()
returnfig

..code::ipython3

generator=torch.Generator(device="cpu").manual_seed(576)

image=load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")

result=ov_pipe(
prompt="",
ip_adapter_image=image,
gaidance_scale=1,
negative_prompt="",
num_inference_steps=4,
generator=generator,
)

fig=visualize_results([image,result.images[0]],["inputimage","result"])



..parsed-literal::

0%||0/4[00:00<?,?it/s]



..image::stable-diffusion-ip-adapter-with-output_files/stable-diffusion-ip-adapter-with-output_22_1.png


Generationconditionedbyimageandtext
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

IP-Adapterallowsyoutousebothimageandtexttoconditiontheimage
generationprocess.BothIP-Adapterimageandtextpromptserveas
extensionforeachother,forexamplewecanuseatextprompttoadd
“sunglasses”😎onpreviousimage.

..code::ipython3

generator=torch.Generator(device="cpu").manual_seed(576)

result=ov_pipe(
prompt="bestquality,highquality,wearingsunglasses",
ip_adapter_image=image,
gaidance_scale=1,
negative_prompt="monochrome,low-res,badanatomy,worstquality,lowquality",
num_inference_steps=4,
generator=generator,
)



..parsed-literal::

0%||0/4[00:00<?,?it/s]


..code::ipython3

fig=visualize_results([image,result.images[0]],["inputimage","result"])



..image::stable-diffusion-ip-adapter-with-output_files/stable-diffusion-ip-adapter-with-output_25_0.png


Generationimageblending
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

IP-AdapteralsoworksgreatwithImage-to-Imagetranslation.Ithelpsto
achieveimageblendingeffect.

..code::ipython3

image=load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/vermeer.jpg")
ip_image=load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/river.png")

result=ov_pipe(
prompt="bestquality,highquality",
image=image,
ip_adapter_image=ip_image,
gaidance_scale=1,
generator=generator,
strength=0.7,
num_inference_steps=8,
)



..parsed-literal::

0%||0/5[00:00<?,?it/s]


..code::ipython3

fig=visualize_results([image,ip_image,result.images[0]],["inputimage","ip-adapterimage","result"])



..image::stable-diffusion-ip-adapter-with-output_files/stable-diffusion-ip-adapter-with-output_28_0.png


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

Now,youcantrymodelusingownimagesandtextprompts.

..code::ipython3

importgradioasgr


defgenerate_from_text(
positive_prompt,
negative_prompt,
ip_adapter_image,
seed,
num_steps,
guidance_scale,
_=gr.Progress(track_tqdm=True),
):
generator=torch.Generator(device="cpu").manual_seed(seed)
result=ov_pipe(
positive_prompt,
ip_adapter_image=ip_adapter_image,
negative_prompt=negative_prompt,
guidance_scale=guidance_scale,
num_inference_steps=num_steps,
generator=generator,
)
returnresult.images[0]


defgenerate_from_image(
img,
ip_adapter_image,
positive_prompt,
negative_prompt,
seed,
num_steps,
guidance_scale,
strength,
_=gr.Progress(track_tqdm=True),
):
generator=torch.Generator(device="cpu").manual_seed(seed)
result=ov_pipe(
positive_prompt,
image=img,
ip_adapter_image=ip_adapter_image,
negative_prompt=negative_prompt,
num_inference_steps=num_steps,
guidance_scale=guidance_scale,
strength=strength,
generator=generator,
)
returnresult.images[0]


withgr.Blocks()asdemo:
withgr.Tab("Text-to-Imagegeneration"):
withgr.Row():
withgr.Column():
ip_adapter_input=gr.Image(label="IP-AdapterImage",type="pil")
text_input=gr.Textbox(lines=3,label="Positiveprompt")
neg_text_input=gr.Textbox(lines=3,label="Negativeprompt")
withgr.Accordion("Advancedoptions",open=False):
seed_input=gr.Slider(0,10000000,value=42,label="Seed")
steps_input=gr.Slider(1,12,value=4,step=1,label="Steps")
guidance_scale_input=gr.Slider(
label="Guidancescale",
minimum=0.1,
maximum=2,
step=0.1,
value=0.5,
)
out=gr.Image(label="Result",type="pil")
btn=gr.Button()
btn.click(
generate_from_text,
[
text_input,
neg_text_input,
ip_adapter_input,
seed_input,
steps_input,
guidance_scale_input,
],
out,
)
gr.Examples(
[
[
"https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/woman.png",
"bestquality,highquality",
"lowresolution",
],
[
"https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/statue.png",
"wearingahat",
"",
],
],
[ip_adapter_input,text_input,neg_text_input],
)
withgr.Tab("Image-to-Imagegeneration"):
withgr.Row():
withgr.Column():
i2i_input=gr.Image(label="Image",type="pil")
i2i_ip_adapter_input=gr.Image(label="IP-AdapterImage",type="pil")
i2i_text_input=gr.Textbox(lines=3,label="Text")
i2i_neg_text_input=gr.Textbox(lines=3,label="Negativeprompt")
withgr.Accordion("Advancedoptions",open=False):
i2i_seed_input=gr.Slider(0,10000000,value=42,label="Seed")
i2i_steps_input=gr.Slider(1,12,value=8,step=1,label="Steps")
strength_input=gr.Slider(0,1,value=0.7,label="Strength")
i2i_guidance_scale=gr.Slider(
label="Guidancescale",
minimum=0.1,
maximum=2,
step=0.1,
value=0.5,
)
i2i_out=gr.Image(label="Result")
i2i_btn=gr.Button()
i2i_btn.click(
generate_from_image,
[
i2i_input,
i2i_ip_adapter_input,
i2i_text_input,
i2i_neg_text_input,
i2i_seed_input,
i2i_steps_input,
i2i_guidance_scale,
strength_input,
],
i2i_out,
)
gr.Examples(
[
[
"https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/river.png",
"https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/statue.png",
],
],
[i2i_ip_adapter_input,i2i_input],
)
try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

ThanksforbeingaGradiouser!Ifyouhavequestionsorfeedback,pleasejoinourDiscordserverandchatwithus:https://discord.gg/feTf9x3ZSB

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>

