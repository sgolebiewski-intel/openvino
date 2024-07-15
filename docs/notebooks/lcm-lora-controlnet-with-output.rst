Text-to-ImageGenerationwithLCMLoRAandControlNetConditioning
==================================================================

DiffusionmodelsmakearevolutioninAI-generatedart.Thistechnology
enablesthecreationofhigh-qualityimagessimplybywritingatext
prompt.Eventhoughthistechnologygivesverypromisingresults,the
diffusionprocess,inthefirstorder,istheprocessofgenerating
imagesfromrandomnoiseandtextconditions,whichdonotalways
clarifyhowdesiredcontentshouldlook,whichformsitshouldhave,and
whereitislocatedinrelationtootherobjectsontheimage.
Researchershavebeenlookingforwaystohavemorecontroloverthe
resultsofthegenerationprocess.ControlNetprovidesaminimal
interfaceallowinguserstocustomizethegenerationprocesstoagreat
extent.

ControlNetwasintroducedin`AddingConditionalControlto
Text-to-ImageDiffusionModels<https://arxiv.org/abs/2302.05543>`__
paper.Itprovidesaframeworkthatenablessupportforvariousspatial
contextssuchasadepthmap,asegmentationmap,ascribble,andkey
pointsthatcanserveasadditionalconditioningstoDiffusionmodels
suchasStableDiffusion.

LatentConsistencyModels(LCM)areawaytodecreasethenumberof
stepsrequiredtogenerateanimagewithStableDiffusion(orSDXL)by
distillingtheoriginalmodelintoanotherversionthatrequiresfewer
steps(4to8insteadoftheoriginal25to50).Distillationisatype
oftrainingprocedurethatattemptstoreplicatetheoutputsfroma
sourcemodelusinganewone.Thedistilledmodelmaybedesignedtobe
smalleror,inthiscase,requirefewerstepstorun.It’susuallya
lengthyandcostlyprocessthatrequireshugeamountsofdata,patience,
andpowerfultraininghardware.

Forlatentconsistencydistillation,eachmodelneedstobedistilled
separately.TheLCMLoRAallowstotrainjustasmallnumberof
adapters,knownasLoRAlayers,insteadofthefullmodel.Theresulting
LoRAscanthenbeappliedtoanyfine-tunedversionofthemodelwithout
havingtodistilthemseparately.ThebenefitofthisLCMLoRA
distillationprocessisthatitcanbeintegratedintotheexisting
inferencepipelineswithoutchangestothemaincode,forexample,into
theControlNet-guidedStableDiffusionpipeline.MoredetailsaboutLCM
LoRAcanbefoundinthe`technical
report<https://arxiv.org/abs/2311.05556>`__and`blog
post<https://huggingface.co/blog/lcm_lora>`__

ThisnotebookexploreshowtospeedupControlNetpipelineusingLCM
LoRA,OpenVINOandquantizationwith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__.Letusget
“controlling”!

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Background<#background>`__

-`StableDiffusion<#stable-diffusion>`__
-`ControlNet<#controlnet>`__
-`Low-RankAdaptationofLargeLanguageModels
(LoRA)<#low-rank-adaptation-of-large-language-models-lora>`__

-`Prerequisites<#prerequisites>`__
-`LoadOriginalDiffuserspipelineandpreparemodelsfor
conversion<#load-original-diffusers-pipeline-and-prepare-models-for-conversion>`__
-`ConditionImage<#condition-image>`__
-`ConvertmodelstoOpenVINOIntermediaterepresentation(IR)
format<#convert-models-to-openvino-intermediate-representation-ir-format>`__

-`ControlNetconversion<#controlnet-conversion>`__
-`U-Net<#u-net>`__
-`TextEncoder<#text-encoder>`__
-`VAEDecoderconversion<#vae-decoder-conversion>`__

-`PrepareInferencepipeline<#prepare-inference-pipeline>`__

-`Preparetokenizerand
LCMScheduler<#prepare-tokenizer-and-lcmscheduler>`__
-`SelectinferencedeviceforStableDiffusion
pipeline<#select-inference-device-for-stable-diffusion-pipeline>`__

-`RunningText-to-ImageGenerationwithControlNetConditioningand
OpenVINO<#running-text-to-image-generation-with-controlnet-conditioning-and-openvino>`__
-`Quantization<#quantization>`__

-`Preparecalibrationdatasets<#prepare-calibration-datasets>`__
-`Runquantization<#run-quantization>`__
-`CompareinferencetimeoftheFP16andINT8
models<#compare-inference-time-of-the-fp16-and-int8-models>`__

-`Comparemodelfilesizes<#compare-model-file-sizes>`__

-`InteractiveDemo<#interactive-demo>`__

Background
----------

`backtotop⬆️<#table-of-contents>`__

StableDiffusion
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

`StableDiffusion<https://github.com/CompVis/stable-diffusion>`__isa
text-to-imagelatentdiffusionmodelcreatedbyresearchersand
engineersfromCompVis,StabilityAI,andLAION.Diffusionmodelsas
mentionedabovecangeneratehigh-qualityimages.StableDiffusionis
basedonaparticulartypeofdiffusionmodelcalledLatentDiffusion,
proposedin`High-ResolutionImageSynthesiswithLatentDiffusion
Models<https://arxiv.org/abs/2112.10752>`__paper.Generallyspeaking,
diffusionmodelsaremachinelearningsystemsthataretrainedto
denoiserandomGaussiannoisestepbystep,togettoasampleof
interest,suchasanimage.Diffusionmodelshavebeenshowntoachieve
state-of-the-artresultsforgeneratingimagedata.Butonedownsideof
diffusionmodelsisthatthereversedenoisingprocessisslowbecause
ofitsrepeated,sequentialnature.Inaddition,thesemodelsconsumea
lotofmemorybecausetheyoperateinpixelspace,whichbecomeshuge
whengeneratinghigh-resolutionimages.Latentdiffusioncanreducethe
memoryandcomputecomplexitybyapplyingthediffusionprocessovera
lowerdimensionallatentspace,insteadofusingtheactualpixelspace.
Thisisthekeydifferencebetweenstandarddiffusionandlatent
diffusionmodels:inlatentdiffusion,themodelistrainedtogenerate
latent(compressed)representationsoftheimages.

Therearethreemaincomponentsinlatentdiffusion:

-Atext-encoder,forexample`CLIP’sText
Encoder<https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel>`__
forcreationconditiontogenerateimagefromtextprompt.
-AU-Netforstep-by-stepdenoisinglatentimagerepresentation.
-Anautoencoder(VAE)forencodinginputimagetolatentspace(if
required)anddecodinglatentspacetoimagebackaftergeneration.

FormoredetailsregardingStableDiffusionwork,refertothe`project
website<https://ommer-lab.com/research/latent-diffusion-models/>`__.
ThereisatutorialforStableDiffusionText-to-Imagegenerationwith
OpenVINO,seethefollowing
`notebook<stable-diffusion-text-to-image-with-output.html>`__.

ControlNet
~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__ControlNetisaneuralnetwork
structuretocontroldiffusionmodelsbyaddingextraconditions.Using
thisnewframework,wecancaptureascene,structure,object,or
subjectposefromaninputtedimage,andthentransferthatqualityto
thegenerationprocess.Inpractice,thisenablesthemodelto
completelyretaintheoriginalinputshape,andcreateanovelimage
thatconservestheshape,pose,oroutlinewhileusingthenovel
featuresfromtheinputtedprompt.

..figure::https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/he.png
:alt:controlnetblock

controlnetblock

Functionally,ControlNetoperatesbywrappingaroundanimagesynthesis
processtoimpartattentiontotheshaperequiredtooperatethemodel
usingeitheritsinbuiltpredictionoroneofmanyadditionalannotator
models.Referringtothediagramabove,wecansee,onarudimentary
level,howControlNetusesatrainablecopyinconjunctionwiththe
originalnetworktomodifythefinaloutputwithrespecttotheshapeof
theinputcontrolsource.

Byrepeatingtheabovesimplestructure14times,wecancontrolstable
diffusioninthefollowingway:

..figure::https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/sd.png
:alt:sd+controlnet

sd+controlnet

TheinputissimultaneouslypassedthroughtheSDblocks,representedon
theleft,whilesimultaneouslybeingprocessedbytheControlNetblocks
ontheright.Thisprocessisalmostthesameduringencoding.When
denoisingtheimage,ateachsteptheSDdecoderblockswillreceive
controladjustmentsfromtheparallelprocessingpathfromControlNet.

Intheend,weareleftwithaverysimilarimagesynthesispipeline
withanadditionalcontroladdedfortheshapeoftheoutputfeaturesin
thefinalimage.

Low-RankAdaptationofLargeLanguageModels(LoRA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

`Low-RankAdaptationofLargeLanguageModels
(LoRA)<https://arxiv.org/abs/2106.09685>`__isatrainingmethodthat
acceleratesthetrainingoflargemodelswhileconsuminglessmemory.It
addspairsofrank-decompositionweightmatrices(calledupdate
matrices)toexistingweights,andonlytrainsthosenewlyadded
weights.Thishasacoupleofadvantages:

-LoRAmakesfine-tuningmoreefficientbydrasticallyreducingthe
numberoftrainableparameters.
-Theoriginalpre-trainedweightsarekeptfrozen,whichmeansyoucan
havemultiplelightweightandportableLoRAmodelsforvarious
downstreamtasksbuiltontopofthem.
-LoRAisorthogonaltomanyotherparameter-efficientmethodsandcan
becombinedwithmanyofthem.
-Performanceofmodelsfine-tunedusingLoRAiscomparabletothe
performanceoffullyfine-tunedmodels.
-LoRAdoesnotaddanyinferencelatencybecauseadapterweightscan
bemergedwiththebasemodel.

Inprinciple,LoRAcanbeappliedtoanysubsetofweightmatricesina
neuralnetworktoreducethenumberoftrainableparameters.However,
forsimplicityandfurtherparameterefficiency,inTransformermodels
LoRAistypicallyappliedtoattentionblocksonly.Theresultingnumber
oftrainableparametersinaLoRAmodeldependsonthesizeofthe
low-rankupdatematrices,whichisdeterminedmainlybytherankrand
theshapeoftheoriginalweightmatrix.MoredetailsaboutLoRAcanbe
foundinHuggingFace`conceptual
guide<https://huggingface.co/docs/peft/conceptual_guides/lora>`__,
`Diffusers
documentation<https://huggingface.co/docs/diffusers/training/lora>`__
and`blogpost<https://huggingface.co/blog/peft>`__.

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installrequiredpackages

..code::ipython3

%pipinstall-q"torch"transformers"diffusers>=0.22.0""controlnet-aux>=0.0.6""peft==0.6.2"accelerate--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2023.2.0"pillow"gradio>=4.19""datasets>=2.14.6""nncf>=2.7.0"

PreparePyTorchmodels

..code::ipython3

frompathlibimportPath

controlnet_id="lllyasviel/control_v11p_sd15_normalbae"
adapter_id="latent-consistency/lcm-lora-sdv1-5"
stable_diffusion_id="runwayml/stable-diffusion-v1-5"

TEXT_ENCODER_OV_PATH=Path("model/text_encoder.xml")
UNET_OV_PATH=Path("model/unet_controlnet.xml")
CONTROLNET_OV_PATH=Path("model/controlnet-normalbae.xml")
VAE_DECODER_OV_PATH=Path("model/vae_decoder.xml")
TOKENIZER_PATH=Path("model/tokenizer")
SCHEDULER_PATH=Path("model/scheduler")

skip_models=TEXT_ENCODER_OV_PATH.exists()andUNET_OV_PATH.exists()andCONTROLNET_OV_PATH.exists()andVAE_DECODER_OV_PATH.exists()

LoadOriginalDiffuserspipelineandpreparemodelsforconversion
------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

ForworkingwithStableDiffusionandControlNetmodels,wewilluse
HuggingFace`Diffusers<https://github.com/huggingface/diffusers>`__
library.ToexperimentwithControlNet,Diffusersexposesthe
`StableDiffusionControlNetPipeline<https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/controlnet>`__
similartothe`otherDiffusers
pipelines<https://huggingface.co/docs/diffusers/api/pipelines/overview>`__.
Centraltothe``StableDiffusionControlNetPipeline``isthe
``controlnet``argumentwhichenablesprovidingaparticularlytrained
`ControlNetModel<https://huggingface.co/docs/diffusers/main/en/api/models#diffusers.ControlNetModel>`__
instancewhilekeepingthepre-traineddiffusionmodelweightsthesame.

Thecodebelowdemonstrateshowtocreate
``StableDiffusionControlNetPipeline``.Theprocessconsistsofthe
followingsteps:1.Create``ControlNetModel``forpassingtopipeline
using``from_pretrained``method.2.Create
``StableDiffusionControlNetPipeline``usingStableDiffusionand
ControlNetmodel3.LoadLoRAweightstothepipelineusing
``load_lora_weights``method.

..code::ipython3

fromdiffusersimportStableDiffusionControlNetPipeline,ControlNetModel
importgc


defload_original_pytorch_pipeline_components(controlnet_id:str,stable_diffusion_id:str,adapter_id:str):
"""
HelperfunctionforloadingStableDiffusionControlNetpipelineandapplyingLCMLoRA

Parameters:
controlnet_id:modelidfromHuggingFacehuborlocalpathforloadingControlNetmodel
stable_diffusion_id:modelidfromHuggingFacehuborlocalpathforloadingStableDiffusionmodel
adapter_id:LCMLoRAidfromHuggingFacehuborlocalpath
Returns:
controlnet:ControlNetmodel
text_encoder:StableDiffusionTextEncoder
unet:StableDiffusionU-Net
vae:StableDiffusionVariationalAutoencoder(VAE)
"""

#loadcontrolnetmodel
controlnet=ControlNetModel.from_pretrained(controlnet_id)
#loadstablediffusionpipeline
pipe=StableDiffusionControlNetPipeline.from_pretrained(stable_diffusion_id,controlnet=controlnet)
#loadLCMLoRAweights
pipe.load_lora_weights(adapter_id)
#fuseLoRAweightswithUNet
pipe.fuse_lora()
text_encoder=pipe.text_encoder
text_encoder.eval()
unet=pipe.unet
unet.eval()
vae=pipe.vae
vae.eval()
delpipe
gc.collect()
returncontrolnet,text_encoder,unet,vae

..code::ipython3

controlnet,text_encoder,unet,vae=None,None,None,None
ifnotskip_models:
controlnet,text_encoder,unet,vae=load_original_pytorch_pipeline_components(controlnet_id,stable_diffusion_id,adapter_id)

ConditionImage
---------------

`backtotop⬆️<#table-of-contents>`__

Theprocessofextractingspecificinformationfromtheinputimageis
calledanannotation.ControlNetcomespre-packagedwithcompatibility
withseveralannotators-modelsthathelpittoidentifytheshape/form
ofthetargetintheimage:

-CannyEdgeDetection
-M-LSDLines
-HEDBoundary
-Scribbles
-NormalMap
-HumanPoseEstimation
-SemanticSegmentation
-DepthEstimation

Inthistutorialwewilluse`Normal
Mapping<https://en.wikipedia.org/wiki/Normal_mapping>`__for
controllingdiffusionprocess.Forthiscase,ControlNetconditionimage
isanimagewithsurfacenormalinformation,usuallyrepresentedasa
color-codedimage.

..code::ipython3

fromcontrolnet_auximportNormalBaeDetector
fromdiffusers.utilsimportload_image
importrequests
importmatplotlib.pyplotasplt
fromPILimportImage
importnumpyasnp

example_image_url="https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/images/input.png"
r=requests.get(example_image_url)
withopen("example.png","wb")asf:
f.write(r.content)

processor=NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

image=load_image("example.png")
control_image=processor(image)


defvisualize_results(
orig_img:Image.Image,
normal_img:Image.Image,
result_img:Image.Image=None,
save_fig:bool=False,
):
"""
Helperfunctionforresultsvisualization

Parameters:
orig_img(Image.Image):originalimage
normal_img(Image.Image):imagewithbwithsurfacenormalinformation
result_img(Image.Image,optional,defaultNone):generatedimage
safe_fig(bool,optional,defaultFalse):allowsavingvisualizationresultondisk
Returns:
fig(matplotlib.pyplot.Figure):matplotlibgeneratedfigurecontainsdrawingresult
"""
orig_title="Originalimage"
control_title="Normalmap"
orig_img=orig_img.resize(normal_img.sizeifresult_imgisNoneelseresult_img.size)
im_w,im_h=orig_img.size
is_horizontal=im_h<=im_w
figsize=(20,20)
num_images=3ifresult_imgisnotNoneelse2
fig,axs=plt.subplots(
num_imagesifis_horizontalelse1,
1ifis_horizontalelsenum_images,
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
list_axes[0].imshow(np.array(orig_img))
list_axes[1].imshow(np.array(normal_img))
list_axes[0].set_title(orig_title,fontsize=15)
list_axes[1].set_title(control_title,fontsize=15)
ifresult_imgisnotNone:
list_axes[2].imshow(np.array(result_img))
list_axes[2].set_title("Result",fontsize=15)

fig.subplots_adjust(wspace=0.01ifis_horizontalelse0.00,hspace=0.01ifis_horizontalelse0.1)
fig.tight_layout()
ifsave_fig:
fig.savefig("result.png",bbox_inches="tight")
returnfig


fig=visualize_results(image,control_image)


..parsed-literal::

Loadingbasemodel()...Done.
Removinglasttwolayers(global_pool&classifier).



..image::lcm-lora-controlnet-with-output_files/lcm-lora-controlnet-with-output_10_1.png


ConvertmodelstoOpenVINOIntermediaterepresentation(IR)format
------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Startingfrom2023.0release,OpenVINOsupportsPyTorchmodels
conversiondirectly.Weneedtoprovideamodelobject,inputdatafor
modeltracingto``ov.convert_model``functiontoobtainOpenVINO
``ov.Model``objectinstance.Modelcanbesavedondiskfornext
deploymentusing``ov.save_model``function.

Thepipelineconsistsoffiveimportantparts:

-ControlNetforconditioningbyimageannotation.
-TextEncoderforcreationconditiontogenerateanimagefromatext
prompt.
-Unetforstep-by-stepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)fordecodinglatentspacetoimage.

Letusconverteachpart:

ControlNetconversion
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheControlNetmodelacceptsthesameinputslikeUNetinStable
Diffusionpipelineandadditionalconditionsample-skeletonkeypoints
mappredictedbyposeestimator:

-``sample``-latentimagesamplefromthepreviousstep,generation
processhasnotbeenstartedyet,sowewilluserandomnoise,
-``timestep``-currentschedulerstep,
-``encoder_hidden_state``-hiddenstateoftextencoder,
-``controlnet_cond``-conditioninputannotation.

Theoutputofthemodelisattentionhiddenstatesfromdownandmiddle
blocks,whichservesadditionalcontextfortheUNetmodel.

..code::ipython3

importtorch
importopenvinoasov
fromfunctoolsimportpartial


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


defflattenize_inputs(inputs):
"""
Helperfunctionforresolvenestedinputstructure(e.g.listsortuplesoftensors)
"""
flatten_inputs=[]
forinput_dataininputs:
ifinput_dataisNone:
continue
ifisinstance(input_data,(list,tuple)):
flatten_inputs.extend(flattenize_inputs(input_data))
else:
flatten_inputs.append(input_data)
returnflatten_inputs


dtype_mapping={
torch.float32:ov.Type.f32,
torch.float64:ov.Type.f64,
torch.int32:ov.Type.i32,
torch.int64:ov.Type.i64,
}


defprepare_input_info(input_dict):
"""
Helperfunctionforpreparinginputinfo(shapesanddatatypes)forconversionbasedonexampleinputs
"""
flatten_inputs=flattenize_inputs(inputs.values())
input_info=[]
forinput_datainflatten_inputs:
updated_shape=list(input_data.shape)
ifupdated_shape:
updated_shape[0]=-1
ifinput_data.ndim==4:
updated_shape[2]=-1
updated_shape[3]=-1

input_info.append((dtype_mapping[input_data.dtype],updated_shape))
returninput_info


inputs={
"sample":torch.randn((1,4,64,64)),
"timestep":torch.tensor(1,dtype=torch.float32),
"encoder_hidden_states":torch.randn((1,77,768)),
"controlnet_cond":torch.randn((1,3,512,512)),
}


#PrepareconditionalinputsforU-Net
ifnotUNET_OV_PATH.exists():
controlnet.eval()
withtorch.no_grad():
down_block_res_samples,mid_block_res_sample=controlnet(**inputs,return_dict=False)

ifnotCONTROLNET_OV_PATH.exists():
input_info=prepare_input_info(inputs)
withtorch.no_grad():
controlnet.forward=partial(controlnet.forward,return_dict=False)
ov_model=ov.convert_model(controlnet,example_input=inputs,input=input_info)
ov.save_model(ov_model,CONTROLNET_OV_PATH)
delov_model
cleanup_torchscript_cache()
print("ControlNetsuccessfullyconvertedtoIR")
else:
print(f"ControlNetwillbeloadedfrom{CONTROLNET_OV_PATH}")

delcontrolnet
gc.collect()


..parsed-literal::

ControlNetwillbeloadedfrommodel/controlnet-normalbae.xml




..parsed-literal::

9



U-Net
~~~~~

`backtotop⬆️<#table-of-contents>`__

TheprocessofU-Netmodelconversionremainsthesame,likefor
originalStableDiffusionmodel,butwithrespecttothenewinputs
generatedbyControlNet.

..code::ipython3

fromtypingimportTuple


classUnetWrapper(torch.nn.Module):
def__init__(
self,
unet,
sample_dtype=torch.float32,
timestep_dtype=torch.int64,
encoder_hidden_states=torch.float32,
down_block_additional_residuals=torch.float32,
mid_block_additional_residual=torch.float32,
):
super().__init__()
self.unet=unet
self.sample_dtype=sample_dtype
self.timestep_dtype=timestep_dtype
self.encoder_hidden_states_dtype=encoder_hidden_states
self.down_block_additional_residuals_dtype=down_block_additional_residuals
self.mid_block_additional_residual_dtype=mid_block_additional_residual

defforward(
self,
sample:torch.Tensor,
timestep:torch.Tensor,
encoder_hidden_states:torch.Tensor,
down_block_additional_residuals:Tuple[torch.Tensor],
mid_block_additional_residual:torch.Tensor,
):
sample.to(self.sample_dtype)
timestep.to(self.timestep_dtype)
encoder_hidden_states.to(self.encoder_hidden_states_dtype)
down_block_additional_residuals=[res.to(self.down_block_additional_residuals_dtype)forresindown_block_additional_residuals]
mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
returnself.unet(
sample,
timestep,
encoder_hidden_states,
down_block_additional_residuals=down_block_additional_residuals,
mid_block_additional_residual=mid_block_additional_residual,
)


ifnotUNET_OV_PATH.exists():
inputs.pop("controlnet_cond",None)
inputs["down_block_additional_residuals"]=down_block_res_samples
inputs["mid_block_additional_residual"]=mid_block_res_sample
input_info=prepare_input_info(inputs)

wrapped_unet=UnetWrapper(unet)
wrapped_unet.eval()

withtorch.no_grad():
ov_model=ov.convert_model(wrapped_unet,example_input=inputs)

for(input_dtype,input_shape),input_tensorinzip(input_info,ov_model.inputs):
input_tensor.get_node().set_partial_shape(ov.PartialShape(input_shape))
input_tensor.get_node().set_element_type(input_dtype)
ov_model.validate_nodes_and_infer_types()
ov.save_model(ov_model,UNET_OV_PATH)
delov_model
cleanup_torchscript_cache()
delwrapped_unet
delunet
gc.collect()
print("UnetsuccessfullyconvertedtoIR")
else:
delunet
print(f"Unetwillbeloadedfrom{UNET_OV_PATH}")
gc.collect()


..parsed-literal::

Unetwillbeloadedfrommodel/unet_controlnet.xml




..parsed-literal::

0



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

defconvert_encoder(text_encoder:torch.nn.Module,ir_path:Path):
"""
ConvertTextEncodermodeltoOpenVINOIR.
Functionacceptstextencodermodel,preparesexampleinputsforconversion,andconvertittoOpenVINOModel
Parameters:
text_encoder(torch.nn.Module):text_encodermodel
ir_path(Path):Fileforstoringmodel
Returns:
None
"""
ifnotir_path.exists():
input_ids=torch.ones((1,77),dtype=torch.long)
#switchmodeltoinferencemode
text_encoder.eval()

#disablegradientscalculationforreducingmemoryconsumption
withtorch.no_grad():
ov_model=ov.convert_model(
text_encoder,#modelinstance
example_input=input_ids,#inputsformodeltracing
input=([1,77],),
)
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
print("TextEncodersuccessfullyconvertedtoIR")


ifnotTEXT_ENCODER_OV_PATH.exists():
convert_encoder(text_encoder,TEXT_ENCODER_OV_PATH)
else:
print(f"Textencoderwillbeloadedfrom{TEXT_ENCODER_OV_PATH}")
deltext_encoder
gc.collect()


..parsed-literal::

Textencoderwillbeloadedfrommodel/text_encoder.xml




..parsed-literal::

0



VAEDecoderconversion
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheVAEmodelhastwoparts,anencoder,andadecoder.Theencoderis
usedtoconverttheimageintoalow-dimensionallatentrepresentation,
whichwillserveastheinputtotheU-Netmodel.Thedecoder,
conversely,transformsthelatentrepresentationbackintoanimage.

Duringlatentdiffusiontraining,theencoderisusedtogetthelatent
representations(latents)oftheimagesfortheforwarddiffusion
process,whichappliesmoreandmorenoiseateachstep.During
inference,thedenoisedlatentsgeneratedbythereversediffusion
processareconvertedbackintoimagesusingtheVAEdecoder.During
inference,wewillseethatwe**onlyneedtheVAEdecoder**.Youcan
findinstructionsonhowtoconverttheencoderpartinastable
diffusion
`notebook<stable-diffusion-text-to-image-with-output.html>`__.

..code::ipython3

defconvert_vae_decoder(vae:torch.nn.Module,ir_path:Path):
"""
ConvertVAEmodeltoIRformat.
Functionacceptspipeline,createswrapperclassforexportonlynecessaryforinferencepart,
preparesexampleinputsforconvert,
Parameters:
vae(torch.nn.Module):VAEmodel
ir_path(Path):Fileforstoringmodel
Returns:
None
"""

classVAEDecoderWrapper(torch.nn.Module):
def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,latents):
returnself.vae.decode(latents)

ifnotir_path.exists():
vae_decoder=VAEDecoderWrapper(vae)
latents=torch.zeros((1,4,64,64))

vae_decoder.eval()
withtorch.no_grad():
ov_model=ov.convert_model(vae_decoder,example_input=latents,input=[-1,4,-1,-1])
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
print("VAEdecodersuccessfullyconvertedtoIR")


ifnotVAE_DECODER_OV_PATH.exists():
convert_vae_decoder(vae,VAE_DECODER_OV_PATH)
else:
print(f"VAEdecoderwillbeloadedfrom{VAE_DECODER_OV_PATH}")

delvae


..parsed-literal::

VAEdecoderwillbeloadedfrommodel/vae_decoder.xml


PrepareInferencepipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

WealreadydeeplydiscussedhowtheControlNet-guidedpipelineworkson
examplepose-controlledgenerationin`controlnet
notebook<../controlnet-stable-diffusion>`__.Inourcurrentexample,
thepipelineremainswithoutchanges.SimilarlytoDiffusers
``StableDiffusionControlNetPipeline``,wedefineourown
``OVControlNetStableDiffusionPipeline``inferencepipelinebasedon
OpenVINO.

..code::ipython3

fromdiffusersimportDiffusionPipeline
fromtransformersimportCLIPTokenizer
fromtypingimportUnion,List,Optional,Tuple
importcv2


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


defpreprocess(image:Image.Image,dst_height:int=512,dst_width:int=512):
"""
Imagepreprocessingfunction.TakesimageinPIL.Imageformat,resizesittokeepaspectrationandfitstomodelinputwindow512x512,
thenconvertsittonp.ndarrayandaddspaddingwithzerosonrightorbottomsideofimage(dependsfromaspectratio),afterthat
convertsdatatofloat32datatypeandchangerangeofvaluesfrom[0,255]to[-1,1],finally,convertsdatalayoutfromplanarNHWCtoNCHW.
Thefunctionreturnspreprocessedinputtensorandpaddingsize,whichcanbeusedinpostprocessing.

Parameters:
image(Image.Image):inputimage
dst_width:destinationimagewidth
dst_height:destinationimageheight
Returns:
image(np.ndarray):preprocessedimagetensor
pad(Tuple[int]):padingsizeforeachdimensionforrestoringimagesizeinpostprocessing
"""
src_width,src_height=image.size
res_width,res_height=scale_fit_to_window(dst_width,dst_height,src_width,src_height)
image=np.array(image.resize((res_width,res_height),resample=Image.Resampling.LANCZOS))[None,:]
pad_width=dst_width-res_width
pad_height=dst_height-res_height
pad=((0,0),(0,pad_height),(0,pad_width),(0,0))
image=np.pad(image,pad,mode="constant")
image=image.astype(np.float32)/255.0
image=image.transpose(0,3,1,2)
returnimage,pad


defrandn_tensor(
shape:Union[Tuple,List],
dtype:Optional[torch.dtype]=torch.float32,
):
"""
Helperfunctionforgenerationrandomvaluestensorwithgivenshapeanddatatype

Parameters:
shape(Union[Tuple,List]):shapeforfillingrandomvalues
dtype(torch.dtype,*optiona*,torch.float32):datatypeforresult
Returns:
latents(np.ndarray):tensorwithrandomvalueswithgivendatatypeandshape(usuallyrepresentsnoiseinlatentspace)
"""
latents=torch.randn(shape,dtype=dtype)
returnlatents.numpy()


classOVControlNetStableDiffusionPipeline(DiffusionPipeline):
"""
OpenVINOinferencepipelineforStableDiffusionwithControlNetguidence
"""

def__init__(
self,
tokenizer:CLIPTokenizer,
scheduler,
core:ov.Core,
controlnet:ov.Model,
text_encoder:ov.Model,
unet:ov.Model,
vae_decoder:ov.Model,
device:str="AUTO",
):
super().__init__()
self.tokenizer=tokenizer
self.vae_scale_factor=8
self.scheduler=scheduler
self.load_models(core,device,controlnet,text_encoder,unet,vae_decoder)

defload_models(
self,
core:ov.Core,
device:str,
controlnet:ov.Model,
text_encoder:ov.Model,
unet:ov.Model,
vae_decoder:ov.Model,
):
"""
FunctionforloadingmodelsondeviceusingOpenVINO

Parameters:
core(Core):OpenVINOruntimeCoreclassinstance
device(str):inferencedevice
controlnet(Model):OpenVINOModelobjectrepresentsControlNet
text_encoder(Model):OpenVINOModelobjectrepresentstextencoder
unet(Model):OpenVINOModelobjectrepresentsUNet
vae_decoder(Model):OpenVINOModelobjectrepresentsvaedecoder
Returns
None
"""
self.text_encoder=core.compile_model(text_encoder,device)
self.register_to_config(controlnet=core.compile_model(controlnet,device))
self.register_to_config(unet=core.compile_model(unet,device))
ov_config={"INFERENCE_PRECISION_HINT":"f32"}ifdevice!="CPU"else{}
self.vae_decoder=core.compile_model(vae_decoder,device,ov_config)

def__call__(
self,
prompt:Union[str,List[str]],
image:Image.Image,
num_inference_steps:int=4,
height:int=512,
width:int=512,
negative_prompt:Union[str,List[str]]=None,
guidance_scale:float=0.5,
controlnet_conditioning_scale:float=1.0,
latents:Optional[np.array]=None,
output_type:Optional[str]="pil",
):
"""
Functioninvokedwhencallingthepipelineforgeneration.

Parameters:
prompt(`str`or`List[str]`):
Thepromptorpromptstoguidetheimagegeneration.
image(`Image.Image`):
`Image`,ortensorrepresentinganimagebatchwhichwillberepaintedaccordingto`prompt`.
num_inference_steps(`int`,*optional*,defaultsto100):
Thenumberofdenoisingsteps.Moredenoisingstepsusuallyleadtoahigherqualityimageatthe
expenseofslowerinference.
height(int,*optional*,defaultsto512):generatedimageheight
width(int,*optional*,defaultsto512):generatedimagewidth
negative_prompt(`str`or`List[str]`):
negativepromptorpromptsforgeneration
guidance_scale(`float`,*optional*,defaultsto0.5):
Guidancescaleasdefinedin[Classifier-FreeDiffusionGuidance](https://arxiv.org/abs/2207.12598).
`guidance_scale`isdefinedas`w`ofequation2.of[Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf).Guidancescaleisenabledbysetting`guidance_scale>
1`.Higherguidancescaleencouragestogenerateimagesthatarecloselylinkedtothetext`prompt`,
usuallyattheexpenseoflowerimagequality.Thispipelinerequiresavalueofatleast`1`.
latents(`np.ndarray`,*optional*):
Pre-generatednoisylatents,sampledfromaGaussiandistribution,tobeusedasinputsforimage
generation.Canbeusedtotweakthesamegenerationwithdifferentprompts.Ifnotprovided,alatents
tensorwillgegeneratedbysamplingusingthesuppliedrandom`generator`.
output_type(`str`,*optional*,defaultsto`"pil"`):
Theoutputformatofthegenerateimage.Choosebetween
[PIL](https://pillow.readthedocs.io/en/stable/):`Image.Image`or`np.array`.
Returns:
image([List[Union[np.ndarray,Image.Image]]):generaitedimages

"""

#1.Definecallparameters
batch_size=1ifisinstance(prompt,str)elselen(prompt)
ifguidance_scale<1andnegative_prompt:
guidance_scale+=1
#here`guidance_scale`isdefinedanalogtotheguidanceweight`w`ofequation(2)
#oftheImagenpaper:https://arxiv.org/pdf/2205.11487.pdf.`guidance_scale=1`
#correspondstodoingnoclassifierfreeguidance.
do_classifier_free_guidance=guidance_scale>1.0
#2.Encodeinputprompt
text_embeddings=self._encode_prompt(
prompt,
do_classifier_free_guidance=do_classifier_free_guidance,
negative_prompt=negative_prompt,
)

#3.Preprocessimage
orig_width,orig_height=image.size
image,pad=preprocess(image,height,width)
ifdo_classifier_free_guidance:
image=np.concatenate(([image]*2))

#4.settimesteps
self.scheduler.set_timesteps(num_inference_steps)
timesteps=self.scheduler.timesteps

#5.Preparelatentvariables
num_channels_latents=4
latents=self.prepare_latents(
batch_size,
num_channels_latents,
height,
width,
latents=latents,
)

#6.Denoisingloop
withself.progress_bar(total=num_inference_steps)asprogress_bar:
fori,tinenumerate(timesteps):
#Expandthelatentsifwearedoingclassifierfreeguidance.
#Thelatentsareexpanded3timesbecauseforpix2pixtheguidance\
#isappliedforboththetextandtheinputimage.
latent_model_input=np.concatenate([latents]*2)ifdo_classifier_free_guidanceelselatents
latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)

result=self.controlnet(
[latent_model_input,t,text_embeddings,image],
share_inputs=True,
share_outputs=True,
)
down_and_mid_blok_samples=[sample*controlnet_conditioning_scalefor_,sampleinresult.items()]

#predictthenoiseresidual
noise_pred=self.unet(
[
latent_model_input,
t,
text_embeddings,
*down_and_mid_blok_samples,
],
share_inputs=True,
share_outputs=True,
)[0]

#performguidance
ifdo_classifier_free_guidance:
noise_pred_uncond,noise_pred_text=noise_pred[0],noise_pred[1]
noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

#computethepreviousnoisysamplex_t->x_t-1
latents=self.scheduler.step(torch.from_numpy(noise_pred),t,torch.from_numpy(latents)).prev_sample.numpy()
progress_bar.update()

#7.Post-processing
image=self.decode_latents(latents,pad)

#8.ConverttoPIL
ifoutput_type=="pil":
image=self.numpy_to_pil(image)
image=[img.resize((orig_width,orig_height),Image.Resampling.LANCZOS)forimginimage]
else:
image=[cv2.resize(img,(orig_width,orig_width))forimginimage]

returnimage

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
negative_prompt(strorlist(str)):negativeprompttobeencoded
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

text_embeddings=self.text_encoder(text_input_ids,share_inputs=True,share_outputs=True)[0]

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

uncond_embeddings=self.text_encoder(uncond_input.input_ids,share_inputs=True,share_outputs=True)[0]

#duplicateunconditionalembeddingsforeachgenerationperprompt,usingmpsfriendlymethod
seq_len=uncond_embeddings.shape[1]
uncond_embeddings=np.tile(uncond_embeddings,(1,num_images_per_prompt,1))
uncond_embeddings=np.reshape(uncond_embeddings,(batch_size*num_images_per_prompt,seq_len,-1))

#Forclassifierfreeguidance,weneedtodotwoforwardpasses.
#Hereweconcatenatetheunconditionalandtextembeddingsintoasinglebatch
#toavoiddoingtwoforwardpasses
text_embeddings=np.concatenate([uncond_embeddings,text_embeddings])

returntext_embeddings

defprepare_latents(
self,
batch_size:int,
num_channels_latents:int,
height:int,
width:int,
dtype:np.dtype=torch.float32,
latents:np.ndarray=None,
):
"""
Preparingnoisetoimagegeneration.Ifinitiallatentsarenotprovided,theywillbegeneratedrandomly,
thenpreparedlatentsscaledbythestandarddeviationrequiredbythescheduler

Parameters:
batch_size(int):inputbatchsize
num_channels_latents(int):numberofchannelsfornoisegeneration
height(int):imageheight
width(int):imagewidth
dtype(np.dtype,*optional*,np.float32):dtypeforlatentsgeneration
latents(np.ndarray,*optional*,None):initiallatentnoisetensor,ifnotprovidedwillbegenerated
Returns:
latents(np.ndarray):scaledinitialnoisefordiffusion
"""
shape=(
batch_size,
num_channels_latents,
height//self.vae_scale_factor,
width//self.vae_scale_factor,
)
iflatentsisNone:
latents=randn_tensor(shape,dtype=dtype)
else:
latents=latents

#scaletheinitialnoisebythestandarddeviationrequiredbythescheduler
latents=latents*self.scheduler.init_noise_sigma
returnlatents

defdecode_latents(self,latents:np.array,pad:Tuple[int]):
"""
DecodepredictedimagefromlatentspaceusingVAEDecoderandunpadimageresult

Parameters:
latents(np.ndarray):imageencodedindiffusionlatentspace
pad(Tuple[int]):eachsidepaddingsizesobtainedonpreprocessingstep
Returns:
image:decodedbyVAEdecoderimage
"""
latents=1/0.18215*latents
image=self.vae_decoder(latents)[0]
(_,end_h),(_,end_w)=pad[1:3]
h,w=image.shape[2:]
unpad_h=h-end_h
unpad_w=w-end_w
image=image[:,:,:unpad_h,:unpad_w]
image=np.clip(image/2+0.5,0,1)
image=np.transpose(image,(0,2,3,1))
returnimage

PreparetokenizerandLCMScheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tokenizerandschedulerarealsoimportantpartsofthediffusion
pipeline.Thetokenizerisresponsibleforpreprocessinguser-provided
promptsintotokenidsthatthenusedbyTextEncoder.

Theschedulertakesamodel’soutput(thesamplewhichthediffusion
processisiteratingon)andatimesteptoreturnadenoisedsample.The
timestepisimportantbecauseitdictateswhereinthediffusionprocess
thestepis;dataisgeneratedbyiteratingforwardntimestepsand
inferenceoccursbypropagatingbackwardthroughthetimesteps.There
aremany
`schedulers<https://huggingface.co/docs/diffusers/api/schedulers/overview>`__
implementedinsidethediffuserslibrary,LCMpipelinerequiredchanging
theoriginalpipelineschedulerwith
`LCMScheduler<https://huggingface.co/docs/diffusers/api/schedulers/lcm>`__.

..code::ipython3

fromdiffusersimportLCMScheduler
fromtransformersimportAutoTokenizer

ifnotTOKENIZER_PATH.exists():
tokenizer=AutoTokenizer.from_pretrained(stable_diffusion_id,subfolder="tokenizer")
tokenizer.save_pretrained(TOKENIZER_PATH)
else:
tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH)
ifnotSCHEDULER_PATH.exists():
scheduler=LCMScheduler.from_pretrained(stable_diffusion_id,subfolder="scheduler")
scheduler.save_pretrained(SCHEDULER_PATH)
else:
scheduler=LCMScheduler.from_config(SCHEDULER_PATH)

SelectinferencedeviceforStableDiffusionpipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

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

Dropdown(description='Device:',options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='CPU')



..code::ipython3

ov_pipe=OVControlNetStableDiffusionPipeline(
tokenizer,
scheduler,
core,
CONTROLNET_OV_PATH,
TEXT_ENCODER_OV_PATH,
UNET_OV_PATH,
VAE_DECODER_OV_PATH,
device=device.value,
)

RunningText-to-ImageGenerationwithControlNetConditioningandOpenVINO
--------------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Now,wearereadytostartgeneration.Forimprovingthegeneration
process,wealsointroduceanopportunitytoprovidea
``negativeprompt``.Technically,positivepromptsteersthediffusion
towardtheimagesassociatedwithit,whilenegativepromptsteersthe
diffusionawayfromit.Moreexplanationofhowitworkscanbefoundin
this
`article<https://stable-diffusion-art.com/how-negative-prompt-work/>`__.
Wecankeepthisfieldemptyifwewanttogenerateimagewithout
negativeprompting.

`Classifier-freeguidance(CFG)<https://arxiv.org/abs/2207.12598>`__or
guidancescaleisaparameterthatcontrolshowmuchtheimage
generationprocessfollowsthetextprompt.Thehigherthevalue,the
moretheimagestickstoagiventextinput.Butthisdoesnotmeanthat
thevalueshouldalwaysbesettomaximum,asmoreguidancemeansless
diversityandquality.Accordingtoexperiments,theoptimalvalueof
guidanceforLCMmodelsisinrangebetween0and2.>Pleasenote,that
negativepromptisapplicableonlywhenguidancescale>1.

Let’sseemodelinaction

..code::ipython3

prompt="Aheadfullofroses"
torch.manual_seed(4257)

result=ov_pipe(prompt,control_image,4)
result[0]



..parsed-literal::

0%||0/4[00:00<?,?it/s]


..parsed-literal::

/home/ltalamanova/omz/lib/python3.8/site-packages/diffusers/configuration_utils.py:135:FutureWarning:Accessingconfigattribute`controlnet`directlyvia'OVControlNetStableDiffusionPipeline'objectattributeisdeprecated.Pleaseaccess'controlnet'over'OVControlNetStableDiffusionPipeline'sconfigobjectinstead,e.g.'scheduler.config.controlnet'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)
/home/ltalamanova/omz/lib/python3.8/site-packages/diffusers/configuration_utils.py:135:FutureWarning:Accessingconfigattribute`unet`directlyvia'OVControlNetStableDiffusionPipeline'objectattributeisdeprecated.Pleaseaccess'unet'over'OVControlNetStableDiffusionPipeline'sconfigobjectinstead,e.g.'scheduler.config.unet'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)




..image::lcm-lora-controlnet-with-output_files/lcm-lora-controlnet-with-output_27_2.png



..code::ipython3

fig=visualize_results(image,control_image,result[0])



..image::lcm-lora-controlnet-with-output_files/lcm-lora-controlnet-with-output_28_0.png


Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``OVControlNetStableDiffusionPipeline``structure,
ControlNetandUNetareusedinthecyclerepeatinginferenceoneach
diffusionstep,whileotherpartsofpipelinetakepartonlyonce.That
iswhycomputationcostandspeedofControlNetandUNetbecomethe
criticalpathinthepipeline.QuantizingtherestoftheSDpipeline
doesnotsignificantlyimproveinferenceperformancebutcanleadtoa
substantialdegradationofaccuracy.

Theoptimizationprocesscontainsthefollowingsteps:

1.Createacalibrationdatasetforquantization.
2.Run``nncf.quantize()``toobtainquantizedmodel.
3.Savethe``INT8``modelusing``openvino.save_model()``function.

Pleaseselectbelowwhetheryouwouldliketorunquantizationto
improvemodelinferencespeed.

..code::ipython3

skip_for_device="GPU"indevice.value
to_quantize=widgets.Checkbox(value=notskip_for_device,description="Quantization",disabled=skip_for_device)

to_quantize

Let’sload``skipmagic``extensiontoskipquantizationif
``to_quantize``isnotselected

..code::ipython3

#Fetch`skip_kernel_extension`module
r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

int8_pipe=None

%load_extskip_kernel_extension

Preparecalibrationdatasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseaportionof
`fusing/instructpix2pix-1000-samples<https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples>`__
datasetfromHuggingFaceascalibrationdataforControlNetandUNet.

Tocollectintermediatemodelinputsforcalibrationweshouldcustomize
``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

importdatasets
fromtqdm.notebookimporttqdm
fromtransformersimportset_seed
fromtypingimportAny,Dict,List

set_seed(1)

classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model,prob:float):
super().__init__(compiled_model)
self.data_cache=[]
self.prob=np.clip(prob,0,1)

def__call__(self,*args,**kwargs):
ifnp.random.rand()>=self.prob:
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)

defcollect_calibration_data(pipeline:OVControlNetStableDiffusionPipeline,subset_size:int)->List[Dict]:
original_unet=pipeline.unet
pipeline.unet=CompiledModelDecorator(original_unet,prob=0.3)

dataset=datasets.load_dataset("fusing/instructpix2pix-1000-samples",split="train",streaming=True).shuffle(seed=42)
pipeline.set_progress_bar_config(disable=True)

#Runinferencefordatacollection
pbar=tqdm(total=subset_size)
diff=0
control_images=[]
forbatchindataset:
prompt=batch["edit_prompt"]
iflen(prompt)>tokenizer.model_max_length:
continue
image=batch["input_image"]
control_image=processor(image)

_=pipeline(prompt,image=control_image,num_inference_steps=4)
collected_subset_size=len(pipeline.unet.data_cache)
control_images.append((min(collected_subset_size,subset_size),control_image))
ifcollected_subset_size>=subset_size:
pbar.update(subset_size-pbar.n)
break
pbar.update(collected_subset_size-diff)
diff=collected_subset_size

control_calibration_dataset=pipeline.unet.data_cache
pipeline.set_progress_bar_config(disable=False)
pipeline.unet=original_unet
returncontrol_calibration_dataset,control_images

..code::ipython3

%%skipnot$to_quantize.value

CONTROLNET_INT8_OV_PATH=Path("model/controlnet-normalbae_int8.xml")
UNET_INT8_OV_PATH=Path("model/unet_controlnet_int8.xml")
ifnot(CONTROLNET_INT8_OV_PATH.exists()andUNET_INT8_OV_PATH.exists()):
subset_size=200
unet_calibration_data,control_images=collect_calibration_data(ov_pipe,subset_size=subset_size)



..parsed-literal::

0%||0/200[00:00<?,?it/s]


ThefirstthreeinputsofControlNetarethesameastheinputsofUNet,
thelastControlNetinputisapreprocessed``control_image``.

..code::ipython3

%%skipnot$to_quantize.value

ifnotCONTROLNET_INT8_OV_PATH.exists():
control_calibration_data=[]
prev_idx=0
forupper_bound,imageincontrol_images:
preprocessed_image,_=preprocess(image)
foriinrange(prev_idx,upper_bound):
control_calibration_data.append(unet_calibration_data[i][:3]+[preprocessed_image])
prev_idx=upper_bound

Runquantization
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Createaquantizedmodelfromthepre-trainedconvertedOpenVINOmodel.
``FastBiasCorrection``algorithmisdisabledduetominimalaccuracy
improvementinSDmodelsandincreasedquantizationtime.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

%%skipnot$to_quantize.value

importnncf

ifnotUNET_INT8_OV_PATH.exists():
unet=core.read_model(UNET_OV_PATH)
quantized_unet=nncf.quantize(
model=unet,
calibration_dataset=nncf.Dataset(unet_calibration_data),
model_type=nncf.ModelType.TRANSFORMER,
advanced_parameters=nncf.AdvancedQuantizationParameters(
disable_bias_correction=True
)
)
ov.save_model(quantized_unet,UNET_INT8_OV_PATH)

..code::ipython3

%%skipnot$to_quantize.value

ifnotCONTROLNET_INT8_OV_PATH.exists():
controlnet=core.read_model(CONTROLNET_OV_PATH)
quantized_controlnet=nncf.quantize(
model=controlnet,
calibration_dataset=nncf.Dataset(control_calibration_data),
model_type=nncf.ModelType.TRANSFORMER,
advanced_parameters=nncf.AdvancedQuantizationParameters(
disable_bias_correction=True
)
)
ov.save_model(quantized_controlnet,CONTROLNET_INT8_OV_PATH)

LetuscheckpredictionswiththequantizedControlNetandUNetusing
thesameinputdata.

..code::ipython3

%%skipnot$to_quantize.value

fromIPython.displayimportdisplay

int8_pipe=OVControlNetStableDiffusionPipeline(
tokenizer,
scheduler,
core,
CONTROLNET_INT8_OV_PATH,
TEXT_ENCODER_OV_PATH,
UNET_INT8_OV_PATH,
VAE_DECODER_OV_PATH,
device=device.value
)

prompt="Aheadfullofroses"
torch.manual_seed(4257)

int8_result=int8_pipe(prompt,control_image,4)

fig=visualize_results(result[0],int8_result[0])
fig.axes[0].set_title('FP16result',fontsize=15)
fig.axes[1].set_title('INT8result',fontsize=15)




..parsed-literal::

0%||0/4[00:00<?,?it/s]



..image::lcm-lora-controlnet-with-output_files/lcm-lora-controlnet-with-output_42_1.png


CompareinferencetimeoftheFP16andINT8models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tomeasuretheinferenceperformanceofthe``FP16``and``INT8``
pipelines,weusemedianinferencetimeoncalibrationsubset.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.

..code::ipython3

%%skipnot$to_quantize.value

importtime

validation_size=10
calibration_dataset=datasets.load_dataset("fusing/instructpix2pix-1000-samples",split="train",streaming=True).take(validation_size)
validation_data=[]
forbatchincalibration_dataset:
prompt=batch["edit_prompt"]
image=batch["input_image"]
control_image=processor(image)
validation_data.append((prompt,control_image))

defcalculate_inference_time(pipeline,calibration_dataset):
inference_time=[]
pipeline.set_progress_bar_config(disable=True)
forprompt,control_imageincalibration_dataset:
start=time.perf_counter()
_=pipeline(prompt,control_image,num_inference_steps=4)
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
returnnp.median(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

fp_latency=calculate_inference_time(ov_pipe,validation_data)
int8_latency=calculate_inference_time(int8_pipe,validation_data)
print(f"Performancespeedup:{fp_latency/int8_latency:.3f}")


..parsed-literal::

Performancespeedup:1.257


Comparemodelfilesizes
^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fp16_ir_model_size=UNET_OV_PATH.with_suffix(".bin").stat().st_size/2**20
quantized_model_size=UNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size/2**20

print(f"FP16UNetsize:{fp16_ir_model_size:.2f}MB")
print(f"INT8UNetsize:{quantized_model_size:.2f}MB")
print(f"UNetcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16UNetsize:1639.41MB
INT8UNetsize:820.96MB
UNetcompressionrate:1.997


..code::ipython3

%%skipnot$to_quantize.value

fp16_ir_model_size=CONTROLNET_OV_PATH.with_suffix(".bin").stat().st_size/2**20
quantized_model_size=CONTROLNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size/2**20

print(f"FP16ControlNetsize:{fp16_ir_model_size:.2f}MB")
print(f"INT8ControlNetsize:{quantized_model_size:.2f}MB")
print(f"ControlNetcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16ControlNetsize:689.07MB
INT8ControlNetsize:345.12MB
ControlNetcompressionrate:1.997


InteractiveDemo
----------------

`backtotop⬆️<#table-of-contents>`__

Now,youcantestmodelonownimages.Please,provideimageinto
``InputImage``windowandpromptsforgenerationandclick``Run``
button.Toachievethebestresults,youalsocanselectadditional
optionsforgeneration:``Guidancescale``,``Seed``and``Steps``.

..code::ipython3

importgradioasgr

MAX_SEED=np.iinfo(np.int32).max

quantized_model_present=int8_pipeisnotNone

gr.close_all()
withgr.Blocks()asdemo:
withgr.Row():
withgr.Column():
inp_img=gr.Image(label="Inputimage")
withgr.Column(visible=True)asstep1:
out_normal=gr.Image(label="NormalMap",type="pil",interactive=False)
btn=gr.Button()
inp_prompt=gr.Textbox(label="Prompt")
inp_neg_prompt=gr.Textbox(
"",
label="Negativeprompt",
)
withgr.Accordion("Advancedoptions",open=False):
guidance_scale=gr.Slider(
label="Guidancescale",
minimum=0.1,
maximum=2,
step=0.1,
value=0.5,
)
inp_seed=gr.Slider(label="Seed",value=42,maximum=MAX_SEED)
inp_steps=gr.Slider(label="Steps",value=4,minimum=1,maximum=50,step=1)
withgr.Column(visible=True)asstep2:
out_result=gr.Image(label="Result(Original)")
withgr.Column(visible=quantized_model_present)asquantization_step:
int_result=gr.Image(label="Result(Quantized)")
examples=gr.Examples([["example.png","aheadfullofroses"]],[inp_img,inp_prompt])

defextract_normal_map(img):
ifimgisNone:
raisegr.Error("Pleaseuploadtheimageoruseonefromtheexampleslist")
returnprocessor(img)

defgenerate(img,prompt,negative_prompt,seed,num_steps,guidance_scale):
torch.manual_seed(seed)
control_img=extract_normal_map(img)

result=ov_pipe(
prompt,
control_img,
num_steps,
guidance_scale=guidance_scale,
negative_prompt=negative_prompt,
)[0]
ifint8_pipeisnotNone:
torch.manual_seed(seed)
int8_result=int8_pipe(
prompt,
control_img,
num_steps,
guidance_scale=guidance_scale,
negative_prompt=negative_prompt,
)[0]
returncontrol_img,result,int8_result
returncontrol_img,result

output_images=[out_normal,out_result]
ifquantized_model_present:
output_images.append(int_result)
btn.click(
generate,
[inp_img,inp_prompt,inp_neg_prompt,inp_seed,inp_steps,guidance_scale],
output_images,
)


try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False,height=800)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
