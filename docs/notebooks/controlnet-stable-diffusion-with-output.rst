Text-to-ImageGenerationwithControlNetConditioning
=====================================================

DiffusionmodelsmakearevolutioninAI-generatedart.Thistechnology
enablescreationofhigh-qualityimagessimplybywritingatextprompt.
Eventhoughthistechnologygivesverypromisingresults,thediffusion
process,inthefirstorder,istheprocessofgeneratingimagesfrom
randomnoiseandtextconditions,whichdonotalwaysclarifyhow
desiredcontentshouldlook,whichformsitshouldhaveandwhereitis
locatedinrelationtootherobjectsontheimage.Researchershavebeen
lookingforwaystohavemorecontrolovertheresultsofthegeneration
process.ControlNetprovidesaminimalinterfaceallowingusersto
customizethegenerationprocesstoagreatextent.

ControlNetwasintroducedin`AddingConditionalControlto
Text-to-ImageDiffusionModels<https://arxiv.org/abs/2302.05543>`__
paper.Itprovidesaframeworkthatenablessupportforvariousspatial
contextssuchasadepthmap,asegmentationmap,ascribble,andkey
pointsthatcanserveasadditionalconditioningstoDiffusionmodels
suchasStableDiffusion.

ThisnotebookexploresControlNetindepth,especiallyanewtechnique
forimpartinghighlevelsofcontrolovertheshapeofsynthesized
images.Itdemonstrateshowtorunit,usingOpenVINO.Anadditional
partdemonstrateshowtorunquantizationwith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__tospeedup
pipeline.Letusget“controlling”!

Background
----------

StableDiffusion
~~~~~~~~~~~~~~~~

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

ControlNetisaneuralnetworkstructuretocontroldiffusionmodelsby
addingextraconditions.Usingthisnewframework,wecancapturea
scene,structure,object,orsubjectposefromaninputtedimage,and
thentransferthatqualitytothegenerationprocess.Inpractice,this
enablesthemodeltocompletelyretaintheoriginalinputshape,and
createanovelimagethatconservestheshape,pose,oroutlinewhile
usingthenovelfeaturesfromtheinputtedprompt.

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

TrainingControlNetconsistsofthefollowingsteps:

1.Cloningthepre-trainedparametersofaDiffusionmodel,suchas
StableDiffusion’slatentUNet,(referredtoas“trainablecopy”)
whilealsomaintainingthepre-trainedparametersseparately(”locked
copy”).Itisdonesothatthelockedparametercopycanpreservethe
vastknowledgelearnedfromalargedataset,whereasthetrainable
copyisemployedtolearntask-specificaspects.
2.Thetrainableandlockedcopiesoftheparametersareconnectedvia
“zeroconvolution”layers(seehereformoreinformation)whichare
optimizedasapartoftheControlNetframework.Thisisatraining
tricktopreservethesemanticsalreadylearnedbyafrozenmodelas
thenewconditionsaretrained.

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

Thistutorialfocusesmainlyonconditioningbypose.However,the
discussedstepsarealsoapplicabletootherannotationmodes.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`InstantiatingGeneration
Pipeline<#instantiating-generation-pipeline>`__

-`ControlNetinDiffusers
library<#controlnet-in-diffusers-library>`__
-`OpenPose<#openpose>`__

-`ConvertmodelstoOpenVINOIntermediaterepresentation(IR)
format<#convert-models-to-openvino-intermediate-representation-ir-format>`__

-`OpenPoseconversion<#openpose-conversion>`__

-`Selectinferencedevice<#select-inference-device>`__

-`ControlNetconversion<#controlnet-conversion>`__
-`UNetconversion<#unet-conversion>`__
-`TextEncoder<#text-encoder>`__
-`VAEDecoderconversion<#vae-decoder-conversion>`__

-`PrepareInferencepipeline<#prepare-inference-pipeline>`__
-`RunningText-to-ImageGenerationwithControlNetConditioningand
OpenVINO<#running-text-to-image-generation-with-controlnet-conditioning-and-openvino>`__
-`SelectinferencedeviceforStableDiffusion
pipeline<#select-inference-device-for-stable-diffusion-pipeline>`__
-`Quantization<#quantization>`__

-`Preparecalibrationdatasets<#prepare-calibration-datasets>`__
-`Runquantization<#run-quantization>`__
-`Comparemodelfilesizes<#compare-model-file-sizes>`__
-`CompareinferencetimeoftheFP16andINT8
pipelines<#compare-inference-time-of-the-fp16-and-int8-pipelines>`__

-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"torch>=2.1""torchvision"
%pipinstall-q"diffusers>=0.14.0""transformers>=4.30.2""controlnet-aux>=0.0.6""gradio>=3.36"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2023.1.0""datasets>=2.14.6""nncf>=2.7.0"

InstantiatingGenerationPipeline
---------------------------------

`backtotop⬆️<#table-of-contents>`__

ControlNetinDiffuserslibrary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
``StableDiffusionControlNetPipeline``,usingthe``controlnet-openpose``
controlnetmodeland``stable-diffusion-v1-5``:

..code::ipython3

importtorch
fromdiffusersimportStableDiffusionControlNetPipeline,ControlNetModel

controlnet=ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose",torch_dtype=torch.float32)
pipe=StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",controlnet=controlnet)

OpenPose
~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

AnnotationisanimportantpartofworkingwithControlNet.
`OpenPose<https://github.com/CMU-Perceptual-Computing-Lab/openpose>`__
isafastkeypointdetectionmodelthatcanextracthumanposeslike
positionsofhands,legs,andhead.BelowistheControlNetworkflow
usingOpenPose.Keypointsareextractedfromtheinputimageusing
OpenPoseandsavedasacontrolmapcontainingthepositionsof
keypoints.ItisthenfedtoStableDiffusionasanextraconditioning
togetherwiththetextprompt.Imagesaregeneratedbasedonthesetwo
conditionings.

..figure::https://user-images.githubusercontent.com/29454499/224248986-eedf6492-dd7a-402b-b65d-36de952094ec.png
:alt:controlnet-openpose-pipe

controlnet-openpose-pipe

ThecodebelowdemonstrateshowtoinstantiatetheOpenPosemodel.

..code::ipython3

fromcontrolnet_auximportOpenposeDetector

pose_estimator=OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

Now,letuscheckitsresultonexampleimage:

..code::ipython3

importrequests
fromPILimportImage
importmatplotlib.pyplotasplt
importnumpyasnp


example_url="https://user-images.githubusercontent.com/29454499/224540208-c172c92a-9714-4a7b-857a-b1e54b4d4791.jpg"
img=Image.open(requests.get(example_url,stream=True).raw)
pose=pose_estimator(img)


defvisualize_pose_results(
orig_img:Image.Image,
skeleton_img:Image.Image,
left_title:str="Originalimage",
right_title:str="Pose",
):
"""
Helperfunctionforposeestimationresultsvisualization

Parameters:
orig_img(Image.Image):originalimage
skeleton_img(Image.Image):processedimagewithbodykeypoints
left_title(str):titlefortheleftimage
right_title(str):titlefortherightimage
Returns:
fig(matplotlib.pyplot.Figure):matplotlibgeneratedfigurecontainsdrawingresult
"""
orig_img=orig_img.resize(skeleton_img.size)
im_w,im_h=orig_img.size
is_horizontal=im_h<=im_w
figsize=(20,10)ifis_horizontalelse(10,20)
fig,axs=plt.subplots(
2ifis_horizontalelse1,
1ifis_horizontalelse2,
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
list_axes[1].imshow(np.array(skeleton_img))
list_axes[0].set_title(left_title,fontsize=15)
list_axes[1].set_title(right_title,fontsize=15)
fig.subplots_adjust(wspace=0.01ifis_horizontalelse0.00,hspace=0.01ifis_horizontalelse0.1)
fig.tight_layout()
returnfig


fig=visualize_pose_results(img,pose)



..image::controlnet-stable-diffusion-with-output_files/controlnet-stable-diffusion-with-output_8_0.png


ConvertmodelstoOpenVINOIntermediaterepresentation(IR)format
------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Startingfrom2023.0release,OpenVINOsupportsPyTorchmodels
conversiondirectly.Weneedtoprovideamodelobject,inputdatafor
modeltracingto``ov.convert_model``functiontoobtainOpenVINO
``ov.Model``objectinstance.Modelcanbesavedondiskfornext
deploymentusing``ov.save_model``function.

Thepipelineconsistsoffiveimportantparts:

-OpenPoseforobtainingannotationbasedonanestimatedpose.
-ControlNetforconditioningbyimageannotation.
-TextEncoderforcreationconditiontogenerateanimagefromatext
prompt.
-Unetforstep-by-stepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)fordecodinglatentspacetoimage.

Letusconverteachpart:

OpenPoseconversion
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OpenPosemodelisrepresentedinthepipelineasawrapperonthe
PyTorchmodelwhichnotonlydetectsposesonaninputimagebutisalso
responsiblefordrawingposemaps.Weneedtoconvertonlythepose
estimationpart,whichislocatedinsidethewrapper
``pose_estimator.body_estimation.model``.

..code::ipython3

frompathlibimportPath
importtorch
importopenvinoasov

OPENPOSE_OV_PATH=Path("openpose.xml")


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


ifnotOPENPOSE_OV_PATH.exists():
withtorch.no_grad():
ov_model=ov.convert_model(
pose_estimator.body_estimation.model,
example_input=torch.zeros([1,3,184,136]),
input=[[1,3,184,136]],
)
ov.save_model(ov_model,OPENPOSE_OV_PATH)
delov_model
cleanup_torchscript_cache()
print("OpenPosesuccessfullyconvertedtoIR")
else:
print(f"OpenPosewillbeloadedfrom{OPENPOSE_OV_PATH}")


..parsed-literal::

OpenPosewillbeloadedfromopenpose.xml


Toreusetheoriginaldrawingprocedure,wereplacethePyTorchOpenPose
modelwiththeOpenVINOmodel,usingthefollowingcode:

..code::ipython3

fromcollectionsimportnamedtuple


classOpenPoseOVModel:
"""HelperwrapperforOpenPosemodelinference"""

def__init__(self,core,model_path,device="AUTO"):
self.core=core
self.model=core.read_model(model_path)
self.compiled_model=core.compile_model(self.model,device)

def__call__(self,input_tensor:torch.Tensor):
"""
inferencestep

Parameters:
input_tensor(torch.Tensor):tensorwithprerpcessedinputimage
Returns:
predictedkeypointsheatmaps
"""
h,w=input_tensor.shape[2:]
input_shape=self.model.input(0).shape
ifh!=input_shape[2]orw!=input_shape[3]:
self.reshape_model(h,w)
results=self.compiled_model(input_tensor)
returntorch.from_numpy(results[self.compiled_model.output(0)]),torch.from_numpy(results[self.compiled_model.output(1)])

defreshape_model(self,height:int,width:int):
"""
helpermethodforreshapingmodeltofitinputdata

Parameters:
height(int):inputtensorheight
width(int):inputtensorwidth
Returns:
None
"""
self.model.reshape({0:[1,3,height,width]})
self.compiled_model=self.core.compile_model(self.model)

defparameters(self):
Device=namedtuple("Device",["device"])
return[Device(torch.device("cpu"))]


core=ov.Core()

Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importipywidgetsaswidgets

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

ov_openpose=OpenPoseOVModel(core,OPENPOSE_OV_PATH,device=device.value)
pose_estimator.body_estimation.model=ov_openpose

..code::ipython3

pose=pose_estimator(img)
fig=visualize_pose_results(img,pose)



..image::controlnet-stable-diffusion-with-output_files/controlnet-stable-diffusion-with-output_17_0.png


Great!Aswecansee,itworksperfectly.

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

importgc
fromfunctoolsimportpartial

inputs={
"sample":torch.randn((2,4,64,64)),
"timestep":torch.tensor(1),
"encoder_hidden_states":torch.randn((2,77,768)),
"controlnet_cond":torch.randn((2,3,512,512)),
}

input_info=[(name,ov.PartialShape(inp.shape))forname,inpininputs.items()]

CONTROLNET_OV_PATH=Path("controlnet-pose.xml")
controlnet.eval()
withtorch.no_grad():
down_block_res_samples,mid_block_res_sample=controlnet(**inputs,return_dict=False)

ifnotCONTROLNET_OV_PATH.exists():
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

ControlNetwillbeloadedfromcontrolnet-pose.xml




..parsed-literal::

4890



UNetconversion
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheprocessofUNetmodelconversionremainsthesame,likefororiginal
StableDiffusionmodel,butwithrespecttothenewinputsgeneratedby
ControlNet.

..code::ipython3

fromtypingimportTuple

UNET_OV_PATH=Path("unet_controlnet.xml")

dtype_mapping={
torch.float32:ov.Type.f32,
torch.float64:ov.Type.f64,
torch.int32:ov.Type.i32,
torch.int64:ov.Type.i64,
}


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


defflattenize_inputs(inputs):
flatten_inputs=[]
forinput_dataininputs:
ifinput_dataisNone:
continue
ifisinstance(input_data,(list,tuple)):
flatten_inputs.extend(flattenize_inputs(input_data))
else:
flatten_inputs.append(input_data)
returnflatten_inputs


ifnotUNET_OV_PATH.exists():
inputs.pop("controlnet_cond",None)
inputs["down_block_additional_residuals"]=down_block_res_samples
inputs["mid_block_additional_residual"]=mid_block_res_sample

unet=UnetWrapper(pipe.unet)
unet.eval()

withtorch.no_grad():
ov_model=ov.convert_model(unet,example_input=inputs)

flatten_inputs=flattenize_inputs(inputs.values())
forinput_data,input_tensorinzip(flatten_inputs,ov_model.inputs):
input_tensor.get_node().set_partial_shape(ov.PartialShape(input_data.shape))
input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])
ov_model.validate_nodes_and_infer_types()
ov.save_model(ov_model,UNET_OV_PATH)
delov_model
cleanup_torchscript_cache()
delunet
delpipe.unet
gc.collect()
print("UnetsuccessfullyconvertedtoIR")
else:
delpipe.unet
print(f"Unetwillbeloadedfrom{UNET_OV_PATH}")
gc.collect()


..parsed-literal::

Unetwillbeloadedfromunet_controlnet.xml




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

TEXT_ENCODER_OV_PATH=Path("text_encoder.xml")


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
convert_encoder(pipe.text_encoder,TEXT_ENCODER_OV_PATH)
else:
print(f"Textencoderwillbeloadedfrom{TEXT_ENCODER_OV_PATH}")
delpipe.text_encoder
gc.collect()


..parsed-literal::

Textencoderwillbeloadedfromtext_encoder.xml




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

VAE_DECODER_OV_PATH=Path("vae_decoder.xml")


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
ov_model=ov.convert_model(
vae_decoder,
example_input=latents,
input=[
(1,4,64,64),
],
)
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
print("VAEdecodersuccessfullyconvertedtoIR")


ifnotVAE_DECODER_OV_PATH.exists():
convert_vae_decoder(pipe.vae,VAE_DECODER_OV_PATH)
else:
print(f"VAEdecoderwillbeloadedfrom{VAE_DECODER_OV_PATH}")


..parsed-literal::

VAEdecoderwillbeloadedfromvae_decoder.xml


PrepareInferencepipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

Puttingitalltogether,letusnowtakeacloserlookathowthemodel
worksininferencebyillustratingthelogicalflow.|detailedworkflow|

Thestablediffusionmodeltakesbothalatentseedandatextpromptas
input.Thelatentseedisthenusedtogeneraterandomlatentimage
representationsofsize:math:`64\times64`whereasthetextpromptis
transformedtotextembeddingsofsize:math:`77\times768`viaCLIP’s
textencoder.

Next,theU-Netiteratively*denoises*therandomlatentimage
representationswhilebeingconditionedonthetextembeddings.In
comparisonwiththeoriginalstable-diffusionpipeline,latentimage
representation,encoderhiddenstates,andcontrolconditionannotation
passedviaControlNetoneachdenoisingstepforobtainingmiddleand
downblocksattentionparameters,theseattentionblocksresults
additionallywillbeprovidedtotheUNetmodelforthecontrol
generationprocess.TheoutputoftheU-Net,beingthenoiseresidual,
isusedtocomputeadenoisedlatentimagerepresentationviaa
scheduleralgorithm.Manydifferentscheduleralgorithmscanbeusedfor
thiscomputation,eachhavingitsprosandcons.ForStableDiffusion,
itisrecommendedtouseoneof:

-`PNDM
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py>`__
-`DDIM
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py>`__
-`K-LMS
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py>`__

Theoryonhowthescheduleralgorithmfunctionworksisoutofscopefor
thisnotebook,butinshort,youshouldrememberthattheycomputethe
predicteddenoisedimagerepresentationfromthepreviousnoise
representationandthepredictednoiseresidual.Formoreinformation,
itisrecommendedtolookinto`ElucidatingtheDesignSpaceof
Diffusion-BasedGenerativeModels<https://arxiv.org/abs/2206.00364>`__

Inthistutorial,insteadofusingStableDiffusion’sdefault
`PNDMScheduler<https://huggingface.co/docs/diffusers/main/en/api/schedulers/pndm>`__,
weuseoneofthecurrentlyfastestdiffusionmodelschedulers,called
`UniPCMultistepScheduler<https://huggingface.co/docs/diffusers/main/en/api/schedulers/unipc>`__.
Choosinganimprovedschedulercandrasticallyreduceinferencetime-
inthiscase,wecanreducethenumberofinferencestepsfrom50to20
whilemoreorlesskeepingthesameimagegenerationquality.More
informationregardingschedulerscanbefound
`here<https://huggingface.co/docs/diffusers/main/en/using-diffusers/schedulers>`__.

The*denoising*processisrepeatedagivennumberoftimes(bydefault
50)tostep-by-stepretrievebetterlatentimagerepresentations.Once
complete,thelatentimagerepresentationisdecodedbythedecoderpart
ofthevariationalauto-encoder.

SimilarlytoDiffusers``StableDiffusionControlNetPipeline``,wedefine
ourown``OVContrlNetStableDiffusionPipeline``inferencepipelinebased
onOpenVINO.

..|detailedworkflow|image::https://user-images.githubusercontent.com/29454499/224261720-2d20ca42-f139-47b7-b8b9-0b9f30e1ae1e.png

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


defpreprocess(image:Image.Image):
"""
Imagepreprocessingfunction.TakesimageinPIL.Imageformat,resizesittokeepaspectrationandfitstomodelinputwindow512x512,
thenconvertsittonp.ndarrayandaddspaddingwithzerosonrightorbottomsideofimage(dependsfromaspectratio),afterthat
convertsdatatofloat32datatypeandchangerangeofvaluesfrom[0,255]to[-1,1],finally,convertsdatalayoutfromplanarNHWCtoNCHW.
Thefunctionreturnspreprocessedinputtensorandpaddingsize,whichcanbeusedinpostprocessing.

Parameters:
image(Image.Image):inputimage
Returns:
image(np.ndarray):preprocessedimagetensor
pad(Tuple[int]):padingsizeforeachdimensionforrestoringimagesizeinpostprocessing
"""
src_width,src_height=image.size
dst_width,dst_height=scale_fit_to_window(512,512,src_width,src_height)
image=np.array(image.resize((dst_width,dst_height),resample=Image.Resampling.LANCZOS))[None,:]
pad_width=512-dst_width
pad_height=512-dst_height
pad=((0,0),(0,pad_height),(0,pad_width),(0,0))
image=np.pad(image,pad,mode="constant")
image=image.astype(np.float32)/255.0
image=image.transpose(0,3,1,2)
returnimage,pad


defrandn_tensor(
shape:Union[Tuple,List],
dtype:Optional[np.dtype]=np.float32,
):
"""
Helperfunctionforgenerationrandomvaluestensorwithgivenshapeanddatatype

Parameters:
shape(Union[Tuple,List]):shapeforfillingrandomvalues
dtype(np.dtype,*optiona*,np.float32):datatypeforresult
Returns:
latents(np.ndarray):tensorwithrandomvalueswithgivendatatypeandshape(usuallyrepresentsnoiseinlatentspace)
"""
latents=np.random.randn(*shape).astype(dtype)

returnlatents


classOVContrlNetStableDiffusionPipeline(DiffusionPipeline):
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
self.set_progress_bar_config(disable=True)

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
self.text_encoder_out=self.text_encoder.output(0)
self.register_to_config(controlnet=core.compile_model(controlnet,device))
self.register_to_config(unet=core.compile_model(unet,device))
self.unet_out=self.unet.output(0)
self.vae_decoder=core.compile_model(vae_decoder)
self.vae_decoder_out=self.vae_decoder.output(0)

def__call__(
self,
prompt:Union[str,List[str]],
image:Image.Image,
num_inference_steps:int=10,
negative_prompt:Union[str,List[str]]=None,
guidance_scale:float=7.5,
controlnet_conditioning_scale:float=1.0,
eta:float=0.0,
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
negative_prompt(`str`or`List[str]`):
negativepromptorpromptsforgeneration
guidance_scale(`float`,*optional*,defaultsto7.5):
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
#here`guidance_scale`isdefinedanalogtotheguidanceweight`w`ofequation(2)
#oftheImagenpaper:https://arxiv.org/pdf/2205.11487.pdf.`guidance_scale=1`
#correspondstodoingnoclassifierfreeguidance.
do_classifier_free_guidance=guidance_scale>1.0
#2.Encodeinputprompt
text_embeddings=self._encode_prompt(prompt,negative_prompt=negative_prompt)

#3.Preprocessimage
orig_width,orig_height=image.size
image,pad=preprocess(image)
height,width=image.shape[-2:]
ifdo_classifier_free_guidance:
image=np.concatenate(([image]*2))

#4.settimesteps
self.scheduler.set_timesteps(num_inference_steps)
timesteps=self.scheduler.timesteps

#6.Preparelatentvariables
num_channels_latents=4
latents=self.prepare_latents(
batch_size,
num_channels_latents,
height,
width,
text_embeddings.dtype,
latents,
)

#7.Denoisingloop
num_warmup_steps=len(timesteps)-num_inference_steps*self.scheduler.order
withself.progress_bar(total=num_inference_steps)asprogress_bar:
fori,tinenumerate(timesteps):
#Expandthelatentsifwearedoingclassifierfreeguidance.
#Thelatentsareexpanded3timesbecauseforpix2pixtheguidance\
#isappliedforboththetextandtheinputimage.
latent_model_input=np.concatenate([latents]*2)ifdo_classifier_free_guidanceelselatents
latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)

result=self.controlnet([latent_model_input,t,text_embeddings,image])
down_and_mid_blok_samples=[sample*controlnet_conditioning_scalefor_,sampleinresult.items()]

#predictthenoiseresidual
noise_pred=self.unet([latent_model_input,t,text_embeddings,*down_and_mid_blok_samples])[self.unet_out]

#performguidance
ifdo_classifier_free_guidance:
noise_pred_uncond,noise_pred_text=noise_pred[0],noise_pred[1]
noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

#computethepreviousnoisysamplex_t->x_t-1
latents=self.scheduler.step(torch.from_numpy(noise_pred),t,torch.from_numpy(latents)).prev_sample.numpy()

#updateprogress
ifi==len(timesteps)-1or((i+1)>num_warmup_stepsand(i+1)%self.scheduler.order==0):
progress_bar.update()

#8.Post-processing
image=self.decode_latents(latents,pad)

#9.ConverttoPIL
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

text_embeddings=self.text_encoder(text_input_ids)[self.text_encoder_out]

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

uncond_embeddings=self.text_encoder(uncond_input.input_ids)[self.text_encoder_out]

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
dtype:np.dtype=np.float32,
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
image=self.vae_decoder(latents)[self.vae_decoder_out]
(_,end_h),(_,end_w)=pad[1:3]
h,w=image.shape[2:]
unpad_h=h-end_h
unpad_w=w-end_w
image=image[:,:,:unpad_h,:unpad_w]
image=np.clip(image/2+0.5,0,1)
image=np.transpose(image,(0,2,3,1))
returnimage

..code::ipython3

fromtransformersimportCLIPTokenizer
fromdiffusersimportUniPCMultistepScheduler

tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config)


defvisualize_results(orig_img:Image.Image,skeleton_img:Image.Image,result_img:Image.Image):
"""
Helperfunctionforresultsvisualization

Parameters:
orig_img(Image.Image):originalimage
skeleton_img(Image.Image):imagewithbodyposekeypoints
result_img(Image.Image):generatedimage
Returns:
fig(matplotlib.pyplot.Figure):matplotlibgeneratedfigurecontainsdrawingresult
"""
orig_title="Originalimage"
skeleton_title="Pose"
orig_img=orig_img.resize(result_img.size)
im_w,im_h=orig_img.size
is_horizontal=im_h<=im_w
figsize=(20,20)
fig,axs=plt.subplots(
3ifis_horizontalelse1,
1ifis_horizontalelse3,
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
list_axes[1].imshow(np.array(skeleton_img))
list_axes[2].imshow(np.array(result_img))
list_axes[0].set_title(orig_title,fontsize=15)
list_axes[1].set_title(skeleton_title,fontsize=15)
list_axes[2].set_title("Result",fontsize=15)
fig.subplots_adjust(wspace=0.01ifis_horizontalelse0.00,hspace=0.01ifis_horizontalelse0.1)
fig.tight_layout()
fig.savefig("result.png",bbox_inches="tight")
returnfig

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

SelectinferencedeviceforStableDiffusionpipeline
-----------------------------------------------------

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

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



..code::ipython3

ov_pipe=OVContrlNetStableDiffusionPipeline(
tokenizer,
scheduler,
core,
CONTROLNET_OV_PATH,
TEXT_ENCODER_OV_PATH,
UNET_OV_PATH,
VAE_DECODER_OV_PATH,
device=device.value,
)

..code::ipython3

np.random.seed(42)

pose=pose_estimator(img)

prompt="DancingDarthVader,bestquality,extremelydetailed"
negative_prompt="monochrome,lowres,badanatomy,worstquality,lowquality"
result=ov_pipe(prompt,pose,20,negative_prompt=negative_prompt)

result[0]


..parsed-literal::

/home/ltalamanova/tmp_venv/lib/python3.11/site-packages/diffusers/configuration_utils.py:139:FutureWarning:Accessingconfigattribute`controlnet`directlyvia'OVContrlNetStableDiffusionPipeline'objectattributeisdeprecated.Pleaseaccess'controlnet'over'OVContrlNetStableDiffusionPipeline'sconfigobjectinstead,e.g.'scheduler.config.controlnet'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)




..image::controlnet-stable-diffusion-with-output_files/controlnet-stable-diffusion-with-output_34_1.png



Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``OVContrlNetStableDiffusionPipeline``structure,
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

to_quantize=widgets.Checkbox(value=True,description="Quantization")

to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



Let’sload``skipmagic``extensiontoskipquantizationif
``to_quantize``isnotselected

..code::ipython3

#Fetch`skip_kernel_extension`module
importrequests

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
`jschoormans/humanpose_densepose<https://huggingface.co/datasets/jschoormans/humanpose_densepose>`__
datasetfromHuggingFaceascalibrationdata.Weuseapromptsbelowas
negativepromptsforControlNetandUNet.Tocollectintermediatemodel
inputsforcalibrationweshouldcustomize``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

negative_prompts=[
"blurryunrealoccluded",
"lowcontrastdisfigureduncenteredmangled",
"amateuroutofframelowqualitynsfw",
"uglyunderexposedjpegartifacts",
"lowsaturationdisturbingcontent",
"overexposedseveredistortion",
"amateurNSFW",
"uglymutilatedoutofframedisfigured",
]

..code::ipython3

%%skipnot$to_quantize.value

importdatasets

num_inference_steps=20
subset_size=200

dataset=datasets.load_dataset("jschoormans/humanpose_densepose",split="train",streaming=True).shuffle(seed=42)
input_data=[]
forbatchindataset:
caption=batch["caption"]
iflen(caption)>tokenizer.model_max_length:
continue
img=batch["file_name"]
input_data.append((caption,pose_estimator(img)))
iflen(input_data)>=subset_size//num_inference_steps:
break

..code::ipython3

%%skipnot$to_quantize.value

importdatasets
fromtqdm.notebookimporttqdm
fromtransformersimportset_seed
fromtypingimportAny,Dict,List

set_seed(42)

classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model:ov.CompiledModel,keep_prob:float=1.0):
super().__init__(compiled_model)
self.data_cache=[]
self.keep_prob=np.clip(keep_prob,0,1)

def__call__(self,*args,**kwargs):
ifnp.random.rand()<=self.keep_prob:
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)

defcollect_calibration_data(pipeline:OVContrlNetStableDiffusionPipeline,subset_size:int)->List[Dict]:
original_unet=pipeline.unet
pipeline.unet=CompiledModelDecorator(original_unet)
pipeline.set_progress_bar_config(disable=True)

pbar=tqdm(total=subset_size)
forprompt,poseininput_data:
img=batch["file_name"]
negative_prompt=np.random.choice(negative_prompts)
_=pipeline(prompt,pose,num_inference_steps,negative_prompt=negative_prompt)
collected_subset_size=len(pipeline.unet.data_cache)
pbar.update(collected_subset_size-pbar.n)
ifcollected_subset_size>=subset_size:
break

calibration_dataset=pipeline.unet.data_cache[:subset_size]
pipeline.set_progress_bar_config(disable=False)
pipeline.unet=original_unet
returncalibration_dataset

..code::ipython3

%%skipnot$to_quantize.value

CONTROLNET_INT8_OV_PATH=Path("controlnet-pose_int8.xml")
UNET_INT8_OV_PATH=Path("unet_controlnet_int8.xml")

ifnot(CONTROLNET_INT8_OV_PATH.exists()andUNET_INT8_OV_PATH.exists()):
unet_calibration_data=collect_calibration_data(ov_pipe,subset_size=subset_size)



..parsed-literal::

0%||0/200[00:00<?,?it/s]


..code::ipython3

%%skipnot$to_quantize.value

ifnotCONTROLNET_INT8_OV_PATH.exists():
control_calibration_data=[]
prev_idx=0
for_,pose_imgininput_data:
preprocessed_image,_=preprocess(pose_img)
preprocessed_image=np.concatenate(([preprocessed_image]*2))
foriinrange(prev_idx,prev_idx+num_inference_steps):
control_calibration_data.append(unet_calibration_data[i][:3]+[preprocessed_image])
prev_idx+=num_inference_steps

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
subset_size=subset_size,
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
subset_size=subset_size,
model_type=nncf.ModelType.TRANSFORMER,
advanced_parameters=nncf.AdvancedQuantizationParameters(
disable_bias_correction=True
)
)
ov.save_model(quantized_controlnet,CONTROLNET_INT8_OV_PATH)

Let’scomparetheimagesgeneratedbytheoriginalandoptimized
pipelines.

..code::ipython3

%%skipnot$to_quantize.value

int8_pipe=OVContrlNetStableDiffusionPipeline(
tokenizer,
scheduler,
core,
CONTROLNET_INT8_OV_PATH,
TEXT_ENCODER_OV_PATH,
UNET_INT8_OV_PATH,
VAE_DECODER_OV_PATH,
device=device.value
)

..code::ipython3

%%skipnot$to_quantize.value

np.random.seed(42)
int8_image=int8_pipe(prompt,pose,20,negative_prompt=negative_prompt)[0]
fig=visualize_pose_results(result[0],int8_image,left_title="FP16pipeline",right_title="INT8pipeline")



..image::controlnet-stable-diffusion-with-output_files/controlnet-stable-diffusion-with-output_50_0.png


Comparemodelfilesizes
~~~~~~~~~~~~~~~~~~~~~~~~

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


CompareinferencetimeoftheFP16andINT8pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tomeasuretheinferenceperformanceofthe``FP16``and``INT8``
pipelines,weusemeaninferencetimeon3samples.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.

..code::ipython3

%%skipnot$to_quantize.value

importtime

defcalculate_inference_time(pipeline):
inference_time=[]
pipeline.set_progress_bar_config(disable=True)
foriinrange(3):
prompt,pose=input_data[i]
negative_prompt=np.random.choice(negative_prompts)
start=time.perf_counter()
_=pipeline(prompt,pose,num_inference_steps=num_inference_steps,negative_prompt=negative_prompt)
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
pipeline.set_progress_bar_config(disable=False)
returnnp.mean(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

fp_latency=calculate_inference_time(ov_pipe)
print(f"FP16pipeline:{fp_latency:.3f}seconds")
int8_latency=calculate_inference_time(int8_pipe)
print(f"INT8pipeline:{int8_latency:.3f}seconds")
print(f"Performancespeed-up:{fp_latency/int8_latency:.3f}")


..parsed-literal::

FP16pipeline:31.296seconds


..parsed-literal::

/home/ltalamanova/tmp_venv/lib/python3.11/site-packages/diffusers/configuration_utils.py:139:FutureWarning:Accessingconfigattribute`unet`directlyvia'OVContrlNetStableDiffusionPipeline'objectattributeisdeprecated.Pleaseaccess'unet'over'OVContrlNetStableDiffusionPipeline'sconfigobjectinstead,e.g.'scheduler.config.unet'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)


..parsed-literal::

INT8pipeline:24.183seconds
Performancespeed-up:1.294


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

Pleaseselectbelowwhetheryouwouldliketousethequantizedmodelto
launchtheinteractivedemo.

..code::ipython3

quantized_model_present=int8_pipeisnotNone

use_quantized_model=widgets.Checkbox(
value=Trueifquantized_model_presentelseFalse,
description="Usequantizedmodel",
disabled=notquantized_model_present,
)

use_quantized_model

..code::ipython3

importgradioasgr

pipeline=int8_pipeifuse_quantized_model.valueelseov_pipe

r=requests.get(example_url)

img_path=Path("example.jpg")

withimg_path.open("wb")asf:
f.write(r.content)

gr.close_all()
withgr.Blocks()asdemo:
withgr.Row():
withgr.Column():
inp_img=gr.Image(label="Inputimage")
pose_btn=gr.Button("Extractpose")
examples=gr.Examples(["example.jpg"],inp_img)
withgr.Column(visible=False)asstep1:
out_pose=gr.Image(label="Estimatedpose",type="pil")
inp_prompt=gr.Textbox("DancingDarthVader,bestquality,extremelydetailed",label="Prompt")
inp_neg_prompt=gr.Textbox(
"monochrome,lowres,badanatomy,worstquality,lowquality",
label="Negativeprompt",
)
inp_seed=gr.Slider(label="Seed",value=42,maximum=1024000000)
inp_steps=gr.Slider(label="Steps",value=20,minimum=1,maximum=50)
btn=gr.Button()
withgr.Column(visible=False)asstep2:
out_result=gr.Image(label="Result")

defextract_pose(img):
ifimgisNone:
raisegr.Error("Pleaseuploadtheimageoruseonefromtheexampleslist")
return{
step1:gr.update(visible=True),
step2:gr.update(visible=True),
out_pose:pose_estimator(img),
}

defgenerate(
pose,
prompt,
negative_prompt,
seed,
num_steps,
progress=gr.Progress(track_tqdm=True),
):
np.random.seed(seed)
result=pipeline(prompt,pose,num_steps,negative_prompt)[0]
returnresult

pose_btn.click(extract_pose,inp_img,[out_pose,step1,step2])
btn.click(
generate,
[out_pose,inp_prompt,inp_neg_prompt,inp_seed,inp_steps],
out_result,
)


try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
