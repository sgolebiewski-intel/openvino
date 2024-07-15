GeneratecreativeQRcodeswithControlNetQRCodeMonsterandOpenVINO™
========================================================================

`StableDiffusion<https://github.com/CompVis/stable-diffusion>`__,a
cutting-edgeimagegenerationtechnique,butitcanbefurtherenhanced
bycombiningitwith`ControlNet<https://arxiv.org/abs/2302.05543>`__,
awidelyusedcontrolnetworkapproach.ThecombinationallowsStable
Diffusiontouseaconditioninputtoguidetheimagegeneration
process,resultinginhighlyaccurateandvisuallyappealingimages.The
conditioninputcouldbeintheformofvarioustypesofdatasuchas
scribbles,edgemaps,posekeypoints,depthmaps,segmentationmaps,
normalmaps,oranyotherrelevantinformationthathelpstoguidethe
contentofthegeneratedimage,forexample-QRcodes!Thismethodcan
beparticularlyusefulincompleximagegenerationscenarioswhere
precisecontrolandfine-tuningarerequiredtoachievethedesired
results.

Inthistutorial,wewilllearnhowtoconvertandrun`ControlnetQR
CodeMonsterFor
SD-1.5<https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster>`__
by`monster-labs<https://qrcodemonster.art/>`__.Anadditionalpart
demonstrateshowtorunquantizationwith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__tospeedup
pipeline.

|image0|

IfyouwanttolearnmoreaboutControlNetandparticularlyon
conditioningbypose,pleaserefertothis
`tutorial<controlnet-stable-diffusion-with-output.html>`__

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`InstantiatingGeneration
Pipeline<#instantiating-generation-pipeline>`__

-`ControlNetinDiffusers
library<#controlnet-in-diffusers-library>`__

-`ConvertmodelstoOpenVINOIntermediaterepresentation(IR)
format<#convert-models-to-openvino-intermediate-representation-ir-format>`__

-`ControlNetconversion<#controlnet-conversion>`__
-`TextEncoder<#text-encoder>`__
-`UNetconversion<#unet-conversion>`__
-`VAEDecoderconversion<#vae-decoder-conversion>`__

-`SelectinferencedeviceforStableDiffusion
pipeline<#select-inference-device-for-stable-diffusion-pipeline>`__
-`PrepareInferencepipeline<#prepare-inference-pipeline>`__
-`Quantization<#quantization>`__

-`Preparecalibrationdatasets<#prepare-calibration-datasets>`__
-`Runquantization<#run-quantization>`__
-`Comparemodelfilesizes<#compare-model-file-sizes>`__
-`CompareinferencetimeoftheFP16andINT8
pipelines<#compare-inference-time-of-the-fp16-and-int8-pipelines>`__

-`RunningText-to-ImageGenerationwithControlNetConditioningand
OpenVINO<#running-text-to-image-generation-with-controlnet-conditioning-and-openvino>`__

..|image0|image::https://github.com/openvinotoolkit/openvino_notebooks/assets/76463150/1a5978c6-e7a0-4824-9318-a3d8f4912c47

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-qacceleratediffuserstransformers"torch>=2.1""gradio>=4.19"qrcodeopencv-python"peft==0.6.2"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2023.1.0""nncf>=2.7.0"

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

fromdiffusersimport(
StableDiffusionControlNetPipeline,
ControlNetModel,
)

controlnet=ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster")

pipe=StableDiffusionControlNetPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
controlnet=controlnet,
)

ConvertmodelstoOpenVINOIntermediaterepresentation(IR)format
------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Weneedtoprovideamodelobject,inputdataformodeltracingto
``ov.convert_model``functiontoobtainOpenVINO``ov.Model``object
instance.Modelcanbesavedondiskfornextdeploymentusing
``ov.save_model``function.

Thepipelineconsistsoffourimportantparts:

-ControlNetforconditioningbyimageannotation.
-TextEncoderforcreationconditiontogenerateanimagefromatext
prompt.
-Unetforstep-by-stepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)fordecodinglatentspacetoimage.

..code::ipython3

importgc
fromfunctoolsimportpartial
frompathlibimportPath
fromPILimportImage
importopenvinoasov
importtorch


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()

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

controlnet_ir_path=Path("./controlnet.xml")

controlnet_inputs={
"sample":torch.randn((2,4,96,96)),
"timestep":torch.tensor(1),
"encoder_hidden_states":torch.randn((2,77,768)),
"controlnet_cond":torch.randn((2,3,768,768)),
}

withtorch.no_grad():
down_block_res_samples,mid_block_res_sample=controlnet(**controlnet_inputs,return_dict=False)

ifnotcontrolnet_ir_path.exists():
controlnet.forward=partial(controlnet.forward,return_dict=False)
withtorch.no_grad():
ov_model=ov.convert_model(controlnet,example_input=controlnet_inputs)
ov.save_model(ov_model,controlnet_ir_path)
delov_model
delpipe.controlnet,controlnet
cleanup_torchscript_cache()
print("ControlNetsuccessfullyconvertedtoIR")
else:
delpipe.controlnet,controlnet
print(f"ControlNetwillbeloadedfrom{controlnet_ir_path}")


..parsed-literal::

ControlNetwillbeloadedfromcontrolnet.xml


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

text_encoder_ir_path=Path("./text_encoder.xml")

ifnottext_encoder_ir_path.exists():
pipe.text_encoder.eval()
withtorch.no_grad():
ov_model=ov.convert_model(
pipe.text_encoder,#modelinstance
example_input=torch.ones((1,77),dtype=torch.long),#inputsformodeltracing
)
ov.save_model(ov_model,text_encoder_ir_path)
delov_model
delpipe.text_encoder
cleanup_torchscript_cache()
print("TextEncodersuccessfullyconvertedtoIR")
else:
delpipe.text_encoder
print(f"TextEncoderwillbeloadedfrom{controlnet_ir_path}")


..parsed-literal::

TextEncoderwillbeloadedfromcontrolnet.xml


UNetconversion
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheprocessofUNetmodelconversionremainsthesame,likefororiginal
StableDiffusionmodel,butwithrespecttothenewinputsgeneratedby
ControlNet.

..code::ipython3

fromtypingimportTuple

unet_ir_path=Path("./unet.xml")

dtype_mapping={
torch.float32:ov.Type.f32,
torch.float64:ov.Type.f64,
torch.int32:ov.Type.i32,
torch.int64:ov.Type.i64,
}


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


pipe.unet.eval()
unet_inputs={
"sample":torch.randn((2,4,96,96)),
"timestep":torch.tensor(1),
"encoder_hidden_states":torch.randn((2,77,768)),
"down_block_additional_residuals":down_block_res_samples,
"mid_block_additional_residual":mid_block_res_sample,
}

ifnotunet_ir_path.exists():
withtorch.no_grad():
ov_model=ov.convert_model(UnetWrapper(pipe.unet),example_input=unet_inputs)

flatten_inputs=flattenize_inputs(unet_inputs.values())
forinput_data,input_tensorinzip(flatten_inputs,ov_model.inputs):
input_tensor.get_node().set_partial_shape(ov.PartialShape(input_data.shape))
input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])
ov_model.validate_nodes_and_infer_types()

ov.save_model(ov_model,unet_ir_path)
delov_model
cleanup_torchscript_cache()
delpipe.unet
gc.collect()
print("UnetsuccessfullyconvertedtoIR")
else:
delpipe.unet
print(f"Unetwillbeloadedfrom{unet_ir_path}")


..parsed-literal::

Unetwillbeloadedfromunet.xml


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

vae_ir_path=Path("./vae.xml")


classVAEDecoderWrapper(torch.nn.Module):
def__init__(self,vae):
super().__init__()
vae.eval()
self.vae=vae

defforward(self,latents):
returnself.vae.decode(latents)


ifnotvae_ir_path.exists():
vae_decoder=VAEDecoderWrapper(pipe.vae)
latents=torch.zeros((1,4,96,96))

vae_decoder.eval()
withtorch.no_grad():
ov_model=ov.convert_model(vae_decoder,example_input=latents)
ov.save_model(ov_model,vae_ir_path)
delov_model
delpipe.vae
cleanup_torchscript_cache()
print("VAEdecodersuccessfullyconvertedtoIR")
else:
delpipe.vae
print(f"VAEdecoderwillbeloadedfrom{vae_ir_path}")


..parsed-literal::

VAEdecoderwillbeloadedfromvae.xml


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

Dropdown(description='Device:',options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='CPU')



PrepareInferencepipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

Thestablediffusionmodeltakesbothalatentseedandatextpromptas
input.Thelatentseedisthenusedtogeneraterandomlatentimage
representationsofsize:math:`96\times96`whereasthetextpromptis
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
weuse
`EulerAncestralDiscreteScheduler<https://huggingface.co/docs/diffusers/api/schedulers/euler_ancestral>`__,
recommendedbyauthors.Moreinformationregardingschedulerscanbe
found
`here<https://huggingface.co/docs/diffusers/main/en/using-diffusers/schedulers>`__.

The*denoising*processisrepeatedagivennumberoftimes(bydefault
50)tostep-by-stepretrievebetterlatentimagerepresentations.Once
complete,thelatentimagerepresentationisdecodedbythedecoderpart
ofthevariationalauto-encoder.

SimilarlytoDiffusers``StableDiffusionControlNetPipeline``,wedefine
ourown``OVContrlNetStableDiffusionPipeline``inferencepipelinebased
onOpenVINO.

..code::ipython3

fromdiffusersimportDiffusionPipeline
fromtransformersimportCLIPTokenizer
fromtypingimportUnion,List,Optional,Tuple
importcv2
importnumpyasnp


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
Imagepreprocessingfunction.TakesimageinPIL.Imageformat,resizesittokeepaspectrationandfitstomodelinputwindow768x768,
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
dst_width,dst_height=scale_fit_to_window(768,768,src_width,src_height)
image=image.convert("RGB")
image=np.array(image.resize((dst_width,dst_height),resample=Image.Resampling.LANCZOS))[None,:]
pad_width=768-dst_width
pad_height=768-dst_height
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
self.vae_decoder=core.compile_model(vae_decoder,device)
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
latents=latents*np.array(self.scheduler.init_noise_sigma)
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

importqrcode


defcreate_code(content:str):
"""CreatesQRcodeswithprovidedcontent."""
qr=qrcode.QRCode(
version=1,
error_correction=qrcode.constants.ERROR_CORRECT_H,
box_size=16,
border=0,
)
qr.add_data(content)
qr.make(fit=True)
img=qr.make_image(fill_color="black",back_color="white")

#findsmallestimagesizemultipleof256thatcanfitqr
offset_min=8*16
w,h=img.size
w=(w+255+offset_min)//256*256
h=(h+255+offset_min)//256*256
ifw>1024:
raiseRuntimeError("QRcodeistoolarge,pleaseuseashortercontent")
bg=Image.new("L",(w,h),128)

#alignon16pxgrid
coords=((w-img.size[0])//2//16*16,(h-img.size[1])//2//16*16)
bg.paste(img,coords)
returnbg

..code::ipython3

fromtransformersimportCLIPTokenizer
fromdiffusersimportEulerAncestralDiscreteScheduler

tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

ov_pipe=OVContrlNetStableDiffusionPipeline(
tokenizer,
scheduler,
core,
controlnet_ir_path,
text_encoder_ir_path,
unet_ir_path,
vae_ir_path,
device=device.value,
)

Now,let’sseemodelinaction

..code::ipython3

np.random.seed(42)

qrcode_image=create_code("HiOpenVINO")
image=ov_pipe(
"cozytownonsnowymountainslope8k",
qrcode_image,
negative_prompt="blurryunrealoccluded",
num_inference_steps=25,
guidance_scale=7.7,
controlnet_conditioning_scale=1.4,
)[0]

image


..parsed-literal::

/home/ltalamanova/omz/lib/python3.8/site-packages/diffusers/configuration_utils.py:135:FutureWarning:Accessingconfigattribute`controlnet`directlyvia'OVContrlNetStableDiffusionPipeline'objectattributeisdeprecated.Pleaseaccess'controlnet'over'OVContrlNetStableDiffusionPipeline'sconfigobjectinstead,e.g.'scheduler.config.controlnet'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)




..image::qrcode-monster-with-output_files/qrcode-monster-with-output_22_1.png



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

skip_for_device="GPU"indevice.value
to_quantize=widgets.Checkbox(value=notskip_for_device,description="Quantization",disabled=skip_for_device)
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

WeuseapromptsbelowascalibrationdataforControlNetandUNet.To
collectintermediatemodelinputsforcalibrationweshouldcustomize
``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

text_prompts=[
"abilboardinNYCwithaqrcode",
"asamuraisideprofile,realistic,8K,fantasy",
"Askyviewofacolorfullakesandriversflowingthroughthedesert",
"Brightsunshinecomingthroughthecracksofawet,cavewallofbigrocks",
"Acityviewwithclouds",
"Aforestoverlookingamountain",
"Skyviewofhighlyaesthetic,ancientgreekthermalbathsinbeautifulnature",
"Adream-likefuturisticcitywiththelighttrailsofcarszippingthroughit'smanystreets",
]

negative_prompts=[
"blurryunrealoccluded",
"lowcontrastdisfigureduncenteredmangled",
"amateuroutofframelowqualitynsfw",
"uglyunderexposedjpegartifacts",
"lowsaturationdisturbingcontent",
"overexposedseveredistortion",
"amateurNSFW",
"uglymutilatedoutofframedisfigured.",
]

qr_code_contents=[
"HuggingFace",
"pre-traineddiffusionmodel",
"imagegenerationtechnique",
"controlnetwork",
"AIQRCodeGenerator",
"ExploreNNCFtoday!",
"JoinOpenVINOcommunity",
"networkcompression",
]
qrcode_images=[create_code(content)forcontentinqr_code_contents]

..code::ipython3

%%skipnot$to_quantize.value

fromtqdm.notebookimporttqdm
fromtransformersimportset_seed
fromtypingimportAny,Dict,List

set_seed(1)

num_inference_steps=25

classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model,prob:float):
super().__init__(compiled_model)
self.data_cache=[]
self.prob=np.clip(prob,0,1)

def__call__(self,*args,**kwargs):
ifnp.random.rand()>=self.prob:
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)

defcollect_calibration_data(pipeline:OVContrlNetStableDiffusionPipeline,subset_size:int)->List[Dict]:
original_unet=pipeline.unet
pipeline.unet=CompiledModelDecorator(original_unet,prob=0)
pipeline.set_progress_bar_config(disable=True)

pbar=tqdm(total=subset_size)
diff=0
forprompt,qrcode_image,negative_promptinzip(text_prompts,qrcode_images,negative_prompts):
_=pipeline(
prompt,
qrcode_image,
negative_prompt=negative_prompt,
num_inference_steps=num_inference_steps,
)
collected_subset_size=len(pipeline.unet.data_cache)
pbar.update(collected_subset_size-diff)
ifcollected_subset_size>=subset_size:
break
diff=collected_subset_size

calibration_dataset=pipeline.unet.data_cache
pipeline.set_progress_bar_config(disable=False)
pipeline.unet=original_unet
returncalibration_dataset

..code::ipython3

%%skipnot$to_quantize.value

CONTROLNET_INT8_OV_PATH=Path("controlnet_int8.xml")
UNET_INT8_OV_PATH=Path("unet_int8.xml")

ifnot(CONTROLNET_INT8_OV_PATH.exists()andUNET_INT8_OV_PATH.exists()):
subset_size=200
unet_calibration_data=collect_calibration_data(ov_pipe,subset_size=subset_size)



..parsed-literal::

0%||0/100[00:00<?,?it/s]


..parsed-literal::

/home/ltalamanova/omz/lib/python3.8/site-packages/diffusers/configuration_utils.py:135:FutureWarning:Accessingconfigattribute`controlnet`directlyvia'OVContrlNetStableDiffusionPipeline'objectattributeisdeprecated.Pleaseaccess'controlnet'over'OVContrlNetStableDiffusionPipeline'sconfigobjectinstead,e.g.'scheduler.config.controlnet'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)


ThefirstthreeinputsofControlNetarethesameastheinputsofUNet,
thelastControlNetinputisapreprocessed``qrcode_image``.

..code::ipython3

%%skipnot$to_quantize.value

ifnotCONTROLNET_INT8_OV_PATH.exists():
control_calibration_data=[]
prev_idx=0
forqrcode_imageinqrcode_images:
preprocessed_image,_=preprocess(qrcode_image)
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
unet=core.read_model(unet_ir_path)
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
controlnet=core.read_model(controlnet_ir_path)
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

np.random.seed(int(42))
int8_pipe=OVContrlNetStableDiffusionPipeline(tokenizer,scheduler,core,CONTROLNET_INT8_OV_PATH,text_encoder_ir_path,UNET_INT8_OV_PATH,vae_ir_path,device=device.value)

int8_image=int8_pipe(
"cozytownonsnowymountainslope8k",
qrcode_image,
negative_prompt="blurryunrealoccluded",
num_inference_steps=25,
guidance_scale=7.7,
controlnet_conditioning_scale=1.4
)[0]

..code::ipython3

%%skipnot$to_quantize.value

importmatplotlib.pyplotasplt

defvisualize_results(orig_img:Image.Image,optimized_img:Image.Image):
"""
Helperfunctionforresultsvisualization

Parameters:
orig_img(Image.Image):generatedimageusingFP16models
optimized_img(Image.Image):generatedimageusingquantizedmodels
Returns:
fig(matplotlib.pyplot.Figure):matplotlibgeneratedfigurecontainsdrawingresult
"""
orig_title="FP16pipeline"
control_title="INT8pipeline"
figsize=(20,20)
fig,axs=plt.subplots(1,2,figsize=figsize,sharex='all',sharey='all')
list_axes=list(axs.flat)
forainlist_axes:
a.set_xticklabels([])
a.set_yticklabels([])
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.grid(False)
list_axes[0].imshow(np.array(orig_img))
list_axes[1].imshow(np.array(optimized_img))
list_axes[0].set_title(orig_title,fontsize=15)
list_axes[1].set_title(control_title,fontsize=15)

fig.subplots_adjust(wspace=0.01,hspace=0.01)
fig.tight_layout()
returnfig

..code::ipython3

%%skipnot$to_quantize.value

fig=visualize_results(image,int8_image)



..image::qrcode-monster-with-output_files/qrcode-monster-with-output_39_0.png


Comparemodelfilesizes
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fp16_ir_model_size=unet_ir_path.with_suffix(".bin").stat().st_size/2**20
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

fp16_ir_model_size=controlnet_ir_path.with_suffix(".bin").stat().st_size/2**20
quantized_model_size=CONTROLNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size/2**20

print(f"FP16ControlNetsize:{fp16_ir_model_size:.2f}MB")
print(f"INT8ControlNetsize:{quantized_model_size:.2f}MB")
print(f"ControlNetcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16ControlNetsize:689.09MB
INT8ControlNetsize:345.14MB
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
prompt,qrcode_image=text_prompts[i],qrcode_images[i]
start=time.perf_counter()
_=pipeline(prompt,qrcode_image,num_inference_steps=25)
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
print(f"Performancespeedup:{fp_latency/int8_latency:.3f}")


..parsed-literal::

FP16pipeline:190.245seconds
INT8pipeline:166.540seconds
Performancespeedup:1.142


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




..parsed-literal::

Checkbox(value=True,description='Usequantizedmodel')



..code::ipython3

importgradioasgr

pipeline=int8_pipeifuse_quantized_model.valueelseov_pipe


def_generate(
qr_code_content:str,
prompt:str,
negative_prompt:str,
seed:Optional[int]=42,
guidance_scale:float=10.0,
controlnet_conditioning_scale:float=2.0,
num_inference_steps:int=5,
progress=gr.Progress(track_tqdm=True),
):
ifseedisnotNone:
np.random.seed(int(seed))
qrcode_image=create_code(qr_code_content)
returnpipeline(
prompt,
qrcode_image,
negative_prompt=negative_prompt,
num_inference_steps=int(num_inference_steps),
guidance_scale=guidance_scale,
controlnet_conditioning_scale=controlnet_conditioning_scale,
)[0]


demo=gr.Interface(
_generate,
inputs=[
gr.Textbox(label="QRCodecontent"),
gr.Textbox(label="TextPrompt"),
gr.Textbox(label="NegativeTextPrompt"),
gr.Number(
minimum=-1,
maximum=9999999999,
step=1,
value=42,
label="Seed",
info="Seedfortherandomnumbergenerator",
),
gr.Slider(
minimum=0.0,
maximum=25.0,
step=0.25,
value=7,
label="GuidanceScale",
info="Controlstheamountofguidancethetextpromptguidestheimagegeneration",
),
gr.Slider(
minimum=0.5,
maximum=2.5,
step=0.01,
value=1.5,
label="ControlnetConditioningScale",
info="""Controlsthereadability/creativityoftheQRcode.
Highvalues:ThegeneratedQRcodewillbemorereadable.
Lowvalues:ThegeneratedQRcodewillbemorecreative.
""",
),
gr.Slider(label="Steps",step=1,value=5,minimum=1,maximum=50),
],
outputs=["image"],
examples=[
[
"HiOpenVINO",
"cozytownonsnowymountainslope8k",
"blurryunrealoccluded",
42,
7.7,
1.4,
25,
],
],
)
try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)

#Ifyouarelaunchingremotely,specifyserver_nameandserver_port
#EXAMPLE:`demo.launch(server_name='yourservername',server_port='serverportinint')`
#TolearnmorepleaserefertotheGradiodocs:https://gradio.app/docs/
