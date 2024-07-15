ImageGenerationwithTiny-SDandOpenVINO™
===========================================

Inrecenttimes,theAIcommunityhaswitnessedaremarkablesurgein
thedevelopmentoflargerandmoreperformantlanguagemodels,suchas
Falcon40B,LLaMa-270B,Falcon40B,MPT30B,andintheimagingdomain
withmodelslikeSD2.1andSDXL.Theseadvancementshaveundoubtedly
pushedtheboundariesofwhatAIcanachieve,enablinghighlyversatile
andstate-of-the-artimagegenerationandlanguageunderstanding
capabilities.However,thebreakthroughoflargemodelscomeswith
substantialcomputationaldemands.Toresolvethisissue,recent
researchonefficientStableDiffusionhasprioritizedreducingthe
numberofsamplingstepsandutilizingnetworkquantization.

Movingtowardsthegoalofmakingimagegenerativemodelsfaster,
smaller,andcheaper,Tiny-SDwasproposedbySegmind.TinySDisa
compressedStableDiffusion(SD)modelthathasbeentrainedon
Knowledge-Distillation(KD)techniquesandtheworkhasbeenlargely
basedonthis`paper<https://arxiv.org/pdf/2305.15798.pdf>`__.The
authorsdescribeaBlock-removalKnowledge-Distillationmethodwhere
someoftheUNetlayersareremovedandthestudentmodelweightsare
trained.UsingtheKDmethodsdescribedinthepaper,theywereableto
traintwocompressedmodelsusingthe🧨diffuserslibrary;Smalland
Tiny,thathave35%and55%fewerparameters,respectivelythanthebase
modelwhileachievingcomparableimagefidelityasthebasemodel.More
detailsaboutmodelcanbefoundin`model
card<https://huggingface.co/segmind/tiny-sd>`__,`blog
post<https://huggingface.co/blog/sd_distillation>`__andtraining
`repository<https://github.com/segmind/distill-sd>`__.

ThisnotebookdemonstrateshowtoconvertandruntheTiny-SDmodel
usingOpenVINO.

Thenotebookcontainsthefollowingsteps:

1.ConvertPyTorchmodelstoOpenVINOIntermediateRepresentationusing
OpenVINOConverterTool(OVC).
2.PrepareInferencePipeline.
3.RunInferencepipelinewithOpenVINO.
4.RunInteractivedemoforTiny-SDmodel

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`CreatePyTorchModelspipeline<#create-pytorch-models-pipeline>`__
-`ConvertmodelstoOpenVINOIntermediaterepresentation
format<#convert-models-to-openvino-intermediate-representation-format>`__

-`TextEncoder<#text-encoder>`__
-`U-net<#u-net>`__
-`VAE<#vae>`__

-`PrepareInferencePipeline<#prepare-inference-pipeline>`__
-`ConfigureInferencePipeline<#configure-inference-pipeline>`__

-`CalibrateUNetforGPU
inference<#calibrate-unet-for-gpu-inference>`__
-`Text-to-Imagegeneration<#text-to-image-generation>`__
-`Image-to-Imagegeneration<#image-to-image-generation>`__
-`InteractiveDemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installrequireddependencies

..code::ipython3

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"torch>=2.1"torchvision"openvino>=2023.3.0""opencv-python""pillow""diffusers>=0.18.0""transformers>=4.30.2""gradio>=4.19"

CreatePyTorchModelspipeline
------------------------------

`backtotop⬆️<#table-of-contents>`__

``StableDiffusionPipeline``isanend-to-endinferencepipelinethatyou
canusetogenerateimagesfromtextwithjustafewlinesofcode.

First,loadthepre-trainedweightsofallcomponentsofthemodel.

..code::ipython3

importgc
fromdiffusersimportStableDiffusionPipeline

model_id="segmind/tiny-sd"

pipe=StableDiffusionPipeline.from_pretrained(model_id).to("cpu")
text_encoder=pipe.text_encoder
text_encoder.eval()
unet=pipe.unet
unet.eval()
vae=pipe.vae
vae.eval()

delpipe
gc.collect()


..parsed-literal::

2023-09-1815:58:40.831193:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2023-09-1815:58:40.870576:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2023-09-1815:58:41.537042:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
text_encoder/model.safetensorsnotfound



..parsed-literal::

Loadingpipelinecomponents...:0%||0/5[00:00<?,?it/s]




..parsed-literal::

27



ConvertmodelstoOpenVINOIntermediaterepresentationformat
-------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchthroughconversiontoOpenVINOIntermediate
Representation(IR)format.TotaketheadvantageofOpenVINO
optimizationtoolsandfeatures,themodelshouldbeconvertedusingthe
OpenVINOConvertertool(OVC).The``openvino.convert_model``function
providesPythonAPIforOVCusage.Thefunctionreturnstheinstanceof
theOpenVINOModelclass,whichisreadyforuseinthePython
interface.However,itcanalsobesavedondiskusing
``openvino.save_model``forfutureexecution.

StartingfromOpenVINO2023.0.0releaseOpenVINOsupportsdirect
conversionPyTorchmodels.Toperformconversion,weshouldprovide
PyTorchmodelinstanceandexampleinputinto
``openvino.convert_model``.Bydefault,modelconvertedwithdynamic
shapespreserving,inordertofixateinputshapetogenerateimageof
specificresolution,``input``parameteradditionallycanbespecified.

Themodelconsistsofthreeimportantparts:

-TextEncoderforcreationconditiontogenerateimagefromtext
prompt.
-U-netforstepbystepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)forencodinginputimagetolatentspace(if
required)anddecodinglatentspacetoimagebackaftergeneration.

Letusconverteachpart.

TextEncoder
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thetext-encoderisresponsiblefortransformingtheinputprompt,for
example,“aphotoofanastronautridingahorse”intoanembedding
spacethatcanbeunderstoodbytheU-Net.Itisusuallyasimple
transformer-basedencoderthatmapsasequenceofinputtokenstoa
sequenceoflatenttextembeddings.

Inputofthetextencoderisthetensor``input_ids``whichcontains
indexesoftokensfromtextprocessedbytokenizerandpaddedtomaximum
lengthacceptedbymodel.Modeloutputsaretwotensors:
``last_hidden_state``-hiddenstatefromthelastMultiHeadAttention
layerinthemodeland``pooler_out``-Pooledoutputforwholemodel
hiddenstates.

..code::ipython3

frompathlibimportPath
importtorch
importopenvinoasov

TEXT_ENCODER_OV_PATH=Path("text_encoder.xml")


defconvert_encoder(text_encoder:torch.nn.Module,ir_path:Path):
"""
ConvertTextEncodermode.
Functionacceptstextencodermodel,andpreparesexampleinputsforconversion,
Parameters:
text_encoder(torch.nn.Module):text_encodermodelfromStableDiffusionpipeline
ir_path(Path):Fileforstoringmodel
Returns:
None
"""
input_ids=torch.ones((1,77),dtype=torch.long)
#switchmodeltoinferencemode
text_encoder.eval()

#disablegradientscalculationforreducingmemoryconsumption
withtorch.no_grad():
#ExportmodeltoIRformat
ov_model=ov.convert_model(
text_encoder,
example_input=input_ids,
input=[
(1,77),
],
)
ov.save_model(ov_model,ir_path)
delov_model
print(f"TextEncodersuccessfullyconvertedtoIRandsavedto{ir_path}")


ifnotTEXT_ENCODER_OV_PATH.exists():
convert_encoder(text_encoder,TEXT_ENCODER_OV_PATH)
else:
print(f"Textencoderwillbeloadedfrom{TEXT_ENCODER_OV_PATH}")

deltext_encoder
gc.collect()


..parsed-literal::

Textencoderwillbeloadedfromtext_encoder.xml




..parsed-literal::

0



U-net
~~~~~

`backtotop⬆️<#table-of-contents>`__

U-netmodelhasthreeinputs:

-``sample``-latentimagesamplefrompreviousstep.Generation
processhasnotbeenstartedyet,soyouwilluserandomnoise.
-``timestep``-currentschedulerstep.
-``encoder_hidden_state``-hiddenstateoftextencoder.

Modelpredictsthe``sample``stateforthenextstep.

..code::ipython3

importnumpyasnp
fromopenvinoimportPartialShape,Type

UNET_OV_PATH=Path("unet.xml")

dtype_mapping={torch.float32:Type.f32,torch.float64:Type.f64}


defconvert_unet(unet:torch.nn.Module,ir_path:Path):
"""
ConvertU-netmodeltoIRformat.
Functionacceptsunetmodel,preparesexampleinputsforconversion,
Parameters:
unet(StableDiffusionPipeline):unetfromStableDiffusionpipeline
ir_path(Path):Fileforstoringmodel
Returns:
None
"""
#prepareinputs
encoder_hidden_state=torch.ones((2,77,768))
latents_shape=(2,4,512//8,512//8)
latents=torch.randn(latents_shape)
t=torch.from_numpy(np.array(1,dtype=float))
dummy_inputs=(latents,t,encoder_hidden_state)
input_info=[]
forinput_tensorindummy_inputs:
shape=PartialShape(tuple(input_tensor.shape))
element_type=dtype_mapping[input_tensor.dtype]
input_info.append((shape,element_type))

unet.eval()
withtorch.no_grad():
ov_model=ov.convert_model(unet,example_input=dummy_inputs,input=input_info)
ov.save_model(ov_model,ir_path)
delov_model
print(f"UnetsuccessfullyconvertedtoIRandsavedto{ir_path}")


ifnotUNET_OV_PATH.exists():
convert_unet(unet,UNET_OV_PATH)
gc.collect()
else:
print(f"Unetwillbeloadedfrom{UNET_OV_PATH}")
delunet
gc.collect()


..parsed-literal::

Unetwillbeloadedfromunet.xml




..parsed-literal::

0



VAE
~~~

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
runinferencefortext-to-image,thereisnoinitialimageasastarting
point.Youcanskipthisstepanddirectlygenerateinitialrandom
noise.

Astheencoderandthedecoderareusedindependentlyindifferentparts
ofthepipeline,itwillbebettertoconvertthemtoseparatemodels.

..code::ipython3

VAE_ENCODER_OV_PATH=Path("vae_encodr.xml")


defconvert_vae_encoder(vae:torch.nn.Module,ir_path:Path):
"""
ConvertVAEmodelforencodingtoIRformat.
Functionacceptsvaemodel,createswrapperclassforexportonlynecessaryforinferencepart,
preparesexampleinputsforconversion,
Parameters:
vae(torch.nn.Module):VAEmodelfromStableDiffusiopipeline
ir_path(Path):Fileforstoringmodel
Returns:
None
"""

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
ov_model=ov.convert_model(vae_encoder,example_input=image,input=[((1,3,512,512),)])
ov.save_model(ov_model,ir_path)
delov_model
print(f"VAEencodersuccessfullyconvertedtoIRandsavedto{ir_path}")


ifnotVAE_ENCODER_OV_PATH.exists():
convert_vae_encoder(vae,VAE_ENCODER_OV_PATH)
else:
print(f"VAEencoderwillbeloadedfrom{VAE_ENCODER_OV_PATH}")

VAE_DECODER_OV_PATH=Path("vae_decoder.xml")


defconvert_vae_decoder(vae:torch.nn.Module,ir_path:Path):
"""
ConvertVAEmodelfordecodingtoIRformat.
Functionacceptsvaemodel,createswrapperclassforexportonlynecessaryforinferencepart,
preparesexampleinputsforconversion,
Parameters:
vae(torch.nn.Module):VAEmodelfrmStableDiffusionpipeline
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

vae_decoder=VAEDecoderWrapper(vae)
latents=torch.zeros((1,4,64,64))

vae_decoder.eval()
withtorch.no_grad():
ov_model=ov.convert_model(vae_decoder,example_input=latents,input=[((1,4,64,64),)])
ov.save_model(ov_model,ir_path)
delov_model
print(f"VAEdecodersuccessfullyconvertedtoIRandsavedto{ir_path}")


ifnotVAE_DECODER_OV_PATH.exists():
convert_vae_decoder(vae,VAE_DECODER_OV_PATH)
else:
print(f"VAEdecoderwillbeloadedfrom{VAE_DECODER_OV_PATH}")

delvae
gc.collect()


..parsed-literal::

VAEencoderwillbeloadedfromvae_encodr.xml
VAEdecoderwillbeloadedfromvae_decoder.xml




..parsed-literal::

0



PrepareInferencePipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

Puttingitalltogether,letusnowtakeacloserlookathowthemodel
worksininferencebyillustratingthelogicalflow.

..figure::https://user-images.githubusercontent.com/29454499/260981188-c112dd0a-5752-4515-adca-8b09bea5d14a.png
:alt:sd-pipeline

sd-pipeline

Asyoucanseefromthediagram,theonlydifferencebetween
Text-to-Imageandtext-guidedImage-to-Imagegenerationinapproachis
howinitiallatentstateisgenerated.IncaseofImage-to-Image
generation,youadditionallyhaveanimageencodedbyVAEencodermixed
withthenoiseproducedbyusinglatentseed,whileinText-to-Imageyou
useonlynoiseasinitiallatentstate.Thestablediffusionmodeltakes
bothalatentimagerepresentationofsize:math:`64\times64`anda
textpromptistransformedtotextembeddingsofsize
:math:`77\times768`viaCLIP’stextencoderasaninput.

Next,theU-Netiteratively*denoises*therandomlatentimage
representationswhilebeingconditionedonthetextembeddings.The
outputoftheU-Net,beingthenoiseresidual,isusedtocomputea
denoisedlatentimagerepresentationviaascheduleralgorithm.Many
differentscheduleralgorithmscanbeusedforthiscomputation,each
havingitsprosandcons.ForStableDiffusion,itisrecommendedtouse
oneof:

-`PNDM
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py>`__
-`DDIM
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py>`__
-`K-LMS
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py>`__\(you
willuseitinyourpipeline)

Theoryonhowthescheduleralgorithmfunctionworksisoutofscopefor
thisnotebook.Nonetheless,inshort,youshouldrememberthatyou
computethepredicteddenoisedimagerepresentationfromtheprevious
noiserepresentationandthepredictednoiseresidual.Formore
information,refertotherecommended`ElucidatingtheDesignSpaceof
Diffusion-BasedGenerativeModels<https://arxiv.org/abs/2206.00364>`__

The*denoising*processisrepeatedgivennumberoftimes(bydefault
50)tostep-by-stepretrievebetterlatentimagerepresentations.When
complete,thelatentimagerepresentationisdecodedbythedecoderpart
ofthevariationalautoencoder.

..code::ipython3

importinspect
fromtypingimportList,Optional,Union,Dict

importPIL
importcv2

fromtransformersimportCLIPTokenizer
fromdiffusers.pipelines.pipeline_utilsimportDiffusionPipeline
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


defpreprocess(image:PIL.Image.Image):
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
dst_width,dst_height=scale_fit_to_window(512,512,src_width,src_height)
image=np.array(image.resize((dst_width,dst_height),resample=PIL.Image.Resampling.LANCZOS))[None,:]
pad_width=512-dst_width
pad_height=512-dst_height
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
vae_encoder:ov.Model=None,
):
"""
Pipelinefortext-to-imagegenerationusingStableDiffusion.
Parameters:
vae(Model):
VariationalAuto-Encoder(VAE)Modeltodecodeimagestoandfromlatentrepresentations.
text_encoder(Model):
Frozentext-encoder.StableDiffusionusesthetextportionof
[CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel),specifically
theclip-vit-large-patch14(https://huggingface.co/openai/clip-vit-large-patch14)variant.
tokenizer(CLIPTokenizer):
TokenizerofclassCLIPTokenizer(https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
unet(Model):ConditionalU-Netarchitecturetodenoisetheencodedimagelatents.
scheduler(SchedulerMixin):
Aschedulertobeusedincombinationwithunettodenoisetheencodedimagelatents.Canbeoneof
DDIMScheduler,LMSDiscreteScheduler,orPNDMScheduler.
"""
super().__init__()
self.scheduler=scheduler
self.vae_decoder=vae_decoder
self.vae_encoder=vae_encoder
self.text_encoder=text_encoder
self.unet=unet
self._text_encoder_output=text_encoder.output(0)
self._unet_output=unet.output(0)
self._vae_d_output=vae_decoder.output(0)
self._vae_e_output=vae_encoder.output(0)ifvae_encoderisnotNoneelseNone
self.height=512
self.width=512
self.tokenizer=tokenizer

def__call__(
self,
prompt:Union[str,List[str]],
image:PIL.Image.Image=None,
num_inference_steps:Optional[int]=50,
negative_prompt:Union[str,List[str]]=None,
guidance_scale:Optional[float]=7.5,
eta:Optional[float]=0.0,
output_type:Optional[str]="pil",
seed:Optional[int]=None,
strength:float=1.0,
gif:Optional[bool]=False,
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
negative_prompt(strorList[str]):
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
seed(int,*optional*,None):
Seedforrandomgeneratorstateinitialization.
gif(bool,*optional*,False):
Flagforstoringallstepsresultsornot.
Returns:
Dictionarywithkeys:
sample-thelastgeneratedimagePIL.Image.Imageornp.array
iterations-*optional*(ifgif=True)imagesforalldiffusionsteps,ListofPIL.Image.Imageornp.array.
"""
ifseedisnotNone:
np.random.seed(seed)

img_buffer=[]
do_classifier_free_guidance=guidance_scale>1.0
#getprompttextembeddings
text_embeddings=self._encode_prompt(
prompt,
do_classifier_free_guidance=do_classifier_free_guidance,
negative_prompt=negative_prompt,
)

#settimesteps
accepts_offset="offset"inset(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
extra_set_kwargs={}
ifaccepts_offset:
extra_set_kwargs["offset"]=1

self.scheduler.set_timesteps(num_inference_steps,**extra_set_kwargs)
timesteps,num_inference_steps=self.get_timesteps(num_inference_steps,strength)
latent_timestep=timesteps[:1]

#gettheinitialrandomnoiseunlesstheusersuppliedit
latents,meta=self.prepare_latents(image,latent_timestep)

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
noise_pred=self.unet([latent_model_input,t,text_embeddings])[self._unet_output]
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
ifgif:
image=self.vae_decoder(latents*(1/0.18215))[self._vae_d_output]
image=self.postprocess_image(image,meta,output_type)
img_buffer.extend(image)

#scaleanddecodetheimagelatentswithvae
image=self.vae_decoder(latents*(1/0.18215))[self._vae_d_output]

image=self.postprocess_image(image,meta,output_type)
return{"sample":image,"iterations":img_buffer}

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

text_embeddings=self.text_encoder(text_input_ids)[self._text_encoder_output]

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

uncond_embeddings=self.text_encoder(uncond_input.input_ids)[self._text_encoder_output]

#duplicateunconditionalembeddingsforeachgenerationperprompt,usingmpsfriendlymethod
seq_len=uncond_embeddings.shape[1]
uncond_embeddings=np.tile(uncond_embeddings,(1,num_images_per_prompt,1))
uncond_embeddings=np.reshape(uncond_embeddings,(batch_size*num_images_per_prompt,seq_len,-1))

#Forclassifierfreeguidance,weneedtodotwoforwardpasses.
#Hereweconcatenatetheunconditionalandtextembeddingsintoasinglebatch
#toavoiddoingtwoforwardpasses
text_embeddings=np.concatenate([uncond_embeddings,text_embeddings])

returntext_embeddings

defprepare_latents(self,image:PIL.Image.Image=None,latent_timestep:torch.Tensor=None):
"""
Functionforgettinginitiallatentsforstartinggeneration

Parameters:
image(PIL.Image.Image,*optional*,None):
Inputimageforgeneration,ifnotprovidedrandonnoisewillbeusedasstartingpoint
latent_timestep(torch.Tensor,*optional*,None):
Predictedbyschedulerinitialstepforimagegeneration,requiredforlatentimagemixingwithnosie
Returns:
latents(np.ndarray):
Imageencodedinlatentspace
"""
latents_shape=(1,4,self.height//8,self.width//8)
noise=np.random.randn(*latents_shape).astype(np.float32)
ifimageisNone:
#ifyouuseLMSDiscreteScheduler,let'smakesurelatentsaremultipliedbysigmas
ifisinstance(self.scheduler,LMSDiscreteScheduler):
noise=noise*self.scheduler.sigmas[0].numpy()
returnnoise,{}
input_image,meta=preprocess(image)
latents=self.vae_encoder(input_image)[self._vae_e_output]*0.18215
latents=self.scheduler.add_noise(torch.from_numpy(latents),torch.from_numpy(noise),latent_timestep).numpy()
returnlatents,meta

defpostprocess_image(self,image:np.ndarray,meta:Dict,output_type:str="pil"):
"""
Postprocessingfordecodedimage.TakesgeneratedimagedecodedbyVAEdecoder,unpadittoinitilaimagesize(ifrequired),
normalizeandconvertto[0,255]pixelsrange.Optionally,convertesitfromnp.ndarraytoPIL.Imageformat

Parameters:
image(np.ndarray):
Generatedimage
meta(Dict):
Metadataobtainedonlatentspreparingstep,canbeempty
output_type(str,*optional*,pil):
Outputformatforresult,canbepilornumpy
Returns:
image(Listofnp.ndarrayorPIL.Image.Image):
Postprocessedimages
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

defget_timesteps(self,num_inference_steps:int,strength:float):
"""
Helperfunctionforgettingschedulertimestepsforgeneration
Incaseofimage-to-imagegeneration,itupdatesnumberofstepsaccordingtostrength

Parameters:
num_inference_steps(int):
numberofinferencestepsforgeneration
strength(float):
valuebetween0.0and1.0,thatcontrolstheamountofnoisethatisaddedtotheinputimage.
Valuesthatapproach1.0enablelotsofvariationsbutwillalsoproduceimagesthatarenotsemanticallyconsistentwiththeinput.
"""
#gettheoriginaltimestepusinginit_timestep
init_timestep=min(int(num_inference_steps*strength),num_inference_steps)

t_start=max(num_inference_steps-init_timestep,0)
timesteps=self.scheduler.timesteps[t_start:]

returntimesteps,num_inference_steps-t_start

ConfigureInferencePipeline
----------------------------

`backtotop⬆️<#table-of-contents>`__

First,youshouldcreateinstancesofOpenVINOModel.

..code::ipython3

core=ov.Core()

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO.

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

Dropdown(description='Device:',index=2,options=('CPU','GPU','AUTO'),value='AUTO')



..code::ipython3

text_enc=core.compile_model(TEXT_ENCODER_OV_PATH,device.value)

CalibrateUNetforGPUinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OnaGPUdeviceamodelisexecutedinFP16precision.ForTiny-SDUNet
modelthereknowntobeaccuracyissuescausedbythis.Therefore,a
specialcalibrationprocedureisusedtoselectivelymarksome
operationstobeexecutedinfullprecision.

..code::ipython3

importpickle
importrequests
importos

#Fetch`model_upcast_utils`whichhelpstorestoreaccuracywheninferredonGPU
r=requests.get("https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/model_upcast_utils.py")
withopen("model_upcast_utils.py","w")asf:
f.write(r.text)

#FetchanexampleinputforUNetmodelneededforupcastingcalibrationprocess
r=requests.get("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/pkl/unet_calibration_example_input.pkl")
withopen("unet_calibration_example_input.pkl","wb")asf:
f.write(r.content)

frommodel_upcast_utilsimport(
is_model_partially_upcasted,
partially_upcast_nodes_to_fp32,
)

unet_model=core.read_model(UNET_OV_PATH)
if"GPU"incore.available_devicesandnotis_model_partially_upcasted(unet_model):
withopen("unet_calibration_example_input.pkl","rb")asf:
example_input=pickle.load(f)
unet_model=partially_upcast_nodes_to_fp32(unet_model,example_input,upcast_ratio=0.7,operation_types=["Convolution"])

ov.save_model(unet_model,UNET_OV_PATH.with_suffix("._tmp.xml"))
delunet_model
os.remove(UNET_OV_PATH)
os.remove(str(UNET_OV_PATH).replace(".xml",".bin"))
UNET_OV_PATH.with_suffix("._tmp.xml").rename(UNET_OV_PATH)
UNET_OV_PATH.with_suffix("._tmp.bin").rename(UNET_OV_PATH.with_suffix(".bin"))

..code::ipython3

unet_model=core.compile_model(UNET_OV_PATH,device.value)

..code::ipython3

ov_config={"INFERENCE_PRECISION_HINT":"f32"}ifdevice.value!="CPU"else{}

vae_decoder=core.compile_model(VAE_DECODER_OV_PATH,device.value,ov_config)
vae_encoder=core.compile_model(VAE_ENCODER_OV_PATH,device.value,ov_config)

Modeltokenizerandschedulerarealsoimportantpartsofthepipeline.
Letusdefinethemandputallcomponentstogether

..code::ipython3

fromtransformersimportCLIPTokenizer
fromdiffusers.schedulersimportLMSDiscreteScheduler

lms=LMSDiscreteScheduler(beta_start=0.00085,beta_end=0.012,beta_schedule="scaled_linear")
tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

ov_pipe=OVStableDiffusionPipeline(
tokenizer=tokenizer,
text_encoder=text_enc,
unet=unet_model,
vae_encoder=vae_encoder,
vae_decoder=vae_decoder,
scheduler=lms,
)

Text-to-Imagegeneration
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,let’sseemodelinaction

..code::ipython3

text_prompt="RAWstudiophotoofAnintricateforestminitownlandscapetrappedinabottle,atmosphericolivalighting,onthetable,intricatedetails,darkshot,soothingtones,mutedcolors"
seed=431
num_steps=20

..code::ipython3

print("Pipelinesettings")
print(f"Inputtext:{text_prompt}")
print(f"Seed:{seed}")
print(f"Numberofsteps:{num_steps}")


..parsed-literal::

Pipelinesettings
Inputtext:RAWstudiophotoofAnintricateforestminitownlandscapetrappedinabottle,atmosphericolivalighting,onthetable,intricatedetails,darkshot,soothingtones,mutedcolors
Seed:431
Numberofsteps:20


..code::ipython3

result=ov_pipe(text_prompt,num_inference_steps=num_steps,seed=seed)



..parsed-literal::

0%||0/20[00:00<?,?it/s]


Finally,letussavegenerationresults.Thepipelinereturnsseveral
results:``sample``containsfinalgeneratedimage,``iterations``
containslistofintermediateresultsforeachstep.

..code::ipython3

final_image=result["sample"][0]
final_image.save("result.png")

Nowisshowtime!

..code::ipython3

text="\n\t".join(text_prompt.split("."))
print("Inputtext:")
print("\t"+text)
display(final_image)


..parsed-literal::

Inputtext:
	RAWstudiophotoofAnintricateforestminitownlandscapetrappedinabottle,atmosphericolivalighting,onthetable,intricatedetails,darkshot,soothingtones,mutedcolors



..image::tiny-sd-image-generation-with-output_files/tiny-sd-image-generation-with-output_35_1.png


Nice.Asyoucansee,thepicturehasquiteahighdefinition🔥.

Image-to-Imagegeneration
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OneofthemostamazingfeaturesofStableDiffusionmodelisthe
abilitytoconditionimagegenerationfromanexistingimageorsketch.
Givena(potentiallycrude)imageandtherighttextprompt,latent
diffusionmodelscanbeusedto“enhance”animage.

Image-to-Imagegeneration,inadditionallytothetextprompt,requires
providingtheinitialimage.Optionally,youcanalsochange
``strength``parameter,whichisavaluebetween0.0and1.0,that
controlstheamountofnoisethatisaddedtotheinputimage.Values
thatapproach1.0enablelotsofvariationsbutwillalsoproduceimages
thatarenotsemanticallyconsistentwiththeinput.Oneofthe
interestingusecasesforImage-to-Imagegenerationisdepainting-
turningsketchesorpaintingsintorealisticphotographs.

Additionally,toimproveimagegenerationquality,modelsupports
negativeprompting.Technically,positivepromptsteersthediffusion
towardtheimagesassociatedwithit,whilenegativepromptsteersthe
diffusionawayfromit.Inotherwords,negativepromptdeclares
undesiredconceptsforgenerationimage,e.g. ifwewanttohave
colorfulandbrightimage,grayscaleimagewillberesultwhichwewant
toavoid,inthiscasegrayscalecanbetreatedasnegativeprompt.The
positiveandnegativepromptareinequalfooting.Youcanalwaysuse
onewithorwithouttheother.Moreexplanationofhowitworkscanbe
foundinthis
`article<https://stable-diffusion-art.com/how-negative-prompt-work/>`__.

..code::ipython3

text_prompt_i2i="professionalphotoportraitofwoman,highlydetailed,hyperrealistic,cinematiceffects,softlighting"
negative_prompt_i2i=(
"blurry,poorquality,lowres,worstquality,cropped,ugly,poorlydrawnface,withouteyes,mutation,unreal,animate,poorlydrawneyes"
)
num_steps_i2i=40
seed_i2i=82698152
strength=0.68

..code::ipython3

fromdiffusers.utilsimportload_image

default_image_url="https://user-images.githubusercontent.com/29454499/260418860-69cc443a-9ee6-493c-a393-3a97af080be7.jpg"
#readuploadedimage
image=load_image(default_image_url)
print("Pipelinesettings")
print(f"Inputpositiveprompt:\n\t{text_prompt_i2i}")
print(f"Inputnegativeprompt:\n\t{negative_prompt_i2i}")
print(f"Seed:{seed_i2i}")
print(f"Numberofsteps:{num_steps_i2i}")
print(f"Strength:{strength}")
print("Inputimage:")
display(image)
processed_image=ov_pipe(
text_prompt_i2i,
image,
negative_prompt=negative_prompt_i2i,
num_inference_steps=num_steps_i2i,
seed=seed_i2i,
strength=strength,
)


..parsed-literal::

Pipelinesettings
Inputpositiveprompt:
	professionalphotoportraitofwoman,highlydetailed,hyperrealistic,cinematiceffects,softlighting
Inputnegativeprompt:
	blurry,poorquality,lowres,worstquality,cropped,ugly,poorlydrawnface,withouteyes,mutation,unreal,animate,poorlydrawneyes
Seed:82698152
Numberofsteps:40
Strength:0.68
Inputimage:



..image::tiny-sd-image-generation-with-output_files/tiny-sd-image-generation-with-output_39_1.png



..parsed-literal::

0%||0/27[00:00<?,?it/s]


..code::ipython3

final_image_i2i=processed_image["sample"][0]
final_image_i2i.save("result_i2i.png")

..code::ipython3

text_i2i="\n\t".join(text_prompt_i2i.split("."))
print("Inputtext:")
print("\t"+text_i2i)
display(final_image_i2i)


..parsed-literal::

Inputtext:
	professionalphotoportraitofwoman,highlydetailed,hyperrealistic,cinematiceffects,softlighting



..image::tiny-sd-image-generation-with-output_files/tiny-sd-image-generation-with-output_41_1.png


InteractiveDemo
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr

sample_img_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/tower.jpg"

img=load_image(sample_img_url).save("tower.jpg")


defgenerate_from_text(text,negative_text,seed,num_steps,_=gr.Progress(track_tqdm=True)):
result=ov_pipe(text,negative_prompt=negative_text,num_inference_steps=num_steps,seed=seed)
returnresult["sample"][0]


defgenerate_from_image(img,text,negative_text,seed,num_steps,strength,_=gr.Progress(track_tqdm=True)):
result=ov_pipe(
text,
img,
negative_prompt=negative_text,
num_inference_steps=num_steps,
seed=seed,
strength=strength,
)
returnresult["sample"][0]


withgr.Blocks()asdemo:
withgr.Tab("Text-to-Imagegeneration"):
withgr.Row():
withgr.Column():
text_input=gr.Textbox(lines=3,label="Positiveprompt")
negative_text_input=gr.Textbox(lines=3,label="Negativeprompt")
seed_input=gr.Slider(0,10000000,value=751,label="Seed")
steps_input=gr.Slider(1,50,value=20,step=1,label="Steps")
out=gr.Image(label="Result",type="pil")
sample_text=(
"futuristicsynthwavecity,retrosunset,crystals,spires,volumetriclighting,studioGhiblistyle,renderedinunrealenginewithcleandetails"
)
sample_text2="RAWstudiophotooftinycutehappycatinayellowraincoatinthewoods,rain,acharacterportrait,softlighting,highresolution,photorealistic,extremelydetailed"
negative_sample_text=""
negative_sample_text2="badanatomy,blurry,noisy,jpegartifacts,lowquality,geometry,mutation,disgusting.ugly"
btn=gr.Button()
btn.click(
generate_from_text,
[text_input,negative_text_input,seed_input,steps_input],
out,
)
gr.Examples(
[
[sample_text,negative_sample_text,42,20],
[sample_text2,negative_sample_text2,1561,25],
],
[text_input,negative_text_input,seed_input,steps_input],
)
withgr.Tab("Image-to-Imagegeneration"):
withgr.Row():
withgr.Column():
i2i_input=gr.Image(label="Image",type="pil")
i2i_text_input=gr.Textbox(lines=3,label="Text")
i2i_negative_text_input=gr.Textbox(lines=3,label="Negativeprompt")
i2i_seed_input=gr.Slider(0,10000000,value=42,label="Seed")
i2i_steps_input=gr.Slider(1,50,value=10,step=1,label="Steps")
strength_input=gr.Slider(0,1,value=0.5,label="Strength")
i2i_out=gr.Image(label="Result",type="pil")
i2i_btn=gr.Button()
sample_i2i_text="amazingwatercolorpainting"
i2i_btn.click(
generate_from_image,
[
i2i_input,
i2i_text_input,
i2i_negative_text_input,
i2i_seed_input,
i2i_steps_input,
strength_input,
],
i2i_out,
)
gr.Examples(
[["tower.jpg",sample_i2i_text,"",6400023,40,0.3]],
[
i2i_input,
i2i_text_input,
i2i_negative_text_input,
i2i_seed_input,
i2i_steps_input,
strength_input,
],
)

try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7863

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7863/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>

