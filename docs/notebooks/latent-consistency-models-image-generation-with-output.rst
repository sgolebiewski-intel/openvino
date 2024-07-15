ImagegenerationwithLatentConsistencyModelandOpenVINO
===========================================================

LCMs:ThenextgenerationofgenerativemodelsafterLatentDiffusion
Models(LDMs).LatentDiffusionmodels(LDMs)haveachievedremarkable
resultsinsynthesizinghigh-resolutionimages.However,theiterative
samplingiscomputationallyintensiveandleadstoslowgeneration.

Inspiredby`ConsistencyModels<https://arxiv.org/abs/2303.01469>`__,
`LatentConsistencyModels<https://arxiv.org/pdf/2310.04378.pdf>`__
(LCMs)wereproposed,enablingswiftinferencewithminimalstepsonany
pre-trainedLDMs,includingStableDiffusion.The`ConsistencyModel
(CM)(Songetal.,2023)<https://arxiv.org/abs/2303.01469>`__isanew
familyofgenerativemodelsthatenablesone-steporfew-step
generation.ThecoreideaoftheCMistolearnthefunctionthatmaps
anypointsonatrajectoryofthePF-ODE(probabilityflowof`ordinary
differential
equation<https://en.wikipedia.org/wiki/Ordinary_differential_equation>`__)
tothattrajectory’sorigin(i.e.,thesolutionofthePF-ODE).By
learningconsistencymappingsthatmaintainpointconsistencyon
ODE-trajectory,thesemodelsallowforsingle-stepgeneration,
eliminatingtheneedforcomputation-intensiveiterations.However,CM
isconstrainedtopixelspaceimagegenerationtasks,makingit
unsuitableforsynthesizinghigh-resolutionimages.LCMsadopta
consistencymodelintheimagelatentspaceforgeneration
high-resolutionimages.Viewingtheguidedreversediffusionprocessas
solvinganaugmentedprobabilityflowODE(PF-ODE),LCMsaredesignedto
directlypredictthesolutionofsuchODEinlatentspace,mitigating
theneedfornumerousiterationsandallowingrapid,high-fidelity
sampling.Utilizingimagelatentspaceinlarge-scalediffusionmodels
likeStableDiffusion(SD)haseffectivelyenhancedimagegeneration
qualityandreducedcomputationalload.TheauthorsofLCMsprovidea
simpleandefficientone-stageguidedconsistencydistillationmethod
namedLatentConsistencyDistillation(LCD)todistillSDforfew-step
(2∼4)oreven1-stepsamplingandproposetheSKIPPING-STEPtechniqueto
furtheracceleratetheconvergence.Moredetailsaboutproposedapproach
andmodelscanbefoundin`project
page<https://latent-consistency-models.github.io/>`__,
`paper<https://arxiv.org/abs/2310.04378>`__and`original
repository<https://github.com/luosiallen/latent-consistency-model>`__.

Inthistutorial,weconsiderhowtoconvertandrunLCMusingOpenVINO.
Anadditionalpartdemonstrateshowtorunquantizationwith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__tospeedup
pipeline.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`PreparemodelsforOpenVINOformat
conversion<#prepare-models-for-openvino-format-conversion>`__
-`ConvertmodelstoOpenVINO
format<#convert-models-to-openvino-format>`__

-`TextEncoder<#text-encoder>`__
-`U-Net<#u-net>`__
-`VAE<#vae>`__

-`Prepareinferencepipeline<#prepare-inference-pipeline>`__

-`ConfigureInferencePipeline<#configure-inference-pipeline>`__

-`Text-to-imagegeneration<#text-to-image-generation>`__
-`Quantization<#quantization>`__

-`Preparecalibrationdataset<#prepare-calibration-dataset>`__
-`Runquantization<#run-quantization>`__
-`CompareinferencetimeoftheFP16andINT8
models<#compare-inference-time-of-the-fp16-and-int8-models>`__

-`CompareUNetfilesize<#compare-unet-file-size>`__

-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"torch>=2.1"--index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2023.1.0"transformers"diffusers>=0.23.1"pillow"gradio>=4.19""nncf>=2.7.0""datasets>=2.14.6""peft==0.6.2"--extra-index-urlhttps://download.pytorch.org/whl/cpu

PreparemodelsforOpenVINOformatconversion
---------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Inthistutorialwewilluse
`LCM_Dreamshaper_v7<https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7>`__
from`HuggingFacehub<https://huggingface.co/>`__.Thismodeldistilled
from`Dreamshaperv7<https://huggingface.co/Lykon/dreamshaper-7>`__
fine-tuneof`Stable-Diffusion
v1-5<https://huggingface.co/runwayml/stable-diffusion-v1-5>`__using
LatentConsistencyDistillation(LCD)approachdiscussedabove.This
modelisalsointegratedinto
`Diffusers<https://huggingface.co/docs/diffusers/index>`__library.
Diffusersisthego-tolibraryforstate-of-the-artpretraineddiffusion
modelsforgeneratingimages,audio,andeven3Dstructuresof
molecules.ThisallowsustocomparerunningoriginalStableDiffusion
(fromthis
`notebook<stable-diffusion-text-to-image-with-output.html>`__)
anddistilledusingLCD.Thedistillationapproachefficientlyconverts
apre-trainedguideddiffusionmodelintoalatentconsistencymodelby
solvinganaugmentedPF-ODE.

ForstartingworkwithLCM,weshouldinstantiategenerationpipeline
first.``DiffusionPipeline.from_pretrained``methoddownloadall
pipelinecomponentsforLCMandconfigurethem.Thismodelusescustom
inferencepipelinestoredaspartofmodelrepository,wealsoshould
providewhichmoduleshouldbeloadedforinitializationusing
``custom_pipeline``argumentandrevisionforit.

..code::ipython3

importgc
importwarnings
frompathlibimportPath
fromdiffusersimportDiffusionPipeline
importnumpyasnp


warnings.filterwarnings("ignore")

TEXT_ENCODER_OV_PATH=Path("model/text_encoder.xml")
UNET_OV_PATH=Path("model/unet.xml")
VAE_DECODER_OV_PATH=Path("model/vae_decoder.xml")


defload_orginal_pytorch_pipeline_componets(skip_models=False,skip_safety_checker=False):
pipe=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
scheduler=pipe.scheduler
tokenizer=pipe.tokenizer
feature_extractor=pipe.feature_extractorifnotskip_safety_checkerelseNone
safety_checker=pipe.safety_checkerifnotskip_safety_checkerelseNone
text_encoder,unet,vae=None,None,None
ifnotskip_models:
text_encoder=pipe.text_encoder
text_encoder.eval()
unet=pipe.unet
unet.eval()
vae=pipe.vae
vae.eval()
delpipe
gc.collect()
return(
scheduler,
tokenizer,
feature_extractor,
safety_checker,
text_encoder,
unet,
vae,
)

..code::ipython3

skip_conversion=TEXT_ENCODER_OV_PATH.exists()andUNET_OV_PATH.exists()andVAE_DECODER_OV_PATH.exists()

(
scheduler,
tokenizer,
feature_extractor,
safety_checker,
text_encoder,
unet,
vae,
)=load_orginal_pytorch_pipeline_componets(skip_conversion)



..parsed-literal::

Fetching15files:0%||0/15[00:00<?,?it/s]



..parsed-literal::

diffusion_pytorch_model.safetensors:0%||0.00/3.44G[00:00<?,?B/s]



..parsed-literal::

model.safetensors:0%||0.00/1.22G[00:00<?,?B/s]



..parsed-literal::

model.safetensors:0%||0.00/492M[00:00<?,?B/s]



..parsed-literal::

Loadingpipelinecomponents...:0%||0/7[00:00<?,?it/s]


ConvertmodelstoOpenVINOformat
---------------------------------

`backtotop⬆️<#table-of-contents>`__

Startingfrom2023.0release,OpenVINOsupportsPyTorchmodelsdirectly
viaModelConversionAPI.``ov.convert_model``functionacceptsinstance
ofPyTorchmodelandexampleinputsfortracingandreturnsobjectof
``ov.Model``class,readytouseorsaveondiskusing``ov.save_model``
function.

LikeoriginalStableDiffusionpipeline,theLCMpipelineconsistsof
threeimportantparts:

-TextEncodertocreateconditiontogenerateanimagefromatext
prompt.
-U-Netforstep-by-stepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)fordecodinglatentspacetoimage.

Letusconverteachpart:

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

importtorch
importopenvinoasov


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


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
(-1,77),
],
)
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
gc.collect()
print(f"TextEncodersuccessfullyconvertedtoIRandsavedto{ir_path}")


ifnotTEXT_ENCODER_OV_PATH.exists():
convert_encoder(text_encoder,TEXT_ENCODER_OV_PATH)
else:
print(f"Textencoderwillbeloadedfrom{TEXT_ENCODER_OV_PATH}")

deltext_encoder
gc.collect()


..parsed-literal::

Textencoderwillbeloadedfrommodel/text_encoder.xml




..parsed-literal::

9



U-Net
~~~~~

`backtotop⬆️<#table-of-contents>`__

U-Netmodel,similartoStableDiffusionUNetmodel,hasfourinputs:

-``sample``-latentimagesamplefrompreviousstep.Generation
processhasnotbeenstartedyet,soyouwilluserandomnoise.
-``timestep``-currentschedulerstep.
-``encoder_hidden_state``-hiddenstateoftextencoder.
-``timestep_cond``-timestepconditionforgeneration.Thisinputis
notpresentinoriginalStableDiffusionU-Netmodelandintroduced
byLCMforimprovinggenerationqualityusingClassifier-Free
Guidance.`Classifier-freeguidance
(CFG)<https://arxiv.org/abs/2207.12598>`__iscrucialfor
synthesizinghigh-qualitytext-alignedimagesinStableDiffusion,
becauseitcontrolshowsimilarthegeneratedimagewillbetothe
prompt.InLatentConsistencyModels,CFGservesasaugmentation
parameterforPF-ODE.

Modelpredictsthe``sample``stateforthenextstep.

..code::ipython3

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
dummy_inputs={
"sample":torch.randn((1,4,64,64)),
"timestep":torch.ones([1]).to(torch.float32),
"encoder_hidden_states":torch.randn((1,77,768)),
"timestep_cond":torch.randn((1,256)),
}
unet.eval()
withtorch.no_grad():
ov_model=ov.convert_model(unet,example_input=dummy_inputs)
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
gc.collect()
print(f"UnetsuccessfullyconvertedtoIRandsavedto{ir_path}")


ifnotUNET_OV_PATH.exists():
convert_unet(unet,UNET_OV_PATH)
else:
print(f"Unetwillbeloadedfrom{UNET_OV_PATH}")
delunet
gc.collect()


..parsed-literal::

UnetsuccessfullyconvertedtoIRandsavedtomodel/unet.xml




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

Inourinferencepipeline,wewillnotuseVAEencoderpartandskipits
conversionforreducingmemoryconsumption.Theprocessofconversion
VAEencoder,canbefoundinStableDiffusionnotebook.

..code::ipython3

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
ov_model=ov.convert_model(vae_decoder,example_input=latents)
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
print(f"VAEdecodersuccessfullyconvertedtoIRandsavedto{ir_path}")


ifnotVAE_DECODER_OV_PATH.exists():
convert_vae_decoder(vae,VAE_DECODER_OV_PATH)
else:
print(f"VAEdecoderwillbeloadedfrom{VAE_DECODER_OV_PATH}")

delvae
gc.collect()


..parsed-literal::

VAEdecoderwillbeloadedfrommodel/vae_decoder.xml




..parsed-literal::

0



Prepareinferencepipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

Puttingitalltogether,letusnowtakeacloserlookathowthemodel
worksininferencebyillustratingthelogicalflow.

..figure::https://user-images.githubusercontent.com/29454499/277402235-079bacfb-3b6d-424b-8d47-5ddf601e1639.png
:alt:lcm-pipeline

lcm-pipeline

Thepipelinetakesalatentimagerepresentationandatextpromptis
transformedtotextembeddingviaCLIP’stextencoderasaninput.The
initiallatentimagerepresentationgeneratedusingrandomnoise
generator.Indifference,withoriginalStableDiffusionpipeline,LCM
alsousesguidancescaleforgettingtimestepconditionalembeddingsas
inputfordiffusionprocess,whileinStableDiffusion,itusedfor
scalingoutputlatents.

Next,theU-Netiteratively*denoises*therandomlatentimage
representationswhilebeingconditionedonthetextembeddings.The
outputoftheU-Net,beingthenoiseresidual,isusedtocomputea
denoisedlatentimagerepresentationviaascheduleralgorithm.LCM
introducesownschedulingalgorithmthatextendsthedenoisingprocedure
introducedindenoisingdiffusionprobabilisticmodels(DDPMs)with
non-Markovianguidance.The*denoising*processisrepeatedgivennumber
oftimes(bydefault50inoriginalSDpipeline,butforLCMsmall
numberofstepsrequired~2-8)tostep-by-stepretrievebetterlatent
imagerepresentations.Whencomplete,thelatentimagerepresentationis
decodedbythedecoderpartofthevariationalautoencoder.

..code::ipython3

fromtypingimportUnion,Optional,Any,List,Dict
fromtransformersimportCLIPTokenizer,CLIPImageProcessor
fromdiffusers.pipelines.stable_diffusion.safety_checkerimport(
StableDiffusionSafetyChecker,
)
fromdiffusers.pipelines.stable_diffusionimportStableDiffusionPipelineOutput
fromdiffusers.image_processorimportVaeImageProcessor


classOVLatentConsistencyModelPipeline(DiffusionPipeline):
def__init__(
self,
vae_decoder:ov.Model,
text_encoder:ov.Model,
tokenizer:CLIPTokenizer,
unet:ov.Model,
scheduler:None,
safety_checker:StableDiffusionSafetyChecker,
feature_extractor:CLIPImageProcessor,
requires_safety_checker:bool=True,
):
super().__init__()
self.vae_decoder=vae_decoder
self.text_encoder=text_encoder
self.tokenizer=tokenizer
self.register_to_config(unet=unet)
self.scheduler=scheduler
self.safety_checker=safety_checker
self.feature_extractor=feature_extractor
self.vae_scale_factor=2**3
self.image_processor=VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

def_encode_prompt(
self,
prompt,
num_images_per_prompt,
prompt_embeds:None,
):
r"""
Encodesthepromptintotextencoderhiddenstates.
Args:
prompt(`str`or`List[str]`,*optional*):
prompttobeencoded
num_images_per_prompt(`int`):
numberofimagesthatshouldbegeneratedperprompt
prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatedtextembeddings.Canbeusedtoeasilytweaktextinputs,*e.g.*promptweighting.Ifnot
provided,textembeddingswillbegeneratedfrom`prompt`inputargument.
"""

ifprompt_embedsisNone:
text_inputs=self.tokenizer(
prompt,
padding="max_length",
max_length=self.tokenizer.model_max_length,
truncation=True,
return_tensors="pt",
)
text_input_ids=text_inputs.input_ids

prompt_embeds=self.text_encoder(text_input_ids,share_inputs=True,share_outputs=True)
prompt_embeds=torch.from_numpy(prompt_embeds[0])

bs_embed,seq_len,_=prompt_embeds.shape
#duplicatetextembeddingsforeachgenerationperprompt
prompt_embeds=prompt_embeds.repeat(1,num_images_per_prompt,1)
prompt_embeds=prompt_embeds.view(bs_embed*num_images_per_prompt,seq_len,-1)

#Don'tneedtogetuncondpromptembeddingbecauseofLCMGuidedDistillation
returnprompt_embeds

defrun_safety_checker(self,image,dtype):
ifself.safety_checkerisNone:
has_nsfw_concept=None
else:
iftorch.is_tensor(image):
feature_extractor_input=self.image_processor.postprocess(image,output_type="pil")
else:
feature_extractor_input=self.image_processor.numpy_to_pil(image)
safety_checker_input=self.feature_extractor(feature_extractor_input,return_tensors="pt")
image,has_nsfw_concept=self.safety_checker(images=image,clip_input=safety_checker_input.pixel_values.to(dtype))
returnimage,has_nsfw_concept

defprepare_latents(self,batch_size,num_channels_latents,height,width,dtype,latents=None):
shape=(
batch_size,
num_channels_latents,
height//self.vae_scale_factor,
width//self.vae_scale_factor,
)
iflatentsisNone:
latents=torch.randn(shape,dtype=dtype)
#scaletheinitialnoisebythestandarddeviationrequiredbythescheduler
latents=latents*self.scheduler.init_noise_sigma
returnlatents

defget_w_embedding(self,w,embedding_dim=512,dtype=torch.float32):
"""
seehttps://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
Args:
timesteps:torch.Tensor:generateembeddingvectorsatthesetimesteps
embedding_dim:int:dimensionoftheembeddingstogenerate
dtype:datatypeofthegeneratedembeddings
Returns:
embeddingvectorswithshape`(len(timesteps),embedding_dim)`
"""
assertlen(w.shape)==1
w=w*1000.0

half_dim=embedding_dim//2
emb=torch.log(torch.tensor(10000.0))/(half_dim-1)
emb=torch.exp(torch.arange(half_dim,dtype=dtype)*-emb)
emb=w.to(dtype)[:,None]*emb[None,:]
emb=torch.cat([torch.sin(emb),torch.cos(emb)],dim=1)
ifembedding_dim%2==1:#zeropad
emb=torch.nn.functional.pad(emb,(0,1))
assertemb.shape==(w.shape[0],embedding_dim)
returnemb

@torch.no_grad()
def__call__(
self,
prompt:Union[str,List[str]]=None,
height:Optional[int]=512,
width:Optional[int]=512,
guidance_scale:float=7.5,
num_images_per_prompt:Optional[int]=1,
latents:Optional[torch.FloatTensor]=None,
num_inference_steps:int=4,
lcm_origin_steps:int=50,
prompt_embeds:Optional[torch.FloatTensor]=None,
output_type:Optional[str]="pil",
return_dict:bool=True,
cross_attention_kwargs:Optional[Dict[str,Any]]=None,
):
#1.Definecallparameters
ifpromptisnotNoneandisinstance(prompt,str):
batch_size=1
elifpromptisnotNoneandisinstance(prompt,list):
batch_size=len(prompt)
else:
batch_size=prompt_embeds.shape[0]

#do_classifier_free_guidance=guidance_scale>0.0
#InLCMImplementation:cfg_noise=noise_cond+cfg_scale*(noise_cond-noise_uncond),(cfg_scale>0.0usingCFG)

#2.Encodeinputprompt
prompt_embeds=self._encode_prompt(
prompt,
num_images_per_prompt,
prompt_embeds=prompt_embeds,
)

#3.Preparetimesteps
self.scheduler.set_timesteps(num_inference_steps,original_inference_steps=lcm_origin_steps)
timesteps=self.scheduler.timesteps

#4.Preparelatentvariable
num_channels_latents=4
latents=self.prepare_latents(
batch_size*num_images_per_prompt,
num_channels_latents,
height,
width,
prompt_embeds.dtype,
latents,
)

bs=batch_size*num_images_per_prompt

#5.GetGuidanceScaleEmbedding
w=torch.tensor(guidance_scale).repeat(bs)
w_embedding=self.get_w_embedding(w,embedding_dim=256)

#6.LCMMultiStepSamplingLoop:
withself.progress_bar(total=num_inference_steps)asprogress_bar:
fori,tinenumerate(timesteps):
ts=torch.full((bs,),t,dtype=torch.long)

#modelprediction(v-prediction,eps,x)
model_pred=self.unet(
[latents,ts,prompt_embeds,w_embedding],
share_inputs=True,
share_outputs=True,
)[0]

#computethepreviousnoisysamplex_t->x_t-1
latents,denoised=self.scheduler.step(torch.from_numpy(model_pred),t,latents,return_dict=False)
progress_bar.update()

ifnotoutput_type=="latent":
image=torch.from_numpy(self.vae_decoder(denoised/0.18215,share_inputs=True,share_outputs=True)[0])
image,has_nsfw_concept=self.run_safety_checker(image,prompt_embeds.dtype)
else:
image=denoised
has_nsfw_concept=None

ifhas_nsfw_conceptisNone:
do_denormalize=[True]*image.shape[0]
else:
do_denormalize=[nothas_nsfwforhas_nsfwinhas_nsfw_concept]

image=self.image_processor.postprocess(image,output_type=output_type,do_denormalize=do_denormalize)

ifnotreturn_dict:
return(image,has_nsfw_concept)

returnStableDiffusionPipelineOutput(images=image,nsfw_content_detected=has_nsfw_concept)

ConfigureInferencePipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

First,youshouldcreateinstancesofOpenVINOModelandcompileit
usingselecteddevice.Selectdevicefromdropdownlistforrunning
inferenceusingOpenVINO.

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

text_enc=core.compile_model(TEXT_ENCODER_OV_PATH,device.value)
unet_model=core.compile_model(UNET_OV_PATH,device.value)

ov_config={"INFERENCE_PRECISION_HINT":"f32"}ifdevice.value!="CPU"else{}

vae_decoder=core.compile_model(VAE_DECODER_OV_PATH,device.value,ov_config)

Modeltokenizerandschedulerarealsoimportantpartsofthepipeline.
ThispipelineisalsocanuseSafetyChecker,thefilterfordetecting
thatcorrespondinggeneratedimagecontains“not-safe-for-work”(nsfw)
content.Theprocessofnsfwcontentdetectionrequirestoobtainimage
embeddingsusingCLIPmodel,soadditionallyfeatureextractorcomponent
shouldbeaddedinthepipeline.Wereusetokenizer,featureextractor,
schedulerandsafetycheckerfromoriginalLCMpipeline.

..code::ipython3

ov_pipe=OVLatentConsistencyModelPipeline(
tokenizer=tokenizer,
text_encoder=text_enc,
unet=unet_model,
vae_decoder=vae_decoder,
scheduler=scheduler,
feature_extractor=feature_extractor,
safety_checker=safety_checker,
)

Text-to-imagegeneration
------------------------

`backtotop⬆️<#table-of-contents>`__

Now,let’sseemodelinaction

..code::ipython3

prompt="abeautifulpinkunicorn,8k"
num_inference_steps=4
torch.manual_seed(1234567)

images=ov_pipe(
prompt=prompt,
num_inference_steps=num_inference_steps,
guidance_scale=8.0,
lcm_origin_steps=50,
output_type="pil",
height=512,
width=512,
).images



..parsed-literal::

0%||0/4[00:00<?,?it/s]


..code::ipython3

images[0]




..image::latent-consistency-models-image-generation-with-output_files/latent-consistency-models-image-generation-with-output_21_0.png



Nice.Asyoucansee,thepicturehasquiteahighdefinition🔥.

Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``LatentConsistencyModelPipeline``structure,UNetusedfor
iterativedenoisingofinput.Itmeansthatmodelrunsinthecycle
repeatinginferenceoneachdiffusionstep,whileotherpartsof
pipelinetakepartonlyonce.Thatiswhycomputationcostandspeedof
UNetdenoisingbecomesthecriticalpathinthepipeline.Quantizingthe
restoftheSDpipelinedoesnotsignificantlyimproveinference
performancebutcanleadtoasubstantialdegradationofaccuracy.

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

int8_pipe=None

#Fetch`skip_kernel_extension`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)
%load_extskip_kernel_extension

Preparecalibrationdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseaportionof
`conceptual_captions<https://huggingface.co/datasets/google-research-datasets/conceptual_captions>`__
datasetfromHuggingFaceascalibrationdata.Tocollectintermediate
modelinputsforcalibrationweshouldcustomize``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

importdatasets
fromtqdm.notebookimporttqdm
fromtransformersimportset_seed
fromtypingimportAny,Dict,List

set_seed(1)

classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model,prob:float,data_cache:List[Any]=None):
super().__init__(compiled_model)
self.data_cache=data_cacheifdata_cacheelse[]
self.prob=np.clip(prob,0,1)

def__call__(self,*args,**kwargs):
ifnp.random.rand()>=self.prob:
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)

defcollect_calibration_data(lcm_pipeline:OVLatentConsistencyModelPipeline,subset_size:int)->List[Dict]:
original_unet=lcm_pipeline.unet
lcm_pipeline.unet=CompiledModelDecorator(original_unet,prob=0.3)

dataset=datasets.load_dataset("google-research-datasets/conceptual_captions",split="train",trust_remote_code=True).shuffle(seed=42)
lcm_pipeline.set_progress_bar_config(disable=True)
safety_checker=lcm_pipeline.safety_checker
lcm_pipeline.safety_checker=None

#Runinferencefordatacollection
pbar=tqdm(total=subset_size)
diff=0
forbatchindataset:
prompt=batch["caption"]
iflen(prompt)>tokenizer.model_max_length:
continue
_=lcm_pipeline(
prompt,
num_inference_steps=num_inference_steps,
guidance_scale=8.0,
lcm_origin_steps=50,
output_type="pil",
height=512,
width=512,
)
collected_subset_size=len(lcm_pipeline.unet.data_cache)
ifcollected_subset_size>=subset_size:
pbar.update(subset_size-pbar.n)
break
pbar.update(collected_subset_size-diff)
diff=collected_subset_size

calibration_dataset=lcm_pipeline.unet.data_cache
lcm_pipeline.set_progress_bar_config(disable=False)
lcm_pipeline.unet=original_unet
lcm_pipeline.safety_checker=safety_checker
returncalibration_dataset

..code::ipython3

%%skipnot$to_quantize.value

importlogging
logging.basicConfig(level=logging.WARNING)
logger=logging.getLogger(__name__)

UNET_INT8_OV_PATH=Path("model/unet_int8.xml")
ifnotUNET_INT8_OV_PATH.exists():
subset_size=200
unet_calibration_data=collect_calibration_data(ov_pipe,subset_size=subset_size)



..parsed-literal::

0%||0/200[00:00<?,?it/s]


Runquantization
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Createaquantizedmodelfromthepre-trainedconvertedOpenVINOmodel.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

%%skipnot$to_quantize.value

importnncf
fromnncf.scopesimportIgnoredScope

ifUNET_INT8_OV_PATH.exists():
print("Loadingquantizedmodel")
quantized_unet=core.read_model(UNET_INT8_OV_PATH)
else:
unet=core.read_model(UNET_OV_PATH)
quantized_unet=nncf.quantize(
model=unet,
subset_size=subset_size,
calibration_dataset=nncf.Dataset(unet_calibration_data),
model_type=nncf.ModelType.TRANSFORMER,
advanced_parameters=nncf.AdvancedQuantizationParameters(
disable_bias_correction=True
)
)
ov.save_model(quantized_unet,UNET_INT8_OV_PATH)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,onnx,openvino



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>




..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:122ignorednodeswerefoundbynameintheNNCFGraph



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..code::ipython3

%%skipnot$to_quantize.value

unet_optimized=core.compile_model(UNET_INT8_OV_PATH,device.value)

int8_pipe=OVLatentConsistencyModelPipeline(
tokenizer=tokenizer,
text_encoder=text_enc,
unet=unet_optimized,
vae_decoder=vae_decoder,
scheduler=scheduler,
feature_extractor=feature_extractor,
safety_checker=safety_checker,
)

LetuscheckpredictionswiththequantizedUNetusingthesameinput
data.

..code::ipython3

%%skipnot$to_quantize.value

fromIPython.displayimportdisplay

prompt="abeautifulpinkunicorn,8k"
num_inference_steps=4
torch.manual_seed(1234567)

images=int8_pipe(
prompt=prompt,
num_inference_steps=num_inference_steps,
guidance_scale=8.0,
lcm_origin_steps=50,
output_type="pil",
height=512,
width=512,
).images

display(images[0])



..parsed-literal::

0%||0/4[00:00<?,?it/s]



..image::latent-consistency-models-image-generation-with-output_files/latent-consistency-models-image-generation-with-output_34_1.png


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
calibration_dataset=datasets.load_dataset("google-research-datasets/conceptual_captions",split="train",trust_remote_code=True)
validation_data=[]
foridx,batchinenumerate(calibration_dataset):
ifidx>=validation_size:
break
prompt=batch["caption"]
validation_data.append(prompt)

defcalculate_inference_time(pipeline,calibration_dataset):
inference_time=[]
pipeline.set_progress_bar_config(disable=True)
foridx,promptinenumerate(validation_data):
start=time.perf_counter()
_=pipeline(
prompt,
num_inference_steps=num_inference_steps,
guidance_scale=8.0,
lcm_origin_steps=50,
output_type="pil",
height=512,
width=512,
)
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
ifidx>=validation_size:
break
returnnp.median(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

fp_latency=calculate_inference_time(ov_pipe,validation_data)
int8_latency=calculate_inference_time(int8_pipe,validation_data)
print(f"Performancespeedup:{fp_latency/int8_latency:.3f}")


..parsed-literal::

Performancespeedup:1.319


CompareUNetfilesize
^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fp16_ir_model_size=UNET_OV_PATH.with_suffix(".bin").stat().st_size/1024
quantized_model_size=UNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size/1024

print(f"FP16modelsize:{fp16_ir_model_size:.2f}KB")
print(f"INT8modelsize:{quantized_model_size:.2f}KB")
print(f"Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16modelsize:1678912.37KB
INT8modelsize:840792.93KB
Modelcompressionrate:1.997


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importrandom
importgradioasgr
fromfunctoolsimportpartial

MAX_SEED=np.iinfo(np.int32).max

examples=[
"portraitphotoofagirl,photograph,highlydetailedface,depthoffield,moodylight,goldenhour,"
"stylebyDanWinters,RussellJames,SteveMcCurry,centered,extremelydetailed,NikonD850,awardwinningphotography",
"Self-portraitoilpainting,abeautifulcyborgwithgoldenhair,8k",
"Astronautinajungle,coldcolorpalette,mutedcolors,detailed,8k",
"Aphotoofbeautifulmountainwithrealisticsunsetandbluelake,highlydetailed,masterpiece",
]


defrandomize_seed_fn(seed:int,randomize_seed:bool)->int:
ifrandomize_seed:
seed=random.randint(0,MAX_SEED)
returnseed


MAX_IMAGE_SIZE=768


defgenerate(
pipeline:OVLatentConsistencyModelPipeline,
prompt:str,
seed:int=0,
width:int=512,
height:int=512,
guidance_scale:float=8.0,
num_inference_steps:int=4,
randomize_seed:bool=False,
num_images:int=1,
progress=gr.Progress(track_tqdm=True),
):
seed=randomize_seed_fn(seed,randomize_seed)
torch.manual_seed(seed)
result=pipeline(
prompt=prompt,
width=width,
height=height,
guidance_scale=guidance_scale,
num_inference_steps=num_inference_steps,
num_images_per_prompt=num_images,
lcm_origin_steps=50,
output_type="pil",
).images[0]
returnresult,seed


generate_original=partial(generate,ov_pipe)
generate_optimized=partial(generate,int8_pipe)
quantized_model_present=int8_pipeisnotNone

withgr.Blocks()asdemo:
withgr.Group():
withgr.Row():
prompt=gr.Text(
label="Prompt",
show_label=False,
max_lines=1,
placeholder="Enteryourprompt",
container=False,
)
withgr.Row():
withgr.Column():
result=gr.Image(
label="Result(Original)"ifquantized_model_presentelse"Image",
type="pil",
)
run_button=gr.Button("Run")
withgr.Column(visible=quantized_model_present):
result_optimized=gr.Image(
label="Result(Optimized)",
type="pil",
visible=quantized_model_present,
)
run_quantized_button=gr.Button(value="Runquantized",visible=quantized_model_present)

withgr.Accordion("Advancedoptions",open=False):
seed=gr.Slider(label="Seed",minimum=0,maximum=MAX_SEED,step=1,value=0,randomize=True)
randomize_seed=gr.Checkbox(label="Randomizeseedacrossruns",value=True)
withgr.Row():
width=gr.Slider(
label="Width",
minimum=256,
maximum=MAX_IMAGE_SIZE,
step=32,
value=512,
)
height=gr.Slider(
label="Height",
minimum=256,
maximum=MAX_IMAGE_SIZE,
step=32,
value=512,
)
withgr.Row():
guidance_scale=gr.Slider(
label="Guidancescaleforbase",
minimum=2,
maximum=14,
step=0.1,
value=8.0,
)
num_inference_steps=gr.Slider(
label="Numberofinferencestepsforbase",
minimum=1,
maximum=8,
step=1,
value=4,
)

gr.Examples(
examples=examples,
inputs=prompt,
outputs=result,
cache_examples=False,
)

gr.on(
triggers=[
prompt.submit,
run_button.click,
],
fn=generate_original,
inputs=[
prompt,
seed,
width,
height,
guidance_scale,
num_inference_steps,
randomize_seed,
],
outputs=[result,seed],
)

ifquantized_model_present:
gr.on(
triggers=[
prompt.submit,
run_quantized_button.click,
],
fn=generate_optimized,
inputs=[
prompt,
seed,
width,
height,
guidance_scale,
num_inference_steps,
randomize_seed,
],
outputs=[result_optimized,seed],
)

..code::ipython3

try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
