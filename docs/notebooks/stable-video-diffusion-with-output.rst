ImagetoVideoGenerationwithStableVideoDiffusion
=====================================================

StableVideoDiffusion(SVD)Image-to-Videoisadiffusionmodelthat
takesinastillimageasaconditioningframe,andgeneratesavideo
fromit.InthistutorialweconsiderhowtoconvertandrunStable
VideoDiffusionusingOpenVINO.Wewilluse
`stable-video-diffusion-img2video-xt<https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt>`__
modelasexample.Additionally,tospeedupvideogenerationprocesswe
apply`AnimateLCM<https://arxiv.org/abs/2402.00769>`__LoRAweightsand
runoptimizationwith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__.

Tableofcontents:
------------------

-`Prerequisites<#prerequisites>`__
-`DownloadPyTorchModel<#download-pytorch-model>`__
-`ConvertModeltoOpenVINOIntermediate
Representation<#convert-model-to-openvino-intermediate-representation>`__

-`ImageEncoder<#image-encoder>`__
-`U-net<#u-net>`__
-`VAEEncoderandDecoder<#vae-encoder-and-decoder>`__

-`PrepareInferencePipeline<#prepare-inference-pipeline>`__
-`RunVideoGeneration<#run-video-generation>`__

-`SelectInferenceDevice<#select-inference-device>`__

-`Quantization<#quantization>`__

-`Preparecalibrationdataset<#prepare-calibration-dataset>`__
-`RunHybridModelQuantization<#run-hybrid-model-quantization>`__
-`RunWeightCompression<#run-weight-compression>`__
-`Comparemodelfilesizes<#compare-model-file-sizes>`__
-`CompareinferencetimeoftheFP16andINT8
pipelines<#compare-inference-time-of-the-fp16-and-int8-pipelines>`__

-`InteractiveDemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"torch>=2.1""diffusers>=0.25""peft==0.6.2""transformers""openvino>=2024.1.0"Pillowopencv-pythontqdm"gradio>=4.19"safetensors--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-qdatasets"nncf>=2.10.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


DownloadPyTorchModel
----------------------

`backtotop⬆️<#table-of-contents>`__

ThecodebelowloadStableVideoDiffusionXTmodelusing
`Diffusers<https://huggingface.co/docs/diffusers/index>`__libraryand
applyConsistencyDistilledAnimateLCMweights.

..code::ipython3

importtorch
frompathlibimportPath
fromdiffusersimportStableVideoDiffusionPipeline
fromdiffusers.utilsimportload_image,export_to_video
fromdiffusers.models.attention_processorimportAttnProcessor
fromsafetensorsimportsafe_open
importgc
importrequests

lcm_scheduler_url="https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/raw/main/lcm_scheduler.py"

r=requests.get(lcm_scheduler_url)

withopen("lcm_scheduler.py","w")asf:
f.write(r.text)

fromlcm_schedulerimportAnimateLCMSVDStochasticIterativeScheduler
fromhuggingface_hubimporthf_hub_download

MODEL_DIR=Path("model")

IMAGE_ENCODER_PATH=MODEL_DIR/"image_encoder.xml"
VAE_ENCODER_PATH=MODEL_DIR/"vae_encoder.xml"
VAE_DECODER_PATH=MODEL_DIR/"vae_decoder.xml"
UNET_PATH=MODEL_DIR/"unet.xml"


load_pt_pipeline=not(VAE_ENCODER_PATH.exists()andVAE_DECODER_PATH.exists()andUNET_PATH.exists()andIMAGE_ENCODER_PATH.exists())

unet,vae,image_encoder=None,None,None
ifload_pt_pipeline:
noise_scheduler=AnimateLCMSVDStochasticIterativeScheduler(
num_train_timesteps=40,
sigma_min=0.002,
sigma_max=700.0,
sigma_data=1.0,
s_noise=1.0,
rho=7,
clip_denoised=False,
)
pipe=StableVideoDiffusionPipeline.from_pretrained(
"stabilityai/stable-video-diffusion-img2vid-xt",
variant="fp16",
scheduler=noise_scheduler,
)
pipe.unet.set_attn_processor(AttnProcessor())
hf_hub_download(
repo_id="wangfuyun/AnimateLCM-SVD-xt",
filename="AnimateLCM-SVD-xt.safetensors",
local_dir="./checkpoints",
)
state_dict={}
LCM_LORA_PATH=Path(
"checkpoints/AnimateLCM-SVD-xt.safetensors",
)
withsafe_open(LCM_LORA_PATH,framework="pt",device="cpu")asf:
forkeyinf.keys():
state_dict[key]=f.get_tensor(key)
missing,unexpected=pipe.unet.load_state_dict(state_dict,strict=True)

pipe.scheduler.save_pretrained(MODEL_DIR/"scheduler")
pipe.feature_extractor.save_pretrained(MODEL_DIR/"feature_extractor")
unet=pipe.unet
unet.eval()
vae=pipe.vae
vae.eval()
image_encoder=pipe.image_encoder
image_encoder.eval()
delpipe
gc.collect()

#Loadtheconditioningimage
image=load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
image=image.resize((512,256))

ConvertModeltoOpenVINOIntermediateRepresentation
-----------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchmodelsviaconversionintoIntermediate
Representation(IR)format.Weneedtoprovideamodelobject,input
dataformodeltracingto``ov.convert_model``functiontoobtain
OpenVINO``ov.Model``objectinstance.Modelcanbesavedondiskfor
nextdeploymentusing``ov.save_model``function.

StableVideoDiffusionconsistsof3parts:

-**ImageEncoder**forextractionembeddingsfromtheinputimage.
-**U-Net**forstep-by-stepdenoisingvideoclip.
-**VAE**forencodinginputimageintolatentspaceanddecoding
generatedvideo.

Let’sconverteachpart.

ImageEncoder
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importopenvinoasov


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


ifnotIMAGE_ENCODER_PATH.exists():
withtorch.no_grad():
ov_model=ov.convert_model(
image_encoder,
example_input=torch.zeros((1,3,224,224)),
input=[-1,3,224,224],
)
ov.save_model(ov_model,IMAGE_ENCODER_PATH)
delov_model
cleanup_torchscript_cache()
print(f"ImageEncodersuccessfullyconvertedtoIRandsavedto{IMAGE_ENCODER_PATH}")
delimage_encoder
gc.collect();

U-net
~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotUNET_PATH.exists():
unet_inputs={
"sample":torch.ones([2,2,8,32,32]),
"timestep":torch.tensor(1.256),
"encoder_hidden_states":torch.zeros([2,1,1024]),
"added_time_ids":torch.ones([2,3]),
}
withtorch.no_grad():
ov_model=ov.convert_model(unet,example_input=unet_inputs)
ov.save_model(ov_model,UNET_PATH)
delov_model
cleanup_torchscript_cache()
print(f"UNetsuccessfullyconvertedtoIRandsavedto{UNET_PATH}")

delunet
gc.collect();

VAEEncoderandDecoder
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

AsdiscussedaboveVAEmodelusedforencodinginitialimageand
decodinggeneratedvideo.EncodingandDecodinghappenondifferent
pipelinestages,soforconvenientusageweseparateVAEon2parts:
EncoderandDecoder.

..code::ipython3

classVAEEncoderWrapper(torch.nn.Module):
def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,image):
returnself.vae.encode(x=image)["latent_dist"].sample()


classVAEDecoderWrapper(torch.nn.Module):
def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,latents,num_frames:int):
returnself.vae.decode(latents,num_frames=num_frames)


ifnotVAE_ENCODER_PATH.exists():
vae_encoder=VAEEncoderWrapper(vae)
withtorch.no_grad():
ov_model=ov.convert_model(vae_encoder,example_input=torch.zeros((1,3,576,1024)))
ov.save_model(ov_model,VAE_ENCODER_PATH)
cleanup_torchscript_cache()
print(f"VAEEncodersuccessfullyconvertedtoIRandsavedto{VAE_ENCODER_PATH}")
delvae_encoder
gc.collect()

ifnotVAE_DECODER_PATH.exists():
vae_decoder=VAEDecoderWrapper(vae)
withtorch.no_grad():
ov_model=ov.convert_model(vae_decoder,example_input=(torch.zeros((8,4,72,128)),torch.tensor(8)))
ov.save_model(ov_model,VAE_DECODER_PATH)
cleanup_torchscript_cache()
print(f"VAEDecodersuccessfullyconvertedtoIRandsavedto{VAE_ENCODER_PATH}")
delvae_decoder
gc.collect()

delvae
gc.collect();

PrepareInferencePipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

Thecodebellowimplements``OVStableVideoDiffusionPipeline``classfor
runningvideogenerationusingOpenVINO.Thepipelineacceptsinput
imageandreturnsthesequenceofgeneratedframesThediagrambelow
representsasimplifiedpipelineworkflow.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/a5671c5b-415b-4ae0-be82-9bf36527d452
:alt:svd

svd

Thepipelineisverysimilarto`StableDiffusionImagetoImage
Generation
pipeline<stable-diffusion-text-to-image-with-output.html>`__
withtheonlydifferencethatImageEncoderisusedinsteadofText
Encoder.Modeltakesinputimageandrandomseedasinitialprompt.Then
imageencodedintoembeddingsspaceusingImageEncoderandintolatent
spaceusingVAEEncoderandpassedasinputtoU-Netmodel.Next,the
U-Netiteratively*denoises*therandomlatentvideorepresentations
whilebeingconditionedontheimageembeddings.Theoutputofthe
U-Net,beingthenoiseresidual,isusedtocomputeadenoisedlatent
imagerepresentationviaascheduleralgorithmfornextiterationin
generationcycle.Thisprocessrepeatsthegivennumberoftimesand,
finally,VAEdecoderconvertsdenoisedlatentsintosequenceofvideo
frames.

..code::ipython3

fromdiffusers.pipelines.pipeline_utilsimportDiffusionPipeline
importPIL.Image
fromdiffusers.image_processorimportVaeImageProcessor
fromdiffusers.utils.torch_utilsimportrandn_tensor
fromtypingimportCallable,Dict,List,Optional,Union
fromdiffusers.pipelines.stable_video_diffusionimport(
StableVideoDiffusionPipelineOutput,
)


def_append_dims(x,target_dims):
"""Appendsdimensionstotheendofatensoruntilithastarget_dimsdimensions."""
dims_to_append=target_dims-x.ndim
ifdims_to_append<0:
raiseValueError(f"inputhas{x.ndim}dimsbuttarget_dimsis{target_dims},whichisless")
returnx[(...,)+(None,)*dims_to_append]


deftensor2vid(video:torch.Tensor,processor,output_type="np"):
#Basedon:
#https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

batch_size,channels,num_frames,height,width=video.shape
outputs=[]
forbatch_idxinrange(batch_size):
batch_vid=video[batch_idx].permute(1,0,2,3)
batch_output=processor.postprocess(batch_vid,output_type)

outputs.append(batch_output)

returnoutputs


classOVStableVideoDiffusionPipeline(DiffusionPipeline):
r"""
PipelinetogeneratevideofromaninputimageusingStableVideoDiffusion.

Thismodelinheritsfrom[`DiffusionPipeline`].Checkthesuperclassdocumentationforthegenericmethods
implementedforallpipelines(downloading,saving,runningonaparticulardevice,etc.).

Args:
vae([`AutoencoderKL`]):
VariationalAuto-Encoder(VAE)modeltoencodeanddecodeimagestoandfromlatentrepresentations.
image_encoder([`~transformers.CLIPVisionModelWithProjection`]):
FrozenCLIPimage-encoder([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
unet([`UNetSpatioTemporalConditionModel`]):
A`UNetSpatioTemporalConditionModel`todenoisetheencodedimagelatents.
scheduler([`EulerDiscreteScheduler`]):
Aschedulertobeusedincombinationwith`unet`todenoisetheencodedimagelatents.
feature_extractor([`~transformers.CLIPImageProcessor`]):
A`CLIPImageProcessor`toextractfeaturesfromgeneratedimages.
"""

def__init__(
self,
vae_encoder,
image_encoder,
unet,
vae_decoder,
scheduler,
feature_extractor,
):
super().__init__()
self.vae_encoder=vae_encoder
self.vae_decoder=vae_decoder
self.image_encoder=image_encoder
self.register_to_config(unet=unet)
self.scheduler=scheduler
self.feature_extractor=feature_extractor
self.vae_scale_factor=2**(4-1)
self.image_processor=VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

def_encode_image(self,image,device,num_videos_per_prompt,do_classifier_free_guidance):
dtype=torch.float32

ifnotisinstance(image,torch.Tensor):
image=self.image_processor.pil_to_numpy(image)
image=self.image_processor.numpy_to_pt(image)

#Wenormalizetheimagebeforeresizingtomatchwiththeoriginalimplementation.
#Thenweunnormalizeitafterresizing.
image=image*2.0-1.0
image=_resize_with_antialiasing(image,(224,224))
image=(image+1.0)/2.0

#NormalizetheimagewithforCLIPinput
image=self.feature_extractor(
images=image,
do_normalize=True,
do_center_crop=False,
do_resize=False,
do_rescale=False,
return_tensors="pt",
).pixel_values

image=image.to(device=device,dtype=dtype)
image_embeddings=torch.from_numpy(self.image_encoder(image)[0])
image_embeddings=image_embeddings.unsqueeze(1)

#duplicateimageembeddingsforeachgenerationperprompt,usingmpsfriendlymethod
bs_embed,seq_len,_=image_embeddings.shape
image_embeddings=image_embeddings.repeat(1,num_videos_per_prompt,1)
image_embeddings=image_embeddings.view(bs_embed*num_videos_per_prompt,seq_len,-1)

ifdo_classifier_free_guidance:
negative_image_embeddings=torch.zeros_like(image_embeddings)

#Forclassifierfreeguidance,weneedtodotwoforwardpasses.
#Hereweconcatenatetheunconditionalandtextembeddingsintoasinglebatch
#toavoiddoingtwoforwardpasses
image_embeddings=torch.cat([negative_image_embeddings,image_embeddings])
returnimage_embeddings

def_encode_vae_image(
self,
image:torch.Tensor,
device,
num_videos_per_prompt,
do_classifier_free_guidance,
):
image_latents=torch.from_numpy(self.vae_encoder(image)[0])

ifdo_classifier_free_guidance:
negative_image_latents=torch.zeros_like(image_latents)

#Forclassifierfreeguidance,weneedtodotwoforwardpasses.
#Hereweconcatenatetheunconditionalandtextembeddingsintoasinglebatch
#toavoiddoingtwoforwardpasses
image_latents=torch.cat([negative_image_latents,image_latents])

#duplicateimage_latentsforeachgenerationperprompt,usingmpsfriendlymethod
image_latents=image_latents.repeat(num_videos_per_prompt,1,1,1)

returnimage_latents

def_get_add_time_ids(
self,
fps,
motion_bucket_id,
noise_aug_strength,
dtype,
batch_size,
num_videos_per_prompt,
do_classifier_free_guidance,
):
add_time_ids=[fps,motion_bucket_id,noise_aug_strength]

passed_add_embed_dim=256*len(add_time_ids)
expected_add_embed_dim=3*256

ifexpected_add_embed_dim!=passed_add_embed_dim:
raiseValueError(
f"Modelexpectsanaddedtimeembeddingvectoroflength{expected_add_embed_dim},butavectorof{passed_add_embed_dim}wascreated.Themodelhasanincorrectconfig.Pleasecheck`unet.config.time_embedding_type`and`text_encoder_2.config.projection_dim`."
)

add_time_ids=torch.tensor([add_time_ids],dtype=dtype)
add_time_ids=add_time_ids.repeat(batch_size*num_videos_per_prompt,1)

ifdo_classifier_free_guidance:
add_time_ids=torch.cat([add_time_ids,add_time_ids])

returnadd_time_ids

defdecode_latents(self,latents,num_frames,decode_chunk_size=14):
#[batch,frames,channels,height,width]->[batch*frames,channels,height,width]
latents=latents.flatten(0,1)

latents=1/0.18215*latents

#decodedecode_chunk_sizeframesatatimetoavoidOOM
frames=[]
foriinrange(0,latents.shape[0],decode_chunk_size):
frame=torch.from_numpy(self.vae_decoder([latents[i:i+decode_chunk_size],num_frames])[0])
frames.append(frame)
frames=torch.cat(frames,dim=0)

#[batch*frames,channels,height,width]->[batch,channels,frames,height,width]
frames=frames.reshape(-1,num_frames,*frames.shape[1:]).permute(0,2,1,3,4)

#wealwayscasttofloat32asthisdoesnotcausesignificantoverheadandiscompatiblewithbfloat16
frames=frames.float()
returnframes

defcheck_inputs(self,image,height,width):
ifnotisinstance(image,torch.Tensor)andnotisinstance(image,PIL.Image.Image)andnotisinstance(image,list):
raiseValueError("`image`hastobeoftype`torch.FloatTensor`or`PIL.Image.Image`or`List[PIL.Image.Image]`butis"f"{type(image)}")

ifheight%8!=0orwidth%8!=0:
raiseValueError(f"`height`and`width`havetobedivisibleby8butare{height}and{width}.")

defprepare_latents(
self,
batch_size,
num_frames,
num_channels_latents,
height,
width,
dtype,
device,
generator,
latents=None,
):
shape=(
batch_size,
num_frames,
num_channels_latents//2,
height//self.vae_scale_factor,
width//self.vae_scale_factor,
)
ifisinstance(generator,list)andlen(generator)!=batch_size:
raiseValueError(
f"Youhavepassedalistofgeneratorsoflength{len(generator)},butrequestedaneffectivebatch"
f"sizeof{batch_size}.Makesurethebatchsizematchesthelengthofthegenerators."
)

iflatentsisNone:
latents=randn_tensor(shape,generator=generator,device=device,dtype=dtype)
else:
latents=latents.to(device)

#scaletheinitialnoisebythestandarddeviationrequiredbythescheduler
latents=latents*self.scheduler.init_noise_sigma
returnlatents

@torch.no_grad()
def__call__(
self,
image:Union[PIL.Image.Image,List[PIL.Image.Image],torch.FloatTensor],
height:int=320,
width:int=512,
num_frames:Optional[int]=8,
num_inference_steps:int=4,
min_guidance_scale:float=1.0,
max_guidance_scale:float=1.2,
fps:int=7,
motion_bucket_id:int=80,
noise_aug_strength:int=0.01,
decode_chunk_size:Optional[int]=None,
num_videos_per_prompt:Optional[int]=1,
generator:Optional[Union[torch.Generator,List[torch.Generator]]]=None,
latents:Optional[torch.FloatTensor]=None,
output_type:Optional[str]="pil",
callback_on_step_end:Optional[Callable[[int,int,Dict],None]]=None,
callback_on_step_end_tensor_inputs:List[str]=["latents"],
return_dict:bool=True,
):
r"""
Thecallfunctiontothepipelineforgeneration.

Args:discussed
image(`PIL.Image.Image`or`List[PIL.Image.Image]`or`torch.FloatTensor`):
Imageorimagestoguideimagegeneration.Ifyouprovideatensor,itneedstobecompatiblewith
[`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
height(`int`,*optional*,defaultsto`self.unet.config.sample_size*self.vae_scale_factor`):
Theheightinpixelsofthegeneratedimage.
width(`int`,*optional*,defaultsto`self.unet.config.sample_size*self.vae_scale_factor`):
Thewidthinpixelsofthegeneratedimage.
num_frames(`int`,*optional*):
Thenumberofvideoframestogenerate.Defaultsto14for`stable-video-diffusion-img2vid`andto25for`stable-video-diffusion-img2vid-xt`
num_inference_steps(`int`,*optional*,defaultsto25):


Thenumberofdenoisingsteps.Moredenoisingstepsusuallyleadtoahigherqualityimageatthe
expenseofslowerinference.Thisparameterismodulatedby`strength`.
min_guidance_scale(`float`,*optional*,defaultsto1.0):
Theminimumguidancescale.Usedfortheclassifierfreeguidancewithfirstframe.
max_guidance_scale(`float`,*optional*,defaultsto3.0):
Themaximumguidancescale.Usedfortheclassifierfreeguidancewithlastframe.
fps(`int`,*optional*,defaultsto7):
Framespersecond.Therateatwhichthegeneratedimagesshallbeexportedtoavideoaftergeneration.
NotethatStableDiffusionVideo'sUNetwasmicro-conditionedonfps-1duringtraining.
motion_bucket_id(`int`,*optional*,defaultsto127):
ThemotionbucketID.Usedasconditioningforthegeneration.Thehigherthenumberthemoremotionwillbeinthevideo.
noise_aug_strength(`int`,*optional*,defaultsto0.02):
Theamountofnoiseaddedtotheinitimage,thehigheritisthelessthevideowilllookliketheinitimage.Increaseitformoremotion.
decode_chunk_size(`int`,*optional*):
Thenumberofframestodecodeatatime.Thehigherthechunksize,thehigherthetemporalconsistency
betweenframes,butalsothehigherthememoryconsumption.Bydefault,thedecoderwilldecodeallframesatonce
formaximalquality.Reduce`decode_chunk_size`toreducememoryusage.
num_videos_per_prompt(`int`,*optional*,defaultsto1):
Thenumberofimagestogenerateperprompt.
generator(`torch.Generator`or`List[torch.Generator]`,*optional*):
A[`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)tomake
generationdeterministic.
latents(`torch.FloatTensor`,*optional*):
Pre-generatednoisylatentssampledfromaGaussiandistribution,tobeusedasinputsforimage
generation.Canbeusedtotweakthesamegenerationwithdifferentprompts.Ifnotprovided,alatents
tensorisgeneratedbysamplingusingthesuppliedrandom`generator`.
output_type(`str`,*optional*,defaultsto`"pil"`):
Theoutputformatofthegeneratedimage.Choosebetween`PIL.Image`or`np.array`.
callback_on_step_end(`Callable`,*optional*):
Afunctionthatcallsattheendofeachdenoisingstepsduringtheinference.Thefunctioniscalled
withthefollowingarguments:`callback_on_step_end(self:DiffusionPipeline,step:int,timestep:int,
callback_kwargs:Dict)`.`callback_kwargs`willincludealistofalltensorsasspecifiedby
`callback_on_step_end_tensor_inputs`.
callback_on_step_end_tensor_inputs(`List`,*optional*):
Thelistoftensorinputsforthe`callback_on_step_end`function.Thetensorsspecifiedinthelist
willbepassedas`callback_kwargs`argument.Youwillonlybeabletoincludevariableslistedinthe
`._callback_tensor_inputs`attributeofyourpipelineclass.
return_dict(`bool`,*optional*,defaultsto`True`):
Whetherornottoreturna[`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]insteadofa
plaintuple.

Returns:
[`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`]or`tuple`:
If`return_dict`is`True`,[`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`]isreturned,
otherwisea`tuple`isreturnedwherethefirstelementisalistoflistwiththegeneratedframes.

Examples:

```py
fromdiffusersimportStableVideoDiffusionPipeline
fromdiffusers.utilsimportload_image,export_to_video

pipe=StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",torch_dtype=torch.float16,variant="fp16")
pipe.to("cuda")

image=load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
image=image.resize((1024,576))

frames=pipe(image,num_frames=25,decode_chunk_size=8).frames[0]
export_to_video(frames,"generated.mp4",fps=7)
```
"""
#0.Defaultheightandwidthtounet
height=heightor96*self.vae_scale_factor
width=widthor96*self.vae_scale_factor

num_frames=num_framesifnum_framesisnotNoneelse25
decode_chunk_size=decode_chunk_sizeifdecode_chunk_sizeisnotNoneelsenum_frames

#1.Checkinputs.Raiseerrorifnotcorrect
self.check_inputs(image,height,width)

#2.Definecallparameters
ifisinstance(image,PIL.Image.Image):
batch_size=1
elifisinstance(image,list):
batch_size=len(image)
else:
batch_size=image.shape[0]
device=torch.device("cpu")

#here`guidance_scale`isdefinedanalogtotheguidanceweight`w`ofequation(2)
#oftheImagenpaper:https://arxiv.org/pdf/2205.11487.pdf.`guidance_scale=1`
#correspondstodoingnoclassifierfreeguidance.
do_classifier_free_guidance=max_guidance_scale>1.0

#3.Encodeinputimage
image_embeddings=self._encode_image(image,device,num_videos_per_prompt,do_classifier_free_guidance)

#NOTE:StableDiffusionVideowasconditionedonfps-1,which
#iswhyitisreducedhere.
#See:https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
fps=fps-1

#4.EncodeinputimageusingVAE
image=self.image_processor.preprocess(image,height=height,width=width)
noise=randn_tensor(image.shape,generator=generator,device=image.device,dtype=image.dtype)
image=image+noise_aug_strength*noise

image_latents=self._encode_vae_image(image,device,num_videos_per_prompt,do_classifier_free_guidance)
image_latents=image_latents.to(image_embeddings.dtype)

#Repeattheimagelatentsforeachframesowecanconcatenatethemwiththenoise
#image_latents[batch,channels,height,width]->[batch,num_frames,channels,height,width]
image_latents=image_latents.unsqueeze(1).repeat(1,num_frames,1,1,1)

#5.GetAddedTimeIDs
added_time_ids=self._get_add_time_ids(
fps,
motion_bucket_id,
noise_aug_strength,
image_embeddings.dtype,
batch_size,
num_videos_per_prompt,
do_classifier_free_guidance,
)
added_time_ids=added_time_ids

#4.Preparetimesteps
self.scheduler.set_timesteps(num_inference_steps,device=device)
timesteps=self.scheduler.timesteps
#5.Preparelatentvariables
num_channels_latents=8
latents=self.prepare_latents(
batch_size*num_videos_per_prompt,
num_frames,
num_channels_latents,
height,
width,
image_embeddings.dtype,
device,
generator,
latents,
)

#7.Prepareguidancescale
guidance_scale=torch.linspace(min_guidance_scale,max_guidance_scale,num_frames).unsqueeze(0)
guidance_scale=guidance_scale.to(device,latents.dtype)
guidance_scale=guidance_scale.repeat(batch_size*num_videos_per_prompt,1)
guidance_scale=_append_dims(guidance_scale,latents.ndim)

#8.Denoisingloop
num_warmup_steps=len(timesteps)-num_inference_steps*self.scheduler.order
num_timesteps=len(timesteps)
withself.progress_bar(total=num_inference_steps)asprogress_bar:
fori,tinenumerate(timesteps):
#expandthelatentsifwearedoingclassifierfreeguidance
latent_model_input=torch.cat([latents]*2)ifdo_classifier_free_guidanceelselatents
latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)

#Concatenateimage_latentsoverchannelsdimention
latent_model_input=torch.cat([latent_model_input,image_latents],dim=2)
#predictthenoiseresidual
noise_pred=torch.from_numpy(
self.unet(
[
latent_model_input,
t,
image_embeddings,
added_time_ids,
]
)[0]
)
#performguidance
ifdo_classifier_free_guidance:
noise_pred_uncond,noise_pred_cond=noise_pred.chunk(2)
noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_cond-noise_pred_uncond)

#computethepreviousnoisysamplex_t->x_t-1
latents=self.scheduler.step(noise_pred,t,latents).prev_sample

ifcallback_on_step_endisnotNone:
callback_kwargs={}
forkincallback_on_step_end_tensor_inputs:
callback_kwargs[k]=locals()[k]
callback_outputs=callback_on_step_end(self,i,t,callback_kwargs)

latents=callback_outputs.pop("latents",latents)

ifi==len(timesteps)-1or((i+1)>num_warmup_stepsand(i+1)%self.scheduler.order==0):
progress_bar.update()

ifnotoutput_type=="latent":
frames=self.decode_latents(latents,num_frames,decode_chunk_size)
frames=tensor2vid(frames,self.image_processor,output_type=output_type)
else:
frames=latents

ifnotreturn_dict:
returnframes

returnStableVideoDiffusionPipelineOutput(frames=frames)


#resizingutils
def_resize_with_antialiasing(input,size,interpolation="bicubic",align_corners=True):
h,w=input.shape[-2:]
factors=(h/size[0],w/size[1])

#First,wehavetodeterminesigma
#Takenfromskimage:https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
sigmas=(
max((factors[0]-1.0)/2.0,0.001),
max((factors[1]-1.0)/2.0,0.001),
)
#Nowkernelsize.Goodresultsarefor3sigma,butthatiskindofslow.Pillowuses1sigma
#https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
#Buttheydoitinthe2passes,whichgivesbetterresults.Let'stry2sigmasfornow
ks=int(max(2.0*2*sigmas[0],3)),int(max(2.0*2*sigmas[1],3))

#Makesureitisodd
if(ks[0]%2)==0:
ks=ks[0]+1,ks[1]

if(ks[1]%2)==0:

ks=ks[0],ks[1]+1

input=_gaussian_blur2d(input,ks,sigmas)

output=torch.nn.functional.interpolate(input,size=size,mode=interpolation,align_corners=align_corners)
returnoutput


def_compute_padding(kernel_size):
"""Computepaddingtuple."""
#4or6ints:(padding_left,padding_right,padding_top,padding_bottom)
#https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
iflen(kernel_size)<2:
raiseAssertionError(kernel_size)
computed=[k-1forkinkernel_size]

#forevenkernelsweneedtodoasymmetricpadding:(
out_padding=2*len(kernel_size)*[0]

foriinrange(len(kernel_size)):
computed_tmp=computed[-(i+1)]

pad_front=computed_tmp//2
pad_rear=computed_tmp-pad_front

out_padding[2*i+0]=pad_front
out_padding[2*i+1]=pad_rear

returnout_padding


def_filter2d(input,kernel):
#preparekernel
b,c,h,w=input.shape
tmp_kernel=kernel[:,None,...].to(device=input.device,dtype=input.dtype)

tmp_kernel=tmp_kernel.expand(-1,c,-1,-1)

height,width=tmp_kernel.shape[-2:]

padding_shape:list[int]=_compute_padding([height,width])
input=torch.nn.functional.pad(input,padding_shape,mode="reflect")

#kernelandinputtensorreshapetoalignelement-wiseorbatch-wiseparams
tmp_kernel=tmp_kernel.reshape(-1,1,height,width)
input=input.view(-1,tmp_kernel.size(0),input.size(-2),input.size(-1))

#convolvethetensorwiththekernel.
output=torch.nn.functional.conv2d(input,tmp_kernel,groups=tmp_kernel.size(0),padding=0,stride=1)

out=output.view(b,c,h,w)
returnout


def_gaussian(window_size:int,sigma):
ifisinstance(sigma,float):
sigma=torch.tensor([[sigma]])

batch_size=sigma.shape[0]

x=(torch.arange(window_size,device=sigma.device,dtype=sigma.dtype)-window_size//2).expand(batch_size,-1)

ifwindow_size%2==0:

x=x+0.5

gauss=torch.exp(-x.pow(2.0)/(2*sigma.pow(2.0)))

returngauss/gauss.sum(-1,keepdim=True)


def_gaussian_blur2d(input,kernel_size,sigma):
ifisinstance(sigma,tuple):
sigma=torch.tensor([sigma],dtype=input.dtype)
else:
sigma=sigma.to(dtype=input.dtype)

ky,kx=int(kernel_size[0]),int(kernel_size[1])
bs=sigma.shape[0]
kernel_x=_gaussian(kx,sigma[:,1].view(bs,1))
kernel_y=_gaussian(ky,sigma[:,0].view(bs,1))
out_x=_filter2d(input,kernel_x[...,None,:])
out=_filter2d(out_x,kernel_y[...,None])

returnout

RunVideoGeneration
--------------------

`backtotop⬆️<#table-of-contents>`__

SelectInferenceDevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importipywidgetsaswidgets

core=ov.Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=4,options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='AUTO')



..code::ipython3

fromtransformersimportCLIPImageProcessor


vae_encoder=core.compile_model(VAE_ENCODER_PATH,device.value)
image_encoder=core.compile_model(IMAGE_ENCODER_PATH,device.value)
unet=core.compile_model(UNET_PATH,device.value)
vae_decoder=core.compile_model(VAE_DECODER_PATH,device.value)
scheduler=AnimateLCMSVDStochasticIterativeScheduler.from_pretrained(MODEL_DIR/"scheduler")
feature_extractor=CLIPImageProcessor.from_pretrained(MODEL_DIR/"feature_extractor")

Now,let’sseemodelinaction.>Please,note,videogenerationis
memoryandtimeconsumingprocess.Forreducingmemoryconsumption,we
decreasedinputvideoresolutionto576x320andnumberofgenerated
framesthatmayaffectqualityofgeneratedvideo.Youcanchangethese
settingsmanuallyproviding``height``,``width``and``num_frames``
parametersintopipeline.

..code::ipython3

ov_pipe=OVStableVideoDiffusionPipeline(vae_encoder,image_encoder,unet,vae_decoder,scheduler,feature_extractor)

..code::ipython3

frames=ov_pipe(
image,
num_inference_steps=4,
motion_bucket_id=60,
num_frames=8,
height=320,
width=512,
generator=torch.manual_seed(12342),
).frames[0]



..parsed-literal::

0%||0/4[00:00<?,?it/s]


..parsed-literal::

denoisecurrently
tensor(128.5637)
denoisecurrently
tensor(13.6784)
denoisecurrently
tensor(0.4969)
denoisecurrently
tensor(0.)


..code::ipython3

out_path=Path("generated.mp4")

export_to_video(frames,str(out_path),fps=7)
frames[0].save(
"generated.gif",
save_all=True,
append_images=frames[1:],
optimize=False,
duration=120,
loop=0,
)

..code::ipython3

fromIPython.displayimportHTML

HTML('<imgsrc="generated.gif">')




..raw::html

<imgsrc="generated.gif">



Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``OVStableVideoDiffusionPipeline``structure,thediffusion
modeltakesupsignificantportionoftheoverallpipelineexecution
time.NowwewillshowyouhowtooptimizetheUNetpartusing
`NNCF<https://github.com/openvinotoolkit/nncf/>`__toreduce
computationcostandspeedupthepipeline.Quantizingtherestofthe
pipelinedoesnotsignificantlyimproveinferenceperformancebutcan
leadtoasubstantialdegradationofaccuracy.That’swhyweuseonly
weightcompressionforthe``vaeencoder``and``vaedecoder``toreduce
thememoryfootprint.

FortheUNetmodelweapplyquantizationinhybridmodewhichmeansthat
wequantize:(1)weightsofMatMulandEmbeddinglayersand(2)
activationsofotherlayers.Thestepsarethefollowing:

1.Createacalibrationdatasetforquantization.
2.Collectoperationswithweights.
3.Run``nncf.compress_model()``tocompressonlythemodelweights.
4.Run``nncf.quantize()``onthecompressedmodelwithweighted
operationsignoredbyproviding``ignored_scope``parameter.
5.Savethe``INT8``modelusing``openvino.save_model()``function.

Pleaseselectbelowwhetheryouwouldliketorunquantizationto
improvemodelinferencespeed.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

to_quantize=widgets.Checkbox(
value=True,
description="Quantization",
disabled=False,
)

to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



..code::ipython3

#Fetch`skip_kernel_extension`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

ov_int8_pipeline=None
OV_INT8_UNET_PATH=MODEL_DIR/"unet_int8.xml"
OV_INT8_VAE_ENCODER_PATH=MODEL_DIR/"vae_encoder_int8.xml"
OV_INT8_VAE_DECODER_PATH=MODEL_DIR/"vae_decoder_int8.xml"

%load_extskip_kernel_extension

Preparecalibrationdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseaportionof
`fusing/instructpix2pix-1000-samples<https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples>`__
datasetfromHuggingFaceascalibrationdata.Tocollectintermediate
modelinputsforUNetoptimizationweshouldcustomize
``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

fromtypingimportAny

importdatasets
importnumpyasnp
fromtqdm.notebookimporttqdm
fromIPython.utilsimportio


classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model:ov.CompiledModel,data_cache:List[Any]=None,keep_prob:float=0.5):
super().__init__(compiled_model)
self.data_cache=data_cacheifdata_cacheisnotNoneelse[]
self.keep_prob=keep_prob

def__call__(self,*args,**kwargs):
ifnp.random.rand()<=self.keep_prob:
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)


defcollect_calibration_data(ov_pipe,calibration_dataset_size:int,num_inference_steps:int=50)->List[Dict]:
original_unet=ov_pipe.unet
calibration_data=[]
ov_pipe.unet=CompiledModelDecorator(original_unet,calibration_data,keep_prob=1)

dataset=datasets.load_dataset("fusing/instructpix2pix-1000-samples",split="train",streaming=False).shuffle(seed=42)
#Runinferencefordatacollection
pbar=tqdm(total=calibration_dataset_size)
forbatchindataset:
image=batch["input_image"]

withio.capture_output()ascaptured:
ov_pipe(
image,
num_inference_steps=4,
motion_bucket_id=60,
num_frames=8,
height=256,
width=256,
generator=torch.manual_seed(12342),
)
pbar.update(len(calibration_data)-pbar.n)
iflen(calibration_data)>=calibration_dataset_size:
break

ov_pipe.unet=original_unet
returncalibration_data[:calibration_dataset_size]

..code::ipython3

%%skipnot$to_quantize.value

ifnotOV_INT8_UNET_PATH.exists():
subset_size=200
calibration_data=collect_calibration_data(ov_pipe,calibration_dataset_size=subset_size)

RunHybridModelQuantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fromcollectionsimportdeque

defget_operation_const_op(operation,const_port_id:int):
node=operation.input_value(const_port_id).get_node()
queue=deque([node])
constant_node=None
allowed_propagation_types_list=["Convert","FakeQuantize","Reshape"]

whilelen(queue)!=0:
curr_node=queue.popleft()
ifcurr_node.get_type_name()=="Constant":
constant_node=curr_node
break
iflen(curr_node.inputs())==0:
break
ifcurr_node.get_type_name()inallowed_propagation_types_list:
queue.append(curr_node.input_value(0).get_node())

returnconstant_node


defis_embedding(node)->bool:
allowed_types_list=["f16","f32","f64"]
const_port_id=0
input_tensor=node.input_value(const_port_id)
ifinput_tensor.get_element_type().get_type_name()inallowed_types_list:
const_node=get_operation_const_op(node,const_port_id)
ifconst_nodeisnotNone:
returnTrue

returnFalse


defcollect_ops_with_weights(model):
ops_with_weights=[]
foropinmodel.get_ops():
ifop.get_type_name()=="MatMul":
constant_node_0=get_operation_const_op(op,const_port_id=0)
constant_node_1=get_operation_const_op(op,const_port_id=1)
ifconstant_node_0orconstant_node_1:
ops_with_weights.append(op.get_friendly_name())
ifop.get_type_name()=="Gather"andis_embedding(op):
ops_with_weights.append(op.get_friendly_name())

returnops_with_weights

..code::ipython3

%%skipnot$to_quantize.value

importnncf
importlogging
fromnncf.quantization.advanced_parametersimportAdvancedSmoothQuantParameters

nncf.set_log_level(logging.ERROR)

ifnotOV_INT8_UNET_PATH.exists():
diffusion_model=core.read_model(UNET_PATH)
unet_ignored_scope=collect_ops_with_weights(diffusion_model)
compressed_diffusion_model=nncf.compress_weights(diffusion_model,ignored_scope=nncf.IgnoredScope(types=['Convolution']))
quantized_diffusion_model=nncf.quantize(
model=diffusion_model,
calibration_dataset=nncf.Dataset(calibration_data),
subset_size=subset_size,
model_type=nncf.ModelType.TRANSFORMER,
#Weadditionallyignorethefirstconvolutiontoimprovethequalityofgenerations
ignored_scope=nncf.IgnoredScope(names=unet_ignored_scope+["__module.conv_in/aten::_convolution/Convolution"]),
advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=-1))
)
ov.save_model(quantized_diffusion_model,OV_INT8_UNET_PATH)

RunWeightCompression
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Quantizingofthe``vaeencoder``and``vaedecoder``doesnot
significantlyimproveinferenceperformancebutcanleadtoa
substantialdegradationofaccuracy.Onlyweightcompressionwillbe
appliedforfootprintreduction.

..code::ipython3

%%skipnot$to_quantize.value

nncf.set_log_level(logging.INFO)

ifnotOV_INT8_VAE_ENCODER_PATH.exists():
text_encoder_model=core.read_model(VAE_ENCODER_PATH)
compressed_text_encoder_model=nncf.compress_weights(text_encoder_model,mode=nncf.CompressWeightsMode.INT4_SYM,group_size=64)
ov.save_model(compressed_text_encoder_model,OV_INT8_VAE_ENCODER_PATH)

ifnotOV_INT8_VAE_DECODER_PATH.exists():
decoder_model=core.read_model(VAE_DECODER_PATH)
compressed_decoder_model=nncf.compress_weights(decoder_model,mode=nncf.CompressWeightsMode.INT4_SYM,group_size=64)
ov.save_model(compressed_decoder_model,OV_INT8_VAE_DECODER_PATH)


..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│98%(29/32)│0%(0/3)│
├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│4│2%(3/32)│100%(3/3)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│99%(65/68)│0%(0/3)│
├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│4│1%(3/68)│100%(3/3)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



Let’scomparethevideogeneratedbytheoriginalandoptimized
pipelines.

..code::ipython3

%%skipnot$to_quantize.value

ov_int8_vae_encoder=core.compile_model(OV_INT8_VAE_ENCODER_PATH,device.value)
ov_int8_unet=core.compile_model(OV_INT8_UNET_PATH,device.value)
ov_int8_decoder=core.compile_model(OV_INT8_VAE_DECODER_PATH,device.value)

ov_int8_pipeline=OVStableVideoDiffusionPipeline(
ov_int8_vae_encoder,image_encoder,ov_int8_unet,ov_int8_decoder,scheduler,feature_extractor
)

int8_frames=ov_int8_pipeline(
image,
num_inference_steps=4,
motion_bucket_id=60,
num_frames=8,
height=320,
width=512,
generator=torch.manual_seed(12342),
).frames[0]



..parsed-literal::

0%||0/4[00:00<?,?it/s]


..parsed-literal::

/home/ltalamanova/env_ci/lib/python3.8/site-packages/diffusers/configuration_utils.py:139:FutureWarning:Accessingconfigattribute`unet`directlyvia'OVStableVideoDiffusionPipeline'objectattributeisdeprecated.Pleaseaccess'unet'over'OVStableVideoDiffusionPipeline'sconfigobjectinstead,e.g.'scheduler.config.unet'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)


..parsed-literal::

denoisecurrently
tensor(128.5637)
denoisecurrently
tensor(13.6784)
denoisecurrently
tensor(0.4969)
denoisecurrently
tensor(0.)


..code::ipython3

int8_out_path=Path("generated_int8.mp4")

export_to_video(frames,str(out_path),fps=7)
int8_frames[0].save(
"generated_int8.gif",
save_all=True,
append_images=int8_frames[1:],
optimize=False,
duration=120,
loop=0,
)
HTML('<imgsrc="generated_int8.gif">')




..raw::html

<imgsrc="generated_int8.gif">



Comparemodelfilesizes
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fp16_model_paths=[VAE_ENCODER_PATH,UNET_PATH,VAE_DECODER_PATH]
int8_model_paths=[OV_INT8_VAE_ENCODER_PATH,OV_INT8_UNET_PATH,OV_INT8_VAE_DECODER_PATH]

forfp16_path,int8_pathinzip(fp16_model_paths,int8_model_paths):
fp16_ir_model_size=fp16_path.with_suffix(".bin").stat().st_size
int8_model_size=int8_path.with_suffix(".bin").stat().st_size
print(f"{fp16_path.stem}compressionrate:{fp16_ir_model_size/int8_model_size:.3f}")


..parsed-literal::

vae_encodercompressionrate:2.018
unetcompressionrate:1.996
vae_decodercompressionrate:2.007


CompareinferencetimeoftheFP16andINT8pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tomeasuretheinferenceperformanceofthe``FP16``and``INT8``
pipelines,weusemedianinferencetimeoncalibrationsubset.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.

..code::ipython3

%%skipnot$to_quantize.value

importtime

defcalculate_inference_time(pipeline,validation_data):
inference_time=[]
forpromptinvalidation_data:
start=time.perf_counter()
withio.capture_output()ascaptured:
_=pipeline(
image,
num_inference_steps=4,
motion_bucket_id=60,
num_frames=8,
height=320,
width=512,
generator=torch.manual_seed(12342),
)
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
returnnp.median(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

validation_size=3
validation_dataset=datasets.load_dataset("fusing/instructpix2pix-1000-samples",split="train",streaming=True).shuffle(seed=42).take(validation_size)
validation_data=[data["input_image"]fordatainvalidation_dataset]

fp_latency=calculate_inference_time(ov_pipe,validation_data)
int8_latency=calculate_inference_time(ov_int8_pipeline,validation_data)
print(f"Performancespeed-up:{fp_latency/int8_latency:.3f}")


..parsed-literal::

Performancespeed-up:1.243


InteractiveDemo
----------------

`backtotop⬆️<#table-of-contents>`__

Pleaseselectbelowwhetheryouwouldliketousethequantizedmodelto
launchtheinteractivedemo.

..code::ipython3

quantized_model_present=ov_int8_pipelineisnotNone

use_quantized_model=widgets.Checkbox(
value=quantized_model_present,
description="Usequantizedmodel",
disabled=notquantized_model_present,
)

use_quantized_model




..parsed-literal::

Checkbox(value=True,description='Usequantizedmodel')



..code::ipython3

importgradioasgr
importrandom

max_64_bit_int=2**63-1
pipeline=ov_int8_pipelineifuse_quantized_model.valueelseov_pipe

example_images_urls=[
"https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/ship-7833921_1280.jpg?download=true",
"https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/ai-generated-8476858_1280.png?download=true",
"https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/ai-generated-8481641_1280.jpg?download=true",
"https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/dog-7396912_1280.jpg?download=true",
"https://huggingface.co/spaces/wangfuyun/AnimateLCM-SVD/resolve/main/test_imgs/cupcakes-380178_1280.jpg?download=true",
]

example_images_dir=Path("example_images")
example_images_dir.mkdir(exist_ok=True)
example_imgs=[]

forimage_id,urlinenumerate(example_images_urls):
img=load_image(url)
image_path=example_images_dir/f"{image_id}.png"
img.save(image_path)
example_imgs.append([image_path])


defsample(
image:PIL.Image,
seed:Optional[int]=42,
randomize_seed:bool=True,
motion_bucket_id:int=127,
fps_id:int=6,
num_inference_steps:int=15,
num_frames:int=4,
max_guidance_scale=1.0,
min_guidance_scale=1.0,
decoding_t:int=8,#Numberofframesdecodedatatime!ThiseatsmostVRAM.Reduceifnecessary.
output_folder:str="outputs",
progress=gr.Progress(track_tqdm=True),
):
ifimage.mode=="RGBA":
image=image.convert("RGB")

ifrandomize_seed:
seed=random.randint(0,max_64_bit_int)
generator=torch.manual_seed(seed)

output_folder=Path(output_folder)
output_folder.mkdir(exist_ok=True)
base_count=len(list(output_folder.glob("*.mp4")))
video_path=output_folder/f"{base_count:06d}.mp4"

frames=pipeline(
image,
decode_chunk_size=decoding_t,
generator=generator,
motion_bucket_id=motion_bucket_id,
noise_aug_strength=0.1,
num_frames=num_frames,
num_inference_steps=num_inference_steps,
max_guidance_scale=max_guidance_scale,
min_guidance_scale=min_guidance_scale,
).frames[0]
export_to_video(frames,str(video_path),fps=fps_id)

returnvideo_path,seed


defresize_image(image,output_size=(512,320)):
#Calculateaspectratios
target_aspect=output_size[0]/output_size[1]#Aspectratioofthedesiredsize
image_aspect=image.width/image.height#Aspectratiooftheoriginalimage

#Resizethencropiftheoriginalimageislarger
ifimage_aspect>target_aspect:
#Resizetheimagetomatchthetargetheight,maintainingaspectratio
new_height=output_size[1]
new_width=int(new_height*image_aspect)
resized_image=image.resize((new_width,new_height),PIL.Image.LANCZOS)
#Calculatecoordinatesforcropping
left=(new_width-output_size[0])/2
top=0
right=(new_width+output_size[0])/2
bottom=output_size[1]
else:
#Resizetheimagetomatchthetargetwidth,maintainingaspectratio
new_width=output_size[0]
new_height=int(new_width/image_aspect)
resized_image=image.resize((new_width,new_height),PIL.Image.LANCZOS)
#Calculatecoordinatesforcropping
left=0
top=(new_height-output_size[1])/2
right=output_size[0]
bottom=(new_height+output_size[1])/2

#Croptheimage
cropped_image=resized_image.crop((left,top,right,bottom))
returncropped_image


withgr.Blocks()asdemo:
gr.Markdown(
"""#StableVideoDiffusion:ImagetoVideoGenerationwithOpenVINO.
"""
)
withgr.Row():
withgr.Column():
image_in=gr.Image(label="Uploadyourimage",type="pil")
generate_btn=gr.Button("Generate")
video=gr.Video()
withgr.Accordion("Advancedoptions",open=False):
seed=gr.Slider(
label="Seed",
value=42,
randomize=True,
minimum=0,
maximum=max_64_bit_int,
step=1,
)
randomize_seed=gr.Checkbox(label="Randomizeseed",value=True)
motion_bucket_id=gr.Slider(
label="Motionbucketid",
info="Controlshowmuchmotiontoadd/removefromtheimage",
value=127,
minimum=1,
maximum=255,
)
fps_id=gr.Slider(
label="Framespersecond",
info="Thelengthofyourvideoinsecondswillbenum_frames/fps",
value=6,
minimum=5,
maximum=30,
step=1,
)
num_frames=gr.Slider(label="NumberofFrames",value=8,minimum=2,maximum=25,step=1)
num_steps=gr.Slider(label="Numberofgenerationsteps",value=4,minimum=1,maximum=8,step=1)
max_guidance_scale=gr.Slider(
label="Maxguidancescale",
info="classifier-freeguidancestrength",
value=1.2,
minimum=1,
maximum=2,
)
min_guidance_scale=gr.Slider(
label="Minguidancescale",
info="classifier-freeguidancestrength",
value=1,
minimum=1,
maximum=1.5,
)
examples=gr.Examples(
examples=example_imgs,
inputs=[image_in],
outputs=[video,seed],
)

image_in.upload(fn=resize_image,inputs=image_in,outputs=image_in)
generate_btn.click(
fn=sample,
inputs=[
image_in,
seed,
randomize_seed,
motion_bucket_id,
fps_id,
num_steps,
num_frames,
max_guidance_scale,
min_guidance_scale,
],
outputs=[video,seed],
api_name="video",
)


try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(debug=False,share=True)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
