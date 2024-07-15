ImagegenerationwithHunyuanDITandOpenVINO
=============================================

Hunyuan-DiTisapowerfultext-to-imagediffusiontransformerwith
fine-grainedunderstandingofbothEnglishandChinese.

|image0|

Themodelarchitectureexpertlyblendsdiffusionmodelsandtransformer
networkstounlockthepotentialoftext-to-imagegeneration.The
diffusiontransformerconsistsofencoderanddecoderblocksthatwork
togethertotranslatethetextpromptintoavisualrepresentation.Each
blockcontainsthreekeymodules:self-attention,cross-attention,anda
feed-forwardnetwork.Self-attentionanalyzesrelationshipswithinthe
image,whilecross-attentionfusesthetextencodingfromCLIPandT5,
guidingtheimagegenerationprocessbasedontheuser’sinput.The
Hunyuan-DiTblock,specifically,consistsoftheseencoderanddecoder
blocks.Theencoderblockprocessestheimagepatches,capturing
patternsanddependencies,whilethedecoderreconstructstheimagefrom
theencodedinformation.Thedecoderalsoincludesaskipmodulethat
directlyconnectstotheencoder,facilitatinginformationflowand
enhancingdetailreconstruction.RotaryPositionalEmbedding(RoPE)
ensuresthatthemodelunderstandsthespatialrelationshipsbetween
imagepatches,accuratelyreconstructingthevisualcomposition.
Additionally,CentralizedInterpolativePositionalEncodingenables
multi-resolutiontraining,allowingHunyuan-DiTtohandlevariousimage
sizesseamlessly.

Moredetailsaboutmodelcanbefoundinoriginal
`repository<https://github.com/Tencent/HunyuanDiT>`__,`projectweb
page<https://dit.hunyuan.tencent.com/>`__and
`paper<https://arxiv.org/abs/2405.08748>`__.

InthistutorialweconsiderhowtoconvertandrunHunyuan-DITmodel
usingOpenVINO.Additionally,wewilluse
`NNCF<https://github.com/openvinotoolkit/nncf>`__foroptimizingmodel
inlowprecision.####Tableofcontents:

-`Prerequisites<#prerequisites>`__
-`DownloadPyTorchmodel<#download-pytorch-model>`__
-`BuildPyTorchpipeline<#build-pytorch-pipeline>`__
-`ConvertandOptimizemodelswithOpenVINOand
NNCF<#convert-and-optimize-models-with-openvino-and-nncf>`__

-`DiT<#dit>`__
-`TextEncoder<#text-encoder>`__
-`TextEmbedder<#text-embedder>`__
-`VAEDecoder<#vae-decoder>`__

-`CreateInferencepipeline<#create-inference-pipeline>`__

-`Runmodel<#run-model>`__

-`Interactivedemo<#interactive-demo>`__

..|image0|image::https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/framework.png

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__##Prerequisites

..code::ipython3

%pipinstall-q"torch>=2.1"torchvisioneinopstimmpeftacceleratetransformersdiffusershuggingface-hubtokenizerssentencepieceprotobufloguru--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"nncf>=2.11""gradio>=4.19""pillow""opencv-python"
%pipinstall-pre-Uqopenvino--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly

..code::ipython3

frompathlibimportPath
importsys

repo_dir=Path("HunyuanDiT")

ifnotrepo_dir.exists():
!gitclonehttps://github.com/tencent/HunyuanDiT
%cdHunyuanDiT
!gitcheckoutebfb7936490287616c38519f87084a34a1d75362
%cd..

sys.path.append(str(repo_dir))

DownloadPyTorchmodel
----------------------

`backtotop⬆️<#table-of-contents>`__

Forstartingworkwithmodel,weshoulddownloaditfromHuggingFace
Hub.Wewilluse
`Distilled<https://huggingface.co/Tencent-Hunyuan/Distillation>`__
versionof
`hunyuan-DIT<https://huggingface.co/Tencent-Hunyuan/HunyuanDiT>`__.In
thefirsttime,modeldownloadingmaytakesometime.

..code::ipython3

importhuggingface_hubashf_hub

weights_dir=Path("ckpts")
weights_dir.mkdir(exist_ok=True)
models_dir=Path("models")
models_dir.mkdir(exist_ok=True)

OV_DIT_MODEL=models_dir/"dit.xml"
OV_TEXT_ENCODER=models_dir/"text_encoder.xml"
OV_TEXT_EMBEDDER=models_dir/"text_embedder.xml"
OV_VAE_DECODER=models_dir/"vae_decoder.xml"

model_conversion_required=notall([OV_DIT_MODEL.exists(),OV_TEXT_ENCODER.exists(),OV_TEXT_EMBEDDER.exists(),OV_VAE_DECODER.exists()])
distilled_repo_id="Tencent-Hunyuan/Distillation"
orig_repo_id="Tencent-Hunyuan/HunyuanDiT"

ifmodel_conversion_requiredandnot(weights_dir/"t2i").exists():
hf_hub.snapshot_download(repo_id=orig_repo_id,local_dir=weights_dir,allow_patterns=["t2i/*"],ignore_patterns=["t2i/model/*"])
hf_hub.hf_hub_download(repo_id=distilled_repo_id,filename="pytorch_model_distill.pt",local_dir=weights_dir/"t2i/model")

BuildPyTorchpipeline
----------------------

`backtotop⬆️<#table-of-contents>`__

Thecodebellow,initializePyTorchinferencepipelineforhunyuan-DIT
model.

..code::ipython3

fromhydit.inferenceimportEnd2End
fromhydit.configimportget_args

gen=None

ifmodel_conversion_required:
args=get_args({})
args.load_key="distill"
args.model_root=weights_dir

#Loadmodels
gen=End2End(args,weights_dir)


..parsed-literal::

/home/ea/work/notebooks_env/lib/python3.8/site-packages/diffusers/models/transformers/transformer_2d.py:34:FutureWarning:`Transformer2DModelOutput`isdeprecatedandwillberemovedinversion1.0.0.Importing`Transformer2DModelOutput`from`diffusers.models.transformer_2d`isdeprecatedandthiswillberemovedinafutureversion.Pleaseuse`fromdiffusers.models.modeling_outputsimportTransformer2DModelOutput`,instead.
deprecate("Transformer2DModelOutput","1.0.0",deprecation_message)


..parsed-literal::

flash_attnimportfailed:Nomodulenamed'flash_attn'


ConvertandOptimizemodelswithOpenVINOandNNCF
--------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Startingfrom2023.0release,OpenVINOsupportsPyTorchmodelsdirectly
viaModelConversionAPI.``ov.convert_model``functionacceptsinstance
ofPyTorchmodelandexampleinputsfortracingandreturnsobjectof
``ov.Model``class,readytouseorsaveondiskusing``ov.save_model``
function.

Thepipelineconsistsoffourimportantparts:

-ClipandT5TextEncodertocreateconditiontogenerateanimage
fromatextprompt.
-DITforstep-by-stepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)fordecodinglatentspacetoimage.

Forreducingmodelmemoryconsumptionandimprovingperformancewewill
useweightscompression.The`Weights
Compression<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__
algorithmisaimedatcompressingtheweightsofthemodelsandcanbe
usedtooptimizethemodelfootprintandperformanceoflargemodels
wherethesizeofweightsisrelativelylargerthanthesizeof
activations,forexample,LargeLanguageModels(LLM).ComparedtoINT8
compression,INT4compressionimprovesperformanceevenmore,but
introducesaminordropinpredictionquality.

Letusconvertandoptimizeeachpart:

DiT
~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtorch
importnncf
importgc
importopenvinoasov


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


ifnotOV_DIT_MODEL.exists():
latent_model_input=torch.randn(2,4,64,64)
t_expand=torch.randint(0,1000,[2])
prompt_embeds=torch.randn(2,77,1024)
attention_mask=torch.randint(0,2,[2,77])
prompt_embeds_t5=torch.randn(2,256,2048)
attention_mask_t5=torch.randint(0,2,[2,256])
ims=torch.tensor([[512,512,512,512,0,0],[512,512,512,512,0,0]])
style=torch.tensor([0,0])
freqs_cis_img=(
torch.randn(1024,88),
torch.randn(1024,88),
)
model_args=(
latent_model_input,
t_expand,
prompt_embeds,
attention_mask,
prompt_embeds_t5,
attention_mask_t5,
ims,
style,
freqs_cis_img[0],
freqs_cis_img[1],
)

gen.model.to(torch.device("cpu"))
gen.model.to(torch.float32)
gen.model.args.use_fp16=False
ov_model=ov.convert_model(gen.model,example_input=model_args)
ov_model=nncf.compress_weights(ov_model,mode=nncf.CompressWeightsMode.INT4_SYM,ratio=0.8,group_size=64)
ov.save_model(ov_model,OV_DIT_MODEL)
delov_model
cleanup_torchscript_cache()
delgen.model
gc.collect()


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,onnx,openvino


TextEncoder
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotOV_TEXT_ENCODER.exists():
gen.clip_text_encoder.to("cpu")
gen.clip_text_encoder.to(torch.float32)
ov_model=ov.convert_model(
gen.clip_text_encoder,example_input={"input_ids":torch.ones([1,77],dtype=torch.int64),"attention_mask":torch.ones([1,77],dtype=torch.int64)}
)
ov_model=nncf.compress_weights(ov_model,mode=nncf.CompressWeightsMode.INT4_SYM,ratio=0.8,group_size=64)
ov.save_model(ov_model,OV_TEXT_ENCODER)
delov_model
cleanup_torchscript_cache()
delgen.clip_text_encoder
gc.collect()

TextEmbedder
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotOV_TEXT_EMBEDDER.exists():
gen.embedder_t5.model.to("cpu")
gen.embedder_t5.model.to(torch.float32)

ov_model=ov.convert_model(gen.embedder_t5,example_input=(torch.ones([1,256],dtype=torch.int64),torch.ones([1,256],dtype=torch.int64)))
ov_model=nncf.compress_weights(ov_model,mode=nncf.CompressWeightsMode.INT4_SYM,ratio=0.8,group_size=64)
ov.save_model(ov_model,OV_TEXT_EMBEDDER)
delov_model
cleanup_torchscript_cache()
delgen.embedder_t5
gc.collect()

VAEDecoder
~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotOV_VAE_DECODER.exists():
vae_decoder=gen.vae
vae_decoder.to("cpu")
vae_decoder.to(torch.float32)

vae_decoder.forward=vae_decoder.decode

ov_model=ov.convert_model(vae_decoder,example_input=torch.zeros((1,4,128,128)))
ov.save_model(ov_model,OV_VAE_DECODER)
delov_model
cleanup_torchscript_cache()
delvae_decoder
delgen.vae
gc.collect()

..code::ipython3

delgen
gc.collect();

CreateInferencepipeline
-------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importinspect
fromtypingimportAny,Callable,Dict,List,Optional,Union

importtorch
fromdiffusers.configuration_utilsimportFrozenDict
fromdiffusers.image_processorimportVaeImageProcessor
fromdiffusers.modelsimportAutoencoderKL,UNet2DConditionModel
fromdiffusers.pipelines.pipeline_utilsimportDiffusionPipeline
fromdiffusers.pipelines.stable_diffusionimportStableDiffusionPipelineOutput
fromdiffusers.schedulersimportKarrasDiffusionSchedulers
fromdiffusers.utils.torch_utilsimportrandn_tensor
fromtransformersimportBertModel,BertTokenizer
fromtransformersimportCLIPImageProcessor,CLIPTextModel,CLIPTokenizer


defrescale_noise_cfg(noise_cfg,noise_pred_text,guidance_rescale=0.0):
"""
Rescale`noise_cfg`accordingto`guidance_rescale`.Basedonfindingsof[CommonDiffusionNoiseSchedulesand
SampleStepsareFlawed](https://arxiv.org/pdf/2305.08891.pdf).SeeSection3.4
"""
std_text=noise_pred_text.std(dim=list(range(1,noise_pred_text.ndim)),keepdim=True)
std_cfg=noise_cfg.std(dim=list(range(1,noise_cfg.ndim)),keepdim=True)
#rescaletheresultsfromguidance(fixesoverexposure)
noise_pred_rescaled=noise_cfg*(std_text/std_cfg)
#mixwiththeoriginalresultsfromguidancebyfactorguidance_rescaletoavoid"plainlooking"images
noise_cfg=guidance_rescale*noise_pred_rescaled+(1-guidance_rescale)*noise_cfg
returnnoise_cfg


classOVHyDiTPipeline(DiffusionPipeline):
def__init__(
self,
vae:AutoencoderKL,
text_encoder:Union[BertModel,CLIPTextModel],
tokenizer:Union[BertTokenizer,CLIPTokenizer],
unet:UNet2DConditionModel,
scheduler:KarrasDiffusionSchedulers,
feature_extractor:CLIPImageProcessor,
progress_bar_config:Dict[str,Any]=None,
embedder_t5=None,
embedder_tokenizer=None,
):
self.embedder_t5=embedder_t5
self.embedder_tokenizer=embedder_tokenizer

ifprogress_bar_configisNone:
progress_bar_config={}
ifnothasattr(self,"_progress_bar_config"):
self._progress_bar_config={}
self._progress_bar_config.update(progress_bar_config)

ifhasattr(scheduler.config,"steps_offset")andscheduler.config.steps_offset!=1:
new_config=dict(scheduler.config)
new_config["steps_offset"]=1
scheduler._internal_dict=FrozenDict(new_config)

ifhasattr(scheduler.config,"clip_sample")andscheduler.config.clip_sampleisTrue:
new_config=dict(scheduler.config)
new_config["clip_sample"]=False
scheduler._internal_dict=FrozenDict(new_config)

self.vae=vae
self.text_encoder=text_encoder
self.tokenizer=tokenizer
self.unet=unet
self.scheduler=scheduler
self.feature_extractor=feature_extractor
self.vae_scale_factor=2**3
self.image_processor=VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

defencode_prompt(
self,
prompt,
num_images_per_prompt,
do_classifier_free_guidance,
negative_prompt=None,
prompt_embeds:Optional[torch.FloatTensor]=None,
negative_prompt_embeds:Optional[torch.FloatTensor]=None,
embedder=None,
):
r"""
Encodesthepromptintotextencoderhiddenstates.

Args:
prompt(`str`or`List[str]`,*optional*):
prompttobeencoded
num_images_per_prompt(`int`):
numberofimagesthatshouldbegeneratedperprompt
do_classifier_free_guidance(`bool`):
whethertouseclassifierfreeguidanceornot
negative_prompt(`str`or`List[str]`,*optional*):
Thepromptorpromptsnottoguidetheimagegeneration.Ifnotdefined,onehastopass
`negative_prompt_embeds`instead.Ignoredwhennotusingguidance(i.e.,ignoredif`guidance_scale`is
lessthan`1`).
prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatedtextembeddings.Canbeusedtoeasilytweaktextinputs,*e.g.*promptweighting.Ifnot
provided,textembeddingswillbegeneratedfrom`prompt`inputargument.
negative_prompt_embeds(`torch.FloatTensor`,*optional*):
Pre-generatednegativetextembeddings.Canbeusedtoeasilytweaktextinputs,*e.g.*prompt
weighting.Ifnotprovided,negative_prompt_embedswillbegeneratedfrom`negative_prompt`input
argument.
embedder:
T5embedder
"""
ifembedderisNone:
text_encoder=self.text_encoder
tokenizer=self.tokenizer
max_length=self.tokenizer.model_max_length
else:
text_encoder=embedder
tokenizer=self.embedder_tokenizer
max_length=256

ifpromptisnotNoneandisinstance(prompt,str):
batch_size=1
elifpromptisnotNoneandisinstance(prompt,list):
batch_size=len(prompt)
else:
batch_size=prompt_embeds.shape[0]

ifprompt_embedsisNone:
text_inputs=tokenizer(
prompt,
padding="max_length",
max_length=max_length,
truncation=True,
return_attention_mask=True,
return_tensors="pt",
)
text_input_ids=text_inputs.input_ids
attention_mask=text_inputs.attention_mask

prompt_embeds=text_encoder([text_input_ids,attention_mask])
prompt_embeds=torch.from_numpy(prompt_embeds[0])
attention_mask=attention_mask.repeat(num_images_per_prompt,1)
else:
attention_mask=None

bs_embed,seq_len,_=prompt_embeds.shape
#duplicatetextembeddingsforeachgenerationperprompt,usingmpsfriendlymethod
prompt_embeds=prompt_embeds.repeat(1,num_images_per_prompt,1)
prompt_embeds=prompt_embeds.view(bs_embed*num_images_per_prompt,seq_len,-1)

#getunconditionalembeddingsforclassifierfreeguidance
ifdo_classifier_free_guidanceandnegative_prompt_embedsisNone:
uncond_tokens:List[str]
ifnegative_promptisNone:
uncond_tokens=[""]*batch_size
elifpromptisnotNoneandtype(prompt)isnottype(negative_prompt):
raiseTypeError(f"`negative_prompt`shouldbethesametypeto`prompt`,butgot{type(negative_prompt)}!="f"{type(prompt)}.")
elifisinstance(negative_prompt,str):
uncond_tokens=[negative_prompt]
elifbatch_size!=len(negative_prompt):
raiseValueError(
f"`negative_prompt`:{negative_prompt}hasbatchsize{len(negative_prompt)},but`prompt`:"
f"{prompt}hasbatchsize{batch_size}.Pleasemakesurethatpassed`negative_prompt`matches"
"thebatchsizeof`prompt`."
)
else:
uncond_tokens=negative_prompt

max_length=prompt_embeds.shape[1]
uncond_input=tokenizer(
uncond_tokens,
padding="max_length",
max_length=max_length,
truncation=True,
return_tensors="pt",
)
uncond_attention_mask=uncond_input.attention_mask
negative_prompt_embeds=text_encoder([uncond_input.input_ids,uncond_attention_mask])
negative_prompt_embeds=torch.from_numpy(negative_prompt_embeds[0])
uncond_attention_mask=uncond_attention_mask.repeat(num_images_per_prompt,1)
else:
uncond_attention_mask=None

ifdo_classifier_free_guidance:
#duplicateunconditionalembeddingsforeachgenerationperprompt,usingmpsfriendlymethod
seq_len=negative_prompt_embeds.shape[1]

negative_prompt_embeds=negative_prompt_embeds

negative_prompt_embeds=negative_prompt_embeds.repeat(1,num_images_per_prompt,1)
negative_prompt_embeds=negative_prompt_embeds.view(batch_size*num_images_per_prompt,seq_len,-1)

returnprompt_embeds,negative_prompt_embeds,attention_mask,uncond_attention_mask

defprepare_extra_step_kwargs(self,generator,eta):
#prepareextrakwargsfortheschedulerstep,sincenotallschedulershavethesamesignature
#eta(η)isonlyusedwiththeDDIMScheduler,itwillbeignoredforotherschedulers.
#etacorrespondstoηinDDIMpaper:https://arxiv.org/abs/2010.02502
#andshouldbebetween[0,1]

accepts_eta="eta"inset(inspect.signature(self.scheduler.step).parameters.keys())
extra_step_kwargs={}
ifaccepts_eta:
extra_step_kwargs["eta"]=eta

#checkifthescheduleracceptsgenerator
accepts_generator="generator"inset(inspect.signature(self.scheduler.step).parameters.keys())
ifaccepts_generator:
extra_step_kwargs["generator"]=generator
returnextra_step_kwargs

defcheck_inputs(
self,
prompt,
height,
width,
callback_steps,
negative_prompt=None,
prompt_embeds=None,
negative_prompt_embeds=None,
):
ifheight%8!=0orwidth%8!=0:
raiseValueError(f"`height`and`width`havetobedivisibleby8butare{height}and{width}.")

if(callback_stepsisNone)or(callback_stepsisnotNoneand(notisinstance(callback_steps,int)orcallback_steps<=0)):
raiseValueError(f"`callback_steps`hastobeapositiveintegerbutis{callback_steps}oftype"f"{type(callback_steps)}.")
ifpromptisnotNoneandprompt_embedsisnotNone:
raiseValueError(
f"Cannotforwardboth`prompt`:{prompt}and`prompt_embeds`:{prompt_embeds}.Pleasemakesureto""onlyforwardoneofthetwo."
)
elifpromptisNoneandprompt_embedsisNone:
raiseValueError("Provideeither`prompt`or`prompt_embeds`.Cannotleaveboth`prompt`and`prompt_embeds`undefined.")
elifpromptisnotNoneand(notisinstance(prompt,str)andnotisinstance(prompt,list)):
raiseValueError(f"`prompt`hastobeoftype`str`or`list`butis{type(prompt)}")

ifnegative_promptisnotNoneandnegative_prompt_embedsisnotNone:
raiseValueError(
f"Cannotforwardboth`negative_prompt`:{negative_prompt}and`negative_prompt_embeds`:"
f"{negative_prompt_embeds}.Pleasemakesuretoonlyforwardoneofthetwo."
)

ifprompt_embedsisnotNoneandnegative_prompt_embedsisnotNone:
ifprompt_embeds.shape!=negative_prompt_embeds.shape:
raiseValueError(
"`prompt_embeds`and`negative_prompt_embeds`musthavethesameshapewhenpasseddirectly,but"
f"got:`prompt_embeds`{prompt_embeds.shape}!=`negative_prompt_embeds`"
f"{negative_prompt_embeds.shape}."
)

defprepare_latents(self,batch_size,num_channels_latents,height,width,dtype,generator,latents=None):
shape=(batch_size,num_channels_latents,height//self.vae_scale_factor,width//self.vae_scale_factor)
ifisinstance(generator,list)andlen(generator)!=batch_size:
raiseValueError(
f"Youhavepassedalistofgeneratorsoflength{len(generator)},butrequestedaneffectivebatch"
f"sizeof{batch_size}.Makesurethebatchsizematchesthelengthofthegenerators."
)

iflatentsisNone:
latents=randn_tensor(shape,generator=generator,device=torch.device("cpu"),dtype=dtype)

#scaletheinitialnoisebythestandarddeviationrequiredbythescheduler
latents=latents*self.scheduler.init_noise_sigma
returnlatents

def__call__(
self,
height:int,
width:int,
prompt:Union[str,List[str]]=None,
num_inference_steps:Optional[int]=50,
guidance_scale:Optional[float]=7.5,
negative_prompt:Optional[Union[str,List[str]]]=None,
num_images_per_prompt:Optional[int]=1,
eta:Optional[float]=0.0,
generator:Optional[Union[torch.Generator,List[torch.Generator]]]=None,
latents:Optional[torch.FloatTensor]=None,
prompt_embeds:Optional[torch.FloatTensor]=None,
prompt_embeds_t5:Optional[torch.FloatTensor]=None,
negative_prompt_embeds:Optional[torch.FloatTensor]=None,
negative_prompt_embeds_t5:Optional[torch.FloatTensor]=None,
output_type:Optional[str]="pil",
return_dict:bool=True,
callback:Optional[Callable[[int,int,torch.FloatTensor,torch.FloatTensor],None]]=None,
callback_steps:int=1,
guidance_rescale:float=0.0,
image_meta_size:Optional[torch.LongTensor]=None,
style:Optional[torch.LongTensor]=None,
freqs_cis_img:Optional[tuple]=None,
learn_sigma:bool=True,
):
#1.Checkinputs.Raiseerrorifnotcorrect
self.check_inputs(prompt,height,width,callback_steps,negative_prompt,prompt_embeds,negative_prompt_embeds)

#2.Definecallparameters
ifpromptisnotNoneandisinstance(prompt,str):
batch_size=1
elifpromptisnotNoneandisinstance(prompt,list):
batch_size=len(prompt)
else:
batch_size=prompt_embeds.shape[0]

#here`guidance_scale`isdefinedanalogtotheguidanceweight`w`ofequation(2)
#oftheImagenpaper:https://arxiv.org/pdf/2205.11487.pdf.`guidance_scale=1`
#correspondstodoingnoclassifierfreeguidance.from
do_classifier_free_guidance=guidance_scale>1.0

prompt_embeds,negative_prompt_embeds,attention_mask,uncond_attention_mask=self.encode_prompt(
prompt,
num_images_per_prompt,
do_classifier_free_guidance,
negative_prompt,
prompt_embeds=prompt_embeds,
negative_prompt_embeds=negative_prompt_embeds,
)
prompt_embeds_t5,negative_prompt_embeds_t5,attention_mask_t5,uncond_attention_mask_t5=self.encode_prompt(
prompt,
num_images_per_prompt,
do_classifier_free_guidance,
negative_prompt,
prompt_embeds=prompt_embeds_t5,
negative_prompt_embeds=negative_prompt_embeds_t5,
embedder=self.embedder_t5,
)

#Forclassifierfreeguidance,weneedtodotwoforwardpasses.
#Hereweconcatenatetheunconditionalandtextembeddingsintoasinglebatch
#toavoiddoingtwoforwardpasses
ifdo_classifier_free_guidance:
prompt_embeds=torch.cat([negative_prompt_embeds,prompt_embeds])
attention_mask=torch.cat([uncond_attention_mask,attention_mask])
prompt_embeds_t5=torch.cat([negative_prompt_embeds_t5,prompt_embeds_t5])
attention_mask_t5=torch.cat([uncond_attention_mask_t5,attention_mask_t5])

#4.Preparetimesteps
self.scheduler.set_timesteps(num_inference_steps,device=torch.device("cpu"))
timesteps=self.scheduler.timesteps

#5.Preparelatentvariables
num_channels_latents=4
latents=self.prepare_latents(
batch_size*num_images_per_prompt,
num_channels_latents,
height,
width,
prompt_embeds.dtype,
generator,
latents,
)

#6.Prepareextrastepkwargs.
extra_step_kwargs=self.prepare_extra_step_kwargs(generator,eta)

#7.Denoisingloop
num_warmup_steps=len(timesteps)-num_inference_steps*self.scheduler.order
withself.progress_bar(total=num_inference_steps)asprogress_bar:
fori,tinenumerate(timesteps):
#expandthelatentsifwearedoingclassifierfreeguidance
latent_model_input=torch.cat([latents]*2)ifdo_classifier_free_guidanceelselatents
latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)
#expandscalartto1-Dtensortomatchthe1stdimoflatent_model_input
t_expand=torch.tensor([t]*latent_model_input.shape[0],device=latent_model_input.device)

ims=image_meta_sizeifimage_meta_sizeisnotNoneelsetorch.tensor([[1024,1024,1024,1024,0,0],[1024,1024,1024,1024,0,0]])

noise_pred=torch.from_numpy(
self.unet(
[
latent_model_input,
t_expand,
prompt_embeds,
attention_mask,
prompt_embeds_t5,
attention_mask_t5,
ims,
style,
freqs_cis_img[0],
freqs_cis_img[1],
]
)[0]
)
iflearn_sigma:
noise_pred,_=noise_pred.chunk(2,dim=1)

#performguidance
ifdo_classifier_free_guidance:
noise_pred_uncond,noise_pred_text=noise_pred.chunk(2)
noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

ifdo_classifier_free_guidanceandguidance_rescale>0.0:
#Basedon3.4.inhttps://arxiv.org/pdf/2305.08891.pdf
noise_pred=rescale_noise_cfg(noise_pred,noise_pred_text,guidance_rescale=guidance_rescale)

#computethepreviousnoisysamplex_t->x_t-1
results=self.scheduler.step(noise_pred,t,latents,**extra_step_kwargs,return_dict=True)
latents=results.prev_sample
pred_x0=results.pred_original_sampleifhasattr(results,"pred_original_sample")elseNone

#callthecallback,ifprovided
ifi==len(timesteps)-1or((i+1)>num_warmup_stepsand(i+1)%self.scheduler.order==0):
progress_bar.update()
ifcallbackisnotNoneandi%callback_steps==0:
callback(i,t,latents,pred_x0)

has_nsfw_concept=None
ifnotoutput_type=="latent":
image=torch.from_numpy(self.vae(latents/0.13025)[0])
else:
image=latents

ifhas_nsfw_conceptisNone:
do_denormalize=[True]*image.shape[0]
else:
do_denormalize=[nothas_nsfwforhas_nsfwinhas_nsfw_concept]

image=self.image_processor.postprocess(image,output_type=output_type,do_denormalize=do_denormalize)

ifnotreturn_dict:
return(image,has_nsfw_concept)

returnStableDiffusionPipelineOutput(images=image,nsfw_content_detected=has_nsfw_concept)

Runmodel
~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Pleaseselectinferencedeviceusingdropdownwidget:

..code::ipython3

importopenvinoasov
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

Dropdown(description='Device:',index=3,options=('CPU','GPU.0','GPU.1','AUTO'),value='AUTO')



..code::ipython3

importgc

core=ov.Core()
ov_dit=core.read_model(OV_DIT_MODEL)
dit=core.compile_model(ov_dit,device.value)
ov_text_encoder=core.read_model(OV_TEXT_ENCODER)
text_encoder=core.compile_model(ov_text_encoder,device.value)
ov_text_embedder=core.read_model(OV_TEXT_EMBEDDER)

text_embedder=core.compile_model(ov_text_embedder,device.value)
vae_decoder=core.compile_model(OV_VAE_DECODER,device.value)

delov_dit,ov_text_encoder,ov_text_embedder

gc.collect();

..code::ipython3

fromtransformersimportAutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("./ckpts/t2i/tokenizer/")
embedder_tokenizer=AutoTokenizer.from_pretrained("./ckpts/t2i/mt5")


..parsed-literal::

Youareusingthedefaultlegacybehaviourofthe<class'transformers.models.t5.tokenization_t5.T5Tokenizer'>.Thisisexpected,andsimplymeansthatthe`legacy`(previous)behaviorwillbeusedsonothingchangesforyou.Ifyouwanttousethenewbehaviour,set`legacy=False`.Thisshouldonlybesetifyouunderstandwhatitmeans,andthoroughlyreadthereasonwhythiswasaddedasexplainedinhttps://github.com/huggingface/transformers/pull/24565
/home/ea/work/notebooks_env/lib/python3.8/site-packages/transformers/convert_slow_tokenizer.py:562:UserWarning:Thesentencepiecetokenizerthatyouareconvertingtoafasttokenizerusesthebytefallbackoptionwhichisnotimplementedinthefasttokenizers.Inpracticethismeansthatthefastversionofthetokenizercanproduceunknowntokenswhereasthesentencepieceversionwouldhaveconvertedtheseunknowntokensintoasequenceofbytetokensmatchingtheoriginalpieceoftext.
warnings.warn(


..code::ipython3

fromhydit.constantsimportSAMPLER_FACTORY,NEGATIVE_PROMPT

..code::ipython3

sampler="ddpm"
kwargs=SAMPLER_FACTORY[sampler]["kwargs"]
scheduler=SAMPLER_FACTORY[sampler]["scheduler"]

..code::ipython3

fromdiffusersimportschedulers

scheduler_class=getattr(schedulers,scheduler)
scheduler=scheduler_class(**kwargs)

..code::ipython3

ov_pipe=OVHyDiTPipeline(vae_decoder,text_encoder,tokenizer,dit,scheduler,None,None,embedder_t5=text_embedder,embedder_tokenizer=embedder_tokenizer)

..code::ipython3

fromhydit.modules.posemb_layersimportget_2d_rotary_pos_embed,get_fill_resize_and_crop


defcalc_rope(height,width,patch_size=2,head_size=88):
th=height//8//patch_size
tw=width//8//patch_size
base_size=512//8//patch_size
start,stop=get_fill_resize_and_crop((th,tw),base_size)
sub_args=[start,stop,(th,tw)]
rope=get_2d_rotary_pos_embed(head_size,*sub_args)
returnrope

..code::ipython3

fromhydit.utils.toolsimportset_seeds

height,width=880,880
style=torch.as_tensor([0,0])
target_height=int((height//16)*16)
target_width=int((width//16)*16)

size_cond=[height,width,target_width,target_height,0,0]
image_meta_size=torch.as_tensor([size_cond]*2)
freqs_cis_img_cache={}

if(target_height,target_width)notinfreqs_cis_img_cache:
freqs_cis_img_cache[(target_height,target_width)]=calc_rope(target_height,target_width)

freqs_cis_img=freqs_cis_img_cache[(target_height,target_width)]
images=ov_pipe(
prompt="cutecat",
negative_prompt=NEGATIVE_PROMPT,
height=target_height,
width=target_width,
num_inference_steps=10,
image_meta_size=image_meta_size,
style=style,
return_dict=False,
guidance_scale=7.5,
freqs_cis_img=freqs_cis_img,
generator=set_seeds(42),
)



..parsed-literal::

0%||0/10[00:00<?,?it/s]


..code::ipython3

images[0][0]




..image::hunyuan-dit-image-generation-with-output_files/hunyuan-dit-image-generation-with-output_30_0.png



Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


definference(input_prompt,negative_prompt,seed,num_steps,height,width,progress=gr.Progress(track_tqdm=True)):
style=torch.as_tensor([0,0])
target_height=int((height//16)*16)
target_width=int((width//16)*16)

size_cond=[height,width,target_width,target_height,0,0]
image_meta_size=torch.as_tensor([size_cond]*2)
freqs_cis_img=calc_rope(target_height,target_width)
images=ov_pipe(
prompt=input_prompt,
negative_prompt=negative_prompt,
height=target_height,
width=target_width,
num_inference_steps=num_steps,
image_meta_size=image_meta_size,
style=style,
return_dict=False,
guidance_scale=7.5,
freqs_cis_img=freqs_cis_img,
generator=set_seeds(seed),
)
returnimages[0][0]


withgr.Blocks()asdemo:
withgr.Row():
withgr.Column():
prompt=gr.Textbox(label="Inputprompt",lines=3)
withgr.Row():
infer_steps=gr.Slider(
label="NumberInferencesteps",
minimum=1,
maximum=200,
value=15,
step=1,
)
seed=gr.Number(
label="Seed",
minimum=-1,
maximum=1_000_000_000,
value=42,
step=1,
precision=0,
)
withgr.Accordion("Advancedsettings",open=False):
withgr.Row():
negative_prompt=gr.Textbox(
label="Negativeprompt",
value=NEGATIVE_PROMPT,
lines=2,
)
withgr.Row():
oriW=gr.Number(
label="Width",
minimum=768,
maximum=1024,
value=880,
step=16,
precision=0,
min_width=80,
)
oriH=gr.Number(
label="Height",
minimum=768,
maximum=1024,
value=880,
step=16,
precision=0,
min_width=80,
)
cfg_scale=gr.Slider(label="Guidancescale",minimum=1.0,maximum=16.0,value=7.5,step=0.5)
withgr.Row():
advanced_button=gr.Button()
withgr.Column():
output_img=gr.Image(
label="Generatedimage",
interactive=False,
)
advanced_button.click(
fn=inference,
inputs=[
prompt,
negative_prompt,
seed,
infer_steps,
oriH,
oriW,
],
outputs=output_img,
)

withgr.Row():
gr.Examples(
[
["一只小猫"],
["akitten"],
["一只聪明的狐狸走在阔叶树林里,旁边是一条小溪,细节真实,摄影"],
["Acleverfoxwalksinabroadleafforestnexttoastream,realisticdetails,photography"],
["请将“杞人忧天”的样子画出来"],
['Pleasedrawapictureof"unfoundedworries"'],
["枯藤老树昏鸦，小桥流水人家"],
["Witheredvines,oldtreesanddimcrows,smallbridgesandflowingwater,people'shouses"],
["湖水清澈，天空湛蓝，阳光灿烂。一只优雅的白天鹅在湖边游泳。它周围有几只小鸭子，看起来非常可爱，整个画面给人一种宁静祥和的感觉。"],
[
"Thelakeisclear,theskyisblue,andthesunisbright.Anelegantwhiteswanswimsbythelake.Thereareseverallittleducksaroundit,whichlookverycute,andthewholepicturegivespeopleasenseofpeaceandtranquility."
],
["一朵鲜艳的红色玫瑰花，花瓣撒有一些水珠，晶莹剔透，特写镜头"],
["Abrightredroseflowerwithpetalssprinkledwithsomewaterdrops,crystalclear,close-up"],
["风格是写实，画面主要描述一个亚洲戏曲艺术家正在表演，她穿着华丽的戏服，脸上戴着精致的面具，身姿优雅，背景是古色古香的舞台，镜头是近景"],
[
"Thestyleisrealistic.ThepicturemainlydepictsanAsianoperaartistperforming.Sheiswearingagorgeouscostumeandadelicatemaskonherface.Herpostureiselegant.Thebackgroundisanantiquestageandthecameraisaclose-up."
],
],
[prompt],
)

try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>


..parsed-literal::

Keyboardinterruptioninmainthread...closingserver.

