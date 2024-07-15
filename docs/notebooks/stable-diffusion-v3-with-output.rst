ImagegenerationwithStableDiffusionv3andOpenVINO
======================================================

StableDiffusionV3isnextgenerationoflatentdiffusionimageStable
Diffusionmodelsfamilythatoutperformsstate-of-the-arttext-to-image
generationsystemsintypographyandpromptadherence,basedonhuman
preferenceevaluations.Incomparisonwithpreviousversions,itbased
onMultimodalDiffusionTransformer(MMDiT)text-to-imagemodelthat
featuresgreatlyimprovedperformanceinimagequality,typography,
complexpromptunderstanding,andresource-efficiency.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/dd079427-89f2-4d28-a10e-c80792d750bf
:alt:mmdit.png

mmdit.png

Moredetailsaboutmodelcanbefoundin`model
card<https://huggingface.co/stabilityai/stable-diffusion-3-medium>`__,
`research
paper<https://stability.ai/news/stable-diffusion-3-research-paper>`__
and`Stability.AIblog
post<https://stability.ai/news/stable-diffusion-3-medium>`__.Inthis
tutorial,wewillconsiderhowtoconvertandoptimizeStableDiffusion
v3forrunningwithOpenVINO.IfyouwanttorunpreviousStable
Diffusionversions,pleasecheckourothernotebooks:

-`StableDiffusion<../stable-diffusion-text-to-image>`__
-`StableDiffusionv2<../stable-diffusion-v2>`__
-`StableDiffusionXL<../stable-diffusion-xl>`__
-`LCMStable
Diffusion<../latent-consistency-models-image-generation>`__
-`TurboSDXL<../sdxl-turbo>`__
-`TurboSD<../sketch-to-image-pix2pix-turbo>`__

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`BuildPyTorchpipeline<#build-pytorch-pipeline>`__
-`ConvertandOptimizemodelswithOpenVINOand
NNCF<#convert-and-optimize-models-with-openvino-and-nncf>`__

-`Transformer<#transformer>`__
-`T5TextEncoder<#t5-text-encoder>`__
-`Cliptextencoders<#clip-text-encoders>`__
-`VAE<#vae>`__

-`PrepareOpenVINOinference
pipeline<#prepare-openvino-inference-pipeline>`__
-`RunOpenVINOmodel<#run-openvino-model>`__
-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"git+https://github.com/initml/diffusers.git@clement/feature/flash_sd3""gradio>=4.19""torch>=2.1""transformers""nncf>=2.11.0""opencv-python""pillow""peft>=0.7.0"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-qU--pre"openvino"--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly

BuildPyTorchpipeline
----------------------

`backtotop⬆️<#table-of-contents>`__

**Note**:runmodelwithnotebook,youwillneedtoacceptlicense
agreement.Youmustbearegistereduserin🤗HuggingFaceHub.
Pleasevisit`HuggingFacemodel
card<https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers>`__,
carefullyreadtermsofusageandclickacceptbutton.Youwillneed
touseanaccesstokenforthecodebelowtorun.Formore
informationonaccesstokens,referto`thissectionofthe
documentation<https://huggingface.co/docs/hub/security-tokens>`__.
YoucanloginonHuggingFaceHubinnotebookenvironment,using
followingcode:

..code::ipython3

#uncommenttheselinestologintohuggingfacehubtogetaccesstopretrainedmodel

#fromhuggingface_hubimportnotebook_login,whoami

#try:
#whoami()
#print('Authorizationtokenalreadyprovided')
#exceptOSError:
#notebook_login()

Wewilluse
`Diffusers<https://huggingface.co/docs/diffusers/main/en/index>`__
libraryintegrationforrunningStableDiffusionv3model.Youcanfind
moredetailsinDiffusers
`documentation<https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3>`__.
Additionally,wecanapplyoptimizationforpipelineperformanceand
memoryconsumption:

-**UseflashSD3**.FlashDiffusionisadiffusiondistillationmethod
proposedin`FlashDiffusion:AcceleratingAnyConditionalDiffusion
ModelforFewStepsImage
Generation<http://arxiv.org/abs/2406.02347>`__.Themodel
representedasa90.4MLoRAdistilledversionofSD3modelthatis
abletogenerate1024x1024imagesin4steps.Ifyouwantdisableit,
youcanunsetcheckbox**UseflashSD3**
-**RemoveT5textencoder**.Removingthememory-intensive4.7B
parameterT5-XXLtextencoderduringinferencecansignificantly
decreasethememoryrequirementsforSD3withonlyaslightlossin
performance.Ifyouwanttousethismodelinpipeline,pleaseset
**uset5textencoder**checkbox.

..code::ipython3

importipywidgetsaswidgets

use_flash_lora=widgets.Checkbox(
value=True,
description="UseflashSD3",
disabled=False,
)

load_t5=widgets.Checkbox(
value=False,
description="Uset5textencoder",
disabled=False,
)

pt_pipeline_options=widgets.VBox([use_flash_lora,load_t5])
display(pt_pipeline_options)



..parsed-literal::

VBox(children=(Checkbox(value=True,description='UseflashSD3'),Checkbox(value=False,description='Uset5te…


..code::ipython3

frompathlibimportPath
importtorch
fromdiffusersimportStableDiffusion3Pipeline,SD3Transformer2DModel
frompeftimportPeftModel


MODEL_DIR=Path("stable-diffusion-3")
MODEL_DIR.mkdir(exist_ok=True)

TRANSFORMER_PATH=MODEL_DIR/"transformer.xml"
VAE_DECODER_PATH=MODEL_DIR/"vae_decoder.xml"
TEXT_ENCODER_PATH=MODEL_DIR/"text_encoder.xml"
TEXT_ENCODER_2_PATH=MODEL_DIR/"text_encoder_2.xml"
TEXT_ENCODER_3_PATH=MODEL_DIR/"text_encoder_3.xml"

conversion_statuses=[TRANSFORMER_PATH.exists(),VAE_DECODER_PATH.exists(),TEXT_ENCODER_PATH.exists(),TEXT_ENCODER_2_PATH.exists()]

ifload_t5.value:
conversion_statuses.append(TEXT_ENCODER_3_PATH.exists())

requires_conversion=notall(conversion_statuses)

transformer,vae,text_encoder,text_encoder_2,text_encoder_3=None,None,None,None,None


defget_pipeline_components():
pipe_kwargs={}
ifuse_flash_lora.value:
#LoadLoRA
transformer=SD3Transformer2DModel.from_pretrained(
"stabilityai/stable-diffusion-3-medium-diffusers",
subfolder="transformer",
)
transformer=PeftModel.from_pretrained(transformer,"jasperai/flash-sd3")
pipe_kwargs["transformer"]=transformer
ifnotload_t5.value:
pipe_kwargs.update({"text_encoder_3":None,"tokenizer_3":None})
pipe=StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",**pipe_kwargs)
pipe.tokenizer.save_pretrained(MODEL_DIR/"tokenizer")
pipe.tokenizer_2.save_pretrained(MODEL_DIR/"tokenizer_2")
ifload_t5.value:
pipe.tokenizer_3.save_pretrained(MODEL_DIR/"tokenizer_3")
pipe.scheduler.save_pretrained(MODEL_DIR/"scheduler")
transformer,vae,text_encoder,text_encoder_2,text_encoder_3=None,None,None,None,None
ifnotTRANSFORMER_PATH.exists():
transformer=pipe.transformer
transformer.eval()
ifnotVAE_DECODER_PATH.exists():
vae=pipe.vae
vae.eval()
ifnotTEXT_ENCODER_PATH.exists():
text_encoder=pipe.text_encoder
text_encoder.eval()
ifnotTEXT_ENCODER_2_PATH.exists():
text_encoder_2=pipe.text_encoder_2
text_encoder_2.eval()
ifnotTEXT_ENCODER_3_PATH.exists()andload_t5.value:
text_encoder_3=pipe.text_encoder_3
text_encoder_3.eval()
returntransformer,vae,text_encoder,text_encoder_2,text_encoder_3


ifrequires_conversion:
transformer,vae,text_encoder,text_encoder_2,text_encoder_3=get_pipeline_components()


..parsed-literal::

/home/ea/work/notebooks_env/lib/python3.8/site-packages/diffusers/models/transformers/transformer_2d.py:34:FutureWarning:`Transformer2DModelOutput`isdeprecatedandwillberemovedinversion1.0.0.Importing`Transformer2DModelOutput`from`diffusers.models.transformer_2d`isdeprecatedandthiswillberemovedinafutureversion.Pleaseuse`fromdiffusers.models.modeling_outputsimportTransformer2DModelOutput`,instead.
deprecate("Transformer2DModelOutput","1.0.0",deprecation_message)


ConvertandOptimizemodelswithOpenVINOandNNCF
--------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Startingfrom2023.0release,OpenVINOsupportsPyTorchmodelsdirectly
viaModelConversionAPI.``ov.convert_model``functionacceptsinstance
ofPyTorchmodelandexampleinputsfortracingandreturnsobjectof
``ov.Model``class,readytouseorsaveondiskusing``ov.save_model``
function.

Thepipelineconsistsoffourimportantparts:

-ClipandT5TextEncoderstocreateconditiontogenerateanimage
fromatextprompt.
-Transformerforstep-by-stepdenoisinglatentimagerepresentation.
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

Transformer
~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importopenvinoasov
fromfunctoolsimportpartial
importgc


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


classTransformerWrapper(torch.nn.Module):
def__init__(self,model):
super().__init__()
self.model=model

defforward(self,hidden_states,encoder_hidden_states,pooled_projections,timestep,return_dict=False):
returnself.model(
hidden_states=hidden_states,
encoder_hidden_states=encoder_hidden_states,
pooled_projections=pooled_projections,
timestep=timestep,
return_dict=return_dict,
)


ifnotTRANSFORMER_PATH.exists():
ifisinstance(transformer,PeftModel):
transformer=TransformerWrapper(transformer)
transformer.forward=partial(transformer.forward,return_dict=False)

withtorch.no_grad():
ov_model=ov.convert_model(
transformer,
example_input={
"hidden_states":torch.zeros((2,16,64,64)),
"timestep":torch.tensor([1,1]),
"encoder_hidden_states":torch.ones([2,154,4096]),
"pooled_projections":torch.ones([2,2048]),
},
)
ov.save_model(ov_model,TRANSFORMER_PATH)
delov_model
cleanup_torchscript_cache()

deltransformer
gc.collect()




..parsed-literal::

20



..code::ipython3

importipywidgetsaswidgets

to_compress_weights=widgets.Checkbox(
value=True,
description="WeightsCompression",
disabled=False,
)

to_compress_weights




..parsed-literal::

Checkbox(value=True,description='WeightsCompression')



..code::ipython3

importnncf

core=ov.Core()

TRANSFORMER_INT4_PATH=MODEL_DIR/"transformer_int4.xml"

ifto_compress_weights.valueandnotTRANSFORMER_INT4_PATH.exists():
transformer=core.read_model(TRANSFORMER_PATH)
compressed_transformer=nncf.compress_weights(transformer,mode=nncf.CompressWeightsMode.INT4_SYM,ratio=0.8,group_size=64)
ov.save_model(compressed_transformer,TRANSFORMER_INT4_PATH)
delcompressed_transformer
deltransformer
gc.collect()

ifTRANSFORMER_INT4_PATH.exists():
fp16_ir_model_size=TRANSFORMER_PATH.with_suffix(".bin").stat().st_size/1024
compressed_model_size=TRANSFORMER_INT4_PATH.with_suffix(".bin").stat().st_size/1024

print(f"FP16modelsize:{fp16_ir_model_size:.2f}KB")
print(f"INT8modelsize:{compressed_model_size:.2f}KB")
print(f"Modelcompressionrate:{fp16_ir_model_size/compressed_model_size:.3f}")


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,onnx,openvino
FP16modelsize:4243354.63KB
INT8modelsize:1411706.74KB
Modelcompressionrate:3.006


T5TextEncoder
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotTEXT_ENCODER_3_PATH.exists()andload_t5.value:
withtorch.no_grad():
ov_model=ov.convert_model(text_encoder_3,example_input=torch.ones([1,77],dtype=torch.long))
ov.save_model(ov_model,TEXT_ENCODER_3_PATH)
delov_model
cleanup_torchscript_cache()

deltext_encoder_3
gc.collect()




..parsed-literal::

11



..code::ipython3

ifload_t5.value:
display(to_compress_weights)

..code::ipython3

TEXT_ENCODER_3_INT4_PATH=MODEL_DIR/"text_encoder_3_int4.xml"

ifload_t5.valueandto_compress_weights.valueandnotTEXT_ENCODER_3_INT4_PATH.exists():
encoder=core.read_model(TEXT_ENCODER_3_PATH)
compressed_encoder=nncf.compress_weights(encoder,mode=nncf.CompressWeightsMode.INT4_SYM,ratio=0.8,group_size=64)
ov.save_model(compressed_encoder,TEXT_ENCODER_3_INT4_PATH)
delcompressed_encoder
delencoder
gc.collect()

ifTEXT_ENCODER_3_INT4_PATH.exists():
fp16_ir_model_size=TEXT_ENCODER_3_PATH.with_suffix(".bin").stat().st_size/1024
compressed_model_size=TEXT_ENCODER_3_INT4_PATH.with_suffix(".bin").stat().st_size/1024

print(f"FP16modelsize:{fp16_ir_model_size:.2f}KB")
print(f"INT8modelsize:{compressed_model_size:.2f}KB")
print(f"Modelcompressionrate:{fp16_ir_model_size/compressed_model_size:.3f}")

Cliptextencoders
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotTEXT_ENCODER_PATH.exists():
withtorch.no_grad():
text_encoder.forward=partial(text_encoder.forward,output_hidden_states=True,return_dict=False)
ov_model=ov.convert_model(text_encoder,example_input=torch.ones([1,77],dtype=torch.long))
ov.save_model(ov_model,TEXT_ENCODER_PATH)
delov_model
cleanup_torchscript_cache()

deltext_encoder
gc.collect()




..parsed-literal::

0



..code::ipython3

ifnotTEXT_ENCODER_2_PATH.exists():
withtorch.no_grad():
text_encoder_2.forward=partial(text_encoder_2.forward,output_hidden_states=True,return_dict=False)
ov_model=ov.convert_model(text_encoder_2,example_input=torch.ones([1,77],dtype=torch.long))
ov.save_model(ov_model,TEXT_ENCODER_2_PATH)
delov_model
cleanup_torchscript_cache()

deltext_encoder_2
gc.collect()




..parsed-literal::

0



VAE
~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotVAE_DECODER_PATH.exists():
withtorch.no_grad():
vae.forward=vae.decode
ov_model=ov.convert_model(vae,example_input=torch.ones([1,16,64,64]))
ov.save_model(ov_model,VAE_DECODER_PATH)

delvae
gc.collect()




..parsed-literal::

0



PrepareOpenVINOinferencepipeline
-----------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importinspect
fromtypingimportCallable,Dict,List,Optional,Union

importtorch
fromtransformersimport(
CLIPTextModelWithProjection,
CLIPTokenizer,
T5EncoderModel,
T5TokenizerFast,
)

fromdiffusers.image_processorimportVaeImageProcessor
fromdiffusers.models.autoencodersimportAutoencoderKL
fromdiffusers.schedulersimportFlowMatchEulerDiscreteScheduler
fromdiffusers.utilsimport(
logging,
)
fromdiffusers.utils.torch_utilsimportrandn_tensor
fromdiffusers.pipelines.pipeline_utilsimportDiffusionPipeline
fromdiffusers.pipelines.stable_diffusion_3.pipeline_outputimportStableDiffusion3PipelineOutput


logger=logging.get_logger(__name__)#pylint:disable=invalid-name


#Copiedfromdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
defretrieve_timesteps(
scheduler,
num_inference_steps:Optional[int]=None,
device:Optional[Union[str,torch.device]]=None,
timesteps:Optional[List[int]]=None,
sigmas:Optional[List[float]]=None,
**kwargs,
):
"""
Callsthescheduler's`set_timesteps`methodandretrievestimestepsfromtheschedulerafterthecall.Handles
customtimesteps.Anykwargswillbesuppliedto`scheduler.set_timesteps`.

Args:
scheduler(`SchedulerMixin`):
Theschedulertogettimestepsfrom.
num_inference_steps(`int`):
Thenumberofdiffusionstepsusedwhengeneratingsampleswithapre-trainedmodel.Ifused,`timesteps`
mustbe`None`.
device(`str`or`torch.device`,*optional*):
Thedevicetowhichthetimestepsshouldbemovedto.If`None`,thetimestepsarenotmoved.
timesteps(`List[int]`,*optional*):
Customtimestepsusedtooverridethetimestepspacingstrategyofthescheduler.If`timesteps`ispassed,
`num_inference_steps`and`sigmas`mustbe`None`.
sigmas(`List[float]`,*optional*):
Customsigmasusedtooverridethetimestepspacingstrategyofthescheduler.If`sigmas`ispassed,
`num_inference_steps`and`timesteps`mustbe`None`.

Returns:
`Tuple[torch.Tensor,int]`:Atuplewherethefirstelementisthetimestepschedulefromtheschedulerandthe
secondelementisthenumberofinferencesteps.
"""
iftimestepsisnotNoneandsigmasisnotNone:
raiseValueError("Onlyoneof`timesteps`or`sigmas`canbepassed.Pleasechooseonetosetcustomvalues")
iftimestepsisnotNone:
accepts_timesteps="timesteps"inset(inspect.signature(scheduler.set_timesteps).parameters.keys())
ifnotaccepts_timesteps:
raiseValueError(
f"Thecurrentschedulerclass{scheduler.__class__}'s`set_timesteps`doesnotsupportcustom"
f"timestepschedules.Pleasecheckwhetheryouareusingthecorrectscheduler."
)
scheduler.set_timesteps(timesteps=timesteps,device=device,**kwargs)
timesteps=scheduler.timesteps
num_inference_steps=len(timesteps)
elifsigmasisnotNone:
accept_sigmas="sigmas"inset(inspect.signature(scheduler.set_timesteps).parameters.keys())
ifnotaccept_sigmas:
raiseValueError(
f"Thecurrentschedulerclass{scheduler.__class__}'s`set_timesteps`doesnotsupportcustom"
f"sigmasschedules.Pleasecheckwhetheryouareusingthecorrectscheduler."
)
scheduler.set_timesteps(sigmas=sigmas,device=device,**kwargs)
timesteps=scheduler.timesteps
num_inference_steps=len(timesteps)
else:
scheduler.set_timesteps(num_inference_steps,device=device,**kwargs)
timesteps=scheduler.timesteps
returntimesteps,num_inference_steps


classOVStableDiffusion3Pipeline(DiffusionPipeline):
r"""
Args:
transformer([`SD3Transformer2DModel`]):
ConditionalTransformer(MMDiT)architecturetodenoisetheencodedimagelatents.
scheduler([`FlowMatchEulerDiscreteScheduler`]):
Aschedulertobeusedincombinationwith`transformer`todenoisetheencodedimagelatents.
vae([`AutoencoderKL`]):
VariationalAuto-Encoder(VAE)Modeltoencodeanddecodeimagestoandfromlatentrepresentations.
text_encoder([`CLIPTextModelWithProjection`]):
[CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
specificallythe[clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)variant,
withanadditionaladdedprojectionlayerthatisinitializedwithadiagonalmatrixwiththe`hidden_size`
asitsdimension.
text_encoder_2([`CLIPTextModelWithProjection`]):
[CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
specificallythe
[laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
variant.
text_encoder_3([`T5EncoderModel`]):
Frozentext-encoder.StableDiffusion3uses
[T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel),specificallythe
[t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl)variant.
tokenizer(`CLIPTokenizer`):
Tokenizerofclass
[CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
tokenizer_2(`CLIPTokenizer`):
SecondTokenizerofclass
[CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
tokenizer_3(`T5TokenizerFast`):
Tokenizerofclass
[T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
"""

_optional_components=[]
_callback_tensor_inputs=["latents","prompt_embeds","negative_prompt_embeds","negative_pooled_prompt_embeds"]

def__init__(
self,
transformer:SD3Transformer2DModel,
scheduler:FlowMatchEulerDiscreteScheduler,
vae:AutoencoderKL,
text_encoder:CLIPTextModelWithProjection,
tokenizer:CLIPTokenizer,
text_encoder_2:CLIPTextModelWithProjection,
tokenizer_2:CLIPTokenizer,
text_encoder_3:T5EncoderModel,
tokenizer_3:T5TokenizerFast,
):
super().__init__()

self.register_modules(
vae=vae,
text_encoder=text_encoder,
text_encoder_2=text_encoder_2,
text_encoder_3=text_encoder_3,
tokenizer=tokenizer,
tokenizer_2=tokenizer_2,
tokenizer_3=tokenizer_3,
transformer=transformer,
scheduler=scheduler,
)
self.vae_scale_factor=2**3
self.image_processor=VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
self.tokenizer_max_length=self.tokenizer.model_max_lengthifhasattr(self,"tokenizer")andself.tokenizerisnotNoneelse77
self.vae_scaling_factor=1.5305
self.vae_shift_factor=0.0609
self.default_sample_size=64

def_get_t5_prompt_embeds(
self,
prompt:Union[str,List[str]]=None,
num_images_per_prompt:int=1,
):
prompt=[prompt]ifisinstance(prompt,str)elseprompt
batch_size=len(prompt)

ifself.text_encoder_3isNone:
returntorch.zeros(
(batch_size,self.tokenizer_max_length,4096),
)

text_inputs=self.tokenizer_3(
prompt,
padding="max_length",
max_length=self.tokenizer_max_length,
truncation=True,
add_special_tokens=True,
return_tensors="pt",
)
text_input_ids=text_inputs.input_ids
prompt_embeds=torch.from_numpy(self.text_encoder_3(text_input_ids)[0])
_,seq_len,_=prompt_embeds.shape
prompt_embeds=prompt_embeds.repeat(1,num_images_per_prompt,1)
prompt_embeds=prompt_embeds.view(batch_size*num_images_per_prompt,seq_len,-1)

returnprompt_embeds

def_get_clip_prompt_embeds(
self,
prompt:Union[str,List[str]],
num_images_per_prompt:int=1,
clip_skip:Optional[int]=None,
clip_model_index:int=0,
):
clip_tokenizers=[self.tokenizer,self.tokenizer_2]
clip_text_encoders=[self.text_encoder,self.text_encoder_2]

tokenizer=clip_tokenizers[clip_model_index]
text_encoder=clip_text_encoders[clip_model_index]

prompt=[prompt]ifisinstance(prompt,str)elseprompt
batch_size=len(prompt)

text_inputs=tokenizer(prompt,padding="max_length",max_length=self.tokenizer_max_length,truncation=True,return_tensors="pt")

text_input_ids=text_inputs.input_ids
prompt_embeds=text_encoder(text_input_ids)
pooled_prompt_embeds=torch.from_numpy(prompt_embeds[0])
hidden_states=list(prompt_embeds.values())[1:]

ifclip_skipisNone:
prompt_embeds=torch.from_numpy(hidden_states[-2])
else:
prompt_embeds=torch.from_numpy(hidden_states[-(clip_skip+2)])

_,seq_len,_=prompt_embeds.shape
prompt_embeds=prompt_embeds.repeat(1,num_images_per_prompt,1)
prompt_embeds=prompt_embeds.view(batch_size*num_images_per_prompt,seq_len,-1)

pooled_prompt_embeds=pooled_prompt_embeds.repeat(1,num_images_per_prompt,1)
pooled_prompt_embeds=pooled_prompt_embeds.view(batch_size*num_images_per_prompt,-1)

returnprompt_embeds,pooled_prompt_embeds

defencode_prompt(
self,
prompt:Union[str,List[str]],
prompt_2:Union[str,List[str]],
prompt_3:Union[str,List[str]],
num_images_per_prompt:int=1,
do_classifier_free_guidance:bool=True,
negative_prompt:Optional[Union[str,List[str]]]=None,
negative_prompt_2:Optional[Union[str,List[str]]]=None,
negative_prompt_3:Optional[Union[str,List[str]]]=None,
prompt_embeds:Optional[torch.FloatTensor]=None,
negative_prompt_embeds:Optional[torch.FloatTensor]=None,
pooled_prompt_embeds:Optional[torch.FloatTensor]=None,
negative_pooled_prompt_embeds:Optional[torch.FloatTensor]=None,
clip_skip:Optional[int]=None,
):
prompt=[prompt]ifisinstance(prompt,str)elseprompt
ifpromptisnotNone:
batch_size=len(prompt)
else:
batch_size=prompt_embeds.shape[0]

ifprompt_embedsisNone:
prompt_2=prompt_2orprompt
prompt_2=[prompt_2]ifisinstance(prompt_2,str)elseprompt_2

prompt_3=prompt_3orprompt
prompt_3=[prompt_3]ifisinstance(prompt_3,str)elseprompt_3

prompt_embed,pooled_prompt_embed=self._get_clip_prompt_embeds(
prompt=prompt,
num_images_per_prompt=num_images_per_prompt,
clip_skip=clip_skip,
clip_model_index=0,
)
prompt_2_embed,pooled_prompt_2_embed=self._get_clip_prompt_embeds(
prompt=prompt_2,
num_images_per_prompt=num_images_per_prompt,
clip_skip=clip_skip,
clip_model_index=1,
)
clip_prompt_embeds=torch.cat([prompt_embed,prompt_2_embed],dim=-1)

t5_prompt_embed=self._get_t5_prompt_embeds(
prompt=prompt_3,
num_images_per_prompt=num_images_per_prompt,
)

clip_prompt_embeds=torch.nn.functional.pad(clip_prompt_embeds,(0,t5_prompt_embed.shape[-1]-clip_prompt_embeds.shape[-1]))

prompt_embeds=torch.cat([clip_prompt_embeds,t5_prompt_embed],dim=-2)
pooled_prompt_embeds=torch.cat([pooled_prompt_embed,pooled_prompt_2_embed],dim=-1)

ifdo_classifier_free_guidanceandnegative_prompt_embedsisNone:
negative_prompt=negative_promptor""
negative_prompt_2=negative_prompt_2ornegative_prompt
negative_prompt_3=negative_prompt_3ornegative_prompt

#normalizestrtolist
negative_prompt=batch_size*[negative_prompt]ifisinstance(negative_prompt,str)elsenegative_prompt
negative_prompt_2=batch_size*[negative_prompt_2]ifisinstance(negative_prompt_2,str)elsenegative_prompt_2
negative_prompt_3=batch_size*[negative_prompt_3]ifisinstance(negative_prompt_3,str)elsenegative_prompt_3

ifpromptisnotNoneandtype(prompt)isnottype(negative_prompt):
raiseTypeError(f"`negative_prompt`shouldbethesametypeto`prompt`,butgot{type(negative_prompt)}!="f"{type(prompt)}.")
elifbatch_size!=len(negative_prompt):
raiseValueError(
f"`negative_prompt`:{negative_prompt}hasbatchsize{len(negative_prompt)},but`prompt`:"
f"{prompt}hasbatchsize{batch_size}.Pleasemakesurethatpassed`negative_prompt`matches"
"thebatchsizeof`prompt`."
)

negative_prompt_embed,negative_pooled_prompt_embed=self._get_clip_prompt_embeds(
negative_prompt,
num_images_per_prompt=num_images_per_prompt,
clip_skip=None,
clip_model_index=0,
)
negative_prompt_2_embed,negative_pooled_prompt_2_embed=self._get_clip_prompt_embeds(
negative_prompt_2,
num_images_per_prompt=num_images_per_prompt,
clip_skip=None,
clip_model_index=1,
)
negative_clip_prompt_embeds=torch.cat([negative_prompt_embed,negative_prompt_2_embed],dim=-1)

t5_negative_prompt_embed=self._get_t5_prompt_embeds(prompt=negative_prompt_3,num_images_per_prompt=num_images_per_prompt)

negative_clip_prompt_embeds=torch.nn.functional.pad(
negative_clip_prompt_embeds,
(0,t5_negative_prompt_embed.shape[-1]-negative_clip_prompt_embeds.shape[-1]),
)

negative_prompt_embeds=torch.cat([negative_clip_prompt_embeds,t5_negative_prompt_embed],dim=-2)
negative_pooled_prompt_embeds=torch.cat([negative_pooled_prompt_embed,negative_pooled_prompt_2_embed],dim=-1)

returnprompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds

defcheck_inputs(
self,
prompt,
prompt_2,
prompt_3,
height,
width,
negative_prompt=None,
negative_prompt_2=None,
negative_prompt_3=None,
prompt_embeds=None,
negative_prompt_embeds=None,
pooled_prompt_embeds=None,
negative_pooled_prompt_embeds=None,
callback_on_step_end_tensor_inputs=None,
):
ifheight%8!=0orwidth%8!=0:
raiseValueError(f"`height`and`width`havetobedivisibleby8butare{height}and{width}.")

ifcallback_on_step_end_tensor_inputsisnotNoneandnotall(kinself._callback_tensor_inputsforkincallback_on_step_end_tensor_inputs):
raiseValueError(
f"`callback_on_step_end_tensor_inputs`hastobein{self._callback_tensor_inputs},butfound{[kforkincallback_on_step_end_tensor_inputsifknotinself._callback_tensor_inputs]}"
)

ifpromptisnotNoneandprompt_embedsisnotNone:
raiseValueError(
f"Cannotforwardboth`prompt`:{prompt}and`prompt_embeds`:{prompt_embeds}.Pleasemakesureto""onlyforwardoneofthetwo."
)
elifprompt_2isnotNoneandprompt_embedsisnotNone:
raiseValueError(
f"Cannotforwardboth`prompt_2`:{prompt_2}and`prompt_embeds`:{prompt_embeds}.Pleasemakesureto""onlyforwardoneofthetwo."
)
elifprompt_3isnotNoneandprompt_embedsisnotNone:
raiseValueError(
f"Cannotforwardboth`prompt_3`:{prompt_2}and`prompt_embeds`:{prompt_embeds}.Pleasemakesureto""onlyforwardoneofthetwo."
)
elifpromptisNoneandprompt_embedsisNone:
raiseValueError("Provideeither`prompt`or`prompt_embeds`.Cannotleaveboth`prompt`and`prompt_embeds`undefined.")
elifpromptisnotNoneand(notisinstance(prompt,str)andnotisinstance(prompt,list)):
raiseValueError(f"`prompt`hastobeoftype`str`or`list`butis{type(prompt)}")
elifprompt_2isnotNoneand(notisinstance(prompt_2,str)andnotisinstance(prompt_2,list)):
raiseValueError(f"`prompt_2`hastobeoftype`str`or`list`butis{type(prompt_2)}")
elifprompt_3isnotNoneand(notisinstance(prompt_3,str)andnotisinstance(prompt_3,list)):
raiseValueError(f"`prompt_3`hastobeoftype`str`or`list`butis{type(prompt_3)}")

ifnegative_promptisnotNoneandnegative_prompt_embedsisnotNone:
raiseValueError(
f"Cannotforwardboth`negative_prompt`:{negative_prompt}and`negative_prompt_embeds`:"
f"{negative_prompt_embeds}.Pleasemakesuretoonlyforwardoneofthetwo."
)
elifnegative_prompt_2isnotNoneandnegative_prompt_embedsisnotNone:
raiseValueError(
f"Cannotforwardboth`negative_prompt_2`:{negative_prompt_2}and`negative_prompt_embeds`:"
f"{negative_prompt_embeds}.Pleasemakesuretoonlyforwardoneofthetwo."
)
elifnegative_prompt_3isnotNoneandnegative_prompt_embedsisnotNone:
raiseValueError(
f"Cannotforwardboth`negative_prompt_3`:{negative_prompt_3}and`negative_prompt_embeds`:"
f"{negative_prompt_embeds}.Pleasemakesuretoonlyforwardoneofthetwo."
)

ifprompt_embedsisnotNoneandnegative_prompt_embedsisnotNone:
ifprompt_embeds.shape!=negative_prompt_embeds.shape:
raiseValueError(
"`prompt_embeds`and`negative_prompt_embeds`musthavethesameshapewhenpasseddirectly,but"
f"got:`prompt_embeds`{prompt_embeds.shape}!=`negative_prompt_embeds`"
f"{negative_prompt_embeds.shape}."
)

ifprompt_embedsisnotNoneandpooled_prompt_embedsisNone:
raiseValueError(
"If`prompt_embeds`areprovided,`pooled_prompt_embeds`alsohavetobepassed.Makesuretogenerate`pooled_prompt_embeds`fromthesametextencoderthatwasusedtogenerate`prompt_embeds`."
)

ifnegative_prompt_embedsisnotNoneandnegative_pooled_prompt_embedsisNone:
raiseValueError(
"If`negative_prompt_embeds`areprovided,`negative_pooled_prompt_embeds`alsohavetobepassed.Makesuretogenerate`negative_pooled_prompt_embeds`fromthesametextencoderthatwasusedtogenerate`negative_prompt_embeds`."
)

defprepare_latents(self,batch_size,num_channels_latents,height,width,generator,latents=None):
iflatentsisnotNone:
returnlatents

shape=(batch_size,num_channels_latents,int(height)//self.vae_scale_factor,int(width)//self.vae_scale_factor)

ifisinstance(generator,list)andlen(generator)!=batch_size:
raiseValueError(
f"Youhavepassedalistofgeneratorsoflength{len(generator)},butrequestedaneffectivebatch"
f"sizeof{batch_size}.Makesurethebatchsizematchesthelengthofthegenerators."
)

latents=randn_tensor(shape,generator=generator,device=torch.device("cpu"),dtype=torch.float32)

returnlatents

@property
defguidance_scale(self):
returnself._guidance_scale

@property
defclip_skip(self):
returnself._clip_skip

#here`guidance_scale`isdefinedanalogtotheguidanceweight`w`ofequation(2)
#oftheImagenpaper:https://arxiv.org/pdf/2205.11487.pdf.`guidance_scale=1`
#correspondstodoingnoclassifierfreeguidance.
@property
defdo_classifier_free_guidance(self):
returnself._guidance_scale>1

@property
defjoint_attention_kwargs(self):
returnself._joint_attention_kwargs

@property
defnum_timesteps(self):
returnself._num_timesteps

@property
definterrupt(self):
returnself._interrupt

@torch.no_grad()
def__call__(
self,
prompt:Union[str,List[str]]=None,
prompt_2:Optional[Union[str,List[str]]]=None,
prompt_3:Optional[Union[str,List[str]]]=None,
height:Optional[int]=None,
width:Optional[int]=None,
num_inference_steps:int=28,
timesteps:List[int]=None,
guidance_scale:float=7.0,
negative_prompt:Optional[Union[str,List[str]]]=None,
negative_prompt_2:Optional[Union[str,List[str]]]=None,
negative_prompt_3:Optional[Union[str,List[str]]]=None,
num_images_per_prompt:Optional[int]=1,
generator:Optional[Union[torch.Generator,List[torch.Generator]]]=None,
latents:Optional[torch.FloatTensor]=None,
prompt_embeds:Optional[torch.FloatTensor]=None,
negative_prompt_embeds:Optional[torch.FloatTensor]=None,
pooled_prompt_embeds:Optional[torch.FloatTensor]=None,
negative_pooled_prompt_embeds:Optional[torch.FloatTensor]=None,
output_type:Optional[str]="pil",
return_dict:bool=True,
clip_skip:Optional[int]=None,
callback_on_step_end:Optional[Callable[[int,int,Dict],None]]=None,
callback_on_step_end_tensor_inputs:List[str]=["latents"],
):
height=heightorself.default_sample_size*self.vae_scale_factor
width=widthorself.default_sample_size*self.vae_scale_factor

#1.Checkinputs.Raiseerrorifnotcorrect
self.check_inputs(
prompt,
prompt_2,
prompt_3,
height,
width,
negative_prompt=negative_prompt,
negative_prompt_2=negative_prompt_2,
negative_prompt_3=negative_prompt_3,
prompt_embeds=prompt_embeds,
negative_prompt_embeds=negative_prompt_embeds,
pooled_prompt_embeds=pooled_prompt_embeds,
negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
)

self._guidance_scale=guidance_scale
self._clip_skip=clip_skip
self._interrupt=False

#2.Definecallparameters
ifpromptisnotNoneandisinstance(prompt,str):
batch_size=1
elifpromptisnotNoneandisinstance(prompt,list):
batch_size=len(prompt)
else:
batch_size=prompt_embeds.shape[0]
results=self.encode_prompt(
prompt=prompt,
prompt_2=prompt_2,
prompt_3=prompt_3,
negative_prompt=negative_prompt,
negative_prompt_2=negative_prompt_2,
negative_prompt_3=negative_prompt_3,
do_classifier_free_guidance=self.do_classifier_free_guidance,
prompt_embeds=prompt_embeds,
negative_prompt_embeds=negative_prompt_embeds,
pooled_prompt_embeds=pooled_prompt_embeds,
negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
clip_skip=self.clip_skip,
num_images_per_prompt=num_images_per_prompt,
)

(prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds)=results

ifself.do_classifier_free_guidance:
prompt_embeds=torch.cat([negative_prompt_embeds,prompt_embeds],dim=0)
pooled_prompt_embeds=torch.cat([negative_pooled_prompt_embeds,pooled_prompt_embeds],dim=0)

#4.Preparetimesteps
timesteps,num_inference_steps=retrieve_timesteps(self.scheduler,num_inference_steps,timesteps)
num_warmup_steps=max(len(timesteps)-num_inference_steps*self.scheduler.order,0)
self._num_timesteps=len(timesteps)

#5.Preparelatentvariables
num_channels_latents=16
latents=self.prepare_latents(batch_size*num_images_per_prompt,num_channels_latents,height,width,generator,latents)

#6.Denoisingloop
withself.progress_bar(total=num_inference_steps)asprogress_bar:
fori,tinenumerate(timesteps):
ifself.interrupt:
continue

#expandthelatentsifwearedoingclassifierfreeguidance
latent_model_input=torch.cat([latents]*2)ifself.do_classifier_free_guidanceelselatents
#broadcasttobatchdimensioninawaythat'scompatiblewithONNX/CoreML
timestep=t.expand(latent_model_input.shape[0])

noise_pred=self.transformer([latent_model_input,prompt_embeds,pooled_prompt_embeds,timestep])[0]

noise_pred=torch.from_numpy(noise_pred)

#performguidance
ifself.do_classifier_free_guidance:
noise_pred_uncond,noise_pred_text=noise_pred.chunk(2)
noise_pred=noise_pred_uncond+self.guidance_scale*(noise_pred_text-noise_pred_uncond)

#computethepreviousnoisysamplex_t->x_t-1
latents=self.scheduler.step(noise_pred,t,latents,return_dict=False)[0]

ifcallback_on_step_endisnotNone:
callback_kwargs={}
forkincallback_on_step_end_tensor_inputs:
callback_kwargs[k]=locals()[k]
callback_outputs=callback_on_step_end(self,i,t,callback_kwargs)

latents=callback_outputs.pop("latents",latents)
prompt_embeds=callback_outputs.pop("prompt_embeds",prompt_embeds)
negative_prompt_embeds=callback_outputs.pop("negative_prompt_embeds",negative_prompt_embeds)
negative_pooled_prompt_embeds=callback_outputs.pop("negative_pooled_prompt_embeds",negative_pooled_prompt_embeds)

#callthecallback,ifprovided
ifi==len(timesteps)-1or((i+1)>num_warmup_stepsand(i+1)%self.scheduler.order==0):
progress_bar.update()

ifoutput_type=="latent":
image=latents

else:
latents=(latents/self.vae_scaling_factor)+self.vae_shift_factor

image=torch.from_numpy(self.vae(latents)[0])
image=self.image_processor.postprocess(image,output_type=output_type)

ifnotreturn_dict:
return(image,)

returnStableDiffusion3PipelineOutput(images=image)

RunOpenVINOmodel
------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','GPU.0','GPU.1','AUTO'),value='CPU')



..code::ipython3

use_int4_transformer=widgets.Checkbox(value=TRANSFORMER_INT4_PATH.exists(),description="INT4transformer",disabled=notTRANSFORMER_INT4_PATH.exists())

use_int4_t5=widgets.Checkbox(value=TEXT_ENCODER_3_INT4_PATH.exists(),description="INT4t5textencoder",disabled=notTEXT_ENCODER_3_INT4_PATH.exists())

v_box_widgets=[]
ifTRANSFORMER_INT4_PATH.exists():
v_box_widgets.append(use_int4_transformer)

ifload_t5.valueandTEXT_ENCODER_3_INT4_PATH.exists():
v_box_widgets.append(use_int4_t5)

ifv_box_widgets:
model_options=widgets.VBox(v_box_widgets)
display(model_options)



..parsed-literal::

VBox(children=(Checkbox(value=True,description='INT4transformer'),))


..code::ipython3

ov_config={}
if"GPU"indevice.value:
ov_config["INFERENCE_PRECISION_HINT"]="f32"

transformer=core.compile_model(TRANSFORMER_PATHifnotuse_int4_transformer.valueelseTRANSFORMER_INT4_PATH,device.value)
text_encoder_3=(
core.compile_model(TEXT_ENCODER_3_PATHifnotuse_int4_t5.valueelseTEXT_ENCODER_3_INT4_PATH,device.value,ov_config)ifload_t5.valueelseNone
)
text_encoder=core.compile_model(TEXT_ENCODER_PATH,device.value,ov_config)
text_encoder_2=core.compile_model(TEXT_ENCODER_2_PATH,device.value,ov_config)
vae=core.compile_model(VAE_DECODER_PATH,device.value)

..code::ipython3

fromdiffusers.schedulersimportFlowMatchEulerDiscreteScheduler,FlashFlowMatchEulerDiscreteScheduler
fromtransformersimportAutoTokenizer

scheduler=(
FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_DIR/"scheduler")
ifnotuse_flash_lora.value
elseFlashFlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_DIR/"scheduler")
)

tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR/"tokenizer")
tokenizer_2=AutoTokenizer.from_pretrained(MODEL_DIR/"tokenizer_2")
tokenizer_3=AutoTokenizer.from_pretrained(MODEL_DIR/"tokenizer_3")ifload_t5.valueelseNone

..code::ipython3

ov_pipe=OVStableDiffusion3Pipeline(transformer,scheduler,vae,text_encoder,tokenizer,text_encoder_2,tokenizer_2,text_encoder_3,tokenizer_3)

..code::ipython3

image=ov_pipe(
"Araccoontrappedinsideaglassjarfullofcolorfulcandies,thebackgroundissteamywithvividcolors",
negative_prompt="",
num_inference_steps=28ifnotuse_flash_lora.valueelse4,
guidance_scale=5ifnotuse_flash_lora.valueelse0,
height=512,
width=512,
generator=torch.Generator().manual_seed(141),
).images[0]
image



..parsed-literal::

0%||0/4[00:00<?,?it/s]




..image::stable-diffusion-v3-with-output_files/stable-diffusion-v3-with-output_30_1.png



Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr
importnumpyasnp
importrandom

MAX_SEED=np.iinfo(np.int32).max
MAX_IMAGE_SIZE=1344


definfer(prompt,negative_prompt,seed,randomize_seed,width,height,guidance_scale,num_inference_steps,progress=gr.Progress(track_tqdm=True)):
ifrandomize_seed:
seed=random.randint(0,MAX_SEED)

generator=torch.Generator().manual_seed(seed)

image=ov_pipe(
prompt=prompt,
negative_prompt=negative_prompt,
guidance_scale=guidance_scale,
num_inference_steps=num_inference_steps,
width=width,
height=height,
generator=generator,
).images[0]

returnimage,seed


examples=[
"Astronautinajungle,coldcolorpalette,mutedcolors,detailed,8k",
"Anastronautridingagreenhorse",
"Adeliciouscevichecheesecakeslice",
"Apandareadingabookinalushforest.",
"A3drenderofafuturisticcitywithagiantrobotinthemiddlefullofneonlights,pinkandbluecolors",
'awizardkittenholdingasignsaying"openvino"withamagicwand.',
"photoofahugeredcatwithgreeneyessittingonacloudinthesky,lookingatthecamera",
"Pirateshipsailingonaseawiththemilkywaygalaxyintheskyandpurpleglowlights",
]

css="""
#col-container{
margin:0auto;
max-width:580px;
}
"""

withgr.Blocks(css=css)asdemo:
withgr.Column(elem_id="col-container"):
gr.Markdown(
"""
#Demo[StableDiffusion3Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)withOpenVINO
"""
)

withgr.Row():
prompt=gr.Text(
label="Prompt",
show_label=False,
max_lines=1,
placeholder="Enteryourprompt",
container=False,
)

run_button=gr.Button("Run",scale=0)

result=gr.Image(label="Result",show_label=False)

withgr.Accordion("AdvancedSettings",open=False):
negative_prompt=gr.Text(
label="Negativeprompt",
max_lines=1,
placeholder="Enteranegativeprompt",
)

seed=gr.Slider(
label="Seed",
minimum=0,
maximum=MAX_SEED,
step=1,
value=0,
)

randomize_seed=gr.Checkbox(label="Randomizeseed",value=True)

withgr.Row():
width=gr.Slider(
label="Width",
minimum=256,
maximum=MAX_IMAGE_SIZE,
step=64,
value=512,
)

height=gr.Slider(
label="Height",
minimum=256,
maximum=MAX_IMAGE_SIZE,
step=64,
value=512,
)

withgr.Row():
guidance_scale=gr.Slider(
label="Guidancescale",
minimum=0.0,
maximum=10.0ifnotuse_flash_lora.valueelse2,
step=0.1,
value=5.0ifnotuse_flash_lora.valueelse0,
)

num_inference_steps=gr.Slider(
label="Numberofinferencesteps",
minimum=1,
maximum=50,
step=1,
value=28ifnotuse_flash_lora.valueelse4,
)

gr.Examples(examples=examples,inputs=[prompt])
gr.on(
triggers=[run_button.click,prompt.submit,negative_prompt.submit],
fn=infer,
inputs=[prompt,negative_prompt,seed,randomize_seed,width,height,guidance_scale,num_inference_steps],
outputs=[result,seed],
)

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#ifyouhaveanyissuetolaunchonyourplatform,youcanpassshare=Truetolaunchmethod:
#demo.launch(share=True)
#itcreatesapubliclyshareablelinkfortheinterface.Readmoreinthedocs:https://gradio.app/docs/
try:
demo.launch(debug=False)
exceptException:
demo.launch(debug=False,share=True)
