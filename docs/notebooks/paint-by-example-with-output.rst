PaintByExample:Exemplar-basedImageEditingwithDiffusionModels
====================================================================

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`StableDiffusioninDiffusers
library<#stable-diffusion-in-diffusers-library>`__
-`Downloaddefaultimages<#download-default-images>`__
-`ConvertmodelstoOpenVINOIntermediaterepresentation(IR)
format<#convert-models-to-openvino-intermediate-representation-ir-format>`__
-`PrepareInferencepipeline<#prepare-inference-pipeline>`__
-`Selectinferencedevice<#select-inference-device>`__
-`ConfigureInferencePipeline<#configure-inference-pipeline>`__
-`Quantization<#quantization>`__

-`PrepareInferencepipeline<#prepare-inference-pipeline>`__
-`Runquantization<#run-quantization>`__
-`Runinferenceandcompareinference
time<#run-inference-and-compare-inference-time>`__
-`CompareUNetfilesize<#compare-unet-file-size>`__

-`Interactiveinference<#interactive-inference>`__

StableDiffusioninDiffuserslibrary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__ToworkwithStableDiffusion,
wewillusetheHuggingFace
`Diffusers<https://github.com/huggingface/diffusers>`__library.To
experimentwithin-paintingwecanuseDiffuserswhichexposesthe
`StableDiffusionInpaintPipeline<https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation>`__
similartothe`otherDiffusers
pipelines<https://huggingface.co/docs/diffusers/api/pipelines/overview>`__.
Thecodebelowdemonstrateshowtocreate
``StableDiffusionInpaintPipeline``using
``stable-diffusion-2-inpainting``.Tocreatethedrawingtoolwewill
installGradioforhandlinguserinteraction.

Thisistheoverallflowoftheapplication:

..figure::https://user-images.githubusercontent.com/103226580/236954918-f364b227-293c-4f78-a9bf-9dcebcb1034a.png
:alt:FlowDiagram

FlowDiagram

..code::ipython3

%pipinstall-q"torch>=2.1"torchvision--extra-index-url"https://download.pytorch.org/whl/cpu"
%pipinstall-q"diffusers>=0.25.0""peft==0.6.2""openvino>=2023.2.0""transformers>=4.25.1"ipywidgetsopencv-pythonpillow"nncf>=2.7.0""gradio==3.44.1"tqdm


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Downloadthemodelfrom`HuggingFace
Paint-by-Example<https://huggingface.co/Fantasy-Studio/Paint-by-Example>`__.
Thismighttakeseveralminutesbecauseitisover5GB

..code::ipython3

fromdiffusersimportDiffusionPipeline
fromdiffusers.schedulersimportDDIMScheduler,LMSDiscreteScheduler,PNDMScheduler


pipeline=DiffusionPipeline.from_pretrained("Fantasy-Studio/Paint-By-Example")

scheduler_inpaint=DDIMScheduler.from_config(pipeline.scheduler.config)

..code::ipython3

importgc

extractor=pipeline.feature_extractor
image_encoder=pipeline.image_encoder
image_encoder.eval()
unet_inpaint=pipeline.unet
unet_inpaint.eval()
vae_inpaint=pipeline.vae
vae_inpaint.eval()

delpipeline
gc.collect();

Downloaddefaultimages
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Downloaddefaultimages.

..code::ipython3

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file

download_file(
"https://github-production-user-asset-6210df.s3.amazonaws.com/103226580/286377210-edc98e97-0e43-4796-b771-dacd074c39ea.png",
"0.png",
"data/image",
)

download_file(
"https://github-production-user-asset-6210df.s3.amazonaws.com/103226580/286377233-b2c2d902-d379-415a-8183-5bdd37c52429.png",
"1.png",
"data/image",
)

download_file(
"https://github-production-user-asset-6210df.s3.amazonaws.com/103226580/286377248-da1db61e-3521-4cdb-85c8-1386d360ce22.png",
"2.png",
"data/image",
)

download_file(
"https://github-production-user-asset-6210df.s3.amazonaws.com/103226580/286377279-fa496f17-e850-4351-87c5-2552dfbc4633.jpg",
"bird.jpg",
"data/reference",
)

download_file(
"https://github-production-user-asset-6210df.s3.amazonaws.com/103226580/286377298-06a25ff2-84d8-4d46-95cd-8c25efa690d8.jpg",
"car.jpg",
"data/reference",
)

download_file(
"https://github-production-user-asset-6210df.s3.amazonaws.com/103226580/286377318-8841a801-1933-4523-a433-7d2fb64c47e6.jpg",
"dog.jpg",
"data/reference",
)

ConvertmodelstoOpenVINOIntermediaterepresentation(IR)format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Adaptedfrom`StableDiffusionv2InfiniteZoom
notebook<stable-diffusion-v2-with-output.html>`__

..code::ipython3

frompathlibimportPath
importtorch
importnumpyasnp
importopenvinoasov

model_dir=Path("model")
model_dir.mkdir(exist_ok=True)
sd2_inpainting_model_dir=Path("model/paint_by_example")
sd2_inpainting_model_dir.mkdir(exist_ok=True)

FunctionstoconverttoOpenVINOIRformat

..code::ipython3

defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


defconvert_image_encoder(image_encoder:torch.nn.Module,ir_path:Path):
"""
ConvertImageEncodermodeltoIR.
Functionacceptspipeline,preparesexampleinputsforconversion
Parameters:
image_encoder(torch.nn.Module):imageencoderPyTorchmodel
ir_path(Path):Fileforstoringmodel
Returns:
None
"""

classImageEncoderWrapper(torch.nn.Module):
def__init__(self,image_encoder):
super().__init__()
self.image_encoder=image_encoder

defforward(self,image):
image_embeddings,negative_prompt_embeds=self.image_encoder(image,return_uncond_vector=True)
returnimage_embeddings,negative_prompt_embeds

ifnotir_path.exists():
image_encoder=ImageEncoderWrapper(image_encoder)
image_encoder.eval()
input_ids=torch.randn((1,3,224,224))
#switchmodeltoinferencemode

#disablegradientscalculationforreducingmemoryconsumption
withtorch.no_grad():
ov_model=ov.convert_model(image_encoder,example_input=input_ids,input=([1,3,224,224],))
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
print("ImageEncodersuccessfullyconvertedtoIR")


defconvert_unet(
unet:torch.nn.Module,
ir_path:Path,
num_channels:int=4,
width:int=64,
height:int=64,
):
"""
ConvertUnetmodeltoIRformat.
Functionacceptspipeline,preparesexampleinputsforconversion
Parameters:
unet(torch.nn.Module):UNetPyTorchmodel
ir_path(Path):Fileforstoringmodel
num_channels(int,optional,4):numberofinputchannels
width(int,optional,64):inputwidth
height(int,optional,64):inputheight
Returns:
None
"""
dtype_mapping={torch.float32:ov.Type.f32,torch.float64:ov.Type.f64}
ifnotir_path.exists():
#prepareinputs
encoder_hidden_state=torch.ones((2,1,768))
latents_shape=(2,num_channels,width,height)
latents=torch.randn(latents_shape)
t=torch.from_numpy(np.array(1,dtype=np.float32))
unet.eval()
dummy_inputs=(latents,t,encoder_hidden_state)
input_info=[]
forinput_tensorindummy_inputs:
shape=ov.PartialShape(tuple(input_tensor.shape))
element_type=dtype_mapping[input_tensor.dtype]
input_info.append((shape,element_type))

withtorch.no_grad():
ov_model=ov.convert_model(unet,example_input=dummy_inputs,input=input_info)
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
print("U-NetsuccessfullyconvertedtoIR")


defconvert_vae_encoder(vae:torch.nn.Module,ir_path:Path,width:int=512,height:int=512):
"""
ConvertVAEmodeltoIRformat.
FunctionacceptsVAEmodel,createswrapperclassforexportonlynecessaryforinferencepart,
preparesexampleinputsforconversion,
Parameters:
vae(torch.nn.Module):VAEPyTorchmodel
ir_path(Path):Fileforstoringmodel
width(int,optional,512):inputwidth
height(int,optional,512):inputheight
Returns:
None
"""

classVAEEncoderWrapper(torch.nn.Module):
def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,image):
latents=self.vae.encode(image).latent_dist.sample()
returnlatents

ifnotir_path.exists():
vae_encoder=VAEEncoderWrapper(vae)
vae_encoder.eval()
image=torch.zeros((1,3,width,height))
withtorch.no_grad():
ov_model=ov.convert_model(vae_encoder,example_input=image,input=([1,3,width,height],))
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
print("VAEencodersuccessfullyconvertedtoIR")


defconvert_vae_decoder(vae:torch.nn.Module,ir_path:Path,width:int=64,height:int=64):
"""
ConvertVAEdecodermodeltoIRformat.
FunctionacceptsVAEmodel,createswrapperclassforexportonlynecessaryforinferencepart,
preparesexampleinputsforconversion,
Parameters:
vae(torch.nn.Module):VAEmodel
ir_path(Path):Fileforstoringmodel
width(int,optional,64):inputwidth
height(int,optional,64):inputheight
Returns:
None
"""

classVAEDecoderWrapper(torch.nn.Module):
def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,latents):
latents=1/0.18215*latents
returnself.vae.decode(latents)

ifnotir_path.exists():
vae_decoder=VAEDecoderWrapper(vae)
latents=torch.zeros((1,4,width,height))

vae_decoder.eval()
withtorch.no_grad():
ov_model=ov.convert_model(vae_decoder,example_input=latents,input=([1,4,width,height],))
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
print("VAEdecodersuccessfullyconvertedto")

Dotheconversionofthein-paintingmodel:

..code::ipython3

IMAGE_ENCODER_OV_PATH_INPAINT=sd2_inpainting_model_dir/"image_encoder.xml"

ifnotIMAGE_ENCODER_OV_PATH_INPAINT.exists():
convert_image_encoder(image_encoder,IMAGE_ENCODER_OV_PATH_INPAINT)
else:
print(f"Imageencoderwillbeloadedfrom{IMAGE_ENCODER_OV_PATH_INPAINT}")

delimage_encoder
gc.collect();

DotheconversionoftheUnetmodel

..code::ipython3

UNET_OV_PATH_INPAINT=sd2_inpainting_model_dir/"unet.xml"
ifnotUNET_OV_PATH_INPAINT.exists():
convert_unet(unet_inpaint,UNET_OV_PATH_INPAINT,num_channels=9,width=64,height=64)
delunet_inpaint
gc.collect()
else:
delunet_inpaint
print(f"U-Netwillbeloadedfrom{UNET_OV_PATH_INPAINT}")
gc.collect();

DotheconversionoftheVAEEncodermodel

..code::ipython3

VAE_ENCODER_OV_PATH_INPAINT=sd2_inpainting_model_dir/"vae_encoder.xml"

ifnotVAE_ENCODER_OV_PATH_INPAINT.exists():
convert_vae_encoder(vae_inpaint,VAE_ENCODER_OV_PATH_INPAINT,512,512)
else:
print(f"VAEencoderwillbeloadedfrom{VAE_ENCODER_OV_PATH_INPAINT}")

VAE_DECODER_OV_PATH_INPAINT=sd2_inpainting_model_dir/"vae_decoder.xml"
ifnotVAE_DECODER_OV_PATH_INPAINT.exists():
convert_vae_decoder(vae_inpaint,VAE_DECODER_OV_PATH_INPAINT,64,64)
else:
print(f"VAEdecoderwillbeloadedfrom{VAE_DECODER_OV_PATH_INPAINT}")

delvae_inpaint
gc.collect();

PrepareInferencepipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Functiontopreparethemaskandmaskedimage.

Adaptedfrom`StableDiffusionv2InfiniteZoom
notebook<stable-diffusion-v2-with-output.html>`__

Themaindifferenceisthatinsteadofencodingatextpromptitwill
nowencodeanimageastheprompt.

Thisisthedetailedflowchartforthepipeline:

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/103226580/cde2d5c4-2540-4a45-ad9c-339f7a69459d
:alt:pipeline-flowchart

pipeline-flowchart

..code::ipython3

importinspect
fromtypingimportOptional,Union,Dict

importPIL
importcv2

fromtransformersimportCLIPImageProcessor
fromdiffusers.pipelines.pipeline_utilsimportDiffusionPipeline
fromopenvino.runtimeimportModel


defprepare_mask_and_masked_image(image:PIL.Image.Image,mask:PIL.Image.Image):
"""
Preparesapair(image,mask)tobeconsumedbytheStableDiffusionpipeline.Thismeansthatthoseinputswillbe
convertedto``np.array``withshapes``batchxchannelsxheightxwidth``where``channels``is``3``forthe
``image``and``1``forthe``mask``.

The``image``willbeconvertedto``np.float32``andnormalizedtobein``[-1,1]``.The``mask``willbe
binarized(``mask>0.5``)andcastto``np.float32``too.

Args:
image(Union[np.array,PIL.Image]):Theimagetoinpaint.
Itcanbea``PIL.Image``,ora``heightxwidthx3````np.array``
mask(_type_):Themasktoapplytotheimage,i.e.regionstoinpaint.
Itcanbea``PIL.Image``,ora``heightxwidth````np.array``.

Returns:
tuple[np.array]:Thepair(mask,masked_image)as``torch.Tensor``with4
dimensions:``batchxchannelsxheightxwidth``.
"""
ifisinstance(image,(PIL.Image.Image,np.ndarray)):
image=[image]

ifisinstance(image,list)andisinstance(image[0],PIL.Image.Image):
image=[np.array(i.convert("RGB"))[None,:]foriinimage]
image=np.concatenate(image,axis=0)
elifisinstance(image,list)andisinstance(image[0],np.ndarray):
image=np.concatenate([i[None,:]foriinimage],axis=0)

image=image.transpose(0,3,1,2)
image=image.astype(np.float32)/127.5-1.0

#preprocessmask
ifisinstance(mask,(PIL.Image.Image,np.ndarray)):
mask=[mask]

ifisinstance(mask,list)andisinstance(mask[0],PIL.Image.Image):
mask=np.concatenate([np.array(m.convert("L"))[None,None,:]forminmask],axis=0)
mask=mask.astype(np.float32)/255.0
elifisinstance(mask,list)andisinstance(mask[0],np.ndarray):
mask=np.concatenate([m[None,None,:]forminmask],axis=0)

mask=1-mask

mask[mask<0.5]=0
mask[mask>=0.5]=1

masked_image=image*mask

returnmask,masked_image

Classforthepipelinewhichwillconnectallthemodelstogether:VAE
decode–>imageencode–>tokenizer–>Unet–>VAEmodel–>scheduler

..code::ipython3

classOVStableDiffusionInpaintingPipeline(DiffusionPipeline):
def__init__(
self,
vae_decoder:Model,
image_encoder:Model,
image_processor:CLIPImageProcessor,
unet:Model,
scheduler:Union[DDIMScheduler,PNDMScheduler,LMSDiscreteScheduler],
vae_encoder:Model=None,
):
"""
Pipelinefortext-to-imagegenerationusingStableDiffusion.
Parameters:
vae_decoder(Model):
VariationalAuto-Encoder(VAE)Modeltodecodeimagestoandfromlatentrepresentations.
image_encoder(Model):
https://huggingface.co/Fantasy-Studio/Paint-by-Example/blob/main/image_encoder/config.json
tokenizer(CLIPTokenizer):
TokenizerofclassCLIPTokenizer(https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
unet(Model):ConditionalU-Netarchitecturetodenoisetheencodedimagelatents.
vae_encoder(Model):
VariationalAuto-Encoder(VAE)Modeltoencodeimagestolatentrepresentation.
scheduler(SchedulerMixin):
Aschedulertobeusedincombinationwithunettodenoisetheencodedimagelatents.Canbeoneof
DDIMScheduler,LMSDiscreteScheduler,orPNDMScheduler.
"""
super().__init__()
self.scheduler=scheduler
self.vae_decoder=vae_decoder
self.vae_encoder=vae_encoder
self.image_encoder=image_encoder
self.unet=unet
self.register_to_config(unet=unet)
self._unet_output=unet.output(0)
self._vae_d_output=vae_decoder.output(0)
self._vae_e_output=vae_encoder.output(0)ifvae_encoderisnotNoneelseNone
self.height=self.unet.input(0).shape[2]*8
self.width=self.unet.input(0).shape[3]*8
self.image_processor=image_processor

defprepare_mask_latents(
self,
mask,
masked_image,
height=512,
width=512,
do_classifier_free_guidance=True,
):
"""
PreparemaskasUnetnputandencodeinputmaskedimagetolatentspaceusingvaeencoder

Parameters:
mask(np.array):inputmaskarray
masked_image(np.array):maskedinputimagetensor
heigh(int,*optional*,512):generatedimageheight
width(int,*optional*,512):generatedimagewidth
do_classifier_free_guidance(bool,*optional*,True):whethertouseclassifierfreeguidanceornot
Returns:
mask(np.array):resizedmasktensor
masked_image_latents(np.array):maskedimageencodedintolatentspaceusingVAE
"""
mask=torch.nn.functional.interpolate(torch.from_numpy(mask),size=(height//8,width//8))
mask=mask.numpy()

#encodethemaskimageintolatentsspacesowecanconcatenateittothelatents
masked_image_latents=self.vae_encoder(masked_image)[self._vae_e_output]
masked_image_latents=0.18215*masked_image_latents

mask=np.concatenate([mask]*2)ifdo_classifier_free_guidanceelsemask
masked_image_latents=np.concatenate([masked_image_latents]*2)ifdo_classifier_free_guidanceelsemasked_image_latents
returnmask,masked_image_latents

def__call__(
self,
image:PIL.Image.Image,
mask_image:PIL.Image.Image,
reference_image:PIL.Image.Image,
num_inference_steps:Optional[int]=50,
guidance_scale:Optional[float]=7.5,
eta:Optional[float]=0,
output_type:Optional[str]="pil",
seed:Optional[int]=None,
):
"""
Functioninvokedwhencallingthepipelineforgeneration.
Parameters:
image(PIL.Image.Image):
Sourceimageforinpainting.
mask_image(PIL.Image.Image):
Maskareaforinpainting
reference_image(PIL.Image.Image):
Referenceimagetoinpaintinmaskarea
num_inference_steps(int,*optional*,defaultsto50):
Thenumberofdenoisingsteps.Moredenoisingstepsusuallyleadtoahigherqualityimageatthe
expenseofslowerinference.
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
Returns:
Dictionarywithkeys:
sample-thelastgeneratedimagePIL.Image.Imageornp.array
"""
ifseedisnotNone:
np.random.seed(seed)
#here`guidance_scale`isdefinedanalogtotheguidanceweight`w`ofequation(2)
#oftheImagenpaper:https://arxiv.org/pdf/2205.11487.pdf.`guidance_scale=1`
#correspondstodoingnoclassifierfreeguidance.
do_classifier_free_guidance=guidance_scale>1.0

#getreferenceimageembeddings
image_embeddings=self._encode_image(reference_image,do_classifier_free_guidance=do_classifier_free_guidance)

#preparemask
mask,masked_image=prepare_mask_and_masked_image(image,mask_image)
#settimesteps
accepts_offset="offset"inset(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
extra_set_kwargs={}
ifaccepts_offset:
extra_set_kwargs["offset"]=1

self.scheduler.set_timesteps(num_inference_steps,**extra_set_kwargs)
timesteps,num_inference_steps=self.get_timesteps(num_inference_steps,1)
latent_timestep=timesteps[:1]

#gettheinitialrandomnoiseunlesstheusersuppliedit
latents,meta=self.prepare_latents(latent_timestep)
mask,masked_image_latents=self.prepare_mask_latents(
mask,
masked_image,
do_classifier_free_guidance=do_classifier_free_guidance,
)

#prepareextrakwargsfortheschedulerstep,sincenotallschedulershavethesamesignature
#eta(η)isonlyusedwiththeDDIMScheduler,itwillbeignoredforotherschedulers.
#etacorrespondstoηinDDIMpaper:https://arxiv.org/abs/2010.02502
#andshouldbebetween[0,1]
accepts_eta="eta"inset(inspect.signature(self.scheduler.step).parameters.keys())
extra_step_kwargs={}
ifaccepts_eta:
extra_step_kwargs["eta"]=eta

fortinself.progress_bar(timesteps):
#expandthelatentsifwearedoingclassifierfreeguidance
latent_model_input=np.concatenate([latents]*2)ifdo_classifier_free_guidanceelselatents
latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)
latent_model_input=np.concatenate([latent_model_input,masked_image_latents,mask],axis=1)
#predictthenoiseresidual
noise_pred=self.unet([latent_model_input,np.array(t,dtype=np.float32),image_embeddings])[self._unet_output]
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
image=self.vae_decoder(latents)[self._vae_d_output]

image=self.postprocess_image(image,meta,output_type)
return{"sample":image}

def_encode_image(self,image:PIL.Image.Image,do_classifier_free_guidance:bool=True):
"""
Encodestheimageintoimageencoderhiddenstates.

Parameters:
image(PIL.Image.Image):baseimagetoencode
do_classifier_free_guidance(bool):whethertouseclassifierfreeguidanceornot
Returns:
image_embeddings(np.ndarray):imageencoderhiddenstates
"""
processed_image=self.image_processor(image)
processed_image=processed_image["pixel_values"][0]
processed_image=np.expand_dims(processed_image,axis=0)

output=self.image_encoder(processed_image)
image_embeddings=output[self.image_encoder.output(0)]
negative_embeddings=output[self.image_encoder.output(1)]

image_embeddings=np.concatenate([negative_embeddings,image_embeddings])

returnimage_embeddings

defprepare_latents(self,latent_timestep:torch.Tensor=None):
"""
Functionforgettinginitiallatentsforstartinggeneration

Parameters:
latent_timestep(torch.Tensor,*optional*,None):
Predictedbyschedulerinitialstepforimagegeneration,requiredforlatentimagemixingwithnosie
Returns:
latents(np.ndarray):
Imageencodedinlatentspace
"""
latents_shape=(1,4,self.height//8,self.width//8)
noise=np.random.randn(*latents_shape).astype(np.float32)
#ifweuseLMSDiscreteScheduler,let'smakesurelatentsaremulitpliedbysigmas
ifisinstance(self.scheduler,LMSDiscreteScheduler):
noise=noise*self.scheduler.sigmas[0].numpy()
returnnoise,{}

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
Valuesthatapproach1.0allowforlotsofvariationsbutwillalsoproduceimagesthatarenotsemanticallyconsistentwiththeinput.
"""
#gettheoriginaltimestepusinginit_timestep
init_timestep=min(int(num_inference_steps*strength),num_inference_steps)

t_start=max(num_inference_steps-init_timestep,0)
timesteps=self.scheduler.timesteps[t_start:]

returntimesteps,num_inference_steps-t_start

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

fromopenvinoimportCore
importipywidgetsaswidgets

core=Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=4,options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='AUTO')



ConfigureInferencePipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Configurationsteps:1.Loadmodelsondevice2.Configuretokenizerand
scheduler3.CreateinstanceofOvStableDiffusionInpaintingPipeline
class

Thiscantakeawhiletorun.

..code::ipython3

ov_config={"INFERENCE_PRECISION_HINT":"f32"}ifdevice.value!="CPU"else{}


defget_ov_pipeline():
image_encoder_inpaint=core.compile_model(IMAGE_ENCODER_OV_PATH_INPAINT,device.value)
unet_model_inpaint=core.compile_model(UNET_OV_PATH_INPAINT,device.value)
vae_decoder_inpaint=core.compile_model(VAE_DECODER_OV_PATH_INPAINT,device.value,ov_config)
vae_encoder_inpaint=core.compile_model(VAE_ENCODER_OV_PATH_INPAINT,device.value,ov_config)

ov_pipe_inpaint=OVStableDiffusionInpaintingPipeline(
image_processor=extractor,
image_encoder=image_encoder_inpaint,
unet=unet_model_inpaint,
vae_encoder=vae_encoder_inpaint,
vae_decoder=vae_decoder_inpaint,
scheduler=scheduler_inpaint,
)

returnov_pipe_inpaint


ov_pipe_inpaint=get_ov_pipeline()

Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``StableDiffusionInpaintingPipeline``structure,UNetused
foriterativedenoisingofinput.Itmeansthatmodelrunsinthecycle
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

importipywidgetsaswidgets

UNET_INT8_OV_PATH=Path("model/unet_int8.xml")
int8_ov_pipe_inpaint=None


to_quantize=widgets.Checkbox(
value=True,
description="Quantization",
disabled=False,
)

to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



Let’sload``skipmagic``extensiontoskipquantizationif
``to_quantize``isnotselected

..code::ipython3

#Fetch`skip_kernel_extension`module
r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

ifto_quantize.valueand"GPU"indevice.value:
to_quantize.value=False

%load_extskip_kernel_extension

Preparecalibrationdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuse3examplesfrom
`Paint-by-Example<https://github.com/Fantasy-Studio/Paint-by-Example>`__
tocreateacalibrationdataset.

..code::ipython3

importPIL
importrequests
fromioimportBytesIO


defdownload_image(url):
response=requests.get(url)
returnPIL.Image.open(BytesIO(response.content)).convert("RGB")


example1=[
"https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/examples/image/example_1.png?raw=true",
"https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/examples/mask/example_1.png?raw=true",
"https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/examples/reference/example_1.jpg?raw=true",
]
example2=[
"https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/examples/image/example_2.png?raw=true",
"https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/examples/mask/example_2.png?raw=true",
"https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/examples/reference/example_2.jpg?raw=true",
]
example3=[
"https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/examples/image/example_3.png?raw=true",
"https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/examples/mask/example_3.png?raw=true",
"https://github.com/Fantasy-Studio/Paint-by-Example/blob/main/examples/reference/example_3.jpg?raw=true",
]
examples=[example1,example2,example3]


img_examples=[]
forinit_image_url,mask_image_url,example_image_urlinexamples:
init_image=download_image(init_image_url).resize((512,512))
mask_image=download_image(mask_image_url).resize((512,512))
example_image=download_image(example_image_url).resize((512,512))
img_examples.append((init_image,mask_image,example_image))

Tocollectintermediatemodelinputsforcalibrationweshouldcustomize
``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

fromtqdm.notebookimporttqdm
fromtransformersimportset_seed
fromtypingimportAny,Dict,List


classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model,data_cache:List[Any]=None):
super().__init__(compiled_model)
self.data_cache=data_cacheifdata_cacheelse[]

def__call__(self,*args,**kwargs):
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)


defcollect_calibration_data(pipeline)->List[Dict]:
original_unet=pipeline.unet
pipeline.unet=CompiledModelDecorator(original_unet)
pipeline.set_progress_bar_config(disable=True)
prev_example_image=None
forinit_image,mask_image,example_imageinimg_examples:

_=pipeline(
image=init_image,
mask_image=mask_image,
reference_image=example_image,
)
ifprev_example_image:
_=pipeline(
image=init_image,
mask_image=mask_image,
reference_image=prev_example_image,
)
prev_example_image=example_image


calibration_dataset=pipeline.unet.data_cache
pipeline.set_progress_bar_config(disable=False)
pipeline.unet=original_unet

returncalibration_dataset

..code::ipython3

%%skipnot$to_quantize.value

UNET_INT8_OV_PATH=Path("model/unet_int8.xml")
ifnotUNET_INT8_OV_PATH.exists():
unet_calibration_data=collect_calibration_data(ov_pipe_inpaint)

Runquantization
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Createaquantizedmodelfromthepre-trainedconvertedOpenVINOmodel.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

%%skipnot$to_quantize.value

importnncf


defget_quantized_pipeline():
ifUNET_INT8_OV_PATH.exists():
print("Loadingquantizedmodel")
quantized_unet=core.read_model(UNET_INT8_OV_PATH)
else:
unet=core.read_model(UNET_OV_PATH_INPAINT)
quantized_unet=nncf.quantize(
model=unet,
preset=nncf.QuantizationPreset.MIXED,
calibration_dataset=nncf.Dataset(unet_calibration_data),
model_type=nncf.ModelType.TRANSFORMER,
)
ov.save_model(quantized_unet,UNET_INT8_OV_PATH)

unet_optimized=core.compile_model(UNET_INT8_OV_PATH,device.value)

image_encoder_inpaint=core.compile_model(IMAGE_ENCODER_OV_PATH_INPAINT,device.value)
vae_decoder_inpaint=core.compile_model(VAE_DECODER_OV_PATH_INPAINT,device.value,ov_config)
vae_encoder_inpaint=core.compile_model(VAE_ENCODER_OV_PATH_INPAINT,device.value,ov_config)

int8_ov_pipe_inpaint=OVStableDiffusionInpaintingPipeline(
image_processor=extractor,
image_encoder=image_encoder_inpaint,
unet=unet_optimized,
vae_encoder=vae_encoder_inpaint,
vae_decoder=vae_decoder_inpaint,
scheduler=scheduler_inpaint,
)

returnint8_ov_pipe_inpaint


int8_ov_pipe_inpaint=get_quantized_pipeline()


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,openvino



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

INFO:nncf:121ignorednodeswerefoundbynameintheNNCFGraph



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



Runinferenceandcompareinferencetime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OVpipeline:

..code::ipython3

init_image,mask_image,example_image=img_examples[1]


ov_image=ov_pipe_inpaint(image=init_image,mask_image=mask_image,reference_image=example_image,seed=2)

Quantizedpipeline:

..code::ipython3

%%skipnot$to_quantize.value

int8_image=int8_ov_pipe_inpaint(image=init_image,mask_image=mask_image,reference_image=example_image,seed=2)

..code::ipython3

%%skipnot$to_quantize.value

importmatplotlib.pyplotasplt
fromPILimportImage

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


visualize_results(ov_image["sample"][0],int8_image["sample"][0])



..image::paint-by-example-with-output_files/paint-by-example-with-output_41_0.png


..code::ipython3

%%skip$to_quantize.value

display(ov_image["sample"][0])

CompareUNetfilesize
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fp16_ir_model_size=UNET_OV_PATH_INPAINT.with_suffix(".bin").stat().st_size/1024
quantized_model_size=UNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size/1024

print(f"FP16modelsize:{fp16_ir_model_size:.2f}KB")
print(f"INT8modelsize:{quantized_model_size:.2f}KB")
print(f"Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16modelsize:1678780.62KB
INT8modelsize:840725.98KB
Modelcompressionrate:1.997


Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

Choosewhatmodeldoyouwanttouseintheinteractiveinterface.You
canchooseboth,FP16andINT8.

..code::ipython3

available_models=["FP16"]

ifUNET_INT8_OV_PATH.exists():
available_models.append("INT8")

model_to_use=widgets.Select(
options=available_models,
value="FP16",
description="Selectmodel:",
disabled=False,
)

model_to_use




..parsed-literal::

Select(description='Selectmodel:',options=('FP16','INT8'),value='FP16')



..code::ipython3

if"INT8"==model_to_use.value:
chosen_pipeline=int8_ov_pipe_inpaintorget_quantized_pipeline()
ov_pipe_inpaint=None
else:
chosen_pipeline=ov_pipe_inpaintorget_ov_pipeline()
int8_ov_pipe_inpaint=None


gc.collect();

Chooseasourceimageandareferenceimage,drawamaskinsourceimage
andpush“Paint!”

..code::ipython3

#Codeadapatedfromhttps://huggingface.co/spaces/Fantasy-Studio/Paint-by-Example/blob/main/app.py

importos
importgradioasgr


defpredict(input_dict,reference,seed,steps):
"""
Thisfunctionrunswhenthe'paint'buttonispressed.Ittakes3inputimages.TakesgeneratedimagedecodedbyVAEdecoder,unpadittoinitilaimagesize(ifrequired),
normalizeandconvertto[0,255]pixelsrange.Optionally,convertesitfromnp.ndarraytoPIL.Imageformat

Parameters:
input_dict(Dict):
Containstwoimagesinadictionary
'image'istheimagethatwillbepaintedon
'mask'istheblack/whiteimagespecifyingwheretopaint(white)andnottopaint(black)
image(PIL.Image.Image):
Referenceimagethatwillbeusedbythemodeltoknowwhattopaintinthespecifiedarea
seed(int):
Usedtoinitializetherandomnumbergeneratorstate
steps(int):
Thenumberofdenoisingstepstorunduringinference.Low=fast/lowquality,High=slow/higherquality
use_quantize_model(bool):
Usefp16orint8model
Returns:
image(PIL.Image.Image):
Postprocessedimages
"""
width,height=input_dict["image"].size

#Iftheimageisnot512x512thenresize
ifwidth<height:
factor=width/512.0
width=512
height=int((height/factor)/8.0)*8
else:
factor=height/512.0
height=512
width=int((width/factor)/8.0)*8

init_image=input_dict["image"].convert("RGB").resize((width,height))
mask=input_dict["mask"].convert("RGB").resize((width,height))

#Iftheimageisnota512x512squarethencrop
ifwidth>height:
buffer=(width-height)/2
input_image=init_image.crop((buffer,0,width-buffer,512))
mask=mask.crop((buffer,0,width-buffer,512))
elifwidth<height:
buffer=(height-width)/2
input_image=init_image.crop((0,buffer,512,height-buffer))
mask=mask.crop((0,buffer,512,height-buffer))
else:
input_image=init_image

ifnotos.path.exists("output"):
os.mkdir("output")
input_image.save("output/init.png")
mask.save("output/mask.png")
reference.save("output/ref.png")

mask=[mask]

result=chosen_pipeline(
image=input_image,
mask_image=mask,
reference_image=reference,
seed=seed,
num_inference_steps=steps,
)[
"sample"
][0]

out_dir=Path("output")
out_dir.mkdir(exist_ok=True)
result.save("output/result.png")

returnresult


example={}
title=f"#{model_to_use.value}pipeline"
ref_dir="data/reference"
image_dir="data/image"
ref_list=[os.path.join(ref_dir,file)forfileinos.listdir(ref_dir)iffile.endswith(".jpg")]
ref_list.sort()
image_list=[os.path.join(image_dir,file)forfileinos.listdir(image_dir)iffile.endswith(".png")]
image_list.sort()


image_blocks=gr.Blocks()
withimage_blocksasdemo:
gr.Markdown(title)
withgr.Group():
withgr.Row():
withgr.Column():
image=gr.Image(
source="upload",
tool="sketch",
elem_id="image_upload",
type="pil",
label="SourceImage",
)
reference=gr.Image(
source="upload",
elem_id="image_upload",
type="pil",
label="ReferenceImage",
)

withgr.Column():
image_out=gr.Image(label="Output",elem_id="output-img")
steps=gr.Slider(
label="Steps",
value=15,
minimum=2,
maximum=75,
step=1,
interactive=True,
)
seed=gr.Slider(0,10000,label="Seed(0=random)",value=0,step=1)

withgr.Row(elem_id="prompt-container"):
btn=gr.Button("Paint!")

withgr.Row():
withgr.Column():
gr.Examples(
image_list,
inputs=[image],
label="Examples-SourceImage",
examples_per_page=12,
)
withgr.Column():
gr.Examples(
ref_list,
inputs=[reference],
label="Examples-ReferenceImage",
examples_per_page=12,
)

btn.click(
fn=predict,
inputs=[image,reference,seed,steps],
outputs=[image_out],
)

#LaunchingtheGradioapp
try:
image_blocks.launch(debug=False,height=680)
exceptException:
image_blocks.queue().launch(share=True,debug=False,height=680)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#image_blocks.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
