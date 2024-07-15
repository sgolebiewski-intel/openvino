ImageEditingwithInstructPix2PixandOpenVINO
===============================================

TheInstructPix2Pixisaconditionaldiffusionmodelthateditsimages
basedonwritteninstructionsprovidedbytheuser.Generativeimage
editingmodelstraditionallytargetasingleeditingtasklikestyle
transferortranslationbetweenimagedomains.Textguidancegivesusan
opportunitytosolvemultipletaskswithasinglemodel.The
InstructPix2Pixmethodworksdifferentthanexistingtext-basedimage
editinginthatitenableseditingfrominstructionsthattellthemodel
whatactiontoperforminsteadofusingtextlabels,captionsor
descriptionsofinput/outputimages.Akeybenefitoffollowingediting
instructionsisthattheusercanjusttellthemodelexactlywhattodo
innaturalwrittentext.Thereisnoneedfortheusertoprovideextra
information,suchasexampleimagesordescriptionsofvisualcontent
thatremainconstantbetweentheinputandoutputimages.Moredetails
aboutthisapproachcanbefoundinthis
`paper<https://arxiv.org/pdf/2211.09800.pdf>`__and
`repository<https://github.com/timothybrooks/instruct-pix2pix>`__.

ThisnotebookdemonstrateshowtoconvertandruntheInstructPix2Pix
modelusingOpenVINO.

Notebookcontainsthefollowingsteps:

1.ConvertPyTorchmodelstoOpenVINOIRformat,usingModelConversion
API.
2.RunInstructPix2PixpipelinewithOpenVINO.
3.OptimizeInstructPix2Pixpipelinewith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__quantization.
4.Compareresultsoforiginalandoptimizedpipelines.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`CreatePytorchModelspipeline<#create-pytorch-models-pipeline>`__
-`ConvertModelstoOpenVINOIR<#convert-models-to-openvino-ir>`__

-`TextEncoder<#text-encoder>`__
-`VAE<#vae>`__
-`Unet<#unet>`__

-`PrepareInferencePipeline<#prepare-inference-pipeline>`__
-`Quantization<#quantization>`__

-`Preparecalibrationdataset<#prepare-calibration-dataset>`__
-`Runquantization<#run-quantization>`__
-`CompareinferencetimeoftheFP16andINT8
models<#compare-inference-time-of-the-fp16-and-int8-models>`__

-`InteractivedemowithGradio<#interactive-demo-with-gradio>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installnecessarypackages

..code::ipython3

importplatform

%pipinstall-q"transformers>=4.25.1"torchaccelerate"gradio>4.19""datasets>=2.14.6"diffuserspillowopencv-python--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2023.1.0"

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

CreatePytorchModelspipeline
------------------------------

`backtotop⬆️<#table-of-contents>`__

``StableDiffusionInstructPix2PixPipeline``isanend-to-endinference
pipelinethatyoucanusetoeditimagesfromtextinstructionswith
justafewlinesofcodeprovidedaspart
`diffusers<https://huggingface.co/docs/diffusers/index>`__library.

First,weloadthepre-trainedweightsofallcomponentsofthemodel.

**NOTE**:Initially,modelloadingcantakesometimedueto
downloadingtheweights.Also,thedownloadspeeddependsonyour
internetconnection.

..code::ipython3

importtorch
fromdiffusersimport(
StableDiffusionInstructPix2PixPipeline,
EulerAncestralDiscreteScheduler,
)

model_id="timbrooks/instruct-pix2pix"
pipe=StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,torch_dtype=torch.float32,safety_checker=None)
scheduler_config=pipe.scheduler.config
text_encoder=pipe.text_encoder
text_encoder.eval()
unet=pipe.unet
unet.eval()
vae=pipe.vae
vae.eval()

delpipe

ConvertModelstoOpenVINOIR
-----------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchmodelsusing`ModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
toconvertthemodeltoIRformat.``ov.convert_model``functionaccepts
PyTorchmodelobjectandexampleinputandthenconvertsitto
``ov.Model``classinstancethatreadytouseforloadingondeviceor
canbesavedondiskusing``ov.save_model``.

TheInstructPix2PixmodelisbasedonStableDiffusion,alarge-scale
text-to-imagelatentdiffusionmodel.Youcanfindmoredetailsabout
howtorunStableDiffusionfortext-to-imagegenerationwithOpenVINO
inaseparate
`tutorial<stable-diffusion-text-to-image-with-output.html>`__.

Themodelconsistsofthreeimportantparts:

-TextEncoder-tocreateconditionsfromatextprompt.
-Unet-forstep-by-stepdenoisinglatentimagerepresentation.
-Autoencoder(VAE)-toencodetheinitialimagetolatentspacefor
startingthedenoisingprocessanddecodinglatentspacetoimage,
whendenoisingiscomplete.

Letusconverteachpart.

TextEncoder
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thetext-encoderisresponsiblefortransformingtheinputprompt,for
example,“aphotoofanastronautridingahorse”intoanembedding
spacethatcanbeunderstoodbytheUNet.Itisusuallyasimple
transformer-basedencoderthatmapsasequenceofinputtokenstoa
sequenceoflatenttextembeddings.

Inputofthetextencoderistensor``input_ids``,whichcontains
indexesoftokensfromtextprocessedbytokenizerandpaddedtomaximum
lengthacceptedbythemodel.Modeloutputsaretwotensors:
``last_hidden_state``-hiddenstatefromthelastMultiHeadAttention
layerinthemodeland``pooler_out``-pooledoutputforwholemodel
hiddenstates.

..code::ipython3

frompathlibimportPath
importopenvinoasov
importgc

core=ov.Core()

TEXT_ENCODER_OV_PATH=Path("text_encoder.xml")


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
(1,77),
],
)
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
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

32



VAE
~~~

`backtotop⬆️<#table-of-contents>`__

TheVAEmodelconsistsoftwoparts:anencoderandadecoder.

-Theencoderisusedtoconverttheimageintoalowdimensional
latentrepresentation,whichwillserveastheinputtotheUNet
model.
-Thedecoder,conversely,transformsthelatentrepresentationback
intoanimage.

Incomparisonwithatext-to-imageinferencepipeline,whereVAEisused
onlyfordecoding,thepipelinealsoinvolvestheoriginalimage
encoding.Asthetwopartsareusedseparatelyinthepipelineon
differentsteps,anddonotdependoneachother,weshouldconvertthem
intotwoindependentmodels.

..code::ipython3

VAE_ENCODER_OV_PATH=Path("vae_encoder.xml")


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
cleanup_torchscript_cache()
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
cleanup_torchscript_cache()
print(f"VAEdecodersuccessfullyconvertedtoIRandsavedto{ir_path}")


ifnotVAE_DECODER_OV_PATH.exists():
convert_vae_decoder(vae,VAE_DECODER_OV_PATH)
else:
print(f"VAEdecoderwillbeloadedfrom{VAE_DECODER_OV_PATH}")

delvae
gc.collect()


..parsed-literal::

VAEencoderwillbeloadedfromvae_encoder.xml
VAEdecoderwillbeloadedfromvae_decoder.xml




..parsed-literal::

0



Unet
~~~~

`backtotop⬆️<#table-of-contents>`__

TheUnetmodelhasthreeinputs:

-``scaled_latent_model_input``-thelatentimagesamplefromprevious
step.Generationprocesshasnotbeenstartedyet,soyouwilluse
randomnoise.
-``timestep``-acurrentschedulerstep.
-``text_embeddings``-ahiddenstateofthetextencoder.

Modelpredictsthe``sample``stateforthenextstep.

..code::ipython3

importnumpyasnp

UNET_OV_PATH=Path("unet.xml")

dtype_mapping={torch.float32:ov.Type.f32,torch.float64:ov.Type.f64}


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
encoder_hidden_state=torch.ones((3,77,768))
latents_shape=(3,8,512//8,512//8)
latents=torch.randn(latents_shape)
t=torch.from_numpy(np.array(1,dtype=float))
dummy_inputs=(latents,t,encoder_hidden_state)
input_info=[]
forinput_tensorindummy_inputs:
shape=ov.PartialShape(tuple(input_tensor.shape))
element_type=dtype_mapping[input_tensor.dtype]
input_info.append((shape,element_type))

unet.eval()
withtorch.no_grad():
ov_model=ov.convert_model(unet,example_input=dummy_inputs,input=input_info)
ov.save_model(ov_model,ir_path)
delov_model
cleanup_torchscript_cache()
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



PrepareInferencePipeline
--------------------------

`backtotop⬆️<#table-of-contents>`__

Puttingitalltogether,letusnowtakeacloserlookathowthemodel
inferenceworksbyillustratingthelogicalflow.

..figure::https://user-images.githubusercontent.com/29454499/214895365-3063ac11-0486-4d9b-9e25-8f469aba5e5d.png
:alt:diagram

diagram

TheInstructPix2Pixmodeltakesbothanimageandatextpromptasan
input.Theimageistransformedtolatentimagerepresentationsofsize
:math:`64\times64`,usingtheencoderpartofvariationalautoencoder,
whereasthetextpromptistransformedtotextembeddingsofsize
:math:`77\times768`viaCLIP’stextencoder.

Next,theUNetmodeliteratively*denoises*therandomlatentimage
representationswhilebeingconditionedonthetextembeddings.The
outputoftheUNet,beingthenoiseresidual,isusedtocomputea
denoisedlatentimagerepresentationviaascheduleralgorithm.

The*denoising*processisrepeatedagivennumberoftimes(bydefault
100)toretrievestep-by-stepbetterlatentimagerepresentations.Once
ithasbeencompleted,thelatentimagerepresentationisdecodedbythe
decoderpartofthevariationalautoencoder.

..code::ipython3

fromdiffusersimportDiffusionPipeline
fromtransformersimportCLIPTokenizer
fromtypingimportUnion,List,Optional,Tuple
importPIL
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
pad(Tuple[int]):padingsizeforeachdimensionforrestoringimagesizeinpostprocessing
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


classOVInstructPix2PixPipeline(DiffusionPipeline):
"""
OpenVINOinferencepipelineforInstructPix2Pix
"""

def__init__(
self,
tokenizer:CLIPTokenizer,
scheduler:EulerAncestralDiscreteScheduler,
core:ov.Core,
text_encoder:ov.Model,
vae_encoder:ov.Model,
unet:ov.Model,
vae_decoder:ov.Model,
device:str="AUTO",
):
super().__init__()
self.tokenizer=tokenizer
self.vae_scale_factor=8
self.scheduler=scheduler
self.load_models(core,device,text_encoder,vae_encoder,unet,vae_decoder)

defload_models(
self,
core:ov.Core,
device:str,
text_encoder:ov.Model,
vae_encoder:ov.Model,
unet:ov.Model,
vae_decoder:ov.Model,
):
"""
FunctionforloadingmodelsondeviceusingOpenVINO

Parameters:
core(Core):OpenVINOruntimeCoreclassinstance
device(str):inferencedevice
text_encoder(Model):OpenVINOModelobjectrepresentstextencoder
vae_encoder(Model):OpenVINOModelobjectrepresentsvaeencoder
unet(Model):OpenVINOModelobjectrepresentsunet
vae_decoder(Model):OpenVINOModelobjectrepresentsvaedecoder
Returns
None
"""
self.text_encoder=core.compile_model(text_encoder,device)
self.text_encoder_out=self.text_encoder.output(0)
ov_config={"INFERENCE_PRECISION_HINT":"f32"}ifdevice!="CPU"else{}
self.vae_encoder=core.compile_model(vae_encoder,device,ov_config)
self.vae_encoder_out=self.vae_encoder.output(0)
#WehavetoregisterUNetinconfigtobeabletochangeitexternallytocollectcalibrationdata
self.register_to_config(unet=core.compile_model(unet,device))
self.unet_out=self.unet.output(0)
self.vae_decoder=core.compile_model(vae_decoder,device,ov_config)
self.vae_decoder_out=self.vae_decoder.output(0)

def__call__(
self,
prompt:Union[str,List[str]],
image:PIL.Image.Image,
num_inference_steps:int=10,
guidance_scale:float=7.5,
image_guidance_scale:float=1.5,
eta:float=0.0,
latents:Optional[np.array]=None,
output_type:Optional[str]="pil",
):
"""
Functioninvokedwhencallingthepipelineforgeneration.

Parameters:
prompt(`str`or`List[str]`):
Thepromptorpromptstoguidetheimagegeneration.
image(`PIL.Image.Image`):
`Image`,ortensorrepresentinganimagebatchwhichwillberepaintedaccordingto`prompt`.
num_inference_steps(`int`,*optional*,defaultsto100):
Thenumberofdenoisingsteps.Moredenoisingstepsusuallyleadtoahigherqualityimageatthe
expenseofslowerinference.
guidance_scale(`float`,*optional*,defaultsto7.5):
Guidancescaleasdefinedin[Classifier-FreeDiffusionGuidance](https://arxiv.org/abs/2207.12598).
`guidance_scale`isdefinedas`w`ofequation2.of[Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf).Guidancescaleisenabledbysetting`guidance_scale>
1`.Higherguidancescaleencouragestogenerateimagesthatarecloselylinkedtothetext`prompt`,
usuallyattheexpenseoflowerimagequality.Thispipelinerequiresavalueofatleast`1`.
image_guidance_scale(`float`,*optional*,defaultsto1.5):
Imageguidancescaleistopushthegeneratedimagetowardstheinitalimage`image`.Imageguidance
scaleisenabledbysetting`image_guidance_scale>1`.Higherimageguidancescaleencouragesto
generateimagesthatarecloselylinkedtothesourceimage`image`,usuallyattheexpenseoflower
imagequality.Thispipelinerequiresavalueofatleast`1`.
latents(`torch.FloatTensor`,*optional*):
Pre-generatednoisylatents,sampledfromaGaussiandistribution,tobeusedasinputsforimage
generation.Canbeusedtotweakthesamegenerationwithdifferentprompts.Ifnotprovided,alatents
tensorwillgegeneratedbysamplingusingthesuppliedrandom`generator`.
output_type(`str`,*optional*,defaultsto`"pil"`):
Theoutputformatofthegenerateimage.Choosebetween
[PIL](https://pillow.readthedocs.io/en/stable/):`PIL.Image.Image`or`np.array`.
Returns:
image([List[Union[np.ndarray,PIL.Image.Image]]):generaitedimages

"""

#1.Definecallparameters
batch_size=1ifisinstance(prompt,str)elselen(prompt)
#here`guidance_scale`isdefinedanalogtotheguidanceweight`w`ofequation(2)
#oftheImagenpaper:https://arxiv.org/pdf/2205.11487.pdf.`guidance_scale=1`
#correspondstodoingnoclassifierfreeguidance.
do_classifier_free_guidance=guidance_scale>1.0andimage_guidance_scale>=1.0
#checkifschedulerisinsigmasspace
scheduler_is_in_sigma_space=hasattr(self.scheduler,"sigmas")

#2.Encodeinputprompt
text_embeddings=self._encode_prompt(prompt)

#3.Preprocessimage
orig_width,orig_height=image.size
image,pad=preprocess(image)
height,width=image.shape[-2:]

#4.settimesteps
self.scheduler.set_timesteps(num_inference_steps)
timesteps=self.scheduler.timesteps

#5.PrepareImagelatents
image_latents=self.prepare_image_latents(
image,
do_classifier_free_guidance=do_classifier_free_guidance,
)

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
latent_model_input=np.concatenate([latents]*3)ifdo_classifier_free_guidanceelselatents

#concatlatents,image_latentsinthechanneldimension
scaled_latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)
scaled_latent_model_input=np.concatenate([scaled_latent_model_input,image_latents],axis=1)

#predictthenoiseresidual
noise_pred=self.unet([scaled_latent_model_input,t,text_embeddings])[self.unet_out]

#Hack:
#Forkarrasstyleschedulersthemodeldoesclassifierfreeguidanceusingthe
#predicted_original_sampleinsteadofthenoise_pred.Soweneedtocomputethe
#predicted_original_samplehereifweareusingakarrasstylescheduler.
ifscheduler_is_in_sigma_space:
step_index=(self.scheduler.timesteps==t).nonzero().item()
sigma=self.scheduler.sigmas[step_index].numpy()
noise_pred=latent_model_input-sigma*noise_pred

#performguidance
ifdo_classifier_free_guidance:
noise_pred_text,noise_pred_image,noise_pred_uncond=(
noise_pred[0],
noise_pred[1],
noise_pred[2],
)
noise_pred=(
noise_pred_uncond
+guidance_scale*(noise_pred_text-noise_pred_image)
+image_guidance_scale*(noise_pred_image-noise_pred_uncond)
)

#Forkarrasstyleschedulersthemodeldoesclassifierfreeguidanceusingthe
#predicted_original_sampleinsteadofthenoise_pred.Butthescheduler.stepfunction
#expectsthenoise_predandcomputesthepredicted_original_sampleinternally.Sowe
#needtooverwritethenoise_predheresuchthatthevalueofthecomputed
#predicted_original_sampleiscorrect.
ifscheduler_is_in_sigma_space:
noise_pred=(noise_pred-latents)/(-sigma)

#computethepreviousnoisysamplex_t->x_t-1
latents=self.scheduler.step(torch.from_numpy(noise_pred),t,torch.from_numpy(latents)).prev_sample.numpy()

#callthecallback,ifprovided
ifi==len(timesteps)-1or((i+1)>num_warmup_stepsand(i+1)%self.scheduler.order==0):
progress_bar.update()

#8.Post-processing
image=self.decode_latents(latents,pad)

#9.ConverttoPIL
ifoutput_type=="pil":
image=self.numpy_to_pil(image)
image=[img.resize((orig_width,orig_height),PIL.Image.Resampling.LANCZOS)forimginimage]
else:
image=[cv2.resize(img,(orig_width,orig_width))forimginimage]

returnimage

def_encode_prompt(
self,
prompt:Union[str,List[str]],
num_images_per_prompt:int=1,
do_classifier_free_guidance:bool=True,
):
"""
Encodesthepromptintotextencoderhiddenstates.

Parameters:
prompt(strorlist(str)):prompttobeencoded
num_images_per_prompt(int):numberofimagesthatshouldbegeneratedperprompt
do_classifier_free_guidance(bool):whethertouseclassifierfreeguidanceornot
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

#duplicatetextembeddingsforeachgenerationperprompt,usingmpsfriendlymethod
ifnum_images_per_prompt!=1:
bs_embed,seq_len,_=text_embeddings.shape
text_embeddings=np.tile(text_embeddings,(1,num_images_per_prompt,1))
text_embeddings=np.reshape(text_embeddings,(bs_embed*num_images_per_prompt,seq_len,-1))

#getunconditionalembeddingsforclassifierfreeguidance
ifdo_classifier_free_guidance:
uncond_tokens:List[str]
uncond_tokens=[""]*batch_size
max_length=text_input_ids.shape[-1]
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

#Forclassifierfreeguidance,youneedtodotwoforwardpasses.
#Here,youconcatenatetheunconditionalandtextembeddingsintoasinglebatch
#toavoiddoingtwoforwardpasses
text_embeddings=np.concatenate([text_embeddings,uncond_embeddings,uncond_embeddings])

returntext_embeddings

defprepare_image_latents(
self,
image,
batch_size=1,
num_images_per_prompt=1,
do_classifier_free_guidance=True,
):
"""
EncodesinputimagetolatentspaceusingVAEEncoder

Parameters:
image(np.ndarray):inputimagetensor
num_image_per_prompt(int,*optional*,1):numberofimagegeneratedforpromt
do_classifier_free_guidance(bool):whethertouseclassifierfreeguidanceornot
Returns:
image_latents:imageencodedtolatentspace
"""

image=image.astype(np.float32)

batch_size=batch_size*num_images_per_prompt
image_latents=self.vae_encoder(image)[self.vae_encoder_out]

ifbatch_size>image_latents.shape[0]andbatch_size%image_latents.shape[0]==0:
#expandimage_latentsforbatch_size
additional_image_per_prompt=batch_size//image_latents.shape[0]
image_latents=np.concatenate([image_latents]*additional_image_per_prompt,axis=0)
elifbatch_size>image_latents.shape[0]andbatch_size%image_latents.shape[0]!=0:
raiseValueError(f"Cannotduplicate`image`ofbatchsize{image_latents.shape[0]}to{batch_size}textprompts.")
else:
image_latents=np.concatenate([image_latents],axis=0)

ifdo_classifier_free_guidance:
uncond_image_latents=np.zeros_like(image_latents)
image_latents=np.concatenate([image_latents,image_latents,uncond_image_latents],axis=0)

returnimage_latents

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
latents=latents*self.scheduler.init_noise_sigma.numpy()
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

importmatplotlib.pyplotasplt


defvisualize_results(
orig_img:PIL.Image.Image,
processed_img:PIL.Image.Image,
img1_title:str,
img2_title:str,
):
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
im_w,im_h=orig_img.size
is_horizontal=im_h<=im_w
figsize=(20,30)ifis_horizontalelse(30,20)
fig,axs=plt.subplots(
1ifis_horizontalelse2,
2ifis_horizontalelse1,
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
list_axes[1].imshow(np.array(processed_img))
list_axes[0].set_title(img1_title,fontsize=20)
list_axes[1].set_title(img2_title,fontsize=20)
fig.subplots_adjust(wspace=0.0ifis_horizontalelse0.01,hspace=0.01ifis_horizontalelse0.0)
fig.tight_layout()
fig.savefig("result.png",bbox_inches="tight")
returnfig

Modeltokenizerandschedulerarealsoimportantpartsofthepipeline.
Letusdefinethemandputallcomponentstogether.Additionally,you
canprovidedeviceselectingonefromavailableindropdownlist.

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

fromtransformersimportCLIPTokenizer

tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
scheduler=EulerAncestralDiscreteScheduler.from_config(scheduler_config)

ov_pipe=OVInstructPix2PixPipeline(
tokenizer,
scheduler,
core,
TEXT_ENCODER_OV_PATH,
VAE_ENCODER_OV_PATH,
UNET_OV_PATH,
VAE_DECODER_OV_PATH,
device=device.value,
)


..parsed-literal::

/home/ltalamanova/env_ci/lib/python3.8/site-packages/diffusers/configuration_utils.py:134:FutureWarning:Accessingconfigattribute`unet`directlyvia'OVInstructPix2PixPipeline'objectattributeisdeprecated.Pleaseaccess'unet'over'OVInstructPix2PixPipeline'sconfigobjectinstead,e.g.'scheduler.config.unet'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)


Now,youarereadytodefineeditinginstructionsandanimagefor
runningtheinferencepipeline.Youcanfindexampleresultsgenerated
bythemodelonthis
`page<https://www.timothybrooks.com/instruct-pix2pix/>`__,incaseyou
needinspiration.Optionally,youcanalsochangetherandomgenerator
seedforlatentstateinitializationandnumberofsteps.

**Note**:Considerincreasing``steps``togetmorepreciseresults.
Asuggestedvalueis``100``,butitwilltakemoretimetoprocess.

..code::ipython3

style={"description_width":"initial"}
text_prompt=widgets.Text(value="Makeitingalaxy",description="yourtext")
num_steps=widgets.IntSlider(min=1,max=100,value=10,description="steps:")
seed=widgets.IntSlider(min=0,max=1024,description="seed:",value=42)
image_widget=widgets.FileUpload(accept="",multiple=False,description="Uploadimage",style=style)
widgets.VBox([text_prompt,seed,num_steps,image_widget])




..parsed-literal::

VBox(children=(Text(value='Makeitingalaxy',description='yourtext'),IntSlider(value=42,description='see…



**Note**:Diffusionprocesscantakesometime,dependingonwhat
hardwareyouselect.

..code::ipython3

importio
importrequests

default_url="https://user-images.githubusercontent.com/29454499/223343459-4ac944f0-502e-4acf-9813-8e9f0abc8a16.jpg"
#readuploadedimage
image=PIL.Image.open(io.BytesIO(image_widget.value[-1]["content"])ifimage_widget.valueelserequests.get(default_url,stream=True).raw)
image=image.convert("RGB")
print("Pipelinesettings")
print(f"Inputtext:{text_prompt.value}")
print(f"Seed:{seed.value}")
print(f"Numberofsteps:{num_steps.value}")
np.random.seed(seed.value)
processed_image=ov_pipe(text_prompt.value,image,num_steps.value)


..parsed-literal::

Pipelinesettings
Inputtext:Makeitingalaxy
Seed:42
Numberofsteps:10



..parsed-literal::

0%||0/10[00:00<?,?it/s]


Now,letuslookattheresults.Thetopimagerepresentstheoriginal
beforeediting.Thebottomimageistheresultoftheeditingprocess.
Thetitlebetweenthemcontainsthetextinstructionsusedfor
generation.

..code::ipython3

fig=visualize_results(
image,
processed_image[0],
img1_title="Originalimage",
img2_title=f"Prompt:{text_prompt.value}",
)



..image::instruct-pix2pix-image-editing-with-output_files/instruct-pix2pix-image-editing-with-output_24_0.png


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

Accordingto``InstructPix2Pix``pipelinestructure,UNetusedfor
iterativedenoisingofinput.Itmeansthatmodelrunsinthecycle
repeatinginferenceoneachdiffusionstep,whileotherpartsof
pipelinetakepartonlyonce.Thatiswhycomputationcostandspeedof
UNetdenoisingbecomesthecriticalpathinthepipeline.

Theoptimizationprocesscontainsthefollowingsteps:

1.Createacalibrationdatasetforquantization.
2.Run``nncf.quantize()``toobtainquantizedmodel.
3.Savethe``INT8``modelusing``openvino.save_model()``function.

Pleaseselectbelowwhetheryouwouldliketorunquantizationto
improvemodelinferencespeed.

..code::ipython3

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
`fusing/instructpix2pix-1000-samples<https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples>`__
datasetfromHuggingFaceascalibrationdata.Tocollectintermediate
modelinputsforcalibrationweshouldcustomize``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

importdatasets
fromtqdm.notebookimporttqdm
fromtransformersimportPipeline
fromtypingimportAny,Dict,List

classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model,prob:float,data_cache:List[Any]=None):
super().__init__(compiled_model)
self.data_cache=data_cacheifdata_cacheelse[]
self.prob=np.clip(prob,0,1)

def__call__(self,*args,**kwargs):
ifnp.random.rand()>=self.prob:
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)

defcollect_calibration_data(pix2pix_pipeline:Pipeline,subset_size:int)->List[Dict]:
original_unet=pix2pix_pipeline.unet
pix2pix_pipeline.unet=CompiledModelDecorator(original_unet,prob=0.3)
dataset=datasets.load_dataset("fusing/instructpix2pix-1000-samples",split="train",streaming=True).shuffle(seed=42)
pix2pix_pipeline.set_progress_bar_config(disable=True)

#Runinferencefordatacollection
pbar=tqdm(total=subset_size)
diff=0
forbatchindataset:
prompt=batch["edit_prompt"]
image=batch["input_image"].convert("RGB")
_=pix2pix_pipeline(prompt,image)
collected_subset_size=len(pix2pix_pipeline.unet.data_cache)
ifcollected_subset_size>=subset_size:
pbar.update(subset_size-pbar.n)
break
pbar.update(collected_subset_size-diff)
diff=collected_subset_size

calibration_dataset=pix2pix_pipeline.unet.data_cache
pix2pix_pipeline.set_progress_bar_config(disable=False)
pix2pix_pipeline.unet=original_unet
returncalibration_dataset

..code::ipython3

%%skipnot$to_quantize.value

UNET_INT8_OV_PATH=Path("unet_int8.xml")
ifnotUNET_INT8_OV_PATH.exists():
subset_size=300
unet_calibration_data=collect_calibration_data(ov_pipe,subset_size=subset_size)


..parsed-literal::

/home/ltalamanova/env_ci/lib/python3.8/site-packages/diffusers/configuration_utils.py:134:FutureWarning:Accessingconfigattribute`unet`directlyvia'OVInstructPix2PixPipeline'objectattributeisdeprecated.Pleaseaccess'unet'over'OVInstructPix2PixPipeline'sconfigobjectinstead,e.g.'scheduler.config.unet'.
deprecate("directconfignameaccess","1.0.0",deprecation_message,standard_warn=False)



..parsed-literal::

0%||0/300[00:00<?,?it/s]


Runquantization
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Createaquantizedmodelfromthepre-trainedconvertedOpenVINOmodel.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

%%skipnot$to_quantize.value

importnncf

ifUNET_INT8_OV_PATH.exists():
print("Loadingquantizedmodel")
quantized_unet=core.read_model(UNET_INT8_OV_PATH)
else:
unet=core.read_model(UNET_OV_PATH)
quantized_unet=nncf.quantize(
model=unet,
subset_size=subset_size,
calibration_dataset=nncf.Dataset(unet_calibration_data),
model_type=nncf.ModelType.TRANSFORMER
)
ov.save_model(quantized_unet,UNET_INT8_OV_PATH)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..parsed-literal::

Statisticscollection:100%|██████████|300/300[06:48<00:00,1.36s/it]
ApplyingSmoothQuant:100%|██████████|100/100[00:07<00:00,13.51it/s]


..parsed-literal::

INFO:nncf:96ignorednodeswasfoundbynameintheNNCFGraph


..parsed-literal::

Statisticscollection:100%|██████████|300/300[14:34<00:00,2.91s/it]
ApplyingFastBiascorrection:100%|██████████|186/186[05:31<00:00,1.78s/it]


LetuscheckpredictionswiththequantizedUNetusingthesameinput
data.

..code::ipython3

%%skipnot$to_quantize.value

print('Pipelinesettings')
print(f'Inputtext:{text_prompt.value}')
print(f'Seed:{seed.value}')
print(f'Numberofsteps:{num_steps.value}')
np.random.seed(seed.value)

int8_pipe=OVInstructPix2PixPipeline(tokenizer,scheduler,core,TEXT_ENCODER_OV_PATH,VAE_ENCODER_OV_PATH,UNET_INT8_OV_PATH,VAE_DECODER_OV_PATH,device=device.value)
int8_processed_image=int8_pipe(text_prompt.value,image,num_steps.value)

fig=visualize_results(processed_image[0],int8_processed_image[0],img1_title="FP16result",img2_title="INT8result")


..parsed-literal::

Pipelinesettings
Inputtext:Makeitingalaxy
Seed:42
Numberofsteps:10



..parsed-literal::

0%||0/10[00:00<?,?it/s]



..image::instruct-pix2pix-image-editing-with-output_files/instruct-pix2pix-image-editing-with-output_36_2.png


CompareinferencetimeoftheFP16andINT8models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tomeasuretheinferenceperformanceofthe``FP16``and``INT8``
models,weusemedianinferencetimeoncalibrationsubset.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.

..code::ipython3

%%skipnot$to_quantize.value

importtime

calibration_dataset=datasets.load_dataset("fusing/instructpix2pix-1000-samples",split="train",streaming=True)
validation_data=[]
validation_size=10
whilelen(validation_data)<validation_size:
batch=next(iter(calibration_dataset))
prompt=batch["edit_prompt"]
input_image=batch["input_image"].convert("RGB")
validation_data.append((prompt,input_image))

defcalculate_inference_time(pix2pix_pipeline,calibration_dataset,size=10):
inference_time=[]
pix2pix_pipeline.set_progress_bar_config(disable=True)
for(prompt,image)incalibration_dataset:
start=time.perf_counter()
_=pix2pix_pipeline(prompt,image)
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

Performancespeedup:1.437


InteractivedemowithGradio
----------------------------

`backtotop⬆️<#table-of-contents>`__

**Note**:Diffusionprocesscantakesometime,dependingonwhat
hardwareyouselect.

..code::ipython3

pipe_precision=widgets.Dropdown(
options=["FP16"]ifnotto_quantize.valueelse["FP16","INT8"],
value="FP16",
description="Precision:",
disabled=False,
)

pipe_precision




..parsed-literal::

Dropdown(description='Precision:',options=('FP16','INT8'),value='FP16')



..code::ipython3

importgradioasgr
frompathlibimportPath
importnumpyasnp

default_url="https://user-images.githubusercontent.com/29454499/223343459-4ac944f0-502e-4acf-9813-8e9f0abc8a16.jpg"
path=Path("data/example.jpg")
path.parent.mkdir(parents=True,exist_ok=True)

r=requests.get(default_url)

withpath.open("wb")asf:
f.write(r.content)

pipeline=int8_pipeifpipe_precision.value=="INT8"elseov_pipe


defgenerate(img,text,seed,num_steps,_=gr.Progress(track_tqdm=True)):
ifimgisNone:
raisegr.Error("Pleaseuploadanimageorchooseonefromtheexampleslist")
np.random.seed(seed)
result=pipeline(text,img,num_steps)[0]
returnresult


demo=gr.Interface(
generate,
[
gr.Image(label="Image",type="pil"),
gr.Textbox(label="Text"),
gr.Slider(0,1024,label="Seed",value=42),
gr.Slider(
1,
100,
label="Steps",
value=10,
info="Considerincreasingthevaluetogetmorepreciseresults.Asuggestedvalueis100,butitwilltakemoretimetoprocess.",
),
],
gr.Image(label="Result"),
examples=[[path,"Makeitingalaxy"]],
)

try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
