Text-to-ImageGenerationwithStableDiffusionv2andOpenVINO™
===============================================================

StableDiffusionv2isthenextgenerationofStableDiffusionmodela
Text-to-Imagelatentdiffusionmodelcreatedbytheresearchersand
engineersfrom`StabilityAI<https://stability.ai/>`__and
`LAION<https://laion.ai/>`__.

Generaldiffusionmodelsaremachinelearningsystemsthataretrained
todenoiserandomgaussiannoisestepbystep,togettoasampleof
interest,suchasanimage.Diffusionmodelshaveshowntoachieve
state-of-the-artresultsforgeneratingimagedata.Butonedownsideof
diffusionmodelsisthatthereversedenoisingprocessisslow.In
addition,thesemodelsconsumealotofmemorybecausetheyoperatein
pixelspace,whichbecomesunreasonablyexpensivewhengenerating
high-resolutionimages.Therefore,itischallengingtotrainthese
modelsandalsousethemforinference.OpenVINObringscapabilitiesto
runmodelinferenceonIntelhardwareandopensthedoortothe
fantasticworldofdiffusionmodelsforeveryone!

Inpreviousnotebooks,wealreadydiscussedhowtorun`Text-to-Image
generationandImage-to-ImagegenerationusingStableDiffusion
v1<stable-diffusion-text-to-image-with-output.html>`__
and`controllingitsgenerationprocessusing
ControlNet<./controlnet-stable-diffusion/controlnet-stable-diffusion.ipynb>`__.
NowisturnofStableDiffusionv2.

StableDiffusionv2:What’snew?
--------------------------------

Thenewstablediffusionmodeloffersabunchofnewfeaturesinspired
bytheothermodelsthathaveemergedsincetheintroductionofthe
firstiteration.Someofthefeaturesthatcanbefoundinthenewmodel
are:

-Themodelcomeswithanewrobustencoder,OpenCLIP,createdbyLAION
andaidedbyStabilityAI;thisversionv2significantlyenhancesthe
producedphotosovertheV1versions.
-Themodelcannowgenerateimagesina768x768resolution,offering
moreinformationtobeshowninthegeneratedimages.
-Themodelfinetunedwith
`v-objective<https://arxiv.org/abs/2202.00512>`__.The
v-parameterizationisparticularlyusefulfornumericalstability
throughoutthediffusionprocesstoenableprogressivedistillation
formodels.Formodelsthatoperateathigherresolution,itisalso
discoveredthatthev-parameterizationavoidscolorshifting
artifactsthatareknowntoaffecthighresolutiondiffusionmodels,
andinthevideosettingitavoidstemporalcolorshiftingthat
sometimesappearswithepsilon-predictionusedinStableDiffusion
v1.
-Themodelalsocomeswithanewdiffusionmodelcapableofrunning
upscalingontheimagesgenerated.Upscaledimagescanbeadjustedup
to4timestheoriginalimage.Providedasseparatedmodel,formore
detailspleasecheck
`stable-diffusion-x4-upscaler<https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler>`__
-Themodelcomeswithanewrefineddeptharchitecturecapableof
preservingcontextfrompriorgenerationlayersinanimage-to-image
setting.Thisstructurepreservationhelpsgenerateimagesthat
preservingformsandshadowofobjects,butwithdifferentcontent.
-Themodelcomeswithanupdatedinpaintingmodulebuiltuponthe
previousmodel.Thistext-guidedinpaintingmakesswitchingoutparts
intheimageeasierthanbefore.

ThisnotebookdemonstrateshowtoconvertandrunStableDiffusionv2
modelusingOpenVINO.

Notebookcontainsthefollowingsteps:

1.CreatePyTorchmodelspipelineusingDiffuserslibrary.
2.ConvertPyTorchmodelstoOpenVINOIRformat,usingmodelconversion
API.
3.Applyhybridpost-trainingquantizationtoUNetmodelwith
`NNCF<https://github.com/openvinotoolkit/nncf/>`__.
4.RunStableDiffusionv2Text-to-ImagepipelinewithOpenVINO.

**Note:**ThisisthefullversionoftheStableDiffusiontext-to-image
implementation.Ifyouwouldliketogetstartedandrunthenotebook
quickly,checkout`stable-diffusion-v2-text-to-image-demo
notebook<stable-diffusion-v2-with-output.html>`__.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`StableDiffusionv2forText-to-Image
Generation<#stable-diffusion-v2-for-text-to-image-generation>`__

-`StableDiffusioninDiffusers
library<#stable-diffusion-in-diffusers-library>`__
-`ConvertmodelstoOpenVINOIntermediaterepresentation(IR)
format<#convert-models-to-openvino-intermediate-representation-ir-format>`__
-`TextEncoder<#text-encoder>`__
-`U-Net<#u-net>`__
-`VAE<#vae>`__
-`PrepareInferencePipeline<#prepare-inference-pipeline>`__
-`ConfigureInferencePipeline<#configure-inference-pipeline>`__

-`Quantization<#quantization>`__

-`Preparecalibrationdataset<#prepare-calibration-dataset>`__
-`RunHybridModelQuantization<#run-hybrid-model-quantization>`__
-`CompareinferencetimeoftheFP16andINT8
pipelines<#compare-inference-time-of-the-fp16-and-int8-pipelines>`__

-`RunText-to-Imagegeneration<#run-text-to-image-generation>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

installrequiredpackages

..code::ipython3

%pipinstall-q"diffusers>=0.14.0""openvino>=2023.1.0""datasets>=2.14.6""transformers>=4.25.1""gradio>=4.19""torch>=2.1"Pillowopencv-python--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"nncf>=2.9.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


StableDiffusionv2forText-to-ImageGeneration
------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Tostart,let’slookonText-to-ImageprocessforStableDiffusionv2.
Wewilluse`StableDiffusion
v2-1<https://huggingface.co/stabilityai/stable-diffusion-2-1>`__model
forthesepurposes.ThemaindifferencefromStableDiffusionv2and
StableDiffusionv2.1isusageofmoredata,moretraining,andless
restrictivefilteringofthedataset,thatgivespromisingresultsfor
selectingwiderangeofinputtextprompts.Moredetailsaboutmodelcan
befoundin`StabilityAIblog
post<https://stability.ai/blog/stablediffusion2-1-release7-dec-2022>`__
andoriginalmodel
`repository<https://github.com/Stability-AI/stablediffusion>`__.

StableDiffusioninDiffuserslibrary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__ToworkwithStableDiffusion
v2,wewilluseHuggingFace
`Diffusers<https://github.com/huggingface/diffusers>`__library.To
experimentwithStableDiffusionmodels,Diffusersexposesthe
`StableDiffusionPipeline<https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation>`__
similartothe`otherDiffusers
pipelines<https://huggingface.co/docs/diffusers/api/pipelines/overview>`__.
Thecodebelowdemonstrateshowtocreate``StableDiffusionPipeline``
using``stable-diffusion-2-1``:

..code::ipython3

fromdiffusersimportStableDiffusionPipeline

pipe=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to("cpu")

#forreducingmemoryconsumptiongetallcomponentsfrompipelineindependently
text_encoder=pipe.text_encoder
text_encoder.eval()
unet=pipe.unet
unet.eval()
vae=pipe.vae
vae.eval()

conf=pipe.scheduler.config

delpipe



..parsed-literal::

Loadingpipelinecomponents...:0%||0/6[00:00<?,?it/s]


ConvertmodelstoOpenVINOIntermediaterepresentation(IR)format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Startingfrom2023.0release,OpenVINOsupportsPyTorchmodelsdirectly
viaModelConversionAPI.``ov.convert_model``functionacceptsinstance
ofPyTorchmodelandexampleinputsfortracingandreturnsobjectof
``ov.Model``class,readytouseorsaveondiskusing``ov.save_model``
function.

Thepipelineconsistsofthreeimportantparts:

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

Theinputofthetextencoderistensor``input_ids``,whichcontains
indexesoftokensfromtextprocessedbythetokenizerandpaddedtothe
maximumlengthacceptedbythemodel.Modeloutputsaretwotensors:
``last_hidden_state``-hiddenstatefromthelastMultiHeadAttention
layerinthemodeland``pooler_out``-pooledoutputforwholemodel
hiddenstates.

..code::ipython3

frompathlibimportPath

sd2_1_model_dir=Path("sd2.1")
sd2_1_model_dir.mkdir(exist_ok=True)

..code::ipython3

importgc
importtorch
importopenvinoasov

TEXT_ENCODER_OV_PATH=sd2_1_model_dir/"text_encoder.xml"


defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()


defconvert_encoder(text_encoder:torch.nn.Module,ir_path:Path):
"""
ConvertTextEncodermodeltoIR.
Functionacceptspipeline,preparesexampleinputsforconversion
Parameters:
text_encoder(torch.nn.Module):textencoderPyTorchmodel
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
#exportmodel
ov_model=ov.convert_model(
text_encoder,#modelinstance
example_input=input_ids,#exampleinputsformodeltracing
input=([1,77],),#inputshapeforconversion
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
gc.collect();


..parsed-literal::

Textencoderwillbeloadedfromsd2.1/text_encoder.xml


U-Net
~~~~~

`backtotop⬆️<#table-of-contents>`__

U-Netmodelgraduallydenoiseslatentimagerepresentationguidedby
textencoderhiddenstate.

U-Netmodelhasthreeinputs:

-``sample``-latentimagesamplefrompreviousstep.Generation
processhasnotbeenstartedyet,soyouwilluserandomnoise.
-``timestep``-currentschedulerstep.
-``encoder_hidden_state``-hiddenstateoftextencoder.

Modelpredictsthe``sample``stateforthenextstep.

Generally,U-NetmodelconversionprocessremainthesamelikeinStable
Diffusionv1,expectsmallchangesininputsamplesize.Ourmodelwas
pretrainedtogenerateimageswithresolution768x768,initiallatent
samplesizeforthiscaseis96x96.Besidesthat,fordifferentuse
caseslikeinpaintinganddepthtoimagegenerationmodelalsocan
acceptadditionalimageinformation:depthmapormaskaschannel-wise
concatenationwithinitiallatentsample.ForconvertingU-Netmodelfor
suchusecasesrequiredtomodifynumberofinputchannels.

..code::ipython3

importnumpyasnp

UNET_OV_PATH=sd2_1_model_dir/"unet.xml"


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
encoder_hidden_state=torch.ones((2,77,1024))
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


ifnotUNET_OV_PATH.exists():
convert_unet(unet,UNET_OV_PATH,width=96,height=96)
delunet
gc.collect()
else:
delunet
gc.collect();

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
runinferenceforText-to-Image,thereisnoinitialimageasastarting
point.Youcanskipthisstepanddirectlygenerateinitialrandom
noise.

WhenrunningText-to-Imagepipeline,wewillseethatwe**onlyneedthe
VAEdecoder**,butpreserveVAEencoderconversion,itwillbeusefulin
nextchapterofourtutorial.

Note:Thisprocesswilltakeafewminutesandusesignificantamountof
RAM(recommendedatleast32GB).

..code::ipython3

VAE_ENCODER_OV_PATH=sd2_1_model_dir/"vae_encoder.xml"


defconvert_vae_encoder(vae:torch.nn.Module,ir_path:Path,width:int=512,height:int=512):
"""
ConvertVAEmodeltoIRformat.
VAEmodel,createswrapperclassforexportonlynecessaryforinferencepart,
preparesexampleinputsforonversion
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
returnself.vae.encode(x=image)["latent_dist"].sample()

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
preparesexampleinputsforconversion
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
print("VAEdecodersuccessfullyconvertedtoIR")


ifnotVAE_ENCODER_OV_PATH.exists():
convert_vae_encoder(vae,VAE_ENCODER_OV_PATH,768,768)
else:
print(f"VAEencoderwillbeloadedfrom{VAE_ENCODER_OV_PATH}")

VAE_DECODER_OV_PATH=sd2_1_model_dir/"vae_decoder.xml"

ifnotVAE_DECODER_OV_PATH.exists():
convert_vae_decoder(vae,VAE_DECODER_OV_PATH,96,96)
else:
print(f"VAEdecoderwillbeloadedfrom{VAE_DECODER_OV_PATH}")

delvae
gc.collect();


..parsed-literal::

VAEencoderwillbeloadedfromsd2.1/vae_encoder.xml
VAEdecoderwillbeloadedfromsd2.1/vae_decoder.xml


PrepareInferencePipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Puttingitalltogether,letusnowtakeacloserlookathowthemodel
worksininferencebyillustratingthelogicalflow.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/22090501/ec454103-0d28-48e3-a18e-b55da3fab381
:alt:text2img-stable-diffusionv2

text2img-stable-diffusionv2

Thestablediffusionmodeltakesbothalatentseedandatextpromptas
input.Thelatentseedisthenusedtogeneraterandomlatentimage
representationsofsize:math:`96\times96`whereasthetextpromptis
transformedtotextembeddingsofsize:math:`77\times1024`via
OpenCLIP’stextencoder.

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
scheduler<https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py>`__

Theoryonhowthescheduleralgorithmfunctionworksisoutofscopefor
thisnotebook,butinshort,youshouldrememberthattheycomputethe
predicteddenoisedimagerepresentationfromthepreviousnoise
representationandthepredictednoiseresidual.Formoreinformation,
itisrecommendedtolookinto`ElucidatingtheDesignSpaceof
Diffusion-BasedGenerativeModels<https://arxiv.org/abs/2206.00364>`__.

ThechartabovelooksverysimilartoStableDiffusionV1from
`notebook<stable-diffusion-text-to-image-with-output.html>`__,
butthereissomesmalldifferenceindetails:

-ChangedinputresolutionforU-Netmodel.
-Changedtextencoderandastheresultsizeofitshiddenstate
embeddings.
-Additionally,toimproveimagegenerationqualityauthorsintroduced
negativeprompting.Technically,positivepromptsteersthediffusion
towardtheimagesassociatedwithit,whilenegativepromptsteers
thediffusionawayfromit.Inotherwords,negativepromptdeclares
undesiredconceptsforgenerationimage,e.g. ifwewanttohave
colorfulandbrightimage,grayscaleimagewillberesultwhichwe
wanttoavoid,inthiscasegrayscalecanbetreatedasnegative
prompt.Thepositiveandnegativepromptareinequalfooting.You
canalwaysuseonewithorwithouttheother.Moreexplanationofhow
itworkscanbefoundinthis
`article<https://stable-diffusion-art.com/how-negative-prompt-work/>`__.

..code::ipython3

importinspect
fromtypingimportList,Optional,Union,Dict

importPIL
importcv2
importtorch

fromtransformersimportCLIPTokenizer
fromdiffusersimportDiffusionPipeline
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
vae_decoder(Model):
VariationalAuto-Encoder(VAE)Modeltodecodeimagestoandfromlatentrepresentations.
text_encoder(Model):
Frozentext-encoder.StableDiffusionusesthetextportionof
[CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel),specifically
theclip-vit-large-patch14(https://huggingface.co/openai/clip-vit-large-patch14)variant.
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
self.text_encoder=text_encoder
self.unet=unet
self.register_to_config(unet=unet)
self._text_encoder_output=text_encoder.output(0)
self._unet_output=unet.output(0)
self._vae_d_output=vae_decoder.output(0)
self._vae_e_output=vae_encoder.output(0)ifvae_encoderisnotNoneelseNone
self.height=self.unet.input(0).shape[2]*8
self.width=self.unet.input(0).shape[3]*8
self.tokenizer=tokenizer

def__call__(
self,
prompt:Union[str,List[str]],
image:PIL.Image.Image=None,
negative_prompt:Union[str,List[str]]=None,
num_inference_steps:Optional[int]=50,
guidance_scale:Optional[float]=7.5,
eta:Optional[float]=0.0,
output_type:Optional[str]="pil",
seed:Optional[int]=None,
strength:float=1.0,
):
"""
Functioninvokedwhencallingthepipelineforgeneration.
Parameters:
prompt(strorList[str]):
Thepromptorpromptstoguidetheimagegeneration.
image(PIL.Image.Image,*optional*,None):
Intinalimageforgeneration.
negative_prompt(strorList[str]):
Thenegativepromptorpromptstoguidetheimagegeneration.
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
strength(int,*optional*,1.0):
strengthbetweeninitialimageandgeneratedinImage-to-Imagepipeline,donotusedinText-to-Image
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

fortinself.progress_bar(timesteps):
#expandthelatentsifwearedoingclassifierfreeguidance
latent_model_input=np.concatenate([latents]*2)ifdo_classifier_free_guidanceelselatents
latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)

#predictthenoiseresidual
noise_pred=self.unet([latent_model_input,np.array(t,dtype=np.float32),text_embeddings])[self._unet_output]
#performguidance
ifdo_classifier_free_guidance:
noise_pred_uncond,noise_pred_text=noise_pred[0],noise_pred[1]
noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

#computethepreviousnoisysamplex_t->x_t-1
latents=self.scheduler.step(torch.from_numpy(noise_pred),t,torch.from_numpy(latents),**extra_step_kwargs)["prev_sample"].numpy()
#scaleanddecodetheimagelatentswithvae
image=self.vae_decoder(latents*(1/0.18215))[self._vae_d_output]

image=self.postprocess_image(image,meta,output_type)
return{"sample":image}

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
#ifweuseLMSDiscreteScheduler,let'smakesurelatentsaremulitpliedbysigmas
ifisinstance(self.scheduler,LMSDiscreteScheduler):
noise=noise*self.scheduler.sigmas[0].numpy()
returnnoise,{}
input_image,meta=preprocess(image)
latents=self.vae_encoder(input_image)[self._vae_e_output]
latents=latents*0.18215
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
Valuesthatapproach1.0allowforlotsofvariationsbutwillalsoproduceimagesthatarenotsemanticallyconsistentwiththeinput.
"""
#gettheoriginaltimestepusinginit_timestep
init_timestep=min(int(num_inference_steps*strength),num_inference_steps)

t_start=max(num_inference_steps-init_timestep,0)
timesteps=self.scheduler.timesteps[t_start:]

returntimesteps,num_inference_steps-t_start

ConfigureInferencePipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

First,youshouldcreateinstancesofOpenVINOModel.

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

ov_config={"INFERENCE_PRECISION_HINT":"f32"}ifdevice.value!="CPU"else{}

text_enc=core.compile_model(TEXT_ENCODER_OV_PATH,device.value)
unet_model=core.compile_model(UNET_OV_PATH,device.value)
vae_decoder=core.compile_model(VAE_DECODER_OV_PATH,device.value,ov_config)
vae_encoder=core.compile_model(VAE_ENCODER_OV_PATH,device.value,ov_config)

Modeltokenizerandschedulerarealsoimportantpartsofthepipeline.
Letusdefinethemandputallcomponentstogether.

..code::ipython3

fromtransformersimportCLIPTokenizer

scheduler=DDIMScheduler.from_config(conf)#DDIMSchedulerisusedbecauseUNetquantizationproducesbetterresultswithit
tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

ov_pipe=OVStableDiffusionPipeline(
tokenizer=tokenizer,
text_encoder=text_enc,
unet=unet_model,
vae_encoder=vae_encoder,
vae_decoder=vae_decoder,
scheduler=scheduler,
)

Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``StableDiffusionv2``structure,theUNetmodeltakesup
significantportionoftheoverallpipelineexecutiontime.Nowwewill
showyouhowtooptimizetheUNetpartusing
`NNCF<https://github.com/openvinotoolkit/nncf/>`__toreduce
computationcostandspeedupthepipeline.Quantizingtherestofthe
pipelinedoesnotsignificantlyimproveinferenceperformancebutcan
leadtoasubstantialdegradationofaccuracy.

Forthismodelweapplyquantizationinhybridmodewhichmeansthatwe
quantize:(1)weightsofMatMulandEmbeddinglayersand(2)activations
ofotherlayers.Thestepsarethefollowing:

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

int8_ov_pipe=None

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
importnumpyasnp
fromtqdm.notebookimporttqdm
fromtypingimportAny,Dict,List


defdisable_progress_bar(pipeline,disable=True):
ifnothasattr(pipeline,"_progress_bar_config"):
pipeline._progress_bar_config={'disable':disable}
else:
pipeline._progress_bar_config['disable']=disable


classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model:ov.CompiledModel,data_cache:List[Any]=None,keep_prob:float=0.5):
super().__init__(compiled_model)
self.data_cache=data_cacheifdata_cacheisnotNoneelse[]
self.keep_prob=keep_prob

def__call__(self,*args,**kwargs):
ifnp.random.rand()<=self.keep_prob:
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)


defcollect_calibration_data(ov_pipe,calibration_dataset_size:int,num_inference_steps:int)->List[Dict]:
original_unet=ov_pipe.unet
calibration_data=[]
ov_pipe.unet=CompiledModelDecorator(original_unet,calibration_data,keep_prob=0.7)
disable_progress_bar(ov_pipe)

dataset=datasets.load_dataset("google-research-datasets/conceptual_captions",split="train",trust_remote_code=True).shuffle(seed=42)

#Runinferencefordatacollection
pbar=tqdm(total=calibration_dataset_size)
forbatchindataset:
prompt=batch["caption"]
iflen(prompt)>ov_pipe.tokenizer.model_max_length:
continue
ov_pipe(prompt,num_inference_steps=num_inference_steps,seed=1)
pbar.update(len(calibration_data)-pbar.n)
ifpbar.n>=calibration_dataset_size:
break

disable_progress_bar(ov_pipe,disable=False)
ov_pipe.unet=original_unet
returncalibration_data

RunHybridModelQuantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fromcollectionsimportdeque
fromtransformersimportset_seed
importnncf

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

UNET_INT8_OV_PATH=sd2_1_model_dir/'unet_optimized.xml'
ifnotUNET_INT8_OV_PATH.exists():
calibration_dataset_size=300
set_seed(1)
unet_calibration_data=collect_calibration_data(ov_pipe,
calibration_dataset_size=calibration_dataset_size,
num_inference_steps=50)

unet=core.read_model(UNET_OV_PATH)

#Collectoperationswhichweightswillbecompressed
unet_ignored_scope=collect_ops_with_weights(unet)

#Compressmodelweights
compressed_unet=nncf.compress_weights(unet,ignored_scope=nncf.IgnoredScope(types=['Convolution']))

#QuantizebothweightsandactivationsofConvolutionlayers
quantized_unet=nncf.quantize(
model=compressed_unet,
calibration_dataset=nncf.Dataset(unet_calibration_data),
subset_size=calibration_dataset_size,
model_type=nncf.ModelType.TRANSFORMER,
ignored_scope=nncf.IgnoredScope(names=unet_ignored_scope),
advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=-1)
)

ov.save_model(quantized_unet,UNET_INT8_OV_PATH)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,onnx,openvino


..code::ipython3

%%skipnot$to_quantize.value

int8_unet_model=core.compile_model(UNET_INT8_OV_PATH,device.value)
int8_ov_pipe=OVStableDiffusionPipeline(
tokenizer=tokenizer,
text_encoder=text_enc,
unet=int8_unet_model,
vae_encoder=vae_encoder,
vae_decoder=vae_decoder,
scheduler=scheduler
)

CompareUNetfilesize
~~~~~~~~~~~~~~~~~~~~~~

..code::ipython3

%%skipnot$to_quantize.value

fp16_ir_model_size=UNET_OV_PATH.with_suffix(".bin").stat().st_size/1024
quantized_model_size=UNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size/1024

print(f"FP16modelsize:{fp16_ir_model_size:.2f}KB")
print(f"INT8modelsize:{quantized_model_size:.2f}KB")
print(f"Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16modelsize:1691232.51KB
INT8modelsize:846918.58KB
Modelcompressionrate:1.997


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
pipeline.set_progress_bar_config(disable=True)
forpromptinvalidation_data:
start=time.perf_counter()
_=pipeline(prompt,num_inference_steps=10,seed=0)
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
returnnp.median(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

validation_size=10
validation_dataset=datasets.load_dataset("google-research-datasets/conceptual_captions",split="train",streaming=True,trust_remote_code=True).take(validation_size)
validation_data=[batch["caption"]forbatchinvalidation_dataset]

fp_latency=calculate_inference_time(ov_pipe,validation_data)
int8_latency=calculate_inference_time(int8_ov_pipe,validation_data)
print(f"Performancespeed-up:{fp_latency/int8_latency:.3f}")


..parsed-literal::

/home/nsavel/venvs/ov_notebooks_tmp/lib/python3.8/site-packages/datasets/load.py:1429:FutureWarning:Therepositoryforconceptual_captionscontainscustomcodewhichmustbeexecutedtocorrectlyloadthedataset.Youcaninspecttherepositorycontentathttps://hf.co/datasets/conceptual_captions
Youcanavoidthismessageinfuturebypassingtheargument`trust_remote_code=True`.
Passing`trust_remote_code=True`willbemandatorytoloadthisdatasetfromthenextmajorreleaseof`datasets`.
warnings.warn(


..parsed-literal::

Performancespeed-up:1.232


RunText-to-Imagegeneration
----------------------------

`backtotop⬆️<#table-of-contents>`__

Now,youcandefineatextpromptsforimagegenerationandrun
inferencepipeline.Optionally,youcanalsochangetherandomgenerator
seedforlatentstateinitializationandnumberofsteps.

**Note**:Considerincreasing``steps``togetmorepreciseresults.
Asuggestedvalueis``50``,butitwilltakelongertimetoprocess.

Pleaseselectbelowwhetheryouwouldliketousethequantizedmodelto
launchtheinteractivedemo.

..code::ipython3

quantized_model_present=int8_ov_pipeisnotNone

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


pipeline=int8_ov_pipeifuse_quantized_model.valueelseov_pipe


defgenerate(prompt,negative_prompt,seed,num_steps,_=gr.Progress(track_tqdm=True)):
result=pipeline(
prompt,
negative_prompt=negative_prompt,
num_inference_steps=num_steps,
seed=seed,
)
returnresult["sample"][0]


gr.close_all()
demo=gr.Interface(
generate,
[
gr.Textbox(
"valleyintheAlpsatsunset,epicvista,beautifullandscape,4k,8k",
label="Prompt",
),
gr.Textbox(
"frames,borderline,text,charachter,duplicate,error,outofframe,watermark,lowquality,ugly,deformed,blur",
label="Negativeprompt",
),
gr.Slider(value=42,label="Seed",maximum=10000000),
gr.Slider(value=25,label="Steps",minimum=1,maximum=50),
],
"image",
)

try:
demo.queue().launch()
exceptException:
demo.queue().launch(share=True)
