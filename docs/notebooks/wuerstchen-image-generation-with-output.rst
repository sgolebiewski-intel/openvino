ImagegenerationwithWürstchenandOpenVINO
============================================

..figure::attachment:499b779a-61d1-4e68-a1c3-437122622ba7.png
:alt:image.png

image.png

`Würstchen<https://arxiv.org/abs/2306.00637>`__isadiffusionmodel,
whosetext-conditionalmodelworksinahighlycompressedlatentspace
ofimages.Whyisthisimportant?Compressingdatacanreduce
computationalcostsforbothtrainingandinferencebymagnitudes.
Trainingon1024x1024images,iswaymoreexpensivethantrainingat
32x32.Usually,otherworksmakeuseofarelativelysmallcompression,
intherangeof4x-8xspatialcompression.Würstchentakesthistoan
extreme.Throughitsnoveldesign,authorsachievea42xspatial
compression.Thiswasunseenbeforebecausecommonmethodsfailto
faithfullyreconstructdetailedimagesafter16xspatialcompression.
Würstchenemploysatwo-stagecompression(referredbelowas*Decoder*).
ThefirstoneisaVQGAN,andthesecondisaDiffusionAutoencoder
(moredetailscanbefoundinthepaper).Athirdmodel(referredbelow
as*Prior*)islearnedinthathighlycompressedlatentspace.This
trainingrequiresfractionsofthecomputeusedforcurrent
top-performingmodels,allowingalsocheaperandfasterinference.

WewillusePyTorchversionofWürstchen`modelfromHuggingFace
Hub<https://huggingface.co/warp-ai/wuerstchen>`__.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Loadtheoriginalmodel<#load-the-original-model>`__

-`Infertheoriginalmodel<#infer-the-original-model>`__

-`ConvertthemodeltoOpenVINO
IR<#convert-the-model-to-openvino-ir>`__

-`Priorpipeline<#prior-pipeline>`__
-`Decoderpipeline<#decoder-pipeline>`__

-`Compilingmodels<#compiling-models>`__
-`Buildingthepipeline<#building-the-pipeline>`__
-`Inference<#inference>`__
-`Quantization<#quantization>`__

-`Preparecalibrationdatasets<#prepare-calibration-datasets>`__
-`Runquantization<#run-quantization>`__
-`Comparemodelfilesizes<#compare-model-file-sizes>`__
-`CompareinferencetimeoftheFP16andINT8
pipelines<#compare-inference-time-of-the-fp16-and-int8-pipelines>`__

-`Interactiveinference<#interactive-inference>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

%pipinstall-q"diffusers>=0.21.0""torch>=2.1"transformersaccelerate"gradio>=4.19""openvino>=2023.2.0""peft==0.6.2"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-qdatasets"nncf>=2.7.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath
fromcollectionsimportnamedtuple
importgc

importdiffusers
importtorch
importmatplotlib.pyplotasplt
importgradioasgr
importnumpyasnp

importopenvinoasov

..code::ipython3

MODELS_DIR=Path("models")
PRIOR_TEXT_ENCODER_PATH=MODELS_DIR/"prior_text_encoder.xml"
PRIOR_PRIOR_PATH=MODELS_DIR/"prior_prior.xml"
DECODER_PATH=MODELS_DIR/"decoder.xml"
TEXT_ENCODER_PATH=MODELS_DIR/"text_encoder.xml"
VQGAN_PATH=MODELS_DIR/"vqgan.xml"

MODELS_DIR.mkdir(parents=True,exist_ok=True)

..code::ipython3

BaseModelOutputWithPooling=namedtuple("BaseModelOutputWithPooling","last_hidden_state")
DecoderOutput=namedtuple("DecoderOutput","sample")

Loadtheoriginalmodel
-----------------------

`backtotop⬆️<#table-of-contents>`__

Weuse``from_pretrained``methodof
``diffusers.AutoPipelineForText2Image``toloadthepipeline.

..code::ipython3

pipeline=diffusers.AutoPipelineForText2Image.from_pretrained("warp-diffusion/wuerstchen")

Loadedmodelhas``WuerstchenCombinedPipeline``typeandconsistsof2
parts:prioranddecoder.

Infertheoriginalmodel
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

caption="Anthropomorphiccatdressedasafirefighter"
negative_prompt=""
generator=torch.Generator().manual_seed(1)
output=pipeline(
prompt=caption,
height=1024,
width=1024,
negative_prompt=negative_prompt,
prior_guidance_scale=4.0,
decoder_guidance_scale=0.0,
output_type="pil",
generator=generator,
).images

..code::ipython3

plt.figure(figsize=(8*len(output),8),dpi=128)
fori,xinenumerate(output):
plt.subplot(1,len(output),i+1)
plt.imshow(x)
plt.axis("off")



..image::wuerstchen-image-generation-with-output_files/wuerstchen-image-generation-with-output_11_0.png


ConvertthemodeltoOpenVINOIR
--------------------------------

`backtotop⬆️<#table-of-contents>`__

Mainmodelcomponents:-Priorstage:createlow-dimensionallatent
spacerepresentationoftheimageusingtext-conditionalLDM-Decoder
stage:usingrepresentationfromPriorStage,producealatentimagein
latentspaceofhigherdimensionalityusinganotherLDMandusing
VQGAN-decoder,decodethelatentimagetoyieldafull-resolutionoutput
image

Thepipelineconsistsof2sub-pipelines:Priorpipelineaccessedby
``prior_pipe``property,andDecoderPipelineaccessedby
``decoder_pipe``property.

..code::ipython3

#Priorpipeline
pipeline.prior_text_encoder.eval()
pipeline.prior_prior.eval()

#Decoderpipeline
pipeline.decoder.eval()
pipeline.text_encoder.eval()
pipeline.vqgan.eval();

Next,let’sdefinetheconversionfunctionforPyTorchmodules.Weuse
``ov.convert_model``functiontoobtainOpenVINOIntermediate
Representationobjectand``ov.save_model``functiontosaveitasXML
file.

..code::ipython3

defconvert(model:torch.nn.Module,xml_path:Path,**convert_kwargs):
ifnotxml_path.exists():
converted_model=ov.convert_model(model,**convert_kwargs)
ov.save_model(converted_model,xml_path,compress_to_fp16=False)
delconverted_model

#Cleantorchjitcache
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()

gc.collect()

Priorpipeline
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thispipelineconsistsoftextencoderandpriordiffusionmodel.From
here,wealwaysusefixedshapesinconversionbyusingan``input``
parametertogeneratealessmemory-demandingmodel.

Textencodermodelhas2inputs:-``input_ids``:vectoroftokenized
inputsentence.Defaulttokenizervectorlengthis77.-
``attention_mask``:vectorofsamelengthas``input_ids``describing
theattentionmask.

..code::ipython3

convert(
pipeline.prior_text_encoder,
PRIOR_TEXT_ENCODER_PATH,
example_input={
"input_ids":torch.zeros(1,77,dtype=torch.int32),
"attention_mask":torch.zeros(1,77),
},
input={"input_ids":((1,77),),"attention_mask":((1,77),)},
)
delpipeline.prior_text_encoder
delpipeline.prior_pipe.text_encoder
gc.collect()




..parsed-literal::

2058



PriormodelisthecanonicalunCLIPpriortoapproximatetheimage
embeddingfromthetextembedding.LikeUNet,ithas3inputs:sample,
timestepandencoderhiddenstates.

..code::ipython3

convert(
pipeline.prior_prior,
PRIOR_PRIOR_PATH,
example_input=[
torch.zeros(2,16,24,24),
torch.zeros(2),
torch.zeros(2,77,1280),
],
input=[((2,16,24,24),),((2),),((2,77,1280),)],
)
delpipeline.prior_prior
delpipeline.prior_pipe.prior
gc.collect()




..parsed-literal::

0



Decoderpipeline
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Decoderpipelineconsistsof3parts:decoder,textencoderandVQGAN.

DecodermodelistheWuerstchenDiffNeXtUNetdecoder.Inputsare:-
``x``:sample-``r``:timestep-``effnet``:interpolationblock-
``clip``:encoderhiddenstates

..code::ipython3

convert(
pipeline.decoder,
DECODER_PATH,
example_input={
"x":torch.zeros(1,4,256,256),
"r":torch.zeros(1),
"effnet":torch.zeros(1,16,24,24),
"clip":torch.zeros(1,77,1024),
},
input={
"x":((1,4,256,256),),
"r":((1),),
"effnet":((1,16,24,24),),
"clip":((1,77,1024),),
},
)
delpipeline.decoder
delpipeline.decoder_pipe.decoder
gc.collect()




..parsed-literal::

0



Themaintextencoderhasthesameinputparametersandshapesastext
encoderin`priorpipeline<#prior-pipeline>`__.

..code::ipython3

convert(
pipeline.text_encoder,
TEXT_ENCODER_PATH,
example_input={
"input_ids":torch.zeros(1,77,dtype=torch.int32),
"attention_mask":torch.zeros(1,77),
},
input={"input_ids":((1,77),),"attention_mask":((1,77),)},
)
delpipeline.text_encoder
delpipeline.decoder_pipe.text_encoder
gc.collect()




..parsed-literal::

0



PipelineusesVQGANmodel``decode``methodtogetthefull-sizeoutput
image.Herewecreatethewrappermodulefordecodingpartonly.Our
decodertakesasinput4x256x256latentimage.

..code::ipython3

classVqganDecoderWrapper(torch.nn.Module):
def__init__(self,vqgan):
super().__init__()
self.vqgan=vqgan

defforward(self,h):
returnself.vqgan.decode(h)

..code::ipython3

convert(
VqganDecoderWrapper(pipeline.vqgan),
VQGAN_PATH,
example_input=torch.zeros(1,4,256,256),
input=(1,4,256,256),
)
delpipeline.decoder_pipe.vqgan
gc.collect()




..parsed-literal::

0



Compilingmodels
----------------

`backtotop⬆️<#table-of-contents>`__

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

Dropdown(description='Device:',index=4,options=('CPU','GPU.0','GPU.1','GPU.2','AUTO'),value='AUTO')



..code::ipython3

ov_prior_text_encoder=core.compile_model(PRIOR_TEXT_ENCODER_PATH,device.value)

..code::ipython3

ov_prior_prior=core.compile_model(PRIOR_PRIOR_PATH,device.value)

..code::ipython3

ov_decoder=core.compile_model(DECODER_PATH,device.value)

..code::ipython3

ov_text_encoder=core.compile_model(TEXT_ENCODER_PATH,device.value)

..code::ipython3

ov_vqgan=core.compile_model(VQGAN_PATH,device.value)

Buildingthepipeline
---------------------

`backtotop⬆️<#table-of-contents>`__

Let’screatecallablewrapperclassesforcompiledmodelstoallow
interactionwithoriginal``WuerstchenCombinedPipeline``class.Note
thatallofwrapperclassesreturn``torch.Tensor``\sinsteadof
``np.array``\s.

..code::ipython3

classTextEncoderWrapper:
dtype=torch.float32#accessedintheoriginalworkflow

def__init__(self,text_encoder):
self.text_encoder=text_encoder

def__call__(self,input_ids,attention_mask):
output=self.text_encoder({"input_ids":input_ids,"attention_mask":attention_mask})["last_hidden_state"]
output=torch.tensor(output)
returnBaseModelOutputWithPooling(output)

..code::ipython3

classPriorPriorWrapper:
config=namedtuple("PriorPriorWrapperConfig","c_in")(16)#accessedintheoriginalworkflow

def__init__(self,prior):
self.prior=prior

def__call__(self,x,r,c):
output=self.prior([x,r,c])[0]
returntorch.tensor(output)

..code::ipython3

classDecoderWrapper:
dtype=torch.float32#accessedintheoriginalworkflow

def__init__(self,decoder):
self.decoder=decoder

def__call__(self,x,r,effnet,clip):
output=self.decoder({"x":x,"r":r,"effnet":effnet,"clip":clip})[0]
output=torch.tensor(output)
returnoutput

..code::ipython3

classVqganWrapper:
config=namedtuple("VqganWrapperConfig","scale_factor")(0.3764)#accessedintheoriginalworkflow

def__init__(self,vqgan):
self.vqgan=vqgan

defdecode(self,h):
output=self.vqgan(h)[0]
output=torch.tensor(output)
returnDecoderOutput(output)

Andinsertwrappersinstancesinthepipeline:

..code::ipython3

pipeline.prior_pipe.text_encoder=TextEncoderWrapper(ov_prior_text_encoder)
pipeline.prior_pipe.prior=PriorPriorWrapper(ov_prior_prior)

pipeline.decoder_pipe.decoder=DecoderWrapper(ov_decoder)
pipeline.decoder_pipe.text_encoder=TextEncoderWrapper(ov_text_encoder)
pipeline.decoder_pipe.vqgan=VqganWrapper(ov_vqgan)

Inference
---------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

caption="Anthropomorphiccatdressedasafirefighter"
negative_prompt=""
generator=torch.Generator().manual_seed(1)

output=pipeline(
prompt=caption,
height=1024,
width=1024,
negative_prompt=negative_prompt,
prior_guidance_scale=4.0,
decoder_guidance_scale=0.0,
output_type="pil",
generator=generator,
).images

..code::ipython3

plt.figure(figsize=(8*len(output),8),dpi=128)
fori,xinenumerate(output):
plt.subplot(1,len(output),i+1)
plt.imshow(x)
plt.axis("off")



..image::wuerstchen-image-generation-with-output_files/wuerstchen-image-generation-with-output_45_0.png


Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``WuerstchenPriorPipeline``structure,priormodelisused
inthecyclerepeatinginferenceoneachdiffusionstep,whiletext
encodertakespartonlyonce,andinthe``WuerstchenDecoderPipeline``,
thedecodermodelisusedinaloop,andotherpipelinecomponentsare
inferredonlyonce.Thatiswhycomputationcostandspeedofpriorand
decodermodelsbecomethecriticalpathinthepipeline.Quantizingthe
restofthepipelinedoesnotsignificantlyimproveinference
performancebutcanleadtoasubstantialdegradationofaccuracy.

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

Let’sload``skipmagic``extensiontoskipquantizationif
``to_quantize``isnotselected

..code::ipython3

#Fetch`skip_kernel_extension`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

int8_pipeline=None

%load_extskip_kernel_extension

Preparecalibrationdatasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseaportionof
`conceptual_captions<https://huggingface.co/datasets/google-research-datasets/conceptual_captions>`__
datasetfromHuggingFaceascalibrationdata.Tocollectintermediate
modelinputsforcalibrationweshouldcustomize``CompiledModel``.

..code::ipython3

%%skipnot$to_quantize.value

classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model):
super().__init__(compiled_model)
self.data_cache=[]

def__call__(self,*args,**kwargs):
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)

..code::ipython3

%%skipnot$to_quantize.value

importdatasets
fromtqdm.notebookimporttqdm
fromtransformersimportset_seed

set_seed(1)

defcollect_calibration_data(pipeline,subset_size):
pipeline.set_progress_bar_config(disable=True)

original_prior=pipeline.prior_pipe.prior.prior
original_decoder=pipeline.decoder_pipe.decoder.decoder
pipeline.prior_pipe.prior.prior=CompiledModelDecorator(original_prior)
pipeline.decoder_pipe.decoder.decoder=CompiledModelDecorator(original_decoder)

dataset=datasets.load_dataset("google-research-datasets/conceptual_captions",split="train",trust_remote_code=True).shuffle(seed=42)
pbar=tqdm(total=subset_size)
diff=0
forbatchindataset:
prompt=batch["caption"]
iflen(prompt)>pipeline.tokenizer.model_max_length:
continue
_=pipeline(
prompt=prompt,
height=1024,
width=1024,
negative_prompt="",
prior_guidance_scale=4.0,
decoder_guidance_scale=0.0,
output_type="pil",
)
collected_subset_size=len(pipeline.prior_pipe.prior.prior.data_cache)
ifcollected_subset_size>=subset_size:
pbar.update(subset_size-pbar.n)
break
pbar.update(collected_subset_size-diff)
diff=collected_subset_size

prior_calibration_dataset=pipeline.prior_pipe.prior.prior.data_cache
decoder_calibration_dataset=pipeline.decoder_pipe.decoder.decoder.data_cache
pipeline.prior_pipe.prior.prior=original_prior
pipeline.decoder_pipe.decoder.decoder=original_decoder
pipeline.set_progress_bar_config(disable=False)
returnprior_calibration_dataset,decoder_calibration_dataset

..code::ipython3

%%skipnot$to_quantize.value

PRIOR_PRIOR_INT8_PATH=MODELS_DIR/"prior_prior_int8.xml"
DECODER_INT8_PATH=MODELS_DIR/"decoder_int8.xml"

ifnot(PRIOR_PRIOR_INT8_PATH.exists()andDECODER_INT8_PATH.exists()):
subset_size=300
prior_calibration_dataset,decoder_calibration_dataset=collect_calibration_data(pipeline,subset_size=subset_size)

Runquantization
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Createaquantizedmodelfromthepre-trainedconvertedOpenVINOmodel.
``BiasCorrection``algorithmisdisabledduetominimalaccuracy
improvementinWürstchenmodelandincreasedquantizationtime.The
prioranddecodermodelsaretransformer-basedbackbonenetworks,weuse
``model_type=nncf.ModelType.TRANSFORMER``tospecifyadditional
transformerpatternsinthemodel.ItpreservesaccuracyafterNNCFPTQ
byretainingseveralaccuracy-sensitivelayersinFP16precision.

Thequantizationofthefirstandlast``Convolution``layersinthe
priormodeldramaticallyimpactsthegenerationresultsaccordingtoour
experiments.Werecommendusing``IgnoredScope``tokeeptheminFP16
precision.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

%%skipnot$to_quantize.value

importnncf
fromnncf.scopesimportIgnoredScope

ifnotPRIOR_PRIOR_INT8_PATH.exists():
prior_model=core.read_model(PRIOR_PRIOR_PATH)
quantized_prior_prior=nncf.quantize(
model=prior_model,
subset_size=subset_size,
calibration_dataset=nncf.Dataset(prior_calibration_dataset),
model_type=nncf.ModelType.TRANSFORMER,
ignored_scope=IgnoredScope(names=[
"__module.projection/aten::_convolution/Convolution",
"__module.out.1/aten::_convolution/Convolution"
]),
advanced_parameters=nncf.AdvancedQuantizationParameters(
disable_bias_correction=True
)
)
ov.save_model(quantized_prior_prior,PRIOR_PRIOR_INT8_PATH)

..code::ipython3

%%skipnot$to_quantize.value

ifnotDECODER_INT8_PATH.exists():
decoder_model=core.read_model(DECODER_PATH)
quantized_decoder=nncf.quantize(
model=decoder_model,
calibration_dataset=nncf.Dataset(decoder_calibration_dataset),
subset_size=len(decoder_calibration_dataset),
model_type=nncf.ModelType.TRANSFORMER,
advanced_parameters=nncf.AdvancedQuantizationParameters(
disable_bias_correction=True
)
)
ov.save_model(quantized_decoder,DECODER_INT8_PATH)

Let’scomparetheimagesgeneratedbytheoriginalandoptimized
pipelines.

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

..code::ipython3

%%skipnot$to_quantize.value

caption="Anthropomorphiccatdressedasafirefighter"
negative_prompt=""

int8_pipeline=diffusers.AutoPipelineForText2Image.from_pretrained("warp-diffusion/wuerstchen")

int8_prior_prior=core.compile_model(PRIOR_PRIOR_INT8_PATH)
int8_pipeline.prior_pipe.prior=PriorPriorWrapper(int8_prior_prior)

int8_decoder=core.compile_model(DECODER_INT8_PATH)
int8_pipeline.decoder_pipe.decoder=DecoderWrapper(int8_decoder)

int8_pipeline.prior_pipe.text_encoder=TextEncoderWrapper(ov_prior_text_encoder)
int8_pipeline.decoder_pipe.text_encoder=TextEncoderWrapper(ov_text_encoder)
int8_pipeline.decoder_pipe.vqgan=VqganWrapper(ov_vqgan)

..code::ipython3

%%skipnot$to_quantize.value

generator=torch.Generator().manual_seed(1)
int8_output=int8_pipeline(
prompt=caption,
height=1024,
width=1024,
negative_prompt=negative_prompt,
prior_guidance_scale=4.0,
decoder_guidance_scale=0.0,
output_type="pil",
generator=generator,
).images

..code::ipython3

%%skipnot$to_quantize.value

fig=visualize_results(output[0],int8_output[0])



..image::wuerstchen-image-generation-with-output_files/wuerstchen-image-generation-with-output_61_0.png


Comparemodelfilesizes
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fp16_ir_model_size=PRIOR_PRIOR_PATH.with_suffix(".bin").stat().st_size/2**20
quantized_model_size=PRIOR_PRIOR_INT8_PATH.with_suffix(".bin").stat().st_size/2**20

print(f"FP16Priorsize:{fp16_ir_model_size:.2f}MB")
print(f"INT8Priorsize:{quantized_model_size:.2f}MB")
print(f"Priorcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16Priorsize:3790.42MB
INT8Priorsize:951.03MB
Priorcompressionrate:3.986


..code::ipython3

%%skipnot$to_quantize.value

fp16_ir_model_size=DECODER_PATH.with_suffix(".bin").stat().st_size/2**20
quantized_model_size=DECODER_INT8_PATH.with_suffix(".bin").stat().st_size/2**20

print(f"FP16Decodersize:{fp16_ir_model_size:.2f}MB")
print(f"INT8Decodersize:{quantized_model_size:.2f}MB")
print(f"Decodercompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16Decodersize:4025.90MB
INT8Decodersize:1010.20MB
Decodercompressionrate:3.985


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
caption="Anthropomorphiccatdressedasafirefighter"
foriinrange(3):
start=time.perf_counter()
_=pipeline(
prompt=caption,
height=1024,
width=1024,
prior_guidance_scale=4.0,
decoder_guidance_scale=0.0,
output_type="pil",
)
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
pipeline.set_progress_bar_config(disable=False)
returnnp.mean(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

fp_latency=calculate_inference_time(pipeline)
print(f"FP16pipeline:{fp_latency:.3f}seconds")
int8_latency=calculate_inference_time(int8_pipeline)
print(f"INT8pipeline:{int8_latency:.3f}seconds")
print(f"Performancespeedup:{fp_latency/int8_latency:.3f}")


..parsed-literal::

FP16pipeline:199.484seconds
INT8pipeline:78.734seconds
Performancespeedup:2.534


Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

Pleaseselectbelowwhetheryouwouldliketousethequantizedmodelto
launchtheinteractivedemo.

..code::ipython3

quantized_model_present=int8_pipelineisnotNone

use_quantized_model=widgets.Checkbox(
value=quantized_model_present,
description="Usequantizedmodel",
disabled=notquantized_model_present,
)

use_quantized_model

..code::ipython3

pipe=int8_pipelineifuse_quantized_model.valueelsepipeline


defgenerate(caption,negative_prompt,prior_guidance_scale,seed):
generator=torch.Generator().manual_seed(seed)
image=pipe(
prompt=caption,
height=1024,
width=1024,
negative_prompt=negative_prompt,
prior_num_inference_steps=30,
prior_guidance_scale=prior_guidance_scale,
generator=generator,
output_type="pil",
).images[0]
returnimage

..code::ipython3

demo=gr.Interface(
generate,
[
gr.Textbox(label="Caption"),
gr.Textbox(label="Negativeprompt"),
gr.Slider(2,20,step=1,label="Priorguidancescale"),
gr.Slider(0,np.iinfo(np.int32).max,label="Seed"),
],
"image",
examples=[["Anthropomorphiccatdressedasafirefighter","",4,0]],
allow_flagging="never",
)
try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(debug=False,share=True)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
