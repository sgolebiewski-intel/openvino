SinglestepimagegenerationusingSDXL-turboandOpenVINO
==========================================================

SDXL-Turboisafastgenerativetext-to-imagemodelthatcansynthesize
photorealisticimagesfromatextpromptinasinglenetworkevaluation.
SDXL-Turboisadistilledversionof`SDXL
1.0<https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`__,
trainedforreal-timesynthesis.SDXLTurboisbasedonanovel
distillationtechniquecalledAdversarialDiffusionDistillation(ADD),
whichenablesthemodeltosynthesizeimageoutputsinasinglestepand
generatereal-timetext-to-imageoutputswhilemaintaininghighsampling
fidelity.Moredetailsaboutthisdistillationapproachcanbefoundin
`technical
report<https://stability.ai/research/adversarial-diffusion-distillation>`__.
Moredetailsaboutmodelcanbefoundin`StabilityAIblog
post<https://stability.ai/news/stability-ai-sdxl-turbo>`__.

Previously,wealreadydiscussedhowtolaunchStableDiffusionXLmodel
usingOpenVINOinthefollowing`notebook<../stable-diffusion-xl>`__,
inthistutorialwewillfocusonthe
`SDXL-turbo<https://huggingface.co/stabilityai/sdxl-turbo>`__version.
Additionally,toimproveimagedecodingspeed,wewilluse`Tiny
Autoencoder<https://github.com/madebyollin/taesd>`__,whichisuseful
forreal-timepreviewingoftheSDXLgenerationprocess.

Wewilluseapre-trainedmodelfromthe`HuggingFace
Diffusers<https://huggingface.co/docs/diffusers/index>`__library.To
simplifytheuserexperience,the`HuggingFaceOptimum
Intel<https://huggingface.co/docs/optimum/intel/index>`__libraryis
usedtoconvertthemodelstoOpenVINO™IRformat.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`ConvertmodeltoOpenVINO
format<#convert-model-to-openvino-format>`__
-`Text-to-imagegeneration<#text-to-image-generation>`__

-`Selectinferencedevicefortext-to-image
generation<#select-inference-device-for-text-to-image-generation>`__

-`Image-to-Imagegeneration<#image-to-image-generation>`__
-`Quantization<#quantization>`__

-`Preparecalibrationdataset<#prepare-calibration-dataset>`__
-`Runquantization<#run-quantization>`__

-`CompareUNetfilesize<#compare-unet-file-size>`__

-`CompareinferencetimeoftheFP16andINT8
models<#compare-inference-time-of-the-fp16-and-int8-models>`__

-`InteractiveDemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu\
"torch>=2.1"transformersdiffusers"git+https://github.com/huggingface/optimum-intel.git""gradio>=4.19""peft==0.6.2""openvino>=2023.3.0"

ConvertmodeltoOpenVINOformat
--------------------------------

`backtotop⬆️<#table-of-contents>`__

`sdxl-turbo<https://huggingface.co/stabilityai/sdxl-turbo>`__is
availablefordownloadingviathe`HuggingFace
hub<https://huggingface.co/models>`__.Wewilluseoptimum-cli
interfaceforexportingitintoOpenVINOIntermediateRepresentation
(IR)format.

OptimumCLIinterfaceforconvertingmodelssupportsexporttoOpenVINO
(supportedstartingoptimum-intel1.12version).Generalcommandformat:

..code::bash

optimum-cliexportopenvino--model<model_id_or_path>--task<task><output_dir>

wheretaskistasktoexportthemodelfor,ifnotspecified,thetask
willbeauto-inferredbasedonthemodel.Availabletasksdependonthe
model,forsdxlshouldbeselected``stable-diffusion-xl``

YoucanfindamappingbetweentasksandmodelclassesinOptimum
TaskManager
`documentation<https://huggingface.co/docs/optimum/exporters/task_manager>`__.

Additionally,youcanspecifyweightscompression``--fp16``forthe
compressionmodeltoFP16and``--int8``forthecompressionmodelto
INT8.Pleasenote,thatforINT8,itisnecessarytoinstallnncf.

Fulllistofsupportedargumentsavailablevia``--help``Formore
detailsandexamplesofusage,pleasecheck`optimum
documentation<https://huggingface.co/docs/optimum/intel/inference#export>`__.

ForTinyAutoencoder,wewilluse``ov.convert_model``functionfor
obtaining``ov.Model``andsaveitusing``ov.save_model``.Model
consistsof2partsthatusedinpipelineseparately:``vae_encoder``
forencodinginputimageinlatentspaceinimage-to-imagegeneration
taskand``vae_decoder``thatresponsiblefordecodingdiffusionresult
backtoimageformat.

..code::ipython3

frompathlibimportPath

model_dir=Path("./model")
sdxl_model_id="stabilityai/sdxl-turbo"
tae_id="madebyollin/taesdxl"
skip_convert_model=model_dir.exists()

..code::ipython3

importtorch
importopenvinoasov
fromdiffusersimportAutoencoderTiny
importgc


classVAEEncoder(torch.nn.Module):
def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,sample):
returnself.vae.encode(sample)


classVAEDecoder(torch.nn.Module):
def__init__(self,vae):
super().__init__()
self.vae=vae

defforward(self,latent_sample):
returnself.vae.decode(latent_sample)


defconvert_tiny_vae(model_id,output_path):
tiny_vae=AutoencoderTiny.from_pretrained(model_id)
tiny_vae.eval()
vae_encoder=VAEEncoder(tiny_vae)
ov_model=ov.convert_model(vae_encoder,example_input=torch.zeros((1,3,512,512)))
ov.save_model(ov_model,output_path/"vae_encoder/openvino_model.xml")
tiny_vae.save_config(output_path/"vae_encoder")
vae_decoder=VAEDecoder(tiny_vae)
ov_model=ov.convert_model(vae_decoder,example_input=torch.zeros((1,4,64,64)))
ov.save_model(ov_model,output_path/"vae_decoder/openvino_model.xml")
tiny_vae.save_config(output_path/"vae_decoder")


ifnotskip_convert_model:
!optimum-cliexportopenvino--model$sdxl_model_id--taskstable-diffusion-xl$model_dir--fp16
convert_tiny_vae(tae_id,model_dir)

Text-to-imagegeneration
------------------------

`backtotop⬆️<#table-of-contents>`__

Text-to-imagegenerationletsyoucreateimagesusingtextdescription.
Tostartgeneratingimages,weneedtoloadmodelsfirst.Toloadan
OpenVINOmodelandrunaninferencewithOptimumandOpenVINORuntime,
youneedtoreplacediffusers``StableDiffusionXLPipeline``withOptimum
``OVStableDiffusionXLPipeline``.Pipelineinitializationstartswith
using``from_pretrained``method,whereadirectorywithOpenVINOmodels
shouldbepassed.Additionally,youcanspecifyaninferencedevice.

Selectinferencedevicefortext-to-imagegeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

fromoptimum.intel.openvinoimportOVStableDiffusionXLPipeline

text2image_pipe=OVStableDiffusionXLPipeline.from_pretrained(model_dir,device=device.value)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,onnx,openvino


..parsed-literal::

/home/ea/work/genai_env/lib/python3.8/site-packages/torch/cuda/__init__.py:138:UserWarning:CUDAinitialization:TheNVIDIAdriveronyoursystemistooold(foundversion11080).PleaseupdateyourGPUdriverbydownloadingandinstallinganewversionfromtheURL:http://www.nvidia.com/Download/index.aspxAlternatively,goto:https://pytorch.orgtoinstallaPyTorchversionthathasbeencompiledwithyourversionoftheCUDAdriver.(Triggeredinternallyat../c10/cuda/CUDAFunctions.cpp:108.)
returntorch._C._cuda_getDeviceCount()>0
NoCUDAruntimeisfound,usingCUDA_HOME='/usr/local/cuda'
Compilingthevae_decodertoAUTO...
CompilingtheunettoAUTO...
Compilingthetext_encodertoAUTO...
Compilingthetext_encoder_2toAUTO...
Compilingthevae_encodertoAUTO...


Thepipelineinterfaceissimilartooriginal
``StableDiffusionXLPipeline``.Weshouldprovidetextprompt.The
defaultnumberofstepsis50,whilesdxl-turborequiredonly1step.
Accordingtotheinformationprovidedinmodelcard,modeldoesnotuse
negativepromptandguidancescaleandthisparametersshouldbe
disabledusing``guidance_scale=0``

..code::ipython3

importnumpyasnp

prompt="cutecat"
image=text2image_pipe(
prompt,
num_inference_steps=1,
height=512,
width=512,
guidance_scale=0.0,
generator=np.random.RandomState(987),
).images[0]
image.save("cat.png")
image



..parsed-literal::

0%||0/1[00:00<?,?it/s]




..image::sdxl-turbo-with-output_files/sdxl-turbo-with-output_11_1.png



..code::ipython3

deltext2image_pipe
gc.collect();

Image-to-Imagegeneration
-------------------------

`backtotop⬆️<#table-of-contents>`__

Image-to-imagegenerationletsyoutransformimagestomatchthe
characteristicsprovidedinthetextdescription.Wecanreusethe
alreadyconvertedmodelforrunningtheImage2Imagegenerationpipeline.
Forthat,weshouldreplace``OVStableDiffusionXLPipeline``with
``OVStableDiffusionXLImage2ImagePipeline``.

..code::ipython3

fromoptimum.intelimportOVStableDiffusionXLImg2ImgPipeline

image2image_pipe=OVStableDiffusionXLImg2ImgPipeline.from_pretrained(model_dir,device=device.value)


..parsed-literal::

Compilingthevae_decodertoAUTO...
CompilingtheunettoAUTO...
Compilingthetext_encoder_2toAUTO...
Compilingthevae_encodertoAUTO...
Compilingthetext_encodertoAUTO...


..code::ipython3

photo_prompt="acutecatwithbowtie"

``strength``parameterisimportantfortheimage-to-imagegeneration
pipeline.Itisavaluebetween0.0and1.0,thatcontrolstheamountof
noisethatisaddedtotheinputimage.Valuesthatapproach1.0enable
lotsofvariationsbutwillalsoproduceimagesthatarenot
semanticallyconsistentwiththeinput,thencloseto0,lessnoisewill
beaddedandthetargetimagewillpreservesourceimagecontent.
strengthhasanimpactnotonlyonanumberofnoisebutalsothenumber
ofgenerationsteps.Thenumberofdenoisingiterationsinthe
image-to-imagegenerationpipelineiscalculatedas
``int(num_inference_steps*strength)``.Withsdxl-turboweshouldbe
carefulwithselecting``num_inference_steps``and``strength``to
producethecorrectresultandmakesurethatthenumberofstepsused
inpipeline>=1afterapplyingstrengthmultiplication.e.g. inexample
below,wewilluse``num_inference_steps=2``and``stength=0.5``,
finally,weget0.5\*2.0=1stepinourpipeline.

..code::ipython3

photo_image=image2image_pipe(
photo_prompt,
image=image,
num_inference_steps=2,
generator=np.random.RandomState(511),
guidance_scale=0.0,
strength=0.5,
).images[0]
photo_image.save("cat_tie.png")
photo_image



..parsed-literal::

0%||0/1[00:00<?,?it/s]




..image::sdxl-turbo-with-output_files/sdxl-turbo-with-output_17_1.png



..code::ipython3

delimage2image_pipe
gc.collect();

Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Accordingto``SDXL-TurboModel``structure,theUNetmodeltakesup
significantportionoftheoverallpipelineexecutiontime.Nowwewill
showyouhowtooptimizetheUNetpartusing
`NNCF<https://github.com/openvinotoolkit/nncf/>`__toreduce
computationcostandspeedupthepipeline.Quantizingtherestofthe
SDXLpipelinedoesnotsignificantlyimproveinferenceperformancebut
canleadtoasubstantialdegradationofaccuracy.

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



..code::ipython3

#Fetch`skip_kernel_extension`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

int8_pipe=None

%load_extskip_kernel_extension

Preparecalibrationdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseaportionof
`conceptual_captions<https://huggingface.co/datasets/conceptual_captions>`__
datasetfromHuggingFaceascalibrationdata.Tocollectintermediate
modelinputsforcalibrationweshouldcustomize``CompiledModel``.

..code::ipython3

UNET_INT8_OV_PATH=model_dir/"optimized_unet"/"openvino_model.xml"


defdisable_progress_bar(pipeline,disable=True):
ifnothasattr(pipeline,"_progress_bar_config"):
pipeline._progress_bar_config={"disable":disable}
else:
pipeline._progress_bar_config["disable"]=disable

..code::ipython3

%%skipnot$to_quantize.value

importdatasets
importnumpyasnp
fromtqdm.notebookimporttqdm
fromtransformersimportset_seed
fromtypingimportAny,Dict,List

set_seed(1)

classCompiledModelDecorator(ov.CompiledModel):
def__init__(self,compiled_model:ov.CompiledModel,data_cache:List[Any]=None):
super().__init__(compiled_model)
self.data_cache=data_cacheifdata_cacheelse[]

def__call__(self,*args,**kwargs):
self.data_cache.append(*args)
returnsuper().__call__(*args,**kwargs)

defcollect_calibration_data(pipe,subset_size:int)->List[Dict]:
original_unet=pipe.unet.request
pipe.unet.request=CompiledModelDecorator(original_unet)

dataset=datasets.load_dataset("google-research-datasets/conceptual_captions",split="train",trust_remote_code=True).shuffle(seed=42)
disable_progress_bar(pipe)

#Runinferencefordatacollection
pbar=tqdm(total=subset_size)
diff=0
forbatchindataset:
prompt=batch["caption"]
iflen(prompt)>pipe.tokenizer.model_max_length:
continue
_=pipe(
prompt,
num_inference_steps=1,
height=512,
width=512,
guidance_scale=0.0,
generator=np.random.RandomState(987)
)
collected_subset_size=len(pipe.unet.request.data_cache)
ifcollected_subset_size>=subset_size:
pbar.update(subset_size-pbar.n)
break
pbar.update(collected_subset_size-diff)
diff=collected_subset_size

calibration_dataset=pipe.unet.request.data_cache
disable_progress_bar(pipe,disable=False)
pipe.unet.request=original_unet
returncalibration_dataset

..code::ipython3

%%skipnot$to_quantize.value

ifnotUNET_INT8_OV_PATH.exists():
text2image_pipe=OVStableDiffusionXLPipeline.from_pretrained(model_dir,device=device.value)
unet_calibration_data=collect_calibration_data(text2image_pipe,subset_size=200)

Runquantization
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Createaquantizedmodelfromthepre-trainedconvertedOpenVINOmodel.
Quantizationofthefirstandlast``Convolution``layersimpactsthe
generationresults.Werecommendusing``IgnoredScope``tokeepaccuracy
sensitive``Convolution``layersinFP16precision.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

..code::ipython3

%%skipnot$to_quantize.value

importnncf
fromnncf.scopesimportIgnoredScope

UNET_OV_PATH=model_dir/"unet"/"openvino_model.xml"
ifnotUNET_INT8_OV_PATH.exists():
unet=core.read_model(UNET_OV_PATH)
quantized_unet=nncf.quantize(
model=unet,
model_type=nncf.ModelType.TRANSFORMER,
calibration_dataset=nncf.Dataset(unet_calibration_data),
ignored_scope=IgnoredScope(
names=[
"__module.model.conv_in/aten::_convolution/Convolution",
"__module.model.up_blocks.2.resnets.2.conv_shortcut/aten::_convolution/Convolution",
"__module.model.conv_out/aten::_convolution/Convolution"
],
),
)
ov.save_model(quantized_unet,UNET_INT8_OV_PATH)

LetuscheckpredictionswiththequantizedUNetusingthesameinput
data.

..code::ipython3

%%skipnot$to_quantize.value

fromIPython.displayimportdisplay

int8_text2image_pipe=OVStableDiffusionXLPipeline.from_pretrained(model_dir,device=device.value,compile=False)
int8_text2image_pipe.unet.model=core.read_model(UNET_INT8_OV_PATH)
int8_text2image_pipe.unet.request=None

prompt="cutecat"
image=int8_text2image_pipe(prompt,num_inference_steps=1,height=512,width=512,guidance_scale=0.0,generator=np.random.RandomState(987)).images[0]
display(image)


..parsed-literal::

Compilingthetext_encodertoAUTO...
Compilingthetext_encoder_2toAUTO...



..parsed-literal::

0%||0/1[00:00<?,?it/s]


..parsed-literal::

CompilingtheunettoAUTO...
Compilingthevae_decodertoAUTO...



..image::sdxl-turbo-with-output_files/sdxl-turbo-with-output_29_3.png


..code::ipython3

%%skipnot$to_quantize.value

int8_image2image_pipe=OVStableDiffusionXLImg2ImgPipeline.from_pretrained(model_dir,device=device.value,compile=False)
int8_image2image_pipe.unet.model=core.read_model(UNET_INT8_OV_PATH)
int8_image2image_pipe.unet.request=None

photo_prompt="acutecatwithbowtie"
photo_image=int8_image2image_pipe(photo_prompt,image=image,num_inference_steps=2,generator=np.random.RandomState(511),guidance_scale=0.0,strength=0.5).images[0]
display(photo_image)


..parsed-literal::

Compilingthetext_encodertoAUTO...
Compilingthetext_encoder_2toAUTO...
Compilingthevae_encodertoAUTO...



..parsed-literal::

0%||0/1[00:00<?,?it/s]


..parsed-literal::

CompilingtheunettoAUTO...
Compilingthevae_decodertoAUTO...



..image::sdxl-turbo-with-output_files/sdxl-turbo-with-output_30_3.png


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

FP16modelsize:5014578.27KB
INT8modelsize:2513541.44KB
Modelcompressionrate:1.995


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

validation_size=7
calibration_dataset=datasets.load_dataset("google-research-datasets/conceptual_captions",split="train",trust_remote_code=True)
validation_data=[]
forbatchincalibration_dataset:
prompt=batch["caption"]
validation_data.append(prompt)

defcalculate_inference_time(pipe,dataset):
inference_time=[]
disable_progress_bar(pipe)

foridx,promptinenumerate(dataset):
start=time.perf_counter()
image=pipe(
prompt,
num_inference_steps=1,
guidance_scale=0.0,
generator=np.random.RandomState(23)
).images[0]
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
ifidx>=validation_size:
break
disable_progress_bar(pipe,disable=False)
returnnp.median(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

int8_latency=calculate_inference_time(int8_text2image_pipe,validation_data)
text2image_pipe=OVStableDiffusionXLPipeline.from_pretrained(model_dir,device=device.value)
fp_latency=calculate_inference_time(text2image_pipe,validation_data)
print(f"FP16pipelinelatency:{fp_latency:.3f}")
print(f"INT8pipelinelatency:{int8_latency:.3f}")
print(f"Text-to-Imagegenerationspeedup:{fp_latency/int8_latency:.3f}")


..parsed-literal::

Compilingthevae_decodertoAUTO...
CompilingtheunettoAUTO...
Compilingthetext_encoder_2toAUTO...
Compilingthetext_encodertoAUTO...
Compilingthevae_encodertoAUTO...


..parsed-literal::

FP16pipelinelatency:1.391
INT8pipelinelatency:0.781
Text-to-Imagegenerationspeedup:1.780


InteractiveDemo
----------------

`backtotop⬆️<#table-of-contents>`__

Now,youcancheckmodelworkusingowntextdescriptions.Providetext
promptinthetextboxandlaunchgenerationusingRunbutton.
Additionallyyoucancontrolgenerationwithadditionalparameters:\*
Seed-randomseedforinitialization\*Steps-numberofgeneration
steps\*HeightandWidth-sizeofgeneratedimage

Pleasenotethatincreasingimagesizemayrequiretoincreasing
numberofstepsforaccurateresult.Werecommendrunning104x1024
resolutionimagegenerationusing4steps.

Pleaseselectbelowwhetheryouwouldliketousethequantizedmodelto
launchtheinteractivedemo.

..code::ipython3

quantized_model_present=UNET_INT8_OV_PATH.exists()

use_quantized_model=widgets.Checkbox(
value=Trueifquantized_model_presentelseFalse,
description="Usequantizedmodel",
disabled=False,
)

use_quantized_model




..parsed-literal::

Checkbox(value=True,description='Usequantizedmodel')



..code::ipython3

importgradioasgr

text2image_pipe=OVStableDiffusionXLPipeline.from_pretrained(model_dir,device=device.value)
ifuse_quantized_model.value:
ifnotquantized_model_present:
raiseRuntimeError("Quantizedmodelnotfound.")
text2image_pipe.unet.model=core.read_model(UNET_INT8_OV_PATH)
text2image_pipe.unet.request=core.compile_model(text2image_pipe.unet.model,device.value)


defgenerate_from_text(text,seed,num_steps,height,width):
result=text2image_pipe(
text,
num_inference_steps=num_steps,
guidance_scale=0.0,
generator=np.random.RandomState(seed),
height=height,
width=width,
).images[0]
returnresult


withgr.Blocks()asdemo:
withgr.Column():
positive_input=gr.Textbox(label="Textprompt")
withgr.Row():
seed_input=gr.Number(precision=0,label="Seed",value=42,minimum=0)
steps_input=gr.Slider(label="Steps",value=1,minimum=1,maximum=4,step=1)
height_input=gr.Slider(label="Height",value=512,minimum=256,maximum=1024,step=32)
width_input=gr.Slider(label="Width",value=512,minimum=256,maximum=1024,step=32)
btn=gr.Button()
out=gr.Image(
label=("Result(Quantized)"ifuse_quantized_model.valueelse"Result(Original)"),
type="pil",
width=512,
)
btn.click(
generate_from_text,
[positive_input,seed_input,steps_input,height_input,width_input],
out,
)
gr.Examples(
[
["cutecat",999],
[
"underwaterworldcoralreef,colorfuljellyfish,35mm,cinematiclighting,shallowdepthoffield,ultraquality,masterpiece,realistic",
89,
],
[
"aphotorealistichappywhitepoodledog​​playinginthegrass,extremelydetailed,highres,8k,masterpiece,dynamicangle",
1569,
],
[
"AstronautonMarswatchingsunset,bestquality,cinematiceffects,",
65245,
],
[
"BlackandwhitestreetphotographyofarainynightinNewYork,reflectionsonwetpavement",
48199,
],
],
[positive_input,seed_input],
)

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
#ifyouwantcreatepubliclinkforsharingdemo,pleaseaddshare=True
try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
