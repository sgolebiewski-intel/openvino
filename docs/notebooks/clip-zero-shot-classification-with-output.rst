Zero-shotImageClassificationwithOpenAICLIPandOpenVINO™
=============================================================

Zero-shotimageclassificationisacomputervisiontasktoclassify
imagesintooneofseveralclasseswithoutanypriortrainingor
knowledgeoftheclasses.

..figure::https://user-images.githubusercontent.com/29454499/207773481-d77cacf8-6cdc-4765-a31b-a1669476d620.png
:alt:zero-shot-pipeline

zero-shot-pipeline

`\**image
source\*<https://huggingface.co/tasks/zero-shot-image-classification>`__

Zero-shotlearningresolvesseveralchallengesinimageretrieval
systems.Forexample,withtherapidgrowthofcategoriesontheweb,it
ischallengingtoindeximagesbasedonunseencategories.Wecan
associateunseencategoriestoimageswithzero-shotlearningby
exploitingattributestomodel’srelationshipbetweenvisualfeatures
andlabels.Inthistutorial,wewillusethe`OpenAI
CLIP<https://github.com/openai/CLIP>`__modeltoperformzero-shot
imageclassification.Additionally,thenotebookdemonstrateshowto
optimizethemodelusing
`NNCF<https://github.com/openvinotoolkit/nncf/>`__.

Thenotebookcontainsthefollowingsteps:

1.Downloadthemodel.
2.InstantiatethePyTorchmodel.
3.ConvertmodeltoOpenVINOIR,usingthemodelconversionAPI.
4.RunCLIPwithOpenVINO.
5.QuantizetheconvertedmodelwithNNCF.
6.Checkthequantizedmodelinferenceresult.
7.Comparemodelsizeofconvertedandquantizedmodels.
8.Compareperformanceofconvertedandquantizedmodels.
9.Launchinteractivedemo

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Instantiatemodel<#instantiate-model>`__
-`RunPyTorchmodelinference<#run-pytorch-model-inference>`__
-`ConvertmodeltoOpenVINOIntermediateRepresentation(IR)
format.<#convert-model-to-openvino-intermediate-representation-ir-format->`__
-`RunOpenVINOmodel<#run-openvino-model>`__

-`Selectinferencedevice<#select-inference-device>`__

-`QuantizemodeltoINT8using
NNCF<#quantize-model-to-int8-using-nncf>`__

-`Preparedatasets<#prepare-datasets>`__
-`Performquantization<#perform-quantization>`__
-`RunquantizedOpenVINOmodel<#run-quantized-openvino-model>`__
-`CompareFileSize<#compare-file-size>`__
-`CompareinferencetimeoftheFP16IRandquantized
models<#compare-inference-time-of-the-fp16-ir-and-quantized-models>`__

-`Interactivedemo<#interactive-demo>`__

Instantiatemodel
-----------------

`backtotop⬆️<#table-of-contents>`__

CLIP(ContrastiveLanguage-ImagePre-Training)isaneuralnetwork
trainedonvarious(image,text)pairs.Itcanbeinstructedinnatural
languagetopredictthemostrelevanttextsnippet,givenanimage,
withoutdirectlyoptimizingforthetask.CLIPusesa
`ViT<https://arxiv.org/abs/2010.11929>`__liketransformertoget
visualfeaturesandacausallanguagemodeltogetthetextfeatures.
Thetextandvisualfeaturesarethenprojectedintoalatentspacewith
identicaldimensions.Thedotproductbetweentheprojectedimageand
textfeaturesisthenusedasasimilarityscore.

..figure::https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png
:alt:clip

clip

`\**image_source\*<https://github.com/openai/CLIP/blob/main/README.md>`__

Youcanfindmoreinformationaboutthismodelinthe`research
paper<https://arxiv.org/abs/2103.00020>`__,`OpenAI
blog<https://openai.com/blog/clip/>`__,`model
card<https://github.com/openai/CLIP/blob/main/model-card.md>`__and
GitHub`repository<https://github.com/openai/CLIP>`__.

Inthisnotebook,wewilluse
`openai/clip-vit-base-patch16<https://huggingface.co/openai/clip-vit-base-patch16>`__,
availableviaHuggingFaceTransformers,butthesamestepsare
applicableforotherCLIPfamilymodels.

First,weneedtocreate``CLIPModel``classobjectandinitializeit
withmodelconfigurationandweights,using``from_pretrained``method.
ThemodelwillbeautomaticallydownloadedfromHuggingFaceHuband
cachedforthenextusage.``CLIPProcessor``classisawrapperfor
inputdatapreprocessing.Itincludesbothencodingthetextusing
tokenizerandpreparingtheimages.

..code::ipython3

importplatform

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"gradio>=4.19""openvino>=2023.1.0""transformers[torch]>=4.30""datasets""nncf>=2.6.0""torch>=2.1"Pillow

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

fromtransformersimportCLIPProcessor,CLIPModel

#loadpre-trainedmodel
model=CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
#loadpreprocessorformodelinput
processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


..parsed-literal::

2024-02-2612:23:32.559340:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-02-2612:23:32.561128:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-02-2612:23:32.599733:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-02-2612:23:33.401048:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


..code::ipython3

fromtypingimportList
importmatplotlib.pyplotasplt
importnumpyasnp
fromPILimportImage


defvisualize_result(image:Image,labels:List[str],probs:np.ndarray,top:int=5):
"""
Utilityfunctionforvisualizationclassificationresults
params:
image:inputimage
labels:listofclassificationlabels
probs:modelpredictedsoftmaxedprobabilitiesforeachlabel
top:numberofthehighestprobabilityresultsforvisualization
returns:
None
"""
plt.figure(figsize=(64,64))
top_labels=np.argsort(-probs)[:min(top,probs.shape[0])]
top_probs=probs[top_labels]
plt.subplot(8,8,1)
plt.imshow(image)
plt.axis("off")

plt.subplot(8,8,2)
y=np.arange(top_probs.shape[-1])
plt.grid()
plt.barh(y,top_probs)
plt.gca().invert_yaxis()
plt.gca().set_axisbelow(True)
plt.yticks(y,[labels[index]forindexintop_labels])
plt.xlabel("probability")

RunPyTorchmodelinference
---------------------------

`backtotop⬆️<#table-of-contents>`__

Toperformclassification,definelabelsandloadanimageinRGB
format.Togivethemodelwidertextcontextandimproveguidance,we
extendthelabelsdescriptionusingthetemplate“Thisisaphotoofa”.
Boththelistoflabeldescriptionsandimageshouldbepassedthrough
theprocessortoobtainadictionarywithinputdatainthe
model-specificformat.Themodelpredictsanimage-textsimilarityscore
inrawlogitsformat,whichcanbenormalizedtothe``[0,1]``range
usingthe``softmax``function.Then,weselectlabelswiththehighest
similarityscoreforthefinalresult.

..code::ipython3

importrequests
frompathlibimportPath


sample_path=Path("data/coco.jpg")
sample_path.parent.mkdir(parents=True,exist_ok=True)
r=requests.get("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg")

withsample_path.open("wb")asf:
f.write(r.content)

image=Image.open(sample_path)

input_labels=[
"cat",
"dog",
"wolf",
"tiger",
"man",
"horse",
"frog",
"tree",
"house",
"computer",
]
text_descriptions=[f"Thisisaphotoofa{label}"forlabelininput_labels]

inputs=processor(text=text_descriptions,images=[image],return_tensors="pt",padding=True)

results=model(**inputs)
logits_per_image=results["logits_per_image"]#thisistheimage-textsimilarityscore
probs=logits_per_image.softmax(dim=1).detach().numpy()#wecantakethesoftmaxtogetthelabelprobabilities
visualize_result(image,input_labels,probs[0])



..image::clip-zero-shot-classification-with-output_files/clip-zero-shot-classification-with-output_6_0.png


ConvertmodeltoOpenVINOIntermediateRepresentation(IR)format.
------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

ForbestresultswithOpenVINO,itisrecommendedtoconvertthemodel
toOpenVINOIRformat.OpenVINOsupportsPyTorchviaModelconversion
API.ToconvertthePyTorchmodeltoOpenVINOIRformatwewilluse
``ov.convert_model``of`modelconversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
The``ov.convert_model``PythonfunctionreturnsanOpenVINOModel
objectreadytoloadonthedeviceandstartmakingpredictions.Wecan
saveitondiskforthenextusagewith``ov.save_model``.

..code::ipython3

importopenvinoasov

fp16_model_path=Path("clip-vit-base-patch16.xml")
model.config.torchscript=True

ifnotfp16_model_path.exists():
ov_model=ov.convert_model(model,example_input=dict(inputs))
ov.save_model(ov_model,fp16_model_path)

RunOpenVINOmodel
------------------

`backtotop⬆️<#table-of-contents>`__

ThestepsformakingpredictionswiththeOpenVINOCLIPmodelare
similartothePyTorchmodel.Letuscheckthemodelresultusingthe
sameinputdatafromtheexampleabovewithPyTorch.

..code::ipython3

fromscipy.specialimportsoftmax

#createOpenVINOcoreobjectinstance
core=ov.Core()

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

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

Dropdown(description='Device:',index=3,options=('CPU','GPU.0','GPU.1','AUTO'),value='AUTO')



..code::ipython3

#compilemodelforloadingondevice
compiled_model=core.compile_model(fp16_model_path,device.value)
#runinferenceonpreprocesseddataandgetimage-textsimilarityscore
ov_logits_per_image=compiled_model(dict(inputs))[0]
#performsoftmaxonscore
probs=softmax(ov_logits_per_image,axis=1)
#visualizeprediction
visualize_result(image,input_labels,probs[0])



..image::clip-zero-shot-classification-with-output_files/clip-zero-shot-classification-with-output_13_0.png


Great!Lookslikewegotthesameresult.

QuantizemodeltoINT8usingNNCF
---------------------------------

`backtotop⬆️<#table-of-contents>`__##QuantizemodeltoINT8using
NNCF

Thegoalofthispartoftutorialistodemonstratehowtospeedupthe
modelbyapplying8-bitpost-trainingquantizationfrom
`NNCF<https://github.com/openvinotoolkit/nncf/>`__(NeuralNetwork
CompressionFramework)andinferquantizedmodelviaOpenVINO™Toolkit.
`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.Theoptimizationprocesscontainsthefollowingsteps:

1.Preparequantizationdataset
2.QuantizetheconvertedOpenVINOmodelwithNNCF.
3.Checkthemodelresultusingthesameinputdatalikeweuse.
4.Comparemodelsizeofconvertedandquantizedmodels.
5.Compareperformanceofconvertedandquantizedmodels.

..

**Note:**quantizationprocessmayrequireadditionaltimeandmemory
forperforming.Youcandisableitusingwidgetbelow:

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

#Fetchskip_kernel_extensionmodule
r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

%load_extskip_kernel_extension

Preparedatasets
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The`Conceptual
Captions<https://ai.google.com/research/ConceptualCaptions/>`__dataset
consistingof~3.3Mimagesannotatedwithcaptionsisusedtoquantize
model.

..code::ipython3

%%skipnot$to_quantize.value

importrequests
fromioimportBytesIO
importnumpyasnp
fromPILimportImage
fromrequests.packages.urllib3.exceptionsimportInsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

max_length=model.config.text_config.max_position_embeddings

defcheck_text_data(data):
"""
Checkifthegivendataistext-based.
"""
ifisinstance(data,str):
returnTrue
ifisinstance(data,list):
returnall(isinstance(x,str)forxindata)
returnFalse

defget_pil_from_url(url):
"""
DownloadsandconvertsanimagefromaURLtoaPILImageobject.
"""
response=requests.get(url,verify=False,timeout=20)
image=Image.open(BytesIO(response.content))
returnimage.convert("RGB")

defcollate_fn(example,image_column="image_url",text_column="caption"):
"""
Preprocessesanexamplebyloadingandtransformingimageandtextdata.
Checksifthetextdataintheexampleisvalidbycallingthe`check_text_data`function.
DownloadstheimagespecifiedbytheURLintheimage_columnbycallingthe`get_pil_from_url`function.
Ifthereisanyerrorduringthedownloadprocess,returnsNone.
Returnsthepreprocessedinputswithtransformedimageandtextdata.
"""
assertlen(example)==1
example=example[0]

ifnotcheck_text_data(example[text_column]):
raiseValueError("Textdataisnotvalid")

url=example[image_column]
try:
image=get_pil_from_url(url)
h,w=image.size
ifh==1orw==1:
returnNone
exceptException:
returnNone

inputs=processor(text=example[text_column],images=[image],return_tensors="pt",padding=True)
ifinputs['input_ids'].shape[1]>max_length:
returnNone
returninputs

..code::ipython3

%%skipnot$to_quantize.value

importtorch
fromdatasetsimportload_dataset
fromtqdm.notebookimporttqdm

defprepare_calibration_data(dataloader,init_steps):
"""
Thisfunctionpreparescalibrationdatafromadataloaderforaspecifiednumberofinitializationsteps.
Ititeratesoverthedataloader,fetchingbatchesandstoringtherelevantdata.
"""
data=[]
print(f"Fetching{init_steps}samplesfortheinitialization...")
withtqdm(total=init_steps)aspbar:
forbatchindataloader:
iflen(data)==init_steps:
break
ifbatch:
pbar.update(1)
withtorch.no_grad():
data.append(
{
"pixel_values":batch["pixel_values"].to("cpu"),
"input_ids":batch["input_ids"].to("cpu"),
"attention_mask":batch["attention_mask"].to("cpu")
}
)
returndata


defprepare_dataset(opt_init_steps=50,max_train_samples=1000):
"""
Preparesavision-textdatasetforquantization.
"""
dataset=load_dataset("google-research-datasets/conceptual_captions",trust_remote_code=True)
train_dataset=dataset["train"].shuffle(seed=42)
dataloader=torch.utils.data.DataLoader(train_dataset,collate_fn=collate_fn,batch_size=1)
calibration_data=prepare_calibration_data(dataloader,opt_init_steps)
returncalibration_data

..code::ipython3

%%skipnot$to_quantize.value

importlogging
importnncf

core=ov.Core()

nncf.set_log_level(logging.ERROR)

int8_model_path='clip-vit-base-patch16_int8.xml'
calibration_data=prepare_dataset()
ov_model=core.read_model(fp16_model_path)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..parsed-literal::

/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/datasets/load.py:1429:FutureWarning:Therepositoryforconceptual_captionscontainscustomcodewhichmustbeexecutedtocorrectlyloadthedataset.Youcaninspecttherepositorycontentathttps://hf.co/datasets/conceptual_captions
Youcanavoidthismessageinfuturebypassingtheargument`trust_remote_code=True`.
Passing`trust_remote_code=True`willbemandatorytoloadthisdatasetfromthenextmajorreleaseof`datasets`.
warnings.warn(


..parsed-literal::

Fetching50samplesfortheinitialization...



..parsed-literal::

0%||0/50[00:00<?,?it/s]


Performquantization
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Createaquantizedmodelfromthepre-trained``FP16``model.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakealongtime.

..code::ipython3

%%skipnot$to_quantize.value

iflen(calibration_data)==0:
raiseRuntimeError(
'Calibrationdatasetisempty.Pleasecheckinternetconnectionandtrytodownloadimagesmanually.'
)

calibration_dataset=nncf.Dataset(calibration_data)
quantized_model=nncf.quantize(
model=ov_model,
calibration_dataset=calibration_dataset,
model_type=nncf.ModelType.TRANSFORMER,
#SmoothQuantalgorithmreducesactivationquantizationerror;optimalalphavaluewasobtainedthroughgridsearch
advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6)
)
ov.save_model(quantized_model,int8_model_path)


..parsed-literal::

/home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/nncf/quantization/algorithms/post_training/pipeline.py:87:FutureWarning:`AdvancedQuantizationParameters(smooth_quant_alpha=..)`isdeprecated.Please,use`AdvancedQuantizationParameters(smooth_quant_alphas)`optionwithAdvancedSmoothQuantParameters(convolution=..,matmul=..)asvalueinstead.
warning_deprecated(



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



RunquantizedOpenVINOmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThestepsformakingpredictionswiththequantizedOpenVINOCLIPmodel
aresimilartothePyTorchmodel.Letuscheckthemodelresultusing
thesameinputdatathatweusedbefore.

..code::ipython3

%%skipnot$to_quantize.value

#compilemodelforloadingondevice
compiled_model=core.compile_model(quantized_model,device.value)
#runinferenceonpreprocesseddataandgetimage-textsimilarityscore
ov_logits_per_image=compiled_model(dict(inputs))[0]
#performsoftmaxonscore
probs=softmax(ov_logits_per_image,axis=1)
#visualizeprediction
visualize_result(image,input_labels,probs[0])



..image::clip-zero-shot-classification-with-output_files/clip-zero-shot-classification-with-output_26_0.png


Nice!Resultslookssimilartofp16modelresultsbeforequantization.

CompareFileSize
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

frompathlibimportPath

fp16_ir_model_size=Path(fp16_model_path).with_suffix(".bin").stat().st_size/1024/1024
quantized_model_size=Path(int8_model_path).with_suffix(".bin").stat().st_size/1024/1024
print(f"FP16IRmodelsize:{fp16_ir_model_size:.2f}MB")
print(f"INT8modelsize:{quantized_model_size:.2f}MB")
print(f"Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16IRmodelsize:285.38MB
INT8modelsize:143.60MB
Modelcompressionrate:1.987


CompareinferencetimeoftheFP16IRandquantizedmodels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__Tomeasuretheinference
performanceofthe``FP16``and``INT8``models,weusemedianinference
timeoncalibrationdataset.Sowecanapproximatelyestimatethespeed
upofthedynamicquantizedmodels.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplicationswithstaticshapes.

..code::ipython3

%%skipnot$to_quantize.value

importtime

defcalculate_inference_time(model_path,calibration_data):
model=core.compile_model(model_path,device.value)
inference_time=[]
forbatchincalibration_data:
start=time.perf_counter()
_=model(batch)[0]
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
returnnp.median(inference_time)

..code::ipython3

%%skipnot$to_quantize.value

fp16_latency=calculate_inference_time(fp16_model_path,calibration_data)
int8_latency=calculate_inference_time(int8_model_path,calibration_data)
print(f"Performancespeedup:{fp16_latency/int8_latency:.3f}")


..parsed-literal::

Performancespeedup:1.639


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__##Interactivedemo

Now,itisyourturn!Youcanprovideyourownimageandcomma-separated
listoflabelsforzero-shotclassification.

Feelfreetouploadanimage,usingthefileuploadwindowandtype
labelnamesintothetextfield,usingcommaastheseparator(for
example,``cat,dog,bird``)

..code::ipython3

importgradioasgr

model_path=Path("clip-vit-base-patch16-int8.xml")
ifnotmodel_path.exists():
model_path=Path("clip-vit-base-patch16.xml")
compiled_model=core.compile_model(model_path,device.value)


defclassify(image,text):
"""Classifyimageusingclasseslisting.
Args:
image(np.ndarray):imagethatneedstobeclassifiedinCHWformat.
text(str):comma-separatedlistofclasslabels
Returns:
(dict):Mappingbetweenclasslabelsandclassprobabilities.
"""
labels=text.split(",")
text_descriptions=[f"Thisisaphotoofa{label}"forlabelinlabels]
inputs=processor(text=text_descriptions,images=[image],return_tensors="np",padding=True)
ov_logits_per_image=compiled_model(dict(inputs))[0]
probs=softmax(ov_logits_per_image,axis=1)[0]

return{label:float(prob)forlabel,probinzip(labels,probs)}


demo=gr.Interface(
classify,
[
gr.Image(label="Image",type="pil"),
gr.Textbox(label="Labels",info="Comma-separatedlistofclasslabels"),
],
gr.Label(label="Result"),
examples=[[sample_path,"cat,dog,bird"]],
)
try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
