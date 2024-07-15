Zero-shotImageClassificationwithSigLIP
==========================================

|Colab|

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
andlabels.Inthistutorial,wewillusethe
`SigLIP<https://huggingface.co/docs/transformers/main/en/model_doc/siglip>`__
modeltoperformzero-shotimageclassification.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Instantiatemodel<#instantiate-model>`__
-`RunPyTorchmodelinference<#run-pytorch-model-inference>`__
-`ConvertmodeltoOpenVINOIntermediateRepresentation(IR)
format<#convert-model-to-openvino-intermediate-representation-ir-format>`__
-`RunOpenVINOmodel<#run-openvino-model>`__
-`Applypost-trainingquantizationusing
NNCF<#apply-post-training-quantization-using-nncf>`__

-`Preparedataset<#prepare-dataset>`__
-`Quantizemodel<#quantize-model>`__
-`RunquantizedOpenVINOmodel<#run-quantized-openvino-model>`__
-`CompareFileSize<#compare-file-size>`__
-`CompareinferencetimeoftheFP16IRandquantized
models<#compare-inference-time-of-the-fp16-ir-and-quantized-models>`__

-`Interactiveinference<#interactive-inference>`__

..|Colab|image::https://colab.research.google.com/assets/colab-badge.svg
:target:https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/siglip-zero-shot-image-classification/siglip-zero-shot-image-classification.ipynb

Instantiatemodel
-----------------

`backtotop⬆️<#table-of-contents>`__

TheSigLIPmodelwasproposedin`SigmoidLossforLanguageImage
Pre-Training<https://arxiv.org/abs/2303.15343>`__.SigLIPproposesto
replacethelossfunctionusedin
`CLIP<https://github.com/openai/CLIP>`__(ContrastiveLanguage–Image
Pre-training)byasimplepairwisesigmoidloss.Thisresultsinbetter
performanceintermsofzero-shotclassificationaccuracyonImageNet.

Theabstractfromthepaperisthefollowing:

WeproposeasimplepairwiseSigmoidlossforLanguage-Image
Pre-training(SigLIP).Unlikestandardcontrastivelearningwith
softmaxnormalization,thesigmoidlossoperatessolelyonimage-text
pairsanddoesnotrequireaglobalviewofthepairwisesimilarities
fornormalization.Thesigmoidlosssimultaneouslyallowsfurther
scalingupthebatchsize,whilealsoperformingbetteratsmaller
batchsizes.

Youcanfindmoreinformationaboutthismodelinthe`research
paper<https://arxiv.org/abs/2303.15343>`__,`GitHub
repository<https://github.com/google-research/big_vision>`__,`Hugging
Facemodel
page<https://huggingface.co/docs/transformers/main/en/model_doc/siglip>`__.

Inthisnotebook,wewilluse
`google/siglip-base-patch16-224<https://huggingface.co/google/siglip-base-patch16-224>`__,
availableviaHuggingFaceTransformers,butthesamestepsare
applicableforotherCLIPfamilymodels.

First,weneedtocreate``AutoModel``classobjectandinitializeit
withmodelconfigurationandweights,using``from_pretrained``method.
ThemodelwillbeautomaticallydownloadedfromHuggingFaceHuband
cachedforthenextusage.``AutoProcessor``classisawrapperfor
inputdatapreprocessing.Itincludesbothencodingthetextusing
tokenizerandpreparingtheimages.

..code::ipython3

importplatform

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"gradio>=4.19""openvino>=2023.3.0""transformers>=4.37""torch>=2.1"Pillowsentencepieceprotobufscipydatasetsnncf

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

fromtransformersimportAutoProcessor,AutoModel

model=AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor=AutoProcessor.from_pretrained("google/siglip-base-patch16-224")


..parsed-literal::

2024-07-1302:43:57.477894:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1302:43:57.512131:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1302:43:58.111651:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1132:FutureWarning:`resume_download`isdeprecatedandwillberemovedinversion1.0.0.Downloadsalwaysresumewhenpossible.Ifyouwanttoforceanewdownload,use`force_download=True`.
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1132:FutureWarning:`resume_download`isdeprecatedandwillberemovedinversion1.0.0.Downloadsalwaysresumewhenpossible.Ifyouwanttoforceanewdownload,use`force_download=True`.
warnings.warn(


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

#Resultsvisualizationfunction
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
plt.figure(figsize=(72,64))
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

print([{labels[x]:round(y,2)}forx,yinzip(top_labels,top_probs)])

..code::ipython3

importrequests
frompathlibimportPath
importtorch
fromPILimportImage

image_path=Path("test_image.jpg")
r=requests.get(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
)

withimage_path.open("wb")asf:
f.write(r.content)
image=Image.open(image_path)

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

inputs=processor(text=text_descriptions,images=[image],padding="max_length",return_tensors="pt")

withtorch.no_grad():
model.config.torchscript=False
results=model(**inputs)

logits_per_image=results["logits_per_image"]#thisistheimage-textsimilarityscore

probs=logits_per_image.softmax(dim=1).detach().numpy()
visualize_result(image,input_labels,probs[0])


..parsed-literal::

[{'dog':0.99},{'cat':0.0},{'horse':0.0},{'wolf':0.0},{'tiger':0.0}]



..image::siglip-zero-shot-image-classification-with-output_files/siglip-zero-shot-image-classification-with-output_6_1.png


ConvertmodeltoOpenVINOIntermediateRepresentation(IR)format
-----------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

ForbestresultswithOpenVINO,itisrecommendedtoconvertthemodel
toOpenVINOIRformat.OpenVINOsupportsPyTorchviaModelconversion
API.ToconvertthePyTorchmodeltoOpenVINOIRformatwewilluse
``ov.convert_model``of`modelconversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
The``ov.convert_model``PythonfunctionreturnsanOpenVINOModel
objectreadytoloadonthedeviceandstartmakingpredictions.

..code::ipython3

importopenvinoasov

model.config.torchscript=True
ov_model=ov.convert_model(model,example_input=dict(inputs))


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/siglip/modeling_siglip.py:354:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_weights.size()!=(batch_size,self.num_heads,q_len,k_v_seq_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/siglip/modeling_siglip.py:372:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_output.size()!=(batch_size,self.num_heads,q_len,self.head_dim):


..parsed-literal::

['input_ids','pixel_values']


RunOpenVINOmodel
------------------

`backtotop⬆️<#table-of-contents>`__

ThestepsformakingpredictionswiththeOpenVINOSigLIPmodelare
similartothePyTorchmodel.Letuscheckthemodelresultusingthe
sameinputdatafromtheexampleabovewithPyTorch.

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO

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



RunOpenVINOmodel

..code::ipython3

fromscipy.specialimportsoftmax

#compilemodelforloadingondevice
compiled_ov_model=core.compile_model(ov_model,device.value)
#obtainoutputtensorforgettingpredictions
logits_per_image_out=compiled_ov_model.output(0)
#runinferenceonpreprocesseddataandgetimage-textsimilarityscore
ov_logits_per_image=compiled_ov_model(dict(inputs))[logits_per_image_out]
#performsoftmaxonscore
probs=softmax(ov_logits_per_image[0])
#visualizeprediction
visualize_result(image,input_labels,probs)


..parsed-literal::

[{'dog':0.99},{'cat':0.0},{'horse':0.0},{'wolf':0.0},{'tiger':0.0}]



..image::siglip-zero-shot-image-classification-with-output_files/siglip-zero-shot-image-classification-with-output_13_1.png


Great!Lookslikewegotthesameresult.

Applypost-trainingquantizationusingNNCF
-------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingthequantizationlayersintothe
modelgraphandthenusingasubsetofthetrainingdatasetto
initializetheparametersoftheseadditionalquantizationlayers.The
frameworkisdesignedsothatmodificationstoyouroriginaltraining
codeareminor.Quantizationisthesimplestscenarioandrequiresafew
modifications.

Theoptimizationprocesscontainsthefollowingsteps:

1.Createadatasetforquantization.
2.Run``nncf.quantize``forgettingaquantizedmodel.

Preparedataset
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The`Conceptual
Captions<https://ai.google.com/research/ConceptualCaptions/>`__dataset
consistingof~3.3Mimagesannotatedwithcaptionsisusedtoquantize
model.

..code::ipython3

importrequests
fromioimportBytesIO
fromPILimportImage
fromrequests.packages.urllib3.exceptionsimportInsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


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

inputs=processor(
text=example[text_column],
images=[image],
return_tensors="pt",
padding="max_length",
)
ifinputs["input_ids"].shape[1]>model.config.text_config.max_position_embeddings:
returnNone
returninputs

..code::ipython3

importtorch
fromdatasetsimportload_dataset
fromtqdm.notebookimporttqdm


defprepare_calibration_data(dataloader,init_steps):
"""
Thisfunctionpreparescalibrationdatafromadataloaderforaspecifiednumberofinitializationsteps.
Ititeratesoverthedataloader,fetchingbatchesandstoringtherelevantdata.
"""
data=[]
print(f"Fetching{init_steps}fortheinitialization...")
counter=0
forbatchintqdm(dataloader):
ifcounter==init_steps:
break
ifbatch:
counter+=1
withtorch.no_grad():
data.append(
{
"pixel_values":batch["pixel_values"].to("cpu"),
"input_ids":batch["input_ids"].to("cpu"),
}
)
returndata


defprepare_dataset(opt_init_steps=300,max_train_samples=1000):
"""
Preparesavision-textdatasetforquantization.
"""
dataset=load_dataset("google-research-datasets/conceptual_captions",streaming=True,trust_remote_code=True)
train_dataset=dataset["train"].shuffle(seed=42,buffer_size=max_train_samples)
dataloader=torch.utils.data.DataLoader(train_dataset,collate_fn=collate_fn,batch_size=1)
calibration_data=prepare_calibration_data(dataloader,opt_init_steps)
returncalibration_data

..code::ipython3

calibration_data=prepare_dataset()


..parsed-literal::

Fetching300fortheinitialization...



..parsed-literal::

0it[00:00,?it/s]


Quantizemodel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Createaquantizedmodelfromthepre-trained``FP16``model.

**NOTE**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakealongtime.

..code::ipython3

importnncf
importlogging

nncf.set_log_level(logging.ERROR)

iflen(calibration_data)==0:
raiseRuntimeError("Calibrationdatasetisempty.Pleasecheckinternetconnectionandtrytodownloadimagesmanually.")

calibration_dataset=nncf.Dataset(calibration_data)
quantized_ov_model=nncf.quantize(
model=ov_model,
calibration_dataset=calibration_dataset,
model_type=nncf.ModelType.TRANSFORMER,
)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino



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



NNCFalsosupportsquantization-awaretraining,andotheralgorithms
thanquantization.Seethe`NNCF
documentation<https://github.com/openvinotoolkit/nncf/#documentation>`__
intheNNCFrepositoryformoreinformation.

RunquantizedOpenVINOmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThestepsformakingpredictionswiththequantizedOpenVINOSigLIP
modelaresimilartothePyTorchmodel.

..code::ipython3

fromscipy.specialimportsoftmax


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

inputs=processor(text=text_descriptions,images=[image],return_tensors="pt",padding="max_length")
compiled_int8_ov_model=ov.compile_model(quantized_ov_model,device.value)

logits_per_image_out=compiled_int8_ov_model.output(0)
ov_logits_per_image=compiled_int8_ov_model(dict(inputs))[logits_per_image_out]
probs=softmax(ov_logits_per_image,axis=1)
visualize_result(image,input_labels,probs[0])


..parsed-literal::

[{'dog':0.99},{'cat':0.0},{'horse':0.0},{'wolf':0.0},{'tiger':0.0}]



..image::siglip-zero-shot-image-classification-with-output_files/siglip-zero-shot-image-classification-with-output_24_1.png


CompareFileSize
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frompathlibimportPath

fp16_model_path="siglip-base-patch16-224.xml"
ov.save_model(ov_model,fp16_model_path)

int8_model_path="siglip-base-patch16-224_int8.xml"
ov.save_model(quantized_ov_model,int8_model_path)

fp16_ir_model_size=Path(fp16_model_path).with_suffix(".bin").stat().st_size/1024/1024
quantized_model_size=Path(int8_model_path).with_suffix(".bin").stat().st_size/1024/1024
print(f"FP16IRmodelsize:{fp16_ir_model_size:.2f}MB")
print(f"INT8modelsize:{quantized_model_size:.2f}MB")
print(f"Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16IRmodelsize:387.49MB
INT8modelsize:201.26MB
Modelcompressionrate:1.925


CompareinferencetimeoftheFP16IRandquantizedmodels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tomeasuretheinferenceperformanceofthe``FP16``and``INT8``
models,weusemedianinferencetimeoncalibrationdataset.Sowecan
approximatelyestimatethespeedupofthedynamicquantizedmodels.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplicationswithstaticshapes.

..code::ipython3

importtime


defcalculate_inference_time(model_path,calibration_data):
model=ov.compile_model(model_path,device.value)
output_layer=model.output(0)
inference_time=[]
forbatchincalibration_data:
start=time.perf_counter()
_=model(batch)[output_layer]
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
returnnp.median(inference_time)

..code::ipython3

fp16_latency=calculate_inference_time(fp16_model_path,calibration_data)
int8_latency=calculate_inference_time(int8_model_path,calibration_data)
print(f"Performancespeedup:{fp16_latency/int8_latency:.3f}")


..parsed-literal::

Performancespeedup:2.088


Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

Now,itisyourturn!Youcanprovideyourownimageandcomma-separated
listoflabelsforzero-shotclassification.Feelfreetouploadan
image,usingthefileuploadwindowandtypelabelnamesintothetext
field,usingcommaastheseparator(forexample,``cat,dog,bird``)

..code::ipython3

importgradioasgr


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
inputs=processor(
text=text_descriptions,
images=[image],
return_tensors="np",
padding="max_length",
)
ov_logits_per_image=compiled_int8_ov_model(dict(inputs))[logits_per_image_out]
probs=softmax(ov_logits_per_image[0])

return{label:float(prob)forlabel,probinzip(labels,probs)}


demo=gr.Interface(
classify,
[
gr.Image(label="Image",type="pil"),
gr.Textbox(label="Labels",info="Comma-separatedlistofclasslabels"),
],
gr.Label(label="Result"),
examples=[[image_path,"cat,dog,bird"]],
)
try:
demo.launch(debug=False,height=1000)
exceptException:
demo.launch(share=True,debug=False,height=1000)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="1000"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>

