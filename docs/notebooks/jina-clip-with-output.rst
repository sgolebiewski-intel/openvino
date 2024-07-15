CLIPmodelwithJinaCLIPandOpenVINO
--------------------------------------

`jina-clip-v1<https://huggingface.co/jinaai/jina-clip-v1>`__isa
state-of-the-artEnglishmultimodal(text-image)embeddingmodeltrained
by`JinaAI<https://aimodels.fyi/creators/huggingFace/jinaai>`__.It
bridgesthegapbetweentraditionaltextembeddingmodels,whichexcel
intext-to-textretrievalbutareincapableofcross-modaltasks,and
modelsthateffectivelyalignimageandtextembeddingsbutarenot
optimizedfortext-to-textretrieval.jina-clip-v1offersrobust
performanceinbothdomains.Itsdualcapabilitymakesitanexcellent
toolformultimodalretrieval-augmentedgeneration(MuRAG)applications,
allowingseamlesstext-to-textandtext-to-imagesearcheswithina
singlemodel.jina-clip-v1canbeusedforavarietyofmultimodal
applications,suchas:imagesearchbydescribingthemintext,
multimodalquestionanswering,multimodalcontentgeneration.JinaAI
hasalsoprovidedtheEmbeddingsAPIasaneasy-to-useinterfacefor
workingwithjina-clip-v1andtheirotherembeddingmodels.

InthisnotebookwewillloadthemodelwithHuggingFaceTransformers,
convertittoOpenVINOIRformat,optimizeitwithNNCFandshowthe
lifedemo.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Instantiatemodel<#instantiate-model>`__

-`Prepareinputdata<#prepare-input-data>`__
-`RunPyTorchmodelinference<#run-pytorch-model-inference>`__

-`RunOpenVINOmodelinference<#run-openvino-model-inference>`__

-`Prepareinputdata<#prepare-input-data>`__
-`ConvertModeltoOpenVINOIR
format<#convert-model-to-openvino-ir-format>`__
-`Selectinferencedevice<#select-inference-device>`__
-`Compilemodelandrun
inference<#compile-model-and-run-inference>`__

-`QuantizemodeltoINT8using
NNCF<#quantize-model-to-int8-using-nncf>`__

-`Preparedatasets<#prepare-datasets>`__

-`Datasetwithtextdata<#dataset-with-text-data>`__
-`Datasetwithimagedata<#dataset-with-image-data>`__

-`Performquantization<#perform-quantization>`__

-`Quantizationoftextmodel<#quantization-of-text-model>`__
-`Quantizationofimagemodel<#quantization-of-image-model>`__

-`CompareFileSize<#compare-file-size>`__
-`CompareinferencetimeoftheFP16IRandquantized
models<#compare-inference-time-of-the-fp16-ir-and-quantized-models>`__

-`Gradiodemo<#gradio-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"openvino>=2024.2.0""datasets>=2.20""nncf>=2.11.0"
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"gradio>=4.19""pillow""einops""timm""transformers[torch]>=4.39""torch>=2.1"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Instantiatemodel
-----------------

`backtotop⬆️<#table-of-contents>`__

Let’sloadthe
`jinaai/jina-clip-v1<https://huggingface.co/jinaai/jina-clip-v1>`__
withHuggingFaceTransformers.WecreatesPyTorchmodelclassinstance
with``AutoModel``,loadandinitializeitwithmodelconfigurationand
weights,using``from_pretrained``method.

..code::ipython3

fromtransformersimportAutoModel

model=AutoModel.from_pretrained("jinaai/jina-clip-v1",trust_remote_code=True)


..parsed-literal::

2024-07-1300:37:54.543689:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1300:37:54.579011:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1300:37:55.252367:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


Prepareinputdata
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThemodelcanencodemeaningfulsentencesinEnglishastextinput.
Imagecouldbeprovidedtomodelaslocalfilepath,URLsordirectly
passinginthePIL.Imageobjects.

..code::ipython3

fromPILimportImage
importrequests

#imageinputdata
r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

download_file(
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/3f779fc1-c1b2-4dec-915a-64dae510a2bb",
"furseal.png",
directory="data",
)

img_furseal=Image.open("./data/furseal.png")

image_path=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
directory="data",
)

img_coco=Image.open("./data/coco.jpg")

IMAGE_INPUTS=[img_furseal,img_coco]

#textinputdata
TEXT_INPUTS=["Seal","Cobra","Rat","Penguin","Dog"]



..parsed-literal::

data/furseal.png:0%||0.00/2.55M[00:00<?,?B/s]



..parsed-literal::

data/coco.jpg:0%||0.00/202k[00:00<?,?B/s]


..code::ipython3

fromtypingimportList
importmatplotlib.pyplotasplt
importnumpyasnp
fromPILimportImage
fromscipy.specialimportsoftmax


defcalc_simularity_softmax(embeddings1,embeddings2,apply_softmax=True):
simularity=[]
foremb1inembeddings1:
temp_simularity=[]
foremb2inembeddings2:
temp_simularity.append(emb1@emb2)
temp_simularity=softmax(temp_simularity)ifapply_softmaxelsetemp_simularity
simularity.append(temp_simularity)

returnsimularity


defvisionize_result(image:Image,labels:List[str],probs:np.ndarray,top:int=5):
"""
Utilityfunctionforvisionizationclassificationresults
params:
image:inputimage
labels:listofclassificationlabels
probs:modelpredictedsoftmaxedprobabilitiesforeachlabel
top:numberofthehighestprobabilityresultsforvisionization
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
plt.xlabel("simularity")

Wewillusetokenizerandpreprocessfromjina-clipmodel.Wewilltake
``tokenizer``toencodetextinputdatausing``model.get_tokenizer()``
andtake``preprocess``forimagedatausing``model.get_preprocess()``.

..code::ipython3

tokenizer=model.get_tokenizer()

tokenizer_kwargs=dict()
tokenizer_kwargs["padding"]="max_length"
tokenizer_kwargs["max_length"]=512
tokenizer_kwargs["truncation"]=True

text_inputs=tokenizer(
TEXT_INPUTS,
return_tensors="pt",
**tokenizer_kwargs,
).to("cpu")


processor=model.get_preprocess()
vision_inputs=processor(images=IMAGE_INPUTS,return_tensors="pt")

RunPyTorchmodelinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

text_embeddings=model.text_model(text_inputs["input_ids"])
image_embeddings=model.vision_model(vision_inputs["pixel_values"])

res=calc_simularity_softmax(image_embeddings.detach().numpy(),text_embeddings.detach().numpy())
visionize_result(img_furseal,TEXT_INPUTS,np.array(res[0]))



..image::jina-clip-with-output_files/jina-clip-with-output_11_0.png


RunOpenVINOmodelinference
----------------------------

`backtotop⬆️<#table-of-contents>`__

ConvertModeltoOpenVINOIRformat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchmodelsviaconversiontoOpenVINOIntermediate
Representation(IR).OpenVINOmodelconversionAPIshouldbeusedfor
thesepurposes.``ov.convert_model``functionacceptsoriginalPyTorch
modelinstanceandexampleinputfortracingandreturns``ov.Model``
representingthismodelinOpenVINOframework.Convertedmodelcanbe
usedforsavingondiskusing``ov.save_model``functionordirectly
loadingondeviceusing``core.complie_model``.

..code::ipython3

importopenvinoasov
frompathlibimportPath

core=ov.Core()

..code::ipython3

fp16_text_model_path=Path("jina-clip-text_v1_fp16.xml")

ifnotfp16_text_model_path.exists():
ov_text_model=ov.convert_model(model.text_model,example_input=text_inputs["input_ids"])
ov.save_model(ov_text_model,fp16_text_model_path)


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4565:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/jinaai/jina-bert-flash-implementation/b78d1595de294f13ffe7b19d6cd63892a6e4e7a4/mha.py:333:TracerWarning:ConvertingatensortoaPythonfloatmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
softmax_scale=self.softmax_scaleor1.0/math.sqrt(q.shape[-1])
/opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/jinaai/jina-bert-flash-implementation/b78d1595de294f13ffe7b19d6cd63892a6e4e7a4/mha.py:343:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifseqlen>self.linear_biases.shape[-1]:


..code::ipython3

fp16_vision_model_path=Path("jina-clip-vision_v1_fp16.xml")

ifnotfp16_vision_model_path.exists():
ov_vision_model=ov.convert_model(model.vision_model,example_input=vision_inputs["pixel_values"])
ov.save_model(ov_vision_model,fp16_vision_model_path)


..parsed-literal::

/opt/home/k8sworker/.cache/huggingface/modules/transformers_modules/jinaai/jina-clip-implementation/952897b38094b9f6a47b3d9a1d8239523e374098/eva_model.py:468:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
assertH==self.img_size[0]andW==self.img_size[1],(


Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Forstartingwork,pleaseselectinferencedevicefromdropdownlist.

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



Compilemodelandruninference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

compiled_text_model=core.compile_model(fp16_text_model_path,device.value)
compiled_vision_model=core.compile_model(fp16_vision_model_path,device.value)

..code::ipython3

text_ov_res=compiled_text_model(text_inputs["input_ids"])
vis_ov_res=compiled_vision_model(vision_inputs["pixel_values"])

res=calc_simularity_softmax(vis_ov_res[0],text_ov_res[0])
visionize_result(img_furseal,TEXT_INPUTS,np.array(res[0]))



..image::jina-clip-with-output_files/jina-clip-with-output_21_0.png


QuantizemodeltoINT8usingNNCF
---------------------------------

`backtotop⬆️<#table-of-contents>`__

Letsspeedupthemodelbyapplying8-bitpost-trainingquantization
from`NNCF<https://github.com/openvinotoolkit/nncf/>`__(NeuralNetwork
CompressionFramework)andinferquantizedmodelviaOpenVINO™Toolkit.
`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.Theoptimizationprocesscontainsthefollowingsteps:

1.Preparequantizationdataset
2.QuantizetheconvertedOpenVINOmodelwithNNCFwith
``nncf.quantize()``.
3.Savethe``INT8``modelusing``openvino.save_model()``function.
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

#Fetch`skip_kernel_extension`module
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

Datasetwithtextdata
^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

importtorch
fromdatasetsimportload_dataset
fromtqdm.notebookimporttqdm
importrequests
fromioimportBytesIO
importnumpyasnp
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


defcollate_fn_text(example,text_column="caption"):
"""
Preprocessesanexamplebyloadingandtransformingtextdata.
Checksifthetextdataintheexampleisvalidbycallingthe`check_text_data`function.
Ifthereisanyerrorduringthedownloadprocess,returnsNone.
Returnsthepreprocessedinputswithtransformedimageandtextdata.
"""
assertlen(example)==1
example=example[0]

ifnotcheck_text_data(example[text_column]):
raiseValueError("Textdataisnotvalid")

text_input=tokenizer(
example[text_column],
return_tensors='pt',
**tokenizer_kwargs)

returntext_input


defprepare_calibration_data_text(dataloader,init_steps):
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
data.append(batch["input_ids"].to("cpu"))
returndata

..code::ipython3

%%skipnot$to_quantize.value

importlogging
importnncf

dataset=load_dataset("google-research-datasets/conceptual_captions",trust_remote_code=True)
train_dataset=dataset["train"].shuffle(seed=42)

dataloader_text=torch.utils.data.DataLoader(train_dataset,collate_fn=collate_fn_text,batch_size=1)
calibration_data_text=prepare_calibration_data_text(dataloader_text,50)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino
Fetching50samplesfortheinitialization...



..parsed-literal::

0%||0/50[00:00<?,?it/s]


Datasetwithimagedata
^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value


defget_pil_from_url(url):
"""
DownloadsandconvertsanimagefromaURLtoaPILImageobject.
"""
response=requests.get(url,verify=False,timeout=20)
image=Image.open(BytesIO(response.content))
returnimage.convert("RGB")


defcollate_fn_vision(example,image_column="image_url"):
"""
Preprocessesanexamplebyloadingandtransformingimagedata.
DownloadstheimagespecifiedbytheURLintheimage_columnbycallingthe`get_pil_from_url`function.
Ifthereisanyerrorduringthedownloadprocess,returnsNone.
Returnsthepreprocessedinputswithtransformedimageandtextdata.
"""
assertlen(example)==1
example=example[0]

url=example[image_column]
try:
image=get_pil_from_url(url)
h,w=image.size
ifh==1orw==1:
returnNone
exceptException:
returnNone

vision_input=processor(images=[image])
returnvision_input


defprepare_calibration_data_vis(dataloader,init_steps):
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
data.append(batch["pixel_values"].to("cpu"))
returndata

..code::ipython3

%%skipnot$to_quantize.value

dataset=load_dataset("google-research-datasets/conceptual_captions",trust_remote_code=True)
train_dataset=dataset["train"].shuffle(seed=42)

dataloader_vis=torch.utils.data.DataLoader(train_dataset,collate_fn=collate_fn_vision,batch_size=1)
calibration_data_vision=prepare_calibration_data_vis(dataloader_vis,50)


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

Quantizationoftextmodel
^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

int8_text_model_path="jina-clip-text_v1_int8.xml"

..code::ipython3

%%skipnot$to_quantize.value

iflen(calibration_data_text)==0:
raiseRuntimeError(
'Calibrationdatasetisempty.Pleasecheckinternetconnectionandtrytodownloadimagesmanually.'
)

ov_model_text=core.read_model(fp16_text_model_path)

calibration_dataset=nncf.Dataset(calibration_data_text)
quantized_model=nncf.quantize(
model=ov_model_text,
calibration_dataset=calibration_dataset
)
ov.save_model(quantized_model,int8_text_model_path)



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



Quantizationofimagemodel
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

int8_vision_model_path="jina-clip-vision_v1_int8.xml"

..code::ipython3

%%skipnot$to_quantize.value

iflen(calibration_data_vision)==0:
raiseRuntimeError(
'Calibrationdatasetisempty.Pleasecheckinternetconnectionandtrytodownloadimagesmanually.'
)

ov_model_vision=core.read_model(fp16_vision_model_path)

calibration_dataset=nncf.Dataset(calibration_data_vision)
quantized_model=nncf.quantize(
model=ov_model_vision,
calibration_dataset=calibration_dataset
)
ov.save_model(quantized_model,int8_vision_model_path)



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



..code::ipython3

%%skipnot$to_quantize.value

compiled_text_model_int8=core.compile_model(int8_text_model_path,device.value)
compiled_vision_model_int8=core.compile_model(int8_vision_model_path,device.value)

text_ov_res_int8=compiled_text_model_int8(text_inputs["input_ids"])
vis_ov_res_int8=compiled_vision_model_int8(vision_inputs["pixel_values"])

res=calc_simularity_softmax(vis_ov_res_int8[0],text_ov_res_int8[0])
visionize_result(img_furseal,TEXT_INPUTS,np.array(res[0]))



..image::jina-clip-with-output_files/jina-clip-with-output_39_0.png


CompareFileSize
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

frompathlibimportPath

fp16_ir_model_size=Path(fp16_text_model_path).with_suffix(".bin").stat().st_size/1024/1024
quantized_model_size=Path(int8_text_model_path).with_suffix(".bin").stat().st_size/1024/1024
print(
f"Textmodel:FP16modelsize-{fp16_ir_model_size:.2f}MB;INT8modelsize-{quantized_model_size:.2f}MB;Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}"
)


fp16_ir_model_size=Path(fp16_vision_model_path).with_suffix(".bin").stat().st_size/1024/1024
quantized_model_size=Path(int8_vision_model_path).with_suffix(".bin").stat().st_size/1024/1024
print(
f"Visionmodel:FP16modelsize-{fp16_ir_model_size:.2f}MB;INT8modelsize-{quantized_model_size:.2f}MB;Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}"
)


..parsed-literal::

Textmodel:FP16modelsize-266.88MB;INT8modelsize-136.98MB;Modelcompressionrate:1.948
Visionmodel:FP16modelsize-163.83MB;INT8modelsize-82.64MB;Modelcompressionrate:1.983


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

fp16_latency=calculate_inference_time(fp16_text_model_path,calibration_data_text)
int8_latency=calculate_inference_time(int8_text_model_path,calibration_data_text)
print(f"Performancespeedupfortextmodel:{fp16_latency/int8_latency:.3f}")


fp16_latency=calculate_inference_time(fp16_vision_model_path,calibration_data_vision)
int8_latency=calculate_inference_time(int8_vision_model_path,calibration_data_vision)
print(f"Performancespeedupforvisionmodel:{fp16_latency/int8_latency:.3f}")


..parsed-literal::

Performancespeedupfortextmodel:1.539
Performancespeedupforvisionmodel:1.509


Gradiodemo
-----------

`backtotop⬆️<#table-of-contents>`__

Youcanprovideyourownimageandcomma-separatedlistoflabelsfor
zero-shotclassification.

Feelfreetouploadanimage,usingthefileuploadwindowandtype
labelnamesintothetextfield,usingcommaastheseparator(for
example,``cat,dog,bird``)

..code::ipython3

importgradioasgr

core=ov.Core()

compiled_text_model_int8=None
compiled_vision_model_int8=None
ifPath(int8_text_model_path).existsandPath(int8_vision_model_path).exists:
compiled_text_model_int8=core.compile_model(int8_text_model_path,device.value)
compiled_vision_model_int8=core.compile_model(int8_vision_model_path,device.value)

compiled_text_model_f16=core.compile_model(fp16_text_model_path,device.value)
compiled_vision_model_f16=core.compile_model(fp16_vision_model_path,device.value)


defimage_text_sim(text,image,quantized_model):
compiled_text_model=compiled_text_model_int8ifquantized_modelelsecompiled_text_model_f16
text=text.split(",")
text_inputs=tokenizer(text,return_tensors="pt",**tokenizer_kwargs)
emb1_res=compiled_text_model(text_inputs["input_ids"])

compiled_vision_model=compiled_vision_model_int8ifquantized_modelelsecompiled_vision_model_f16
vision_input=processor(images=[image])
emb2_res=compiled_vision_model(vision_input["pixel_values"])

text_description="Simularity:"
simularity=calc_simularity_softmax(emb2_res[0],emb1_res[0],False)
iflen(text)==1:
text_description+=f"{simularity[0]}"
else:
simularity_text="\n".join([f"{text[i]}{sim:.4f}"fori,siminenumerate(simularity[0])])
text_description+=f"\n{simularity_text}"
returntext_description


deftext_text_sim(text1,text2,quantized_model):
compiled_text_model=compiled_text_model_int8ifquantized_modelelsecompiled_text_model_f16

text_inputs=tokenizer(text1,return_tensors="pt",**tokenizer_kwargs)
emb1_res=compiled_text_model(text_inputs["input_ids"])

text_inputs=tokenizer(text2,return_tensors="pt",**tokenizer_kwargs)
emb2_res=compiled_text_model(text_inputs["input_ids"])

returnf"Simularity:{calc_simularity_softmax(emb1_res[0],emb2_res[0],False)[0][0]:.4f}"


defimage_image_sim(image1,image2,quantized_model):
compiled_vision_model=compiled_vision_model_int8ifquantized_modelelsecompiled_vision_model_f16

vision_input=processor(images=[image1])
emb1_res=compiled_vision_model(vision_input["pixel_values"])

vision_input=processor(images=[image2])
emb2_res=compiled_vision_model(vision_input["pixel_values"])

returnf"Simularity:{calc_simularity_softmax(emb1_res[0],emb2_res[0],False)[0][0]:.4f}"


withgr.Blocks()asdemo:
gr.Markdown("Discoversimularityoftextorimagefilesusingthisdemo.")
model_choice_visible=Path(int8_text_model_path).existsandPath(int8_vision_model_path).exists
quantized_model=gr.Checkbox(
label="Usequantizedint8model",info="Modeltype.FP16modelisusedbydefault.",visible=model_choice_visible,value=False
)
withgr.Tab("Text-Image"):
withgr.Row():
image_text_vis=gr.Image(label="Image",type="pil")
text_text_vis=gr.Textbox(label="Labels",info="Usecommatoseparatesentences")
text_image_button=gr.Button("Submit")
withgr.Row():
gr.Examples([img_furseal],image_text_vis)
gr.Examples(["seal,rat,cobra"],text_text_vis)
text_image_output=gr.Textbox(label="Results")
withgr.Tab("Text-Text"):
withgr.Row():
text_text_1=gr.Textbox(label="Text")
text_text_2=gr.Textbox(label="Text")
text_text_button=gr.Button("Submit")
withgr.Row():
gr.Examples(["ThebreedingseasonforfursealsisfromMaytotheendofNovember"],text_text_1)
gr.Examples(["Fursealsfeedonfishandsquid"],text_text_2)
text_text_output=gr.Textbox(label="Results")
withgr.Tab("Image-Image"):
withgr.Row():
image_image_1=gr.Image(label="Image",type="pil")
image_image_2=gr.Image(label="Image",type="pil")
image_image_button=gr.Button("Submit")
text_output=gr.Textbox(label="Results")
withgr.Row():
gr.Examples([img_furseal],image_image_1)
gr.Examples([img_coco],image_image_2)
image_image_output=gr.Textbox(label="Results")

text_image_button.click(image_text_sim,inputs=[text_text_vis,image_text_vis,quantized_model],outputs=text_image_output)
text_text_button.click(text_text_sim,inputs=[text_text_1,text_text_2,quantized_model],outputs=text_text_output)
image_image_button.click(image_image_sim,inputs=[image_image_1,image_image_2,quantized_model],outputs=image_image_output)


if__name__=="__main__":
try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>

