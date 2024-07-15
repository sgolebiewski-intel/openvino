QuantizeaSegmentationModelandShowLiveInference
=====================================================

KidneySegmentationwithPyTorchLightningandOpenVINO™-Part3
-----------------------------------------------------------------

Thistutorialisapartofaseriesonhowtotrain,optimize,quantize
andshowliveinferenceonamedicalsegmentationmodel.Thegoalisto
accelerateinferenceonakidneysegmentationmodel.The
`UNet<https://arxiv.org/abs/1505.04597>`__modelistrainedfrom
scratch;thedataisfrom
`Kits19<https://github.com/neheller/kits19>`__.

Thisthirdtutorialintheseriesshowshowto:

-ConvertanOriginalmodeltoOpenVINOIRwith`modelconversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
-QuantizeaPyTorchmodelwithNNCF
-EvaluatetheF1scoremetricoftheoriginalmodelandthequantized
model
-BenchmarkperformanceoftheFP32modelandtheINT8quantizedmodel
-ShowliveinferencewithOpenVINO’sasyncAPI

Allnotebooksinthisseries:

-`DataPreparationfor2DSegmentationof3DMedical
Data<data-preparation-ct-scan.ipynb>`__
-`Traina2D-UNetMedicalImagingModelwithPyTorch
Lightning<pytorch-monai-training.ipynb>`__
-ConvertandQuantizeaSegmentationModelandShowLiveInference
(thisnotebook)
-`LiveInferenceandBenchmarkCT-scan
data<ct-scan-live-inference.ipynb>`__

Instructions
------------

ThisnotebookneedsatrainedUNetmodel.Weprovideapre-trained
model,trainedfor20epochswiththefull
`Kits-19<https://github.com/neheller/kits19>`__framesdataset,which
hasanF1scoreonthevalidationsetof0.9.Thetrainingcodeis
availablein`thisnotebook<pytorch-monai-training.ipynb>`__.

NNCFforPyTorchmodelsrequiresaC++compiler.OnWindows,install
`MicrosoftVisualStudio
2019<https://docs.microsoft.com/en-us/visualstudio/releases/2019/release-notes>`__.
Duringinstallation,chooseDesktopdevelopmentwithC++inthe
Workloadstab.OnmacOS,run``xcode-select–install``fromaTerminal.
OnLinux,install``gcc``.

Runningthisnotebookwiththefulldatasetwilltakealongtime.For
demonstrationpurposes,thistutorialwilldownloadoneconvertedCT
scanandusethatscanforquantizationandinference.Forproduction
purposes,usearepresentativedatasetforquantizingthemodel.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Settings<#settings>`__
-`LoadPyTorchModel<#load-pytorch-model>`__
-`DownloadCT-scanData<#download-ct-scan-data>`__
-`Configuration<#configuration>`__

-`Dataset<#dataset>`__
-`Metric<#metric>`__

-`Quantization<#quantization>`__
-`CompareFP32andINT8Model<#compare-fp32-and-int8-model>`__

-`CompareFileSize<#compare-file-size>`__
-`CompareMetricsfortheoriginalmodelandthequantizedmodelto
besurethatthereno
degradation.<#compare-metrics-for-the-original-model-and-the-quantized-model-to-be-sure-that-there-no-degradation->`__
-`ComparePerformanceoftheFP32IRModelandQuantized
Models<#compare-performance-of-the-fp32-ir-model-and-quantized-models>`__
-`VisuallyCompareInference
Results<#visually-compare-inference-results>`__

-`ShowLiveInference<#show-live-inference>`__

-`LoadModelandListofImage
Files<#load-model-and-list-of-image-files>`__
-`ShowInference<#show-inference>`__

-`References<#references>`__

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.3.0""monai>=0.9.1""torchmetrics>=0.11.0""nncf>=2.8.0""opencv-python"torchtqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importlogging
importos
importrandom
importtime
importwarnings
importzipfile
frompathlibimportPath
fromtypingimportUnion

warnings.filterwarnings("ignore",category=UserWarning)

importcv2
importmatplotlib.pyplotasplt
importmonai
importnumpyasnp
importtorch
importnncf
importopenvinoasov
frommonai.transformsimportLoadImage
fromnncf.common.logging.loggerimportset_log_level
fromtorchmetricsimportF1ScoreasF1
importrequests


set_log_level(logging.ERROR)#DisablesallNNCFinfoandwarningmessages

#Fetch`notebook_utils`module
r=requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

ifnotPath("./custom_segmentation.py").exists():
download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/ct-segmentation-quantize/custom_segmentation.py")
fromcustom_segmentationimportSegmentationModel

ifnotPath("./async_pipeline.py").exists():
download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/ct-segmentation-quantize/async_pipeline.py")
fromasync_pipelineimportshow_live_inference


..parsed-literal::

2024-07-1223:48:24.140201:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1223:48:24.175655:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1223:48:24.759330:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


Settings
--------

`backtotop⬆️<#table-of-contents>`__

Bydefault,thisnotebookwilldownloadoneCTscanfromtheKITS19
datasetthatwillbeusedforquantization.

..code::ipython3

BASEDIR=Path("kits19_frames_1")
#Uncommentthelinebelowtousethefulldataset,aspreparedinthedatapreparationnotebook
#BASEDIR=Path("~/kits19/kits19_frames").expanduser()
MODEL_DIR=Path("model")
MODEL_DIR.mkdir(exist_ok=True)

LoadPyTorchModel
------------------

`backtotop⬆️<#table-of-contents>`__

Downloadthepre-trainedmodelweights,loadthePyTorchmodelandthe
``state_dict``thatwassavedaftertraining.Themodelusedinthis
notebookisa
`BasicUNet<https://docs.monai.io/en/stable/networks.html#basicunet>`__
modelfrom`MONAI<https://monai.io>`__.Weprovideapre-trained
checkpoint.Toseehowthismodelperforms,checkoutthe`training
notebook<pytorch-monai-training.ipynb>`__.

..code::ipython3

state_dict_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/kidney-segmentation-kits19/unet_kits19_state_dict.pth"
state_dict_file=download_file(state_dict_url,directory="pretrained_model")
state_dict=torch.load(state_dict_file,map_location=torch.device("cpu"))

new_state_dict={}
fork,vinstate_dict.items():
new_key=k.replace("_model.","")
new_state_dict[new_key]=v
new_state_dict.pop("loss_function.pos_weight")

model=monai.networks.nets.BasicUNet(spatial_dims=2,in_channels=1,out_channels=1).eval()
model.load_state_dict(new_state_dict)



..parsed-literal::

pretrained_model/unet_kits19_state_dict.pth:0%||0.00/7.58M[00:00<?,?B/s]


..parsed-literal::

BasicUNetfeatures:(32,32,64,128,256,32).




..parsed-literal::

<Allkeysmatchedsuccessfully>



DownloadCT-scanData
---------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#TheCTscancasenumber.Forexample:2fordatafromthecase_00002directory
#Currentlyonly117issupported
CASE=117
ifnot(BASEDIR/f"case_{CASE:05d}").exists():
BASEDIR.mkdir(exist_ok=True)
filename=download_file(f"https://storage.openvinotoolkit.org/data/test_data/openvino_notebooks/kits19/case_{CASE:05d}.zip")
withzipfile.ZipFile(filename,"r")aszip_ref:
zip_ref.extractall(path=BASEDIR)
os.remove(filename)#removezipfile
print(f"Downloadedandextracteddataforcase_{CASE:05d}")
else:
print(f"Dataforcase_{CASE:05d}exists")



..parsed-literal::

case_00117.zip:0%||0.00/5.48M[00:00<?,?B/s]


..parsed-literal::

Downloadedandextracteddataforcase_00117


Configuration
-------------

`backtotop⬆️<#table-of-contents>`__

Dataset
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The``KitsDataset``classinthenextcellexpectsimagesandmasksin
the*``basedir``*directory,inafolderperpatient.Itisasimplified
versionoftheDatasetclassinthe`training
notebook<pytorch-monai-training.ipynb>`__.

ImagesareloadedwithMONAI’s
`LoadImage<https://docs.monai.io/en/stable/transforms.html#loadimage>`__,
toalignwiththeimageloadingmethodinthetrainingnotebook.This
methodrotatesandflipstheimages.Wedefinea``rotate_and_flip``
methodtodisplaytheimagesintheexpectedorientation:

..code::ipython3

defrotate_and_flip(image):
"""Rotate`image`by90degreesandfliphorizontally"""
returncv2.flip(cv2.rotate(image,rotateCode=cv2.ROTATE_90_CLOCKWISE),flipCode=1)


classKitsDataset:
def__init__(self,basedir:str):
"""
DatasetclassforpreparedKits19data,forbinarysegmentation(background/kidney)
Sourcedatashouldexistinbasedir,insubdirectoriescase_00000untilcase_00210,
witheachsubdirectorycontainingdirectoriesimaging_frames,withjpgimages,and
segmentation_frameswithsegmentationmasksaspngfiles.
See[data-preparation-ct-scan](./data-preparation-ct-scan.ipynb)

:parambasedir:DirectorythatcontainsthepreparedCTscans
"""
masks=sorted(BASEDIR.glob("case_*/segmentation_frames/*png"))

self.basedir=basedir
self.dataset=masks
print(f"Createddatasetwith{len(self.dataset)}items."f"Basedirectoryfordata:{basedir}")

def__getitem__(self,index):
"""
Getanitemfromthedatasetatthespecifiedindex.

:return:(image,segmentation_mask)
"""
mask_path=self.dataset[index]
image_path=str(mask_path.with_suffix(".jpg")).replace("segmentation_frames","imaging_frames")

#LoadimageswithMONAI'sLoadImagetomatchdataloadingintrainingnotebook
mask=LoadImage(image_only=True,dtype=np.uint8)(str(mask_path)).numpy()
img=LoadImage(image_only=True,dtype=np.float32)(str(image_path)).numpy()

ifimg.shape[:2]!=(512,512):
img=cv2.resize(img.astype(np.uint8),(512,512)).astype(np.float32)
mask=cv2.resize(mask,(512,512))

input_image=np.expand_dims(img,axis=0)
returninput_image,mask

def__len__(self):
returnlen(self.dataset)

Totestwhetherthedataloaderreturnstheexpectedoutput,weshowan
imageandamask.Theimageandthemaskarereturnedbythedataloader,
afterresizingandpreprocessing.Sincethisdatasetcontainsalotof
sliceswithoutkidneys,weselectaslicethatcontainsatleast5000
kidneypixelstoverifythattheannotationslookcorrect:

..code::ipython3

dataset=KitsDataset(BASEDIR)
#Findaslicethatcontainskidneyannotations
#item[0]istheannotation:(id,annotation_data)
image_data,mask=next(itemforitemindatasetifnp.count_nonzero(item[1])>5000)
#Removeextraimagedimensionandrotateandfliptheimageforvisualization
image=rotate_and_flip(image_data.squeeze())

#Thedataloaderreturnsannotationsas(index,mask)andmaskinshape(H,W)
mask=rotate_and_flip(mask)

fig,ax=plt.subplots(1,2,figsize=(12,6))
ax[0].imshow(image,cmap="gray")
ax[1].imshow(mask,cmap="gray");


..parsed-literal::

Createddatasetwith69items.Basedirectoryfordata:kits19_frames_1



..image::ct-segmentation-quantize-nncf-with-output_files/ct-segmentation-quantize-nncf-with-output_14_1.png


Metric
~~~~~~

`backtotop⬆️<#table-of-contents>`__

Defineametrictodeterminetheperformanceofthemodel.

Forthisdemo,weusethe`F1
score<https://en.wikipedia.org/wiki/F-score>`__,orDicecoefficient,
fromthe
`TorchMetrics<https://torchmetrics.readthedocs.io/en/stable/>`__
library.

..code::ipython3

defcompute_f1(model:Union[torch.nn.Module,ov.CompiledModel],dataset:KitsDataset):
"""
ComputebinaryF1scoreof`model`on`dataset`
F1scoremetricisprovidedbythetorchmetricslibrary
`model`isexpectedtobeabinarysegmentationmodel,imagesinthe
datasetareexpectedin(N,C,H,W)formatwhereN==C==1
"""
metric=F1(ignore_index=0,task="binary",average="macro")
withtorch.no_grad():
forimage,targetindataset:
input_image=torch.as_tensor(image).unsqueeze(0)
ifisinstance(model,ov.CompiledModel):
output_layer=model.output(0)
output=model(input_image)[output_layer]
output=torch.from_numpy(output)
else:
output=model(input_image)
label=torch.as_tensor(target.squeeze()).long()
prediction=torch.sigmoid(output.squeeze()).round().long()
metric.update(label.flatten(),prediction.flatten())
returnmetric.compute()

Quantization
------------

`backtotop⬆️<#table-of-contents>`__

Beforequantizingthemodel,wecomputetheF1scoreonthe``FP32``
model,forcomparison:

..code::ipython3

fp32_f1=compute_f1(model,dataset)
print(f"FP32F1:{fp32_f1:.3f}")


..parsed-literal::

FP32F1:0.999


WeconvertthePyTorchmodeltoOpenVINOIRandserializeitfor
comparingtheperformanceofthe``FP32``and``INT8``modellaterin
thisnotebook.

..code::ipython3

fp32_ir_path=MODEL_DIR/Path("unet_kits19_fp32.xml")

fp32_ir_model=ov.convert_model(model,example_input=torch.ones(1,1,512,512,dtype=torch.float32))
ov.save_model(fp32_ir_model,str(fp32_ir_path))


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/monai/networks/nets/basic_unet.py:168:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifx_e.shape[-i-1]!=x_0.shape[-i-1]:


`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.

**Note**:NNCFPost-trainingQuantizationisavailableinOpenVINO
2023.0release.

Createaquantizedmodelfromthepre-trained``FP32``modelandthe
calibrationdataset.Theoptimizationprocesscontainsthefollowing
steps:

::

1.CreateaDatasetforquantization.
2.Run`nncf.quantize`forgettinganoptimizedmodel.
3.ExportthequantizedmodeltoONNXandthenconverttoOpenVINOIRmodel.
4.SerializetheINT8modelusing`ov.save_model`functionforbenchmarking.

..code::ipython3

deftransform_fn(data_item):
"""
Extractthemodel'sinputfromthedataitem.
Thedataitemhereisthedataitemthatisreturnedfromthedatasourceperiteration.
Thisfunctionshouldbepassedwhenthedataitemcannotbeusedasmodel'sinput.
"""
images,_=data_item
returnimages


data_loader=torch.utils.data.DataLoader(dataset)
calibration_dataset=nncf.Dataset(data_loader,transform_fn)
quantized_model=nncf.quantize(
model,
calibration_dataset,
#DonotquantizeLeakyReLUactivationstoallowtheINT8modeltorunonIntelGPU
ignored_scope=nncf.IgnoredScope(patterns=[".*LeakyReLU.*"]),
)



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



ConvertquantizedmodeltoOpenVINOIRmodelandsaveit.

..code::ipython3

dummy_input=torch.randn(1,1,512,512)
int8_onnx_path=MODEL_DIR/"unet_kits19_int8.onnx"
int8_ir_path=Path(int8_onnx_path).with_suffix(".xml")
int8_ir_model=ov.convert_model(quantized_model,example_input=dummy_input,input=dummy_input.shape)
ov.save_model(int8_ir_model,str(int8_ir_path))


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/torch/quantization/layers.py:339:TracerWarning:ConvertingatensortoaPythonnumbermightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
returnself._level_low.item()
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/torch/quantization/layers.py:347:TracerWarning:ConvertingatensortoaPythonnumbermightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
returnself._level_high.item()
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/monai/networks/nets/basic_unet.py:168:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifx_e.shape[-i-1]!=x_0.shape[-i-1]:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:1116:TracerWarning:Outputnr1.ofthetracedfunctiondoesnotmatchthecorrespondingoutputofthePythonfunction.Detailederror:
Tensor-likesarenotclose!

Mismatchedelements:249132/262144(95.0%)
Greatestabsolutedifference:4.048818826675415atindex(0,0,237,507)(upto1e-05allowed)
Greatestrelativedifference:53465.01457340121atindex(0,0,334,396)(upto1e-05allowed)
_check_trace(


Thisnotebookdemonstratespost-trainingquantizationwithNNCF.

NNCFalsosupportsquantization-awaretraining,andotheralgorithms
thanquantization.Seethe`NNCF
documentation<https://github.com/openvinotoolkit/nncf/>`__intheNNCF
repositoryformoreinformation.

CompareFP32andINT8Model
---------------------------

`backtotop⬆️<#table-of-contents>`__

CompareFileSize
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fp32_ir_model_size=fp32_ir_path.with_suffix(".bin").stat().st_size/1024
quantized_model_size=int8_ir_path.with_suffix(".bin").stat().st_size/1024

print(f"FP32IRmodelsize:{fp32_ir_model_size:.2f}KB")
print(f"INT8modelsize:{quantized_model_size:.2f}KB")


..parsed-literal::

FP32IRmodelsize:3864.14KB
INT8modelsize:1953.48KB


SelectInferenceDevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

core=ov.Core()
#Bydefault,benchmarkonMULTI:CPU,GPUifaGPUisavailable,otherwiseonCPU.
device_list=["MULTI:CPU,GPU"if"GPU"incore.available_deviceselse"AUTO"]

importipywidgetsaswidgets

device=widgets.Dropdown(
options=core.available_devices+device_list,
value=device_list[0],
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



CompareMetricsfortheoriginalmodelandthequantizedmodeltobesurethattherenodegradation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

int8_compiled_model=core.compile_model(int8_ir_model,device.value)
int8_f1=compute_f1(int8_compiled_model,dataset)

print(f"FP32F1:{fp32_f1:.3f}")
print(f"INT8F1:{int8_f1:.3f}")


..parsed-literal::

FP32F1:0.999
INT8F1:0.999


ComparePerformanceoftheFP32IRModelandQuantizedModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tomeasuretheinferenceperformanceofthe``FP32``and``INT8``
models,weuse`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
-OpenVINO’sinferenceperformancemeasurementtool.Benchmarktoolisa
commandlineapplication,partofOpenVINOdevelopmenttools,thatcan
beruninthenotebookwith``!benchmark_app``or
``%sxbenchmark_app``.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.Run
``benchmark_app-mmodel.xml-dCPU``tobenchmarkasyncinferenceon
CPUforoneminute.Change``CPU``to``GPU``tobenchmarkonGPU.
Run``benchmark_app--help``toseeallcommandlineoptions.

..code::ipython3

#!benchmark_app--help

..code::ipython3

#BenchmarkFP32model
!benchmark_app-m$fp32_ir_path-d$device.value-t15-apisync


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.LATENCY.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook9.03ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,?,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.final_conv/aten::_convolution/Add):f32/[...]/[?,1,16..,16..]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,?,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.final_conv/aten::_convolution/Add):f32/[...]/[?,1,16..,16..]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook178.45ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.LATENCY
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:False
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:12
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:1
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]PERFORMANCE_HINT:LATENCY
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[ERROR]Inputxisdynamic.Providedatashapes!
Traceback(mostrecentcalllast):
File"/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/main.py",line486,inmain
data_queue=get_input_data(paths_to_input,app_inputs_info)
File"/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/utils/inputs_filling.py",line123,inget_input_data
raiseException(f"Input{info.name}isdynamic.Providedatashapes!")
Exception:Inputxisdynamic.Providedatashapes!


..code::ipython3

#BenchmarkINT8model
!benchmark_app-m$int8_ir_path-d$device.value-t15-apisync


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.LATENCY.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook10.91ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,1,512,512]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.final_conv/aten::_convolution/Add):f32/[...]/[1,1,512,512]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):f32/[N,C,H,W]/[1,1,512,512]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.final_conv/aten::_convolution/Add):f32/[...]/[1,1,512,512]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook229.32ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model49
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.LATENCY
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:False
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:12
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model49
[INFO]NUM_STREAMS:1
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:1
[INFO]PERFORMANCE_HINT:LATENCY
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferencesynchronously,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook30.15ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:964iterations
[INFO]Duration:15012.54ms
[INFO]Latency:
[INFO]Median:15.32ms
[INFO]Average:15.38ms
[INFO]Min:14.99ms
[INFO]Max:17.28ms
[INFO]Throughput:64.21FPS


VisuallyCompareInferenceResults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Visualizetheresultsofthemodelonfourslicesofthevalidationset.
Comparetheresultsofthe``FP32``IRmodelwiththeresultsofthe
quantized``INT8``modelandthereferencesegmentationannotation.

Medicalimagingdatasetstendtobeveryimbalanced:mostoftheslices
inaCTscandonotcontainkidneydata.Thesegmentationmodelshould
begoodatfindingkidneyswheretheyexist(inmedicalterms:havegood
sensitivity)butalsonotfindspuriouskidneysthatdonotexist(have
goodspecificity).Inthenextcell,therearefourslices:twoslices
thathavenokidneydata,andtwoslicesthatcontainkidneydata.For
thisexample,aslicehaskidneydataifatleast50pixelsinthe
slicesareannotatedaskidney.

Runthiscellagaintoshowresultsonadifferentsubset.Therandom
seedisdisplayedtoenablereproducingspecificrunsofthiscell.

**NOTE**:theimagesareshownafteroptionalaugmentingand
resizing.IntheKits19datasetallbutoneofthecaseshasthe
``(512,512)``inputshape.

..code::ipython3

#Thesigmoidfunctionisusedtotransformtheresultofthenetwork
#tobinarysegmentationmasks
defsigmoid(x):
returnnp.exp(-np.logaddexp(0,-x))


num_images=4
colormap="gray"

#LoadFP32andINT8models
core=ov.Core()
fp_model=core.read_model(fp32_ir_path)
int8_model=core.read_model(int8_ir_path)
compiled_model_fp=core.compile_model(fp_model,device_name=device.value)
compiled_model_int8=core.compile_model(int8_model,device_name=device.value)
output_layer_fp=compiled_model_fp.output(0)
output_layer_int8=compiled_model_int8.output(0)

#Createsubsetofdataset
background_slices=(itemforitemindatasetifnp.count_nonzero(item[1])==0)
kidney_slices=(itemforitemindatasetifnp.count_nonzero(item[1])>50)
data_subset=random.sample(list(background_slices),2)+random.sample(list(kidney_slices),2)

#Setseedtocurrenttime.Toreproducespecificresults,copytheprintedseed
#andmanuallyset`seed`tothatvalue.
seed=int(time.time())
random.seed(seed)
print(f"Visualizingresultswithseed{seed}")

fig,ax=plt.subplots(nrows=num_images,ncols=4,figsize=(24,num_images*4))
fori,(image,mask)inenumerate(data_subset):
display_image=rotate_and_flip(image.squeeze())
target_mask=rotate_and_flip(mask).astype(np.uint8)
#AddbatchdimensiontoimageanddoinferenceonFPandINT8models
input_image=np.expand_dims(image,0)
res_fp=compiled_model_fp([input_image])
res_int8=compiled_model_int8([input_image])

#Processinferenceoutputsandconverttobinarysegementationmasks
result_mask_fp=sigmoid(res_fp[output_layer_fp]).squeeze().round().astype(np.uint8)
result_mask_int8=sigmoid(res_int8[output_layer_int8]).squeeze().round().astype(np.uint8)
result_mask_fp=rotate_and_flip(result_mask_fp)
result_mask_int8=rotate_and_flip(result_mask_int8)

#Displayimages,annotations,FP32resultandINT8result
ax[i,0].imshow(display_image,cmap=colormap)
ax[i,1].imshow(target_mask,cmap=colormap)
ax[i,2].imshow(result_mask_fp,cmap=colormap)
ax[i,3].imshow(result_mask_int8,cmap=colormap)
ax[i,2].set_title("PredictiononFP32model")
ax[i,3].set_title("PredictiononINT8model")


..parsed-literal::

Visualizingresultswithseed1720820974



..image::ct-segmentation-quantize-nncf-with-output_files/ct-segmentation-quantize-nncf-with-output_37_1.png


ShowLiveInference
-------------------

`backtotop⬆️<#table-of-contents>`__

Toshowliveinferenceonthemodelinthenotebook,wewillusethe
asynchronousprocessingfeatureofOpenVINO.

Weusethe``show_live_inference``functionfrom`Notebook
Utils<../utils-with-output.html>`__toshowliveinference.This
functionuses`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__\’sAsync
PipelineandModelAPItoperformasynchronousinference.After
inferenceonthespecifiedCTscanhascompleted,thetotaltimeand
throughput(fps),includingpreprocessinganddisplaying,willbe
printed.

**NOTE**:IfyouexperienceflickeringonFirefox,considerusing
ChromeorEdgetorunthisnotebook.

LoadModelandListofImageFiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

WeloadthesegmentationmodeltoOpenVINORuntimewith
``SegmentationModel``,basedonthe`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__ModelAPI.
Thismodelimplementationincludespreandpostprocessingforthe
model.For``SegmentationModel``,thisincludesthecodetocreatean
overlayofthesegmentationmaskontheoriginalimage/frame.

..code::ipython3

CASE=117

segmentation_model=SegmentationModel(ie=core,model_path=int8_ir_path,sigmoid=True,rotate_and_flip=True)
case_path=BASEDIR/f"case_{CASE:05d}"
image_paths=sorted(case_path.glob("imaging_frames/*jpg"))
print(f"{case_path.name},{len(image_paths)}images")


..parsed-literal::

case_00117,69images


ShowInference
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Inthenextcell,werunthe``show_live_inference``function,which
loadsthe``segmentation_model``tothespecified``device``(using
cachingforfastermodelloadingonGPUdevices),loadstheimages,
performsinference,anddisplaystheresultsontheframesloadedin
``images``inreal-time.

..code::ipython3

reader=LoadImage(image_only=True,dtype=np.uint8)
show_live_inference(
ie=core,
image_paths=image_paths,
model=segmentation_model,
device=device.value,
reader=reader,
)



..image::ct-segmentation-quantize-nncf-with-output_files/ct-segmentation-quantize-nncf-with-output_42_0.jpg


..parsed-literal::

LoadedmodeltoAUTOin0.19seconds.
Totaltimefor68frames:2.39seconds,fps:28.85


References
----------

`backtotop⬆️<#table-of-contents>`__

**OpenVINO**-`NNCF
Repository<https://github.com/openvinotoolkit/nncf/>`__-`Neural
NetworkCompressionFrameworkforfastmodel
inference<https://arxiv.org/abs/2002.08679>`__-`OpenVINOAPI
Tutorial<openvino-api-with-output.html>`__-`OpenVINOPyPI(pip
installopenvino-dev)<https://pypi.org/project/openvino-dev/>`__

**Kits19Data**-`Kits19Challenge
Homepage<https://kits19.grand-challenge.org/>`__-`Kits19GitHub
Repository<https://github.com/neheller/kits19>`__-`TheKiTS19
ChallengeData:300KidneyTumorCaseswithClinicalContext,CT
SemanticSegmentations,andSurgical
Outcomes<https://arxiv.org/abs/1904.00445>`__-`Thestateoftheart
inkidneyandkidneytumorsegmentationincontrast-enhancedCTimaging:
ResultsoftheKiTS19
challenge<https://www.sciencedirect.com/science/article/pii/S1361841520301857>`__
