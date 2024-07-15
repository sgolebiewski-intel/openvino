ConvertandOptimizeYOLOv10withOpenVINO
==========================================

Real-timeobjectdetectionaimstoaccuratelypredictobjectcategories
andpositionsinimageswithlowlatency.TheYOLOserieshasbeenat
theforefrontofthisresearchduetoitsbalancebetweenperformance
andefficiency.However,relianceonNMSandarchitectural
inefficiencieshavehinderedoptimalperformance.YOLOv10addresses
theseissuesbyintroducingconsistentdualassignmentsforNMS-free
trainingandaholisticefficiency-accuracydrivenmodeldesign
strategy.

YOLOv10,builtonthe`UltralyticsPython
package<https://pypi.org/project/ultralytics/>`__byresearchersat
`TsinghuaUniversity<https://www.tsinghua.edu.cn/en/>`__,introducesa
newapproachtoreal-timeobjectdetection,addressingboththe
post-processingandmodelarchitecturedeficienciesfoundinprevious
YOLOversions.Byeliminatingnon-maximumsuppression(NMS)and
optimizingvariousmodelcomponents,YOLOv10achievesstate-of-the-art
performancewithsignificantlyreducedcomputationaloverhead.Extensive
experimentsdemonstrateitssuperioraccuracy-latencytrade-offsacross
multiplemodelscales.

..figure::https://github.com/ultralytics/ultralytics/assets/26833433/f9b1bec0-928e-41ce-a205-e12db3c4929a
:alt:yolov10-approach.png

yolov10-approach.png

Moredetailsaboutmodelarchitectureyoucanfindinoriginal
`repo<https://github.com/THU-MIG/yolov10>`__,
`paper<https://arxiv.org/abs/2405.14458>`__and`Ultralytics
documentation<https://docs.ultralytics.com/models/yolov10/>`__.

Thistutorialdemonstratesstep-by-stepinstructionsonhowtorunand
optimizePyTorchYOLOV10withOpenVINO.

Thetutorialconsistsofthefollowingsteps:

-PreparePyTorchmodel
-ConvertPyTorchmodeltoOpenVINOIR
-RunmodelinferencewithOpenVINO
-PrepareandrunoptimizationpipelineusingNNCF
-CompareperformanceoftheFP16andquantizedmodels.
-Runoptimizedmodelinferenceonvideo
-LaunchinteractiveGradiodemo

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`DownloadPyTorchmodel<#download-pytorch-model>`__
-`ExportPyTorchmodeltoOpenVINOIR
Format<#export-pytorch-model-to-openvino-ir-format>`__
-`RunOpenVINOInferenceonAUTOdeviceusingUltralytics
API<#run-openvino-inference-on-auto-device-using-ultralytics-api>`__
-`RunOpenVINOInferenceonselecteddeviceusingUltralytics
API<#run-openvino-inference-on-selected-device-using-ultralytics-api>`__
-`OptimizemodelusingNNCFPost-trainingQuantization
API<#optimize-model-using-nncf-post-training-quantization-api>`__

-`PrepareQuantizationDataset<#prepare-quantization-dataset>`__
-`QuantizeandSaveINT8model<#quantize-and-save-int8-model>`__

-`RunOptimizedModelInference<#run-optimized-model-inference>`__

-`RunOptimizedModelonAUTO
device<#run-optimized-model-on-auto-device>`__
-`RunOptimizedModelInferenceonselected
device<#run-optimized-model-inference-on-selected-device>`__

-`ComparetheOriginalandQuantized
Models<#compare-the-original-and-quantized-models>`__

-`Modelsize<#model-size>`__
-`Performance<#performance>`__
-`FP16modelperformance<#fp16-model-performance>`__
-`Int8modelperformance<#int8-model-performance>`__

-`Livedemo<#live-demo>`__

-`GradioInteractiveDemo<#gradio-interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importos

os.environ["GIT_CLONE_PROTECTION_ACTIVE"]="false"

%pipinstall-q"nncf>=2.11.0"
%pipinstall--pre-Uqopenvino--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly
%pipinstall-q"git+https://github.com/THU-MIG/yolov10.git"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"torch>=2.1""torchvision>=0.16"tqdmopencv-python"gradio>=4.19"--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

WARNING:Skippingopenvinoasitisnotinstalled.
WARNING:Skippingopenvino-devasitisnotinstalled.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file,VideoPlayer

DownloadPyTorchmodel
----------------------

`backtotop⬆️<#table-of-contents>`__

Thereareseveralversionof`YOLO
V10<https://github.com/THU-MIG/yolov10/tree/main?tab=readme-ov-file#performance>`__
modelsprovidedbymodelauthors.Eachofthemhasdifferent
characteristicsdependsonnumberoftrainingparameters,performance
andaccuracy.Fordemonstrationpurposeswewilluse``yolov10n``,but
thesamestepsarealsoapplicabletoothermodelsinYOLOV10series.

..code::ipython3

models_dir=Path("./models")
models_dir.mkdir(exist_ok=True)

..code::ipython3

model_weights_url="https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt"
file_name=model_weights_url.split("/")[-1]
model_name=file_name.replace(".pt","")

download_file(model_weights_url,directory=models_dir)


..parsed-literal::

'models/yolov10n.pt'alreadyexists.




..parsed-literal::

PosixPath('/home/ea/work/openvino_notebooks_new_clone/openvino_notebooks/notebooks/yolov10-optimization/models/yolov10n.pt')



ExportPyTorchmodeltoOpenVINOIRFormat
------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Asitwasdiscussedbefore,YOLOV10codeisdesignedontopof
`Ultralytics<https://docs.ultralytics.com/>`__libraryandhassimilar
interfacewithYOLOV8(Youcancheck`YOLOV8
notebooks<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization>`__
formoredetailedinstructionhowtoworkwithUltralyticsAPI).
UltralyticssupportOpenVINOmodelexportusing
`export<https://docs.ultralytics.com/modes/export/>`__methodofmodel
class.Additionally,wecanspecifyparametersresponsiblefortarget
inputsize,staticordynamicinputshapesandmodelprecision
(FP32/FP16/INT8).INT8quantizationcanbeadditionallyperformedon
exportstage,butformakingapproachmoreflexible,weconsiderhowto
performquantizationusing
`NNCF<https://github.com/openvinotoolkit/nncf>`__.

..code::ipython3

importtypes
fromultralytics.utilsimportops,yaml_load,yaml_save
fromultralyticsimportYOLOv10
importtorch

detection_labels={
0:"person",
1:"bicycle",
2:"car",
3:"motorcycle",
4:"airplane",
5:"bus",
6:"train",
7:"truck",
8:"boat",
9:"trafficlight",
10:"firehydrant",
11:"stopsign",
12:"parkingmeter",
13:"bench",
14:"bird",
15:"cat",
16:"dog",
17:"horse",
18:"sheep",
19:"cow",
20:"elephant",
21:"bear",
22:"zebra",
23:"giraffe",
24:"backpack",
25:"umbrella",
26:"handbag",
27:"tie",
28:"suitcase",
29:"frisbee",
30:"skis",
31:"snowboard",
32:"sportsball",
33:"kite",
34:"baseballbat",
35:"baseballglove",
36:"skateboard",
37:"surfboard",
38:"tennisracket",
39:"bottle",
40:"wineglass",
41:"cup",
42:"fork",
43:"knife",
44:"spoon",
45:"bowl",
46:"banana",
47:"apple",
48:"sandwich",
49:"orange",
50:"broccoli",
51:"carrot",
52:"hotdog",
53:"pizza",
54:"donut",
55:"cake",
56:"chair",
57:"couch",
58:"pottedplant",
59:"bed",
60:"diningtable",
61:"toilet",
62:"tv",
63:"laptop",
64:"mouse",
65:"remote",
66:"keyboard",
67:"cellphone",
68:"microwave",
69:"oven",
70:"toaster",
71:"sink",
72:"refrigerator",
73:"book",
74:"clock",
75:"vase",
76:"scissors",
77:"teddybear",
78:"hairdrier",
79:"toothbrush",
}


defv10_det_head_forward(self,x):
one2one=self.forward_feat([xi.detach()forxiinx],self.one2one_cv2,self.one2one_cv3)
ifnotself.export:
one2many=super().forward(x)

ifnotself.training:
one2one=self.inference(one2one)
ifnotself.export:
return{"one2many":one2many,"one2one":one2one}
else:
assertself.max_det!=-1
boxes,scores,labels=ops.v10postprocess(one2one.permute(0,2,1),self.max_det,self.nc)
returntorch.cat(
[boxes,scores.unsqueeze(-1),labels.unsqueeze(-1).to(boxes.dtype)],
dim=-1,
)
else:
return{"one2many":one2many,"one2one":one2one}


ov_model_path=models_dir/f"{model_name}_openvino_model/{model_name}.xml"
ifnotov_model_path.exists():
model=YOLOv10(models_dir/file_name)
model.model.model[-1].forward=types.MethodType(v10_det_head_forward,model.model.model[-1])
model.export(format="openvino",dynamic=True,half=True)
config=yaml_load(ov_model_path.parent/"metadata.yaml")
config["names"]=detection_labels
yaml_save(ov_model_path.parent/"metadata.yaml",config)

RunOpenVINOInferenceonAUTOdeviceusingUltralyticsAPI
-----------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Now,whenweexportedmodeltoOpenVINO,wecanloaditdirectlyinto
YOLOv10class,whereautomaticinferencebackendwillprovide
easy-to-useuserexperiencetorunOpenVINOYOLOv10modelonthesimilar
levellikefororiginalPyTorchmodel.Thecodebellowdemonstrateshow
toruninferenceOpenVINOexportedmodelwithUltralyticsAPIonsingle
image.`AUTO
device<https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/auto-device>`__
willbeusedforlaunchingmodel.

..code::ipython3

ov_yolo_model=YOLOv10(ov_model_path.parent,task="detect")

..code::ipython3

fromPILimportImage

IMAGE_PATH=Path("./data/coco_bike.jpg")
download_file(
url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg",
filename=IMAGE_PATH.name,
directory=IMAGE_PATH.parent,
)


..parsed-literal::

'data/coco_bike.jpg'alreadyexists.




..parsed-literal::

PosixPath('/home/ea/work/openvino_notebooks_new_clone/openvino_notebooks/notebooks/yolov10-optimization/data/coco_bike.jpg')



..code::ipython3

res=ov_yolo_model(IMAGE_PATH,iou=0.45,conf=0.2)
Image.fromarray(res[0].plot()[:,:,::-1])


..parsed-literal::

Loadingmodels/yolov10n_openvino_modelforOpenVINOinference...
requirements:Ultralyticsrequirement['openvino>=2024.0.0']notfound,attemptingAutoUpdate...
requirements:❌AutoUpdateskipped(offline)
UsingOpenVINOLATENCYmodeforbatch=1inference...

image1/1/home/ea/work/openvino_notebooks_new_clone/openvino_notebooks/notebooks/yolov10-optimization/data/coco_bike.jpg:640x6401bicycle,2cars,1motorcycle,1dog,72.0ms
Speed:25.6mspreprocess,72.0msinference,0.6mspostprocessperimageatshape(1,3,640,640)




..image::yolov10-optimization-with-output_files/yolov10-optimization-with-output_13_1.png



RunOpenVINOInferenceonselecteddeviceusingUltralyticsAPI
---------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Inthispartofnotebookyoucanselectinferencedeviceforrunning
modelinferencetocompareresultswithAUTO.

..code::ipython3

importopenvinoasov

importipywidgetsaswidgets

core=ov.Core()

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

ov_model=core.read_model(ov_model_path)

#loadmodelonselecteddevice
if"GPU"indevice.valueor"NPU"indevice.value:
ov_model.reshape({0:[1,3,640,640]})
ov_config={}
if"GPU"indevice.value:
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}
det_compiled_model=core.compile_model(ov_model,device.value,ov_config)

..code::ipython3

ov_yolo_model.predictor.model.ov_compiled_model=det_compiled_model

..code::ipython3

res=ov_yolo_model(IMAGE_PATH,iou=0.45,conf=0.2)


..parsed-literal::


image1/1/home/ea/work/openvino_notebooks_new_clone/openvino_notebooks/notebooks/yolov10-optimization/data/coco_bike.jpg:640x6401bicycle,2cars,1motorcycle,1dog,29.1ms
Speed:3.2mspreprocess,29.1msinference,0.3mspostprocessperimageatshape(1,3,640,640)


..code::ipython3

Image.fromarray(res[0].plot()[:,:,::-1])




..image::yolov10-optimization-with-output_files/yolov10-optimization-with-output_19_0.png



OptimizemodelusingNNCFPost-trainingQuantizationAPI
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.Wewilluse8-bitquantizationin
post-trainingmode(withoutthefine-tuningpipeline)tooptimize
YOLOv10.

Theoptimizationprocesscontainsthefollowingsteps:

1.CreateaDatasetforquantization.
2.Run``nncf.quantize``forgettinganoptimizedmodel.
3.SerializeOpenVINOIRmodel,usingthe``openvino.save_model``
function.

Quantizationistimeandmemoryconsumingprocess,youcanskipthis
stepusingcheckboxbellow:

..code::ipython3

importipywidgetsaswidgets

int8_model_det_path=models_dir/"int8"/f"{model_name}_openvino_model/{model_name}.xml"
ov_yolo_int8_model=None

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

PrepareQuantizationDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Forstartingquantization,weneedtopreparedataset.Wewilluse
validationsubsetfrom`MSCOCOdataset<https://cocodataset.org/>`__
formodelquantizationandUltralyticsvalidationdataloaderfor
preparinginputdata.

..code::ipython3

%%skipnot$to_quantize.value

fromzipfileimportZipFile

fromultralytics.data.utilsimportDATASETS_DIR

ifnotint8_model_det_path.exists():

DATA_URL="http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
CFG_URL="https://raw.githubusercontent.com/ultralytics/ultralytics/v8.1.0/ultralytics/cfg/datasets/coco.yaml"

OUT_DIR=DATASETS_DIR

DATA_PATH=OUT_DIR/"val2017.zip"
LABELS_PATH=OUT_DIR/"coco2017labels-segments.zip"
CFG_PATH=OUT_DIR/"coco.yaml"

download_file(DATA_URL,DATA_PATH.name,DATA_PATH.parent)
download_file(LABELS_URL,LABELS_PATH.name,LABELS_PATH.parent)
download_file(CFG_URL,CFG_PATH.name,CFG_PATH.parent)

ifnot(OUT_DIR/"coco/labels").exists():
withZipFile(LABELS_PATH,"r")aszip_ref:
zip_ref.extractall(OUT_DIR)
withZipFile(DATA_PATH,"r")aszip_ref:
zip_ref.extractall(OUT_DIR/"coco/images")

..code::ipython3

%%skipnot$to_quantize.value

fromultralytics.utilsimportDEFAULT_CFG
fromultralytics.cfgimportget_cfg
fromultralytics.data.converterimportcoco80_to_coco91_class
fromultralytics.data.utilsimportcheck_det_dataset

ifnotint8_model_det_path.exists():
args=get_cfg(cfg=DEFAULT_CFG)
args.data=str(CFG_PATH)
det_validator=ov_yolo_model.task_map[ov_yolo_model.task]["validator"](args=args)

det_validator.data=check_det_dataset(args.data)
det_validator.stride=32
det_data_loader=det_validator.get_dataloader(OUT_DIR/"coco",1)

NNCFprovides``nncf.Dataset``wrapperforusingnativeframework
dataloadersinquantizationpipeline.Additionally,wespecifytransform
functionthatwillberesponsibleforpreparinginputdatainmodel
expectedformat.

..code::ipython3

%%skipnot$to_quantize.value

importnncf
fromtypingimportDict


deftransform_fn(data_item:Dict):
"""
Quantizationtransformfunction.Extractsandpreprocessinputdatafromdataloaderitemforquantization.
Parameters:
data_item:DictwithdataitemproducedbyDataLoaderduringiteration
Returns:
input_tensor:Inputdataforquantization
"""
input_tensor=det_validator.preprocess(data_item)['img'].numpy()
returninput_tensor

ifnotint8_model_det_path.exists():
quantization_dataset=nncf.Dataset(det_data_loader,transform_fn)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,openvino


QuantizeandSaveINT8model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The``nncf.quantize``functionprovidesaninterfaceformodel
quantization.ItrequiresaninstanceoftheOpenVINOModeland
quantizationdataset.Optionally,someadditionalparametersforthe
configurationquantizationprocess(numberofsamplesforquantization,
preset,ignoredscope,etc.)canbeprovided.YOLOv10modelcontains
non-ReLUactivationfunctions,whichrequireasymmetricquantizationof
activations.Toachieveabetterresult,wewillusea``mixed``
quantizationpreset.Itprovidessymmetricquantizationofweightsand
asymmetricquantizationofactivations.

**Note**:Modelpost-trainingquantizationistime-consumingprocess.
Bepatient,itcantakeseveralminutesdependingonyourhardware.

..code::ipython3

%%skipnot$to_quantize.value

importshutil

ifnotint8_model_det_path.exists():
quantized_det_model=nncf.quantize(
ov_model,
quantization_dataset,
preset=nncf.QuantizationPreset.MIXED,
)

ov.save_model(quantized_det_model,int8_model_det_path)
shutil.copy(ov_model_path.parent/"metadata.yaml",int8_model_det_path.parent/"metadata.yaml")

RunOptimizedModelInference
-----------------------------

`backtotop⬆️<#table-of-contents>`__

ThewayofusageINT8quantizedmodelisthesamelikeformodelbefore
quantization.Let’scheckinferenceresultofquantizedmodelonsingle
image

RunOptimizedModelonAUTOdevice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value
ov_yolo_int8_model=YOLOv10(int8_model_det_path.parent,task="detect")

..code::ipython3

%%skipnot$to_quantize.value
res=ov_yolo_int8_model(IMAGE_PATH,iou=0.45,conf=0.2)


..parsed-literal::

Loadingmodels/int8/yolov10n_openvino_modelforOpenVINOinference...
requirements:Ultralyticsrequirement['openvino>=2024.0.0']notfound,attemptingAutoUpdate...
requirements:❌AutoUpdateskipped(offline)
UsingOpenVINOLATENCYmodeforbatch=1inference...

image1/1/home/ea/work/openvino_notebooks_new_clone/openvino_notebooks/notebooks/yolov10-optimization/data/coco_bike.jpg:640x6401bicycle,3cars,2motorcycles,1dog,92.3ms
Speed:3.7mspreprocess,92.3msinference,0.4mspostprocessperimageatshape(1,3,640,640)


..code::ipython3

Image.fromarray(res[0].plot()[:,:,::-1])




..image::yolov10-optimization-with-output_files/yolov10-optimization-with-output_34_0.png



RunOptimizedModelInferenceonselecteddevice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

device

..code::ipython3

%%skipnot$to_quantize.value

ov_config={}
if"GPU"indevice.valueor"NPU"indevice.value:
ov_model.reshape({0:[1,3,640,640]})
ov_config={}
if"GPU"indevice.value:
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}

quantized_det_model=core.read_model(int8_model_det_path)
quantized_det_compiled_model=core.compile_model(quantized_det_model,device.value,ov_config)

ov_yolo_int8_model.predictor.model.ov_compiled_model=quantized_det_compiled_model

res=ov_yolo_int8_model(IMAGE_PATH,iou=0.45,conf=0.2)


..parsed-literal::


image1/1/home/ea/work/openvino_notebooks_new_clone/openvino_notebooks/notebooks/yolov10-optimization/data/coco_bike.jpg:640x6401bicycle,3cars,2motorcycles,1dog,26.5ms
Speed:7.4mspreprocess,26.5msinference,0.3mspostprocessperimageatshape(1,3,640,640)


..code::ipython3

Image.fromarray(res[0].plot()[:,:,::-1])




..image::yolov10-optimization-with-output_files/yolov10-optimization-with-output_38_0.png



ComparetheOriginalandQuantizedModels
-----------------------------------------

`backtotop⬆️<#table-of-contents>`__

Modelsize
~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ov_model_weights=ov_model_path.with_suffix(".bin")
print(f"SizeofFP16modelis{ov_model_weights.stat().st_size/1024/1024:.2f}MB")
ifint8_model_det_path.exists():
ov_int8_weights=int8_model_det_path.with_suffix(".bin")
print(f"SizeofmodelwithINT8compressedweightsis{ov_int8_weights.stat().st_size/1024/1024:.2f}MB")
print(f"CompressionrateforINT8model:{ov_model_weights.stat().st_size/ov_int8_weights.stat().st_size:.3f}")


..parsed-literal::

SizeofFP16modelis4.39MB
SizeofmodelwithINT8compressedweightsis2.25MB
CompressionrateforINT8model:1.954


Performance
~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

FP16modelperformance
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

!benchmark_app-m$ov_model_path-d$device.value-apiasync-shape"[1,3,640,640]"-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.2.0-15496-17f8e86e5f2-releases/2024/2
[INFO]
[INFO]Deviceinfo:
[INFO]CPU
[INFO]Build.................................2024.2.0-15496-17f8e86e5f2-releases/2024/2
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(CPU)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook31.92ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.23/aten::cat/Concat_8):f32/[...]/[?,300,6]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook17.77ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.23/aten::cat/Concat_8):f32/[...]/[1,300,6]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook303.83ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]NUM_STREAMS:12
[INFO]INFERENCE_NUM_THREADS:36
[INFO]PERF_COUNT:NO
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]LOG_LEVEL:Level.NO
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]AFFINITY:Affinity.CORE
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,12inferencerequests,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook30.60ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:2424iterations
[INFO]Duration:15093.22ms
[INFO]Latency:
[INFO]Median:72.34ms
[INFO]Average:74.46ms
[INFO]Min:45.87ms
[INFO]Max:147.25ms
[INFO]Throughput:160.60FPS


Int8modelperformance
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifint8_model_det_path.exists():
!benchmark_app-m$int8_model_det_path-d$device.value-apiasync-shape"[1,3,640,640]"-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.2.0-15496-17f8e86e5f2-releases/2024/2
[INFO]
[INFO]Deviceinfo:
[INFO]CPU
[INFO]Build.................................2024.2.0-15496-17f8e86e5f2-releases/2024/2
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(CPU)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook38.75ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[?,3,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.23/aten::cat/Concat_8):f32/[...]/[?,300,6]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,640,640]
[INFO]Reshapemodeltook18.33ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,640,640]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.model.23/aten::cat/Concat_8):f32/[...]/[1,300,6]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook622.99ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:18
[INFO]NUM_STREAMS:18
[INFO]INFERENCE_NUM_THREADS:36
[INFO]PERF_COUNT:NO
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]ENABLE_CPU_PINNING:True
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]LOG_LEVEL:Level.NO
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:0
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]AFFINITY:Affinity.CORE
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,18inferencerequests,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook28.26ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:5886iterations
[INFO]Duration:15067.10ms
[INFO]Latency:
[INFO]Median:44.39ms
[INFO]Average:45.89ms
[INFO]Min:29.73ms
[INFO]Max:110.52ms
[INFO]Throughput:390.65FPS


Livedemo
---------

`backtotop⬆️<#table-of-contents>`__

Thefollowingcoderunsmodelinferenceonavideo:

..code::ipython3

importcollections
importtime
fromIPythonimportdisplay
importcv2
importnumpyasnp


#Mainprocessingfunctiontorunobjectdetection.
defrun_object_detection(
source=0,
flip=False,
use_popup=False,
skip_first_frames=0,
det_model=ov_yolo_int8_model,
device=device.value,
):
player=None
try:
#Createavideoplayertoplaywithtargetfps.
player=VideoPlayer(source=source,flip=flip,fps=30,skip_first_frames=skip_first_frames)
#Startcapturing.
player.start()
ifuse_popup:
title="PressESCtoExit"
cv2.namedWindow(winname=title,flags=cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE)

processing_times=collections.deque()
whileTrue:
#Grabtheframe.
frame=player.next()
ifframeisNone:
print("Sourceended")
break
#IftheframeislargerthanfullHD,reducesizetoimprovetheperformance.
scale=1280/max(frame.shape)
ifscale<1:
frame=cv2.resize(
src=frame,
dsize=None,
fx=scale,
fy=scale,
interpolation=cv2.INTER_AREA,
)
#Gettheresults.
input_image=np.array(frame)

start_time=time.time()
detections=det_model(input_image,iou=0.45,conf=0.2,verbose=False)
stop_time=time.time()
frame=detections[0].plot()

processing_times.append(stop_time-start_time)
#Useprocessingtimesfromlast200frames.
iflen(processing_times)>200:
processing_times.popleft()

_,f_width=frame.shape[:2]
#Meanprocessingtime[ms].
processing_time=np.mean(processing_times)*1000
fps=1000/processing_time
cv2.putText(
img=frame,
text=f"Inferencetime:{processing_time:.1f}ms({fps:.1f}FPS)",
org=(20,40),
fontFace=cv2.FONT_HERSHEY_COMPLEX,
fontScale=f_width/1000,
color=(0,0,255),
thickness=1,
lineType=cv2.LINE_AA,
)
#Usethisworkaroundifthereisflickering.
ifuse_popup:
cv2.imshow(winname=title,mat=frame)
key=cv2.waitKey(1)
#escape=27
ifkey==27:
break
else:
#Encodenumpyarraytojpg.
_,encoded_img=cv2.imencode(ext=".jpg",img=frame,params=[cv2.IMWRITE_JPEG_QUALITY,100])
#CreateanIPythonimage.
i=display.Image(data=encoded_img)
#Displaytheimageinthisnotebook.
display.clear_output(wait=True)
display.display(i)
#ctrl-c
exceptKeyboardInterrupt:
print("Interrupted")
#anydifferenterror
exceptRuntimeErrorase:
print(e)
finally:
ifplayerisnotNone:
#Stopcapturing.
player.stop()
ifuse_popup:
cv2.destroyAllWindows()

..code::ipython3

use_int8=widgets.Checkbox(
value=ov_yolo_int8_modelisnotNone,
description="Useint8model",
disabled=ov_yolo_int8_modelisNone,
)

use_int8




..parsed-literal::

Checkbox(value=True,description='Useint8model')



..code::ipython3

WEBCAM_INFERENCE=False

ifWEBCAM_INFERENCE:
VIDEO_SOURCE=0#Webcam
else:
download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4",
directory="data",
)
VIDEO_SOURCE="data/people.mp4"


..parsed-literal::

'data/people.mp4'alreadyexists.


..code::ipython3

run_object_detection(
det_model=ov_yolo_modelifnotuse_int8.valueelseov_yolo_int8_model,
source=VIDEO_SOURCE,
flip=True,
use_popup=False,
)



..image::yolov10-optimization-with-output_files/yolov10-optimization-with-output_50_0.png


..parsed-literal::

Sourceended


GradioInteractiveDemo
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr


defyolov10_inference(image,int8,conf_threshold,iou_threshold):
model=ov_yolo_modelifnotint8elseov_yolo_int8_model
results=model(source=image,iou=iou_threshold,conf=conf_threshold,verbose=False)[0]
annotated_image=Image.fromarray(results.plot())

returnannotated_image


withgr.Blocks()asdemo:
gr.HTML(
"""
<h1style='text-align:center'>
YOLOv10:Real-TimeEnd-to-EndObjectDetectionusingOpenVINO
</h1>
"""
)
withgr.Row():
withgr.Column():
image=gr.Image(type="numpy",label="Image")
conf_threshold=gr.Slider(
label="ConfidenceThreshold",
minimum=0.1,
maximum=1.0,
step=0.1,
value=0.2,
)
iou_threshold=gr.Slider(
label="IoUThreshold",
minimum=0.1,
maximum=1.0,
step=0.1,
value=0.45,
)
use_int8=gr.Checkbox(
value=ov_yolo_int8_modelisnotNone,
visible=ov_yolo_int8_modelisnotNone,
label="UseINT8model",
)
yolov10_infer=gr.Button(value="DetectObjects")

withgr.Column():
output_image=gr.Image(type="pil",label="AnnotatedImage")

yolov10_infer.click(
fn=yolov10_inference,
inputs=[
image,
use_int8,
conf_threshold,
iou_threshold,
],
outputs=[output_image],
)
examples=gr.Examples(
[
"data/coco_bike.jpg",
],
inputs=[
image,
],
)


try:
demo.launch(debug=False)
exceptException:
demo.launch(debug=False,share=True)
