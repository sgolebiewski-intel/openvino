ObjectsegmentationswithFastSAMandOpenVINO
==============================================

`TheFastSegmentAnythingModel
(FastSAM)<https://docs.ultralytics.com/models/fast-sam/>`__isa
real-timeCNN-basedmodelthatcansegmentanyobjectwithinanimage
basedonvarioususerprompts.``SegmentAnything``taskisdesignedto
makevisiontaskseasierbyprovidinganefficientwaytoidentify
objectsinanimage.FastSAMsignificantlyreducescomputationaldemands
whilemaintainingcompetitiveperformance,makingitapracticalchoice
foravarietyofvisiontasks.

FastSAMisamodelthataimstoovercomethelimitationsofthe`Segment
AnythingModel(SAM)<https://docs.ultralytics.com/models/sam/>`__,
whichisaTransformermodelthatrequiressignificantcomputational
resources.FastSAMtacklesthesegmentanythingtaskbydividingitinto
twoconsecutivestages:all-instancesegmentationandprompt-guided
selection.

Inthefirststage,
`YOLOv8-seg<https://docs.ultralytics.com/tasks/segment/>`__isused
toproducesegmentationmasksforallinstancesintheimage.Inthe
secondstage,FastSAMoutputstheregion-of-interestcorrespondingto
theprompt.

..figure::https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg
:alt:pipeline

pipeline

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__

-`Installrequirements<#install-requirements>`__
-`Imports<#imports>`__

-`FastSAMinUltralytics<#fastsam-in-ultralytics>`__
-`ConvertthemodeltoOpenVINOIntermediaterepresentation(IR)
format<#convert-the-model-to-openvino-intermediate-representation-ir-format>`__
-`Embeddingtheconvertedmodelsintotheoriginal
pipeline<#embedding-the-converted-models-into-the-original-pipeline>`__

-`Selectinferencedevice<#select-inference-device>`__
-`AdaptOpenVINOmodelstotheoriginal
pipeline<#adapt-openvino-models-to-the-original-pipeline>`__

-`OptimizethemodelusingNNCFPost-trainingQuantization
API<#optimize-the-model-using-nncf-post-training-quantization-api>`__

-`ComparetheperformanceoftheOriginalandQuantized
Models<#compare-the-performance-of-the-original-and-quantized-models>`__

-`Tryouttheconvertedpipeline<#try-out-the-converted-pipeline>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installrequirements
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"ultralytics==8.2.24"onnxtqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino-dev>=2024.0.0"
%pipinstall-q"nncf>=2.9.0"
%pipinstall-q"gradio>=4.13"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importipywidgetsaswidgets
frompathlibimportPath

importopenvinoasov
importtorch
fromPILimportImage,ImageDraw
fromultralyticsimportFastSAM

#Fetchskip_kernel_extensionmodule
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)
#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

%load_extskip_kernel_extension

FastSAMinUltralytics
----------------------

`backtotop⬆️<#table-of-contents>`__

Toworkwith`FastSegmentAnything
Model<https://github.com/CASIA-IVA-Lab/FastSAM>`__by
``CASIA-IVA-Lab``,wewillusethe`Ultralytics
package<https://docs.ultralytics.com/>`__.Ultralyticspackageexposes
the``FastSAM``class,simplifyingthemodelinstantiationandweights
loading.Thecodebelowdemonstrateshowtoinitializea``FastSAM``
modelandgenerateasegmentationmap.

..code::ipython3

model_name="FastSAM-x"
model=FastSAM(model_name)

#Runinferenceonanimage
image_uri="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg"
image_uri=download_file(image_uri)
results=model(image_uri,device="cpu",retina_masks=True,imgsz=1024,conf=0.6,iou=0.9)


..parsed-literal::

Downloadinghttps://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-x.ptto'FastSAM-x.pt'...


..parsed-literal::

100%|██████████|138M/138M[00:03<00:00,44.4MB/s]



..parsed-literal::

coco_bike.jpg:0%||0.00/182k[00:00<?,?B/s]


..parsed-literal::


image1/1/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/fast-segment-anything/coco_bike.jpg:768x102437objects,706.9ms
Speed:3.9mspreprocess,706.9msinference,592.3mspostprocessperimageatshape(1,3,768,1024)


Themodelreturnssegmentationmapsforalltheobjectsontheimage.
Observetheresultsbelow.

..code::ipython3

Image.fromarray(results[0].plot()[...,::-1])




..image::fast-segment-anything-with-output_files/fast-segment-anything-with-output_9_0.png



ConvertthemodeltoOpenVINOIntermediaterepresentation(IR)format
---------------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

TheUltralyticsModelexportAPIenablesconversionofPyTorchmodelsto
OpenVINOIRformat.Underthehooditutilizesthe
``openvino.convert_model``methodtoacquireOpenVINOIRversionsofthe
models.Themethodrequiresamodelobjectandexampleinputformodel
tracing.TheFastSAMmodelitselfisbasedonYOLOv8model.

..code::ipython3

#instancesegmentationmodel
ov_model_path=Path(f"{model_name}_openvino_model/{model_name}.xml")
ifnotov_model_path.exists():
ov_model=model.export(format="openvino",dynamic=False,half=False)


..parsed-literal::

UltralyticsYOLOv8.2.24🚀Python-3.8.10torch-2.3.1+cpuCPU(IntelCore(TM)i9-10920X3.50GHz)

PyTorch:startingfrom'FastSAM-x.pt'withinputshape(1,3,1024,1024)BCHWandoutputshape(s)((1,37,21504),(1,32,256,256))(138.3MB)

OpenVINO:startingexportwithopenvino2024.2.0-15519-5c0f38f83f6-releases/2024/2...
OpenVINO:exportsuccess✅6.2s,savedas'FastSAM-x_openvino_model/'(276.1MB)

Exportcomplete(9.2s)
Resultssavedto/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/fast-segment-anything
Predict:yolopredicttask=segmentmodel=FastSAM-x_openvino_modelimgsz=1024
Validate:yolovaltask=segmentmodel=FastSAM-x_openvino_modelimgsz=1024data=ultralytics/datasets/sa.yaml
Visualize:https://netron.app


Embeddingtheconvertedmodelsintotheoriginalpipeline
---------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINO™RuntimePythonAPIisusedtocompilethemodelinOpenVINOIR
format.The
`Core<https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.runtime.Core.html>`__
classprovidesaccesstotheOpenVINORuntimeAPI.The``core``object,
whichisaninstanceofthe``Core``classrepresentstheAPIanditis
usedtocompilethemodel.

..code::ipython3

core=ov.Core()

Selectinferencedevice
^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

SelectdevicethatwillbeusedtodomodelsinferenceusingOpenVINO
fromthedropdownlist:

..code::ipython3

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



AdaptOpenVINOmodelstotheoriginalpipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

HerewecreatewrapperclassesfortheOpenVINOmodelthatwewantto
embedintheoriginalinferencepipeline.Herearesomeofthethingsto
considerwhenadaptinganOVmodel:-Makesurethatparameterspassed
bytheoriginalpipelineareforwardedtothecompiledOVmodel
properly;sometimestheOVmodelusesonlyaportionoftheinput
argumentsandsomeareignored,sometimesyouneedtoconvertthe
argumenttoanotherdatatypeorunwrapsomedatastructuressuchas
tuplesordictionaries.-Guaranteethatthewrapperclassreturns
resultstothepipelineinanexpectedformat.Intheexamplebelowyou
canseehowwepackOVmodeloutputsintoatupleof``torch``tensors.
-Payattentiontothemodelmethodusedintheoriginalpipelinefor
callingthemodel-itmaybenotthe``forward``method!Inthis
example,themodelisapartofa``predictor``objectandcalledasand
object,soweneedtoredefinethemagic``__call__``method.

..code::ipython3

classOVWrapper:
def__init__(self,ov_model,device="CPU",stride=32,ov_config=None)->None:
ov_config=ov_configor{}
self.model=core.compile_model(ov_model,device,ov_config)

self.stride=stride
self.pt=False
self.fp16=False
self.names={0:"object"}

def__call__(self,im,**_):
result=self.model(im)
returntorch.from_numpy(result[0]),torch.from_numpy(result[1])

NowweinitializethewrapperobjectsandloadthemtotheFastSAM
pipeline.

..code::ipython3

ov_config={}
if"GPU"indevice.valueor("AUTO"indevice.valueand"GPU"incore.available_devices):
ov_config={"GPU_DISABLE_WINOGRAD_CONVOLUTION":"YES"}

wrapped_model=OVWrapper(
ov_model_path,
device=device.value,
stride=model.predictor.model.stride,
ov_config=ov_config,
)
model.predictor.model=wrapped_model

ov_results=model(image_uri,device=device.value,retina_masks=True,imgsz=1024,conf=0.6,iou=0.9)


..parsed-literal::


image1/1/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/fast-segment-anything/coco_bike.jpg:1024x102442objects,508.7ms
Speed:7.4mspreprocess,508.7msinference,32.1mspostprocessperimageatshape(1,3,1024,1024)


Onecanobservetheconvertedmodeloutputsinthenextcell,theyis
thesameasoftheoriginalmodel.

..code::ipython3

Image.fromarray(ov_results[0].plot()[...,::-1])




..image::fast-segment-anything-with-output_files/fast-segment-anything-with-output_21_0.png



OptimizethemodelusingNNCFPost-trainingQuantizationAPI
------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf>`__providesasuiteof
advancedalgorithmsforNeuralNetworksinferenceoptimizationin
OpenVINOwithminimalaccuracydrop.Wewilluse8-bitquantizationin
post-trainingmode(withoutthefine-tuningpipeline)tooptimize
FastSAM.

Theoptimizationprocesscontainsthefollowingsteps:

1.CreateaDatasetforquantization.
2.Run``nncf.quantize``toobtainaquantizedmodel.
3.SavetheINT8modelusing``openvino.save_model()``function.

..code::ipython3

do_quantize=widgets.Checkbox(
value=True,
description="Quantization",
disabled=False,
)

do_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



The``nncf.quantize``functionprovidesaninterfaceformodel
quantization.ItrequiresaninstanceoftheOpenVINOModeland
quantizationdataset.Optionally,someadditionalparametersforthe
configurationquantizationprocess(numberofsamplesforquantization,
preset,ignoredscope,etc.)canbeprovided.YOLOv8modelbacking
FastSAMcontainsnon-ReLUactivationfunctions,whichrequireasymmetric
quantizationofactivations.Toachieveabetterresult,wewillusea
``mixed``quantizationpreset.Itprovidessymmetricquantizationof
weightsandasymmetricquantizationofactivations.Formoreaccurate
results,weshouldkeeptheoperationinthepostprocessingsubgraphin
floatingpointprecision,usingthe``ignored_scope``parameter.

Thequantizationalgorithmisbasedon`TheYOLOv8quantization
example<https://github.com/openvinotoolkit/nncf/tree/develop/examples/post_training_quantization/openvino/yolov8>`__
intheNNCFrepo,referthereformoredetails.Moreover,youcancheck
outotherquantizationtutorialsinthe`OVnotebooks
repo<../yolov8-optimization/>`__.

**Note**:Modelpost-trainingquantizationistime-consumingprocess.
Bepatient,itcantakeseveralminutesdependingonyourhardware.

..code::ipython3

%%skipnot$do_quantize.value

importpickle
fromcontextlibimportcontextmanager
fromzipfileimportZipFile

importcv2
fromtqdm.autonotebookimporttqdm

importnncf


COLLECT_CALIBRATION_DATA=False
calibration_data=[]

@contextmanager
defcalibration_data_collection():
globalCOLLECT_CALIBRATION_DATA
try:
COLLECT_CALIBRATION_DATA=True
yield
finally:
COLLECT_CALIBRATION_DATA=False


classNNCFWrapper:
def__init__(self,ov_model,stride=32)->None:
self.model=core.read_model(ov_model)
self.compiled_model=core.compile_model(self.model,device_name="CPU")

self.stride=stride
self.pt=False
self.fp16=False
self.names={0:"object"}

def__call__(self,im,**_):
ifCOLLECT_CALIBRATION_DATA:
calibration_data.append(im)

result=self.compiled_model(im)
returntorch.from_numpy(result[0]),torch.from_numpy(result[1])

#Fetchdatafromthewebanddescibeadataloader
DATA_URL="https://ultralytics.com/assets/coco128.zip"
OUT_DIR=Path('.')

download_file(DATA_URL,directory=OUT_DIR,show_progress=True)

ifnot(OUT_DIR/"coco128/images/train2017").exists():
withZipFile('coco128.zip',"r")aszip_ref:
zip_ref.extractall(OUT_DIR)

classCOCOLoader(torch.utils.data.Dataset):
def__init__(self,images_path):
self.images=list(Path(images_path).iterdir())

def__getitem__(self,index):
ifisinstance(index,slice):
return[self.read_image(image_path)forimage_pathinself.images[index]]
returnself.read_image(self.images[index])

defread_image(self,image_path):
image=cv2.imread(str(image_path))
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
returnimage

def__len__(self):
returnlen(self.images)


defcollect_calibration_data_for_decoder(model,calibration_dataset_size:int,
calibration_cache_path:Path):
globalcalibration_data


ifnotcalibration_cache_path.exists():
coco_dataset=COCOLoader(OUT_DIR/'coco128/images/train2017')
withcalibration_data_collection():
forimageintqdm(coco_dataset[:calibration_dataset_size],desc="Collectingcalibrationdata"):
model(image,retina_masks=True,imgsz=1024,conf=0.6,iou=0.9,verbose=False)
calibration_cache_path.parent.mkdir(parents=True,exist_ok=True)
withopen(calibration_cache_path,"wb")asf:
pickle.dump(calibration_data,f)
else:
withopen(calibration_cache_path,"rb")asf:
calibration_data=pickle.load(f)

returncalibration_data


defquantize(model,save_model_path:Path,calibration_cache_path:Path,
calibration_dataset_size:int,preset:nncf.QuantizationPreset):
calibration_data=collect_calibration_data_for_decoder(
model,calibration_dataset_size,calibration_cache_path)
quantized_ov_decoder=nncf.quantize(
model.predictor.model.model,
calibration_dataset=nncf.Dataset(calibration_data),
preset=preset,
subset_size=len(calibration_data),
fast_bias_correction=True,
ignored_scope=nncf.IgnoredScope(
types=["Multiply","Subtract","Sigmoid"],#ignoreoperations
names=[
"__module.model.22.dfl.conv/aten::_convolution/Convolution",#inthepost-processingsubgraph
"__module.model.22/aten::add/Add",
"__module.model.22/aten::add/Add_1"
],
)
)
ov.save_model(quantized_ov_decoder,save_model_path)

wrapped_model=NNCFWrapper(ov_model_path,stride=model.predictor.model.stride)
model.predictor.model=wrapped_model

calibration_dataset_size=128
quantized_model_path=Path(f"{model_name}_quantized")/"FastSAM-x.xml"
calibration_cache_path=Path(f"calibration_data/coco{calibration_dataset_size}.pkl")
ifnotquantized_model_path.exists():
quantize(model,quantized_model_path,calibration_cache_path,
calibration_dataset_size=calibration_dataset_size,
preset=nncf.QuantizationPreset.MIXED)


..parsed-literal::

<string>:7:TqdmExperimentalWarning:Using`tqdm.autonotebook.tqdm`innotebookmode.Use`tqdm.tqdm`insteadtoforceconsolemode(e.g.injupyterconsole)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino



..parsed-literal::

coco128.zip:0%||0.00/6.66M[00:00<?,?B/s]



..parsed-literal::

Collectingcalibrationdata:0%||0/128[00:00<?,?it/s]


..parsed-literal::

INFO:nncf:3ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:8ignorednodeswerefoundbytypesintheNNCFGraph
INFO:nncf:Notaddingactivationinputquantizerforoperation:271__module.model.22/aten::sigmoid/Sigmoid
INFO:nncf:Notaddingactivationinputquantizerforoperation:312__module.model.22.dfl.conv/aten::_convolution/Convolution
INFO:nncf:Notaddingactivationinputquantizerforoperation:349__module.model.22/aten::sub/Subtract
INFO:nncf:Notaddingactivationinputquantizerforoperation:350__module.model.22/aten::add/Add
INFO:nncf:Notaddingactivationinputquantizerforoperation:362__module.model.22/aten::add/Add_1
374__module.model.22/aten::div/Divide

INFO:nncf:Notaddingactivationinputquantizerforoperation:363__module.model.22/aten::sub/Subtract_1
INFO:nncf:Notaddingactivationinputquantizerforoperation:386__module.model.22/aten::mul/Multiply



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



ComparetheperformanceoftheOriginalandQuantizedModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Finally,weiterateboththeOVmodelandthequantizedmodeloverthe
calibrationdatasettomeasuretheperformance.

..code::ipython3

%%skipnot$do_quantize.value

importdatetime

coco_dataset=COCOLoader(OUT_DIR/'coco128/images/train2017')
calibration_dataset_size=128

wrapped_model=OVWrapper(ov_model_path,device=device.value,stride=model.predictor.model.stride)
model.predictor.model=wrapped_model

start_time=datetime.datetime.now()
forimageintqdm(coco_dataset,desc="Measuringinferencetime"):
model(image,retina_masks=True,imgsz=1024,conf=0.6,iou=0.9,verbose=False)
duration_base=(datetime.datetime.now()-start_time).seconds
print("Segmentedin",duration_base,"seconds.")
print("Resultingin",round(calibration_dataset_size/duration_base,2),"fps")



..parsed-literal::

Measuringinferencetime:0%||0/128[00:00<?,?it/s]


..parsed-literal::

Segmentedin69seconds.
Resultingin1.86fps


..code::ipython3

%%skipnot$do_quantize.value

quantized_wrapped_model=OVWrapper(quantized_model_path,device=device.value,stride=model.predictor.model.stride)
model.predictor.model=quantized_wrapped_model

start_time=datetime.datetime.now()
forimageintqdm(coco_dataset,desc="Measuringinferencetime"):
model(image,retina_masks=True,imgsz=1024,conf=0.6,iou=0.9,verbose=False)
duration_quantized=(datetime.datetime.now()-start_time).seconds
print("Segmentedin",duration_quantized,"seconds")
print("Resultingin",round(calibration_dataset_size/duration_quantized,2),"fps")
print("Thatis",round(duration_base/duration_quantized,2),"timesfaster!")



..parsed-literal::

Measuringinferencetime:0%||0/128[00:00<?,?it/s]


..parsed-literal::

Segmentedin22seconds
Resultingin5.82fps
Thatis3.14timesfaster!


Tryouttheconvertedpipeline
------------------------------

`backtotop⬆️<#table-of-contents>`__

Thedemoappbelowiscreatedusing`Gradio
package<https://www.gradio.app/docs/interface>`__.

Theappallowsyoutoalterthemodeloutputinteractively.Usingthe
Pixelselectortypeswitchyoucanplaceforeground/backgroundpointsor
boundingboxesoninputimage.

..code::ipython3

importcv2
importnumpyasnp
importmatplotlib.pyplotasplt


deffast_process(
annotations,
image,
scale,
better_quality=False,
mask_random_color=True,
bbox=None,
use_retina=True,
with_contours=True,
):
original_h=image.height
original_w=image.width

ifbetter_quality:
fori,maskinenumerate(annotations):
mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
annotations[i]=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,np.ones((8,8),np.uint8))

inner_mask=fast_show_mask(
annotations,
plt.gca(),
random_color=mask_random_color,
bbox=bbox,
retinamask=use_retina,
target_height=original_h,
target_width=original_w,
)

ifwith_contours:
contour_all=[]
temp=np.zeros((original_h,original_w,1))
fori,maskinenumerate(annotations):
annotation=mask.astype(np.uint8)
ifnotuse_retina:
annotation=cv2.resize(
annotation,
(original_w,original_h),
interpolation=cv2.INTER_NEAREST,
)
contours,_=cv2.findContours(annotation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
forcontourincontours:
contour_all.append(contour)
cv2.drawContours(temp,contour_all,-1,(255,255,255),2//scale)
color=np.array([0/255,0/255,255/255,0.9])
contour_mask=temp/255*color.reshape(1,1,-1)

image=image.convert("RGBA")
overlay_inner=Image.fromarray((inner_mask*255).astype(np.uint8),"RGBA")
image.paste(overlay_inner,(0,0),overlay_inner)

ifwith_contours:
overlay_contour=Image.fromarray((contour_mask*255).astype(np.uint8),"RGBA")
image.paste(overlay_contour,(0,0),overlay_contour)

returnimage


#CPUpostprocess
deffast_show_mask(
annotation,
ax,
random_color=False,
bbox=None,
retinamask=True,
target_height=960,
target_width=960,
):
mask_sum=annotation.shape[0]
height=annotation.shape[1]
weight=annotation.shape[2]
#
areas=np.sum(annotation,axis=(1,2))
sorted_indices=np.argsort(areas)[::1]
annotation=annotation[sorted_indices]

index=(annotation!=0).argmax(axis=0)
ifrandom_color:
color=np.random.random((mask_sum,1,1,3))
else:
color=np.ones((mask_sum,1,1,3))*np.array([30/255,144/255,255/255])
transparency=np.ones((mask_sum,1,1,1))*0.6
visual=np.concatenate([color,transparency],axis=-1)
mask_image=np.expand_dims(annotation,-1)*visual

mask=np.zeros((height,weight,4))

h_indices,w_indices=np.meshgrid(np.arange(height),np.arange(weight),indexing="ij")
indices=(index[h_indices,w_indices],h_indices,w_indices,slice(None))

mask[h_indices,w_indices,:]=mask_image[indices]
ifbboxisnotNone:
x1,y1,x2,y2=bbox
ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="b",linewidth=1))

ifnotretinamask:
mask=cv2.resize(mask,(target_width,target_height),interpolation=cv2.INTER_NEAREST)

returnmask

..code::ipython3

importgradioasgr

examples=[
[image_uri],
["https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/empty_road_mapillary.jpg"],
["https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/wall.jpg"],
]

object_points=[]
background_points=[]
bbox_points=[]
last_image=examples[0][0]

Thisisthemaincallbackfunctionthatiscalledtosegmentanimage
basedonuserinput.

..code::ipython3

defsegment(
image,
model_type,
input_size=1024,
iou_threshold=0.75,
conf_threshold=0.4,
better_quality=True,
with_contours=True,
use_retina=True,
mask_random_color=True,
):
ifdo_quantize.valueandmodel_type=="Quantizedmodel":
model.predictor.model=quantized_wrapped_model
else:
model.predictor.model=wrapped_model

input_size=int(input_size)
w,h=image.size
scale=input_size/max(w,h)
new_w=int(w*scale)
new_h=int(h*scale)
image=image.resize((new_w,new_h))

results=model(
image,
retina_masks=use_retina,
iou=iou_threshold,
conf=conf_threshold,
imgsz=input_size,
)

masks=results[0].masks.data
#Calculateannotations
ifnot(object_pointsorbbox_points):
annotations=masks.cpu().numpy()
else:
annotations=[]

ifobject_points:
all_points=object_points+background_points
labels=[1]*len(object_points)+[0]*len(background_points)
scaled_points=[[int(x*scale)forxinpoint]forpointinall_points]
h,w=masks[0].shape[:2]
assertmax(h,w)==input_size
onemask=np.zeros((h,w))
formaskinsorted(masks,key=lambdax:x.sum(),reverse=True):
mask_np=(mask==1.0).cpu().numpy()
forpoint,labelinzip(scaled_points,labels):
ifmask_np[point[1],point[0]]==1andlabel==1:
onemask[mask_np]=1
ifmask_np[point[1],point[0]]==1andlabel==0:
onemask[mask_np]=0
annotations.append(onemask>=1)
iflen(bbox_points)>=2:
scaled_bbox_points=[]
fori,pointinenumerate(bbox_points):
x,y=int(point[0]*scale),int(point[1]*scale)
x=max(min(x,new_w),0)
y=max(min(y,new_h),0)
scaled_bbox_points.append((x,y))

foriinrange(0,len(scaled_bbox_points)-1,2):
x0,y0,x1,y1=*scaled_bbox_points[i],*scaled_bbox_points[i+1]

intersection_area=torch.sum(masks[:,y0:y1,x0:x1],dim=(1,2))
masks_area=torch.sum(masks,dim=(1,2))
bbox_area=(y1-y0)*(x1-x0)

union=bbox_area+masks_area-intersection_area
iou=intersection_area/union
max_iou_index=torch.argmax(iou)

annotations.append(masks[max_iou_index].cpu().numpy())

returnfast_process(
annotations=np.array(annotations),
image=image,
scale=(1024//input_size),
better_quality=better_quality,
mask_random_color=mask_random_color,
bbox=None,
use_retina=use_retina,
with_contours=with_contours,
)

..code::ipython3

defselect_point(img:Image.Image,point_type:str,evt:gr.SelectData)->Image.Image:
"""Gradioselectcallback."""
img=img.convert("RGBA")
x,y=evt.index[0],evt.index[1]
point_radius=np.round(max(img.size)/100)
ifpoint_type=="Objectpoint":
object_points.append((x,y))
color=(30,255,30,200)
elifpoint_type=="Backgroundpoint":
background_points.append((x,y))
color=(255,30,30,200)
elifpoint_type=="BoundingBox":
bbox_points.append((x,y))
color=(10,10,255,255)
iflen(bbox_points)%2==0:
#Drawarectangleifnumberofpointsiseven
new_img=Image.new("RGBA",img.size,(255,255,255,0))
_draw=ImageDraw.Draw(new_img)
x0,y0,x1,y1=*bbox_points[-2],*bbox_points[-1]
x0,x1=sorted([x0,x1])
y0,y1=sorted([y0,y1])
#Savesortedorder
bbox_points[-2]=(x0,y0)
bbox_points[-1]=(x1,y1)
_draw.rectangle((x0,y0,x1,y1),fill=(*color[:-1],90))
img=Image.alpha_composite(img,new_img)
#Drawapoint
ImageDraw.Draw(img).ellipse(
[(x-point_radius,y-point_radius),(x+point_radius,y+point_radius)],
fill=color,
)
returnimg


defclear_points()->(Image.Image,None):
"""Gradioclearpointscallback."""
globalobject_points,background_points,bbox_points
#globalobject_points;globalbackground_points;globalbbox_points
object_points=[]
background_points=[]
bbox_points=[]
returnlast_image,None


defsave_last_picked_image(img:Image.Image)->None:
"""Gradiocallbacksavesthelastusedimage."""
globallast_image
last_image=img
#Ifwechangetheinputimage
#weshouldclearallthepreviouspoints
clear_points()
#Removesthesegmentationmapoutput
returnNone


withgr.Blocks(title="FastSAM")asdemo:
withgr.Row(variant="panel"):
original_img=gr.Image(label="Input",value=examples[0][0],type="pil")
segmented_img=gr.Image(label="SegmentationMap",type="pil")
withgr.Row():
point_type=gr.Radio(
["Objectpoint","Backgroundpoint","BoundingBox"],
value="Objectpoint",
label="Pixelselectortype",
)
model_type=gr.Radio(
["FP32model","Quantizedmodel"]ifdo_quantize.valueelse["FP32model"],
value="FP32model",
label="Selectmodelvariant",
)
withgr.Row(variant="panel"):
segment_button=gr.Button("Segment",variant="primary")
clear_button=gr.Button("Clearpoints",variant="secondary")
gr.Examples(
examples,
inputs=original_img,
fn=save_last_picked_image,
run_on_click=True,
outputs=segmented_img,
)

#Callbacks
original_img.select(select_point,inputs=[original_img,point_type],outputs=original_img)
original_img.upload(save_last_picked_image,inputs=original_img,outputs=segmented_img)
clear_button.click(clear_points,outputs=[original_img,segmented_img])
segment_button.click(segment,inputs=[original_img,model_type],outputs=segmented_img)

try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)

#Ifyouarelaunchingremotely,specifyserver_nameandserver_port
#EXAMPLE:`demo.launch(server_name="yourservername",server_port="serverportinint")`
#TolearnmorepleaserefertotheGradiodocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>

