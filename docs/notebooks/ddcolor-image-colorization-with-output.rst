Colorizegrayscaleimagesusing🎨DDColorandOpenVINO
======================================================

Imagecolorizationistheprocessofaddingcolortograyscaleimages.
Initiallycapturedinblackandwhite,theseimagesaretransformedinto
vibrant,lifelikerepresentationsbyestimatingRGBcolors.This
technologyenhancesbothaestheticappealandperceptualquality.
Historically,artistsmanuallyappliedcolorstomonochromatic
photographs,apainstakingtaskthatcouldtakeuptoamonthfora
singleimage.However,withadvancementsininformationtechnologyand
theriseofdeepneuralnetworks,automatedimagecolorizationhas
becomeincreasinglyimportant.

DDColorisoneofthemostprogressivemethodsofimagecolorizationin
ourdays.Itisanovelapproachusingdualdecoders:apixeldecoder
andaquery-basedcolordecoder,thatstandsoutinitsabilityto
producephoto-realisticcolorization,particularlyincomplexscenes
withmultipleobjectsanddiversecontexts.|image0|

Moredetailsaboutthisapproachcanbefoundinoriginalmodel
`repository<https://github.com/piddnad/DDColor>`__and
`paper<https://arxiv.org/abs/2212.11613>`__.

InthistutorialweconsiderhowtoconvertandrunDDColorusing
OpenVINO.Additionally,wewilldemonstratehowtooptimizethismodel
using`NNCF<https://github.com/openvinotoolkit/nncf/>`__.

🪄Let’sstarttoexploremagicofimagecolorization!####Tableof
contents:

-`Prerequisites<#prerequisites>`__
-`LoadPyTorchmodel<#load-pytorch-model>`__
-`RunPyTorchmodelinference<#run-pytorch-model-inference>`__
-`ConvertPyTorchmodeltoOpenVINOIntermediate
Representation<#convert-pytorch-model-to-openvino-intermediate-representation>`__
-`RunOpenVINOmodelinference<#run-openvino-model-inference>`__
-`OptimizeOpenVINOmodelusing
NNCF<#optimize-openvino-model-using-nncf>`__

-`Collectquantizationdataset<#collect-quantization-dataset>`__
-`Performmodelquantization<#perform-model-quantization>`__

-`RunINT8modelinference<#run-int8-model-inference>`__
-`CompareFP16andINT8model
size<#compare-fp16-and-int8-model-size>`__
-`CompareinferencetimeoftheFP16andINT8
models<#compare-inference-time-of-the-fp16-and-int8-models>`__
-`Interactiveinference<#interactive-inference>`__

..|image0|image::https://github.com/piddnad/DDColor/raw/master/assets/network_arch.jpg

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-q"nncf>=2.11.0""torch>=2.1""torchvision""timm""opencv_python""pillow""PyYAML""scipy""scikit-image""datasets""gradio>=4.19"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-Uq--pre"openvino"--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly

ifplatform.python_version_tuple()[1]in["8","9"]:
%pipinstall-q"gradio-imageslider<=0.0.17""typing-extensions>=4.9.0"
else:
%pipinstall-q"gradio-imageslider"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
openvino-dev2024.2.0requiresopenvino==2024.2.0,butyouhaveopenvino2024.4.0.dev20240712whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importsys
frompathlibimportPath

repo_dir=Path("DDColor")

ifnotrepo_dir.exists():
!gitclonehttps://github.com/piddnad/DDColor.git

sys.path.append(str(repo_dir))


..parsed-literal::

Cloninginto'DDColor'...
remote:Enumeratingobjects:230,done.[K
remote:Countingobjects:100%(76/76),done.[K
remote:Compressingobjects:100%(39/39),done.[K
remote:Total230(delta54),reused40(delta36),pack-reused154[K
Receivingobjects:100%(230/230),13.34MiB|20.76MiB/s,done.
Resolvingdeltas:100%(75/75),done.


..code::ipython3

try:
frominference.colorization_pipeline_hfimportDDColorHF,ImageColorizationPipelineHF
exceptException:
frominference.colorization_pipeline_hfimportDDColorHF,ImageColorizationPipelineHF

LoadPyTorchmodel
------------------

`backtotop⬆️<#table-of-contents>`__

ThereareseveralmodelsfromDDColor’sfamilyprovidedin`model
repository<https://github.com/piddnad/DDColor/blob/master/MODEL_ZOO.md>`__.
WewilluseDDColor-T,themostlightweightversionofddcolormodel,
butdemonstratedinthetutorialstepsarealsoapplicabletoother
modelsfromDDColorfamily.

..code::ipython3

importtorch

model_name="ddcolor_paper_tiny"

ddcolor_model=DDColorHF.from_pretrained(f"piddnad/{model_name}")


colorizer=ImageColorizationPipelineHF(model=ddcolor_model,input_size=512)

ddcolor_model.to("cpu")
colorizer.device=torch.device("cpu")

RunPyTorchmodelinference
---------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcv2
importPIL

IMG_PATH="DDColor/assets/test_images/AnselAdams_MoorePhotography.jpeg"


img=cv2.imread(IMG_PATH)

PIL.Image.fromarray(img[:,:,::-1])




..image::ddcolor-image-colorization-with-output_files/ddcolor-image-colorization-with-output_8_0.png



..code::ipython3

image_out=colorizer.process(img)
PIL.Image.fromarray(image_out[:,:,::-1])




..image::ddcolor-image-colorization-with-output_files/ddcolor-image-colorization-with-output_9_0.png



ConvertPyTorchmodeltoOpenVINOIntermediateRepresentation
-------------------------------------------------------------

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
importtorch

OV_COLORIZER_PATH=Path("ddcolor.xml")

ifnotOV_COLORIZER_PATH.exists():
ov_model=ov.convert_model(ddcolor_model,example_input=torch.ones((1,3,512,512)),input=[1,3,512,512])
ov.save_model(ov_model,OV_COLORIZER_PATH)


..parsed-literal::

['x']


RunOpenVINOmodelinference
----------------------------

`backtotop⬆️<#table-of-contents>`__

Selectoneofsupporteddevicesforinferenceusingdropdownlist.

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

compiled_model=core.compile_model(OV_COLORIZER_PATH,device.value)

..code::ipython3

importcv2
importnumpyasnp
importtorch
importtorch.nn.functionalasF


defprocess(img,compiled_model):
#Preprocessinputimage
height,width=img.shape[:2]

#Normalizeto[0,1]range
img=(img/255.0).astype(np.float32)
orig_l=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)[:,:,:1]#(h,w,1)

#Resizergbimage->lab->getgrey->rgb
img=cv2.resize(img,(512,512))
img_l=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)[:,:,:1]
img_gray_lab=np.concatenate((img_l,np.zeros_like(img_l),np.zeros_like(img_l)),axis=-1)
img_gray_rgb=cv2.cvtColor(img_gray_lab,cv2.COLOR_LAB2RGB)

#TransposeHWC->CHWandaddbatchdimension
tensor_gray_rgb=torch.from_numpy(img_gray_rgb.transpose((2,0,1))).float().unsqueeze(0)

#Runmodelinference
output_ab=compiled_model(tensor_gray_rgb)[0]

#Postprocessresult
#resizeab->concatoriginall->rgb
output_ab_resize=F.interpolate(torch.from_numpy(output_ab),size=(height,width))[0].float().numpy().transpose(1,2,0)
output_lab=np.concatenate((orig_l,output_ab_resize),axis=-1)
output_bgr=cv2.cvtColor(output_lab,cv2.COLOR_LAB2BGR)

output_img=(output_bgr*255.0).round().astype(np.uint8)

returnoutput_img

..code::ipython3

ov_processed_img=process(img,compiled_model)
PIL.Image.fromarray(ov_processed_img[:,:,::-1])




..image::ddcolor-image-colorization-with-output_files/ddcolor-image-colorization-with-output_16_0.png



OptimizeOpenVINOmodelusingNNCF
----------------------------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

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



..code::ipython3

importrequests

OV_INT8_COLORIZER_PATH=Path("ddcolor_int8.xml")
compiled_int8_model=None

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

%load_extskip_kernel_extension

Collectquantizationdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseaportionof
`ummagumm-a/colorization_dataset<https://huggingface.co/datasets/ummagumm-a/colorization_dataset>`__
datasetfromHuggingFaceascalibrationdata.

..code::ipython3

%%skipnot$to_quantize.value

fromdatasetsimportload_dataset

subset_size=300
calibration_data=[]

ifnotOV_INT8_COLORIZER_PATH.exists():
dataset=load_dataset("ummagumm-a/colorization_dataset",split="train",streaming=True).shuffle(seed=42).take(subset_size)
foridx,batchinenumerate(dataset):
ifidx>=subset_size:
break
img=np.array(batch["conditioning_image"])
img=(img/255.0).astype(np.float32)
img=cv2.resize(img,(512,512))
img_l=cv2.cvtColor(np.stack([img,img,img],axis=2),cv2.COLOR_BGR2Lab)[:,:,:1]
img_gray_lab=np.concatenate((img_l,np.zeros_like(img_l),np.zeros_like(img_l)),axis=-1)
img_gray_rgb=cv2.cvtColor(img_gray_lab,cv2.COLOR_LAB2RGB)

image=np.expand_dims(img_gray_rgb.transpose((2,0,1)).astype(np.float32),axis=0)
calibration_data.append(image)

Performmodelquantization
~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

importnncf

ifnotOV_INT8_COLORIZER_PATH.exists():
ov_model=core.read_model(OV_COLORIZER_PATH)
quantized_model=nncf.quantize(
model=ov_model,
subset_size=subset_size,
calibration_dataset=nncf.Dataset(calibration_data),
)
ov.save_model(quantized_model,OV_INT8_COLORIZER_PATH)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..parsed-literal::

2024-07-1223:51:48.961005:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1223:51:49.001135:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1223:51:49.398556:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



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



RunINT8modelinference
------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromIPython.displayimportdisplay

ifOV_INT8_COLORIZER_PATH.exists():
compiled_int8_model=core.compile_model(OV_INT8_COLORIZER_PATH,device.value)
img=cv2.imread("DDColor/assets/test_images/AnselAdams_MoorePhotography.jpeg")
img_out=process(img,compiled_int8_model)
display(PIL.Image.fromarray(img_out[:,:,::-1]))



..image::ddcolor-image-colorization-with-output_files/ddcolor-image-colorization-with-output_25_0.png


CompareFP16andINT8modelsize
--------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fp16_ir_model_size=OV_COLORIZER_PATH.with_suffix(".bin").stat().st_size/2**20

print(f"FP16modelsize:{fp16_ir_model_size:.2f}MB")

ifOV_INT8_COLORIZER_PATH.exists():
quantized_model_size=OV_INT8_COLORIZER_PATH.with_suffix(".bin").stat().st_size/2**20
print(f"INT8modelsize:{quantized_model_size:.2f}MB")
print(f"Modelcompressionrate:{fp16_ir_model_size/quantized_model_size:.3f}")


..parsed-literal::

FP16modelsize:104.89MB
INT8modelsize:52.97MB
Modelcompressionrate:1.980


CompareinferencetimeoftheFP16andINT8models
--------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

TomeasuretheinferenceperformanceofOpenVINOFP16andINT8models,
use`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.

..code::ipython3

!benchmark_app-m$OV_COLORIZER_PATH-d$device.value-apiasync-shape"[1,3,512,512]"-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook41.74ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,512,512]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.refine_net.0.0/aten::_convolution/Add):f32/[...]/[1,2,512,512]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,512,512]
[INFO]Reshapemodeltook0.04ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,512,512]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.refine_net.0.0/aten::_convolution/Add):f32/[...]/[1,2,512,512]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook1302.32ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:32
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:24
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:6
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,6inferencerequests,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook537.09ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:72iterations
[INFO]Duration:16531.03ms
[INFO]Latency:
[INFO]Median:1375.35ms
[INFO]Average:1368.70ms
[INFO]Min:1259.43ms
[INFO]Max:1453.51ms
[INFO]Throughput:4.36FPS


..code::ipython3

ifOV_INT8_COLORIZER_PATH.exists():
!benchmark_app-m$OV_INT8_COLORIZER_PATH-d$device.value-apiasync-shape"[1,3,512,512]"-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook68.54ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]x(node:x):f32/[...]/[1,3,512,512]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.refine_net.0.0/aten::_convolution/Add):f32/[...]/[1,2,512,512]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[INFO]Reshapingmodel:'x':[1,3,512,512]
[INFO]Reshapemodeltook0.04ms
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]x(node:x):u8/[N,C,H,W]/[1,3,512,512]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:__module.refine_net.0.0/aten::_convolution/Add):f32/[...]/[1,2,512,512]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook2180.66ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:32
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:24
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:6
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'x'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'x'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,6inferencerequests,limits:15000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook283.96ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:156iterations
[INFO]Duration:15915.24ms
[INFO]Latency:
[INFO]Median:608.17ms
[INFO]Average:609.92ms
[INFO]Min:550.02ms
[INFO]Max:718.37ms
[INFO]Throughput:9.80FPS


Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr
fromgradio_imagesliderimportImageSlider
fromfunctoolsimportpartial


defgenerate(image,use_int8=True):
image_in=cv2.imread(image)
image_out=process(image_in,compiled_modelifnotuse_int8elsecompiled_int8_model)
image_in_pil=PIL.Image.fromarray(cv2.cvtColor(image_in,cv2.COLOR_BGR2RGB))
image_out_pil=PIL.Image.fromarray(cv2.cvtColor(image_out,cv2.COLOR_BGR2RGB))
return(image_in_pil,image_out_pil)


withgr.Blocks()asdemo:
withgr.Row(equal_height=False):
image=gr.Image(type="filepath")
withgr.Column():
output_image=ImageSlider(show_label=True,type="filepath",interactive=False,label="FP16modeloutput")
button=gr.Button(value="Run{}".format("FP16model"ifcompiled_int8_modelisnotNoneelse""))
withgr.Column(visible=compiled_int8_modelisnotNone):
output_image_int8=ImageSlider(show_label=True,type="filepath",interactive=False,label="INT8modeloutput")
button_i8=gr.Button(value="RunINT8model")
button.click(fn=partial(generate,use_int8=False),inputs=[image],outputs=[output_image])
button_i8.click(fn=partial(generate,use_int8=True),inputs=[image],outputs=[output_image_int8])
examples=gr.Examples(
[
"DDColor/assets/test_images/NewYorkRiverfrontDecember15,1931.jpg",
"DDColor/assets/test_images/AudreyHepburn.jpg",
"DDColor/assets/test_images/AcrobatsBalanceOnTopOfTheEmpireStateBuilding,1934.jpg",
],
inputs=[image],
)


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

