AutomaticDeviceSelectionwithOpenVINO™
=========================================

The`Auto
device<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html>`__
(orAUTOinshort)selectsthemostsuitabledeviceforinferenceby
consideringthemodelprecision,powerefficiencyandprocessing
capabilityoftheavailable`compute
devices<https://docs.openvino.ai/2024/about-openvino/compatibility-and-support/supported-devices.html>`__.
Themodelprecision(suchas``FP32``,``FP16``,``INT8``,etc.)isthe
firstconsiderationtofilteroutthedevicesthatcannotrunthe
networkefficiently.

Next,ifdedicatedacceleratorsareavailable,thesedevicesare
preferred(forexample,integratedanddiscrete
`GPU<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html>`__).
`CPU<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html>`__
isusedasthedefault“fallbackdevice”.KeepinmindthatAUTOmakes
thisselectiononlyonce,duringtheloadingofamodel.

WhenusingacceleratordevicessuchasGPUs,loadingmodelstothese
devicesmaytakealongtime.Toaddressthischallengeforapplications
thatrequirefastfirstinferenceresponse,AUTOstartsinference
immediatelyontheCPUandthentransparentlyshiftsinferencetothe
GPU,onceitisready.Thisdramaticallyreducesthetimetoexecute
firstinference.

..figure::https://user-images.githubusercontent.com/15709723/161451847-759e2bdb-70bc-463d-9818-400c0ccf3c16.png
:alt:auto

auto

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`ImportmodulesandcreateCore<#import-modules-and-create-core>`__
-`ConvertthemodeltoOpenVINOIR
format<#convert-the-model-to-openvino-ir-format>`__
-`(1)Simplifyselectionlogic<#1-simplify-selection-logic>`__

-`DefaultbehaviorofCore::compile_modelAPIwithout
device_name<#default-behavior-of-corecompile_model-api-without-device_name>`__
-`ExplicitlypassAUTOasdevice_nametoCore::compile_model
API<#explicitly-pass-auto-as-device_name-to-corecompile_model-api>`__

-`(2)Improvethefirstinference
latency<#2-improve-the-first-inference-latency>`__

-`LoadanImage<#load-an-image>`__
-`LoadthemodeltoGPUdeviceandperform
inference<#load-the-model-to-gpu-device-and-perform-inference>`__
-`LoadthemodelusingAUTOdeviceanddo
inference<#load-the-model-using-auto-device-and-do-inference>`__

-`(3)Achievedifferentperformancefordifferent
targets<#3-achieve-different-performance-for-different-targets>`__

-`Classandcallbackdefinition<#class-and-callback-definition>`__
-`InferencewithTHROUGHPUT
hint<#inference-with-throughput-hint>`__
-`InferencewithLATENCYhint<#inference-with-latency-hint>`__
-`DifferenceinFPSandlatency<#difference-in-fps-and-latency>`__

ImportmodulesandcreateCore
------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

#Installrequiredpackages
%pipinstall-q"openvino>=2023.1.0"Pillowtorchtorchvisiontqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importtime
importsys

importopenvinoasov

fromIPython.displayimportMarkdown,display

core=ov.Core()

ifnotany("GPU"indevicefordeviceincore.available_devices):
display(
Markdown(
'<divclass="alertalert-blockalert-danger"><b>Warning:</b>AGPUdeviceisnotavailable.ThisnotebookrequiresGPUdevicetohavemeaningfulresults.</div>'
)
)



..container::alertalert-blockalert-danger

Warning:AGPUdeviceisnotavailable.ThisnotebookrequiresGPU
devicetohavemeaningfulresults.


ConvertthemodeltoOpenVINOIRformat
---------------------------------------

`backtotop⬆️<#table-of-contents>`__

Thistutorialuses
`resnet50<https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#resnet50>`__
modelfrom
`torchvision<https://pytorch.org/vision/main/index.html?highlight=torchvision#module-torchvision>`__
library.ResNet50isimageclassificationmodelpre-trainedonImageNet
datasetdescribedinpaper`“DeepResidualLearningforImage
Recognition”<https://arxiv.org/abs/1512.03385>`__.FromOpenVINO
2023.0,wecandirectlyconvertamodelfromthePyTorchformattothe
OpenVINOIRformatusingmodelconversionAPI.Toconvertmodel,we
shouldprovidemodelobjectinstanceinto``ov.convert_model``function,
optionally,wecanspecifyinputshapeforconversion(bydefaultmodels
fromPyTorchconvertedwithdynamicinputshapes).``ov.convert_model``
returnsopenvino.runtime.Modelobjectreadytobeloadedonadevice
with``ov.compile_model``orserializedfornextusagewith
``ov.save_model``.

FormoreinformationaboutmodelconversionAPI,seethis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

..code::ipython3

importtorchvision
frompathlibimportPath

base_model_dir=Path("./model")
base_model_dir.mkdir(exist_ok=True)
model_path=base_model_dir/"resnet50.xml"

ifnotmodel_path.exists():
pt_model=torchvision.models.resnet50(weights="DEFAULT")
ov_model=ov.convert_model(pt_model,input=[[1,3,224,224]])
ov.save_model(ov_model,str(model_path))
print("IRmodelsavedto{}".format(model_path))
else:
print("ReadIRmodelfrom{}".format(model_path))
ov_model=core.read_model(model_path)


..parsed-literal::

IRmodelsavedtomodel/resnet50.xml


(1)Simplifyselectionlogic
----------------------------

`backtotop⬆️<#table-of-contents>`__

DefaultbehaviorofCore::compile_modelAPIwithoutdevice_name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Bydefault,``compile_model``APIwillselect**AUTO**as
``device_name``ifnodeviceisspecified.

..code::ipython3

#SetLOG_LEVELtoLOG_INFO.
core.set_property("AUTO",{"LOG_LEVEL":"LOG_INFO"})

#Loadthemodelontothetargetdevice.
compiled_model=core.compile_model(ov_model)

ifisinstance(compiled_model,ov.CompiledModel):
print("Successfullycompiledmodelwithoutadevice_name.")


..parsed-literal::

[23:26:37.1843]I[plugin.cpp:421][AUTO]device:CPU,config:LOG_LEVEL=LOG_INFO
[23:26:37.1844]I[plugin.cpp:421][AUTO]device:CPU,config:PERFORMANCE_HINT=LATENCY
[23:26:37.1844]I[plugin.cpp:421][AUTO]device:CPU,config:PERFORMANCE_HINT_NUM_REQUESTS=0
[23:26:37.1844]I[plugin.cpp:421][AUTO]device:CPU,config:PERF_COUNT=NO
[23:26:37.1844]I[plugin.cpp:426][AUTO]device:CPU,priority:0
[23:26:37.1844]I[schedule.cpp:17][AUTO]schedulerstarting
[23:26:37.1844]I[auto_schedule.cpp:134][AUTO]selectdevice:CPU
[23:26:37.3288]I[auto_schedule.cpp:336][AUTO]Device:[CPU]:Compilemodeltook144.341797ms
[23:26:37.3290]I[auto_schedule.cpp:112][AUTO]device:CPUcompilingmodelfinished
[23:26:37.3291]I[plugin.cpp:454][AUTO]underlyinghardwaredoesnotsupporthardwarecontext
Successfullycompiledmodelwithoutadevice_name.


..code::ipython3

#Deletedmodelwillwaituntilcompilingontheselecteddeviceiscomplete.
delcompiled_model
print("Deletedcompiled_model")


..parsed-literal::

Deletedcompiled_model
[23:26:37.3399]I[schedule.cpp:308][AUTO]schedulerending


ExplicitlypassAUTOasdevice_nametoCore::compile_modelAPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Itisoptional,butpassingAUTOexplicitlyas``device_name``may
improvereadabilityofyourcode.

..code::ipython3

#SetLOG_LEVELtoLOG_NONE.
core.set_property("AUTO",{"LOG_LEVEL":"LOG_NONE"})

compiled_model=core.compile_model(model=ov_model,device_name="AUTO")

ifisinstance(compiled_model,ov.CompiledModel):
print("SuccessfullycompiledmodelusingAUTO.")


..parsed-literal::

SuccessfullycompiledmodelusingAUTO.


..code::ipython3

#Deletedmodelwillwaituntilcompilingontheselecteddeviceiscomplete.
delcompiled_model
print("Deletedcompiled_model")


..parsed-literal::

Deletedcompiled_model


(2)Improvethefirstinferencelatency
---------------------------------------

`backtotop⬆️<#table-of-contents>`__

OneofthebenefitsofusingAUTOdeviceselectionisreducingFIL
(firstinferencelatency).FIListhemodelcompilationtimecombined
withthefirstinferenceexecutiontime.UsingtheCPUdeviceexplicitly
willproducetheshortestfirstinferencelatency,astheOpenVINOgraph
representationloadsquicklyonCPU,usingjust-in-time(JIT)
compilation.ThechallengeiswithGPUdevicessinceOpenCLgraph
complicationtoGPU-optimizedkernelstakesafewsecondstocomplete.
Thisinitializationtimemaybeintolerableforsomeapplications.To
avoidthisdelay,theAUTOusesCPUtransparentlyasthefirstinference
deviceuntilGPUisready.

LoadanImage
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

torchvisionlibraryprovidesmodelspecificinputtransformation
function,wewillreuseitforpreparinginputdata.

..code::ipython3

#Fetch`notebook_utils`module
importrequests

r=requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file

..code::ipython3

fromPILimportImage

#Downloadtheimagefromtheopenvino_notebooksstorage
image_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
directory="data",
)

image=Image.open(str(image_filename))
input_transform=torchvision.models.ResNet50_Weights.DEFAULT.transforms()

input_tensor=input_transform(image)
input_tensor=input_tensor.unsqueeze(0).numpy()
image



..parsed-literal::

data/coco.jpg:0%||0.00/202k[00:00<?,?B/s]




..image::auto-device-with-output_files/auto-device-with-output_14_1.png



LoadthemodeltoGPUdeviceandperforminference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotany("GPU"indevicefordeviceincore.available_devices):
print(f"AGPUdeviceisnotavailable.Availabledevicesare:{core.available_devices}")
else:
#Starttime.
gpu_load_start_time=time.perf_counter()
compiled_model=core.compile_model(model=ov_model,device_name="GPU")#loadtoGPU

#Executethefirstinference.
results=compiled_model(input_tensor)[0]

#Measuretimetothefirstinference.
gpu_fil_end_time=time.perf_counter()
gpu_fil_span=gpu_fil_end_time-gpu_load_start_time
print(f"TimetoloadmodelonGPUdeviceandgetfirstinference:{gpu_fil_end_time-gpu_load_start_time:.2f}seconds.")
delcompiled_model


..parsed-literal::

AGPUdeviceisnotavailable.Availabledevicesare:['CPU']


LoadthemodelusingAUTOdeviceanddoinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

WhenGPUisthebestavailabledevice,thefirstfewinferenceswillbe
executedonCPUuntilGPUisready.

..code::ipython3

#Starttime.
auto_load_start_time=time.perf_counter()
compiled_model=core.compile_model(model=ov_model)#Thedevice_nameisAUTObydefault.

#Executethefirstinference.
results=compiled_model(input_tensor)[0]


#Measuretimetothefirstinference.
auto_fil_end_time=time.perf_counter()
auto_fil_span=auto_fil_end_time-auto_load_start_time
print(f"TimetoloadmodelusingAUTOdeviceandgetfirstinference:{auto_fil_end_time-auto_load_start_time:.2f}seconds.")


..parsed-literal::

TimetoloadmodelusingAUTOdeviceandgetfirstinference:0.17seconds.


..code::ipython3

#Deletedmodelwillwaitforcompilingontheselecteddevicetocomplete.
delcompiled_model

(3)Achievedifferentperformancefordifferenttargets
-------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Itisanadvantagetodefine**performancehints**whenusingAutomatic
DeviceSelection.Byspecifyinga**THROUGHPUT**or**LATENCY**hint,
AUTOoptimizestheperformancebasedonthedesiredmetric.The
**THROUGHPUT**hintdelivershigherframepersecond(FPS)performance
thanthe**LATENCY**hint,whichdeliverslowerlatency.Theperformance
hintsdonotrequireanydevice-specificsettingsandtheyare
completelyportablebetweendevices–meaningAUTOcanconfigurethe
performancehintonwhicheverdeviceisbeingused.

Formoreinformation,refertothe`Performance
Hints<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html#performance-hints-for-auto>`__
sectionof`AutomaticDevice
Selection<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html>`__
article.

Classandcallbackdefinition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

classPerformanceMetrics:
"""
Recordthelatestperformancemetrics(fpsandlatency),updatethemetricsineach@intervalseconds
:member:fps:Framespersecond,indicatestheaveragenumberofinferencesexecutedeachsecondduringthelast@intervalseconds.
:member:latency:Averagelatencyofinferencesexecutedinthelast@intervalseconds.
:member:start_time:Recordthestarttimestampofonging@intervalsecondsduration.
:member:latency_list:Recordthelatencyofeachinferenceexecutionover@intervalsecondsduration.
:member:interval:Themetricswillbeupdatedevery@intervalseconds
"""

def__init__(self,interval):
"""
CreateandinitilizeoneinstanceofclassPerformanceMetrics.
:param:interval:Themetricswillbeupdatedevery@intervalseconds
:returns:
InstanceofPerformanceMetrics
"""
self.fps=0
self.latency=0

self.start_time=time.perf_counter()
self.latency_list=[]
self.interval=interval

defupdate(self,infer_request:ov.InferRequest)->bool:
"""
Updatethemetricsifcurrentongoing@intervalsecondsdurationisexpired.Recordthelatencyonlyifitisnotexpired.
:param:infer_request:InferRequestreturnedfrominferencecallback,whichincludestheresultofinferencerequest.
:returns:
True,ifmetricsareupdated.
False,if@intervalsecondsdurationisnotexpiredandmetricsarenotupdated.
"""
self.latency_list.append(infer_request.latency)
exec_time=time.perf_counter()-self.start_time
ifexec_time>=self.interval:
#Updatetheperformancemetrics.
self.start_time=time.perf_counter()
self.fps=len(self.latency_list)/exec_time
self.latency=sum(self.latency_list)/len(self.latency_list)
print(f"throughput:{self.fps:.2f}fps,latency:{self.latency:.2f}ms,timeinterval:{exec_time:.2f}s")
sys.stdout.flush()
self.latency_list=[]
returnTrue
else:
returnFalse


classInferContext:
"""
Inferencecontext.Recordandupdatepeforamncemetricsvia@metrics,set@feed_inferencetoFalseonce@remaining_update_num<=0
:member:metrics:instanceofclassPerformanceMetrics
:member:remaining_update_num:theremainingtimesforpeforamncemetricsupdating.
:member:feed_inference:iffeedinferencerequestisrequiredornot.
"""

def__init__(self,update_interval,num):
"""
CreateandinitilizeoneinstanceofclassInferContext.
:param:update_interval:Theperformancemetricswillbeupdatedevery@update_intervalseconds.ThisparameterwillbepassedtoclassPerformanceMetricsdirectly.
:param:num:Thenumberoftimesperformancemetricsareupdated.
:returns:
InstanceofInferContext.
"""
self.metrics=PerformanceMetrics(update_interval)
self.remaining_update_num=num
self.feed_inference=True

defupdate(self,infer_request:ov.InferRequest):
"""
Updatethecontext.Set@feed_inferencetoFalseifthenumberofremainingperformancemetricupdates(@remaining_update_num)reaches0
:param:infer_request:InferRequestreturnedfrominferencecallback,whichincludestheresultofinferencerequest.
:returns:None
"""
ifself.remaining_update_num<=0:
self.feed_inference=False

ifself.metrics.update(infer_request):
self.remaining_update_num=self.remaining_update_num-1
ifself.remaining_update_num<=0:
self.feed_inference=False


defcompletion_callback(infer_request:ov.InferRequest,context)->None:
"""
callbackfortheinferencerequest,passthe@infer_requestto@contextforupdating
:param:infer_request:InferRequestreturnedforthecallback,whichincludestheresultofinferencerequest.
:param:context:userdatawhichispassedasthesecondparametertoAsyncInferQueue:start_async()
:returns:None
"""
context.update(infer_request)


#Performancemetricsupdateinterval(seconds)andnumberoftimes.
metrics_update_interval=10
metrics_update_num=6

InferencewithTHROUGHPUThint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

LoopforinferenceandupdatetheFPS/Latencyevery
@metrics_update_intervalseconds.

..code::ipython3

THROUGHPUT_hint_context=InferContext(metrics_update_interval,metrics_update_num)

print("CompilingModelforAUTOdevicewithTHROUGHPUThint")
sys.stdout.flush()

compiled_model=core.compile_model(model=ov_model,config={"PERFORMANCE_HINT":"THROUGHPUT"})

infer_queue=ov.AsyncInferQueue(compiled_model,0)#Settingto0willqueryoptimalnumberbydefault.
infer_queue.set_callback(completion_callback)

print(f"Startinference,{metrics_update_num:.0f}groupsofFPS/latencywillbemeasuredover{metrics_update_interval:.0f}sintervals")
sys.stdout.flush()

whileTHROUGHPUT_hint_context.feed_inference:
infer_queue.start_async(input_tensor,THROUGHPUT_hint_context)

infer_queue.wait_all()

#TaketheFPSandlatencyofthelatestperiod.
THROUGHPUT_hint_fps=THROUGHPUT_hint_context.metrics.fps
THROUGHPUT_hint_latency=THROUGHPUT_hint_context.metrics.latency

print("Done")

delcompiled_model


..parsed-literal::

CompilingModelforAUTOdevicewithTHROUGHPUThint
Startinference,6groupsofFPS/latencywillbemeasuredover10sintervals
throughput:179.02fps,latency:31.75ms,timeinterval:10.02s
throughput:179.80fps,latency:32.59ms,timeinterval:10.00s
throughput:179.17fps,latency:32.63ms,timeinterval:10.01s
throughput:179.81fps,latency:32.58ms,timeinterval:10.01s
throughput:178.74fps,latency:32.75ms,timeinterval:10.00s
throughput:179.33fps,latency:32.57ms,timeinterval:10.02s
Done


InferencewithLATENCYhint
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

LoopforinferenceandupdatetheFPS/Latencyforeach
@metrics_update_intervalseconds

..code::ipython3

LATENCY_hint_context=InferContext(metrics_update_interval,metrics_update_num)

print("CompilingModelforAUTODevicewithLATENCYhint")
sys.stdout.flush()

compiled_model=core.compile_model(model=ov_model,config={"PERFORMANCE_HINT":"LATENCY"})

#Settingto0willqueryoptimalnumberbydefault.
infer_queue=ov.AsyncInferQueue(compiled_model,0)
infer_queue.set_callback(completion_callback)

print(f"Startinference,{metrics_update_num:.0f}groupsfps/latencywillbeoutwith{metrics_update_interval:.0f}sinterval")
sys.stdout.flush()

whileLATENCY_hint_context.feed_inference:
infer_queue.start_async(input_tensor,LATENCY_hint_context)

infer_queue.wait_all()

#TaketheFPSandlatencyofthelatestperiod.
LATENCY_hint_fps=LATENCY_hint_context.metrics.fps
LATENCY_hint_latency=LATENCY_hint_context.metrics.latency

print("Done")

delcompiled_model


..parsed-literal::

CompilingModelforAUTODevicewithLATENCYhint
Startinference,6groupsfps/latencywillbeoutwith10sinterval
throughput:137.56fps,latency:6.70ms,timeinterval:10.00s
throughput:140.27fps,latency:6.69ms,timeinterval:10.00s
throughput:140.43fps,latency:6.68ms,timeinterval:10.00s
throughput:140.33fps,latency:6.69ms,timeinterval:10.01s
throughput:140.45fps,latency:6.68ms,timeinterval:10.00s
throughput:140.42fps,latency:6.68ms,timeinterval:10.01s
Done


DifferenceinFPSandlatency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importmatplotlib.pyplotasplt

TPUT=0
LAT=1
labels=["THROUGHPUThint","LATENCYhint"]

fig1,ax1=plt.subplots(1,1)
fig1.patch.set_visible(False)
ax1.axis("tight")
ax1.axis("off")

cell_text=[]
cell_text.append(
[
"%.2f%s"%(THROUGHPUT_hint_fps,"FPS"),
"%.2f%s"%(THROUGHPUT_hint_latency,"ms"),
]
)
cell_text.append(["%.2f%s"%(LATENCY_hint_fps,"FPS"),"%.2f%s"%(LATENCY_hint_latency,"ms")])

table=ax1.table(
cellText=cell_text,
colLabels=["FPS(Higherisbetter)","Latency(Lowerisbetter)"],
rowLabels=labels,
rowColours=["deepskyblue"]*2,
colColours=["deepskyblue"]*2,
cellLoc="center",
loc="upperleft",
)
table.auto_set_font_size(False)
table.set_fontsize(18)
table.auto_set_column_width(0)
table.auto_set_column_width(1)
table.scale(1,3)

fig1.tight_layout()
plt.show()



..image::auto-device-with-output_files/auto-device-with-output_27_0.png


..code::ipython3

#Outputthedifference.
width=0.4
fontsize=14

plt.rc("font",size=fontsize)
fig,ax=plt.subplots(1,2,figsize=(10,8))

rects1=ax[0].bar([0],THROUGHPUT_hint_fps,width,label=labels[TPUT],color="#557f2d")
rects2=ax[0].bar([width],LATENCY_hint_fps,width,label=labels[LAT])
ax[0].set_ylabel("framespersecond")
ax[0].set_xticks([width/2])
ax[0].set_xticklabels(["FPS"])
ax[0].set_xlabel("Higherisbetter")

rects1=ax[1].bar([0],THROUGHPUT_hint_latency,width,label=labels[TPUT],color="#557f2d")
rects2=ax[1].bar([width],LATENCY_hint_latency,width,label=labels[LAT])
ax[1].set_ylabel("milliseconds")
ax[1].set_xticks([width/2])
ax[1].set_xticklabels(["Latency(ms)"])
ax[1].set_xlabel("Lowerisbetter")

fig.suptitle("PerformanceHints")
fig.legend(labels,fontsize=fontsize)
fig.tight_layout()

plt.show()



..image::auto-device-with-output_files/auto-device-with-output_28_0.png

