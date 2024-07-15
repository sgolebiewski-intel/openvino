Post-TrainingQuantizationofPyTorchmodelswithNNCF
======================================================

ThegoalofthistutorialistodemonstratehowtousetheNNCF(Neural
NetworkCompressionFramework)8-bitquantizationinpost-trainingmode
(withoutthefine-tuningpipeline)tooptimizeaPyTorchmodelforthe
high-speedinferenceviaOpenVINO™Toolkit.Theoptimizationprocess
containsthefollowingsteps:

1.Evaluatetheoriginalmodel.
2.Transformtheoriginalmodeltoaquantizedone.
3.ExportoptimizedandoriginalmodelstoOpenVINOIR.
4.Compareperformanceoftheobtained``FP32``and``INT8``models.

ThistutorialusesaResNet-50model,pre-trainedonTinyImageNet,
whichcontains100000imagesof200classes(500foreachclass)
downsizedto64×64coloredimages.Thetutorialwilldemonstratethat
onlyatinypartofthedatasetisneededforthepost-training
quantization,notdemandingthefine-tuningofthemodel.

**NOTE**:ThisnotebookrequiresthataC++compilerisaccessibleon
thedefaultbinarysearchpathoftheOSyouarerunningthe
notebook.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Preparations<#preparations>`__

-`Imports<#imports>`__
-`Settings<#settings>`__
-`DownloadandPrepareTinyImageNet
dataset<#download-and-prepare-tiny-imagenet-dataset>`__
-`Helpersclassesandfunctions<#helpers-classes-and-functions>`__
-`Validationfunction<#validation-function>`__
-`Createandloadoriginaluncompressed
model<#create-and-load-original-uncompressed-model>`__
-`Createtrainandvalidation
DataLoaders<#create-train-and-validation-dataloaders>`__

-`Modelquantizationand
benchmarking<#model-quantization-and-benchmarking>`__

-`I.Evaluatetheloadedmodel<#i--evaluate-the-loaded-model>`__
-`II.Createandinitialize
quantization<#ii--create-and-initialize-quantization>`__
-`III.ConvertthemodelstoOpenVINOIntermediateRepresentation
(OpenVINO
IR)<#iii--convert-the-models-to-openvino-intermediate-representation-openvino-ir>`__
-`IV.CompareperformanceofINT8modelandFP32modelin
OpenVINO<#iv--compare-performance-of-int8-model-and-fp32-model-in-openvino>`__

Preparations
------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Installopenvinopackage
%pipinstall-q"openvino>=2024.0.0"torchtorchvisiontqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"nncf>=2.9.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importos
importtime
importzipfile
frompathlibimportPath
fromtypingimportList,Tuple

importnncf
importopenvinoasov

importtorch
fromtorchvision.datasetsimportImageFolder
fromtorchvision.modelsimportresnet50
importtorchvision.transformsastransforms

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


Settings
~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

torch_device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")
print(f"Using{torch_device}device")

MODEL_DIR=Path("model")
OUTPUT_DIR=Path("output")
BASE_MODEL_NAME="resnet50"
IMAGE_SIZE=[64,64]

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

#PathswherePyTorchandOpenVINOIRmodelswillbestored.
fp32_checkpoint_filename=Path(BASE_MODEL_NAME+"_fp32").with_suffix(".pth")
fp32_ir_path=OUTPUT_DIR/Path(BASE_MODEL_NAME+"_fp32").with_suffix(".xml")
int8_ir_path=OUTPUT_DIR/Path(BASE_MODEL_NAME+"_int8").with_suffix(".xml")


fp32_pth_url="https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/304_resnet50_fp32.pth"
download_file(fp32_pth_url,directory=MODEL_DIR,filename=fp32_checkpoint_filename)


..parsed-literal::

Usingcpudevice



..parsed-literal::

model/resnet50_fp32.pth:0%||0.00/91.5M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/pytorch-post-training-quantization-nncf/model/resnet50_fp32.pth')



DownloadandPrepareTinyImageNetdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

-100kimagesofshape3x64x64,
-200differentclasses:snake,spider,cat,truck,grasshopper,gull,
etc.

..code::ipython3

defdownload_tiny_imagenet_200(
output_dir:Path,
url:str="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
tarname:str="tiny-imagenet-200.zip",
):
archive_path=output_dir/tarname
download_file(url,directory=output_dir,filename=tarname)
zip_ref=zipfile.ZipFile(archive_path,"r")
zip_ref.extractall(path=output_dir)
zip_ref.close()
print(f"Successfullydownloadedandextracteddatasetto:{output_dir}")


defcreate_validation_dir(dataset_dir:Path):
VALID_DIR=dataset_dir/"val"
val_img_dir=VALID_DIR/"images"

fp=open(VALID_DIR/"val_annotations.txt","r")
data=fp.readlines()

val_img_dict={}
forlineindata:
words=line.split("\t")
val_img_dict[words[0]]=words[1]
fp.close()

forimg,folderinval_img_dict.items():
newpath=val_img_dir/folder
ifnotnewpath.exists():
os.makedirs(newpath)
if(val_img_dir/img).exists():
os.rename(val_img_dir/img,newpath/img)


DATASET_DIR=OUTPUT_DIR/"tiny-imagenet-200"
ifnotDATASET_DIR.exists():
download_tiny_imagenet_200(OUTPUT_DIR)
create_validation_dir(DATASET_DIR)



..parsed-literal::

output/tiny-imagenet-200.zip:0%||0.00/237M[00:00<?,?B/s]


..parsed-literal::

Successfullydownloadedandextracteddatasetto:output


Helpersclassesandfunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thecodebelowwillhelptocountaccuracyandvisualizevalidation
process.

..code::ipython3

classAverageMeter(object):
"""Computesandstorestheaverageandcurrentvalue"""

def__init__(self,name:str,fmt:str=":f"):
self.name=name
self.fmt=fmt
self.val=0
self.avg=0
self.sum=0
self.count=0

defupdate(self,val:float,n:int=1):
self.val=val
self.sum+=val*n
self.count+=n
self.avg=self.sum/self.count

def__str__(self):
fmtstr="{name}{val"+self.fmt+"}({avg"+self.fmt+"})"
returnfmtstr.format(**self.__dict__)


classProgressMeter(object):
"""Displaystheprogressofvalidationprocess"""

def__init__(self,num_batches:int,meters:List[AverageMeter],prefix:str=""):
self.batch_fmtstr=self._get_batch_fmtstr(num_batches)
self.meters=meters
self.prefix=prefix

defdisplay(self,batch:int):
entries=[self.prefix+self.batch_fmtstr.format(batch)]
entries+=[str(meter)formeterinself.meters]
print("\t".join(entries))

def_get_batch_fmtstr(self,num_batches:int):
num_digits=len(str(num_batches//1))
fmt="{:"+str(num_digits)+"d}"
return"["+fmt+"/"+fmt.format(num_batches)+"]"


defaccuracy(output:torch.Tensor,target:torch.Tensor,topk:Tuple[int]=(1,)):
"""Computestheaccuracyoverthektoppredictionsforthespecifiedvaluesofk"""
withtorch.no_grad():
maxk=max(topk)
batch_size=target.size(0)

_,pred=output.topk(maxk,1,True,True)
pred=pred.t()
correct=pred.eq(target.view(1,-1).expand_as(pred))

res=[]
forkintopk:
correct_k=correct[:k].reshape(-1).float().sum(0,keepdim=True)
res.append(correct_k.mul_(100.0/batch_size))

returnres

Validationfunction
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromtypingimportUnion
fromopenvino.runtime.ie_apiimportCompiledModel


defvalidate(
val_loader:torch.utils.data.DataLoader,
model:Union[torch.nn.Module,CompiledModel],
):
"""Computethemetricsusingdatafromval_loaderforthemodel"""
batch_time=AverageMeter("Time",":3.3f")
top1=AverageMeter("Acc@1",":2.2f")
top5=AverageMeter("Acc@5",":2.2f")
progress=ProgressMeter(len(val_loader),[batch_time,top1,top5],prefix="Test:")
start_time=time.time()
#Switchtoevaluatemode.
ifnotisinstance(model,CompiledModel):
model.eval()
model.to(torch_device)

withtorch.no_grad():
end=time.time()
fori,(images,target)inenumerate(val_loader):
images=images.to(torch_device)
target=target.to(torch_device)

#Computetheoutput.
ifisinstance(model,CompiledModel):
output_layer=model.output(0)
output=model(images)[output_layer]
output=torch.from_numpy(output)
else:
output=model(images)

#Measureaccuracyandrecordloss.
acc1,acc5=accuracy(output,target,topk=(1,5))
top1.update(acc1[0],images.size(0))
top5.update(acc5[0],images.size(0))

#Measureelapsedtime.
batch_time.update(time.time()-end)
end=time.time()

print_frequency=10
ifi%print_frequency==0:
progress.display(i)

print("*Acc@1{top1.avg:.3f}Acc@5{top5.avg:.3f}Totaltime:{total_time:.3f}".format(top1=top1,top5=top5,total_time=end-start_time))
returntop1.avg

Createandloadoriginaluncompressedmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ResNet-50fromthe`torchivision
repository<https://github.com/pytorch/vision>`__ispre-trainedon
ImageNetwithmorepredictionclassesthanTinyImageNet,sothemodel
isadjustedbyswappingthelastFClayertoonewithfeweroutput
values.

..code::ipython3

defcreate_model(model_path:Path):
"""CreatestheResNet-50modelandloadsthepretrainedweights"""
model=resnet50()
#UpdatethelastFClayerforTinyImageNetnumberofclasses.
NUM_CLASSES=200
model.fc=torch.nn.Linear(in_features=2048,out_features=NUM_CLASSES,bias=True)
model.to(torch_device)
ifmodel_path.exists():
checkpoint=torch.load(str(model_path),map_location="cpu")
model.load_state_dict(checkpoint["state_dict"],strict=True)
else:
raiseRuntimeError("Thereisnocheckpointtoload")
returnmodel


model=create_model(MODEL_DIR/fp32_checkpoint_filename)

CreatetrainandvalidationDataLoaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defcreate_dataloaders(batch_size:int=128):
"""Createstraindataloaderthatisusedforquantizationinitializationandvalidationdataloaderforcomputingthemodelaccruacy"""
train_dir=DATASET_DIR/"train"
val_dir=DATASET_DIR/"val"/"images"
normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
train_dataset=ImageFolder(
train_dir,
transforms.Compose(
[
transforms.Resize(IMAGE_SIZE),
transforms.ToTensor(),
normalize,
]
),
)
val_dataset=ImageFolder(
val_dir,
transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.ToTensor(),normalize]),
)

train_loader=torch.utils.data.DataLoader(
train_dataset,
batch_size=batch_size,
shuffle=True,
num_workers=0,
pin_memory=True,
sampler=None,
)

val_loader=torch.utils.data.DataLoader(
val_dataset,
batch_size=batch_size,
shuffle=False,
num_workers=0,
pin_memory=True,
)
returntrain_loader,val_loader


train_loader,val_loader=create_dataloaders()

Modelquantizationandbenchmarking
-----------------------------------

`backtotop⬆️<#table-of-contents>`__

Withthevalidationpipeline,modelfiles,anddata-loadingprocedures
formodelcalibrationnowprepared,it’stimetoproceedwiththeactual
post-trainingquantizationusingNNCF.

I.Evaluatetheloadedmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

acc1=validate(val_loader,model)
print(f"TestaccuracyofFP32model:{acc1:.3f}")


..parsed-literal::

Test:[0/79]	Time0.233(0.233)	Acc@181.25(81.25)	Acc@592.19(92.19)
Test:[10/79]	Time0.223(0.225)	Acc@156.25(66.97)	Acc@586.72(87.50)
Test:[20/79]	Time0.224(0.227)	Acc@167.97(64.29)	Acc@585.16(87.35)
Test:[30/79]	Time0.222(0.225)	Acc@153.12(62.37)	Acc@577.34(85.33)
Test:[40/79]	Time0.227(0.224)	Acc@167.19(60.86)	Acc@590.62(84.51)
Test:[50/79]	Time0.219(0.224)	Acc@160.16(60.80)	Acc@588.28(84.42)
Test:[60/79]	Time0.222(0.225)	Acc@166.41(60.46)	Acc@586.72(83.79)
Test:[70/79]	Time0.229(0.225)	Acc@152.34(60.21)	Acc@580.47(83.33)
*Acc@160.740Acc@583.960Totaltime:17.538
TestaccuracyofFP32model:60.740


II.Createandinitializequantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

NNCFenablespost-trainingquantizationbyaddingthequantization
layersintothemodelgraphandthenusingasubsetofthetraining
datasettoinitializetheparametersoftheseadditionalquantization
layers.Theframeworkisdesignedsothatmodificationstoyouroriginal
trainingcodeareminor.Quantizationisthesimplestscenarioand
requiresafewmodifications.FormoreinformationaboutNNCFPost
TrainingQuantization(PTQ)API,refertothe`BasicQuantizationFlow
Guide<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__.

1.Createatransformationfunctionthatacceptsasamplefromthe
datasetandreturnsdatasuitableformodelinference.Thisenables
thecreationofaninstanceofthenncf.Datasetclass,which
representsthecalibrationdataset(basedonthetrainingdataset)
necessaryforpost-trainingquantization.

..code::ipython3

deftransform_fn(data_item):
images,_=data_item
returnimages


calibration_dataset=nncf.Dataset(train_loader,transform_fn)

2.Createaquantizedmodelfromthepre-trained``FP32``modelandthe
calibrationdataset.

..code::ipython3

quantized_model=nncf.quantize(model,calibration_dataset)


..parsed-literal::

2024-07-1301:43:03.812257:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:43:03.845918:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:43:04.402869:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Compilingandloadingtorchextension:quantized_functions_cpu...
INFO:nncf:Finishedloadingtorchextension:quantized_functions_cpu



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



3.Evaluatethenewmodelonthevalidationsetafterinitializationof
quantization.Theaccuracyshouldbeclosetotheaccuracyofthe
floating-point``FP32``modelforasimplecaseliketheonebeing
demonstratednow.

..code::ipython3

acc1=validate(val_loader,quantized_model)
print(f"AccuracyofinitializedINT8model:{acc1:.3f}")


..parsed-literal::

Test:[0/79]	Time0.436(0.436)	Acc@180.47(80.47)	Acc@591.41(91.41)
Test:[10/79]	Time0.409(0.415)	Acc@154.69(66.69)	Acc@587.50(87.86)
Test:[20/79]	Time0.406(0.413)	Acc@169.53(63.91)	Acc@585.16(87.35)
Test:[30/79]	Time0.410(0.412)	Acc@150.78(62.25)	Acc@574.22(85.26)
Test:[40/79]	Time0.410(0.412)	Acc@168.75(60.79)	Acc@589.84(84.34)
Test:[50/79]	Time0.410(0.412)	Acc@157.81(60.63)	Acc@587.50(84.21)
Test:[60/79]	Time0.412(0.412)	Acc@166.41(60.36)	Acc@585.94(83.61)
Test:[70/79]	Time0.411(0.411)	Acc@153.91(60.07)	Acc@579.69(83.23)
*Acc@160.570Acc@583.850Totaltime:32.242
AccuracyofinitializedINT8model:60.570


ItshouldbenotedthattheinferencetimeforthequantizedPyTorch
modelislongerthanthatoftheoriginalmodel,asfakequantizersare
addedtothemodelbyNNCF.However,themodel’sperformancewill
significantlyimprovewhenitisintheOpenVINOIntermediate
Representation(IR)format.

III.ConvertthemodelstoOpenVINOIntermediateRepresentation(OpenVINOIR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ToconvertthePytorchmodelstoOpenVINOIR,useModelConversion
PythonAPI.Themodelswillbesavedtothe‘OUTPUT’directoryforlater
benchmarking.

Formoreinformationaboutmodelconversion,refertothis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

..code::ipython3

dummy_input=torch.randn(128,3,*IMAGE_SIZE)

model_ir=ov.convert_model(model,example_input=dummy_input,input=[-1,3,*IMAGE_SIZE])

ov.save_model(model_ir,fp32_ir_path)


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.


..parsed-literal::

['x']


..code::ipython3

quantized_model_ir=ov.convert_model(quantized_model,example_input=dummy_input,input=[-1,3,*IMAGE_SIZE])

ov.save_model(quantized_model_ir,int8_ir_path)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/torch/quantization/layers.py:340:TracerWarning:ConvertingatensortoaPythonnumbermightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
returnself._level_low.item()
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/torch/quantization/layers.py:348:TracerWarning:ConvertingatensortoaPythonnumbermightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
returnself._level_high.item()
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:1116:TracerWarning:Outputnr1.ofthetracedfunctiondoesnotmatchthecorrespondingoutputofthePythonfunction.Detailederror:
Tensor-likesarenotclose!

Mismatchedelements:25553/25600(99.8%)
Greatestabsolutedifference:0.1654798984527588atindex(93,149)(upto1e-05allowed)
Greatestrelativedifference:52.04605773950292atindex(14,168)(upto1e-05allowed)
_check_trace(


..parsed-literal::

['x']


SelectinferencedeviceforOpenVINO

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



EvaluatetheFP32andINT8models.

..code::ipython3

core=ov.Core()
fp32_compiled_model=core.compile_model(model_ir,device.value)
acc1=validate(val_loader,fp32_compiled_model)
print(f"AccuracyofFP32IRmodel:{acc1:.3f}")


..parsed-literal::

Test:[0/79]	Time0.189(0.189)	Acc@181.25(81.25)	Acc@592.19(92.19)
Test:[10/79]	Time0.141(0.145)	Acc@156.25(66.97)	Acc@586.72(87.50)
Test:[20/79]	Time0.140(0.142)	Acc@167.97(64.29)	Acc@585.16(87.35)
Test:[30/79]	Time0.140(0.141)	Acc@153.12(62.37)	Acc@577.34(85.33)
Test:[40/79]	Time0.139(0.141)	Acc@167.19(60.86)	Acc@590.62(84.51)
Test:[50/79]	Time0.139(0.141)	Acc@160.16(60.80)	Acc@588.28(84.42)
Test:[60/79]	Time0.140(0.140)	Acc@166.41(60.46)	Acc@586.72(83.79)
Test:[70/79]	Time0.139(0.140)	Acc@152.34(60.21)	Acc@580.47(83.33)
*Acc@160.740Acc@583.960Totaltime:10.971
AccuracyofFP32IRmodel:60.740


..code::ipython3

int8_compiled_model=core.compile_model(quantized_model_ir,device.value)
acc1=validate(val_loader,int8_compiled_model)
print(f"AccuracyofINT8IRmodel:{acc1:.3f}")


..parsed-literal::

Test:[0/79]	Time0.150(0.150)	Acc@180.47(80.47)	Acc@591.41(91.41)
Test:[10/79]	Time0.079(0.085)	Acc@152.34(66.48)	Acc@586.72(87.78)
Test:[20/79]	Time0.078(0.082)	Acc@170.31(64.10)	Acc@585.16(87.28)
Test:[30/79]	Time0.078(0.081)	Acc@151.56(62.40)	Acc@573.44(85.11)
Test:[40/79]	Time0.077(0.080)	Acc@168.75(60.94)	Acc@589.84(84.26)
Test:[50/79]	Time0.078(0.080)	Acc@159.38(60.78)	Acc@587.50(84.13)
Test:[60/79]	Time0.077(0.079)	Acc@165.62(60.49)	Acc@585.94(83.53)
Test:[70/79]	Time0.081(0.079)	Acc@153.91(60.24)	Acc@579.69(83.14)
*Acc@160.700Acc@583.720Totaltime:6.203
AccuracyofINT8IRmodel:60.700


IV.CompareperformanceofINT8modelandFP32modelinOpenVINO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Finally,measuretheinferenceperformanceofthe``FP32``and``INT8``
models,using`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
-aninferenceperformancemeasurementtoolinOpenVINO.Bydefault,
BenchmarkToolrunsinferencefor60secondsinasynchronousmodeon
CPU.Itreturnsinferencespeedaslatency(millisecondsperimage)and
throughput(framespersecond)values.

**NOTE**:Thisnotebookrunsbenchmark_appfor15secondstogivea
quickindicationofperformance.Formoreaccurateperformance,itis
recommendedtorunbenchmark_appinaterminal/commandpromptafter
closingotherapplications.Run``benchmark_app-mmodel.xml-dCPU``
tobenchmarkasyncinferenceonCPUforoneminute.ChangeCPUtoGPU
tobenchmarkonGPU.Run``benchmark_app--help``toseeanoverview
ofallcommand-lineoptions.

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

defparse_benchmark_output(benchmark_output:str):
"""Printstheoutputfrombenchmark_appinhuman-readableformat"""
parsed_output=[lineforlineinbenchmark_outputif"FPS"inline]
print(*parsed_output,sep="\n")


print("BenchmarkFP32model(OpenVINOIR)")
benchmark_output=!benchmark_app-m"$fp32_ir_path"-d$device.value-apiasync-t15-shape"[1,3,512,512]"
parse_benchmark_output(benchmark_output)

print("BenchmarkINT8model(OpenVINOIR)")
benchmark_output=!benchmark_app-m"$int8_ir_path"-d$device.value-apiasync-t15-shape"[1,3,512,512]"
parse_benchmark_output(benchmark_output)

print("BenchmarkFP32model(OpenVINOIR)synchronously")
benchmark_output=!benchmark_app-m"$fp32_ir_path"-d$device.value-apisync-t15-shape"[1,3,512,512]"
parse_benchmark_output(benchmark_output)

print("BenchmarkINT8model(OpenVINOIR)synchronously")
benchmark_output=!benchmark_app-m"$int8_ir_path"-d$device.value-apisync-t15-shape"[1,3,512,512]"
parse_benchmark_output(benchmark_output)


..parsed-literal::

BenchmarkFP32model(OpenVINOIR)
[INFO]Throughput:38.73FPS
BenchmarkINT8model(OpenVINOIR)
[INFO]Throughput:153.53FPS
BenchmarkFP32model(OpenVINOIR)synchronously
[INFO]Throughput:39.67FPS
BenchmarkINT8model(OpenVINOIR)synchronously
[INFO]Throughput:136.23FPS


ShowdeviceInformationforreference:

..code::ipython3

core=ov.Core()
devices=core.available_devices

fordevice_nameindevices:
device_full_name=core.get_property(device_name,"FULL_DEVICE_NAME")
print(f"{device_name}:{device_full_name}")


..parsed-literal::

CPU:Intel(R)Core(TM)i9-10920XCPU@3.50GHz

