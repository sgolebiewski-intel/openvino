QuantizationAwareTrainingwithNNCF,usingPyTorchframework
==============================================================

Thisnotebookisbasedon`ImageNettrainingin
PyTorch<https://github.com/pytorch/examples/blob/master/imagenet/main.py>`__.

ThegoalofthisnotebookistodemonstratehowtousetheNeural
NetworkCompressionFramework
`NNCF<https://github.com/openvinotoolkit/nncf>`__8-bitquantizationto
optimizeaPyTorchmodelforinferencewithOpenVINOToolkit.The
optimizationprocesscontainsthefollowingsteps:

-Transformingtheoriginal``FP32``modelto``INT8``
-Usingfine-tuningtoimprovetheaccuracy.
-ExportingoptimizedandoriginalmodelstoOpenVINOIR
-Measuringandcomparingtheperformanceofmodels.

Formoreadvancedusage,refertothese
`examples<https://github.com/openvinotoolkit/nncf/tree/develop/examples>`__.

ThistutorialusestheResNet-18modelwiththeTinyImageNet-200
dataset.ResNet-18istheversionofResNetmodelsthatcontainsthe
fewestlayers(18).TinyImageNet-200isasubsetofthelargerImageNet
datasetwithsmallerimages.Thedatasetwillbedownloadedinthe
notebook.Usingthesmallermodelanddatasetwillspeeduptrainingand
downloadtime.ToseeotherResNetmodels,visit`PyTorch
hub<https://pytorch.org/hub/pytorch_vision_resnet/>`__.

**NOTE**:ThisnotebookrequiresaC++compilerforcompilingPyTorch
customoperationsforquantization.ForWindowswerecommendto
installVisualStudiowithC++support,youcanfindinstruction
`here<https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170>`__.
ForMacOS``xcode-select--install``commandinstallsmanydeveloper
tools,includingC++.ForLinuxyoucaninstallgccwithyour
distribution’spackagemanager.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`ImportsandSettings<#imports-and-settings>`__
-`Pre-trainFloating-PointModel<#pre-train-floating-point-model>`__

-`TrainFunction<#train-function>`__
-`ValidateFunction<#validate-function>`__
-`Helpers<#helpers>`__
-`GetaPre-trainedFP32Model<#get-a-pre-trained-fp32-model>`__

-`CreateandInitialize
Quantization<#create-and-initialize-quantization>`__
-`Fine-tunetheCompressedModel<#fine-tune-the-compressed-model>`__
-`ExportINT8ModeltoOpenVINO
IR<#export-int8-model-to-openvino-ir>`__
-`BenchmarkModelPerformancebyComputingInference
Time<#benchmark-model-performance-by-computing-inference-time>`__

..code::ipython3

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"openvino>=2024.0.0""torch""torchvision""tqdm"
%pipinstall-q"nncf>=2.9.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


ImportsandSettings
--------------------

`backtotop⬆️<#table-of-contents>`__

OnWindows,addtherequiredC++directoriestothesystemPATH.

ImportNNCFandallauxiliarypackagesfromyourPythoncode.Setaname
forthemodel,andtheimagewidthandheightthatwillbeusedforthe
network.AlsodefinepathswherePyTorchandOpenVINOIRversionsofthe
modelswillbestored.

**NOTE**:AllNNCFloggingmessagesbelowERRORlevel(INFOand
WARNING)aredisabledtosimplifythetutorial.Forproductionuse,
itisrecommendedtoenableloggingbyremoving
``set_log_level(logging.ERROR)``.

..code::ipython3

importtime
importwarnings#Todisablewarningsonexportmodel
importzipfile
frompathlibimportPath

importtorch

importtorch.nnasnn
importtorch.nn.parallel
importtorch.optim
importtorch.utils.data
importtorch.utils.data.distributed
importtorchvision.datasetsasdatasets
importtorchvision.modelsasmodels
importtorchvision.transformsastransforms

importopenvinoasov
fromtorch.jitimportTracerWarning

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

torch.manual_seed(0)
device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")
print(f"Using{device}device")

MODEL_DIR=Path("model")
OUTPUT_DIR=Path("output")
DATA_DIR=Path("data")
BASE_MODEL_NAME="resnet18"
image_size=64

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

#PathswherePyTorchandOpenVINOIRmodelswillbestored.
fp32_pth_path=Path(MODEL_DIR/(BASE_MODEL_NAME+"_fp32")).with_suffix(".pth")
fp32_ir_path=fp32_pth_path.with_suffix(".xml")
int8_ir_path=Path(MODEL_DIR/(BASE_MODEL_NAME+"_int8")).with_suffix(".xml")

#ItispossibletotrainFP32modelfromscratch,butitmightbeslow.Therefore,thepre-trainedweightsaredownloadedbydefault.
pretrained_on_tiny_imagenet=True
fp32_pth_url="https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth"
download_file(fp32_pth_url,directory=MODEL_DIR,filename=fp32_pth_path.name)


..parsed-literal::

Usingcpudevice



..parsed-literal::

model/resnet18_fp32.pth:0%||0.00/43.1M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/pytorch-quantization-aware-training/model/resnet18_fp32.pth')



DownloadTinyImageNetdataset

-100kimagesofshape3x64x64
-200differentclasses:snake,spider,cat,truck,grasshopper,gull,
etc.

..code::ipython3

defdownload_tiny_imagenet_200(
data_dir:Path,
url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
tarname="tiny-imagenet-200.zip",
):
archive_path=data_dir/tarname
download_file(url,directory=data_dir,filename=tarname)
zip_ref=zipfile.ZipFile(archive_path,"r")
zip_ref.extractall(path=data_dir)
zip_ref.close()


defprepare_tiny_imagenet_200(dataset_dir:Path):
#Formatvalidationsetthesamewayastrainsetisformatted.
val_data_dir=dataset_dir/"val"
val_annotations_file=val_data_dir/"val_annotations.txt"
withopen(val_annotations_file,"r")asf:
val_annotation_data=map(lambdaline:line.split("\t")[:2],f.readlines())
val_images_dir=val_data_dir/"images"
forimage_filename,image_labelinval_annotation_data:
from_image_filepath=val_images_dir/image_filename
to_image_dir=val_data_dir/image_label
ifnotto_image_dir.exists():
to_image_dir.mkdir()
to_image_filepath=to_image_dir/image_filename
from_image_filepath.rename(to_image_filepath)
val_annotations_file.unlink()
val_images_dir.rmdir()


DATASET_DIR=DATA_DIR/"tiny-imagenet-200"
ifnotDATASET_DIR.exists():
download_tiny_imagenet_200(DATA_DIR)
prepare_tiny_imagenet_200(DATASET_DIR)
print(f"Successfullydownloadedandprepareddatasetat:{DATASET_DIR}")



..parsed-literal::

data/tiny-imagenet-200.zip:0%||0.00/237M[00:00<?,?B/s]


..parsed-literal::

Successfullydownloadedandprepareddatasetat:data/tiny-imagenet-200


Pre-trainFloating-PointModel
------------------------------

`backtotop⬆️<#table-of-contents>`__

UsingNNCFformodelcompressionassumesthatapre-trainedmodelanda
trainingpipelinearealreadyinuse.

Thistutorialdemonstratesonepossibletrainingpipeline:aResNet-18
modelpre-trainedon1000classesfromImageNetisfine-tunedwith200
classesfromTiny-ImageNet.

Subsequently,thetrainingandvalidationfunctionswillbereusedasis
forquantization-awaretraining.

TrainFunction
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

deftrain(train_loader,model,criterion,optimizer,epoch):
batch_time=AverageMeter("Time",":3.3f")
losses=AverageMeter("Loss",":2.3f")
top1=AverageMeter("Acc@1",":2.2f")
top5=AverageMeter("Acc@5",":2.2f")
progress=ProgressMeter(
len(train_loader),
[batch_time,losses,top1,top5],
prefix="Epoch:[{}]".format(epoch),
)

#Switchtotrainmode.
model.train()

end=time.time()
fori,(images,target)inenumerate(train_loader):
images=images.to(device)
target=target.to(device)

#Computeoutput.
output=model(images)
loss=criterion(output,target)

#Measureaccuracyandrecordloss.
acc1,acc5=accuracy(output,target,topk=(1,5))
losses.update(loss.item(),images.size(0))
top1.update(acc1[0],images.size(0))
top5.update(acc5[0],images.size(0))

#Computegradientanddooptstep.
optimizer.zero_grad()
loss.backward()
optimizer.step()

#Measureelapsedtime.
batch_time.update(time.time()-end)
end=time.time()

print_frequency=50
ifi%print_frequency==0:
progress.display(i)

ValidateFunction
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defvalidate(val_loader,model,criterion):
batch_time=AverageMeter("Time",":3.3f")
losses=AverageMeter("Loss",":2.3f")
top1=AverageMeter("Acc@1",":2.2f")
top5=AverageMeter("Acc@5",":2.2f")
progress=ProgressMeter(len(val_loader),[batch_time,losses,top1,top5],prefix="Test:")

#Switchtoevaluatemode.
model.eval()

withtorch.no_grad():
end=time.time()
fori,(images,target)inenumerate(val_loader):
images=images.to(device)
target=target.to(device)

#Computeoutput.
output=model(images)
loss=criterion(output,target)

#Measureaccuracyandrecordloss.
acc1,acc5=accuracy(output,target,topk=(1,5))
losses.update(loss.item(),images.size(0))
top1.update(acc1[0],images.size(0))
top5.update(acc5[0],images.size(0))

#Measureelapsedtime.
batch_time.update(time.time()-end)
end=time.time()

print_frequency=10
ifi%print_frequency==0:
progress.display(i)

print("*Acc@1{top1.avg:.3f}Acc@5{top5.avg:.3f}".format(top1=top1,top5=top5))
returntop1.avg

Helpers
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

classAverageMeter(object):
"""Computesandstorestheaverageandcurrentvalue"""

def__init__(self,name,fmt=":f"):
self.name=name
self.fmt=fmt
self.reset()

defreset(self):
self.val=0
self.avg=0
self.sum=0
self.count=0

defupdate(self,val,n=1):
self.val=val
self.sum+=val*n
self.count+=n
self.avg=self.sum/self.count

def__str__(self):
fmtstr="{name}{val"+self.fmt+"}({avg"+self.fmt+"})"
returnfmtstr.format(**self.__dict__)


classProgressMeter(object):
def__init__(self,num_batches,meters,prefix=""):
self.batch_fmtstr=self._get_batch_fmtstr(num_batches)
self.meters=meters
self.prefix=prefix

defdisplay(self,batch):
entries=[self.prefix+self.batch_fmtstr.format(batch)]
entries+=[str(meter)formeterinself.meters]
print("\t".join(entries))

def_get_batch_fmtstr(self,num_batches):
num_digits=len(str(num_batches//1))
fmt="{:"+str(num_digits)+"d}"
return"["+fmt+"/"+fmt.format(num_batches)+"]"


defaccuracy(output,target,topk=(1,)):
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

GetaPre-trainedFP32Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Аpre-trainedfloating-pointmodelisaprerequisiteforquantization.
Itcanbeobtainedbytuningfromscratchwiththecodebelow.However,
thisusuallytakesalotoftime.Therefore,thiscodehasalreadybeen
runandreceivedgoodenoughweightsafter4epochs(forthesakeof
simplicity,tuningwasnotdoneuntilthebestaccuracy).Bydefault,
thisnotebookjustloadstheseweightswithoutlaunchingtraining.To
trainthemodelyourselfonamodelpre-trainedonImageNet,set
``pretrained_on_tiny_imagenet=False``intheImportsandSettings
sectionatthetopofthisnotebook.

..code::ipython3

num_classes=200#200isforTinyImageNet,defaultis1000forImageNet
init_lr=1e-4
batch_size=128
epochs=4

model=models.resnet18(pretrained=notpretrained_on_tiny_imagenet)
#UpdatethelastFClayerforTinyImageNetnumberofclasses.
model.fc=nn.Linear(in_features=512,out_features=num_classes,bias=True)
model.to(device)

#Dataloadingcode.
train_dir=DATASET_DIR/"train"
val_dir=DATASET_DIR/"val"
normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

train_dataset=datasets.ImageFolder(
train_dir,
transforms.Compose(
[
transforms.Resize(image_size),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
normalize,
]
),
)
val_dataset=datasets.ImageFolder(
val_dir,
transforms.Compose(
[
transforms.Resize(image_size),
transforms.ToTensor(),
normalize,
]
),
)

train_loader=torch.utils.data.DataLoader(
train_dataset,
batch_size=batch_size,
shuffle=True,
num_workers=0,
pin_memory=True,
sampler=None,
)

val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)

#Definelossfunction(criterion)andoptimizer.
criterion=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=init_lr)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:208:UserWarning:Theparameter'pretrained'isdeprecatedsince0.13andmayberemovedinthefuture,pleaseuse'weights'instead.
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:223:UserWarning:Argumentsotherthanaweightenumor`None`for'weights'aredeprecatedsince0.13andmayberemovedinthefuture.Thecurrentbehaviorisequivalenttopassing`weights=None`.
warnings.warn(msg)


..code::ipython3

ifpretrained_on_tiny_imagenet:
#
#**WARNING:The`torch.load`functionalityusesPython'spicklingmodulethat
#maybeusedtoperformarbitrarycodeexecutionduringunpickling.Onlyloaddatathatyou
#trust.
#
checkpoint=torch.load(str(fp32_pth_path),map_location="cpu")
model.load_state_dict(checkpoint["state_dict"],strict=True)
acc1_fp32=checkpoint["acc1"]
else:
best_acc1=0
#Trainingloop.
forepochinrange(0,epochs):
#Runasingletrainingepoch.
train(train_loader,model,criterion,optimizer,epoch)

#Evaluateonvalidationset.
acc1=validate(val_loader,model,criterion)

is_best=acc1>best_acc1
best_acc1=max(acc1,best_acc1)

ifis_best:
checkpoint={"state_dict":model.state_dict(),"acc1":acc1}
torch.save(checkpoint,fp32_pth_path)
acc1_fp32=best_acc1

print(f"AccuracyofFP32model:{acc1_fp32:.3f}")


..parsed-literal::

AccuracyofFP32model:55.520


Exportthe``FP32``modeltoOpenVINO™IntermediateRepresentation,to
benchmarkitincomparisonwiththe``INT8``model.

..code::ipython3

dummy_input=torch.randn(1,3,image_size,image_size).to(device)

ov_model=ov.convert_model(model,example_input=dummy_input,input=[1,3,image_size,image_size])
ov.save_model(ov_model,fp32_ir_path,compress_to_fp16=False)
print(f"FP32modelwasexportedto{fp32_ir_path}.")


..parsed-literal::

['x']
FP32modelwasexportedtomodel/resnet18_fp32.xml.


CreateandInitializeQuantization
----------------------------------

`backtotop⬆️<#table-of-contents>`__

NNCFenablescompression-awaretrainingbyintegratingintoregular
trainingpipelines.Theframeworkisdesignedsothatmodificationsto
youroriginaltrainingcodeareminor.Quantizationrequiresonly2
modifications.

1.Createaquantizationdataloaderwithbatchsizeequaltooneand
wrapitbythe``nncf.Dataset``,specifyingatransformationfunction
whichpreparesinputdatatofitintomodelduringquantization.In
ourcase,topickinputtensorfrompair(inputtensorandlabel).

..code::ipython3

importnncf


deftransform_fn(data_item):
returndata_item[0]


#Creatingseparatedataloaderwithbatchsize=1
#asdataloaderswithbatches>1isnotsupportedyet.
quantization_loader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)

quantization_dataset=nncf.Dataset(quantization_loader,transform_fn)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


2.Run``nncf.quantize``forGettinganOptimizedModel.

``nncf.quantize``functionacceptsmodelandpreparedquantization
datasetforperformingbasicquantization.Optionally,additional
parameterslike``subset_size``,``preset``,``ignored_scope``canbe
providedtoimprovequantizationresultifapplicable.Moredetails
aboutsupportedparameterscanbefoundonthis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html#tune-quantization-parameters>`__

..code::ipython3

quantized_model=nncf.quantize(model,quantization_dataset)


..parsed-literal::

2024-07-1301:48:22.717129:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:48:22.753125:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:48:23.281403:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



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



Evaluatethenewmodelonthevalidationsetafterinitializationof
quantization.Theaccuracyshouldbeclosetotheaccuracyofthe
floating-point``FP32``modelforasimplecaseliketheonebeing
demonstratedhere.

..code::ipython3

acc1=validate(val_loader,quantized_model,criterion)
print(f"AccuracyofinitializedINT8model:{acc1:.3f}")


..parsed-literal::

Test:[0/79]	Time0.223(0.223)	Loss1.005(1.005)	Acc@178.91(78.91)	Acc@588.28(88.28)
Test:[10/79]	Time0.172(0.176)	Loss1.992(1.625)	Acc@144.53(60.37)	Acc@579.69(83.66)
Test:[20/79]	Time0.179(0.173)	Loss1.814(1.705)	Acc@160.94(58.04)	Acc@580.47(82.66)
Test:[30/79]	Time0.168(0.173)	Loss2.287(1.795)	Acc@150.78(56.48)	Acc@568.75(80.97)
Test:[40/79]	Time0.171(0.173)	Loss1.615(1.832)	Acc@160.94(55.43)	Acc@582.81(80.43)
Test:[50/79]	Time0.172(0.173)	Loss1.952(1.833)	Acc@157.03(55.51)	Acc@575.00(80.16)
Test:[60/79]	Time0.171(0.173)	Loss1.794(1.856)	Acc@157.03(55.16)	Acc@584.38(79.84)
Test:[70/79]	Time0.171(0.173)	Loss2.371(1.889)	Acc@146.88(54.68)	Acc@574.22(79.14)
*Acc@155.040Acc@579.730
AccuracyofinitializedINT8model:55.040


Fine-tunetheCompressedModel
------------------------------

`backtotop⬆️<#table-of-contents>`__

Atthisstep,aregularfine-tuningprocessisappliedtofurther
improvequantizedmodelaccuracy.Normally,severalepochsoftuningare
requiredwithasmalllearningrate,thesamethatisusuallyusedat
theendofthetrainingoftheoriginalmodel.Nootherchangesinthe
trainingpipelinearerequired.Hereisasimpleexample.

..code::ipython3

compression_lr=init_lr/10
optimizer=torch.optim.Adam(quantized_model.parameters(),lr=compression_lr)

#TrainforoneepochwithNNCF.
train(train_loader,quantized_model,criterion,optimizer,epoch=0)

#EvaluateonvalidationsetafterQuantization-AwareTraining(QATcase).
acc1_int8=validate(val_loader,quantized_model,criterion)

print(f"AccuracyoftunedINT8model:{acc1_int8:.3f}")
print(f"AccuracydropoftunedINT8modeloverpre-trainedFP32model:{acc1_fp32-acc1_int8:.3f}")


..parsed-literal::

Epoch:[0][0/782]	Time0.433(0.433)	Loss1.029(1.029)	Acc@175.00(75.00)	Acc@590.62(90.62)
Epoch:[0][50/782]	Time0.380(0.380)	Loss0.672(0.823)	Acc@187.50(79.81)	Acc@594.53(93.84)
Epoch:[0][100/782]	Time0.376(0.379)	Loss0.661(0.799)	Acc@185.94(80.31)	Acc@598.44(94.41)
Epoch:[0][150/782]	Time0.368(0.377)	Loss0.632(0.797)	Acc@185.94(80.50)	Acc@594.53(94.24)
Epoch:[0][200/782]	Time0.380(0.377)	Loss0.742(0.790)	Acc@181.25(80.69)	Acc@594.53(94.31)
Epoch:[0][250/782]	Time0.380(0.377)	Loss0.815(0.785)	Acc@181.25(80.80)	Acc@593.75(94.34)
Epoch:[0][300/782]	Time0.365(0.376)	Loss0.878(0.781)	Acc@176.56(80.87)	Acc@592.19(94.37)
Epoch:[0][350/782]	Time0.372(0.376)	Loss0.746(0.774)	Acc@182.03(81.03)	Acc@593.75(94.44)
Epoch:[0][400/782]	Time0.378(0.376)	Loss0.766(0.772)	Acc@179.69(81.12)	Acc@596.88(94.42)
Epoch:[0][450/782]	Time0.379(0.376)	Loss0.865(0.768)	Acc@177.34(81.28)	Acc@593.75(94.48)
Epoch:[0][500/782]	Time0.372(0.376)	Loss0.526(0.765)	Acc@189.06(81.33)	Acc@597.66(94.53)
Epoch:[0][550/782]	Time0.369(0.376)	Loss0.826(0.762)	Acc@179.69(81.39)	Acc@592.19(94.55)
Epoch:[0][600/782]	Time0.367(0.376)	Loss0.644(0.761)	Acc@185.94(81.45)	Acc@595.31(94.55)
Epoch:[0][650/782]	Time0.371(0.376)	Loss0.585(0.757)	Acc@181.25(81.57)	Acc@598.44(94.59)
Epoch:[0][700/782]	Time0.370(0.376)	Loss0.578(0.755)	Acc@186.72(81.65)	Acc@596.88(94.60)
Epoch:[0][750/782]	Time0.381(0.376)	Loss0.783(0.753)	Acc@179.69(81.69)	Acc@595.31(94.63)
Test:[0/79]	Time0.150(0.150)	Loss1.063(1.063)	Acc@174.22(74.22)	Acc@587.50(87.50)
Test:[10/79]	Time0.150(0.149)	Loss1.785(1.514)	Acc@150.78(63.21)	Acc@581.25(84.38)
Test:[20/79]	Time0.150(0.150)	Loss1.582(1.588)	Acc@164.84(61.09)	Acc@582.03(84.04)
Test:[30/79]	Time0.148(0.150)	Loss2.103(1.691)	Acc@155.47(59.30)	Acc@571.09(82.41)
Test:[40/79]	Time0.149(0.150)	Loss1.597(1.745)	Acc@164.06(57.89)	Acc@583.59(81.48)
Test:[50/79]	Time0.148(0.150)	Loss1.895(1.751)	Acc@153.91(57.74)	Acc@577.34(81.20)
Test:[60/79]	Time0.151(0.150)	Loss1.566(1.783)	Acc@165.62(57.18)	Acc@584.38(80.75)
Test:[70/79]	Time0.151(0.150)	Loss2.457(1.811)	Acc@145.31(56.65)	Acc@573.44(80.27)
*Acc@157.080Acc@580.940
AccuracyoftunedINT8model:57.080
AccuracydropoftunedINT8modeloverpre-trainedFP32model:-1.560


ExportINT8ModeltoOpenVINOIR
--------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ifnotint8_ir_path.exists():
warnings.filterwarnings("ignore",category=TracerWarning)
warnings.filterwarnings("ignore",category=UserWarning)
#ExportINT8modeltoOpenVINO™IR
ov_model=ov.convert_model(quantized_model,example_input=dummy_input,input=[1,3,image_size,image_size])
ov.save_model(ov_model,int8_ir_path)
print(f"INT8modelexportedto{int8_ir_path}.")


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.
['x']
INT8modelexportedtomodel/resnet18_int8.xml.


BenchmarkModelPerformancebyComputingInferenceTime
-------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Finally,measuretheinferenceperformanceofthe``FP32``and``INT8``
models,using`Benchmark
Tool<https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
-inferenceperformancemeasurementtoolinOpenVINO.Bydefault,
BenchmarkToolrunsinferencefor60secondsinasynchronousmodeon
CPU.Itreturnsinferencespeedaslatency(millisecondsperimage)and
throughput(framespersecond)values.

**NOTE**:Thisnotebookruns``benchmark_app``for15secondstogive
aquickindicationofperformance.Formoreaccurateperformance,it
isrecommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.Run
``benchmark_app-mmodel.xml-dCPU``tobenchmarkasyncinferenceon
CPUforoneminute.ChangeCPUtoGPUtobenchmarkonGPU.Run
``benchmark_app--help``toseeanoverviewofallcommand-line
options.

..code::ipython3

importipywidgetsaswidgets

#InitializeOpenVINOruntime
core=ov.Core()
device=widgets.Dropdown(
options=core.available_devices,
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU',),value='CPU')



..code::ipython3

defparse_benchmark_output(benchmark_output):
parsed_output=[lineforlineinbenchmark_outputif"FPS"inline]
print(*parsed_output,sep="\n")


print("BenchmarkFP32model(IR)")
benchmark_output=!benchmark_app-m$fp32_ir_path-d$device.value-apiasync-t15
parse_benchmark_output(benchmark_output)

print("BenchmarkINT8model(IR)")
benchmark_output=!benchmark_app-m$int8_ir_path-d$device.value-apiasync-t15
parse_benchmark_output(benchmark_output)


..parsed-literal::

BenchmarkFP32model(IR)
[INFO]Throughput:2933.90FPS
BenchmarkINT8model(IR)
[INFO]Throughput:11862.93FPS


ShowDeviceInformationforreference.

..code::ipython3

core.get_property(device.value,"FULL_DEVICE_NAME")




..parsed-literal::

'Intel(R)Core(TM)i9-10920XCPU@3.50GHz'


