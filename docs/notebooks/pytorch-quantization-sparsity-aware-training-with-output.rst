Quantization-SparsityAwareTrainingwithNNCF,usingPyTorchframework
=======================================================================

Thisnotebookisbasedon`ImageNettrainingin
PyTorch<https://github.com/pytorch/examples/blob/master/imagenet/main.py>`__.

ThegoalofthisnotebookistodemonstratehowtousetheNeural
NetworkCompressionFramework
`NNCF<https://github.com/openvinotoolkit/nncf>`__8-bitquantizationto
optimizeaPyTorchmodelforinferencewithOpenVINOToolkit.The
optimizationprocesscontainsthefollowingsteps:

-Transformingtheoriginaldense``FP32``modeltosparse``INT8``
-Usingfine-tuningtoimprovetheaccuracy.
-ExportingoptimizedandoriginalmodelstoOpenVINOIR
-Measuringandcomparingtheperformanceofmodels.

Formoreadvancedusage,refertothese
`examples<https://github.com/openvinotoolkit/nncf/tree/develop/examples>`__.

ThistutorialusestheResNet-50modelwiththeImageNetdataset.The
datasetmustbedownloadedseparately.ToseeResNetmodels,visit
`PyTorchhub<https://pytorch.org/hub/pytorch_vision_resnet/>`__.

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
-`ExportINT8SparseModeltoOpenVINO
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
frompathlibimportPath

importtorch

importtorch.nnasnn
importtorch.nn.parallel
importtorch.optim
importtorch.utils.data
importtorch.utils.data.distributed
importtorchvision.datasetsasdatasets
importtorchvision.transformsastransforms
importtorchvision.modelsasmodels

importopenvinoasov
fromtorch.jitimportTracerWarning

torch.manual_seed(0)
device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")
print(f"Using{device}device")

MODEL_DIR=Path("model")
OUTPUT_DIR=Path("output")
#DATA_DIR=Path("...")#Insertpathtofoldercontainingimagenetfolder
#DATASET_DIR=DATA_DIR/"imagenet"


..parsed-literal::

Usingcpudevice


..code::ipython3

#Fetch`notebook_utils`module
importzipfile
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

DATA_DIR=Path("data")


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

BASE_MODEL_NAME="resnet18"
image_size=64

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

#PathswherePyTorchandOpenVINOIRmodelswillbestored.
fp32_pth_path=Path(MODEL_DIR/(BASE_MODEL_NAME+"_fp32")).with_suffix(".pth")
fp32_ir_path=fp32_pth_path.with_suffix(".xml")
int8_sparse_ir_path=Path(MODEL_DIR/(BASE_MODEL_NAME+"_int8_sparse")).with_suffix(".xml")



..parsed-literal::

data/tiny-imagenet-200.zip:0%||0.00/237M[00:00<?,?B/s]


..parsed-literal::

Successfullydownloadedandprepareddatasetat:data/tiny-imagenet-200


TrainFunction
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

deftrain(train_loader,model,compression_ctrl,criterion,optimizer,epoch):
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
compression_ctrl.scheduler.step()

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
Itcanbeobtainedbytuningfromscratchwiththecodebelow.

..code::ipython3

num_classes=1000
init_lr=1e-4
batch_size=128
epochs=20

#model=models.resnet50(pretrained=True)
model=models.resnet18(pretrained=True)
model.fc=nn.Linear(in_features=512,out_features=200,bias=True)
model.to(device)


#Dataloadingcode.
train_dir=DATASET_DIR/"train"
val_dir=DATASET_DIR/"val"
normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

train_dataset=datasets.ImageFolder(
train_dir,
transforms.Compose(
[
transforms.Resize([image_size,image_size]),
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
transforms.Resize([256,256]),
transforms.CenterCrop([image_size,image_size]),
transforms.ToTensor(),
normalize,
]
),
)

train_loader=torch.utils.data.DataLoader(
train_dataset,
batch_size=batch_size,
shuffle=True,
num_workers=1,
pin_memory=True,
sampler=None,
)

val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=1,pin_memory=True)

#Definelossfunction(criterion)andoptimizer.
criterion=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=init_lr)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:208:UserWarning:Theparameter'pretrained'isdeprecatedsince0.13andmayberemovedinthefuture,pleaseuse'weights'instead.
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:223:UserWarning:Argumentsotherthanaweightenumor`None`for'weights'aredeprecatedsince0.13andmayberemovedinthefuture.Thecurrentbehaviorisequivalenttopassing`weights=ResNet18_Weights.IMAGENET1K_V1`.Youcanalsouse`weights=ResNet18_Weights.DEFAULT`togetthemostup-to-dateweights.
warnings.warn(msg)


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


CreateandInitializeQuantizationandSparsityTraining
--------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

NNCFenablescompression-awaretrainingbyintegratingintoregular
trainingpipelines.Theframeworkisdesignedsothatmodificationsto
youroriginaltrainingcodeareminor.

..code::ipython3

fromnncfimportNNCFConfig
fromnncf.torchimportcreate_compressed_model,register_default_init_args

#load
nncf_config=NNCFConfig.from_json("config.json")
nncf_config=register_default_init_args(nncf_config,train_loader)

#Creatingacompressedmodel
compression_ctrl,compressed_model=create_compressed_model(model,nncf_config)
compression_ctrl.scheduler.epoch_step()


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino
INFO:nncf:Ignoredaddingweightsparsifierforoperation:ResNet/NNCFConv2d[conv1]/conv2d_0
INFO:nncf:Collectingtensorstatistics|█|8/79
INFO:nncf:Collectingtensorstatistics|███|16/79
INFO:nncf:Collectingtensorstatistics|████|24/79
INFO:nncf:Collectingtensorstatistics|██████|32/79
INFO:nncf:Collectingtensorstatistics|████████|40/79
INFO:nncf:Collectingtensorstatistics|█████████|48/79
INFO:nncf:Collectingtensorstatistics|███████████|56/79
INFO:nncf:Collectingtensorstatistics|████████████|64/79
INFO:nncf:Collectingtensorstatistics|██████████████|72/79
INFO:nncf:Collectingtensorstatistics|████████████████|79/79
INFO:nncf:Compilingandloadingtorchextension:quantized_functions_cpu...
INFO:nncf:Finishedloadingtorchextension:quantized_functions_cpu


..parsed-literal::

2024-07-1301:55:18.828082:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:55:18.860964:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:55:19.462742:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


..parsed-literal::

INFO:nncf:BatchNormstatisticsadaptation|█|1/16
INFO:nncf:BatchNormstatisticsadaptation|██|2/16
INFO:nncf:BatchNormstatisticsadaptation|███|3/16
INFO:nncf:BatchNormstatisticsadaptation|████|4/16
INFO:nncf:BatchNormstatisticsadaptation|█████|5/16
INFO:nncf:BatchNormstatisticsadaptation|██████|6/16
INFO:nncf:BatchNormstatisticsadaptation|███████|7/16
INFO:nncf:BatchNormstatisticsadaptation|████████|8/16
INFO:nncf:BatchNormstatisticsadaptation|█████████|9/16
INFO:nncf:BatchNormstatisticsadaptation|██████████|10/16
INFO:nncf:BatchNormstatisticsadaptation|███████████|11/16
INFO:nncf:BatchNormstatisticsadaptation|████████████|12/16
INFO:nncf:BatchNormstatisticsadaptation|█████████████|13/16
INFO:nncf:BatchNormstatisticsadaptation|██████████████|14/16
INFO:nncf:BatchNormstatisticsadaptation|███████████████|15/16
INFO:nncf:BatchNormstatisticsadaptation|████████████████|16/16


ValidateCompressedModel

Evaluatethenewmodelonthevalidationsetafterinitializationof
quantizationandsparsity.

..code::ipython3

acc1=validate(val_loader,compressed_model,criterion)
print(f"AccuracyofinitializedsparseINT8model:{acc1:.3f}")


..parsed-literal::

Test:[0/79]	Time0.346(0.346)	Loss6.069(6.069)	Acc@10.00(0.00)	Acc@54.69(4.69)
Test:[10/79]	Time0.147(0.161)	Loss5.368(5.689)	Acc@10.78(0.07)	Acc@53.91(2.41)
Test:[20/79]	Time0.157(0.154)	Loss5.921(5.653)	Acc@10.00(0.56)	Acc@52.34(3.16)
Test:[30/79]	Time0.144(0.151)	Loss5.664(5.670)	Acc@10.00(0.50)	Acc@50.78(2.90)
Test:[40/79]	Time0.139(0.149)	Loss5.608(5.632)	Acc@11.56(0.59)	Acc@53.12(3.09)
Test:[50/79]	Time0.147(0.148)	Loss5.170(5.618)	Acc@10.00(0.72)	Acc@52.34(3.32)
Test:[60/79]	Time0.144(0.147)	Loss6.619(5.634)	Acc@10.00(0.67)	Acc@50.00(3.00)
Test:[70/79]	Time0.146(0.146)	Loss5.771(5.653)	Acc@10.00(0.57)	Acc@51.56(2.77)
*Acc@10.570Acc@52.770
AccuracyofinitializedsparseINT8model:0.570


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
optimizer=torch.optim.Adam(compressed_model.parameters(),lr=compression_lr)
nr_epochs=10
#TrainforoneepochwithNNCF.
print("Training")
forepochinrange(nr_epochs):
compression_ctrl.scheduler.epoch_step()
train(train_loader,compressed_model,compression_ctrl,criterion,optimizer,epoch=epoch)

#EvaluateonvalidationsetafterQuantization-AwareTraining(QATcase).
print("Validating")
acc1_int8_sparse=validate(val_loader,compressed_model,criterion)

print(f"AccuracyoftunedINT8sparsemodel:{acc1_int8_sparse:.3f}")
print(f"AccuracydropoftunedINT8sparsemodeloverpre-trainedFP32model:{acc1-acc1_int8_sparse:.3f}")


..parsed-literal::

Training
Epoch:[0][0/782]	Time0.560(0.560)	Loss5.673(5.673)	Acc@10.78(0.78)	Acc@53.12(3.12)
Epoch:[0][50/782]	Time0.338(0.345)	Loss5.643(5.644)	Acc@10.00(0.78)	Acc@52.34(3.12)
Epoch:[0][100/782]	Time0.336(0.341)	Loss5.565(5.604)	Acc@10.78(0.80)	Acc@52.34(3.23)
Epoch:[0][150/782]	Time0.335(0.340)	Loss5.540(5.559)	Acc@10.78(0.90)	Acc@53.91(3.53)
Epoch:[0][200/782]	Time0.338(0.339)	Loss5.273(5.515)	Acc@12.34(1.07)	Acc@57.81(3.98)
Epoch:[0][250/782]	Time0.339(0.339)	Loss5.358(5.473)	Acc@11.56(1.24)	Acc@56.25(4.52)
Epoch:[0][300/782]	Time0.335(0.338)	Loss5.226(5.431)	Acc@11.56(1.45)	Acc@57.03(5.10)
Epoch:[0][350/782]	Time0.349(0.338)	Loss5.104(5.388)	Acc@11.56(1.67)	Acc@510.16(5.81)
Epoch:[0][400/782]	Time0.329(0.338)	Loss5.052(5.351)	Acc@10.78(1.84)	Acc@512.50(6.42)
Epoch:[0][450/782]	Time0.346(0.337)	Loss5.049(5.312)	Acc@13.91(2.11)	Acc@510.94(7.15)
Epoch:[0][500/782]	Time0.341(0.337)	Loss4.855(5.275)	Acc@15.47(2.38)	Acc@513.28(7.91)
Epoch:[0][550/782]	Time0.333(0.337)	Loss4.707(5.237)	Acc@110.16(2.74)	Acc@524.22(8.75)
Epoch:[0][600/782]	Time0.337(0.337)	Loss4.622(5.197)	Acc@17.81(3.14)	Acc@525.00(9.72)
Epoch:[0][650/782]	Time0.336(0.337)	Loss4.615(5.160)	Acc@110.16(3.55)	Acc@522.66(10.64)
Epoch:[0][700/782]	Time0.333(0.337)	Loss4.655(5.122)	Acc@17.03(3.99)	Acc@522.66(11.62)
Epoch:[0][750/782]	Time0.332(0.337)	Loss4.461(5.084)	Acc@115.62(4.51)	Acc@534.38(12.66)
Epoch:[1][0/782]	Time0.831(0.831)	Loss4.331(4.331)	Acc@115.62(15.62)	Acc@535.16(35.16)
Epoch:[1][50/782]	Time0.339(0.358)	Loss4.327(4.228)	Acc@114.06(16.68)	Acc@532.03(37.44)
Epoch:[1][100/782]	Time0.330(0.348)	Loss4.208(4.187)	Acc@117.97(18.04)	Acc@535.94(38.38)
Epoch:[1][150/782]	Time0.336(0.345)	Loss4.060(4.166)	Acc@117.97(18.56)	Acc@542.97(38.90)
Epoch:[1][200/782]	Time0.333(0.343)	Loss4.100(4.142)	Acc@117.97(18.94)	Acc@541.41(39.69)
Epoch:[1][250/782]	Time0.344(0.342)	Loss4.081(4.119)	Acc@121.88(19.23)	Acc@543.75(40.24)
Epoch:[1][300/782]	Time0.334(0.341)	Loss4.199(4.099)	Acc@115.62(19.49)	Acc@537.50(40.77)
Epoch:[1][350/782]	Time0.337(0.341)	Loss3.830(4.077)	Acc@125.78(19.82)	Acc@545.31(41.33)
Epoch:[1][400/782]	Time0.327(0.340)	Loss4.089(4.054)	Acc@121.09(20.27)	Acc@539.06(41.95)
Epoch:[1][450/782]	Time0.339(0.340)	Loss3.782(4.034)	Acc@126.56(20.62)	Acc@544.53(42.39)
Epoch:[1][500/782]	Time0.337(0.340)	Loss3.816(4.012)	Acc@126.56(21.00)	Acc@550.78(43.00)
Epoch:[1][550/782]	Time0.339(0.340)	Loss3.620(3.989)	Acc@126.56(21.37)	Acc@552.34(43.58)
Epoch:[1][600/782]	Time0.340(0.340)	Loss3.694(3.971)	Acc@128.91(21.63)	Acc@547.66(44.06)
Epoch:[1][650/782]	Time0.340(0.340)	Loss3.738(3.952)	Acc@122.66(21.86)	Acc@545.31(44.52)
Epoch:[1][700/782]	Time0.355(0.341)	Loss3.735(3.936)	Acc@125.00(22.09)	Acc@544.53(44.90)
Epoch:[1][750/782]	Time0.344(0.342)	Loss3.630(3.918)	Acc@129.69(22.32)	Acc@553.12(45.32)
Epoch:[2][0/782]	Time0.673(0.673)	Loss3.419(3.419)	Acc@132.03(32.03)	Acc@557.81(57.81)
Epoch:[2][50/782]	Time0.343(0.357)	Loss3.397(3.466)	Acc@132.03(29.34)	Acc@556.25(54.96)
Epoch:[2][100/782]	Time0.346(0.350)	Loss3.293(3.432)	Acc@133.59(30.02)	Acc@559.38(56.53)
Epoch:[2][150/782]	Time0.338(0.349)	Loss3.358(3.422)	Acc@133.59(30.30)	Acc@559.38(56.64)
Epoch:[2][200/782]	Time0.340(0.347)	Loss3.215(3.410)	Acc@134.38(30.50)	Acc@563.28(56.97)
Epoch:[2][250/782]	Time0.335(0.347)	Loss3.369(3.392)	Acc@132.81(30.82)	Acc@557.81(57.15)
Epoch:[2][300/782]	Time0.337(0.346)	Loss3.487(3.379)	Acc@125.78(30.96)	Acc@551.56(57.35)
Epoch:[2][350/782]	Time0.342(0.345)	Loss3.336(3.370)	Acc@134.38(31.04)	Acc@560.94(57.51)
Epoch:[2][400/782]	Time0.340(0.345)	Loss3.434(3.359)	Acc@125.78(31.16)	Acc@559.38(57.66)
Epoch:[2][450/782]	Time0.341(0.345)	Loss3.440(3.348)	Acc@128.12(31.42)	Acc@557.81(57.85)
Epoch:[2][500/782]	Time0.339(0.345)	Loss3.129(3.336)	Acc@135.16(31.59)	Acc@566.41(58.09)
Epoch:[2][550/782]	Time0.347(0.345)	Loss3.388(3.322)	Acc@126.56(31.77)	Acc@552.34(58.40)
Epoch:[2][600/782]	Time0.344(0.345)	Loss3.078(3.311)	Acc@136.72(31.89)	Acc@563.28(58.57)
Epoch:[2][650/782]	Time0.345(0.345)	Loss3.172(3.300)	Acc@136.72(32.08)	Acc@564.84(58.76)
Epoch:[2][700/782]	Time0.346(0.345)	Loss3.152(3.287)	Acc@132.03(32.23)	Acc@558.59(58.98)
Epoch:[2][750/782]	Time0.345(0.345)	Loss3.228(3.275)	Acc@136.72(32.45)	Acc@556.25(59.21)
Epoch:[3][0/782]	Time0.690(0.690)	Loss3.060(3.060)	Acc@132.03(32.03)	Acc@566.41(66.41)
Epoch:[3][50/782]	Time0.347(0.349)	Loss2.926(2.958)	Acc@144.53(37.94)	Acc@562.50(65.10)
Epoch:[3][100/782]	Time0.346(0.346)	Loss3.022(2.938)	Acc@134.38(38.18)	Acc@561.72(65.66)
Epoch:[3][150/782]	Time0.347(0.348)	Loss2.760(2.934)	Acc@140.62(38.10)	Acc@569.53(65.46)
Epoch:[3][200/782]	Time0.349(0.347)	Loss3.039(2.928)	Acc@134.38(38.21)	Acc@560.94(65.38)
Epoch:[3][250/782]	Time0.345(0.347)	Loss2.829(2.924)	Acc@133.59(38.16)	Acc@567.19(65.41)
Epoch:[3][300/782]	Time0.352(0.346)	Loss2.895(2.919)	Acc@143.75(38.16)	Acc@572.66(65.39)
Epoch:[3][350/782]	Time0.343(0.346)	Loss2.767(2.914)	Acc@141.41(38.23)	Acc@568.75(65.42)
Epoch:[3][400/782]	Time0.342(0.346)	Loss3.116(2.908)	Acc@130.47(38.20)	Acc@560.16(65.48)
Epoch:[3][450/782]	Time0.353(0.346)	Loss2.914(2.903)	Acc@135.94(38.30)	Acc@562.50(65.54)
Epoch:[3][500/782]	Time0.343(0.346)	Loss2.719(2.895)	Acc@144.53(38.36)	Acc@567.97(65.71)
Epoch:[3][550/782]	Time0.345(0.345)	Loss3.138(2.889)	Acc@132.81(38.40)	Acc@560.16(65.79)
Epoch:[3][600/782]	Time0.341(0.345)	Loss3.042(2.884)	Acc@132.03(38.43)	Acc@558.59(65.82)
Epoch:[3][650/782]	Time0.332(0.345)	Loss2.931(2.877)	Acc@142.19(38.54)	Acc@567.19(65.96)
Epoch:[3][700/782]	Time0.340(0.345)	Loss2.968(2.870)	Acc@132.81(38.57)	Acc@561.72(66.06)
Epoch:[3][750/782]	Time0.343(0.345)	Loss2.799(2.864)	Acc@137.50(38.71)	Acc@565.62(66.12)
Epoch:[4][0/782]	Time0.675(0.675)	Loss2.625(2.625)	Acc@146.09(46.09)	Acc@568.75(68.75)
Epoch:[4][50/782]	Time0.345(0.351)	Loss2.682(2.727)	Acc@146.09(40.18)	Acc@567.97(67.98)
Epoch:[4][100/782]	Time0.354(0.348)	Loss2.824(2.699)	Acc@133.59(41.11)	Acc@564.84(68.60)
Epoch:[4][150/782]	Time0.347(0.348)	Loss2.703(2.690)	Acc@146.09(41.44)	Acc@564.84(68.91)
Epoch:[4][200/782]	Time0.350(0.347)	Loss2.523(2.683)	Acc@146.88(41.64)	Acc@574.22(69.03)
Epoch:[4][250/782]	Time0.353(0.347)	Loss2.381(2.677)	Acc@149.22(41.80)	Acc@574.22(69.10)
Epoch:[4][300/782]	Time0.342(0.349)	Loss2.633(2.674)	Acc@142.19(41.82)	Acc@565.62(68.98)
Epoch:[4][350/782]	Time0.341(0.348)	Loss2.621(2.671)	Acc@146.09(41.86)	Acc@571.88(69.01)
Epoch:[4][400/782]	Time0.353(0.347)	Loss2.472(2.662)	Acc@142.97(42.02)	Acc@575.00(69.15)
Epoch:[4][450/782]	Time0.346(0.347)	Loss2.529(2.659)	Acc@142.19(42.03)	Acc@575.78(69.18)
Epoch:[4][500/782]	Time0.338(0.347)	Loss2.793(2.654)	Acc@137.50(42.12)	Acc@564.84(69.27)
Epoch:[4][550/782]	Time0.350(0.347)	Loss2.474(2.646)	Acc@145.31(42.31)	Acc@567.97(69.32)
Epoch:[4][600/782]	Time0.352(0.347)	Loss2.383(2.642)	Acc@151.56(42.36)	Acc@573.44(69.34)
Epoch:[4][650/782]	Time0.336(0.347)	Loss2.595(2.638)	Acc@143.75(42.41)	Acc@571.88(69.35)
Epoch:[4][700/782]	Time0.343(0.347)	Loss2.541(2.634)	Acc@139.84(42.44)	Acc@574.22(69.37)
Epoch:[4][750/782]	Time0.342(0.346)	Loss2.408(2.628)	Acc@145.31(42.52)	Acc@575.00(69.51)
Epoch:[5][0/782]	Time0.688(0.688)	Loss2.310(2.310)	Acc@148.44(48.44)	Acc@575.00(75.00)
Epoch:[5][50/782]	Time0.338(0.351)	Loss2.585(2.521)	Acc@142.97(43.66)	Acc@568.75(71.32)
Epoch:[5][100/782]	Time0.347(0.347)	Loss2.263(2.491)	Acc@148.44(44.46)	Acc@574.22(71.88)
Epoch:[5][150/782]	Time0.341(0.345)	Loss2.296(2.480)	Acc@152.34(44.62)	Acc@575.00(71.90)
Epoch:[5][200/782]	Time0.347(0.345)	Loss2.430(2.479)	Acc@148.44(44.75)	Acc@570.31(71.79)
Epoch:[5][250/782]	Time0.348(0.345)	Loss2.566(2.482)	Acc@140.62(44.74)	Acc@569.53(71.70)
Epoch:[5][300/782]	Time0.347(0.346)	Loss2.414(2.476)	Acc@140.62(44.86)	Acc@578.12(71.78)
Epoch:[5][350/782]	Time0.338(0.346)	Loss2.301(2.477)	Acc@150.78(44.74)	Acc@575.78(71.62)
Epoch:[5][400/782]	Time0.348(0.346)	Loss2.414(2.472)	Acc@144.53(44.87)	Acc@572.66(71.71)
Epoch:[5][450/782]	Time0.343(0.346)	Loss2.352(2.466)	Acc@150.78(44.94)	Acc@572.66(71.85)
Epoch:[5][500/782]	Time0.342(0.345)	Loss2.423(2.464)	Acc@147.66(44.97)	Acc@574.22(71.84)
Epoch:[5][550/782]	Time0.342(0.345)	Loss2.407(2.459)	Acc@140.62(45.03)	Acc@571.88(71.88)
Epoch:[5][600/782]	Time0.339(0.345)	Loss2.326(2.457)	Acc@148.44(45.05)	Acc@577.34(71.91)
Epoch:[5][650/782]	Time0.341(0.345)	Loss2.283(2.452)	Acc@147.66(45.13)	Acc@571.88(72.01)
Epoch:[5][700/782]	Time0.334(0.344)	Loss2.217(2.446)	Acc@146.88(45.21)	Acc@572.66(72.09)
Epoch:[5][750/782]	Time0.339(0.344)	Loss2.474(2.442)	Acc@150.78(45.29)	Acc@565.62(72.12)
Epoch:[6][0/782]	Time0.679(0.679)	Loss2.568(2.568)	Acc@144.53(44.53)	Acc@564.06(64.06)
Epoch:[6][50/782]	Time0.333(0.348)	Loss2.411(2.321)	Acc@145.31(47.50)	Acc@568.75(74.17)
Epoch:[6][100/782]	Time0.335(0.345)	Loss2.401(2.333)	Acc@148.44(47.05)	Acc@572.66(73.89)
Epoch:[6][150/782]	Time0.344(0.344)	Loss2.220(2.331)	Acc@146.88(47.11)	Acc@575.78(73.85)
Epoch:[6][200/782]	Time0.351(0.344)	Loss2.330(2.329)	Acc@149.22(47.21)	Acc@573.44(73.77)
Epoch:[6][250/782]	Time0.348(0.343)	Loss2.581(2.330)	Acc@143.75(47.22)	Acc@567.97(73.84)
Epoch:[6][300/782]	Time0.340(0.343)	Loss2.457(2.321)	Acc@142.97(47.57)	Acc@573.44(74.00)
Epoch:[6][350/782]	Time0.343(0.343)	Loss2.332(2.321)	Acc@150.78(47.49)	Acc@573.44(73.98)
Epoch:[6][400/782]	Time0.348(0.343)	Loss2.057(2.317)	Acc@153.91(47.56)	Acc@580.47(74.01)
Epoch:[6][450/782]	Time0.345(0.344)	Loss2.379(2.316)	Acc@145.31(47.41)	Acc@571.09(74.02)
Epoch:[6][500/782]	Time0.374(0.344)	Loss2.337(2.313)	Acc@148.44(47.44)	Acc@571.09(74.10)
Epoch:[6][550/782]	Time0.342(0.344)	Loss2.207(2.309)	Acc@146.88(47.54)	Acc@574.22(74.18)
Epoch:[6][600/782]	Time0.342(0.344)	Loss2.191(2.305)	Acc@157.03(47.63)	Acc@577.34(74.22)
Epoch:[6][650/782]	Time0.343(0.344)	Loss2.120(2.303)	Acc@153.12(47.62)	Acc@577.34(74.23)
Epoch:[6][700/782]	Time0.337(0.344)	Loss2.312(2.298)	Acc@139.84(47.71)	Acc@571.88(74.30)
Epoch:[6][750/782]	Time0.336(0.344)	Loss2.080(2.295)	Acc@153.12(47.77)	Acc@579.69(74.34)
Epoch:[7][0/782]	Time0.692(0.692)	Loss2.192(2.192)	Acc@144.53(44.53)	Acc@578.12(78.12)
Epoch:[7][50/782]	Time0.344(0.349)	Loss2.139(2.214)	Acc@150.78(48.56)	Acc@576.56(75.32)
Epoch:[7][100/782]	Time0.345(0.346)	Loss2.266(2.213)	Acc@157.03(49.16)	Acc@571.88(75.45)
Epoch:[7][150/782]	Time0.341(0.345)	Loss1.987(2.209)	Acc@154.69(49.10)	Acc@582.03(75.53)
Epoch:[7][200/782]	Time0.345(0.344)	Loss2.232(2.203)	Acc@143.75(49.37)	Acc@575.00(75.62)
Epoch:[7][250/782]	Time0.346(0.345)	Loss2.216(2.203)	Acc@148.44(49.27)	Acc@578.91(75.66)
Epoch:[7][300/782]	Time0.345(0.345)	Loss2.393(2.202)	Acc@149.22(49.30)	Acc@571.09(75.70)
Epoch:[7][350/782]	Time0.345(0.344)	Loss2.084(2.196)	Acc@144.53(49.47)	Acc@580.47(75.84)
Epoch:[7][400/782]	Time0.345(0.345)	Loss1.682(2.194)	Acc@165.62(49.55)	Acc@583.59(75.82)
Epoch:[7][450/782]	Time0.343(0.345)	Loss2.193(2.194)	Acc@147.66(49.62)	Acc@575.78(75.82)
Epoch:[7][500/782]	Time0.343(0.345)	Loss2.166(2.192)	Acc@145.31(49.59)	Acc@578.12(75.81)
Epoch:[7][550/782]	Time0.335(0.346)	Loss2.126(2.187)	Acc@147.66(49.70)	Acc@578.91(75.84)
Epoch:[7][600/782]	Time0.337(0.346)	Loss2.222(2.184)	Acc@149.22(49.73)	Acc@573.44(75.87)
Epoch:[7][650/782]	Time0.341(0.345)	Loss2.075(2.181)	Acc@150.00(49.79)	Acc@578.12(75.89)
Epoch:[7][700/782]	Time0.343(0.345)	Loss2.181(2.179)	Acc@147.66(49.81)	Acc@575.78(75.89)
Epoch:[7][750/782]	Time0.346(0.345)	Loss2.071(2.177)	Acc@153.12(49.82)	Acc@575.78(75.89)
Epoch:[8][0/782]	Time0.669(0.669)	Loss1.829(1.829)	Acc@158.59(58.59)	Acc@582.03(82.03)
Epoch:[8][50/782]	Time0.345(0.354)	Loss2.171(2.096)	Acc@150.78(51.04)	Acc@578.91(77.51)
Epoch:[8][100/782]	Time0.337(0.349)	Loss2.207(2.089)	Acc@152.34(51.26)	Acc@574.22(77.56)
Epoch:[8][150/782]	Time0.343(0.346)	Loss2.289(2.100)	Acc@149.22(51.13)	Acc@573.44(77.32)
Epoch:[8][200/782]	Time0.343(0.346)	Loss2.175(2.101)	Acc@146.88(51.00)	Acc@577.34(77.29)
Epoch:[8][250/782]	Time0.341(0.345)	Loss2.239(2.092)	Acc@147.66(51.30)	Acc@571.88(77.35)
Epoch:[8][300/782]	Time0.366(0.346)	Loss2.070(2.087)	Acc@149.22(51.40)	Acc@575.78(77.41)
Epoch:[8][350/782]	Time0.343(0.346)	Loss1.868(2.083)	Acc@152.34(51.38)	Acc@582.81(77.39)
Epoch:[8][400/782]	Time0.344(0.345)	Loss2.345(2.084)	Acc@140.62(51.47)	Acc@571.88(77.34)
Epoch:[8][450/782]	Time0.343(0.345)	Loss1.731(2.085)	Acc@163.28(51.43)	Acc@582.81(77.32)
Epoch:[8][500/782]	Time0.343(0.345)	Loss2.142(2.082)	Acc@146.09(51.40)	Acc@577.34(77.35)
Epoch:[8][550/782]	Time0.353(0.345)	Loss2.173(2.080)	Acc@153.91(51.45)	Acc@573.44(77.40)
Epoch:[8][600/782]	Time0.352(0.345)	Loss2.184(2.077)	Acc@154.69(51.55)	Acc@573.44(77.43)
Epoch:[8][650/782]	Time0.345(0.345)	Loss2.118(2.075)	Acc@149.22(51.60)	Acc@576.56(77.43)
Epoch:[8][700/782]	Time0.343(0.345)	Loss2.254(2.074)	Acc@151.56(51.61)	Acc@572.66(77.37)
Epoch:[8][750/782]	Time0.346(0.346)	Loss2.056(2.071)	Acc@153.91(51.67)	Acc@575.78(77.41)
Epoch:[9][0/782]	Time0.700(0.700)	Loss1.824(1.824)	Acc@159.38(59.38)	Acc@585.16(85.16)
Epoch:[9][50/782]	Time0.345(0.355)	Loss2.063(1.996)	Acc@150.78(53.09)	Acc@580.47(78.65)
Epoch:[9][100/782]	Time0.340(0.350)	Loss1.874(1.999)	Acc@158.59(53.12)	Acc@582.03(78.38)
Epoch:[9][150/782]	Time0.345(0.347)	Loss2.026(1.994)	Acc@150.78(53.17)	Acc@578.91(78.80)
Epoch:[9][200/782]	Time0.344(0.346)	Loss1.877(1.994)	Acc@159.38(53.10)	Acc@582.81(78.68)
Epoch:[9][250/782]	Time0.347(0.346)	Loss2.166(1.996)	Acc@146.09(53.00)	Acc@573.44(78.60)
Epoch:[9][300/782]	Time0.338(0.346)	Loss2.125(1.997)	Acc@151.56(53.01)	Acc@576.56(78.49)
Epoch:[9][350/782]	Time0.342(0.346)	Loss2.210(1.995)	Acc@146.88(52.89)	Acc@575.00(78.60)
Epoch:[9][400/782]	Time0.347(0.345)	Loss1.897(1.994)	Acc@157.81(52.86)	Acc@579.69(78.56)
Epoch:[9][450/782]	Time0.337(0.346)	Loss2.045(1.989)	Acc@150.78(53.00)	Acc@576.56(78.62)
Epoch:[9][500/782]	Time0.344(0.345)	Loss2.300(1.990)	Acc@146.88(52.97)	Acc@572.66(78.62)
Epoch:[9][550/782]	Time0.342(0.345)	Loss1.604(1.990)	Acc@164.06(53.02)	Acc@582.81(78.61)
Epoch:[9][600/782]	Time0.345(0.345)	Loss1.763(1.987)	Acc@154.69(53.07)	Acc@585.16(78.65)
Epoch:[9][650/782]	Time0.345(0.345)	Loss1.664(1.984)	Acc@163.28(53.11)	Acc@582.81(78.71)
Epoch:[9][700/782]	Time0.344(0.345)	Loss2.284(1.982)	Acc@142.97(53.12)	Acc@578.12(78.76)
Epoch:[9][750/782]	Time0.343(0.345)	Loss1.698(1.983)	Acc@159.38(53.11)	Acc@582.03(78.72)
Validating
Test:[0/79]	Time0.399(0.399)	Loss4.175(4.175)	Acc@17.81(7.81)	Acc@529.69(29.69)
Test:[10/79]	Time0.137(0.160)	Loss5.955(4.803)	Acc@13.12(7.81)	Acc@57.03(21.02)
Test:[20/79]	Time0.136(0.148)	Loss6.302(5.109)	Acc@10.00(5.21)	Acc@53.12(17.22)
Test:[30/79]	Time0.137(0.145)	Loss5.520(5.327)	Acc@11.56(4.26)	Acc@516.41(14.36)
Test:[40/79]	Time0.149(0.142)	Loss5.560(5.399)	Acc@16.25(4.12)	Acc@57.81(13.34)
Test:[50/79]	Time0.139(0.141)	Loss4.887(5.498)	Acc@17.81(3.92)	Acc@521.88(12.68)
Test:[60/79]	Time0.116(0.140)	Loss5.905(5.512)	Acc@10.00(3.98)	Acc@57.03(12.58)
Test:[70/79]	Time0.137(0.140)	Loss4.785(5.526)	Acc@12.34(3.75)	Acc@511.72(11.99)
*Acc@15.320Acc@515.300
AccuracyoftunedINT8sparsemodel:5.320
AccuracydropoftunedINT8sparsemodeloverpre-trainedFP32model:-4.750


ExportINT8SparseModeltoOpenVINOIR
---------------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

warnings.filterwarnings("ignore",category=TracerWarning)
warnings.filterwarnings("ignore",category=UserWarning)
#ExportINT8modeltoOpenVINO™IR
ov_model=ov.convert_model(compressed_model,example_input=dummy_input,input=[1,3,image_size,image_size])
ov.save_model(ov_model,int8_sparse_ir_path)
print(f"INT8sparsemodelexportedto{int8_sparse_ir_path}.")


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.
['x']
INT8sparsemodelexportedtomodel/resnet18_int8_sparse.xml.


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

print("BenchmarkINT8sparsemodel(IR)")
benchmark_output=!benchmark_app-m$int8_ir_path-d$device.value-apiasync-t15
parse_benchmark_output(benchmark_output)


..parsed-literal::

BenchmarkFP32model(IR)
[INFO]Throughput:2943.43FPS
BenchmarkINT8sparsemodel(IR)



ShowDeviceInformationforreference.

..code::ipython3

core.get_property(device.value,"FULL_DEVICE_NAME")




..parsed-literal::

'Intel(R)Core(TM)i9-10920XCPU@3.50GHz'


