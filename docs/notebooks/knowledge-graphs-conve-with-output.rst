OpenVINOoptimizationsforKnowledgegraphs
===========================================

Thegoalofthisnotebookistoshowcaseperformanceoptimizationsfor
theConvEknowledgegraphembeddingsmodelusingtheIntel®Distribution
ofOpenVINO™Toolkit.Theoptimizationsprocesscontainsthefollowing
steps:

1.ExportthetrainedmodeltoaformatsuitableforOpenVINO
optimizationsandinference
2.Reporttheinferenceperformancespeedupobtainedwiththeoptimized
OpenVINOmodel

TheConvEmodelisanimplementationofthepaper-“Convolutional2D
KnowledgeGraphEmbeddings”(https://arxiv.org/abs/1707.01476).The
sampledatasetcanbedownloadedfrom:
https://github.com/TimDettmers/ConvE/tree/master/countries/countries_S1

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Windowsspecificsettings<#windows-specific-settings>`__
-`Importthepackagesneededforsuccessful
execution<#import-the-packages-needed-for-successful-execution>`__

-`Settings:Includingpathtotheserializedmodelfilesandinput
data
files<#settings-including-path-to-the-serialized-model-files-and-input-data-files>`__
-`DownloadModelCheckpoint<#download-model-checkpoint>`__
-`DefiningtheConvEmodel
class<#defining-the-conve-model-class>`__
-`Definingthedataloader<#defining-the-dataloader>`__
-`EvaluatethetrainedConvE
model<#evaluate-the-trained-conve-model>`__
-`PredictionontheKnowledge
graph.<#prediction-on-the-knowledge-graph->`__
-`ConvertthetrainedPyTorchmodeltoIRformatforOpenVINO
inference<#convert-the-trained-pytorch-model-to-ir-format-for-openvino-inference>`__
-`Evaluatethemodelperformancewith
OpenVINO<#evaluate-the-model-performance-with-openvino>`__

-`Selectinferencedevice<#select-inference-device>`__

-`DeterminetheplatformspecificspeedupobtainedthroughOpenVINO
graph
optimizations<#determine-the-platform-specific-speedup-obtained-through-openvino-graph-optimizations>`__
-`BenchmarktheconvertedOpenVINOmodelusingbenchmark
app<#benchmark-the-converted-openvino-model-using-benchmark-app>`__
-`Conclusions<#conclusions>`__
-`References<#references>`__

..code::ipython3

%pipinstall-q"openvino>=2023.1.0"torchscikit-learntqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


Windowsspecificsettings
-------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#OnWindows,addthedirectorythatcontainscl.exetothePATH
#toenablePyTorchtofindtherequiredC++tools.
#ThiscodeassumesthatVisualStudio2019isinstalledinthedefaultdirectory.
#IfyouhaveadifferentC++compiler,pleaseaddthecorrectpath
#toos.environ["PATH"]directly.
#NotethattheC++Redistributableisnotenoughtorunthisnotebook.

#Addingthepathtoos.environ["LIB"]isnotalwaysrequired
#-itdependsonthesystem'sconfiguration

importsys

ifsys.platform=="win32":
importdistutils.command.build_ext
importos
frompathlibimportPath

VS_INSTALL_DIR=r"C:/ProgramFiles(x86)/MicrosoftVisualStudio"
cl_paths=sorted(list(Path(VS_INSTALL_DIR).glob("**/Hostx86/x64/cl.exe")))
iflen(cl_paths)==0:
raiseValueError(
"CannotfindVisualStudio.ThisnotebookrequiresaC++compiler.Ifyouinstalled"
"aC++compiler,pleaseaddthedirectorythatcontains"
"cl.exeto`os.environ['PATH']`."
)
else:
#IfmultipleversionsofMSVCareinstalled,getthemostrecentversion
cl_path=cl_paths[-1]
vs_dir=str(cl_path.parent)
os.environ["PATH"]+=f"{os.pathsep}{vs_dir}"
#Codeforfindingthelibrarydirsfrom
#https://stackoverflow.com/questions/47423246/get-pythons-lib-path
d=distutils.core.Distribution()
b=distutils.command.build_ext.build_ext(d)
b.finalize_options()
os.environ["LIB"]=os.pathsep.join(b.library_dirs)
print(f"Added{vs_dir}toPATH")

Importthepackagesneededforsuccessfulexecution
---------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importjson
frompathlibimportPath
importsys
importtime

importnumpyasnp
importtorch
fromsklearn.metricsimportaccuracy_score
fromtorch.nnimportfunctionalasF,Parameter
fromtorch.nn.initimportxavier_normal_

importopenvinoasov

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

Settings:Includingpathtotheserializedmodelfilesandinputdatafiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Pathtothepretrainedmodelcheckpoint
modelpath=Path("models/conve.pt")

#Entityandrelationembeddingdimensions
EMB_DIM=300

#TopKvalstoconsiderfromthepredictions
TOP_K=2

#RequiredforOpenVINOconversion
output_dir=Path("models")
base_model_name="conve"

output_dir.mkdir(exist_ok=True)

#PathswherePyTorchandOpenVINOIRmodelswillbestored
ir_path=Path(output_dir/base_model_name).with_suffix(".xml")

..code::ipython3

data_folder="data"

#DownloadthefilecontainingtheentitiesandentityIDs
entdatapath=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/text/countries_S1/kg_training_entids.txt",
directory=data_folder,
)

#DownloadthefilecontainingtherelationsandrelationIDs
reldatapath=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/text/countries_S1/kg_training_relids.txt",
directory=data_folder,
)

#Downloadthetestdatafile
testdatapath=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/json/countries_S1/e1rel_to_e2_ranking_test.json",
directory=data_folder,
)



..parsed-literal::

data/kg_training_entids.txt:0%||0.00/3.79k[00:00<?,?B/s]



..parsed-literal::

data/kg_training_relids.txt:0%||0.00/62.0[00:00<?,?B/s]



..parsed-literal::

data/e1rel_to_e2_ranking_test.json:0%||0.00/19.1k[00:00<?,?B/s]


DownloadModelCheckpoint
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/knowledge-graph-embeddings/conve.pt"

download_file(model_url,filename=modelpath.name,directory=modelpath.parent)



..parsed-literal::

models/conve.pt:0%||0.00/18.8M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/knowledge-graphs-conve/models/conve.pt')



DefiningtheConvEmodelclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Modelimplementationreference:https://github.com/TimDettmers/ConvE
classConvE(torch.nn.Module):
def__init__(self,num_entities,num_relations,emb_dim):
super(ConvE,self).__init__()
#Embeddingtablesforentityandrelationswithnum_uniq_entiny-dim,emb_diminx-dim
self.emb_e=torch.nn.Embedding(num_entities,emb_dim,padding_idx=0)
self.ent_weights_matrix=torch.ones([num_entities,emb_dim],dtype=torch.float64)
self.emb_rel=torch.nn.Embedding(num_relations,emb_dim,padding_idx=0)
self.ne=num_entities
self.nr=num_relations
self.inp_drop=torch.nn.Dropout(0.2)
self.hidden_drop=torch.nn.Dropout(0.3)
self.feature_map_drop=torch.nn.Dropout2d(0.2)
self.loss=torch.nn.BCELoss()
self.conv1=torch.nn.Conv2d(1,32,(3,3),1,0,bias=True)
self.bn0=torch.nn.BatchNorm2d(1)
self.bn1=torch.nn.BatchNorm2d(32)
self.ln0=torch.nn.LayerNorm(emb_dim)
self.register_parameter("b",Parameter(torch.zeros(num_entities)))
self.fc=torch.nn.Linear(16128,emb_dim)

definit(self):
"""Initializesthemodel"""
#Xavierinitialization
xavier_normal_(self.emb_e.weight.data)
xavier_normal_(self.emb_rel.weight.data)

defforward(self,e1,rel):
"""Forwardpassonthemodel.
:parame1:sourceentity
:paramrel:relationbetweenthesourceandtargetentities
Returnsthemodelpredictionsforthetargetentities
"""
e1_embedded=self.emb_e(e1).view(-1,1,10,30)
rel_embedded=self.emb_rel(rel).view(-1,1,10,30)
stacked_inputs=torch.cat([e1_embedded,rel_embedded],2)
stacked_inputs=self.bn0(stacked_inputs)
x=self.inp_drop(stacked_inputs)
x=self.conv1(x)
x=self.bn1(x)
x=F.relu(x)
x=self.feature_map_drop(x)
x=x.view(1,-1)
x=self.fc(x)
x=self.hidden_drop(x)
x=self.ln0(x)
x=F.relu(x)
x=torch.mm(x,self.emb_e.weight.transpose(1,0))
x=self.hidden_drop(x)
x+=self.b.expand_as(x)
pred=torch.nn.functional.softmax(x,dim=1)
returnpred

Definingthedataloader
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

classDataLoader:
def__init__(self):
super(DataLoader,self).__init__()

self.ent_path=entdatapath
self.rel_path=reldatapath
self.test_file=testdatapath
self.entity_ids,self.ids2entities=self.load_data(data_path=self.ent_path)
self.rel_ids,self.ids2rel=self.load_data(data_path=self.rel_path)
self.test_triples_list=self.convert_triples(data_path=self.test_file)

defload_data(self,data_path):
"""Createsadictionaryofdataitemswithcorrespondingids"""
item_dict,ids_dict={},{}
fp=open(data_path,"r")
lines=fp.readlines()
forlineinlines:
name,id=line.strip().split("\t")
item_dict[name]=int(id)
ids_dict[int(id)]=name
fp.close()
returnitem_dict,ids_dict

defconvert_triples(self,data_path):
"""Createsatripleofsourceentity,relationandtargetentities"""
triples_list=[]
dp=open(data_path,"r")
lines=dp.readlines()
forlineinlines:
item_dict=json.loads(line.strip())
h=item_dict["e1"]
r=item_dict["rel"]
t=item_dict["e2_multi1"].split("\t")
hrt_list=[]
hrt_list.append(self.entity_ids[h])
hrt_list.append(self.rel_ids[r])
t_ents=[]
fort_idxint:
t_ents.append(self.entity_ids[t_idx])
hrt_list.append(t_ents)
triples_list.append(hrt_list)
dp.close()
returntriples_list

EvaluatethetrainedConvEmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

First,wewillevaluatethemodelperformanceusingPyTorch.Thegoalis
tomakesuretherearenoaccuracydifferencesbetweentheoriginal
modelinferenceandthemodelconvertedtoOpenVINOintermediate
representationinferenceresults.Here,weuseasimpleaccuracymetric
toevaluatethemodelperformanceonatestdataset.However,itis
typicaltousemetricssuchasMeanReciprocalRank,Hits@10etc.

..code::ipython3

data=DataLoader()
num_entities=len(data.entity_ids)
num_relations=len(data.rel_ids)

model=ConvE(num_entities=num_entities,num_relations=num_relations,emb_dim=EMB_DIM)
model.load_state_dict(torch.load(modelpath))
model.eval()

pt_inf_times=[]

triples_list=data.test_triples_list
num_test_samples=len(triples_list)
pt_acc=0.0
foriinrange(num_test_samples):
test_sample=triples_list[i]
h,r,t=test_sample
start_time=time.time()
logits=model.forward(e1=torch.tensor(h),rel=torch.tensor(r))
end_time=time.time()
pt_inf_times.append(end_time-start_time)
score,pred=torch.topk(logits,TOP_K,1)

gt=np.array(sorted(t))
pred=np.array(sorted(pred[0].cpu().detach()))
pt_acc+=accuracy_score(gt,pred)

avg_pt_time=np.mean(pt_inf_times)*1000
print(f"Averagetimetakenforinference:{avg_pt_time}ms")
print(f"Meanaccuracyofthemodelonthetestdataset:{pt_acc/num_test_samples}")


..parsed-literal::

Averagetimetakenforinference:0.7081826527913412ms
Meanaccuracyofthemodelonthetestdataset:0.875


PredictionontheKnowledgegraph.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Here,weperformtheentitypredictionontheknowledgegraph,asa
sampleevaluationtask.Wepassthesourceentity``san_marino``and
relation``locatedIn``totheknowledgegraphandobtainthetarget
entitypredictions.Expectedpredictionsaretargetentitiesthatforma
factualtriplewiththeentityandrelationpassedasinputstothe
knowledgegraph.

..code::ipython3

entitynames_dict=data.ids2entities

ent="san_marino"
rel="locatedin"

h_idx=data.entity_ids[ent]
r_idx=data.rel_ids[rel]

logits=model.forward(torch.tensor(h_idx),torch.tensor(r_idx))
score,pred=torch.topk(logits,TOP_K,1)

forj,idinenumerate(pred[0].cpu().detach().numpy()):
pred_entity=entitynames_dict[id]
print(f"SourceEntity:{ent},Relation:{rel},Targetentityprediction:{pred_entity}")


..parsed-literal::

SourceEntity:san_marino,Relation:locatedin,Targetentityprediction:southern_europe
SourceEntity:san_marino,Relation:locatedin,Targetentityprediction:europe


ConvertthetrainedPyTorchmodeltoIRformatforOpenVINOinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ToevaluateperformancewithOpenVINO,wecaneitherconvertthetrained
PyTorchmodeltoanintermediaterepresentation(IR)format.
``ov.convert_model``functioncanbeusedforconversionPyTorchmodels
toOpenVINOModelclassinstance,thatisreadytoloadondeviceorcan
besavedondiskinOpenVINOIntermediateRepresentation(IR)format
using``ov.save_model``.

..code::ipython3

print("ConvertingthetrainedconvemodeltoIRformat")

ov_model=ov.convert_model(model,example_input=(torch.tensor(1),torch.tensor(1)))
ov.save_model(ov_model,ir_path)


..parsed-literal::

ConvertingthetrainedconvemodeltoIRformat


EvaluatethemodelperformancewithOpenVINO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,weevaluatethemodelperformancewiththeOpenVINOframework.In
ordertodoso,makethreemainAPIcalls:

1.InitializetheInferenceenginewith``Core()``
2.Loadthemodelwith``read_model()``
3.Compilethemodelwith``compile_model()``

Then,themodelcanbeinferredonbyusingthe
``create_infer_request()``APIcall.

..code::ipython3

core=ov.Core()
ov_model=core.read_model(model=ir_path)

Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importipywidgetsaswidgets

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



..code::ipython3

compiled_model=core.compile_model(model=ov_model,device_name=device.value)
input_layer_source=compiled_model.inputs[0]
input_layer_relation=compiled_model.inputs[1]
output_layer=compiled_model.output(0)

ov_acc=0.0
ov_inf_times=[]
foriinrange(num_test_samples):
test_sample=triples_list[i]
source,relation,target=test_sample
model_inputs={
input_layer_source:np.int64(source),
input_layer_relation:np.int64(relation),
}
start_time=time.time()
result=compiled_model(model_inputs)[output_layer]
end_time=time.time()
ov_inf_times.append(end_time-start_time)
top_k_idxs=list(np.argpartition(result[0],-TOP_K)[-TOP_K:])

gt=np.array(sorted(t))
pred=np.array(sorted(top_k_idxs))
ov_acc+=accuracy_score(gt,pred)

avg_ov_time=np.mean(ov_inf_times)*1000
print(f"Averagetimetakenforinference:{avg_ov_time}ms")
print(f"Meanaccuracyofthemodelonthetestdataset:{ov_acc/num_test_samples}")


..parsed-literal::

Averagetimetakenforinference:0.6203154722849528ms
Meanaccuracyofthemodelonthetestdataset:0.10416666666666667


DeterminetheplatformspecificspeedupobtainedthroughOpenVINOgraphoptimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#preventdivisionbyzero
delimiter=max(avg_ov_time,np.finfo(float).eps)

print(f"SpeedupwithOpenVINOoptimizations:{round(float(avg_pt_time)/float(delimiter),2)}X")


..parsed-literal::

SpeedupwithOpenVINOoptimizations:1.14X


BenchmarktheconvertedOpenVINOmodelusingbenchmarkapp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheOpenVINOtoolkitprovidesabenchmarkingapplicationtogaugethe
platformspecificruntimeperformancethatcanbeobtainedunderoptimal
configurationparametersforagivenmodel.Formoredetailsreferto:
https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html

Here,weusethebenchmarkapplicationtoobtainperformanceestimates
underoptimalconfigurationfortheknowledgegraphmodelinference.We
obtaintheaverage(AVG),minimum(MIN)aswellasmaximum(MAX)latency
aswellasthethroughputperformance(insamples/s)observedwhile
runningthebenchmarkapplication.Theplatformspecificoptimal
configurationparametersdeterminedbythebenchmarkingappforOpenVINO
inferencecanalsobeobtainedbylookingatthebenchmarkappresults.

..code::ipython3

print("BenchmarkOpenVINOmodelusingthebenchmarkapp")
!benchmark_app-m$ir_path-d$device.value-apiasync-t10-shape"input.1[1],input.2[1]"


..parsed-literal::

BenchmarkOpenVINOmodelusingthebenchmarkapp
[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]Deviceinfo:
[INFO]CPU
[INFO]Build.................................2024.2.0-15519-5c0f38f83f6-releases/2024/2
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(CPU)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook4.60ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]e1(node:e1):i64/[...]/[]
[INFO]rel(node:rel):i64/[...]/[]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:aten::softmax/Softmax):f32/[...]/[1,271]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]e1(node:e1):i64/[...]/[]
[INFO]rel(node:rel):i64/[...]/[]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:aten::softmax/Softmax):f32/[...]/[1,271]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook71.70ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:12
[INFO]NUM_STREAMS:12
[INFO]INFERENCE_NUM_THREADS:24
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
[WARNING]Noinputfilesweregivenforinput'e1'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'rel'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'e1'withrandomvalues
[INFO]Fillinput'rel'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,12inferencerequests,limits:10000msduration)
[INFO]Benchmarkingininferenceonlymode(inputsfillingarenotincludedinmeasurementloop).
[INFO]Firstinferencetook1.49ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:100644iterations
[INFO]Duration:10001.03ms
[INFO]Latency:
[INFO]Median:1.01ms
[INFO]Average:1.03ms
[INFO]Min:0.70ms
[INFO]Max:8.21ms
[INFO]Throughput:10063.36FPS


Conclusions
~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Inthisnotebook,weconvertthetrainedPyTorchknowledgegraph
embeddingsmodeltotheOpenVINOformat.Weconfirmthatthereareno
accuracydifferencespostconversion.Wealsoperformasample
evaluationontheknowledgegraph.Then,wedeterminetheplatform
specificspeedupinruntimeperformancethatcanbeobtainedthrough
OpenVINOgraphoptimizations.TolearnmoreabouttheOpenVINO
performanceoptimizations,referto:
https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference.html

References
~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

1.Convolutional2DKnowledgeGraphEmbeddings,TimDettmerset
al. (https://arxiv.org/abs/1707.01476)
2.Modelimplementation:https://github.com/TimDettmers/ConvE

TheConvEmodelimplementationusedinthisnotebookislicensedunder
theMITLicense.Thelicenseisdisplayedbelow:MITLicense

Copyright(c)2017TimDettmers

Permissionisherebygranted,freeofcharge,toanypersonobtaininga
copyofthissoftwareandassociateddocumentationfiles(the
“Software”),todealintheSoftwarewithoutrestriction,including
withoutlimitationtherightstouse,copy,modify,merge,publish,
distribute,sublicense,and/orsellcopiesoftheSoftware,andto
permitpersonstowhomtheSoftwareisfurnishedtodoso,subjectto
thefollowingconditions:

Theabovecopyrightnoticeandthispermissionnoticeshallbeincluded
inallcopiesorsubstantialportionsoftheSoftware.

THESOFTWAREISPROVIDED“ASIS”,WITHOUTWARRANTYOFANYKIND,EXPRESS
ORIMPLIED,INCLUDINGBUTNOTLIMITEDTOTHEWARRANTIESOF
MERCHANTABILITY,FITNESSFORAPARTICULARPURPOSEANDNONINFRINGEMENT.
INNOEVENTSHALLTHEAUTHORSORCOPYRIGHTHOLDERSBELIABLEFORANY
CLAIM,DAMAGESOROTHERLIABILITY,WHETHERINANACTIONOFCONTRACT,
TORTOROTHERWISE,ARISINGFROM,OUTOFORINCONNECTIONWITHTHE
SOFTWAREORTHEUSEOROTHERDEALINGSINTHESOFTWARE.
