Magika:AIpoweredfastandefficientfiletypeidentificationusingOpenVINO
=============================================================================

MagikaisanovelAIpoweredfiletypedetectiontoolthatreliesonthe
recentadvanceofdeeplearningtoprovideaccuratedetection.Underthe
hood,Magikaemploysacustom,highlyoptimizedmodelthatonlyweighs
about1MB,andenablesprecisefileidentificationwithinmilliseconds,
evenwhenrunningonasingleCPU.

Whyidentifyingfiletypeisdifficult
--------------------------------------

Sincetheearlydaysofcomputing,accuratelydetectingfiletypeshas
beencrucialindetermininghowtoprocessfiles.Linuxcomesequipped
with``libmagic``andthe``file``utility,whichhaveservedasthede
factostandardforfiletypeidentificationforover50years.Todayweb
browsers,codeeditors,andcountlessothersoftwarerelyonfile-type
detectiontodecidehowtoproperlyrenderafile.Forexample,modern
codeeditorsusefile-typedetectiontochoosewhichsyntaxcoloring
schemetouseasthedeveloperstartstypinginanewfile.

Accuratefile-typedetectionisanotoriouslydifficultproblembecause
eachfileformathasadifferentstructure,ornostructureatall.This
isparticularlychallengingfortextualformatsandprogramming
languagesastheyhaveverysimilarconstructs.Sofar,``libmagic``and
mostotherfile-type-identificationsoftwarehavebeenrelyingona
handcraftedcollectionofheuristicsandcustomrulestodetecteach
fileformat.

Thismanualapproachisbothtimeconsuminganderrorproneasitis
hardforhumanstocreategeneralizedrulesbyhand.Inparticularfor
securityapplications,creatingdependabledetectionisespecially
challengingasattackersareconstantlyattemptingtoconfusedetection
withadversarially-craftedpayloads.

Toaddressthisissueandprovidefastandaccuratefile-typedetection
Magikawasdeveloped.Moredetailsaboutapproachandmodelcanbefound
inoriginal`repo<https://github.com/google/magika>`__and`Google’s
blog
post<https://opensource.googleblog.com/2024/02/magika-ai-powered-fast-and-efficient-file-type-identification.html>`__.

InthistutorialweconsiderhowtobringOpenVINOpowerintoMagika.
####Tableofcontents:

-`Prerequisites<#prerequisites>`__
-`Definemodelloadingclass<#define-model-loading-class>`__
-`RunOpenVINOmodelinference<#run-openvino-model-inference>`__

-`Selectdevice<#select-device>`__
-`Createmodel<#create-model>`__
-`Runinferenceonbytesinput<#run-inference-on-bytes-input>`__
-`Runinferenceonfileinput<#run-inference-on-file-input>`__

-`Interactivedemo<#interactive-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__##Prerequisites

..code::ipython3

%pipinstall-qmagika"openvino>=2024.1.0""gradio>=4.19"


..parsed-literal::

ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
openvino-dev2024.2.0requiresopenvino==2024.2.0,butyouhaveopenvino2024.3.0.dev20240711whichisincompatible.
tensorflow2.12.0requiresnumpy<1.24,>=1.22,butyouhavenumpy1.24.4whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Definemodelloadingclass
--------------------------

`backtotop⬆️<#table-of-contents>`__

AtinferencetimeMagikausesONNXasaninferenceenginetoensure
filesareidentifiedinamatterofmilliseconds,almostasfastasa
non-AItoolevenonCPU.ThecodebelowextendingoriginalMagika
inferenceclasswithOpenVINOAPI.Theprovidedcodeisfullycompatible
withoriginal`MagikaPython
API<https://github.com/google/magika/blob/main/docs/python.md>`__.

..code::ipython3

importtime
frompathlibimportPath
fromfunctoolsimportpartial
fromtypingimportList,Tuple,Optional,Dict

frommagikaimportMagika
frommagika.typesimportModelFeatures,ModelOutput,MagikaResult
frommagika.prediction_modeimportPredictionMode
importnumpy.typingasnpt
importnumpyasnp

importopenvinoasov


classOVMagika(Magika):
def__init__(
self,
model_dir:Optional[Path]=None,
prediction_mode:PredictionMode=PredictionMode.HIGH_CONFIDENCE,
no_dereference:bool=False,
verbose:bool=False,
debug:bool=False,
use_colors:bool=False,
device="CPU",
)->None:
self._device=device
super().__init__(model_dir,prediction_mode,no_dereference,verbose,debug,use_colors)

def_init_onnx_session(self):
#overloadmodelloadingusingOpenVINO
start_time=time.time()
core=ov.Core()
ov_model=core.compile_model(self._model_path,self._device.upper())
elapsed_time=1000*(time.time()-start_time)
self._log.debug(f'ONNXDLmodel"{self._model_path}"loadedin{elapsed_time:.03f}mson{self._device}')
returnov_model

def_get_raw_predictions(self,features:List[Tuple[Path,ModelFeatures]])->npt.NDArray:
"""
Givenalistof(path,features),returna(files_num,features_size)
matrixencodingthepredictions.
"""

dataset_format=self._model_config["train_dataset_info"]["dataset_format"]
assertdataset_format=="int-concat/one-hot"
start_time=time.time()
X_bytes=[]
for_,fsinfeatures:
sample_bytes=[]
ifself._input_sizes["beg"]>0:
sample_bytes.extend(fs.beg[:self._input_sizes["beg"]])
ifself._input_sizes["mid"]>0:
sample_bytes.extend(fs.mid[:self._input_sizes["mid"]])
ifself._input_sizes["end"]>0:
sample_bytes.extend(fs.end[-self._input_sizes["end"]:])
X_bytes.append(sample_bytes)
X=np.array(X_bytes).astype(np.float32)
elapsed_time=time.time()-start_time
self._log.debug(f"DLinputpreparedin{elapsed_time:.03f}seconds")

start_time=time.time()
raw_predictions_list=[]
samples_num=X.shape[0]

max_internal_batch_size=1000
batches_num=samples_num//max_internal_batch_size
ifsamples_num%max_internal_batch_size!=0:
batches_num+=1

forbatch_idxinrange(batches_num):
self._log.debug(f"Gettingrawpredictionsfor(internal)batch{batch_idx+1}/{batches_num}")
start_idx=batch_idx*max_internal_batch_size
end_idx=min((batch_idx+1)*max_internal_batch_size,samples_num)
batch_raw_predictions=self._onnx_session({"bytes":X[start_idx:end_idx,:]})["target_label"]
raw_predictions_list.append(batch_raw_predictions)
elapsed_time=time.time()-start_time
self._log.debug(f"DLrawpredictionin{elapsed_time:.03f}seconds")
returnnp.concatenate(raw_predictions_list)

def_get_topk_model_outputs_from_features(self,all_features:List[Tuple[Path,ModelFeatures]],k:int=5)->List[Tuple[Path,List[ModelOutput]]]:
"""
Helperfunctionforgettingtopkthehighestrankedmodelresultsforeachfeature
"""
raw_preds=self._get_raw_predictions(all_features)
top_preds_idxs=np.argsort(raw_preds,axis=1)[:,-k:][:,::-1]
scores=[raw_preds[i,idx]fori,idxinenumerate(top_preds_idxs)]
results=[]
for(path,_),scores,top_idxesinzip(all_features,raw_preds,top_preds_idxs):
model_outputs_for_path=[]
foridxintop_idxes:
ct_label=self._target_labels_space_np[idx]
score=scores[idx]
model_outputs_for_path.append(ModelOutput(ct_label=ct_label,score=float(score)))
results.append((path,model_outputs_for_path))
returnresults

def_get_results_from_features_topk(self,all_features:List[Tuple[Path,ModelFeatures]],top_k=5)->Dict[str,MagikaResult]:
"""
Helperfunctionforgettingtopkthehighestrankedmodelresultsforeachfeature
"""
#Wenowdoinferenceforthosefilesthatneedit.

iflen(all_features)==0:
#nothingtobedone
return{}

outputs:Dict[str,MagikaResult]={}

forpath,model_outputinself._get_topk_model_outputs_from_features(all_features,top_k):
#InadditionaltothecontenttypelabelfromtheDLmodel,we
#alsoallowforotherlogictooverwritesuchresult.For
#debuggingandinformationpurposes,theJSONoutputstores
#boththerawDLmodeloutputandthefinaloutputwereturnto
#theuser.
results=[]
foroutinmodel_output:
output_ct_label=self._get_output_ct_label_from_dl_result(out.ct_label,out.score)

results.append(
self._get_result_from_labels_and_score(
path,
dl_ct_label=out.ct_label,
output_ct_label=output_ct_label,
score=out.score,
)
)
outputs[str(path)]=results

returnoutputs

defidentify_bytes_topk(self,content:bytes,top_k=5)->MagikaResult:
#Helperfunctionforgettingtopkresultsfrombytes
_get_results_from_features=self._get_results_from_features
self._get_results_from_features=partial(self._get_results_from_features_topk,top_k=top_k)
result=super().identify_bytes(content)
self._get_results_from_features=_get_results_from_features
returnresult

RunOpenVINOmodelinference
----------------------------

`backtotop⬆️<#table-of-contents>`__

Nowlet’scheckmodelinferenceresult.

Selectdevice
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Forstartingwork,please,selectoneofrepresenteddevicesfrom
dropdownlist.

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



Createmodel
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Aswediscussedabove,ourOpenVINOextended``OVMagika``classhasthe
sameAPIlikeoriginalone.Let’strytocreateinterfaceinstanceand
launchitondifferentinputformats

..code::ipython3

ov_magika=OVMagika(device=device.value)

Runinferenceonbytesinput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

result=ov_magika.identify_bytes(b"#Example\nThisisanexampleofmarkdown!")
print(f"Contenttype:{result.output.ct_label}-{result.output.score*100:.4}%")


..parsed-literal::

Contenttype:markdown-99.29%


Runinferenceonfileinput
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importrequests

input_file=Path("./README.md")
ifnotinput_file.exists():
r=requests.get("https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/README.md")
withopen("README.md","w")asf:
f.write(r.text)
result=ov_magika.identify_path(input_file)
print(f"Contenttype:{result.output.ct_label}-{result.output.score*100:.4}%")


..parsed-literal::

Contenttype:markdown-100.0%


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

Now,youcantrymodelonownfiles.Uploadfileintoinputfilewindow,
clicksubmitbuttonandlookonpredictedfiletypes.

..code::ipython3

importgradioasgr


defclassify(file_path):
"""Classifyfileusingclasseslisting.
Args:
file_path):pathtoinputfile
Returns:
(dict):Mappingbetweenclasslabelsandclassprobabilities.
"""
results=ov_magika.identify_bytes_topk(file_path)

return{result.dl.ct_label:float(result.output.score)forresultinresults}


demo=gr.Interface(
classify,
[
gr.File(label="Inputfile",type="binary"),
],
gr.Label(label="Result"),
examples=[["./README.md"]],
allow_flagging="never",
)
try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>

