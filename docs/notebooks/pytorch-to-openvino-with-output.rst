ConvertaPyTorchModeltoOpenVINO™IR
=======================================

Thistutorialdemonstratesstep-by-stepinstructionsonhowtodo
inferenceonaPyTorchclassificationmodelusingOpenVINORuntime.
StartingfromOpenVINO2023.0release,OpenVINOsupportsdirectPyTorch
modelconversionwithoutanintermediatesteptoconvertthemintoONNX
format.Inorder,ifyoutrytousethelowerOpenVINOversionorprefer
touseONNX,pleasecheckthis
`tutorial<pytorch-to-openvino-with-output.html>`__.

Inthistutorial,wewillusethe
`RegNetY_800MF<https://arxiv.org/abs/2003.13678>`__modelfrom
`torchvision<https://pytorch.org/vision/stable/index.html>`__to
demonstratehowtoconvertPyTorchmodelstoOpenVINOIntermediate
Representation.

TheRegNetmodelwasproposedin`DesigningNetworkDesign
Spaces<https://arxiv.org/abs/2003.13678>`__byIlijaRadosavovic,Raj
PrateekKosaraju,RossGirshick,KaimingHe,PiotrDollár.Theauthors
designsearchspacestoperformNeuralArchitectureSearch(NAS).They
firststartfromahighdimensionalsearchspaceanditerativelyreduce
thesearchspacebyempiricallyapplyingconstraintsbasedonthe
best-performingmodelssampledbythecurrentsearchspace.Insteadof
focusingondesigningindividualnetworkinstances,authorsdesign
networkdesignspacesthatparametrizepopulationsofnetworks.The
overallprocessisanalogoustotheclassicmanualdesignofnetworks
butelevatedtothedesignspacelevel.TheRegNetdesignspaceprovides
simpleandfastnetworksthatworkwellacrossawiderangeofflop
regimes.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`LoadPyTorchModel<#load-pytorch-model>`__

-`PrepareInputData<#prepare-input-data>`__
-`RunPyTorchModelInference<#run-pytorch-model-inference>`__
-`BenchmarkPyTorchModel
Inference<#benchmark-pytorch-model-inference>`__

-`ConvertPyTorchModeltoOpenVINOIntermediate
Representation<#convert-pytorch-model-to-openvino-intermediate-representation>`__

-`Selectinferencedevice<#select-inference-device>`__
-`RunOpenVINOModelInference<#run-openvino-model-inference>`__
-`BenchmarkOpenVINOModel
Inference<#benchmark-openvino-model-inference>`__

-`ConvertPyTorchModelwithStaticInput
Shape<#convert-pytorch-model-with-static-input-shape>`__

-`Selectinferencedevice<#select-inference-device>`__
-`RunOpenVINOModelInferencewithStaticInput
Shape<#run-openvino-model-inference-with-static-input-shape>`__
-`BenchmarkOpenVINOModelInferencewithStaticInput
Shape<#benchmark-openvino-model-inference-with-static-input-shape>`__

-`ConvertTorchScriptModeltoOpenVINOIntermediate
Representation<#convert-torchscript-model-to-openvino-intermediate-representation>`__

-`ScriptedModel<#scripted-model>`__
-`BenchmarkScriptedModel
Inference<#benchmark-scripted-model-inference>`__
-`ConvertPyTorchScriptedModeltoOpenVINOIntermediate
Representation<#convert-pytorch-scripted-model-to-openvino-intermediate-representation>`__
-`BenchmarkOpenVINOModelInferenceConvertedFromScripted
Model<#benchmark-openvino-model-inference-converted-from-scripted-model>`__
-`TracedModel<#traced-model>`__
-`BenchmarkTracedModel
Inference<#benchmark-traced-model-inference>`__
-`ConvertPyTorchTracedModeltoOpenVINOIntermediate
Representation<#convert-pytorch-traced-model-to-openvino-intermediate-representation>`__
-`BenchmarkOpenVINOModelInferenceConvertedFromTraced
Model<#benchmark-openvino-model-inference-converted-from-traced-model>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installnotebookdependencies

..code::ipython3

%pipinstall-q"openvino>=2023.1.0"scipyPillowtorchtorchvision--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


Downloadinputdataandlabelmap

..code::ipython3

importrequests
frompathlibimportPath
fromPILimportImage

MODEL_DIR=Path("model")
DATA_DIR=Path("data")

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MODEL_NAME="regnet_y_800mf"

image=Image.open(requests.get("https://farm9.staticflickr.com/8225/8511402100_fea15da1c5_z.jpg",stream=True).raw)

labels_file=DATA_DIR/"imagenet_2012.txt"

ifnotlabels_file.exists():
resp=requests.get("https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/imagenet_2012.txt")
withlabels_file.open("wb")asf:
f.write(resp.content)

imagenet_classes=labels_file.open("r").read().splitlines()

LoadPyTorchModel
------------------

`backtotop⬆️<#table-of-contents>`__

Generally,PyTorchmodelsrepresentaninstanceofthe
``torch.nn.Module``class,initializedbyastatedictionarywithmodel
weights.Typicalstepsforgettingapre-trainedmodel:

1.Createaninstanceofamodelclass
2.Loadcheckpointstatedict,whichcontainspre-trainedmodelweights
3.Turnthemodeltoevaluationforswitchingsomeoperationsto
inferencemode

The``torchvision``moduleprovidesaready-to-usesetoffunctionsfor
modelclassinitialization.Wewilluse
``torchvision.models.regnet_y_800mf``.Youcandirectlypasspre-trained
modelweightstothemodelinitializationfunctionusingtheweights
enum``RegNet_Y_800MF_Weights.DEFAULT``.

..code::ipython3

importtorchvision

#getdefaultweightsusingavailableweightsEnumformodel
weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT

#createmodeltopologyandloadweights
model=torchvision.models.regnet_y_800mf(weights=weights)

#switchmodeltoinferencemode
model.eval();

PrepareInputData
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thecodebelowdemonstrateshowtopreprocessinputdatausinga
model-specifictransformsmodulefrom``torchvision``.After
transformation,weshouldconcatenateimagesintobatchedtensor,inour
case,wewillrunthemodelwithbatch1,sowejustunsqueezeinputon
thefirstdimension.

..code::ipython3

importtorch

#InitializetheWeightTransforms
preprocess=weights.transforms()

#Applyittotheinputimage
img_transformed=preprocess(image)

#Addbatchdimensiontoimagetensor
input_tensor=img_transformed.unsqueeze(0)

RunPyTorchModelInference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Themodelreturnsavectorofprobabilitiesinrawlogitsformat,
softmaxcanbeappliedtogetnormalizedvaluesinthe[0,1]range.For
ademonstrationthattheoutputoftheoriginalmodelandOpenVINO
convertedisthesame,wedefinedacommonpostprocessingfunctionwhich
canbereusedlater.

..code::ipython3

importnumpyasnp
fromscipy.specialimportsoftmax

#Performmodelinferenceoninputtensor
result=model(input_tensor)


#PostprocessingfunctionforgettingresultsinthesamewayforbothPyTorchmodelinferenceandOpenVINO
defpostprocess_result(output_tensor:np.ndarray,top_k:int=5):
"""
Posprocessmodelresults.Thisfunctionappliedsofrmaxonoutputtensorandreturnsspecifiedtop_knumberoflabelswithhighestprobability
Parameters:
output_tensor(np.ndarray):modeloutputtensorwithprobabilities
top_k(int,*optional*,default5):numberoflabelswithhighestprobabilityforreturn
Returns:
topk_labels:labelidsforselectedtop_kscores
topk_scores:selectedtop_khighestscorespredictedbymodel
"""
softmaxed_scores=softmax(output_tensor,-1)[0]
topk_labels=np.argsort(softmaxed_scores)[-top_k:][::-1]
topk_scores=softmaxed_scores[topk_labels]
returntopk_labels,topk_scores


#Postprocessresults
top_labels,top_scores=postprocess_result(result.detach().numpy())

#Showresults
display(image)
foridx,(label,score)inenumerate(zip(top_labels,top_scores)):
_,predicted_label=imagenet_classes[label].split("",1)
print(f"{idx+1}:{predicted_label}-{score*100:.2f}%")



..image::pytorch-to-openvino-with-output_files/pytorch-to-openvino-with-output_11_0.png


..parsed-literal::

1:tigercat-25.91%
2:Egyptiancat-10.26%
3:computerkeyboard,keypad-9.22%
4:tabby,tabbycat-9.09%
5:hamper-2.35%


BenchmarkPyTorchModelInference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%timeit

#Runmodelinference
model(input_tensor)


..parsed-literal::

16.4ms±702µsperloop(mean±std.dev.of7runs,100loopseach)


ConvertPyTorchModeltoOpenVINOIntermediateRepresentation
-------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Startingfromthe2023.0releaseOpenVINOsupportsdirectPyTorchmodels
conversiontoOpenVINOIntermediateRepresentation(IR)format.OpenVINO
modelconversionAPIshouldbeusedforthesepurposes.Moredetails
regardingPyTorchmodelconversioncanbefoundinOpenVINO
`documentation<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-pytorch.html>`__

The``convert_model``functionacceptsthePyTorchmodelobjectand
returnsthe``openvino.Model``instancereadytoloadonadeviceusing
``core.compile_model``orsaveondiskfornextusageusing
``ov.save_model``.Optionally,wecanprovideadditionalparameters,
suchas:

-``compress_to_fp16``-flagtoperformmodelweightscompressioninto
FP16dataformat.Itmayreducetherequiredspaceformodelstorage
ondiskandgivespeedupforinferencedevices,whereFP16
calculationissupported.
-``example_input``-inputdatasamplewhichcanbeusedformodel
tracing.
-``input_shape``-theshapeofinputtensorforconversion

andanyotheradvancedoptionssupportedbymodelconversionPythonAPI.
Moredetailscanbefoundonthis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/conversion-parameters.html>`__

..code::ipython3

importopenvinoasov

#CreateOpenVINOCoreobjectinstance
core=ov.Core()

#Convertmodeltoopenvino.runtime.Modelobject
ov_model=ov.convert_model(model)

#Saveopenvino.runtime.Modelobjectondisk
ov.save_model(ov_model,MODEL_DIR/f"{MODEL_NAME}_dynamic.xml")

ov_model




..parsed-literal::

<Model:'Model30'
inputs[
<ConstOutput:names[x]shape[?,3,?,?]type:f32>
]
outputs[
<ConstOutput:names[x.21]shape[?,1000]type:f32>
]>



Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importipywidgetsaswidgets

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

#LoadOpenVINOmodelondevice
compiled_model=core.compile_model(ov_model,device.value)
compiled_model




..parsed-literal::

<CompiledModel:
inputs[
<ConstOutput:names[x]shape[?,3,?,?]type:f32>
]
outputs[
<ConstOutput:names[x.21]shape[?,1000]type:f32>
]>



RunOpenVINOModelInference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Runmodelinference
result=compiled_model(input_tensor)[0]

#Posptorcessresults
top_labels,top_scores=postprocess_result(result)

#Showresults
display(image)
foridx,(label,score)inenumerate(zip(top_labels,top_scores)):
_,predicted_label=imagenet_classes[label].split("",1)
print(f"{idx+1}:{predicted_label}-{score*100:.2f}%")



..image::pytorch-to-openvino-with-output_files/pytorch-to-openvino-with-output_20_0.png


..parsed-literal::

1:tigercat-25.91%
2:Egyptiancat-10.26%
3:computerkeyboard,keypad-9.22%
4:tabby,tabbycat-9.09%
5:hamper-2.35%


BenchmarkOpenVINOModelInference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%timeit

compiled_model(input_tensor)


..parsed-literal::

3.2ms±7.83µsperloop(mean±std.dev.of7runs,100loopseach)


ConvertPyTorchModelwithStaticInputShape
---------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Thedefaultconversionpathpreservesdynamicinputshapes,inorderif
youwanttoconvertthemodelwithstaticshapes,youcanexplicitly
specifyitduringconversionusingthe``input_shape``parameteror
reshapethemodelintothedesiredshapeafterconversion.Forthemodel
reshapingexamplepleasecheckthefollowing
`tutorial<openvino-api-with-output.html>`__.

..code::ipython3

#Convertmodeltoopenvino.runtime.Modelobject
ov_model=ov.convert_model(model,input=[[1,3,224,224]])
#Saveopenvino.runtime.Modelobjectondisk
ov.save_model(ov_model,MODEL_DIR/f"{MODEL_NAME}_static.xml")
ov_model




..parsed-literal::

<Model:'Model65'
inputs[
<ConstOutput:names[x]shape[1,3,224,224]type:f32>
]
outputs[
<ConstOutput:names[x.21]shape[1,1000]type:f32>
]>



Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

#LoadOpenVINOmodelondevice
compiled_model=core.compile_model(ov_model,device.value)
compiled_model




..parsed-literal::

<CompiledModel:
inputs[
<ConstOutput:names[x]shape[1,3,224,224]type:f32>
]
outputs[
<ConstOutput:names[x.21]shape[1,1000]type:f32>
]>



Now,wecanseethatinputofourconvertedmodelistensorofshape[1,
3,224,224]insteadof[?,3,?,?]reportedbypreviouslyconverted
model.

RunOpenVINOModelInferencewithStaticInputShape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Runmodelinference
result=compiled_model(input_tensor)[0]

#Posptorcessresults
top_labels,top_scores=postprocess_result(result)

#Showresults
display(image)
foridx,(label,score)inenumerate(zip(top_labels,top_scores)):
_,predicted_label=imagenet_classes[label].split("",1)
print(f"{idx+1}:{predicted_label}-{score*100:.2f}%")



..image::pytorch-to-openvino-with-output_files/pytorch-to-openvino-with-output_31_0.png


..parsed-literal::

1:tigercat-25.91%
2:Egyptiancat-10.26%
3:computerkeyboard,keypad-9.22%
4:tabby,tabbycat-9.09%
5:hamper-2.35%


BenchmarkOpenVINOModelInferencewithStaticInputShape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%timeit

compiled_model(input_tensor)


..parsed-literal::

2.84ms±8.04µsperloop(mean±std.dev.of7runs,100loopseach)


ConvertTorchScriptModeltoOpenVINOIntermediateRepresentation
-----------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

TorchScriptisawaytocreateserializableandoptimizablemodelsfrom
PyTorchcode.AnyTorchScriptprogramcanbesavedfromaPythonprocess
andloadedinaprocesswherethereisnoPythondependency.More
detailsaboutTorchScriptcanbefoundin`PyTorch
documentation<https://pytorch.org/docs/stable/jit.html>`__.

Thereare2possiblewaystoconvertthePyTorchmodeltoTorchScript:

-``torch.jit.script``-Scriptingafunctionor``nn.Module``will
inspectthesourcecode,compileitasTorchScriptcodeusingthe
TorchScriptcompiler,andreturna``ScriptModule``or
``ScriptFunction``.
-``torch.jit.trace``-Traceafunctionandreturnanexecutableor
``ScriptFunction``thatwillbeoptimizedusingjust-in-time
compilation.

Let’sconsiderbothapproachesandtheirconversionintoOpenVINOIR.

ScriptedModel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

``torch.jit.script``inspectsmodelsourcecodeandcompilesitto
``ScriptModule``.Aftercompilationmodelcanbeusedforinferenceor
savedondiskusingthe``torch.jit.save``functionandafterthat
restoredwith``torch.jit.load``inanyotherenvironmentwithoutthe
originalPyTorchmodelcodedefinitions.

TorchScriptitselfisasubsetofthePythonlanguage,sonotall
featuresinPythonwork,butTorchScriptprovidesenoughfunctionality
tocomputeontensorsanddocontrol-dependentoperations.Fora
completeguide,seethe`TorchScriptLanguage
Reference<https://pytorch.org/docs/stable/jit_language_reference.html#language-reference>`__.

..code::ipython3

#Getmodelpath
scripted_model_path=MODEL_DIR/f"{MODEL_NAME}_scripted.pth"

#Compileandsavemodelifithasnotbeencompiledbeforeorloadcompiledmodel
ifnotscripted_model_path.exists():
scripted_model=torch.jit.script(model)
torch.jit.save(scripted_model,scripted_model_path)
else:
scripted_model=torch.jit.load(scripted_model_path)

#Runscriptedmodelinference
result=scripted_model(input_tensor)

#Postprocessresults
top_labels,top_scores=postprocess_result(result.detach().numpy())

#Showresults
display(image)
foridx,(label,score)inenumerate(zip(top_labels,top_scores)):
_,predicted_label=imagenet_classes[label].split("",1)
print(f"{idx+1}:{predicted_label}-{score*100:.2f}%")



..image::pytorch-to-openvino-with-output_files/pytorch-to-openvino-with-output_35_0.png


..parsed-literal::

1:tigercat-25.91%
2:Egyptiancat-10.26%
3:computerkeyboard,keypad-9.22%
4:tabby,tabbycat-9.09%
5:hamper-2.35%


BenchmarkScriptedModelInference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%timeit

scripted_model(input_tensor)


..parsed-literal::

14.1ms±70.6µsperloop(mean±std.dev.of7runs,100loopseach)


ConvertPyTorchScriptedModeltoOpenVINOIntermediateRepresentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheconversionstepforthescriptedmodeltoOpenVINOIRissimilarto
theoriginalPyTorchmodel.

..code::ipython3

#Convertmodeltoopenvino.runtime.Modelobject
ov_model=ov.convert_model(scripted_model)

#LoadOpenVINOmodelondevice
compiled_model=core.compile_model(ov_model,device.value)

#RunOpenVINOmodelinference
result=compiled_model(input_tensor,device.value)[0]

#Postprocessresults
top_labels,top_scores=postprocess_result(result)

#Showresults
display(image)
foridx,(label,score)inenumerate(zip(top_labels,top_scores)):
_,predicted_label=imagenet_classes[label].split("",1)
print(f"{idx+1}:{predicted_label}-{score*100:.2f}%")



..image::pytorch-to-openvino-with-output_files/pytorch-to-openvino-with-output_39_0.png


..parsed-literal::

1:tigercat-25.91%
2:Egyptiancat-10.26%
3:computerkeyboard,keypad-9.22%
4:tabby,tabbycat-9.09%
5:hamper-2.35%


BenchmarkOpenVINOModelInferenceConvertedFromScriptedModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%timeit

compiled_model(input_tensor)


..parsed-literal::

3.17ms±9.55µsperloop(mean±std.dev.of7runs,100loopseach)


TracedModel
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Using``torch.jit.trace``,youcanturnanexistingmoduleorPython
functionintoaTorchScript``ScriptFunction``or``ScriptModule``.You
mustprovideexampleinputs,andmodelwillbeexecuted,recordingthe
operationsperformedonallthetensors.

-Theresultingrecordingofastandalonefunctionproduces
``ScriptFunction``.

-Theresultingrecordingof``nn.Module.forward``or``nn.Module``
produces``ScriptModule``.

Inthesamewaylikescriptedmodel,tracedmodelcanbeusedfor
inferenceorsavedondiskusing``torch.jit.save``functionandafter
thatrestoredwith``torch.jit.load``inanyotherenvironmentwithout
originalPyTorchmodelcodedefinitions.

..code::ipython3

#Getmodelpath
traced_model_path=MODEL_DIR/f"{MODEL_NAME}_traced.pth"

#Traceandsavemodelifithasnotbeentracedbeforeorloadtracedmodel
ifnottraced_model_path.exists():
traced_model=torch.jit.trace(model,example_inputs=input_tensor)
torch.jit.save(traced_model,traced_model_path)
else:
traced_model=torch.jit.load(traced_model_path)

#Runtracedmodelinference
result=traced_model(input_tensor)

#Postprocessresults
top_labels,top_scores=postprocess_result(result.detach().numpy())

#Showresults
display(image)
foridx,(label,score)inenumerate(zip(top_labels,top_scores)):
_,predicted_label=imagenet_classes[label].split("",1)
print(f"{idx+1}:{predicted_label}-{score*100:.2f}%")



..image::pytorch-to-openvino-with-output_files/pytorch-to-openvino-with-output_43_0.png


..parsed-literal::

1:tigercat-25.91%
2:Egyptiancat-10.26%
3:computerkeyboard,keypad-9.22%
4:tabby,tabbycat-9.09%
5:hamper-2.35%


BenchmarkTracedModelInference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%timeit

traced_model(input_tensor)


..parsed-literal::

14.8ms±412µsperloop(mean±std.dev.of7runs,100loopseach)


ConvertPyTorchTracedModeltoOpenVINOIntermediateRepresentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheconversionstepforatracedmodeltoOpenVINOIRissimilartothe
originalPyTorchmodel.

..code::ipython3

#Convertmodeltoopenvino.runtime.Modelobject
ov_model=ov.convert_model(traced_model)

#LoadOpenVINOmodelondevice
compiled_model=core.compile_model(ov_model,device.value)

#RunOpenVINOmodelinference
result=compiled_model(input_tensor)[0]

#Postprocessresults
top_labels,top_scores=postprocess_result(result)

#Showresults
display(image)
foridx,(label,score)inenumerate(zip(top_labels,top_scores)):
_,predicted_label=imagenet_classes[label].split("",1)
print(f"{idx+1}:{predicted_label}-{score*100:.2f}%")



..image::pytorch-to-openvino-with-output_files/pytorch-to-openvino-with-output_47_0.png


..parsed-literal::

1:tigercat-25.91%
2:Egyptiancat-10.26%
3:computerkeyboard,keypad-9.22%
4:tabby,tabbycat-9.09%
5:hamper-2.35%


BenchmarkOpenVINOModelInferenceConvertedFromTracedModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%timeit

compiled_model(input_tensor)[0]


..parsed-literal::

3.2ms±8.84µsperloop(mean±std.dev.of7runs,100loopseach)

