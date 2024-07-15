#ConvertDetectron2ModelstoOpenVINO™

`Detectron2<https://github.com/facebookresearch/detectron2>`__is
FacebookAIResearch’slibrarythatprovidesstate-of-the-artdetection
andsegmentationalgorithms.Itisthesuccessorof
`Detectron<https://github.com/facebookresearch/Detectron/>`__and
`maskrcnn-benchmark<https://github.com/facebookresearch/maskrcnn-benchmark/>`__.
Itsupportsanumberofcomputervisionresearchprojectsandproduction
applications.

InthistutorialweconsiderhowtoconvertandrunDetectron2models
usingOpenVINO™.Wewilluse``FasterR-CNNFPNx1``modeland
``MaskR-CNNFPNx3``pretrainedon
`COCO<https://cocodataset.org/#home>`__datasetasexamplesforobject
detectionandinstancesegmentationrespectively.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__

-`DefinehelpersforPyTorchmodelinitializationand
conversion<#define-helpers-for-pytorch-model-initialization-and-conversion>`__
-`Prepareinputdata<#prepare-input-data>`__

-`ObjectDetection<#object-detection>`__

-`DownloadPyTorchDetection
model<#download-pytorch-detection-model>`__
-`ConvertDetectionModeltoOpenVINOIntermediate
Representation<#convert-detection-model-to-openvino-intermediate-representation>`__
-`Selectinferencedevice<#select-inference-device>`__
-`RunDetectionmodelinference<#run-detection-model-inference>`__

-`InstanceSegmentation<#instance-segmentation>`__

-`DownloadInstanceSegmentationPyTorch
model<#download-instance-segmentation-pytorch-model>`__
-`ConvertInstanceSegmentationModeltoOpenVINOIntermediate
Representation<#convert-instance-segmentation-model-to-openvino-intermediate-representation>`__
-`Selectinferencedevice<#select-inference-device>`__
-`RunInstanceSegmentationmodel
inference<#run-instance-segmentation-model-inference>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installrequiredpackagesforrunningmodel

..code::ipython3

%pipinstall-q"torch""torchvision""opencv-python""wheel"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"git+https://github.com/facebookresearch/detectron2.git"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2023.1.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


DefinehelpersforPyTorchmodelinitializationandconversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Detectron2providesuniversalandconfigurableAPIforworkingwith
models,itmeansthatallstepsrequiredformodelcreation,conversion
andinferencewillbecommonforallmodels,thatiswhyitisenoughto
definehelperfunctionsonce,thenreusethemfordifferentmodels.For
obtainingmodelswewilluse`Detectron2Model
Zoo<https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md>`__
API.``detecton_zoo.get``functionallowtodownloadandinstantiate
modelbasedonitsconfigfile.Configurationfileisplayingkeyrole
ininteractionwithmodelsinDetectron2projectanddescribesmodel
architectureandtrainingandvalidationprocesses.
``detectron_zoo.get_config``functioncanbeusedforfindingand
readingmodelconfig.

..code::ipython3

importdetectron2.model_zooasdetectron_zoo


defget_model_and_config(model_name:str):
"""
HelperfunctionfordownloadingPyTorchmodelanditsconfigurationfromDetectron2ModelZoo

Parameters:
model_name(str):model_idfromDetectron2ModelZoo
Returns:
model(torch.nn.Module):Pretrainedmodelinstance
cfg(Config):Configurationformodel
"""
cfg=detectron_zoo.get_config(model_name+".yaml",trained=True)
model=detectron_zoo.get(model_name+".yaml",trained=True)
returnmodel,cfg

Detectron2libraryisbasedonPyTorch.Startingfrom2023.0release
OpenVINOsupportsPyTorchmodelsconversiondirectlyviaModel
ConversionAPI.``ov.convert_model``functioncanbeusedforconverting
PyTorchmodeltoOpenVINOModelobjectinstance,thatreadytousefor
loadingondeviceandthenrunninginferenceorcanbesavedondiskfor
nextdeploymentusing``ov.save_model``function.

Detectron2modelsusecustomcomplexdatastructuresinsidethatbrings
somedifficultiesforexportingmodelsindifferentformatsand
frameworksincludingOpenVINO.Foravoidtheseissues,
``detectron2.export.TracingAdapter``providedaspartofDetectron2
deploymentAPI.``TracingAdapter``isamodelwrapperclassthat
simplifymodel’sstructuremakingitmoreexport-friendly.

..code::ipython3

fromdetectron2.modelingimportGeneralizedRCNN
fromdetectron2.exportimportTracingAdapter
importtorch
importopenvinoasov
importwarnings
fromtypingimportList,Dict


defconvert_detectron2_model(model:torch.nn.Module,sample_input:List[Dict[str,torch.Tensor]]):
"""
FunctionforconvertingDetectron2models,createsTracingAdapterformakingmodeltracing-friendly,
preparesinputsandconvertsmodeltoOpenVINOModel

Parameters:
model(torch.nn.Module):Modelobjectforconversion
sample_input(List[Dict[str,torch.Tensor]]):sampleinputfortracing
Returns:
ov_model(ov.Model):OpenVINOModel
"""
#prepareinputfortracingadapter
tracing_input=[{"image":sample_input[0]["image"]}]

#overridemodelforwardanddisablepostprocessingifrequired
ifisinstance(model,GeneralizedRCNN):

definference(model,inputs):
#usedo_postprocess=FalsesoitreturnsROImask
inst=model.inference(inputs,do_postprocess=False)[0]
return[{"instances":inst}]

else:
inference=None#assumethatwejustcallthemodeldirectly

#createtraceablemodel
traceable_model=TracingAdapter(model,tracing_input,inference)
warnings.filterwarnings("ignore")
#convertPyTorchmodeltoOpenVINOmodel
ov_model=ov.convert_model(traceable_model,example_input=sample_input[0]["image"])
returnov_model

Prepareinputdata
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Forrunningmodelconversionandinferenceweneedtoprovideexample
input.Thecellsbelowdownloadsampleimageandapplypreprocessing
stepsbasedonmodelspecifictransformationsdefinedinmodelconfig.

..code::ipython3

importrequests
frompathlibimportPath
fromPILimportImage

MODEL_DIR=Path("model")
DATA_DIR=Path("data")

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

input_image_url="https://farm9.staticflickr.com/8040/8017130856_1b46b5f5fc_z.jpg"

image_file=DATA_DIR/"example_image.jpg"

ifnotimage_file.exists():
image=Image.open(requests.get(input_image_url,stream=True).raw)
image.save(image_file)
else:
image=Image.open(image_file)

image




..image::detectron2-to-openvino-with-output_files/detectron2-to-openvino-with-output_8_0.png



..code::ipython3

importdetectron2.data.transformsasT
fromdetectron2.dataimportdetection_utils
importtorch


defget_sample_inputs(image_path,cfg):
#getasampledata
original_image=detection_utils.read_image(image_path,format=cfg.INPUT.FORMAT)
#DosamepreprocessingasDefaultPredictor
aug=T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MIN_SIZE_TEST],cfg.INPUT.MAX_SIZE_TEST)
height,width=original_image.shape[:2]
image=aug.get_transform(original_image).apply_image(original_image)
image=torch.as_tensor(image.astype("float32").transpose(2,0,1))

inputs={"image":image,"height":height,"width":width}

#Sampleready
sample_inputs=[inputs]
returnsample_inputs

Now,whenallcomponentsrequiredformodelconversionareprepared,we
canconsiderhowtousethemonspecificexamples.

ObjectDetection
----------------

`backtotop⬆️<#table-of-contents>`__

DownloadPyTorchDetectionmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Downloadfaster_rcnn_R_50_FPN_1xfromDetectronModelZoo.

..code::ipython3

model_name="COCO-Detection/faster_rcnn_R_50_FPN_1x"
model,cfg=get_model_and_config(model_name)
sample_input=get_sample_inputs(image_file,cfg)

ConvertDetectionModeltoOpenVINOIntermediateRepresentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Convertmodelusing``convert_detectron2_model``functionand
``sample_input``preparedabove.Afterconversion,modelsavedondisk
using``ov.save_model``functionandcanbefoundin``model``
directory.

..code::ipython3

model_xml_path=MODEL_DIR/(model_name.split("/")[-1]+".xml")
ifnotmodel_xml_path.exists():
ov_model=convert_detectron2_model(model,sample_input)
ov.save_model(ov_model,MODEL_DIR/(model_name.split("/")[-1]+".xml"))
else:
ov_model=model_xml_path


..parsed-literal::

['args']


Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

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



RunDetectionmodelinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Loadourconvertedmodelonselecteddeviceandruninferenceonsample
input.

..code::ipython3

compiled_model=core.compile_model(ov_model,device.value)

..code::ipython3

results=compiled_model(sample_input[0]["image"])

Tracingadaptersimplifiesmodelinputandoutputformat.After
conversion,modelhasmultipleoutputsinfollowingformat:1.Predicted
boxesisfloating-pointtensorinformat[``N``,4],whereNisnumber
ofdetectedboxes.2.Predictedclassesisintegertensorinformat
[``N``],whereNisnumberofpredictedobjectsthatdefineswhichlabel
eachobjectbelongs.Thevaluesrangeofpredictedclassestensoris[0,
``num_labels``],where``num_labels``isnumberofclassessupportedof
model(inourcase80).3.Predictedscoresisfloating-pointtensorin
format[``N``],where``N``isnumberofpredictedobjectsthatdefines
confidenceofeachprediction.4.Inputimagesizeisintegertensor
withvalues[``H``,``W``],where``H``isheightofinputdataand
``W``iswidthofinputdata,usedforrescalingpredictionson
postprocessingstep.

ForreusingDetectron2APIforpostprocessingandvisualization,we
providehelpersforwrappingoutputinoriginalDetectron2format.

..code::ipython3

fromdetectron2.structuresimportInstances,Boxes
fromdetectron2.modeling.postprocessingimportdetector_postprocess
fromdetectron2.utils.visualizerimportColorMode,Visualizer
fromdetectron2.dataimportMetadataCatalog
importnumpyasnp


defpostprocess_detection_result(outputs:Dict,orig_height:int,orig_width:int,conf_threshold:float=0.0):
"""
Helperfunctionforpostprocessingpredictionresults

Parameters:
outputs(Dict):OpenVINOmodeloutputdictionary
orig_height(int):originalimageheightbeforepreprocessing
orig_width(int):originalimagewidthbeforepreprocessing
conf_threshold(float,optional,defaults0.0):confidencethresholdforvalidprediction
Returns:
prediction_result(instances):postprocessedpredictedinstances
"""
boxes=outputs[0]
classes=outputs[1]
has_mask=len(outputs)>=5
masks=Noneifnothas_maskelseoutputs[2]
scores=outputs[2ifnothas_maskelse3]
model_input_size=(
int(outputs[3ifnothas_maskelse4][0]),
int(outputs[3ifnothas_maskelse4][1]),
)
filtered_detections=scores>=conf_threshold
boxes=Boxes(boxes[filtered_detections])
scores=scores[filtered_detections]
classes=classes[filtered_detections]
out_dict={"pred_boxes":boxes,"scores":scores,"pred_classes":classes}
ifmasksisnotNone:
masks=masks[filtered_detections]
out_dict["pred_masks"]=torch.from_numpy(masks)
instances=Instances(model_input_size,**out_dict)
returndetector_postprocess(instances,orig_height,orig_width)


defdraw_instance_prediction(img:np.ndarray,results:Instances,cfg:"Config"):
"""
Helperfunctionforvisualizationpredictionresults

Parameters:
img(np.ndarray):originalimagefordrawingpredictions
results(instances):modelpredictions
cfg(Config):modelconfiguration
Returns:
img_with_res:imagewithresults
"""
metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0])
visualizer=Visualizer(img,metadata,instance_mode=ColorMode.IMAGE)
img_with_res=visualizer.draw_instance_predictions(results)
returnimg_with_res

..code::ipython3

results=postprocess_detection_result(results,sample_input[0]["height"],sample_input[0]["width"],conf_threshold=0.05)
img_with_res=draw_instance_prediction(np.array(image),results,cfg)
Image.fromarray(img_with_res.get_image())




..image::detectron2-to-openvino-with-output_files/detectron2-to-openvino-with-output_22_0.png



InstanceSegmentation
---------------------

`backtotop⬆️<#table-of-contents>`__

Asitwasdiscussedabove,Detectron2providesgenericapproachfor
workingwithmodelsfordifferentusecases.Thestepsthatrequiredto
convertandrunmodelspretrainedforInstanceSegmentationusecase
willbeverysimilartoObjectDetection.

DownloadInstanceSegmentationPyTorchmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

model_name="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x"
model,cfg=get_model_and_config(model_name)
sample_input=get_sample_inputs(image_file,cfg)

ConvertInstanceSegmentationModeltoOpenVINOIntermediateRepresentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

model_xml_path=MODEL_DIR/(model_name.split("/")[-1]+".xml")

ifnotmodel_xml_path.exists():
ov_model=convert_detectron2_model(model,sample_input)
ov.save_model(ov_model,MODEL_DIR/(model_name.split("/")[-1]+".xml"))
else:
ov_model=model_xml_path


..parsed-literal::

['args']


Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



RunInstanceSegmentationmodelinference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

IncomparisonwithObjectDetection,InstanceSegmentationmodelshave
additionaloutputthatrepresentsinstancemasksforeachobject.Our
postprocessingfunctionhandlethisdifference.

..code::ipython3

compiled_model=core.compile_model(ov_model,device.value)

..code::ipython3

results=compiled_model(sample_input[0]["image"])
results=postprocess_detection_result(results,sample_input[0]["height"],sample_input[0]["width"],conf_threshold=0.05)
img_with_res=draw_instance_prediction(np.array(image),results,cfg)
Image.fromarray(img_with_res.get_image())




..image::detectron2-to-openvino-with-output_files/detectron2-to-openvino-with-output_32_0.png


