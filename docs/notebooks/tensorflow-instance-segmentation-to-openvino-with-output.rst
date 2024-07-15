ConvertaTensorFlowInstanceSegmentationModeltoOpenVINO™
=============================================================

`TensorFlow<https://www.tensorflow.org/>`__,orTFforshort,isan
open-sourceframeworkformachinelearning.

The`TensorFlowObjectDetection
API<https://github.com/tensorflow/models/tree/master/research/object_detection>`__
isanopen-sourcecomputervisionframeworkbuiltontopofTensorFlow.
Itisusedforbuildingobjectdetectionandinstancesegmentation
modelsthatcanlocalizemultipleobjectsinthesameimage.TensorFlow
ObjectDetectionAPIsupportsvariousarchitecturesandmodels,which
canbefoundanddownloadedfromthe`TensorFlow
Hub<https://tfhub.dev/tensorflow/collections/object_detection/1>`__.

ThistutorialshowshowtoconvertaTensorFlow`MaskR-CNNwith
InceptionResNet
V2<https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1>`__
instancesegmentationmodeltoOpenVINO`Intermediate
Representation<https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets.html>`__
(OpenVINOIR)format,using`ModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
AftercreatingtheOpenVINOIR,loadthemodelin`OpenVINO
Runtime<https://docs.openvino.ai/2024/openvino-workflow/running-inference.html>`__
anddoinferencewithasampleimage.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Imports<#imports>`__
-`Settings<#settings>`__
-`DownloadModelfromTensorFlow
Hub<#download-model-from-tensorflow-hub>`__
-`ConvertModeltoOpenVINOIR<#convert-model-to-openvino-ir>`__
-`TestInferenceontheConverted
Model<#test-inference-on-the-converted-model>`__
-`Selectinferencedevice<#select-inference-device>`__

-`LoadtheModel<#load-the-model>`__
-`GetModelInformation<#get-model-information>`__
-`GetanImageforTest
Inference<#get-an-image-for-test-inference>`__
-`PerformInference<#perform-inference>`__
-`InferenceResult
Visualization<#inference-result-visualization>`__

-`NextSteps<#next-steps>`__

-`Asyncinferencepipeline<#async-inference-pipeline>`__
-`Integrationpreprocessingto
model<#integration-preprocessing-to-model>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installrequiredpackages:

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.1.0""numpy>=1.21.0""opencv-python""tqdm"

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

%pipinstall-q"tensorflow-macos>=2.5;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version>'3.8'"#macOSM1andM2
%pipinstall-q"tensorflow-macos>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version<='3.8'"#macOSM1andM2
%pipinstall-q"tensorflow>=2.5;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version>'3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version<='3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5;sys_platform!='darwin'andpython_version>'3.8'"
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform!='darwin'andpython_version<='3.8'"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Thenotebookusesutilityfunctions.Thecellbelowwilldownloadthe
``notebook_utils``PythonmodulefromGitHub.

..code::ipython3

#Fetchthenotebookutilsscriptfromtheopenvino_notebooksrepo
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)




..parsed-literal::

23215



Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Standardpythonmodules
frompathlibimportPath

#Externalmodulesanddependencies
importcv2
importmatplotlib.pyplotasplt
importnumpyasnp

#Notebookutilsmodule
fromnotebook_utilsimportdownload_file

#OpenVINOmodules
importopenvinoasov

Settings
--------

`backtotop⬆️<#table-of-contents>`__

Definemodelrelatedvariablesandcreatecorrespondingdirectories:

..code::ipython3

#Createdirectoriesformodelsfiles
model_dir=Path("model")
model_dir.mkdir(exist_ok=True)

#CreatedirectoryforTensorFlowmodel
tf_model_dir=model_dir/"tf"
tf_model_dir.mkdir(exist_ok=True)

#CreatedirectoryforOpenVINOIRmodel
ir_model_dir=model_dir/"ir"
ir_model_dir.mkdir(exist_ok=True)

model_name="mask_rcnn_inception_resnet_v2_1024x1024"

openvino_ir_path=ir_model_dir/f"{model_name}.xml"

tf_model_url=(
"https://www.kaggle.com/models/tensorflow/mask-rcnn-inception-resnet-v2/frameworks/tensorFlow2/variations/1024x1024/versions/1?tf-hub-format=compressed"
)

tf_model_archive_filename=f"{model_name}.tar.gz"

DownloadModelfromTensorFlowHub
----------------------------------

`backtotop⬆️<#table-of-contents>`__

DownloadarchivewithTensorFlowInstanceSegmentationmodel
(`mask_rcnn_inception_resnet_v2_1024x1024<https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1>`__)
fromTensorFlowHub:

..code::ipython3

download_file(url=tf_model_url,filename=tf_model_archive_filename,directory=tf_model_dir);



..parsed-literal::

model/tf/mask_rcnn_inception_resnet_v2_1024x1024.tar.gz:0%||0.00/232M[00:00<?,?B/s]


ExtractTensorFlowInstanceSegmentationmodelfromthedownloaded
archive:

..code::ipython3

importtarfile

withtarfile.open(tf_model_dir/tf_model_archive_filename)asfile:
file.extractall(path=tf_model_dir)

ConvertModeltoOpenVINOIR
----------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOModelOptimizerPythonAPIcanbeusedtoconvertthe
TensorFlowmodeltoOpenVINOIR.

``mo.convert_model``functionacceptpathtoTensorFlowmodeland
returnsOpenVINOModelclassinstancewhichrepresentsthismodel.Also
weneedtoprovidemodelinputshape(``input_shape``)thatisdescribed
at`modeloverviewpageonTensorFlow
Hub<https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1>`__.
Optionally,wecanapplycompressiontoFP16modelweightsusing
``compress_to_fp16=True``optionandintegratepreprocessingusingthis
approach.

Theconvertedmodelisreadytoloadonadeviceusing``compile_model``
orsavedondiskusingthe``serialize``functiontoreduceloadingtime
whenthemodelisruninthefuture.

..code::ipython3

ov_model=ov.convert_model(tf_model_dir)

#SaveconvertedOpenVINOIRmodeltothecorrespondingdirectory
ov.save_model(ov_model,openvino_ir_path)

TestInferenceontheConvertedModel
-------------------------------------

`backtotop⬆️<#table-of-contents>`__

Selectinferencedevice
-----------------------

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



LoadtheModel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

openvino_ir_model=core.read_model(openvino_ir_path)
compiled_model=core.compile_model(model=openvino_ir_model,device_name=device.value)

GetModelInformation
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

MaskR-CNNwithInceptionResNetV2instancesegmentationmodelhasone
input-athree-channelimageofvariablesize.Theinputtensorshape
is``[1,height,width,3]``withvaluesin``[0,255]``.

Modeloutputdictionarycontainsalotoftensors,wewilluseonly5of
them:-``num_detections``:A``tf.int``tensorwithonlyonevalue,the
numberofdetections``[N]``.-``detection_boxes``:A``tf.float32``
tensorofshape``[N,4]``containingboundingboxcoordinatesinthe
followingorder:``[ymin,xmin,ymax,xmax]``.-``detection_classes``:
A``tf.int``tensorofshape``[N]``containingdetectionclassindex
fromthelabelfile.-``detection_scores``:A``tf.float32``tensorof
shape``[N]``containingdetectionscores.-``detection_masks``:A
``[batch,max_detections,mask_height,mask_width]``tensor.Notethata
pixel-wisesigmoidscoreconverterisappliedtothedetectionmasks.

Formoreinformationaboutmodelinputs,outputsandtheirformats,see
the`modeloverviewpageonTensorFlow
Hub<https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1>`__.

Itisimportanttomention,thatvaluesof``detection_boxes``,
``detection_classes``,``detection_scores``,``detection_masks``
correspondtoeachotherandareorderedbythehighestdetectionscore:
thefirstdetectionmaskcorrespondstothefirstdetectionclassandto
thefirst(andhighest)detectionscore.

..code::ipython3

model_inputs=compiled_model.inputs
model_outputs=compiled_model.outputs

print("Modelinputscount:",len(model_inputs))
print("Modelinputs:")
for_inputinmodel_inputs:
print("",_input)

print("Modeloutputscount:",len(model_outputs))
print("Modeloutputs:")
foroutputinmodel_outputs:
print("",output)


..parsed-literal::

Modelinputscount:1
Modelinputs:
<ConstOutput:names[input_tensor]shape[1,?,?,3]type:u8>
Modeloutputscount:23
Modeloutputs:
<ConstOutput:names[]shape[49152,4]type:f32>
<ConstOutput:names[box_classifier_features]shape[300,9,9,1536]type:f32>
<ConstOutput:names[]shape[4]type:f32>
<ConstOutput:names[mask_predictions]shape[100,90,33,33]type:f32>
<ConstOutput:names[num_detections]shape[1]type:f32>
<ConstOutput:names[num_proposals]shape[1]type:f32>
<ConstOutput:names[proposal_boxes]shape[1,?,..8]type:f32>
<ConstOutput:names[proposal_boxes_normalized,final_anchors]shape[1,?,..8]type:f32>
<ConstOutput:names[raw_detection_boxes]shape[1,300,4]type:f32>
<ConstOutput:names[raw_detection_scores]shape[1,300,91]type:f32>
<ConstOutput:names[refined_box_encodings]shape[300,90,4]type:f32>
<ConstOutput:names[rpn_box_encodings]shape[1,49152,4]type:f32>
<ConstOutput:names[class_predictions_with_background]shape[300,91]type:f32>
<ConstOutput:names[rpn_box_predictor_features]shape[1,64,64,512]type:f32>
<ConstOutput:names[rpn_features_to_crop]shape[1,64,64,1088]type:f32>
<ConstOutput:names[rpn_objectness_predictions_with_background]shape[1,49152,2]type:f32>
<ConstOutput:names[detection_anchor_indices]shape[1,?]type:f32>
<ConstOutput:names[detection_boxes]shape[1,?,..8]type:f32>
<ConstOutput:names[detection_classes]shape[1,?]type:f32>
<ConstOutput:names[detection_masks]shape[1,100,33,33]type:f32>
<ConstOutput:names[detection_multiclass_scores]shape[1,?,..182]type:f32>
<ConstOutput:names[detection_scores]shape[1,?]type:f32>
<ConstOutput:names[proposal_boxes_normalized,final_anchors]shape[1,?,..8]type:f32>


GetanImageforTestInference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Loadandsaveanimage:

..code::ipython3

image_path=Path("./data/coco_bike.jpg")

download_file(
url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg",
filename=image_path.name,
directory=image_path.parent,
);



..parsed-literal::

data/coco_bike.jpg:0%||0.00/182k[00:00<?,?B/s]


Readtheimage,resizeandconvertittotheinputshapeofthenetwork:

..code::ipython3

#Readtheimage
image=cv2.imread(filename=str(image_path))

#ThenetworkexpectsimagesinRGBformat
image=cv2.cvtColor(image,code=cv2.COLOR_BGR2RGB)

#Resizetheimagetothenetworkinputshape
resized_image=cv2.resize(src=image,dsize=(255,255))

#Addbatchdimensiontoimage
network_input_image=np.expand_dims(resized_image,0)

#Showtheimage
plt.imshow(image)




..parsed-literal::

<matplotlib.image.AxesImageat0x7f9df57e55b0>




..image::tensorflow-instance-segmentation-to-openvino-with-output_files/tensorflow-instance-segmentation-to-openvino-with-output_25_1.png


PerformInference
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

inference_result=compiled_model(network_input_image)

Aftermodelinferenceonthetestimage,instancesegmentationdatacan
beextractedfromtheresult.Forfurthermodelresultvisualization
``detection_boxes``,``detection_masks``,``detection_classes``and
``detection_scores``outputswillbeused.

..code::ipython3

detection_boxes=compiled_model.output("detection_boxes")
image_detection_boxes=inference_result[detection_boxes]
print("image_detection_boxes:",image_detection_boxes.shape)

detection_masks=compiled_model.output("detection_masks")
image_detection_masks=inference_result[detection_masks]
print("image_detection_masks:",image_detection_masks.shape)

detection_classes=compiled_model.output("detection_classes")
image_detection_classes=inference_result[detection_classes]
print("image_detection_classes:",image_detection_classes.shape)

detection_scores=compiled_model.output("detection_scores")
image_detection_scores=inference_result[detection_scores]
print("image_detection_scores:",image_detection_scores.shape)

num_detections=compiled_model.output("num_detections")
image_num_detections=inference_result[num_detections]
print("image_detections_num:",image_num_detections)

#Alternatively,inferenceresultdatacanbeextractedbymodeloutputnamewith`.get()`method
assert(inference_result[detection_boxes]==inference_result.get("detection_boxes")).all(),"extractedinferenceresultdatashouldbeequal"


..parsed-literal::

image_detection_boxes:(1,100,4)
image_detection_masks:(1,100,33,33)
image_detection_classes:(1,100)
image_detection_scores:(1,100)
image_detections_num:[100.]


InferenceResultVisualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Defineutilityfunctionstovisualizetheinferenceresults

..code::ipython3

importrandom
fromtypingimportOptional


defadd_detection_box(box:np.ndarray,image:np.ndarray,mask:np.ndarray,label:Optional[str]=None)->np.ndarray:
"""
Helperfunctionforaddingsingleboundingboxtotheimage

Parameters
----------
box:np.ndarray
Boundingboxcoordinatesinformat[ymin,xmin,ymax,xmax]
image:np.ndarray
Theimagetowhichdetectionboxisadded
mask:np.ndarray
Segmentationmaskinformat(H,W)
label:str,optional
Detectionboxlabelstring,ifnotprovidedwillnotbeaddedtoresultimage(defaultisNone)

Returns
-------
np.ndarray
NumPyarrayincludingimage,detectionbox,andsegmentationmask

"""
ymin,xmin,ymax,xmax=box
point1,point2=(int(xmin),int(ymin)),(int(xmax),int(ymax))
box_color=[random.randint(0,255)for_inrange(3)]
line_thickness=round(0.002*(image.shape[0]+image.shape[1])/2)+1

result=cv2.rectangle(
img=image,
pt1=point1,
pt2=point2,
color=box_color,
thickness=line_thickness,
lineType=cv2.LINE_AA,
)

iflabel:
font_thickness=max(line_thickness-1,1)
font_face=0
font_scale=line_thickness/3
font_color=(255,255,255)
text_size=cv2.getTextSize(
text=label,
fontFace=font_face,
fontScale=font_scale,
thickness=font_thickness,
)[0]
#Calculaterectanglecoordinates
rectangle_point1=point1
rectangle_point2=(point1[0]+text_size[0],point1[1]-text_size[1]-3)
#Addfilledrectangle
result=cv2.rectangle(
img=result,
pt1=rectangle_point1,
pt2=rectangle_point2,
color=box_color,
thickness=-1,
lineType=cv2.LINE_AA,
)
#Calculatetextposition
text_position=point1[0],point1[1]-3
#Addtextwithlabeltofilledrectangle
result=cv2.putText(
img=result,
text=label,
org=text_position,
fontFace=font_face,
fontScale=font_scale,
color=font_color,
thickness=font_thickness,
lineType=cv2.LINE_AA,
)
mask_img=mask[:,:,np.newaxis]*box_color
result=cv2.addWeighted(result,1,mask_img.astype(np.uint8),0.6,0)
returnresult

..code::ipython3

defget_mask_frame(box,frame,mask):
"""
Transformabinarymasktofitwithinaspecifiedboundingboxinaframeusingperspectivetransformation.

Args:
box(tuple):Aboundingboxrepresentedasatuple(y_min,x_min,y_max,x_max).
frame(numpy.ndarray):Thelargerframeorimagewherethemaskwillbeplaced.
mask(numpy.ndarray):Abinarymaskimagetobetransformed.

Returns:
numpy.ndarray:Atransformedmaskimagethatfitswithinthespecifiedboundingboxintheframe.
"""
x_min=frame.shape[1]*box[1]
y_min=frame.shape[0]*box[0]
x_max=frame.shape[1]*box[3]
y_max=frame.shape[0]*box[2]
rect_src=np.array(
[
[0,0],
[mask.shape[1],0],
[mask.shape[1],mask.shape[0]],
[0,mask.shape[0]],
],
dtype=np.float32,
)
rect_dst=np.array(
[[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]],
dtype=np.float32,
)
M=cv2.getPerspectiveTransform(rect_src[:,:],rect_dst[:,:])
mask_frame=cv2.warpPerspective(mask,M,(frame.shape[1],frame.shape[0]),flags=cv2.INTER_CUBIC)
returnmask_frame

..code::ipython3

fromtypingimportDict

fromopenvino.runtime.utils.data_helpersimportOVDict


defvisualize_inference_result(
inference_result:OVDict,
image:np.ndarray,
labels_map:Dict,
detections_limit:Optional[int]=None,
):
"""
Helperfunctionforvisualizinginferenceresultontheimage

Parameters
----------
inference_result:OVDict
Resultofthecompiledmodelinferenceonthetestimage
image:np.ndarray
Originalimagetouseforvisualization
labels_map:Dict
Dictionarywithmappingsofdetectionclassesnumbersanditsnames
detections_limit:int,optional
Numberofdetectionstoshowontheimage,ifnotprovidedalldetectionswillbeshown(defaultisNone)
"""
detection_boxes=inference_result.get("detection_boxes")
detection_classes=inference_result.get("detection_classes")
detection_scores=inference_result.get("detection_scores")
num_detections=inference_result.get("num_detections")
detection_masks=inference_result.get("detection_masks")

detections_limit=int(min(detections_limit,num_detections[0])ifdetections_limitisnotNoneelsenum_detections[0])

#Normalizedetectionboxescoordinatestooriginalimagesize
original_image_height,original_image_width,_=image.shape
normalized_detection_boxes=detection_boxes[0,:detections_limit]*[
original_image_height,
original_image_width,
original_image_height,
original_image_width,
]
result=np.copy(image)
foriinrange(detections_limit):
detected_class_name=labels_map[int(detection_classes[0,i])]
score=detection_scores[0,i]
mask=detection_masks[0,i]
mask_reframed=get_mask_frame(detection_boxes[0,i],image,mask)
mask_reframed=(mask_reframed>0.5).astype(np.uint8)
label=f"{detected_class_name}{score:.2f}"
result=add_detection_box(
box=normalized_detection_boxes[i],
image=result,
mask=mask_reframed,
label=label,
)

plt.imshow(result)

TensorFlowInstanceSegmentationmodel
(`mask_rcnn_inception_resnet_v2_1024x1024<https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1?tf-hub-format=compressed>`__)
usedinthisnotebookwastrainedon`COCO
2017<https://cocodataset.org/>`__datasetwith91classes.Forbetter
visualizationexperiencewecanuseCOCOdatasetlabelswithhuman
readableclassnamesinsteadofclassnumbersorindexes.

WecandownloadCOCOdatasetclasseslabelsfrom`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__:

..code::ipython3

coco_labels_file_path=Path("./data/coco_91cl.txt")

download_file(
url="https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/coco_91cl.txt",
filename=coco_labels_file_path.name,
directory=coco_labels_file_path.parent,
);



..parsed-literal::

data/coco_91cl.txt:0%||0.00/421[00:00<?,?B/s]


Thenweneedtocreatedictionary``coco_labels_map``withmappings
betweendetectionclassesnumbersanditsnamesfromthedownloaded
file:

..code::ipython3

withopen(coco_labels_file_path,"r")asfile:
coco_labels=file.read().strip().split("\n")
coco_labels_map=dict(enumerate(coco_labels,1))

print(coco_labels_map)


..parsed-literal::

{1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplan',6:'bus',7:'train',8:'truck',9:'boat',10:'trafficlight',11:'firehydrant',12:'streetsign',13:'stopsign',14:'parkingmeter',15:'bench',16:'bird',17:'cat',18:'dog',19:'horse',20:'sheep',21:'cow',22:'elephant',23:'bear',24:'zebra',25:'giraffe',26:'hat',27:'backpack',28:'umbrella',29:'shoe',30:'eyeglasses',31:'handbag',32:'tie',33:'suitcase',34:'frisbee',35:'skis',36:'snowboard',37:'sportsball',38:'kite',39:'baseballbat',40:'baseballglove',41:'skateboard',42:'surfboard',43:'tennisracket',44:'bottle',45:'plate',46:'wineglass',47:'cup',48:'fork',49:'knife',50:'spoon',51:'bowl',52:'banana',53:'apple',54:'sandwich',55:'orange',56:'broccoli',57:'carrot',58:'hotdog',59:'pizza',60:'donut',61:'cake',62:'chair',63:'couch',64:'pottedplant',65:'bed',66:'mirror',67:'diningtable',68:'window',69:'desk',70:'toilet',71:'door',72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',77:'cellphone',78:'microwave',79:'oven',80:'toaster',81:'sink',82:'refrigerator',83:'blender',84:'book',85:'clock',86:'vase',87:'scissors',88:'teddybear',89:'hairdrier',90:'toothbrush',91:'hairbrush'}


Finally,wearereadytovisualizemodelinferenceresultsonthe
originaltestimage:

..code::ipython3

visualize_inference_result(
inference_result=inference_result,
image=image,
labels_map=coco_labels_map,
detections_limit=5,
)



..image::tensorflow-instance-segmentation-to-openvino-with-output_files/tensorflow-instance-segmentation-to-openvino-with-output_39_0.png


NextSteps
----------

`backtotop⬆️<#table-of-contents>`__

Thissectioncontainssuggestionsonhowtoadditionallyimprovethe
performanceofyourapplicationusingOpenVINO.

Asyncinferencepipeline
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__ThekeyadvantageoftheAsync
APIisthatwhenadeviceisbusywithinference,theapplicationcan
performothertasksinparallel(forexample,populatinginputsor
schedulingotherrequests)ratherthanwaitforthecurrentinferenceto
completefirst.Tounderstandhowtoperformasyncinferenceusing
openvino,refertothe`AsyncAPI
tutorial<async-api-with-output.html>`__.

Integrationpreprocessingtomodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

PreprocessingAPIenablesmakingpreprocessingapartofthemodel
reducingapplicationcodeanddependencyonadditionalimageprocessing
libraries.ThemainadvantageofPreprocessingAPIisthatpreprocessing
stepswillbeintegratedintotheexecutiongraphandwillbeperformed
onaselecteddevice(CPU/GPUetc.)ratherthanalwaysbeingexecutedon
CPUaspartofanapplication.Thiswillimproveselecteddevice
utilization.

Formoreinformation,refertothe`OptimizePreprocessing
tutorial<optimize-preprocessing-with-output.html>`__and
totheoverviewof`Preprocessing
API<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details.html>`__.
