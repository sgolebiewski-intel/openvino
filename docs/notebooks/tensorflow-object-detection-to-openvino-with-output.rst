ConvertaTensorFlowObjectDetectionModeltoOpenVINO™
========================================================

`TensorFlow<https://www.tensorflow.org/>`__,orTFforshort,isan
open-sourceframeworkformachinelearning.

The`TensorFlowObjectDetection
API<https://github.com/tensorflow/models/tree/master/research/object_detection>`__
isanopen-sourcecomputervisionframeworkbuiltontopofTensorFlow.
Itisusedforbuildingobjectdetectionandimagesegmentationmodels
thatcanlocalizemultipleobjectsinthesameimage.TensorFlowObject
DetectionAPIsupportsvariousarchitecturesandmodels,whichcanbe
foundanddownloadedfromthe`TensorFlow
Hub<https://tfhub.dev/tensorflow/collections/object_detection/1>`__.

ThistutorialshowshowtoconvertaTensorFlow`FasterR-CNNwith
Resnet-50
V1<https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__
objectdetectionmodeltoOpenVINO`Intermediate
Representation<https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets.html>`__
(OpenVINOIR)format,usingModelConverter.AftercreatingtheOpenVINO
IR,loadthemodelin`OpenVINO
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

#OpenVINOimport
importopenvinoasov

#Notebookutilsmodule
fromnotebook_utilsimportdownload_file

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

model_name="faster_rcnn_resnet50_v1_640x640"

openvino_ir_path=ir_model_dir/f"{model_name}.xml"

tf_model_url="https://www.kaggle.com/models/tensorflow/faster-rcnn-resnet-v1/frameworks/tensorFlow2/variations/faster-rcnn-resnet50-v1-640x640/versions/1?tf-hub-format=compressed"

tf_model_archive_filename=f"{model_name}.tar.gz"

DownloadModelfromTensorFlowHub
----------------------------------

`backtotop⬆️<#table-of-contents>`__

DownloadarchivewithTensorFlowObjectDetectionmodel
(`faster_rcnn_resnet50_v1_640x640<https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__)
fromTensorFlowHub:

..code::ipython3

download_file(url=tf_model_url,filename=tf_model_archive_filename,directory=tf_model_dir)



..parsed-literal::

model/tf/faster_rcnn_resnet50_v1_640x640.tar.gz:0%||0.00/101M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/tensorflow-object-detection-to-openvino/model/tf/faster_rcnn_resnet50_v1_640x640.tar.gz')



ExtractTensorFlowObjectDetectionmodelfromthedownloadedarchive:

..code::ipython3

importtarfile

withtarfile.open(tf_model_dir/tf_model_archive_filename)asfile:
file.extractall(path=tf_model_dir)

ConvertModeltoOpenVINOIR
----------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOModelConversionAPIcanbeusedtoconverttheTensorFlow
modeltoOpenVINOIR.

``ov.convert_model``functionacceptpathtoTensorFlowmodeland
returnsOpenVINOModelclassinstancewhichrepresentsthismodel.Also
weneedtoprovidemodelinputshape(``input_shape``)thatisdescribed
at`modeloverviewpageonTensorFlow
Hub<https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__.

Theconvertedmodelisreadytoloadonadeviceusing``compile_model``
orsavedondiskusingthe``save_model``functiontoreduceloading
timewhenthemodelisruninthefuture.

Seethe`ModelPreparation
Guide<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
formoreinformationaboutmodelconversionandTensorFlow`models
support<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-tensorflow.html>`__.

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

core=ov.Core()
openvino_ir_model=core.read_model(openvino_ir_path)
compiled_model=core.compile_model(model=openvino_ir_model,device_name=device.value)

GetModelInformation
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

FasterR-CNNwithResnet-50V1objectdetectionmodelhasoneinput-a
three-channelimageofvariablesize.Theinputtensorshapeis
``[1,height,width,3]``withvaluesin``[0,255]``.

Modeloutputdictionarycontainsseveraltensors:

-``num_detections``-thenumberofdetectionsin``[N]``format.
-``detection_boxes``-boundingboxcoordinatesforall``N``
detectionsin``[ymin,xmin,ymax,xmax]``format.
-``detection_classes``-``N``detectionclassindexessizefromthe
labelfile.
-``detection_scores``-``N``detectionscores(confidence)foreach
detectedclass.
-``raw_detection_boxes``-decodeddetectionboxeswithoutNon-Max
suppression.
-``raw_detection_scores``-classscorelogitsforrawdetection
boxes.
-``detection_anchor_indices``-theanchorindicesofthedetections
afterNMS.
-``detection_multiclass_scores``-classscoredistribution(including
background)fordetectionboxesintheimageincludingbackground
class.

Inthistutorialwewillmostlyuse``detection_boxes``,
``detection_classes``,``detection_scores``tensors.Itisimportantto
mention,thatvaluesofthesetensorscorrespondtoeachotherandare
orderedbythehighestdetectionscore:thefirstdetectionbox
correspondstothefirstdetectionclassandtothefirst(andhighest)
detectionscore.

Seethe`modeloverviewpageonTensorFlow
Hub<https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__
formoreinformationaboutmodelinputs,outputsandtheirformats.

..code::ipython3

model_inputs=compiled_model.inputs
model_input=compiled_model.input(0)
model_outputs=compiled_model.outputs

print("Modelinputscount:",len(model_inputs))
print("Modelinput:",model_input)

print("Modeloutputscount:",len(model_outputs))
print("Modeloutputs:")
foroutputinmodel_outputs:
print("",output)


..parsed-literal::

Modelinputscount:1
Modelinput:<ConstOutput:names[input_tensor]shape[1,?,?,3]type:u8>
Modeloutputscount:8
Modeloutputs:
<ConstOutput:names[detection_anchor_indices]shape[1,?]type:f32>
<ConstOutput:names[detection_boxes]shape[1,?,..8]type:f32>
<ConstOutput:names[detection_classes]shape[1,?]type:f32>
<ConstOutput:names[detection_multiclass_scores]shape[1,?,..182]type:f32>
<ConstOutput:names[detection_scores]shape[1,?]type:f32>
<ConstOutput:names[num_detections]shape[1]type:f32>
<ConstOutput:names[raw_detection_boxes]shape[1,300,4]type:f32>
<ConstOutput:names[raw_detection_scores]shape[1,300,91]type:f32>


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
)


..parsed-literal::

'data/coco_bike.jpg'alreadyexists.




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/tensorflow-object-detection-to-openvino/data/coco_bike.jpg')



Readtheimage,resizeandconvertittotheinputshapeofthenetwork:

..code::ipython3

#Readtheimage
image=cv2.imread(filename=str(image_path))

#ThenetworkexpectsimagesinRGBformat
image=cv2.cvtColor(image,code=cv2.COLOR_BGR2RGB)

#Resizetheimagetothenetworkinputshape
resized_image=cv2.resize(src=image,dsize=(255,255))

#Transposetheimagetothenetworkinputshape
network_input_image=np.expand_dims(resized_image,0)

#Showtheimage
plt.imshow(image)




..parsed-literal::

<matplotlib.image.AxesImageat0x7f9fdceb46d0>




..image::tensorflow-object-detection-to-openvino-with-output_files/tensorflow-object-detection-to-openvino-with-output_25_1.png


PerformInference
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

inference_result=compiled_model(network_input_image)

Aftermodelinferenceonthetestimage,objectdetectiondatacanbe
extractedfromtheresult.Forfurthermodelresultvisualization
``detection_boxes``,``detection_classes``and``detection_scores``
outputswillbeused.

..code::ipython3

(
_,
detection_boxes,
detection_classes,
_,
detection_scores,
num_detections,
_,
_,
)=model_outputs

image_detection_boxes=inference_result[detection_boxes]
print("image_detection_boxes:",image_detection_boxes)

image_detection_classes=inference_result[detection_classes]
print("image_detection_classes:",image_detection_classes)

image_detection_scores=inference_result[detection_scores]
print("image_detection_scores:",image_detection_scores)

image_num_detections=inference_result[num_detections]
print("image_detections_num:",image_num_detections)

#Alternatively,inferenceresultdatacanbeextractedbymodeloutputnamewith`.get()`method
assert(inference_result[detection_boxes]==inference_result.get("detection_boxes")).all(),"extractedinferenceresultdatashouldbeequal"


..parsed-literal::

image_detection_boxes:[[[0.164478330.54603260.895371440.8550827]
[0.67176810.012388520.98432840.53113335]
[0.492026330.011727620.980521860.8866133]
...
[0.460214470.59246250.487344030.6187243]
[0.43605050.59333980.46925260.6341007]
[0.689981760.41356690.97601980.8143897]]]
image_detection_classes:[[18.2.2.3.2.8.2.2.3.2.4.4.2.4.16.1.1.2.
27.8.62.2.2.4.4.2.18.41.4.4.2.18.2.2.4.2.
27.2.27.2.1.2.16.1.16.2.2.2.2.16.2.2.4.2.
1.33.4.15.3.2.2.1.2.1.4.2.11.3.4.35.4.1.
40.2.62.2.4.4.36.1.36.36.77.31.2.1.51.1.34.3.
90.3.2.2.1.2.2.1.1.1.2.18.4.3.2.2.31.1.
2.1.2.41.33.41.31.3.3.1.36.15.27.4.27.2.4.15.
3.37.1.27.4.35.36.88.4.2.3.15.2.4.2.1.3.27.
4.3.4.16.23.44.1.1.4.1.4.3.15.4.62.36.77.3.
28.1.27.35.2.36.28.27.75.8.3.36.4.44.2.4.35.1.
3.1.1.35.87.1.1.1.15.1.84.1.3.1.1.35.1.2.
1.1.15.62.1.15.44.1.41.1.62.4.35.4.43.3.16.15.
2.4.34.14.3.62.33.41.4.2.35.18.3.15.1.27.4.21.
19.87.1.1.27.1.3.2.3.15.38.1.27.1.15.84.4.4.
3.38.1.15.20.3.62.41.20.58.2.88.4.62.1.15.14.31.
19.4.31.1.2.8.18.15.4.2.2.2.31.84.15.3.18.2.
27.28.15.31.28.1.1.8.20.3.1.41.]]
image_detection_scores:[[0.981009360.940719370.9320540.877722740.840291740.5898775
0.55335830.53980710.493832020.477971970.462484570.44053423
0.401562180.347090660.317498180.274423150.24709810.23665425
0.232172890.223824830.219703940.202136110.194056380.14689012
0.145076110.143437950.127800050.125643480.118098910.10874528
0.104620280.092826810.090718240.089068530.086742420.08082759
0.080100860.0793680.066176830.06282780.060662680.0602232
0.05805670.0536020.051803560.049882550.0485320.04689693
0.044763410.041343170.04080880.039690540.035042780.03275277
0.031099650.029650530.028629010.028582750.02579680.02342912
0.023335450.021425820.021373990.020886130.020248640.01939381
0.01936740.019340380.018638450.018478590.018446650.01834509
0.018030450.017816850.01730030.016670610.015857640.01565674
0.015656290.015248170.015163750.015052810.014359650.01434395
0.014158880.013698950.013591020.01298660.012531290.0120007
0.011567550.011492710.011350320.011331450.011136210.01108707
0.011003620.010908550.010449540.010284270.010012380.00976972
0.009762330.009644470.009605190.009540920.00948810.00940329
0.009350680.009331210.009068780.008875970.00884250.00881775
0.008604510.008546380.00849260.008480490.008454590.00824691
0.008147310.007894080.007853610.007739620.007707730.00766053
0.007656530.007653380.007445460.007040720.006979010.00689811
0.006890550.006597240.006491990.00637550.006355640.00623979
0.006221210.005997850.00588570.005856960.005799750.0057361
0.005725490.00562050.005580060.005567080.005495310.00547659
0.005476340.005469180.005418630.005403050.005355390.00534114
0.005242520.005224220.005058570.00505410.004904340.00482884
0.004790490.004702870.004611440.00460540.004604640.00457361
0.004555930.004551550.004541440.00446960.004372950.00425156
0.004215440.004152560.00410010.004079840.00406960.00404598
0.004032540.003995330.003961390.003933930.003915810.00389289
0.003834190.003832540.003818910.003767520.00375260.00373114
0.00370090.003670860.00366020.003592890.003519310.00350436
0.003483570.003450030.003434770.003433640.003364490.00332134
0.003314930.003295960.00327740.003125070.003119550.00307898
0.003078350.003074190.003063890.00304640.003021920.003013
0.002997570.002972210.002924180.002898390.002897290.00289356
0.002879510.002818610.002809290.002756720.00272630.00269611
0.002672230.002631090.002602420.002564640.00255610.00251843
0.002509940.002502750.002482120.0024740.00246590.00242074
0.002391780.002375580.00237480.002354670.002347260.00234068
0.002323150.002320860.002315380.002307530.002294960.00229319
0.002269350.002239110.002219970.002208660.002199450.00219268
0.002180710.002162850.002158590.002154830.00213130.00211466
0.002106610.002048440.002040420.002040040.002023830.00202068
0.001992530.001988490.001987650.001981620.001976270.00195188
0.001932990.001918650.001902850.001881110.001852290.00182701
0.001788740.001773560.001766280.001760790.00175370.00174401
0.001715740.001695060.001683470.001680530.001671590.00167045
0.001635590.001633020.001630380.001628860.001628660.00162236]]
image_detections_num:[300.]


InferenceResultVisualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Defineutilityfunctionstovisualizetheinferenceresults

..code::ipython3

importrandom
fromtypingimportOptional


defadd_detection_box(box:np.ndarray,image:np.ndarray,label:Optional[str]=None)->np.ndarray:
"""
Helperfunctionforaddingsingleboundingboxtotheimage

Parameters
----------
box:np.ndarray
Boundingboxcoordinatesinformat[ymin,xmin,ymax,xmax]
image:np.ndarray
Theimagetowhichdetectionboxisadded
label:str,optional
Detectionboxlabelstring,ifnotprovidedwillnotbeaddedtoresultimage(defaultisNone)

Returns
-------
np.ndarray
NumPyarrayincludingbothimageanddetectionbox

"""
ymin,xmin,ymax,xmax=box
point1,point2=(int(xmin),int(ymin)),(int(xmax),int(ymax))
box_color=[random.randint(0,255)for_inrange(3)]
line_thickness=round(0.002*(image.shape[0]+image.shape[1])/2)+1

cv2.rectangle(
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
cv2.rectangle(
img=image,
pt1=rectangle_point1,
pt2=rectangle_point2,
color=box_color,
thickness=-1,
lineType=cv2.LINE_AA,
)
#Calculatetextposition
text_position=point1[0],point1[1]-3
#Addtextwithlabeltofilledrectangle
cv2.putText(
img=image,
text=label,
org=text_position,
fontFace=font_face,
fontScale=font_scale,
color=font_color,
thickness=font_thickness,
lineType=cv2.LINE_AA,
)
returnimage

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
detection_boxes:np.ndarray=inference_result.get("detection_boxes")
detection_classes:np.ndarray=inference_result.get("detection_classes")
detection_scores:np.ndarray=inference_result.get("detection_scores")
num_detections:np.ndarray=inference_result.get("num_detections")

detections_limit=int(min(detections_limit,num_detections[0])ifdetections_limitisnotNoneelsenum_detections[0])

#Normalizedetectionboxescoordinatestooriginalimagesize
original_image_height,original_image_width,_=image.shape
normalized_detection_boxex=detection_boxes[::]*[
original_image_height,
original_image_width,
original_image_height,
original_image_width,
]

image_with_detection_boxex=np.copy(image)

foriinrange(detections_limit):
detected_class_name=labels_map[int(detection_classes[0,i])]
score=detection_scores[0,i]
label=f"{detected_class_name}{score:.2f}"
add_detection_box(
box=normalized_detection_boxex[0,i],
image=image_with_detection_boxex,
label=label,
)

plt.imshow(image_with_detection_boxex)

TensorFlowObjectDetectionmodel
(`faster_rcnn_resnet50_v1_640x640<https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__)
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
)



..parsed-literal::

data/coco_91cl.txt:0%||0.00/421[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/tensorflow-object-detection-to-openvino/data/coco_91cl.txt')



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



..image::tensorflow-object-detection-to-openvino-with-output_files/tensorflow-object-detection-to-openvino-with-output_38_0.png


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
