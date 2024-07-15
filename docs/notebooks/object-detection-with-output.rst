LiveObjectDetectionwithOpenVINO™
====================================

ThisnotebookdemonstratesliveobjectdetectionwithOpenVINO,using
the`SSDLite
MobileNetV2<https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssdlite_mobilenet_v2>`__
from`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__.Finalpart
ofthisnotebookshowsliveinferenceresultsfromawebcam.
Additionally,youcanalsouploadavideofile.

**NOTE**:Tousethisnotebookwithawebcam,youneedtorunthe
notebookonacomputerwithawebcam.Ifyourunthenotebookona
server,thewebcamwillnotwork.However,youcanstilldoinference
onavideo.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Preparation<#preparation>`__

-`Installrequirements<#install-requirements>`__
-`Imports<#imports>`__

-`TheModel<#the-model>`__

-`DownloadtheModel<#download-the-model>`__
-`ConverttheModel<#convert-the-model>`__
-`LoadtheModel<#load-the-model>`__

-`Processing<#processing>`__

-`ProcessResults<#process-results>`__
-`MainProcessingFunction<#main-processing-function>`__

-`Run<#run>`__

-`RunLiveObjectDetection<#run-live-object-detection>`__

-`References<#references>`__

Preparation
-----------

`backtotop⬆️<#table-of-contents>`__

Installrequirements
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"openvino-dev>=2024.0.0"
%pipinstall-qtensorflow
%pipinstall-qopencv-pythonrequeststqdm

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)


..parsed-literal::

ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
openvino-tokenizers2024.3.0.0.dev20240711requiresopenvino~=2024.3.0.0.dev,butyouhaveopenvino2024.2.0whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
magika0.5.1requiresnumpy<2.0,>=1.24;python_version>="3.8"andpython_version<"3.9",butyouhavenumpy1.23.5whichisincompatible.
mobileclip0.1.0requirestorch==1.13.1,butyouhavetorch2.3.1+cpuwhichisincompatible.
mobileclip0.1.0requirestorchvision==0.14.1,butyouhavetorchvision0.18.1+cpuwhichisincompatible.
openvino-tokenizers2024.3.0.0.dev20240711requiresopenvino~=2024.3.0.0.dev,butyouhaveopenvino2024.2.0whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.




..parsed-literal::

23215



Imports
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcollections
importtarfile
importtime
frompathlibimportPath

importcv2
importnumpyasnp
fromIPythonimportdisplay
importopenvinoasov
fromopenvino.tools.mo.frontimporttfasov_tf_front
fromopenvino.toolsimportmo

importnotebook_utilsasutils

TheModel
---------

`backtotop⬆️<#table-of-contents>`__

DownloadtheModel
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Usethe``download_file``,afunctionfromthe``notebook_utils``file.
Itautomaticallycreatesadirectorystructureanddownloadsthe
selectedmodel.Thisstepisskippedifthepackageisalready
downloadedandunpacked.Thechosenmodelcomesfromthepublic
directory,whichmeansitmustbeconvertedintoOpenVINOIntermediate
Representation(OpenVINOIR).

**NOTE**:Usingamodelotherthan``ssdlite_mobilenet_v2``may
requiredifferentconversionparametersaswellaspre-and
post-processing.

..code::ipython3

#Adirectorywherethemodelwillbedownloaded.
base_model_dir=Path("model")

#ThenameofthemodelfromOpenModelZoo
model_name="ssdlite_mobilenet_v2"

archive_name=Path(f"{model_name}_coco_2018_05_09.tar.gz")
model_url=f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/{model_name}/{archive_name}"

#Downloadthearchive
downloaded_model_path=base_model_dir/archive_name
ifnotdownloaded_model_path.exists():
utils.download_file(model_url,downloaded_model_path.name,downloaded_model_path.parent)

#Unpackthemodel
tf_model_path=base_model_dir/archive_name.with_suffix("").stem/"frozen_inference_graph.pb"
ifnottf_model_path.exists():
withtarfile.open(downloaded_model_path)asfile:
file.extractall(base_model_dir)



..parsed-literal::

model/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz:0%||0.00/48.7M[00:00<?,?B/s]


ConverttheModel
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thepre-trainedmodelisinTensorFlowformat.TouseitwithOpenVINO,
convertittoOpenVINOIRformat,using`ModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
(``mo.convert_model``function).Ifthemodelhasbeenalready
converted,thisstepisskipped.

..code::ipython3

precision="FP16"
#Theoutputpathfortheconversion.
converted_model_path=Path("model")/f"{model_name}_{precision.lower()}.xml"

#ConvertittoIRifnotpreviouslyconverted
trans_config_path=Path(ov_tf_front.__file__).parent/"ssd_v2_support.json"
ifnotconverted_model_path.exists():
ov_model=mo.convert_model(
tf_model_path,
compress_to_fp16=(precision=="FP16"),
transformations_config=trans_config_path,
tensorflow_object_detection_api_pipeline_config=tf_model_path.parent/"pipeline.config",
reverse_input_channels=True,
)
ov.save_model(ov_model,converted_model_path)
delov_model


..parsed-literal::

[INFO]MOcommandlinetoolisconsideredasthelegacyconversionAPIasofOpenVINO2023.2release.
In2025.0MOcommandlinetoolandopenvino.tools.mo.convert_model()willberemoved.PleaseuseOpenVINOModelConverter(OVC)oropenvino.convert_model().OVCrepresentsalightweightalternativeofMOandprovidessimplifiedmodelconversionAPI.
FindmoreinformationabouttransitionfromMOtoOVCathttps://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


..parsed-literal::

[WARNING]ThePreprocessorblockhasbeenremoved.Onlynodesperformingmeanvaluesubtractionandscaling(ifapplicable)arekept.


LoadtheModel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Onlyafewlinesofcodearerequiredtorunthemodel.First,
initializeOpenVINORuntime.Then,readthenetworkarchitectureand
modelweightsfromthe``.bin``and``.xml``filestocompileforthe
desireddevice.Ifyouchoose``GPU``youneedtowaitforawhile,as
thestartuptimeismuchlongerthaninthecaseof``CPU``.

ThereisapossibilitytoletOpenVINOdecidewhichhardwareoffersthe
bestperformance.Forthatpurpose,justuse``AUTO``.

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



..code::ipython3

#Readthenetworkandcorrespondingweightsfromafile.
model=core.read_model(model=converted_model_path)
#CompilethemodelforCPU(youcanchoosemanuallyCPU,GPUetc.)
#orlettheenginechoosethebestavailabledevice(AUTO).
compiled_model=core.compile_model(model=model,device_name=device.value)

#Gettheinputandoutputnodes.
input_layer=compiled_model.input(0)
output_layer=compiled_model.output(0)

#Gettheinputsize.
height,width=list(input_layer.shape)[1:3]

Inputandoutputlayershavethenamesoftheinputnodeandoutputnode
respectively.InthecaseofSSDLiteMobileNetV2,thereis1inputand1
output.

..code::ipython3

input_layer.any_name,output_layer.any_name




..parsed-literal::

('image_tensor:0','detection_boxes:0')



Processing
----------

`backtotop⬆️<#table-of-contents>`__

ProcessResults
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

First,listallavailableclassesandcreatecolorsforthem.Then,in
thepost-processstage,transformboxeswithnormalizedcoordinates
``[0,1]``intoboxeswithpixelcoordinates``[0,image_size_in_px]``.
Afterward,use`non-maximum
suppression<https://paperswithcode.com/method/non-maximum-suppression>`__
torejectoverlappingdetectionsandthosebelowtheprobability
threshold(0.5).Finally,drawboxesandlabelsinsidethem.

..code::ipython3

#https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
classes=[
"background",
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"trafficlight",
"firehydrant",
"streetsign",
"stopsign",
"parkingmeter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"hat",
"backpack",
"umbrella",
"shoe",
"eyeglasses",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sportsball",
"kite",
"baseballbat",
"baseballglove",
"skateboard",
"surfboard",
"tennisracket",
"bottle",
"plate",
"wineglass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hotdog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"pottedplant",
"bed",
"mirror",
"diningtable",
"window",
"desk",
"toilet",
"door",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cellphone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"blender",
"book",
"clock",
"vase",
"scissors",
"teddybear",
"hairdrier",
"toothbrush",
"hairbrush",
]

#Colorsfortheclassesabove(RainbowColorMap).
colors=cv2.applyColorMap(
src=np.arange(0,255,255/len(classes),dtype=np.float32).astype(np.uint8),
colormap=cv2.COLORMAP_RAINBOW,
).squeeze()


defprocess_results(frame,results,thresh=0.6):
#Thesizeoftheoriginalframe.
h,w=frame.shape[:2]
#The'results'variableisa[1,1,100,7]tensor.
results=results.squeeze()
boxes=[]
labels=[]
scores=[]
for_,label,score,xmin,ymin,xmax,ymaxinresults:
#Createaboxwithpixelscoordinatesfromtheboxwithnormalizedcoordinates[0,1].
boxes.append(tuple(map(int,(xmin*w,ymin*h,(xmax-xmin)*w,(ymax-ymin)*h))))
labels.append(int(label))
scores.append(float(score))

#Applynon-maximumsuppressiontogetridofmanyoverlappingentities.
#Seehttps://paperswithcode.com/method/non-maximum-suppression
#Thisalgorithmreturnsindicesofobjectstokeep.
indices=cv2.dnn.NMSBoxes(bboxes=boxes,scores=scores,score_threshold=thresh,nms_threshold=0.6)

#Iftherearenoboxes.
iflen(indices)==0:
return[]

#Filterdetectedobjects.
return[(labels[idx],scores[idx],boxes[idx])foridxinindices.flatten()]


defdraw_boxes(frame,boxes):
forlabel,score,boxinboxes:
#Choosecolorforthelabel.
color=tuple(map(int,colors[label]))
#Drawabox.
x2=box[0]+box[2]
y2=box[1]+box[3]
cv2.rectangle(img=frame,pt1=box[:2],pt2=(x2,y2),color=color,thickness=3)

#Drawalabelnameinsidethebox.
cv2.putText(
img=frame,
text=f"{classes[label]}{score:.2f}",
org=(box[0]+10,box[1]+30),
fontFace=cv2.FONT_HERSHEY_COMPLEX,
fontScale=frame.shape[1]/1000,
color=color,
thickness=1,
lineType=cv2.LINE_AA,
)

returnframe

MainProcessingFunction
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Runobjectdetectiononthespecifiedsource.Eitherawebcamoravideo
file.

..code::ipython3

#Mainprocessingfunctiontorunobjectdetection.
defrun_object_detection(source=0,flip=False,use_popup=False,skip_first_frames=0):
player=None
try:
#Createavideoplayertoplaywithtargetfps.
player=utils.VideoPlayer(source=source,flip=flip,fps=30,skip_first_frames=skip_first_frames)
#Startcapturing.
player.start()
ifuse_popup:
title="PressESCtoExit"
cv2.namedWindow(winname=title,flags=cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE)

processing_times=collections.deque()
whileTrue:
#Grabtheframe.
frame=player.next()
ifframeisNone:
print("Sourceended")
break
#IftheframeislargerthanfullHD,reducesizetoimprovetheperformance.
scale=1280/max(frame.shape)
ifscale<1:
frame=cv2.resize(
src=frame,
dsize=None,
fx=scale,
fy=scale,
interpolation=cv2.INTER_AREA,
)

#Resizetheimageandchangedimstofitneuralnetworkinput.
input_img=cv2.resize(src=frame,dsize=(width,height),interpolation=cv2.INTER_AREA)
#Createabatchofimages(size=1).
input_img=input_img[np.newaxis,...]

#Measureprocessingtime.

start_time=time.time()
#Gettheresults.
results=compiled_model([input_img])[output_layer]
stop_time=time.time()
#Getposesfromnetworkresults.
boxes=process_results(frame=frame,results=results)

#Drawboxesonaframe.
frame=draw_boxes(frame=frame,boxes=boxes)

processing_times.append(stop_time-start_time)
#Useprocessingtimesfromlast200frames.
iflen(processing_times)>200:
processing_times.popleft()

_,f_width=frame.shape[:2]
#Meanprocessingtime[ms].
processing_time=np.mean(processing_times)*1000
fps=1000/processing_time
cv2.putText(
img=frame,
text=f"Inferencetime:{processing_time:.1f}ms({fps:.1f}FPS)",
org=(20,40),
fontFace=cv2.FONT_HERSHEY_COMPLEX,
fontScale=f_width/1000,
color=(0,0,255),
thickness=1,
lineType=cv2.LINE_AA,
)

#Usethisworkaroundifthereisflickering.
ifuse_popup:
cv2.imshow(winname=title,mat=frame)
key=cv2.waitKey(1)
#escape=27
ifkey==27:
break
else:
#Encodenumpyarraytojpg.
_,encoded_img=cv2.imencode(ext=".jpg",img=frame,params=[cv2.IMWRITE_JPEG_QUALITY,100])
#CreateanIPythonimage.
i=display.Image(data=encoded_img)
#Displaytheimageinthisnotebook.
display.clear_output(wait=True)
display.display(i)
#ctrl-c
exceptKeyboardInterrupt:
print("Interrupted")
#anydifferenterror
exceptRuntimeErrorase:
print(e)
finally:
ifplayerisnotNone:
#Stopcapturing.
player.stop()
ifuse_popup:
cv2.destroyAllWindows()

Run
---

`backtotop⬆️<#table-of-contents>`__

RunLiveObjectDetection
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Useawebcamasthevideoinput.Bydefault,theprimarywebcamisset
with``source=0``.Ifyouhavemultiplewebcams,eachonewillbe
assignedaconsecutivenumberstartingat0.Set``flip=True``when
usingafront-facingcamera.Somewebbrowsers,especiallyMozilla
Firefox,maycauseflickering.Ifyouexperienceflickering,set
``use_popup=True``.

**NOTE**:Tousethisnotebookwithawebcam,youneedtorunthe
notebookonacomputerwithawebcam.Ifyourunthenotebookona
server(forexample,Binder),thewebcamwillnotwork.Popupmode
maynotworkifyourunthisnotebookonaremotecomputer(for
example,Binder).

Ifyoudonothaveawebcam,youcanstillrunthisdemowithavideo
file.Any`formatsupportedby
OpenCV<https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
willwork.

Runtheobjectdetection:

..code::ipython3

USE_WEBCAM=False

video_file="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4"
cam_id=0

source=cam_idifUSE_WEBCAMelsevideo_file

run_object_detection(source=source,flip=isinstance(source,int),use_popup=False)



..image::object-detection-with-output_files/object-detection-with-output_19_0.png


..parsed-literal::

Sourceended


References
----------

`backtotop⬆️<#table-of-contents>`__

1.`SSDLite
MobileNetV2<https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssdlite_mobilenet_v2>`__
2.`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__
3.`Non-Maximum
Suppression<https://paperswithcode.com/method/non-maximum-suppression>`__
