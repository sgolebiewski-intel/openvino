PersonTrackingwithOpenVINO™
==============================

ThisnotebookdemonstrateslivepersontrackingwithOpenVINO:itreads
framesfromaninputvideosequence,detectspeopleintheframes,
uniquelyidentifieseachoneofthemandtracksallofthemuntilthey
leavetheframe.Wewillusethe`Deep
SORT<https://arxiv.org/abs/1703.07402>`__algorithmtoperformobject
tracking,anextensiontoSORT(SimpleOnlineandRealtimeTracking).

DetectionvsTracking
---------------------

-Inobjectdetection,wedetectanobjectinaframe,putabounding
boxoramaskaroundit,andclassifytheobject.Notethat,thejob
ofthedetectorendshere.Itprocesseseachframeindependentlyand
identifiesnumerousobjectsinthatparticularframe.
-Anobjecttrackerontheotherhandneedstotrackaparticular
objectacrosstheentirevideo.Ifthedetectordetectsthreecarsin
theframe,theobjecttrackerhastoidentifythethreeseparate
detectionsandneedstotrackitacrossthesubsequentframes(with
thehelpofauniqueID).

DeepSORT
---------

`DeepSORT<https://arxiv.org/abs/1703.07402>`__canbedefinedasthe
trackingalgorithmwhichtracksobjectsnotonlybasedonthevelocity
andmotionoftheobjectbutalsotheappearanceoftheobject.Itis
madeofthreekeycomponentswhichareasfollows:|deepsort|

1.**Detection**

Thisisthefirststepinthetrackingmodule.Inthisstep,adeep
learningmodelwillbeusedtodetecttheobjectsintheframethat
aretobetracked.Thesedetectionsarethenpassedontothenext
step.

2.**Prediction**

Inthisstep,weuseKalmanfilter[1]frameworktopredictatarget
boundingboxofeachtrackingobjectinthenextframe.Therearetwo
statesofpredictionoutput:``confirmed``and``unconfirmed``.Anew
trackcomeswithastateof``unconfirmed``bydefault,anditcanbe
turnedinto``confirmed``whenacertainnumberofconsecutive
detectionsarematchedwiththisnewtrack.Meanwhile,ifamatched
trackismissedoveraspecifictime,itwillbedeletedaswell.

3.**Dataassociationandupdate**

Now,wehavetomatchthetargetboundingboxwiththedetected
boundingbox,andupdatetrackidentities.Aconventionalwayto
solvetheassociationbetweenthepredictedKalmanstatesandnewly
arrivedmeasurementsistobuildanassignmentproblemwiththe
Hungarianalgorithm[2].Inthisproblemformulation,weintegrate
motionandappearanceinformationthroughacombinationoftwo
appropriatemetrics.Thecostusedforthefirstmatchingstepisset
asacombinationoftheMahalanobisandthecosinedistances.The
`Mahalanobis
distance<https://en.wikipedia.org/wiki/Mahalanobis_distance>`__is
usedtoincorporatemotioninformationandthecosinedistanceis
usedtocalculatesimilaritybetweentwoobjects.Cosinedistanceis
ametricthathelpsthetrackerrecoveridentitiesincaseof
long-termocclusionandmotionestimationalsofails.Forthis
purposes,areidentificationmodelwillbeimplementedtoproducea
vectorinhigh-dimensionalspacethatrepresentstheappearanceof
theobject.Usingthesesimplethingscanmakethetrackerevenmore
powerfulandaccurate.

Inthesecondmatchingstage,wewillrunintersectionover
union(IOU)associationasproposedintheoriginalSORTalgorithm[3]
onthesetofunconfirmedandunmatchedtracksfromtheprevious
step.IftheIOUofdetectionandtargetislessthanacertain
thresholdvaluecalled``IOUmin``thenthatassignmentisrejected.
Thishelpstoaccountforsuddenappearancechanges,forexample,due
topartialocclusionwithstaticscenegeometry,andtoincrease
robustnessagainsterroneous.

Whendetectionresultisassociatedwithatarget,thedetected
boundingboxisusedtoupdatethetargetstate.

--------------

[1]R.Kalman,“ANewApproachtoLinearFilteringandPrediction
Problems”,JournalofBasicEngineering,vol. 82,no.SeriesD,
pp. 35-45,1960.

[2]H.W.Kuhn,“TheHungarianmethodfortheassignmentproblem”,Naval
ResearchLogisticsQuarterly,vol. 2,pp. 83-97,1955.

[3]A.Bewley,G.Zongyuan,F.Ramos,andB.Upcroft,“Simpleonlineand
realtimetracking,”inICIP,2016,pp. 3464–3468.

..|deepsort|image::https://user-images.githubusercontent.com/91237924/221744683-0042eff8-2c41-43b8-b3ad-b5929bafb60b.png

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`DownloadtheModel<#download-the-model>`__
-`Loadmodel<#load-model>`__

-`Selectinferencedevice<#select-inference-device>`__

-`DataProcessing<#data-processing>`__
-`Testpersonreidentification
model<#test-person-reidentification-model>`__

-`Visualizedata<#visualize-data>`__
-`Comparetwopersons<#compare-two-persons>`__

-`MainProcessingFunction<#main-processing-function>`__
-`Run<#run>`__

-`Initializetracker<#initialize-tracker>`__
-`RunLivePersonTracking<#run-live-person-tracking>`__

..code::ipython3

importplatform

%pipinstall-q"openvino-dev>=2024.0.0"
%pipinstall-qopencv-pythonrequestsscipytqdm

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcollections
frompathlibimportPath
importtime

importnumpyasnp
importcv2
fromIPythonimportdisplay
importmatplotlib.pyplotasplt
importopenvinoasov

..code::ipython3

#Importlocalmodules

ifnotPath("./notebook_utils.py").exists():
#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

importnotebook_utilsasutils
fromdeepsort_utils.trackerimportTracker
fromdeepsort_utils.nn_matchingimportNearestNeighborDistanceMetric
fromdeepsort_utils.detectionimport(
Detection,
compute_color_for_labels,
xywh_to_xyxy,
xywh_to_tlwh,
tlwh_to_xyxy,
)

DownloadtheModel
------------------

`backtotop⬆️<#table-of-contents>`__

Wewillusepre-trainedmodelsfromOpenVINO’s`OpenModel
Zoo<https://docs.openvino.ai/2024/documentation/legacy-features/model-zoo.html>`__
tostartthetest.

Use``omz_downloader``,whichisacommand-linetoolfromthe
``openvino-dev``package.Itautomaticallycreatesadirectorystructure
anddownloadstheselectedmodel.Thisstepisskippedifthemodelis
alreadydownloaded.Theselectedmodelcomesfromthepublicdirectory,
whichmeansitmustbeconvertedintoOpenVINOIntermediate
Representation(OpenVINOIR).

**NOTE**:Usingamodeloutsidethelistcanrequiredifferentpre-
andpost-processing.

Inthiscase,`persondetection
model<https://docs.openvino.ai/2024/omz_models_model_person_detection_0202.html>`__
isdeployedtodetectthepersonineachframeofthevideo,and
`reidentification
model<https://docs.openvino.ai/2024/omz_models_model_person_reidentification_retail_0287.html>`__
isusedtooutputembeddingvectortomatchapairofimagesofaperson
bythecosinedistance.

Ifyouwanttodownloadanothermodel(``person-detection-xxx``from
`ObjectDetectionModels
list<https://docs.openvino.ai/2024/omz_models_group_intel.html#object-detection-models>`__,
``person-reidentification-retail-xxx``from`ReidentificationModels
list<https://docs.openvino.ai/2024/omz_models_group_intel.html#reidentification-models>`__),
replacethenameofthemodelinthecodebelow.

..code::ipython3

#Adirectorywherethemodelwillbedownloaded.
base_model_dir="model"
precision="FP16"
#ThenameofthemodelfromOpenModelZoo
detection_model_name="person-detection-0202"

download_command=(
f"omz_downloader"f"--name{detection_model_name}"f"--precisions{precision}"f"--output_dir{base_model_dir}"f"--cache_dir{base_model_dir}"
)
!$download_command

detection_model_path=f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"


reidentification_model_name="person-reidentification-retail-0287"

download_command=(
f"omz_downloader"f"--name{reidentification_model_name}"f"--precisions{precision}"f"--output_dir{base_model_dir}"f"--cache_dir{base_model_dir}"
)
!$download_command

reidentification_model_path=f"model/intel/{reidentification_model_name}/{precision}/{reidentification_model_name}.xml"


..parsed-literal::

################||Downloadingperson-detection-0202||################

==========Downloadingmodel/intel/person-detection-0202/FP16/person-detection-0202.xml


==========Downloadingmodel/intel/person-detection-0202/FP16/person-detection-0202.bin


################||Downloadingperson-reidentification-retail-0287||################

==========Downloadingmodel/intel/person-reidentification-retail-0287/person-reidentification-retail-0267.onnx


==========Downloadingmodel/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml


==========Downloadingmodel/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.bin




Loadmodel
----------

`backtotop⬆️<#table-of-contents>`__

Defineacommonclassformodelloadingandpredicting.

TherearefourmainstepsforOpenVINOmodelinitialization,andthey
arerequiredtorunforonlyoncebeforeinferenceloop.1.Initialize
OpenVINORuntime.2.Readthenetworkfrom``*.bin``and``*.xml``files
(weightsandarchitecture).3.Compilethemodelfordevice.4.Get
inputandoutputnamesofnodes.

Inthiscase,wecanputthemallinaclassconstructorfunction.

ToletOpenVINOautomaticallyselectthebestdeviceforinferencejust
use``AUTO``.Inmostcases,thebestdevicetouseis``GPU``(better
performance,butslightlylongerstartuptime).

..code::ipython3

core=ov.Core()


classModel:
"""
ThisclassrepresentsaOpenVINOmodelobject.

"""

def__init__(self,model_path,batchsize=1,device="AUTO"):
"""
Initializethemodelobject

Parameters
----------
model_path:pathofinferencemodel
batchsize:batchsizeofinputdata
device:deviceusedtoruninference
"""
self.model=core.read_model(model=model_path)
self.input_layer=self.model.input(0)
self.input_shape=self.input_layer.shape
self.height=self.input_shape[2]
self.width=self.input_shape[3]

forlayerinself.model.inputs:
input_shape=layer.partial_shape
input_shape[0]=batchsize
self.model.reshape({layer:input_shape})
self.compiled_model=core.compile_model(model=self.model,device_name=device)
self.output_layer=self.compiled_model.output(0)

defpredict(self,input):
"""
Runinference

Parameters
----------
input:arrayofinputdata
"""
result=self.compiled_model(input)[self.output_layer]
returnresult

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

detector=Model(detection_model_path,device=device.value)
#sincethenumberofdetectionobjectisuncertain,theinputbatchsizeofreidmodelshouldbedynamic
extractor=Model(reidentification_model_path,-1,device.value)

DataProcessing
---------------

`backtotop⬆️<#table-of-contents>`__

DataProcessingincludesdatapreprocessandpostprocessfunctions.-
Datapreprocessfunctionisusedtochangethelayoutandshapeofinput
data,accordingtorequirementofthenetworkinputformat.-Data
postprocessfunctionisusedtoextracttheusefulinformationfrom
network’soriginaloutputandvisualizeit.

..code::ipython3

defpreprocess(frame,height,width):
"""
Preprocessasingleimage

Parameters
----------
frame:inputframe
height:heightofmodelinputdata
width:widthofmodelinputdata
"""
resized_image=cv2.resize(frame,(width,height))
resized_image=resized_image.transpose((2,0,1))
input_image=np.expand_dims(resized_image,axis=0).astype(np.float32)
returninput_image


defbatch_preprocess(img_crops,height,width):
"""
Preprocessbatchedimages

Parameters
----------
img_crops:batchedinputimages
height:heightofmodelinputdata
width:widthofmodelinputdata
"""
img_batch=np.concatenate([preprocess(img,height,width)forimginimg_crops],axis=0)
returnimg_batch


defprocess_results(h,w,results,thresh=0.5):
"""
postprocessdetectionresults

Parameters
----------
h,w:originalheightandwidthofinputimage
results:rawdetectionnetworkoutput
thresh:thresholdforlowconfidencefiltering
"""
#The'results'variableisa[1,1,N,7]tensor.
detections=results.reshape(-1,7)
boxes=[]
labels=[]
scores=[]
fori,detectioninenumerate(detections):
_,label,score,xmin,ymin,xmax,ymax=detection
#Filterdetectedobjects.
ifscore>thresh:
#Createaboxwithpixelscoordinatesfromtheboxwithnormalizedcoordinates[0,1].
boxes.append(
[
(xmin+xmax)/2*w,
(ymin+ymax)/2*h,
(xmax-xmin)*w,
(ymax-ymin)*h,
]
)
labels.append(int(label))
scores.append(float(score))

iflen(boxes)==0:
boxes=np.array([]).reshape(0,4)
scores=np.array([])
labels=np.array([])
returnnp.array(boxes),np.array(scores),np.array(labels)


defdraw_boxes(img,bbox,identities=None):
"""
Drawboundingboxinoriginalimage

Parameters
----------
img:originalimage
bbox:coordinateofboundingbox
identities:identitiesIDs
"""
fori,boxinenumerate(bbox):
x1,y1,x2,y2=[int(i)foriinbox]
#boxtextandbar
id=int(identities[i])ifidentitiesisnotNoneelse0
color=compute_color_for_labels(id)
label="{}{:d}".format("",id)
t_size=cv2.getTextSize(label,cv2.FONT_HERSHEY_PLAIN,2,2)[0]
cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
cv2.rectangle(img,(x1,y1),(x1+t_size[0]+3,y1+t_size[1]+4),color,-1)
cv2.putText(
img,
label,
(x1,y1+t_size[1]+4),
cv2.FONT_HERSHEY_PLAIN,
1.6,
[255,255,255],
2,
)
returnimg


defcosin_metric(x1,x2):
"""
Calculatetheconsindistanceoftwovector

Parameters
----------
x1,x2:inputvectors
"""
returnnp.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))

Testpersonreidentificationmodel
----------------------------------

`backtotop⬆️<#table-of-contents>`__

Thereidentificationnetworkoutputsablobwiththe``(1,256)``shape
named``reid_embedding``,whichcanbecomparedwithotherdescriptors
usingthecosinedistance.

Visualizedata
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

base_file_link="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/person_"
image_indices=["1_1.png","1_2.png","2_1.png"]
image_paths=[utils.download_file(base_file_link+image_index,directory="data")forimage_indexinimage_indices]
image1,image2,image3=[cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)forimage_pathinimage_paths]

#Definetitleswithimages.
data={"Person1":image1,"Person2":image2,"Person3":image3}

#Createasubplottovisualizeimages.
fig,axs=plt.subplots(1,len(data.items()),figsize=(5,5))

#Fillthesubplot.
forax,(name,image)inzip(axs,data.items()):
ax.axis("off")
ax.set_title(name)
ax.imshow(image)

#Displayanimage.
plt.show(fig)



..parsed-literal::

data/person_1_1.png:0%||0.00/68.3k[00:00<?,?B/s]



..parsed-literal::

data/person_1_2.png:0%||0.00/68.9k[00:00<?,?B/s]



..parsed-literal::

data/person_2_1.png:0%||0.00/70.3k[00:00<?,?B/s]



..image::person-tracking-with-output_files/person-tracking-with-output_17_3.png


Comparetwopersons
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Metricparameters
MAX_COSINE_DISTANCE=0.6#thresholdofmatchingobject
input_data=[image2,image3]
img_batch=batch_preprocess(input_data,extractor.height,extractor.width)
features=extractor.predict(img_batch)
sim=cosin_metric(features[0],features[1])
ifsim>=1-MAX_COSINE_DISTANCE:
print(f"Sameperson(confidence:{sim})")
else:
print(f"Differentperson(confidence:{sim})")


..parsed-literal::

Differentperson(confidence:0.02726624347269535)


MainProcessingFunction
------------------------

`backtotop⬆️<#table-of-contents>`__

Runpersontrackingonthespecifiedsource.Eitherawebcamfeedora
videofile.

..code::ipython3

#Mainprocessingfunctiontorunpersontracking.
defrun_person_tracking(source=0,flip=False,use_popup=False,skip_first_frames=0):
"""
Mainfunctiontorunthepersontracking:
1.Createavideoplayertoplaywithtargetfps(utils.VideoPlayer).
2.Prepareasetofframesforpersontracking.
3.RunAIinferenceforpersontracking.
4.Visualizetheresults.

Parameters:
----------
source:Thewebcamnumbertofeedthevideostreamwithprimarywebcamsetto"0",orthevideopath.
flip:TobeusedbyVideoPlayerfunctionforflippingcaptureimage.
use_popup:Falseforshowingencodedframesoverthisnotebook,Trueforcreatingapopupwindow.
skip_first_frames:Numberofframestoskipatthebeginningofthevideo.
"""
player=None
try:
#Createavideoplayertoplaywithtargetfps.
player=utils.VideoPlayer(
source=source,
size=(700,450),
flip=flip,
fps=24,
skip_first_frames=skip_first_frames,
)
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

#Resizetheimageandchangedimstofitneuralnetworkinput.
h,w=frame.shape[:2]
input_image=preprocess(frame,detector.height,detector.width)

#Measureprocessingtime.
start_time=time.time()
#Gettheresults.
output=detector.predict(input_image)
stop_time=time.time()
processing_times.append(stop_time-start_time)
iflen(processing_times)>200:
processing_times.popleft()

_,f_width=frame.shape[:2]
#Meanprocessingtime[ms].
processing_time=np.mean(processing_times)*1100
fps=1000/processing_time

#Getposesfromdetectionresults.
bbox_xywh,score,label=process_results(h,w,results=output)

img_crops=[]
forboxinbbox_xywh:
x1,y1,x2,y2=xywh_to_xyxy(box,h,w)
img=frame[y1:y2,x1:x2]
img_crops.append(img)

#Getreidentificationfeatureofeachperson.
ifimg_crops:
#preprocess
img_batch=batch_preprocess(img_crops,extractor.height,extractor.width)
features=extractor.predict(img_batch)
else:
features=np.array([])

#Wrapthedetectionandreidentificationresultstogether
bbox_tlwh=xywh_to_tlwh(bbox_xywh)
detections=[Detection(bbox_tlwh[i],features[i])foriinrange(features.shape[0])]

#predictthepositionoftrackingtarget
tracker.predict()

#updatetracker
tracker.update(detections)

#updatebboxidentities
outputs=[]
fortrackintracker.tracks:
ifnottrack.is_confirmed()ortrack.time_since_update>1:
continue
box=track.to_tlwh()
x1,y1,x2,y2=tlwh_to_xyxy(box,h,w)
track_id=track.track_id
outputs.append(np.array([x1,y1,x2,y2,track_id],dtype=np.int32))
iflen(outputs)>0:
outputs=np.stack(outputs,axis=0)

#drawboxforvisualization
iflen(outputs)>0:
bbox_tlwh=[]
bbox_xyxy=outputs[:,:4]
identities=outputs[:,-1]
frame=draw_boxes(frame,bbox_xyxy,identities)

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

Initializetracker
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Beforerunninganewtrackingtask,wehavetoreinitializeaTracker
object

..code::ipython3

NN_BUDGET=100
MAX_COSINE_DISTANCE=0.6#thresholdofmatchingobject
metric=NearestNeighborDistanceMetric("cosine",MAX_COSINE_DISTANCE,NN_BUDGET)
tracker=Tracker(metric,max_iou_distance=0.7,max_age=70,n_init=3)

RunLivePersonTracking
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Useawebcamasthevideoinput.Bydefault,theprimarywebcamisset
with``source=0``.Ifyouhavemultiplewebcams,eachonewillbe
assignedaconsecutivenumberstartingat0.Set``flip=True``when
usingafront-facingcamera.Somewebbrowsers,especiallyMozilla
Firefox,maycauseflickering.Ifyouexperienceflickering,set
``use_popup=True``.

Ifyoudonothaveawebcam,youcanstillrunthisdemowithavideo
file.Any`formatsupportedby
OpenCV<https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
willwork.

..code::ipython3

USE_WEBCAM=False

cam_id=0
video_file="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
source=cam_idifUSE_WEBCAMelsevideo_file

run_person_tracking(source=source,flip=USE_WEBCAM,use_popup=False)



..image::person-tracking-with-output_files/person-tracking-with-output_25_0.png


..parsed-literal::

Sourceended

