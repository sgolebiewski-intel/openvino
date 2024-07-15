AsynchronousInferencewithOpenVINO™
=====================================

Thisnotebookdemonstrateshowtousethe`Async
API<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/general-optimizations.html>`__
forasynchronousexecutionwithOpenVINO.

OpenVINORuntimesupportsinferenceineithersynchronousor
asynchronousmode.ThekeyadvantageoftheAsyncAPIisthatwhena
deviceisbusywithinference,theapplicationcanperformothertasks
inparallel(forexample,populatinginputsorschedulingother
requests)ratherthanwaitforthecurrentinferencetocompletefirst.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Preparemodelanddata
processing<#prepare-model-and-data-processing>`__

-`Downloadtestmodel<#download-test-model>`__
-`Loadthemodel<#load-the-model>`__
-`Createfunctionsfordata
processing<#create-functions-for-data-processing>`__
-`Getthetestvideo<#get-the-test-video>`__

-`Howtoimprovethethroughputofvideo
processing<#how-to-improve-the-throughput-of-video-processing>`__

-`SyncMode(default)<#sync-mode-default>`__
-`TestperformanceinSyncMode<#test-performance-in-sync-mode>`__
-`AsyncMode<#async-mode>`__
-`TesttheperformanceinAsync
Mode<#test-the-performance-in-async-mode>`__
-`Comparetheperformance<#compare-the-performance>`__

-`AsyncInferQueue<#asyncinferqueue>`__

-`SettingCallback<#setting-callback>`__
-`Testtheperformancewith
AsyncInferQueue<#test-the-performance-with-asyncinferqueue>`__

Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.1.0"
%pipinstall-qopencv-python
ifplatform.system()!="windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importcv2
importtime
importnumpyasnp
importopenvinoasov
fromIPythonimportdisplay
importmatplotlib.pyplotasplt

#Fetchthenotebookutilsscriptfromtheopenvino_notebooksrepo
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py","w").write(r.text)

importnotebook_utilsasutils

Preparemodelanddataprocessing
---------------------------------

`backtotop⬆️<#table-of-contents>`__

Downloadtestmodel
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseapre-trainedmodelfromOpenVINO’s`OpenModel
Zoo<https://docs.openvino.ai/2024/documentation/legacy-features/model-zoo.html>`__
tostartthetest.Inthiscase,themodelwillbeexecutedtodetect
thepersonineachframeofthevideo.

..code::ipython3

#directorywheremodelwillbedownloaded
base_model_dir="model"

#modelnameasnamedinOpenModelZoo
model_name="person-detection-0202"
precision="FP16"
model_path=f"model/intel/{model_name}/{precision}/{model_name}.xml"
download_command=f"omz_downloader"f"--name{model_name}"f"--precision{precision}"f"--output_dir{base_model_dir}"f"--cache_dir{base_model_dir}"
!$download_command


..parsed-literal::

################||Downloadingperson-detection-0202||################

==========Downloadingmodel/intel/person-detection-0202/FP16/person-detection-0202.xml


==========Downloadingmodel/intel/person-detection-0202/FP16/person-detection-0202.bin




Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importipywidgetsaswidgets

core=ov.Core()
device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','AUTO'),value='CPU')



Loadthemodel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#initializeOpenVINOruntime
core=ov.Core()

#readthenetworkandcorrespondingweightsfromfile
model=core.read_model(model=model_path)

#compilethemodelfortheCPU(youcanchoosemanuallyCPU,GPUetc.)
#orlettheenginechoosethebestavailabledevice(AUTO)
compiled_model=core.compile_model(model=model,device_name=device.value)

#getinputnode
input_layer_ir=model.input(0)
N,C,H,W=input_layer_ir.shape
shape=(H,W)

Createfunctionsfordataprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defpreprocess(image):
"""
Definethepreprocessfunctionforinputdata

:param:image:theorignalinputframe
:returns:
resized_image:theimageprocessed
"""
resized_image=cv2.resize(image,shape)
resized_image=cv2.cvtColor(np.array(resized_image),cv2.COLOR_BGR2RGB)
resized_image=resized_image.transpose((2,0,1))
resized_image=np.expand_dims(resized_image,axis=0).astype(np.float32)
returnresized_image


defpostprocess(result,image,fps):
"""
Definethepostprocessfunctionforoutputdata

:param:result:theinferenceresults
image:theorignalinputframe
fps:averagethroughputcalculatedforeachframe
:returns:
image:theimagewithboundingboxandfpsmessage
"""
detections=result.reshape(-1,7)
fori,detectioninenumerate(detections):
_,image_id,confidence,xmin,ymin,xmax,ymax=detection
ifconfidence>0.5:
xmin=int(max((xmin*image.shape[1]),10))
ymin=int(max((ymin*image.shape[0]),10))
xmax=int(min((xmax*image.shape[1]),image.shape[1]-10))
ymax=int(min((ymax*image.shape[0]),image.shape[0]-10))
cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
cv2.putText(
image,
str(round(fps,2))+"fps",
(5,20),
cv2.FONT_HERSHEY_SIMPLEX,
0.7,
(0,255,0),
3,
)
returnimage

Getthetestvideo
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

video_path="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/CEO%20Pat%20Gelsinger%20on%20Leading%20Intel.mp4"

Howtoimprovethethroughputofvideoprocessing
-------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Below,wecomparetheperformanceofthesynchronousandasync-based
approaches:

SyncMode(default)
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Letusseehowvideoprocessingworkswiththedefaultapproach.Using
thesynchronousapproach,theframeiscapturedwithOpenCVandthen
immediatelyprocessed:

..figure::https://user-images.githubusercontent.com/91237924/168452573-d354ea5b-7966-44e5-813d-f9053be4338a.png
:alt:drawing

drawing

::

while(true){
//captureframe
//populateCURRENTInferRequest
//InferCURRENTInferRequest
//thiscallissynchronous
//displayCURRENTresult
}

\``\`

..code::ipython3

defsync_api(source,flip,fps,use_popup,skip_first_frames):
"""
Definethemainfunctionforvideoprocessinginsyncmode

:param:source:thevideopathortheIDofyourwebcam
:returns:
sync_fps:theinferencethroughputinsyncmode
"""
frame_number=0
infer_request=compiled_model.create_infer_request()
player=None
try:
#Createavideoplayer
player=utils.VideoPlayer(source,flip=flip,fps=fps,skip_first_frames=skip_first_frames)
#Startcapturing
start_time=time.time()
player.start()
ifuse_popup:
title="PressESCtoExit"
cv2.namedWindow(title,cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE)
whileTrue:
frame=player.next()
ifframeisNone:
print("Sourceended")
break
resized_frame=preprocess(frame)
infer_request.set_tensor(input_layer_ir,ov.Tensor(resized_frame))
#Starttheinferencerequestinsynchronousmode
infer_request.infer()
res=infer_request.get_output_tensor(0).data
stop_time=time.time()
total_time=stop_time-start_time
frame_number=frame_number+1
sync_fps=frame_number/total_time
frame=postprocess(res,frame,sync_fps)
#Displaytheresults
ifuse_popup:
cv2.imshow(title,frame)
key=cv2.waitKey(1)
#escape=27
ifkey==27:
break
else:
#Encodenumpyarraytojpg
_,encoded_img=cv2.imencode(".jpg",frame,params=[cv2.IMWRITE_JPEG_QUALITY,90])
#CreateIPythonimage
i=display.Image(data=encoded_img)
#Displaytheimageinthisnotebook
display.clear_output(wait=True)
display.display(i)
#ctrl-c
exceptKeyboardInterrupt:
print("Interrupted")
#Anydifferenterror
exceptRuntimeErrorase:
print(e)
finally:
ifuse_popup:
cv2.destroyAllWindows()
ifplayerisnotNone:
#stopcapturing
player.stop()
returnsync_fps

TestperformanceinSyncMode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

sync_fps=sync_api(source=video_path,flip=False,fps=30,use_popup=False,skip_first_frames=800)
print(f"averagethrouputinsyncmode:{sync_fps:.2f}fps")



..image::async-api-with-output_files/async-api-with-output_17_0.png


..parsed-literal::

Sourceended
averagethrouputinsyncmode:58.66fps


AsyncMode
~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

LetusseehowtheOpenVINOAsyncAPIcanimprovetheoverallframerate
ofanapplication.ThekeyadvantageoftheAsyncapproachisas
follows:whileadeviceisbusywiththeinference,theapplicationcan
dootherthingsinparallel(forexample,populatinginputsor
schedulingotherrequests)ratherthanwaitforthecurrentinferenceto
completefirst.

..figure::https://user-images.githubusercontent.com/91237924/168452572-c2ff1c59-d470-4b85-b1f6-b6e1dac9540e.png
:alt:drawing

drawing

Intheexamplebelow,inferenceisappliedtotheresultsofthevideo
decoding.Soitispossibletokeepmultipleinferrequests,andwhile
thecurrentrequestisprocessed,theinputframeforthenextisbeing
captured.Thisessentiallyhidesthelatencyofcapturing,sothatthe
overallframerateisratherdeterminedonlybytheslowestpartofthe
pipeline(decodingvsinference)andnotbythesumofthestages.

::

while(true){
//captureframe
//populateNEXTInferRequest
//startNEXTInferRequest
//thiscallisasyncandreturnsimmediately
//waitfortheCURRENTInferRequest
//displayCURRENTresult
//swapCURRENTandNEXTInferRequests
}

..code::ipython3

defasync_api(source,flip,fps,use_popup,skip_first_frames):
"""
Definethemainfunctionforvideoprocessinginasyncmode

:param:source:thevideopathortheIDofyourwebcam
:returns:
async_fps:theinferencethroughputinasyncmode
"""
frame_number=0
#Create2inferrequests
curr_request=compiled_model.create_infer_request()
next_request=compiled_model.create_infer_request()
player=None
async_fps=0
try:
#Createavideoplayer
player=utils.VideoPlayer(source,flip=flip,fps=fps,skip_first_frames=skip_first_frames)
#Startcapturing
start_time=time.time()
player.start()
ifuse_popup:
title="PressESCtoExit"
cv2.namedWindow(title,cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE)
#CaptureCURRENTframe
frame=player.next()
resized_frame=preprocess(frame)
curr_request.set_tensor(input_layer_ir,ov.Tensor(resized_frame))
#StarttheCURRENTinferencerequest
curr_request.start_async()
whileTrue:
#CaptureNEXTframe
next_frame=player.next()
ifnext_frameisNone:
print("Sourceended")
break
resized_frame=preprocess(next_frame)
next_request.set_tensor(input_layer_ir,ov.Tensor(resized_frame))
#StarttheNEXTinferencerequest
next_request.start_async()
#WaitingforCURRENTinferenceresult
curr_request.wait()
res=curr_request.get_output_tensor(0).data
stop_time=time.time()
total_time=stop_time-start_time
frame_number=frame_number+1
async_fps=frame_number/total_time
frame=postprocess(res,frame,async_fps)
#Displaytheresults
ifuse_popup:
cv2.imshow(title,frame)
key=cv2.waitKey(1)
#escape=27
ifkey==27:
break
else:
#Encodenumpyarraytojpg
_,encoded_img=cv2.imencode(".jpg",frame,params=[cv2.IMWRITE_JPEG_QUALITY,90])
#CreateIPythonimage
i=display.Image(data=encoded_img)
#Displaytheimageinthisnotebook
display.clear_output(wait=True)
display.display(i)
#SwapCURRENTandNEXTframes
frame=next_frame
#SwapCURRENTandNEXTinferrequests
curr_request,next_request=next_request,curr_request
#ctrl-c
exceptKeyboardInterrupt:
print("Interrupted")
#Anydifferenterror
exceptRuntimeErrorase:
print(e)
finally:
ifuse_popup:
cv2.destroyAllWindows()
ifplayerisnotNone:
#stopcapturing
player.stop()
returnasync_fps

TesttheperformanceinAsyncMode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

async_fps=async_api(source=video_path,flip=False,fps=30,use_popup=False,skip_first_frames=800)
print(f"averagethrouputinasyncmode:{async_fps:.2f}fps")



..image::async-api-with-output_files/async-api-with-output_21_0.png


..parsed-literal::

Sourceended
averagethrouputinasyncmode:103.49fps


Comparetheperformance
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

width=0.4
fontsize=14

plt.rc("font",size=fontsize)
fig,ax=plt.subplots(1,1,figsize=(10,8))

rects1=ax.bar([0],sync_fps,width,color="#557f2d")
rects2=ax.bar([width],async_fps,width)
ax.set_ylabel("framespersecond")
ax.set_xticks([0,width])
ax.set_xticklabels(["Syncmode","Asyncmode"])
ax.set_xlabel("Higherisbetter")

fig.suptitle("SyncmodeVSAsyncmode")
fig.tight_layout()

plt.show()



..image::async-api-with-output_files/async-api-with-output_23_0.png


``AsyncInferQueue``
-------------------

`backtotop⬆️<#table-of-contents>`__

Asynchronousmodepipelinescanbesupportedwiththe
`AsyncInferQueue<https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/python-api-exclusives.html#asyncinferqueue>`__
wrapperclass.Thisclassautomaticallyspawnsthepoolof
``InferRequest``objects(alsocalled“jobs”)andprovides
synchronizationmechanismstocontroltheflowofthepipeline.Itisa
simplerwaytomanagetheinferrequestqueueinAsynchronousmode.

SettingCallback
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

When``callback``isset,anyjobthatendsinferencecallsuponthe
Pythonfunction.The``callback``functionmusthavetwoarguments:one
istherequestthatcallsthe``callback``,whichprovidesthe
``InferRequest``API;theotheriscalled“userdata”,whichprovides
thepossibilityofpassingruntimevalues.

..code::ipython3

defcallback(infer_request,info)->None:
"""
Definethecallbackfunctionforpostprocessing

:param:infer_request:theinfer_requestobject
info:atupleincludesoriginalframeandstartstime
:returns:
None
"""
globalframe_number
globaltotal_time
globalinferqueue_fps
stop_time=time.time()
frame,start_time=info
total_time=stop_time-start_time
frame_number=frame_number+1
inferqueue_fps=frame_number/total_time

res=infer_request.get_output_tensor(0).data[0]
frame=postprocess(res,frame,inferqueue_fps)
#Encodenumpyarraytojpg
_,encoded_img=cv2.imencode(".jpg",frame,params=[cv2.IMWRITE_JPEG_QUALITY,90])
#CreateIPythonimage
i=display.Image(data=encoded_img)
#Displaytheimageinthisnotebook
display.clear_output(wait=True)
display.display(i)

..code::ipython3

definferqueue(source,flip,fps,skip_first_frames)->None:
"""
Definethemainfunctionforvideoprocessingwithasyncinferqueue

:param:source:thevideopathortheIDofyourwebcam
:retuns:
None
"""
#Createinferrequestsqueue
infer_queue=ov.AsyncInferQueue(compiled_model,2)
infer_queue.set_callback(callback)
player=None
try:
#Createavideoplayer
player=utils.VideoPlayer(source,flip=flip,fps=fps,skip_first_frames=skip_first_frames)
#Startcapturing
start_time=time.time()
player.start()
whileTrue:
#Captureframe
frame=player.next()
ifframeisNone:
print("Sourceended")
break
resized_frame=preprocess(frame)
#Starttheinferencerequestwithasyncinferqueue
infer_queue.start_async({input_layer_ir.any_name:resized_frame},(frame,start_time))
exceptKeyboardInterrupt:
print("Interrupted")
#Anydifferenterror
exceptRuntimeErrorase:
print(e)
finally:
infer_queue.wait_all()
player.stop()

Testtheperformancewith``AsyncInferQueue``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frame_number=0
total_time=0
inferqueue(source=video_path,flip=False,fps=30,skip_first_frames=800)
print(f"averagethroughputinasyncmodewithasyncinferqueue:{inferqueue_fps:.2f}fps")



..image::async-api-with-output_files/async-api-with-output_29_0.png


..parsed-literal::

averagethroughputinasyncmodewithasyncinferqueue:149.16fps

