SelfieSegmentationusingTFLiteandOpenVINO
=============================================

TheSelfiesegmentationpipelineallowsdeveloperstoeasilyseparate
thebackgroundfromuserswithinasceneandfocusonwhatmatters.
Addingcooleffectstoselfiesorinsertingyourusersintointeresting
backgroundenvironmentshasneverbeeneasier.Besidesphotoediting,
thistechnologyisalsoimportantforvideoconferencing.Ithelpsto
blurorreplacethebackgroundduringvideocalls.

Inthistutorial,weconsiderhowtoimplementselfiesegmentationusing
OpenVINO.Wewilluse`MulticlassSelfie-segmentation
model<https://developers.google.com/mediapipe/solutions/vision/image_segmenter/#multiclass-model>`__
providedaspartof`Google
MediaPipe<https://developers.google.com/mediapipe>`__solution.

TheMulticlassSelfie-segmentationmodelisamulticlasssemantic
segmentationmodelandclassifieseachpixelasbackground,hair,body,
face,clothes,andothers(e.g. accessories).Themodelsupportssingle
ormultiplepeopleintheframe,selfies,andfull-bodyimages.The
modelisbasedon`Vision
Transformer<https://arxiv.org/abs/2010.11929>`__withcustomized
bottleneckanddecoderarchitectureforreal-timeperformance.More
detailsaboutthemodelcanbefoundin`model
card<https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf>`__.
ThismodelisrepresentedinTensorflowLiteformat.`TensorFlow
Lite<https://www.tensorflow.org/lite/guide>`__,oftenreferredtoas
TFLite,isanopen-sourcelibrarydevelopedfordeployingmachine
learningmodelstoedgedevices.

Thetutorialconsistsoffollowingsteps:

1.DownloadtheTFLitemodelandconvertittoOpenVINOIRformat.
2.Runinferenceontheimage.
3.Runinteractivebackgroundblurringdemoonvideo.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__

-`Installrequireddependencies<#install-required-dependencies>`__
-`Downloadpretrainedmodelandtest
image<#download-pretrained-model-and-test-image>`__

-`ConvertTensorflowLitemodeltoOpenVINOIR
format<#convert-tensorflow-lite-model-to-openvino-ir-format>`__
-`RunOpenVINOmodelinferenceon
image<#run-openvino-model-inference-on-image>`__

-`Loadmodel<#load-model>`__
-`Prepareinputimage<#prepare-input-image>`__
-`Runmodelinference<#run-model-inference>`__
-`Postprocessandvisualizeinference
results<#postprocess-and-visualize-inference-results>`__

-`Interactivebackgroundblurringdemoon
video<#interactive-background-blurring-demo-on-video>`__

-`RunLiveBackgroundBlurring<#run-live-background-blurring>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

Installrequireddependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.1.0""opencv-python""tqdm"

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)




..parsed-literal::

23215



Downloadpretrainedmodelandtestimage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frompathlibimportPath
fromnotebook_utilsimportdownload_file

tflite_model_path=Path("selfie_multiclass_256x256.tflite")
tflite_model_url="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

download_file(tflite_model_url,tflite_model_path)



..parsed-literal::

selfie_multiclass_256x256.tflite:0%||0.00/15.6M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/tflite-selfie-segmentation/selfie_multiclass_256x256.tflite')



ConvertTensorflowLitemodeltoOpenVINOIRformat
---------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Startingfromthe2023.0.0release,OpenVINOsupportsTFLitemodel
conversion.HoweverTFLitemodelformatcanbedirectlypassedin
``read_model``(youcanfindexamplesofthisAPIusageforTFLitein
`TFLitetoOpenVINOconversion
tutorial<tflite-to-openvino-with-output.html>`__and
tutorialwith`basicOpenVINOAPI
capabilities<openvino-api-with-output.html>`__),itisrecommended
toconvertmodeltoOpenVINOIntermediateRepresentationformattoapply
additionaloptimizations(e.g. weightscompressiontoFP16format).To
converttheTFLitemodeltoOpenVINOIR,modelconversionPythonAPIcan
beused.The``ov.convert_model``functionacceptsapathtotheTFLite
modelandreturnstheOpenVINOModelclassinstancewhichrepresents
thismodel.Theobtainedmodelisreadytouseandtobeloadedonthe
deviceusing``compile_model``orcanbesavedonadiskusingthe
``ov.save_model``functionreducingloadingtimeforthenextrunning.
Formoreinformationaboutmodelconversion,seethis
`page<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
ForTensorFlowLite,refertothe`models
support<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-tensorflow-lite.html>`__.

..code::ipython3

importopenvinoasov

core=ov.Core()

ir_model_path=tflite_model_path.with_suffix(".xml")

ifnotir_model_path.exists():
ov_model=ov.convert_model(tflite_model_path)
ov.save_model(ov_model,ir_model_path)
else:
ov_model=core.read_model(ir_model_path)

..code::ipython3

print(f"Modelinputinfo:{ov_model.inputs}")


..parsed-literal::

Modelinputinfo:[<Output:names[input_29]shape[1,256,256,3]type:f32>]


Modelinputisafloatingpointtensorwithshape[1,256,256,3]in
``N,H,W,C``format,where

-``N``-batchsize,numberofinputimages.
-``H``-theheightoftheinputimage.
-``W``-widthoftheinputimage.
-``C``-channelsoftheinputimage.

ThemodelacceptsimagesinRGBformatnormalizedin[0,1]rangeby
divisionon255.

..code::ipython3

print(f"Modeloutputinfo:{ov_model.outputs}")


..parsed-literal::

Modeloutputinfo:[<Output:names[Identity]shape[1,256,256,6]type:f32>]


Modeloutputisafloatingpointtensorwiththesimilarformatand
shape,exceptnumberofchannels-6thatrepresentsnumberofsupported
segmentationclasses:background,hair,bodyskin,faceskin,clothes,
andothers.Eachvalueintheoutputtensorrepresentsofprobability
thatthepixelbelongstothespecifiedclass.Wecanusethe``argmax``
operationtogetthelabelwiththehighestprobabilityforeachpixel.

RunOpenVINOmodelinferenceonimage
-------------------------------------

`backtotop⬆️<#table-of-contents>`__

Let’sseethemodelinaction.Forrunningtheinferencemodelwith
OpenVINOweshouldloadthemodelonthedevicefirst.Pleaseusethe
nextdropdownlistfortheselectioninferencedevice.

Loadmodel
~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

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

compiled_model=core.compile_model(ov_model,device.value)

Prepareinputimage
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Themodelacceptsanimagewithsize256x256,weneedtoresizeour
inputimagetofititinthemodelinputtensor.Usually,segmentation
modelsaresensitivetoproportionsofinputimagedetails,so
preservingtheoriginalaspectratioandaddingpaddingcanhelpimprove
segmentationaccuracy,wewillusethispre-processingapproach.
Additionally,theinputimageisrepresentedasanRGBimageinUINT8
([0,255]datarange),weshouldnormalizeitin[0,1].

..code::ipython3

importcv2
importnumpyasnp
fromnotebook_utilsimportload_image

#ReadinputimageandconvertittoRGB
test_image_url="https://user-images.githubusercontent.com/29454499/251036317-551a2399-303e-4a4a-a7d6-d7ce973e05c5.png"
img=load_image(test_image_url)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


#Preprocessinghelperfunction
defresize_and_pad(image:np.ndarray,height:int=256,width:int=256):
"""
Inputpreprocessingfunction,takesinputimageinnp.ndarrayformat,
resizesittofitspecifiedheightandwidthwithpreservingaspectratio
andaddspaddingonbottomorrightsidetocompletetargetheightxwidthrectangle.

Parameters:
image(np.ndarray):inputimageinnp.ndarrayformat
height(int,*optional*,256):targetheight
width(int,*optional*,256):targetwidth
Returns:
padded_img(np.ndarray):processedimage
padding_info(Tuple[int,int]):informationaboutpaddingsize,requiredforpostprocessing
"""
h,w=image.shape[:2]
ifh<w:
img=cv2.resize(image,(width,np.floor(h/(w/width)).astype(int)))
else:
img=cv2.resize(image,(np.floor(w/(h/height)).astype(int),height))

r_h,r_w=img.shape[:2]
right_padding=width-r_w
bottom_padding=height-r_h
padded_img=cv2.copyMakeBorder(img,0,bottom_padding,0,right_padding,cv2.BORDER_CONSTANT)
returnpadded_img,(bottom_padding,right_padding)


#Applypreprocessigstep-resizeandpadinputimage
padded_img,pad_info=resize_and_pad(np.array(img))

#Convertinputdatafromuint8[0,255]tofloat32[0,1]rangeandaddbatchdimension
normalized_img=np.expand_dims(padded_img.astype(np.float32)/255,0)

Runmodelinference
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

out=compiled_model(normalized_img)[0]

Postprocessandvisualizeinferenceresults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Themodelpredictssegmentationprobabilitiesmaskwiththesize256x
256,weneedtoapplypostprocessingtogetlabelswiththehighest
probabilityforeachpixelandrestoretheresultintheoriginalinput
imagesize.Wecaninterprettheresultofthemodelindifferentways,
e.g. visualizethesegmentationmask,applysomevisualeffectsonthe
selectedbackground(remove,replaceitwithanyotherpicture,blurit)
orotherclasses(forexample,changethecolorofperson’shairoradd
makeup).

..code::ipython3

fromtypingimportTuple
fromnotebook_utilsimportsegmentation_map_to_image,SegmentationMap,Label

#helperforvisualizationsegmentationlabels
labels=[
Label(index=0,color=(192,192,192),name="background"),
Label(index=1,color=(128,0,0),name="hair"),
Label(index=2,color=(255,229,204),name="bodyskin"),
Label(index=3,color=(255,204,204),name="faceskin"),
Label(index=4,color=(0,0,128),name="clothes"),
Label(index=5,color=(128,0,128),name="others"),
]
SegmentationLabels=SegmentationMap(labels)


#helperforpostprocessingoutputmask
defpostprocess_mask(out:np.ndarray,pad_info:Tuple[int,int],orig_img_size:Tuple[int,int]):
"""
Posptprocessingfunctionforsegmentationmask,acceptsmodeloutputtensor,
getslabelsforeachpixelusingargmax,
unpadssegmentationmaskandresizesittooriginalimagesize.

Parameters:
out(np.ndarray):modeloutputtensor
pad_info(Tuple[int,int]):informationaboutpaddingsizefrompreprocessingstep
orig_img_size(Tuple[int,int]):originalimageheightandwidthforresizing
Returns:
label_mask_resized(np.ndarray):postprocessedsegmentationlabelmask
"""
label_mask=np.argmax(out,-1)[0]
pad_h,pad_w=pad_info
unpad_h=label_mask.shape[0]-pad_h
unpad_w=label_mask.shape[1]-pad_w
label_mask_unpadded=label_mask[:unpad_h,:unpad_w]
orig_h,orig_w=orig_img_size
label_mask_resized=cv2.resize(label_mask_unpadded,(orig_w,orig_h),interpolation=cv2.INTER_NEAREST)
returnlabel_mask_resized


#Getinfoaboutoriginalimage
image_data=np.array(img)
orig_img_shape=image_data.shape

#Specifybackgroundcolorforreplacement
BG_COLOR=(192,192,192)

#BlurimageforbackgraundblurringscenariousingGaussianBlur
blurred_image=cv2.GaussianBlur(image_data,(55,55),0)

#Postprocessoutput
postprocessed_mask=postprocess_mask(out,pad_info,orig_img_shape[:2])

#Getcoloredsegmentationmap
output_mask=segmentation_map_to_image(postprocessed_mask,SegmentationLabels.get_colormap())

#Replacebackgroundonoriginalimage
#fillimagewithsolidbackgroundcolor
bg_image=np.full(orig_img_shape,BG_COLOR,dtype=np.uint8)

#defineconditionmaskforseparationbackgroundandforeground
condition=np.stack((postprocessed_mask,)*3,axis=-1)>0
#replacebackgroundwithsolidcolor
output_image=np.where(condition,image_data,bg_image)
#replacebackgroundwithblurredimagecopy
output_blurred_image=np.where(condition,image_data,blurred_image)

Visualizeobtainedresult

..code::ipython3

importmatplotlib.pyplotasplt

titles=["Originalimage","Portraitmask","Removedbackground","Blurredbackground"]
images=[image_data,output_mask,output_image,output_blurred_image]
figsize=(16,16)
fig,axs=plt.subplots(2,2,figsize=figsize,sharex="all",sharey="all")
fig.patch.set_facecolor("white")
list_axes=list(axs.flat)
fori,ainenumerate(list_axes):
a.set_xticklabels([])
a.set_yticklabels([])
a.get_xaxis().set_visible(False)
a.get_yaxis().set_visible(False)
a.grid(False)
a.imshow(images[i].astype(np.uint8))
a.set_title(titles[i])
fig.subplots_adjust(wspace=0.0,hspace=-0.8)
fig.tight_layout()



..image::tflite-selfie-segmentation-with-output_files/tflite-selfie-segmentation-with-output_25_0.png


Interactivebackgroundblurringdemoonvideo
---------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Thefollowingcoderunsmodelinferenceonavideo:

..code::ipython3

importcollections
importtime
fromIPythonimportdisplay
fromtypingimportUnion

fromnotebook_utilsimportVideoPlayer


#Mainprocessingfunctiontorunbackgroundblurring
defrun_background_blurring(
source:Union[str,int]=0,
flip:bool=False,
use_popup:bool=False,
skip_first_frames:int=0,
model:ov.Model=ov_model,
device:str="CPU",
):
"""
Functionforrunningbackgroundblurringinferenceonvideo
Parameters:
source(Union[str,int],*optional*,0):inputvideosource,itcanbepathorlinkonvideofileorwebcameraid.
flip(bool,*optional*,False):flipoutputvideo,usedforfront-cameravideoprocessing
use_popup(bool,*optional*,False):usepopupwindowforavoidflickering
skip_first_frames(int,*optional*,0):specifiednumberofframeswillbeskippedinvideoprocessing
model(ov.Model):OpenVINOmodelforinference
device(str):inferencedevice
Returns:
None
"""
player=None
compiled_model=core.compile_model(model,device)
try:
#Createavideoplayertoplaywithtargetfps.
player=VideoPlayer(source=source,flip=flip,fps=30,skip_first_frames=skip_first_frames)
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
#Gettheresults.
input_image,pad_info=resize_and_pad(frame,256,256)
normalized_img=np.expand_dims(input_image.astype(np.float32)/255,0)

start_time=time.time()
#modelexpectsRGBimage,whilevideocapturinginBGR
segmentation_mask=compiled_model(normalized_img[:,:,:,::-1])[0]
stop_time=time.time()
blurred_image=cv2.GaussianBlur(frame,(55,55),0)
postprocessed_mask=postprocess_mask(segmentation_mask,pad_info,frame.shape[:2])
condition=np.stack((postprocessed_mask,)*3,axis=-1)>0
frame=np.where(condition,frame,blurred_image)
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
color=(255,0,0),
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

RunLiveBackgroundBlurring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Useawebcamasthevideoinput.Bydefault,theprimarywebcamisset
with \``source=0``.Ifyouhavemultiplewebcams,eachonewillbe
assignedaconsecutivenumberstartingat0.Set \``flip=True`` when
usingafront-facingcamera.Somewebbrowsers,especiallyMozilla
Firefox,maycauseflickering.Ifyouexperienceflickering,
set \``use_popup=True``.

**NOTE**:Tousethisnotebookwithawebcam,youneedtorunthe
notebookonacomputerwithawebcam.Ifyourunthenotebookona
remoteserver(forexample,inBinderorGoogleColabservice),the
webcamwillnotwork.Bydefault,thelowercellwillrunmodel
inferenceonavideofile.Ifyouwanttotrytoliveinferenceon
yourwebcamset``WEBCAM_INFERENCE=True``

..code::ipython3

WEBCAM_INFERENCE=False

ifWEBCAM_INFERENCE:
VIDEO_SOURCE=0#Webcam
else:
VIDEO_SOURCE="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/CEO%20Pat%20Gelsinger%20on%20Leading%20Intel.mp4"

Selectdeviceforinference:

..code::ipython3

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



Run:

..code::ipython3

run_background_blurring(source=VIDEO_SOURCE,device=device.value)



..image::tflite-selfie-segmentation-with-output_files/tflite-selfie-segmentation-with-output_33_0.png


..parsed-literal::

Sourceended

