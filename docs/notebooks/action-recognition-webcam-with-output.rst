HumanActionRecognitionwithOpenVINO™
=======================================

ThisnotebookdemonstrateslivehumanactionrecognitionwithOpenVINO,
usingthe`ActionRecognition
Models<https://docs.openvino.ai/2020.2/usergroup13.html>`__from`Open
ModelZoo<https://github.com/openvinotoolkit/open_model_zoo>`__,
specificallyan
`Encoder<https://docs.openvino.ai/2020.2/_models_intel_action_recognition_0001_encoder_description_action_recognition_0001_encoder.html>`__
anda
`Decoder<https://docs.openvino.ai/2020.2/_models_intel_action_recognition_0001_decoder_description_action_recognition_0001_decoder.html>`__.
Bothmodelscreateasequencetosequence(``"seq2seq"``)[1]systemto
identifythehumanactivitiesfor`Kinetics-400
dataset<https://deepmind.com/research/open-source/kinetics>`__.The
modelsusetheVideoTransformerapproachwithResNet34encoder[2].The
notebookshowshowtocreatethefollowingpipeline:

Finalpartofthisnotebookshowsliveinferenceresultsfromawebcam.
Additionally,youcanalsouploadavideofile.

**NOTE**:Touseawebcam,youmustrunthisJupyternotebookona
computerwithawebcam.Ifyourunonaserver,thewebcamwillnot
work.However,youcanstilldoinferenceonavideointhefinalstep.

--------------

[1]seq2seq:Deeplearningmodelsthattakeasequenceofitemstothe
inputandoutput.Inthiscase,input:videoframes,output:actions
sequence.This``"seq2seq"``iscomposedofanencoderandadecoder.
Theencodercaptures``"context"``oftheinputstobeanalyzedbythe
decoder,andfinallygetsthehumanactionandconfidence.

[2]`Video
Transformer<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>`__
and
`ResNet34<https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html>`__.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Themodels<#the-models>`__

-`Downloadthemodels<#download-the-models>`__
-`Loadyourlabels<#load-your-labels>`__
-`Loadthemodels<#load-the-models>`__

-`ModelInitialization
function<#model-initialization-function>`__
-`InitializationforEncoderand
Decoder<#initialization-for-encoder-and-decoder>`__

-`Helperfunctions<#helper-functions>`__
-`AIFunctions<#ai-functions>`__
-`MainProcessingFunction<#main-processing-function>`__
-`RunActionRecognition<#run-action-recognition>`__

..code::ipython3

%pipinstall-q"openvino>=2024.0.0""opencv-python""tqdm"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcollections
importos
importtime
fromtypingimportTuple,List

frompathlibimportPath

importcv2
importnumpyasnp
fromIPythonimportdisplay
importopenvinoasov
fromopenvino.runtime.ie_apiimportCompiledModel

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py","w").write(r.text)
importnotebook_utilsasutils

Themodels
----------

`backtotop⬆️<#table-of-contents>`__

Downloadthemodels
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Usethe``download_ir_model``,afunctionfromthe``notebook_utils``
file.Itautomaticallycreatesadirectorystructureanddownloadsthe
selectedmodel.

Inthiscaseyoucanuse``"action-recognition-0001"``asamodelname,
andthesystemautomaticallydownloadsthetwomodels
``"action-recognition-0001-encoder"``and
``"action-recognition-0001-decoder"``

**NOTE**:Ifyouwanttodownloadanothermodel,suchas
``"driver-action-recognition-adas-0002"``
(``"driver-action-recognition-adas-0002-encoder"``+
``"driver-action-recognition-adas-0002-decoder"``),replacethename
ofthemodelinthecodebelow.Usingamodeloutsidethelistcan
requiredifferentpre-andpost-processing.

..code::ipython3

#Adirectorywherethemodelwillbedownloaded.
base_model_dir="model"
#ThenameofthemodelfromOpenModelZoo.
model_name="action-recognition-0001"
#Selectedprecision(FP32,FP16,FP16-INT8).
precision="FP16"
model_path_decoder=f"model/intel/{model_name}/{model_name}-decoder/{precision}/{model_name}-decoder.xml"
model_path_encoder=f"model/intel/{model_name}/{model_name}-encoder/{precision}/{model_name}-encoder.xml"
encoder_url=f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/temp/{model_name}/{model_name}-encoder/{precision}/{model_name}-encoder.xml"
decoder_url=f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/temp/{model_name}/{model_name}-decoder/{precision}/{model_name}-decoder.xml"

ifnotos.path.exists(model_path_decoder):
utils.download_ir_model(decoder_url,Path(model_path_decoder).parent)
ifnotos.path.exists(model_path_encoder):
utils.download_ir_model(encoder_url,Path(model_path_encoder).parent)



..parsed-literal::

model/intel/action-recognition-0001/action-recognition-0001-decoder/FP16/action-recognition-0001-decoder.bin:…



..parsed-literal::

model/intel/action-recognition-0001/action-recognition-0001-encoder/FP16/action-recognition-0001-encoder.bin:…


Loadyourlabels
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thistutorialuses`Kinetics-400
dataset<https://deepmind.com/research/open-source/kinetics>`__,and
alsoprovidesthetextfileembeddedintothisnotebook.

**NOTE**:Ifyouwanttorun
``"driver-action-recognition-adas-0002"``model,replacethe
``kinetics.txt``fileto``driver_actions.txt``.

..code::ipython3

#Downloadthetextfromtheopenvino_notebooksstorage
vocab_file_path=utils.download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/text/kinetics.txt",
directory="data",
)

withvocab_file_path.open(mode="r")asf:
labels=[line.strip()forlineinf]

print(labels[0:9],np.shape(labels))



..parsed-literal::

data/kinetics.txt:0%||0.00/5.82k[00:00<?,?B/s]


..parsed-literal::

['abseiling','airdrumming','answeringquestions','applauding','applyingcream','archery','armwrestling','arrangingflowers','assemblingcomputer'](400,)


Loadthemodels
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Loadthetwomodelsforthisparticulararchitecture,Encoderand
Decoder.Downloadedmodelsarelocatedinafixedstructure,indicating
avendor,thenameofthemodel,andaprecision.

1.InitializeOpenVINORuntime.
2.Readthenetworkfrom``*.bin``and``*.xml``files(weightsand
architecture).
3.Compilethemodelforspecifieddevice.
4.Getinputandoutputnamesofnodes.

Onlyafewlinesofcodearerequiredtorunthemodel.

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO

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



ModelInitializationfunction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#InitializeOpenVINORuntime.
core=ov.Core()


defmodel_init(model_path:str,device:str)->Tuple:
"""
Readthenetworkandweightsfromafile,loadthe
modelonCPUandgetinputandoutputnamesofnodes

:param:
model:modelarchitecturepath*.xml
device:inferencedevice
:retuns:
compiled_model:Compiledmodel
input_key:Inputnodeformodel
output_key:Outputnodeformodel
"""

#Readthenetworkandcorrespondingweightsfromafile.
model=core.read_model(model=model_path)
#Compilethemodelforspecifieddevice.
compiled_model=core.compile_model(model=model,device_name=device)
#Getinputandoutputnamesofnodes.
input_keys=compiled_model.input(0)
output_keys=compiled_model.output(0)
returninput_keys,output_keys,compiled_model

InitializationforEncoderandDecoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Encoderinitialization
input_key_en,output_keys_en,compiled_model_en=model_init(model_path_encoder,device.value)
#Decoderinitialization
input_key_de,output_keys_de,compiled_model_de=model_init(model_path_decoder,device.value)

#Getinputsize-Encoder.
height_en,width_en=list(input_key_en.shape)[2:]
#Getinputsize-Decoder.
frames2decode=list(input_key_de.shape)[0:][1]

Helperfunctions
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Usethefollowinghelperfunctionsforpreprocessingandpostprocessing
frames:

1.PreprocesstheinputimagebeforerunningtheEncodermodel.
(``center_crop``and``adaptative_resize``)
2.Decodetop-3probabilitiesintolabelnames.(``decode_output``)
3.DrawtheRegionofInterest(ROI)overthevideo.
(``rec_frame_display``)
4.Preparetheframefordisplayinglabelnamesoverthevideo.
(``display_text_fnc``)

..code::ipython3

defcenter_crop(frame:np.ndarray)->np.ndarray:
"""
Centercropsquaredtheoriginalframetostandardizetheinputimagetotheencodermodel

:paramframe:inputframe
:returns:center-crop-squaredframe
"""
img_h,img_w,_=frame.shape
min_dim=min(img_h,img_w)
start_x=int((img_w-min_dim)/2.0)
start_y=int((img_h-min_dim)/2.0)
roi=[start_y,(start_y+min_dim),start_x,(start_x+min_dim)]
returnframe[start_y:(start_y+min_dim),start_x:(start_x+min_dim),...],roi


defadaptive_resize(frame:np.ndarray,size:int)->np.ndarray:
"""
Theframegoingtoberesizedtohaveaheightofsizeorawidthofsize

:paramframe:inputframe
:paramsize:inputsizetoencodermodel
:returns:resizedframe,np.arraytype
"""
h,w,_=frame.shape
scale=size/min(h,w)
w_scaled,h_scaled=int(w*scale),int(h*scale)
ifw_scaled==wandh_scaled==h:
returnframe
returncv2.resize(frame,(w_scaled,h_scaled))


defdecode_output(probs:np.ndarray,labels:np.ndarray,top_k:int=3)->np.ndarray:
"""
Decodestopprobabilitiesintocorrespondinglabelnames

:paramprobs:confidencevectorfor400actions
:paramlabels:listofactions
:paramtop_k:Thekmostprobablepositionsinthelistoflabels
:returns:decoded_labels:Thekmostprobableactionsfromthelabelslist
decoded_top_probs:confidenceforthekmostprobableactions
"""
top_ind=np.argsort(-1*probs)[:top_k]
out_label=np.array(labels)[top_ind.astype(int)]
decoded_labels=[out_label[0][0],out_label[0][1],out_label[0][2]]
top_probs=np.array(probs)[0][top_ind.astype(int)]
decoded_top_probs=[top_probs[0][0],top_probs[0][1],top_probs[0][2]]
returndecoded_labels,decoded_top_probs


defrec_frame_display(frame:np.ndarray,roi)->np.ndarray:
"""
Drawarecframeoveractualframe

:paramframe:inputframe
:paramroi:Regionofinterest,imagesectionprocessedbytheEncoder
:returns:framewithdrawedshape

"""

cv2.line(frame,(roi[2]+3,roi[0]+3),(roi[2]+3,roi[0]+100),(0,200,0),2)
cv2.line(frame,(roi[2]+3,roi[0]+3),(roi[2]+100,roi[0]+3),(0,200,0),2)
cv2.line(frame,(roi[3]-3,roi[1]-3),(roi[3]-3,roi[1]-100),(0,200,0),2)
cv2.line(frame,(roi[3]-3,roi[1]-3),(roi[3]-100,roi[1]-3),(0,200,0),2)
cv2.line(frame,(roi[3]-3,roi[0]+3),(roi[3]-3,roi[0]+100),(0,200,0),2)
cv2.line(frame,(roi[3]-3,roi[0]+3),(roi[3]-100,roi[0]+3),(0,200,0),2)
cv2.line(frame,(roi[2]+3,roi[1]-3),(roi[2]+3,roi[1]-100),(0,200,0),2)
cv2.line(frame,(roi[2]+3,roi[1]-3),(roi[2]+100,roi[1]-3),(0,200,0),2)
#WriteROIoveractualframe
FONT_STYLE=cv2.FONT_HERSHEY_SIMPLEX
org=(roi[2]+3,roi[1]-3)
org2=(roi[2]+2,roi[1]-2)
FONT_SIZE=0.5
FONT_COLOR=(0,200,0)
FONT_COLOR2=(0,0,0)
cv2.putText(frame,"ROI",org2,FONT_STYLE,FONT_SIZE,FONT_COLOR2)
cv2.putText(frame,"ROI",org,FONT_STYLE,FONT_SIZE,FONT_COLOR)
returnframe


defdisplay_text_fnc(frame:np.ndarray,display_text:str,index:int):
"""
Includeatextontheanalyzedframe

:paramframe:inputframe
:paramdisplay_text:texttoaddontheframe
:paramindex:indexlinedoraddingtext

"""
#Configurationfordisplayingimageswithtext.
FONT_COLOR=(255,255,255)
FONT_COLOR2=(0,0,0)
FONT_STYLE=cv2.FONT_HERSHEY_DUPLEX
FONT_SIZE=0.7
TEXT_VERTICAL_INTERVAL=25
TEXT_LEFT_MARGIN=15
#ROIoveractualframe
(processed,roi)=center_crop(frame)
#DrawaROIoveractualframe.
frame=rec_frame_display(frame,roi)
#Putatextoveractualframe.
text_loc=(TEXT_LEFT_MARGIN,TEXT_VERTICAL_INTERVAL*(index+1))
text_loc2=(TEXT_LEFT_MARGIN+1,TEXT_VERTICAL_INTERVAL*(index+1)+1)
cv2.putText(frame,display_text,text_loc2,FONT_STYLE,FONT_SIZE,FONT_COLOR2)
cv2.putText(frame,display_text,text_loc,FONT_STYLE,FONT_SIZE,FONT_COLOR)

AIFunctions
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Followingthepipelineabove,youwillusethenextfunctionsto:

1.PreprocessaframebeforerunningtheEncoder.(``preprocessing``)
2.EncoderInferenceperframe.(``encoder``)
3.Decoderinferencepersetofframes.(``decoder``)
4.NormalizetheDecoderoutputtogetconfidencevaluesperaction
recognitionlabel.(``softmax``)

..code::ipython3

defpreprocessing(frame:np.ndarray,size:int)->np.ndarray:
"""
PreparingframebeforeEncoder.
Theimageshouldbescaledtoitsshortestdimensionat"size"
andcropped,centered,andsquaredsothatbothwidthand
heighthavelengths"size".Theframemustbetransposedfrom
Height-Width-Channels(HWC)toChannels-Height-Width(CHW).

:paramframe:inputframe
:paramsize:inputsizetoencodermodel
:returns:resizedandcroppedframe
"""
#Adaptativeresize
preprocessed=adaptive_resize(frame,size)
#Center_crop
(preprocessed,roi)=center_crop(preprocessed)
#TransposeframeHWC->CHW
preprocessed=preprocessed.transpose((2,0,1))[None,]#HWC->CHW
returnpreprocessed,roi


defencoder(preprocessed:np.ndarray,compiled_model:CompiledModel)->List:
"""
EncoderInferenceperframe.Thisfunctioncallsthenetworkpreviously
configuredfortheencodermodel(compiled_model),extractsthedata
fromtheoutputnode,andappendsitinanarraytobeusedbythedecoder.

:param:preprocessed:preprocessingframe
:param:compiled_model:Encodermodelnetwork
:returns:encoder_output:embeddinglayerthatisappendedwitheacharrivingframe
"""
output_key_en=compiled_model.output(0)

#Getresultsonaction-recognition-0001-encodermodel
infer_result_encoder=compiled_model([preprocessed])[output_key_en]
returninfer_result_encoder


defdecoder(encoder_output:List,compiled_model_de:CompiledModel)->List:
"""
Decoderinferencepersetofframes.Thisfunctionconcatenatestheembeddinglayer
fromstheencoderoutput,transposethearraytomatchwiththedecoderinputsize.
Callsthenetworkpreviouslyconfiguredforthedecodermodel(compiled_model_de),extracts
thelogitsandnormalizethosetogetconfidencevaluesalongspecifiedaxis.
Decodestopprobabilitiesintocorrespondinglabelnames

:param:encoder_output:embeddinglayerfor16frames
:param:compiled_model_de:Decodermodelnetwork
:returns:decoded_labels:Thekmostprobableactionsfromthelabelslist
decoded_top_probs:confidenceforthekmostprobableactions
"""
#Concatenatesample_durationframesinjustonearray
decoder_input=np.concatenate(encoder_output,axis=0)
#OrganizeinputshapevectortotheDecoder(shape:[1x16x512]]
decoder_input=decoder_input.transpose((2,0,1,3))
decoder_input=np.squeeze(decoder_input,axis=3)
output_key_de=compiled_model_de.output(0)
#Getresultsonaction-recognition-0001-decodermodel
result_de=compiled_model_de([decoder_input])[output_key_de]
#Normalizelogitstogetconfidencevaluesalongspecifiedaxis
probs=softmax(result_de-np.max(result_de))
#Decodestopprobabilitiesintocorrespondinglabelnames
decoded_labels,decoded_top_probs=decode_output(probs,labels,top_k=3)
returndecoded_labels,decoded_top_probs


defsoftmax(x:np.ndarray)->np.ndarray:
"""
Normalizeslogitstogetconfidencevaluesalongspecifiedaxis
x:np.array,axis=None
"""
exp=np.exp(x)
returnexp/np.sum(exp,axis=None)

MainProcessingFunction
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Runningactionrecognitionfunctionwillrunindifferentoperations,
eitherawebcamoravideofile.Seethelistofproceduresbelow:

1.Createavideoplayertoplaywithtargetfps
(``utils.VideoPlayer``).
2.Prepareasetofframestobeencoded-decoded.
3.RunAIfunctions
4.Visualizetheresults.

..code::ipython3

defrun_action_recognition(
source:str="0",
flip:bool=True,
use_popup:bool=False,
compiled_model_en:CompiledModel=compiled_model_en,
compiled_model_de:CompiledModel=compiled_model_de,
skip_first_frames:int=0,
):
"""
Usethe"source"webcamorvideofiletorunthecompletepipelineforaction-recognitionproblem
1.Createavideoplayertoplaywithtargetfps
2.Prepareasetofframestobeencoded-decoded
3.PreprocessframebeforeEncoder
4.EncoderInferenceperframe
5.Decoderinferencepersetofframes
6.Visualizetheresults

:param:source:webcam"0"orvideopath
:param:flip:tobeusedbyVideoPlayerfunctionforflippingcaptureimage
:param:use_popup:Falseforshowingencodedframesoverthisnotebook,Trueforcreatingapopupwindow.
:param:skip_first_frames:Numberofframestoskipatthebeginningofthevideo.
:returns:displayvideooverthenotebookorinapopupwindow

"""
size=height_en#Endoderinputsize-FromCell5_9
sample_duration=frames2decode#Decoderinputsize-FromCell5_7
#Selectframespersecondofyoursource.
fps=30
player=None
try:
#Createavideoplayer.
player=utils.VideoPlayer(source,flip=flip,fps=fps,skip_first_frames=skip_first_frames)
#Startcapturing.
player.start()
ifuse_popup:
title="PressESCtoExit"
cv2.namedWindow(title,cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE)

processing_times=collections.deque()
processing_time=0
encoder_output=[]
decoded_labels=[0,0,0]
decoded_top_probs=[0,0,0]
counter=0
#Createatexttemplatetoshowinferenceresultsovervideo.
text_inference_template="InferTime:{Time:.1f}ms,{fps:.1f}FPS"
text_template="{label},{conf:.2f}%"

whileTrue:
counter=counter+1

#Readaframefromthevideostream.
frame=player.next()
ifframeisNone:
print("Sourceended")
break

scale=1280/max(frame.shape)

#Adaptativeresizeforvisualization.
ifscale<1:
frame=cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)

#Selectoneframeeverytwoforprocessingthroughtheencoder.
#After16framesareprocessed,thedecoderwillfindtheaction,
#andthelabelwillbeprintedovertheframes.

ifcounter%2==0:
#PreprocessframebeforeEncoder.
(preprocessed,_)=preprocessing(frame,size)

#Measureprocessingtime.
start_time=time.time()

#EncoderInferenceperframe
encoder_output.append(encoder(preprocessed,compiled_model_en))

#Decoderinferencepersetofframes
#Waitforsampledurationtoworkwithdecodermodel.
iflen(encoder_output)==sample_duration:
decoded_labels,decoded_top_probs=decoder(encoder_output,compiled_model_de)
encoder_output=[]

#Inferencehasfinished.Displaytheresults.
stop_time=time.time()

#Calculateprocessingtime.
processing_times.append(stop_time-start_time)

#Useprocessingtimesfromlast200frames.
iflen(processing_times)>200:
processing_times.popleft()

#Meanprocessingtime[ms]
processing_time=np.mean(processing_times)*1000
fps=1000/processing_time

#Visualizetheresults.
foriinrange(0,3):
display_text=text_template.format(
label=decoded_labels[i],
conf=decoded_top_probs[i]*100,
)
display_text_fnc(frame,display_text,i)

display_text=text_inference_template.format(Time=processing_time,fps=fps)
display_text_fnc(frame,display_text,3)

#Usethisworkaroundifyouexperienceflickering.
ifuse_popup:
cv2.imshow(title,frame)
key=cv2.waitKey(1)
#escape=27
ifkey==27:
break
else:
#Encodenumpyarraytojpg.
_,encoded_img=cv2.imencode(".jpg",frame,params=[cv2.IMWRITE_JPEG_QUALITY,90])
#CreateanIPythonimage.
i=display.Image(data=encoded_img)
#Displaytheimageinthisnotebook.
display.clear_output(wait=True)
display.display(i)

#ctrl-c
exceptKeyboardInterrupt:
print("Interrupted")
#Anydifferenterror
exceptRuntimeErrorase:
print(e)
finally:
ifplayerisnotNone:
#Stopcapturing.
player.stop()
ifuse_popup:
cv2.destroyAllWindows()

RunActionRecognition
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Findouthowthemodelworksinavideofile.`Anyformat
supported<https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
byOpenCVwillwork.Youcanpressthestopbuttonanytimewhilethe
videofileisrunning,anditwillactivatethewebcamforthenext
step.

**NOTE**:Sometimes,thevideocanbecutoffiftherearecorrupted
frames.Inthatcase,youcanconvertit.Ifyouexperienceany
problemswithyourvideo,usethe
`HandBrake<https://handbrake.fr/>`__andselecttheMPEGformat.

ifyouwanttouseawebcameraasaninputsourceforthedemo,please
changethevalueof``USE_WEBCAM``variabletoTrueandspecify
``cam_id``(thedefaultvalueis0,whichcanbedifferentin
multi-camerasystems).

..code::ipython3

USE_WEBCAM=False

cam_id=0
video_file="https://archive.org/serve/ISSVideoResourceLifeOnStation720p/ISS%20Video%20Resource_LifeOnStation_720p.mp4"

source=cam_idifUSE_WEBCAMelsevideo_file
additional_options={"skip_first_frames":600,"flip":False}ifnotUSE_WEBCAMelse{"flip":True}
run_action_recognition(source=source,use_popup=False,**additional_options)



..image::action-recognition-webcam-with-output_files/action-recognition-webcam-with-output_22_0.png


..parsed-literal::

Sourceended

