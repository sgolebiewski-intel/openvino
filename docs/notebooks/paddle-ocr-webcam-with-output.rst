PaddleOCRwithOpenVINO™
========================

ThisdemoshowshowtorunPP-OCRmodelonOpenVINOnatively.Insteadof
exportingthePaddlePaddlemodeltoONNXandthenconvertingtothe
OpenVINOIntermediateRepresentation(OpenVINOIR)formatwithmodel
conversionAPI,youcannowreaddirectlyfromthePaddlePaddleModel
withoutanyconversions.
`PaddleOCR<https://github.com/PaddlePaddle/PaddleOCR>`__isan
ultra-lightOCRmodeltrainedwithPaddlePaddledeeplearningframework,
thataimstocreatemultilingualandpracticalOCRtools.

ThePaddleOCRpre-trainedmodelusedinthedemoreferstothe*“Chinese
andEnglishultra-lightweightPP-OCRmodel(9.4M)”*.Moreopensource
pre-trainedmodelscanbedownloadedat`PaddleOCR
GitHub<https://github.com/PaddlePaddle/PaddleOCR>`__or`PaddleOCR
Gitee<https://gitee.com/paddlepaddle/PaddleOCR>`__.Workingpipelineof
thePaddleOCRisasfollows:

**NOTE**:Tousethisnotebookwithawebcam,youneedtorunthe
notebookonacomputerwithawebcam.Ifyourunthenotebookona
server,thewebcamwillnotwork.Youcanstilldoinferenceona
videofile.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__

-`Selectinferencedevice<#select-inference-device>`__
-`ModelsforPaddleOCR<#models-for-paddleocr>`__

-`DownloadtheModelforText
Detection<#download-the-model-for-text-detection>`__
-`LoadtheModelforText
Detection<#load-the-model-for-text-detection>`__
-`DownloadtheModelforText
Recognition<#download-the-model-for-text-recognition>`__
-`LoadtheModelforTextRecognitionwithDynamic
Shape<#load-the-model-for-text-recognition-with-dynamic-shape>`__

-`PreprocessingImageFunctionsforTextDetectionand
Recognition<#preprocessing-image-functions-for-text-detection-and-recognition>`__
-`PostprocessingImageforText
Detection<#postprocessing-image-for-text-detection>`__
-`MainProcessingFunctionfor
PaddleOCR<#main-processing-function-for-paddleocr>`__

-`RunLivePaddleOCRwith
OpenVINO<#run-live-paddleocr-with-openvino>`__

..code::ipython3

%pipinstall-q"openvino>=2023.1.0"
%pipinstall-q"paddlepaddle>=2.5.1"
%pipinstall-q"pyclipper>=1.2.1""shapely>=1.7.1"tqdm


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcv2
importnumpyasnp
importpaddle
importmath
importtime
importcollections
fromPILimportImage
frompathlibimportPath
importtarfile

importopenvinoasov
fromIPythonimportdisplay
importcopy

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
importpre_post_processingasprocessing

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



ModelsforPaddleOCR
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

PaddleOCRincludestwopartsofdeeplearningmodels,textdetectionand
textrecognition.Pre-trainedmodelsusedinthedemoaredownloadedand
storedinthe“model”folder.

Onlyafewlinesofcodearerequiredtorunthemodel.First,
initializetheruntimeforinference.Then,readthenetwork
architectureandmodelweightsfromthe``.pdmodel``and``.pdiparams``
filestoloadtoCPU/GPU.

..code::ipython3

#DefinethefunctiontodownloadtextdetectionandrecognitionmodelsfromPaddleOCRresources.


defrun_model_download(model_url:str,model_file_path:Path)->None:
"""
Downloadpre-trainedmodelsfromPaddleOCRresources

Parameters:
model_url:urllinktopre-trainedmodels
model_file_path:filepathtostorethedownloadedmodel
"""
archive_path=model_file_path.absolute().parent.parent/model_url.split("/")[-1]
ifmodel_file_path.is_file():
print("Modelalreadyexists")
else:
#Downloadthemodelfromtheserver,anduntarit.
print("Downloadingthepre-trainedmodel...Maytakeawhile...")

#Createadirectory.
utils.download_file(model_url,archive_path.name,archive_path.parent)
print("ModelDownloaded")

file=tarfile.open(archive_path)
res=file.extractall(archive_path.parent)
file.close()
ifnotres:
print(f"ModelExtractedto{model_file_path}.")
else:
print("ErrorExtractingthemodel.Pleasecheckthenetwork.")

DownloadtheModelforText**Detection**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Adirectorywherethemodelwillbedownloaded.

det_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/paddle-ocr/ch_PP-OCRv3_det_infer.tar"
det_model_file_path=Path("model/ch_PP-OCRv3_det_infer/inference.pdmodel")

run_model_download(det_model_url,det_model_file_path)


..parsed-literal::

Downloadingthepre-trainedmodel...Maytakeawhile...



..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/paddle-o…


..parsed-literal::

ModelDownloaded
ModelExtractedtomodel/ch_PP-OCRv3_det_infer/inference.pdmodel.


LoadtheModelforText**Detection**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#InitializeOpenVINORuntimefortextdetection.
core=ov.Core()
det_model=core.read_model(model=det_model_file_path)
det_compiled_model=core.compile_model(model=det_model,device_name=device.value)

#Getinputandoutputnodesfortextdetection.
det_input_layer=det_compiled_model.input(0)
det_output_layer=det_compiled_model.output(0)

DownloadtheModelforText**Recognition**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

rec_model_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/paddle-ocr/ch_PP-OCRv3_rec_infer.tar"
rec_model_file_path=Path("model/ch_PP-OCRv3_rec_infer/inference.pdmodel")

run_model_download(rec_model_url,rec_model_file_path)


..parsed-literal::

Downloadingthepre-trainedmodel...Maytakeawhile...



..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/paddle-o…


..parsed-literal::

ModelDownloaded
ModelExtractedtomodel/ch_PP-OCRv3_rec_infer/inference.pdmodel.


LoadtheModelforText**Recognition**withDynamicShape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Inputtotextrecognitionmodelreferstodetectedboundingboxeswith
differentimagesizes,forexample,dynamicinputshapes.Hence:

1.Inputdimensionwithdynamicinputshapesneedstobespecified
beforeloadingtextrecognitionmodel.
2.Dynamicshapeisspecifiedbyassigning-1totheinputdimensionor
bysettingtheupperboundoftheinputdimensionusing,forexample,
``Dimension(1,512)``.

..code::ipython3

#Readthemodelandcorrespondingweightsfromafile.
rec_model=core.read_model(model=rec_model_file_path)

#Assigndynamicshapestoeveryinputlayeronthelastdimension.
forinput_layerinrec_model.inputs:
input_shape=input_layer.partial_shape
input_shape[3]=-1
rec_model.reshape({input_layer:input_shape})

rec_compiled_model=core.compile_model(model=rec_model,device_name="AUTO")

#Getinputandoutputnodes.
rec_input_layer=rec_compiled_model.input(0)
rec_output_layer=rec_compiled_model.output(0)

PreprocessingImageFunctionsforTextDetectionandRecognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Definepreprocessingfunctionsfortextdetectionandrecognition:1.
Preprocessingfortextdetection:resizeandnormalizeinputimages.2.
Preprocessingfortextrecognition:resizeandnormalizedetectedbox
imagestothesamesize(forexample,``(3,32,320)``sizeforimages
withChinesetext)foreasybatchingininference.

..code::ipython3

#Preprocessfortextdetection.
defimage_preprocess(input_image,size):
"""
Preprocessinputimagefortextdetection

Parameters:
input_image:inputimage
size:valuefortheimagetoberesizedfortextdetectionmodel
"""
img=cv2.resize(input_image,(size,size))
img=np.transpose(img,[2,0,1])/255
img=np.expand_dims(img,0)
#NormalizeImage:{mean:[0.485,0.456,0.406],std:[0.229,0.224,0.225],is_scale:True}
img_mean=np.array([0.485,0.456,0.406]).reshape((3,1,1))
img_std=np.array([0.229,0.224,0.225]).reshape((3,1,1))
img-=img_mean
img/=img_std
returnimg.astype(np.float32)

..code::ipython3

#Preprocessfortextrecognition.
defresize_norm_img(img,max_wh_ratio):
"""
Resizeinputimagefortextrecognition

Parameters:
img:boundingboximagefromtextdetection
max_wh_ratio:valuefortheresizingfortextrecognitionmodel
"""
rec_image_shape=[3,48,320]
imgC,imgH,imgW=rec_image_shape
assertimgC==img.shape[2]
character_type="ch"
ifcharacter_type=="ch":
imgW=int((32*max_wh_ratio))
h,w=img.shape[:2]
ratio=w/float(h)
ifmath.ceil(imgH*ratio)>imgW:
resized_w=imgW
else:
resized_w=int(math.ceil(imgH*ratio))
resized_image=cv2.resize(img,(resized_w,imgH))
resized_image=resized_image.astype("float32")
resized_image=resized_image.transpose((2,0,1))/255
resized_image-=0.5
resized_image/=0.5
padding_im=np.zeros((imgC,imgH,imgW),dtype=np.float32)
padding_im[:,:,0:resized_w]=resized_image
returnpadding_im


defprep_for_rec(dt_boxes,frame):
"""
Preprocessingofthedetectedboundingboxesfortextrecognition

Parameters:
dt_boxes:detectedboundingboxesfromtextdetection
frame:originalinputframe
"""
ori_im=frame.copy()
img_crop_list=[]
forbnoinrange(len(dt_boxes)):
tmp_box=copy.deepcopy(dt_boxes[bno])
img_crop=processing.get_rotate_crop_image(ori_im,tmp_box)
img_crop_list.append(img_crop)

img_num=len(img_crop_list)
#Calculatetheaspectratioofalltextbars.
width_list=[]
forimginimg_crop_list:
width_list.append(img.shape[1]/float(img.shape[0]))

#Sortingcanspeeduptherecognitionprocess.
indices=np.argsort(np.array(width_list))
returnimg_crop_list,img_num,indices


defbatch_text_box(img_crop_list,img_num,indices,beg_img_no,batch_num):
"""
Batchfortextrecognition

Parameters:
img_crop_list:processeddetectedboundingboximages
img_num:numberofboundingboxesfromtextdetection
indices:sortingforboundingboxestospeeduptextrecognition
beg_img_no:thebeginningnumberofboundingboxesforeachbatchoftextrecognitioninference
batch_num:numberofimagesforeachbatch
"""
norm_img_batch=[]
max_wh_ratio=0
end_img_no=min(img_num,beg_img_no+batch_num)
forinoinrange(beg_img_no,end_img_no):
h,w=img_crop_list[indices[ino]].shape[0:2]
wh_ratio=w*1.0/h
max_wh_ratio=max(max_wh_ratio,wh_ratio)
forinoinrange(beg_img_no,end_img_no):
norm_img=resize_norm_img(img_crop_list[indices[ino]],max_wh_ratio)
norm_img=norm_img[np.newaxis,:]
norm_img_batch.append(norm_img)

norm_img_batch=np.concatenate(norm_img_batch)
norm_img_batch=norm_img_batch.copy()
returnnorm_img_batch

PostprocessingImageforTextDetection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defpost_processing_detection(frame,det_results):
"""
Postprocesstheresultsfromtextdetectionintoboundingboxes

Parameters:
frame:inputimage
det_results:inferenceresultsfromtextdetectionmodel
"""
ori_im=frame.copy()
data={"image":frame}
data_resize=processing.DetResizeForTest(data)
data_list=[]
keep_keys=["image","shape"]
forkeyinkeep_keys:
data_list.append(data_resize[key])
img,shape_list=data_list

shape_list=np.expand_dims(shape_list,axis=0)
pred=det_results[0]
ifisinstance(pred,paddle.Tensor):
pred=pred.numpy()
segmentation=pred>0.3

boxes_batch=[]
forbatch_indexinrange(pred.shape[0]):
src_h,src_w,ratio_h,ratio_w=shape_list[batch_index]
mask=segmentation[batch_index]
boxes,scores=processing.boxes_from_bitmap(pred[batch_index],mask,src_w,src_h)
boxes_batch.append({"points":boxes})
post_result=boxes_batch
dt_boxes=post_result[0]["points"]
dt_boxes=processing.filter_tag_det_res(dt_boxes,ori_im.shape)
returndt_boxes

MainProcessingFunctionforPaddleOCR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Run``paddleOCR``functionindifferentoperations,eitherawebcamora
videofile.Seethelistofproceduresbelow:

1.Createavideoplayertoplaywithtargetfps
(``utils.VideoPlayer``).
2.Prepareasetofframesfortextdetectionandrecognition.
3.RunAIinferenceforbothtextdetectionandrecognition.
4.Visualizetheresults.

..code::ipython3

#DownloadfontandacharacterdictionaryforprintingOCRresults.
font_path=utils.download_file(
url="https://raw.githubusercontent.com/Halfish/lstm-ctc-ocr/master/fonts/simfang.ttf",
directory="fonts",
)
character_dictionary_path=utils.download_file(
url="https://raw.githubusercontent.com/WenmuZhou/PytorchOCR/master/torchocr/datasets/alphabets/ppocr_keys_v1.txt",
directory="fonts",
)



..parsed-literal::

fonts/simfang.ttf:0%||0.00/10.1M[00:00<?,?B/s]



..parsed-literal::

fonts/ppocr_keys_v1.txt:0%||0.00/17.3k[00:00<?,?B/s]


..code::ipython3

defrun_paddle_ocr(source=0,flip=False,use_popup=False,skip_first_frames=0):
"""
MainfunctiontorunthepaddleOCRinference:
1.Createavideoplayertoplaywithtargetfps(utils.VideoPlayer).
2.Prepareasetofframesfortextdetectionandrecognition.
3.RunAIinferenceforbothtextdetectionandrecognition.
4.Visualizetheresults.

Parameters:
source:Thewebcamnumbertofeedthevideostreamwithprimarywebcamsetto"0",orthevideopath.
flip:TobeusedbyVideoPlayerfunctionforflippingcaptureimage.
use_popup:Falseforshowingencodedframesoverthisnotebook,Trueforcreatingapopupwindow.
skip_first_frames:Numberofframestoskipatthebeginningofthevideo.
"""
#Createavideoplayertoplaywithtargetfps.
player=None
try:
player=utils.VideoPlayer(source=source,flip=flip,fps=30,skip_first_frames=skip_first_frames)
#Startvideocapturing.
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
#Preprocesstheimagefortextdetection.
test_image=image_preprocess(frame,640)

#Measureprocessingtimefortextdetection.
start_time=time.time()
#Performtheinferencestep.
det_results=det_compiled_model([test_image])[det_output_layer]
stop_time=time.time()

#PostprocessingforPaddleDetection.
dt_boxes=post_processing_detection(frame,det_results)

processing_times.append(stop_time-start_time)
#Useprocessingtimesfromlast200frames.
iflen(processing_times)>200:
processing_times.popleft()
processing_time_det=np.mean(processing_times)*1000

#Preprocessdetectionresultsforrecognition.
dt_boxes=processing.sorted_boxes(dt_boxes)
batch_num=6
img_crop_list,img_num,indices=prep_for_rec(dt_boxes,frame)

#Forstoringrecognitionresults,includetwoparts:
#txtsaretherecognizedtextresults,scoresaretherecognitionconfidencelevel.
rec_res=[["",0.0]]*img_num
txts=[]
scores=[]

forbeg_img_noinrange(0,img_num,batch_num):
#Recognitionstartsfromhere.
norm_img_batch=batch_text_box(img_crop_list,img_num,indices,beg_img_no,batch_num)

#Runinferencefortextrecognition.
rec_results=rec_compiled_model([norm_img_batch])[rec_output_layer]

#Postprocessingrecognitionresults.
postprocess_op=processing.build_post_process(processing.postprocess_params)
rec_result=postprocess_op(rec_results)
forrnoinrange(len(rec_result)):
rec_res[indices[beg_img_no+rno]]=rec_result[rno]
ifrec_res:
txts=[rec_res[i][0]foriinrange(len(rec_res))]
scores=[rec_res[i][1]foriinrange(len(rec_res))]

image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
boxes=dt_boxes
#Drawtextrecognitionresultsbesidetheimage.
draw_img=processing.draw_ocr_box_txt(image,boxes,txts,scores,drop_score=0.5,font_path=str(font_path))

#VisualizethePaddleOCRresults.
f_height,f_width=draw_img.shape[:2]
fps=1000/processing_time_det
cv2.putText(
img=draw_img,
text=f"Inferencetime:{processing_time_det:.1f}ms({fps:.1f}FPS)",
org=(20,40),
fontFace=cv2.FONT_HERSHEY_COMPLEX,
fontScale=f_width/1000,
color=(0,0,255),
thickness=1,
lineType=cv2.LINE_AA,
)

#Usethisworkaroundifthereisflickering.
ifuse_popup:
draw_img=cv2.cvtColor(draw_img,cv2.COLOR_RGB2BGR)
cv2.imshow(winname=title,mat=draw_img)
key=cv2.waitKey(1)
#escape=27
ifkey==27:
break
else:
#Encodenumpyarraytojpg.
draw_img=cv2.cvtColor(draw_img,cv2.COLOR_RGB2BGR)
_,encoded_img=cv2.imencode(ext=".jpg",img=draw_img,params=[cv2.IMWRITE_JPEG_QUALITY,100])
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

RunLivePaddleOCRwithOpenVINO
--------------------------------

`backtotop⬆️<#table-of-contents>`__

Useawebcamasthevideoinput.Bydefault,theprimarywebcamisset
with``source=0``.Ifyouhavemultiplewebcams,eachonewillbe
assignedaconsecutivenumberstartingat0.Set``flip=True``when
usingafront-facingcamera.Somewebbrowsers,especiallyMozilla
Firefox,maycauseflickering.Ifyouexperienceflickering,set
``use_popup=True``.

**NOTE**:Popupmodemaynotworkifyourunthisnotebookona
remotecomputer.

Ifyoudonothaveawebcam,youcanstillrunthisdemowithavideo
file.Any`formatsupportedby
OpenCV<https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
willwork.

RunlivePaddleOCR:

..code::ipython3

USE_WEBCAM=False

cam_id=0
video_file="https://raw.githubusercontent.com/yoyowz/classification/master/images/test.mp4"

source=cam_idifUSE_WEBCAMelsevideo_file

run_paddle_ocr(source,flip=False,use_popup=False)



..image::paddle-ocr-webcam-with-output_files/paddle-ocr-webcam-with-output_30_0.png


..parsed-literal::

Sourceended

