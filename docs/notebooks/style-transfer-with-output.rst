StyleTransferwithOpenVINO™
=============================

ThisnotebookdemonstratesstyletransferwithOpenVINO,usingtheStyle
TransferModelsfrom`ONNXModel
Repository<https://github.com/onnx/models>`__.Specifically,`Fast
NeuralStyle
Transfer<https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style>`__
model,whichisdesignedtomixthecontentofanimagewiththestyle
ofanotherimage.

..figure::https://user-images.githubusercontent.com/109281183/208703143-049f712d-2777-437c-8172-597ef7d53fc3.gif
:alt:styletransfer

styletransfer

Thisnotebookusesfivepre-trainedmodels,forthefollowingstyles:
Mosaic,RainPrincess,Candy,UdnieandPointilism.Themodelsarefrom
`ONNXModelRepository<https://github.com/onnx/models>`__andarebased
ontheresearchpaper`PerceptualLossesforReal-TimeStyleTransfer
andSuper-Resolution<https://arxiv.org/abs/1603.08155>`__alongwith
`InstanceNormalization<https://arxiv.org/abs/1607.08022>`__.Final
partofthisnotebookshowsliveinferenceresultsfromawebcam.
Additionally,youcanalsouploadavideofile.

**NOTE**:Ifyouhaveawebcamonyourcomputer,youcanseelive
resultsstreaminginthenotebook.Ifyourunthenotebookona
server,thewebcamwillnotworkbutyoucanruninference,usinga
videofile.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Preparation<#preparation>`__

-`Installrequirements<#install-requirements>`__
-`Imports<#imports>`__

-`TheModel<#the-model>`__

-`DownloadtheModel<#download-the-model>`__
-`ConvertONNXModeltoOpenVINOIR
Format<#convert-onnx-model-to-openvino-ir-format>`__
-`LoadtheModel<#load-the-model>`__
-`Preprocesstheimage<#preprocess-the-image>`__
-`Helperfunctiontopostprocessthestylized
image<#helper-function-to-postprocess-the-stylized-image>`__
-`MainProcessingFunction<#main-processing-function>`__
-`RunStyleTransfer<#run-style-transfer>`__

-`References<#references>`__

Preparation
-----------

`backtotop⬆️<#table-of-contents>`__

Installrequirements
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"openvino>=2023.1.0"
%pipinstall-qopencv-pythonrequeststqdm

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.




..parsed-literal::

23215



Imports
~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcollections
importtime

importcv2
importnumpyasnp
frompathlibimportPath
importipywidgetsaswidgets
fromIPython.displayimportdisplay,clear_output,Image
importopenvinoasov

importnotebook_utilsasutils

Selectoneofthestylesbelow:Mosaic,RainPrincess,Candy,Udnie,and
Pointilismtodothestyletransfer.

..code::ipython3

#Optiontoselectdifferentstylesusingadropdown
style_dropdown=widgets.Dropdown(
options=["MOSAIC","RAIN-PRINCESS","CANDY","UDNIE","POINTILISM"],
value="MOSAIC",#Setthedefaultvalue
description="SelectStyle:",
disabled=False,
style={"description_width":"initial"},#Adjustthewidthasneeded
)


#Functiontohandlechangesindropdownandprinttheselectedstyle
defprint_style(change):
ifchange["type"]=="change"andchange["name"]=="value":
print(f"Selectedstyle{change['new']}")


#Observechangesinthedropdownvalue
style_dropdown.observe(print_style,names="value")

#Displaythedropdown
display(style_dropdown)



..parsed-literal::

Dropdown(description='SelectStyle:',options=('MOSAIC','RAIN-PRINCESS','CANDY','UDNIE','POINTILISM'),sty…


TheModel
---------

`backtotop⬆️<#table-of-contents>`__

DownloadtheModel
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thestyletransfermodel,selectedinthepreviousstep,willbe
downloadedto``model_path``ifyouhavenotalreadydownloadedit.The
modelsareprovidedbytheONNXModelZooin``.onnx``format,which
meansitcouldbeusedwithOpenVINOdirectly.However,thisnotebook
willalsoshowhowyoucanusetheConversionAPItoconvertONNXto
OpenVINOIntermediateRepresentation(IR)with``FP16``precision.

..code::ipython3

#DirectorytodownloadthemodelfromONNXmodelzoo
base_model_dir="model"
base_url="https://github.com/onnx/models/raw/69d69010b7ed6ba9438c392943d2715026792d40/archive/vision/style_transfer/fast_neural_style/model"

#SelectedONNXmodelwillbedownloadedinthepath
model_path=Path(f"{style_dropdown.value.lower()}-9.onnx")

style_url=f"{base_url}/{model_path}"
utils.download_file(style_url,directory=base_model_dir)



..parsed-literal::

model/mosaic-9.onnx:0%||0.00/6.42M[00:00<?,?B/s]




..parsed-literal::

PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/style-transfer-webcam/model/mosaic-9.onnx')



ConvertONNXModeltoOpenVINOIRFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Inthenextstep,youwillconverttheONNXmodeltoOpenVINOIRformat
with``FP16``precision.WhileONNXmodelsaredirectlysupportedby
OpenVINOruntime,itcanbeusefultoconvertthemtoIRformattotake
advantageofOpenVINOoptimizationtoolsandfeatures.The
``ov.convert_model``PythonfunctionofmodelconversionAPIcanbe
used.Theconvertedmodelissavedtothemodeldirectory.Thefunction
returnsinstanceofOpenVINOModelclass,whichisreadytousein
PythoninterfacebutcanalsobeserializedtoOpenVINOIRformatfor
futureexecution.Ifthemodelhasbeenalreadyconverted,youcanskip
thisstep.

..code::ipython3

#ConstructthecommandformodelconversionAPI.

ov_model=ov.convert_model(f"model/{style_dropdown.value.lower()}-9.onnx")
ov.save_model(ov_model,f"model/{style_dropdown.value.lower()}-9.xml")

..code::ipython3

#ConvertedIRmodelpath
ir_path=Path(f"model/{style_dropdown.value.lower()}-9.xml")
onnx_path=Path(f"model/{model_path}")

LoadtheModel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

BoththeONNXmodel(s)andconvertedIRmodel(s)arestoredinthe
``model``directory.

Onlyafewlinesofcodearerequiredtorunthemodel.First,
initializeOpenVINORuntime.Then,readthenetworkarchitectureand
modelweightsfromthe``.bin``and``.xml``filestocompileforthe
desireddevice.Ifyouselect``GPU``youmayneedtowaitbrieflyfor
ittoload,asthestartuptimeissomewhatlongerthan``CPU``.

ToletOpenVINOautomaticallyselectthebestdeviceforinferencejust
use``AUTO``.Inmostcases,thebestdevicetouseis``GPU``(better
performance,butslightlylongerstartuptime).Youcanselectonefrom
availabledevicesusingdropdownlistbelow.

OpenVINORuntimecanloadONNXmodelsfrom`ONNXModel
Repository<https://github.com/onnx/models>`__directly.Insuchcases,
useONNXpathinsteadofIRmodeltoloadthemodel.Itisrecommended
toloadtheOpenVINOIntermediateRepresentation(IR)modelforthebest
results.

..code::ipython3

#InitializeOpenVINORuntime.
core=ov.Core()

#ReadthenetworkandcorrespondingweightsfromONNXModel.
#model=ie_core.read_model(model=onnx_path)

#ReadthenetworkandcorrespondingweightsfromIRModel.
model=core.read_model(model=ir_path)

..code::ipython3

importipywidgetsaswidgets

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)


#CompilethemodelforCPU(orchangetoGPU,etc.forotherdevices)
#orletOpenVINOselectthebestavailabledevicewithAUTO.
device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



..code::ipython3

compiled_model=core.compile_model(model=model,device_name=device.value)

#Gettheinputandoutputnodes.
input_layer=compiled_model.input(0)
output_layer=compiled_model.output(0)

Inputandoutputlayershavethenamesoftheinputnodeandoutputnode
respectively.For*fast-neural-style-mosaic-onnx*,thereis1inputand
1outputwiththe``(1,3,224,224)``shape.

..code::ipython3

print(input_layer.any_name,output_layer.any_name)
print(input_layer.shape)
print(output_layer.shape)

#Gettheinputsize.
N,C,H,W=list(input_layer.shape)


..parsed-literal::

input1output1
[1,3,224,224]
[1,3,224,224]


Preprocesstheimage
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__Preprocesstheinputimage
beforerunningthemodel.Preparethedimensionsandchannelorderfor
theimagetomatchtheoriginalimagewiththeinputtensor

1.Preprocessaframetoconvertfrom``unit8``to``float32``.
2.Transposethearraytomatchwiththenetworkinputsize

..code::ipython3

#Preprocesstheinputimage.
defpreprocess_images(frame,H,W):
"""
Preprocessinputimagetoalignwithnetworksize

Parameters:
:paramframe:inputframe
:paramH:heightoftheframetostyletransfermodel
:paramW:widthoftheframetostyletransfermodel
:returns:resizedandtransposedframe
"""
image=np.array(frame).astype("float32")
image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
image=cv2.resize(src=image,dsize=(H,W),interpolation=cv2.INTER_AREA)
image=np.transpose(image,[2,0,1])
image=np.expand_dims(image,axis=0)
returnimage

Helperfunctiontopostprocessthestylizedimage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheconvertedIRmodeloutputsaNumPy``float32``arrayofthe`(1,3,
224,
224)<https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/fast-neural-style-mosaic-onnx/README.md>`__
shape.

..code::ipython3

#Postprocesstheresult
defconvert_result_to_image(frame,stylized_image)->np.ndarray:
"""
Postprocessstylizedimageforvisualization

Parameters:
:paramframe:inputframe
:paramstylized_image:stylizedimagewithspecificstyleapplied
:returns:resizedstylizedimageforvisualization
"""
h,w=frame.shape[:2]
stylized_image=stylized_image.squeeze().transpose(1,2,0)
stylized_image=cv2.resize(src=stylized_image,dsize=(w,h),interpolation=cv2.INTER_CUBIC)
stylized_image=np.clip(stylized_image,0,255).astype(np.uint8)
stylized_image=cv2.cvtColor(stylized_image,cv2.COLOR_BGR2RGB)
returnstylized_image

MainProcessingFunction
~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thestyletransferfunctioncanberunindifferentoperatingmodes,
eitherusingawebcamoravideofile.

..code::ipython3

defrun_style_transfer(source=0,flip=False,use_popup=False,skip_first_frames=0):
"""
Mainfunctiontorunthestyleinference:
1.Createavideoplayertoplaywithtargetfps(utils.VideoPlayer).
2.Prepareasetofframesforstyletransfer.
3.RunAIinferenceforstyletransfer.
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
scale=720/max(frame.shape)
ifscale<1:
frame=cv2.resize(
src=frame,
dsize=None,
fx=scale,
fy=scale,
interpolation=cv2.INTER_AREA,
)
#Preprocesstheinputimage.

image=preprocess_images(frame,H,W)

#Measureprocessingtimefortheinputimage.
start_time=time.time()
#Performtheinferencestep.
stylized_image=compiled_model([image])[output_layer]
stop_time=time.time()

#Postprocessingforstylizedimage.
result_image=convert_result_to_image(frame,stylized_image)

processing_times.append(stop_time-start_time)
#Useprocessingtimesfromlast200frames.
iflen(processing_times)>200:
processing_times.popleft()
processing_time_det=np.mean(processing_times)*1000

#Visualizetheresults.
f_height,f_width=frame.shape[:2]
fps=1000/processing_time_det
cv2.putText(
result_image,
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
cv2.imshow(title,result_image)
key=cv2.waitKey(1)
#escape=27
ifkey==27:
break
else:
#Encodenumpyarraytojpg.
_,encoded_img=cv2.imencode(".jpg",result_image,params=[cv2.IMWRITE_JPEG_QUALITY,90])
#CreateanIPythonimage.
i=Image(data=encoded_img)
#Displaytheimageinthisnotebook.
clear_output(wait=True)
display(i)
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

RunStyleTransfer
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,trytoapplythestyletransfermodelusingvideofromyourwebcam
orvideofile.Bydefault,theprimarywebcamissetwith``source=0``.
Ifyouhavemultiplewebcams,eachonewillbeassignedaconsecutive
numberstartingat0.Set``flip=True``whenusingafront-facing
camera.Somewebbrowsers,especiallyMozillaFirefox,maycause
flickering.Ifyouexperienceflickering,set``use_popup=True``.

**NOTE**:Touseawebcam,youmustrunthisJupyternotebookona
computerwithawebcam.Ifyourunitonaserver,youwillnotbe
abletoaccessthewebcam.However,youcanstillperforminference
onavideofileinthefinalstep.

Ifyoudonothaveawebcam,youcanstillrunthisdemowithavideo
file.Any`formatsupportedby
OpenCV<https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__

..code::ipython3

USE_WEBCAM=False

cam_id=0
video_file="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4"

source=cam_idifUSE_WEBCAMelsevideo_file

run_style_transfer(source=source,flip=isinstance(source,int),use_popup=False)



..image::style-transfer-with-output_files/style-transfer-with-output_25_0.png


..parsed-literal::

Sourceended


References
----------

`backtotop⬆️<#table-of-contents>`__

1.`ONNXModelZoo<https://github.com/onnx/models>`__
2.`FastNeuralStyle
Transfer<https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style>`__
3.`FastNeuralStyleMosaicOnnx-OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/fast-neural-style-mosaic-onnx/README.md>`__
