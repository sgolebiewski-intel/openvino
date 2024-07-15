VehicleDetectionAndRecognitionwithOpenVINO™
================================================

Thistutorialdemonstrateshowtousetwopre-trainedmodelsfrom`Open
ModelZoo<https://github.com/openvinotoolkit/open_model_zoo>`__:
`vehicle-detection-0200<https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-detection-0200>`__
forobjectdetectionand
`vehicle-attributes-recognition-barrier-0039<https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0039>`__
forimageclassification.Usingthesemodels,youwilldetectvehicles
fromrawimagesandrecognizeattributesofdetectedvehicles.
|flowchart|

Asaresult,youcanget:

..figure::https://user-images.githubusercontent.com/47499836/157867020-99738b30-62ca-44e2-8d9e-caf13fb724ed.png
:alt:result

result

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`DownloadModels<#download-models>`__
-`LoadModels<#load-models>`__

-`Getattributesfrommodel<#get-attributes-from-model>`__
-`Helperfunction<#helper-function>`__
-`Readanddisplayatestimage<#read-and-display-a-test-image>`__

-`UsetheDetectionModeltoDetect
Vehicles<#use-the-detection-model-to-detect-vehicles>`__

-`DetectionProcessing<#detection-processing>`__
-`Recognizevehicleattributes<#recognize-vehicle-attributes>`__

-`Recognitionprocessing<#recognition-processing>`__

-`Combinetwomodels<#combine-two-models>`__

..|flowchart|image::https://user-images.githubusercontent.com/47499836/157867076-9e997781-f9ef-45f6-9a51-b515bbf41048.png

Imports
-------

`backtotop⬆️<#table-of-contents>`__

Importtherequiredmodules.

..code::ipython3

importplatform

%pipinstall-q"openvino>=2023.1.0"opencv-pythontqdm

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importos
frompathlibimportPath
fromtypingimportTuple

importcv2
importnumpyasnp
importmatplotlib.pyplotasplt
importopenvinoasov

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

importnotebook_utilsasutils

DownloadModels
---------------

`backtotop⬆️<#table-of-contents>`__

Downloadpretrainedmodelsfrom
https://storage.openvinotoolkit.org/repositories/open_model_zoo.Ifthe
modelisalreadydownloaded,thisstepisskipped.

**Note**:Tochangethemodel,replacethenameofthemodelinthe
codebelow,forexampleto``"vehicle-detection-0201"``or
``"vehicle-detection-0202"``.Keepinmindthattheysupport
differentimageinputsizesindetection.Also,youcanchangethe
recognitionmodelto
``"vehicle-attributes-recognition-barrier-0042"``.Theyaretrained
fromdifferentdeeplearningframes.Therefore,ifyouwanttochange
theprecision,youneedtomodifytheprecisionvaluein``"FP32"``,
``"FP16"``,and``"FP16-INT8"``.Adifferenttypehasadifferent
modelsizeandaprecisionvalue.

..code::ipython3

#Adirectorywherethemodelwillbedownloaded.
base_model_dir=Path("model")
#ThenameofthemodelfromOpenModelZoo.
detection_model_name="vehicle-detection-0200"
recognition_model_name="vehicle-attributes-recognition-barrier-0039"
#Selectedprecision(FP32,FP16,FP16-INT8)
precision="FP32"

base_model_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"

#Checkifthemodelexists.
detection_model_url=f"{base_model_url}/{detection_model_name}/{precision}/{detection_model_name}.xml"
recognition_model_url=f"{base_model_url}/{recognition_model_name}/{precision}/{recognition_model_name}.xml"
detection_model_path=(base_model_dir/detection_model_name).with_suffix(".xml")
recognition_model_path=(base_model_dir/recognition_model_name).with_suffix(".xml")

#Downloadthedetectionmodel.
ifnotdetection_model_path.exists():
utils.download_file(detection_model_url,detection_model_name+".xml",base_model_dir)
utils.download_file(
detection_model_url.replace(".xml",".bin"),
detection_model_name+".bin",
base_model_dir,
)
#Downloadtherecognitionmodel.
ifnotos.path.exists(recognition_model_path):
utils.download_file(recognition_model_url,recognition_model_name+".xml",base_model_dir)
utils.download_file(
recognition_model_url.replace(".xml",".bin"),
recognition_model_name+".bin",
base_model_dir,
)



..parsed-literal::

model/vehicle-detection-0200.xml:0%||0.00/181k[00:00<?,?B/s]



..parsed-literal::

model/vehicle-detection-0200.bin:0%||0.00/6.93M[00:00<?,?B/s]



..parsed-literal::

model/vehicle-attributes-recognition-barrier-0039.xml:0%||0.00/33.7k[00:00<?,?B/s]



..parsed-literal::

model/vehicle-attributes-recognition-barrier-0039.bin:0%||0.00/2.39M[00:00<?,?B/s]


LoadModels
-----------

`backtotop⬆️<#table-of-contents>`__

Thistutorialrequiresadetectionmodelandarecognitionmodel.After
downloadingthemodels,initializeOpenVINORuntime,anduse
``read_model()``toreadnetworkarchitectureandweightsfrom``*.xml``
and``*.bin``files.Then,compileitwith``compile_model()``tothe
specifieddevice.

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

#InitializeOpenVINORuntimeruntime.
core=ov.Core()


defmodel_init(model_path:str)->Tuple:
"""
Readthenetworkandweightsfromfile,loadthe
modelontheCPUandgetinputandoutputnamesofnodes

:param:model:modelarchitecturepath*.xml
:retuns:
input_key:Inputnodenetwork
output_key:Outputnodenetwork
exec_net:Encodermodelnetwork
net:Modelnetwork
"""

#Readthenetworkandcorrespondingweightsfromafile.
model=core.read_model(model=model_path)
compiled_model=core.compile_model(model=model,device_name=device.value)
#Getinputandoutputnamesofnodes.
input_keys=compiled_model.input(0)
output_keys=compiled_model.output(0)
returninput_keys,output_keys,compiled_model

Getattributesfrommodel
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Use``input_keys.shape``togetdatashapes.

..code::ipython3

#de->detection
#re->recognition
#Detectionmodelinitialization.
input_key_de,output_keys_de,compiled_model_de=model_init(detection_model_path)
#Recognitionmodelinitialization.
input_key_re,output_keys_re,compiled_model_re=model_init(recognition_model_path)

#Getinputsize-Detection.
height_de,width_de=list(input_key_de.shape)[2:]
#Getinputsize-Recognition.
height_re,width_re=list(input_key_re.shape)[2:]

Helperfunction
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The``plt_show()``functionisusedtoshowimage.

..code::ipython3

defplt_show(raw_image):
"""
Usematplottoshowimageinline
raw_image:inputimage

:param:raw_image:imagearray
"""
plt.figure(figsize=(10,6))
plt.axis("off")
plt.imshow(raw_image)

Readanddisplayatestimage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Theinputshapeofdetectionmodelis``[1,3,256,256]``.Therefore,
youneedtoresizetheimageto``256x256``,andexpandthebatch
channelwith``expand_dims``function.

..code::ipython3

#Loadanimage.
url="https://storage.openvinotoolkit.org/data/test_data/images/person-bicycle-car-detection.bmp"
filename="cars.jpg"
directory="data"
image_file=utils.download_file(
url,
filename=filename,
directory=directory,
show_progress=False,
silent=True,
timeout=30,
)
assertPath(image_file).exists()

#Readtheimage.
image_de=cv2.imread("data/cars.jpg")
#Resizeitto[3,256,256].
resized_image_de=cv2.resize(image_de,(width_de,height_de))
#Expandthebatchchannelto[1,3,256,256].
input_image_de=np.expand_dims(resized_image_de.transpose(2,0,1),0)
#Showtheimage.
plt_show(cv2.cvtColor(image_de,cv2.COLOR_BGR2RGB))



..image::vehicle-detection-and-recognition-with-output_files/vehicle-detection-and-recognition-with-output_14_0.png


UsetheDetectionModeltoDetectVehicles
------------------------------------------

`backtotop⬆️<#table-of-contents>`__

..figure::https://user-images.githubusercontent.com/47499836/157867076-9e997781-f9ef-45f6-9a51-b515bbf41048.png
:alt:pipline

pipline

Asshownintheflowchart,imagesofindividualvehiclesaresenttothe
recognitionmodel.First,use``infer``functiontogettheresult.

Thedetectionmodeloutputhastheformat
``[image_id,label,conf,x_min,y_min,x_max,y_max]``,where:

-``image_id``-IDoftheimageinthebatch
-``label``-predictedclassID(0-vehicle)
-``conf``-confidenceforthepredictedclass
-``(x_min,y_min)``-coordinatesofthetopleftboundingboxcorner
-``(x_max,y_max)``-coordinatesofthebottomrightboundingbox
corner

Deleteunuseddimsandfilteroutresultsthatarenotused.

..code::ipython3

#Runinference.
boxes=compiled_model_de([input_image_de])[output_keys_de]
#Deletethedimof0,1.
boxes=np.squeeze(boxes,(0,1))
#Removezeroonlyboxes.
boxes=boxes[~np.all(boxes==0,axis=1)]

DetectionProcessing
~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Withthefunctionbelow,youchangetheratiototherealpositionin
theimageandfilteroutlow-confidenceresults.

..code::ipython3

defcrop_images(bgr_image,resized_image,boxes,threshold=0.6)->np.ndarray:
"""
Useboundingboxesfromdetectionmodeltofindtheabsolutecarposition

:param:bgr_image:rawimage
:param:resized_image:resizedimage
:param:boxes:detectionmodelreturnsrectangleposition
:param:threshold:confidencethreshold
:returns:car_position:car'sabsoluteposition
"""
#Fetchimageshapestocalculateratio
(real_y,real_x),(resized_y,resized_x)=(
bgr_image.shape[:2],
resized_image.shape[:2],
)
ratio_x,ratio_y=real_x/resized_x,real_y/resized_y

#Findtheboxesratio
boxes=boxes[:,2:]
#Storethevehicle'sposition
car_position=[]
#Iteratethroughnon-zeroboxes
forboxinboxes:
#Pickconfidencefactorfromlastplaceinarray
conf=box[0]
ifconf>threshold:
#Convertfloattointandmultiplycornerpositionofeachboxbyxandyratio
#Incasethatboundingboxisfoundatthetopoftheimage,
#upperboxbarshouldbepositionedalittlebitlowertomakeitvisibleonimage
(x_min,y_min,x_max,y_max)=[
(int(max(corner_position*ratio_y*resized_y,10))ifidx%2elseint(corner_position*ratio_x*resized_x))
foridx,corner_positioninenumerate(box[1:])
]

car_position.append([x_min,y_min,x_max,y_max])

returncar_position

..code::ipython3

#Findthepositionofacar.
car_position=crop_images(image_de,resized_image_de,boxes)

Recognizevehicleattributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Selectoneofthedetectedboxes.Then,croptoanareacontaininga
vehicletotestwiththerecognitionmodel.Again,youneedtoresize
theinputimageandruninference.

..code::ipython3

#Selectavehicletorecognize.
pos=car_position[0]
#Croptheimagewith[y_min:y_max,x_min:x_max].
test_car=image_de[pos[1]:pos[3],pos[0]:pos[2]]
#Resizetheimagetoinput_size.
resized_image_re=cv2.resize(test_car,(width_re,height_re))
input_image_re=np.expand_dims(resized_image_re.transpose(2,0,1),0)
plt_show(cv2.cvtColor(resized_image_re,cv2.COLOR_BGR2RGB))



..image::vehicle-detection-and-recognition-with-output_files/vehicle-detection-and-recognition-with-output_21_0.png


Recognitionprocessing
''''''''''''''''''''''

`backtotop⬆️<#table-of-contents>`__

Theresultcontainscolorsofthevehicles(white,gray,yellow,red,
green,blue,black)andtypesofvehicles(car,bus,truck,van).Next,
youneedtocalculatetheprobabilityofeachattribute.Then,you
determinethemaximumprobabilityastheresult.

..code::ipython3

defvehicle_recognition(compiled_model_re,input_size,raw_image):
"""
Vehicleattributesrecognition,inputasinglevehicle,returnattributes
:param:compiled_model_re:recognitionnet
:param:input_size:recognitioninputsize
:param:raw_image:singlevehicleimage
:returns:attr_color:predictedcolor
attr_type:predictedtype
"""
#Anattributeofavehicle.
colors=["White","Gray","Yellow","Red","Green","Blue","Black"]
types=["Car","Bus","Truck","Van"]

#Resizetheimagetoinputsize.
resized_image_re=cv2.resize(raw_image,input_size)
input_image_re=np.expand_dims(resized_image_re.transpose(2,0,1),0)

#Runinference.
#Predictresult.
predict_colors=compiled_model_re([input_image_re])[compiled_model_re.output(1)]
#Deletethedimof2,3.
predict_colors=np.squeeze(predict_colors,(2,3))
predict_types=compiled_model_re([input_image_re])[compiled_model_re.output(0)]
predict_types=np.squeeze(predict_types,(2,3))

attr_color,attr_type=(
colors[np.argmax(predict_colors)],
types[np.argmax(predict_types)],
)
returnattr_color,attr_type

..code::ipython3

print(f"Attributes:{vehicle_recognition(compiled_model_re,(72,72),test_car)}")


..parsed-literal::

Attributes:('Gray','Car')


Combinetwomodels
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Congratulations!Yousuccessfullyusedadetectionmodeltocropan
imagewithavehicleandrecognizetheattributesofavehicle.

..code::ipython3

defconvert_result_to_image(compiled_model_re,bgr_image,resized_image,boxes,threshold=0.6):
"""
UseDetectionmodelboxestodrawrectanglesandplottheresult

:param:compiled_model_re:recognitionnet
:param:input_key_re:recognitioninputkey
:param:bgr_image:rawimage
:param:resized_image:resizedimage
:param:boxes:detectionmodelreturnsrectangleposition
:param:threshold:confidencethreshold
:returns:rgb_image:processedimage
"""
#Definecolorsforboxesanddescriptions.
colors={"red":(255,0,0),"green":(0,255,0)}

#ConvertthebaseimagefromBGRtoRGBformat.
rgb_image=cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)

#Findpositionsofcars.
car_position=crop_images(image_de,resized_image,boxes)

forx_min,y_min,x_max,y_maxincar_position:
#Runvehiclerecognitioninference.
attr_color,attr_type=vehicle_recognition(compiled_model_re,(72,72),image_de[y_min:y_max,x_min:x_max])

#Closethewindowwithavehicle.
plt.close()

#Drawaboundingboxbasedonposition.
#Parametersinthe`rectangle`functionare:image,start_point,end_point,color,thickness.
rgb_image=cv2.rectangle(rgb_image,(x_min,y_min),(x_max,y_max),colors["red"],2)

#Printtheattributesofavehicle.
#Parametersinthe`putText`functionare:img,text,org,fontFace,fontScale,color,thickness,lineType.
rgb_image=cv2.putText(
rgb_image,
f"{attr_color}{attr_type}",
(x_min,y_min-10),
cv2.FONT_HERSHEY_SIMPLEX,
2,
colors["green"],
10,
cv2.LINE_AA,
)

returnrgb_image

..code::ipython3

plt_show(convert_result_to_image(compiled_model_re,image_de,resized_image_de,boxes))



..image::vehicle-detection-and-recognition-with-output_files/vehicle-detection-and-recognition-with-output_27_0.png

