IndustrialMeterReader
=======================

Thisnotebookshowshowtocreateaindustrialmeterreaderwith
OpenVINORuntime.Weusethepre-trained
`PPYOLOv2<https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyolo>`__
PaddlePaddlemodeland
`DeepLabV3P<https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/deeplabv3p>`__
tobuildupamultipleinferencetaskpipeline:

1.Rundetectionmodeltofindthemeters,andcropthemfromtheorigin
photo.
2.Runsegmentationmodelonthesecroppedmeterstogetthepointerand
scaleinstance.
3.Findthelocationofthepointerinscalemap.

..figure::https://user-images.githubusercontent.com/91237924/166137115-67284fa5-f703-4468-98f4-c43d2c584763.png
:alt:workflow

workflow

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Import<#import>`__
-`PreparetheModelandTest
Image<#prepare-the-model-and-test-image>`__
-`Configuration<#configuration>`__
-`LoadtheModels<#load-the-models>`__
-`DataProcess<#data-process>`__
-`MainFunction<#main-function>`__

-`Initializethemodeland
parameters.<#initialize-the-model-and-parameters->`__
-`Runmeterdetectionmodel<#run-meter-detection-model>`__
-`Runmetersegmentationmodel<#run-meter-segmentation-model>`__
-`Postprocessthemodelsresultandcalculatethefinal
readings<#postprocess-the-models-result-and-calculate-the-final-readings>`__
-`Getthereadingresultonthemeter
picture<#get-the-reading-result-on-the-meter-picture>`__

-`Tryitwithyourmeterphotos!<#try-it-with-your-meter-photos>`__

..code::ipython3

importplatform

#Installopenvinopackage
%pipinstall-q"openvino>=2023.1.0"opencv-pythontqdm

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Import
------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importos
frompathlibimportPath
importnumpyasnp
importmath
importcv2
importtarfile
importmatplotlib.pyplotasplt
importopenvinoasov

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file,segmentation_map_to_image

PreparetheModelandTestImage
--------------------------------

`backtotop⬆️<#table-of-contents>`__DownloadPPYOLOv2and
DeepLabV3Ppre-trainedmodelsfromPaddlePaddlecommunity.

..code::ipython3

MODEL_DIR="model"
DATA_DIR="data"
DET_MODEL_LINK="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/meter-reader/meter_det_model.tar.gz"
SEG_MODEL_LINK="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/meter-reader/meter_seg_model.tar.gz"
DET_FILE_NAME=DET_MODEL_LINK.split("/")[-1]
SEG_FILE_NAME=SEG_MODEL_LINK.split("/")[-1]
IMG_LINK="https://user-images.githubusercontent.com/91237924/170696219-f68699c6-1e82-46bf-aaed-8e2fc3fa5f7b.jpg"
IMG_FILE_NAME=IMG_LINK.split("/")[-1]
IMG_PATH=Path(f"{DATA_DIR}/{IMG_FILE_NAME}")

os.makedirs(MODEL_DIR,exist_ok=True)

download_file(DET_MODEL_LINK,directory=MODEL_DIR,show_progress=True)
file=tarfile.open(f"model/{DET_FILE_NAME}")
res=file.extractall("model")
ifnotres:
print(f'DetectionModelExtractedto"./{MODEL_DIR}".')
else:
print("ErrorExtractingtheDetectionmodel.Pleasecheckthenetwork.")

download_file(SEG_MODEL_LINK,directory=MODEL_DIR,show_progress=True)
file=tarfile.open(f"model/{SEG_FILE_NAME}")
res=file.extractall("model")
ifnotres:
print(f'SegmentationModelExtractedto"./{MODEL_DIR}".')
else:
print("ErrorExtractingtheSegmentationmodel.Pleasecheckthenetwork.")

download_file(IMG_LINK,directory=DATA_DIR,show_progress=True)
ifIMG_PATH.is_file():
print(f'TestImageSavedto"./{DATA_DIR}".')
else:
print("ErrorDownloadingtheTestImage.Pleasecheckthenetwork.")



..parsed-literal::

model/meter_det_model.tar.gz:0%||0.00/192M[00:00<?,?B/s]


..parsed-literal::

DetectionModelExtractedto"./model".



..parsed-literal::

model/meter_seg_model.tar.gz:0%||0.00/94.9M[00:00<?,?B/s]


..parsed-literal::

SegmentationModelExtractedto"./model".



..parsed-literal::

data/170696219-f68699c6-1e82-46bf-aaed-8e2fc3fa5f7b.jpg:0%||0.00/183k[00:00<?,?B/s]


..parsed-literal::

TestImageSavedto"./data".


Configuration
-------------

`backtotop⬆️<#table-of-contents>`__Addparameterconfigurationfor
readingcalculation.

..code::ipython3

METER_SHAPE=[512,512]
CIRCLE_CENTER=[256,256]
CIRCLE_RADIUS=250
PI=math.pi
RECTANGLE_HEIGHT=120
RECTANGLE_WIDTH=1570
TYPE_THRESHOLD=40
COLORMAP=np.array([[28,28,28],[238,44,44],[250,250,250]])

#Thereare2typesofmetersintestimagedatasets
METER_CONFIG=[
{"scale_interval_value":25.0/50.0,"range":25.0,"unit":"(MPa)"},
{"scale_interval_value":1.6/32.0,"range":1.6,"unit":"(MPa)"},
]

SEG_LABEL={"background":0,"pointer":1,"scale":2}

LoadtheModels
---------------

`backtotop⬆️<#table-of-contents>`__Defineacommonclassformodel
loadingandinference

..code::ipython3

#InitializeOpenVINORuntime
core=ov.Core()


classModel:
"""
ThisclassrepresentsaOpenVINOmodelobject.

"""

def__init__(self,model_path,new_shape,device="CPU"):
"""
Initializethemodelobject

Param:
model_path(string):pathofinferencemodel
new_shape(dict):newshapeofmodelinput

"""
self.model=core.read_model(model=model_path)
self.model.reshape(new_shape)
self.compiled_model=core.compile_model(model=self.model,device_name=device)
self.output_layer=self.compiled_model.output(0)

defpredict(self,input_image):
"""
Runinference

Param:
input_image(np.array):inputdata

Retuns:
result(np.array)):modeloutputdata
"""
result=self.compiled_model(input_image)[self.output_layer]
returnresult

DataProcess
------------

`backtotop⬆️<#table-of-contents>`__Includingthepreprocessingand
postprocessingtasksofeachmodel.

..code::ipython3

defdet_preprocess(input_image,target_size):
"""
Preprocessingtheinputdatafordetectiontask

Param:
input_image(np.array):inputdata
size(int):theimagesizerequiredbymodelinputlayer
Retuns:
img.astype(np.array):preprocessedimage

"""
img=cv2.resize(input_image,(target_size,target_size))
img=np.transpose(img,[2,0,1])/255
img=np.expand_dims(img,0)
img_mean=np.array([0.485,0.456,0.406]).reshape((3,1,1))
img_std=np.array([0.229,0.224,0.225]).reshape((3,1,1))
img-=img_mean
img/=img_std
returnimg.astype(np.float32)


deffilter_bboxes(det_results,score_threshold):
"""
Filteroutthedetectionresultswithlowconfidence

Param：
det_results(list[dict]):detectionresults
score_threshold(float)：confidencethreshold

Retuns：
filtered_results(list[dict]):filterdetectionresults

"""
filtered_results=[]
foriinrange(len(det_results)):
ifdet_results[i,1]>score_threshold:
filtered_results.append(det_results[i])
returnfiltered_results


defroi_crop(image,results,scale_x,scale_y):
"""
Croptheareaofdetectedmeteroforiginalimage

Param：
img(np.array)：originalimage。
det_results(list[dict]):detectionresults
scale_x(float):thescalevalueinxaxis
scale_y(float):thescalevalueinyaxis

Retuns：
roi_imgs(list[np.array]):thelistofmeterimages
loc(list[int]):thelistofmeterlocations

"""
roi_imgs=[]
loc=[]
forresultinresults:
bbox=result[2:]
xmin,ymin,xmax,ymax=[
int(bbox[0]*scale_x),
int(bbox[1]*scale_y),
int(bbox[2]*scale_x),
int(bbox[3]*scale_y),
]
sub_img=image[ymin:(ymax+1),xmin:(xmax+1),:]
roi_imgs.append(sub_img)
loc.append([xmin,ymin,xmax,ymax])
returnroi_imgs,loc


defroi_process(input_images,target_size,interp=cv2.INTER_LINEAR):
"""
Preparetheroiimageofdetectionresultsdata
Preprocessingtheinputdataforsegmentationtask

Param：
input_images(list[np.array])：thelistofmeterimages
target_size(list|tuple)：heightandwidthofresizedimage，e.g[heigh,width]
interp(int)：theinterpmethodforimagereszing

Retuns：
img_list(list[np.array])：thelistofprocessedimages
resize_img(list[np.array]):forvisualization

"""
img_list=list()
resize_list=list()
forimgininput_images:
img_shape=img.shape
scale_x=float(target_size[1])/float(img_shape[1])
scale_y=float(target_size[0])/float(img_shape[0])
resize_img=cv2.resize(img,None,None,fx=scale_x,fy=scale_y,interpolation=interp)
resize_list.append(resize_img)
resize_img=resize_img.transpose(2,0,1)/255
img_mean=np.array([0.5,0.5,0.5]).reshape((3,1,1))
img_std=np.array([0.5,0.5,0.5]).reshape((3,1,1))
resize_img-=img_mean
resize_img/=img_std
img_list.append(resize_img)
returnimg_list,resize_list


deferode(seg_results,erode_kernel):
"""
Erodethesegmentationresulttogetthemoreclearinstanceofpointerandscale

Param：
seg_results(list[dict])：segmentationresults
erode_kernel(int):sizeoferode_kernel

Return：
eroded_results(list[dict])：thelabmapoferoded_results

"""
kernel=np.ones((erode_kernel,erode_kernel),np.uint8)
eroded_results=seg_results
foriinrange(len(seg_results)):
eroded_results[i]=cv2.erode(seg_results[i].astype(np.uint8),kernel)
returneroded_results


defcircle_to_rectangle(seg_results):
"""
Switchtheshapeoflabel_mapfromcircletorectangle

Param：
seg_results(list[dict])：segmentationresults

Return：
rectangle_meters(list[np.array])：therectangleoflabelmap

"""
rectangle_meters=list()
fori,seg_resultinenumerate(seg_results):
label_map=seg_result

#Thesizeofrectangle_meterisdeterminedbyRECTANGLE_HEIGHTandRECTANGLE_WIDTH
rectangle_meter=np.zeros((RECTANGLE_HEIGHT,RECTANGLE_WIDTH),dtype=np.uint8)
forrowinrange(RECTANGLE_HEIGHT):
forcolinrange(RECTANGLE_WIDTH):
theta=PI*2*(col+1)/RECTANGLE_WIDTH

#Theradiusofmetercirclewillbemappedtotheheightofrectangleimage
rho=CIRCLE_RADIUS-row-1
y=int(CIRCLE_CENTER[0]+rho*math.cos(theta)+0.5)
x=int(CIRCLE_CENTER[1]-rho*math.sin(theta)+0.5)
rectangle_meter[row,col]=label_map[y,x]
rectangle_meters.append(rectangle_meter)
returnrectangle_meters


defrectangle_to_line(rectangle_meters):
"""
Switchthedimensionofrectanglelabelmapfrom2Dto1D

Param：
rectangle_meters(list[np.array])：2DrectangleOFlabel_map。

Return：
line_scales(list[np.array])：thelistofscalesvalue
line_pointers(list[np.array])：thelistofpointersvalue

"""
line_scales=list()
line_pointers=list()
forrectangle_meterinrectangle_meters:
height,width=rectangle_meter.shape[0:2]
line_scale=np.zeros((width),dtype=np.uint8)
line_pointer=np.zeros((width),dtype=np.uint8)
forcolinrange(width):
forrowinrange(height):
ifrectangle_meter[row,col]==SEG_LABEL["pointer"]:
line_pointer[col]+=1
elifrectangle_meter[row,col]==SEG_LABEL["scale"]:
line_scale[col]+=1
line_scales.append(line_scale)
line_pointers.append(line_pointer)
returnline_scales,line_pointers


defmean_binarization(data_list):
"""
Binarizethedata

Param：
data_list(list[np.array])：inputdata

Return：
binaried_data_list(list[np.array])：outputdata。

"""
batch_size=len(data_list)
binaried_data_list=data_list
foriinrange(batch_size):
mean_data=np.mean(data_list[i])
width=data_list[i].shape[0]
forcolinrange(width):
ifdata_list[i][col]<mean_data:
binaried_data_list[i][col]=0
else:
binaried_data_list[i][col]=1
returnbinaried_data_list


deflocate_scale(line_scales):
"""
Findlocationofcenterofeachscale

Param：
line_scales(list[np.array])：thelistofbinariedscalesvalue

Return：
scale_locations(list[list])：locationofeachscale

"""
batch_size=len(line_scales)
scale_locations=list()
foriinrange(batch_size):
line_scale=line_scales[i]
width=line_scale.shape[0]
find_start=False
one_scale_start=0
one_scale_end=0
locations=list()
forjinrange(width-1):
ifline_scale[j]>0andline_scale[j+1]>0:
ifnotfind_start:
one_scale_start=j
find_start=True
iffind_start:
ifline_scale[j]==0andline_scale[j+1]==0:
one_scale_end=j-1
one_scale_location=(one_scale_start+one_scale_end)/2
locations.append(one_scale_location)
one_scale_start=0
one_scale_end=0
find_start=False
scale_locations.append(locations)
returnscale_locations


deflocate_pointer(line_pointers):
"""
Findlocationofcenterofpointer

Param：
line_scales(list[np.array])：thelistofbinariedpointervalue

Return：
scale_locations(list[list])：locationofpointer

"""
batch_size=len(line_pointers)
pointer_locations=list()
foriinrange(batch_size):
line_pointer=line_pointers[i]
find_start=False
pointer_start=0
pointer_end=0
location=0
width=line_pointer.shape[0]
forjinrange(width-1):
ifline_pointer[j]>0andline_pointer[j+1]>0:
ifnotfind_start:
pointer_start=j
find_start=True
iffind_start:
ifline_pointer[j]==0andline_pointer[j+1]==0:
pointer_end=j-1
location=(pointer_start+pointer_end)/2
find_start=False
break
pointer_locations.append(location)
returnpointer_locations


defget_relative_location(scale_locations,pointer_locations):
"""
Matchlocationofpointerandscales

Param：
scale_locations(list[list])：locationofeachscale
pointer_locations(list[list])：locationofpointer

Return：
pointed_scales(list[dict])：alistofdictwith:
'num_scales':totalnumberofscales
'pointed_scale':predictednumberofscales

"""
pointed_scales=list()
forscale_location,pointer_locationinzip(scale_locations,pointer_locations):
num_scales=len(scale_location)
pointed_scale=-1
ifnum_scales>0:
foriinrange(num_scales-1):
ifscale_location[i]<=pointer_location<scale_location[i+1]:
pointed_scale=i+(pointer_location-scale_location[i])/(scale_location[i+1]-scale_location[i]+1e-05)+1
result={"num_scales":num_scales,"pointed_scale":pointed_scale}
pointed_scales.append(result)
returnpointed_scales


defcalculate_reading(pointed_scales):
"""
Calculatethevalueofmeteraccordingtothetypeofmeter

Param：
pointed_scales(list[list])：predictednumberofscales

Return：
readings(list[float])：thelistofvaluesreadfrommeter

"""
readings=list()
batch_size=len(pointed_scales)
foriinrange(batch_size):
pointed_scale=pointed_scales[i]
#findthetypeofmeteraccordingthetotalnumberofscales
ifpointed_scale["num_scales"]>TYPE_THRESHOLD:
reading=pointed_scale["pointed_scale"]*METER_CONFIG[0]["scale_interval_value"]
else:
reading=pointed_scale["pointed_scale"]*METER_CONFIG[1]["scale_interval_value"]
readings.append(reading)
returnreadings

MainFunction
-------------

`backtotop⬆️<#table-of-contents>`__

Initializethemodelandparameters.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



Thenumberofdetectedmeterfromdetectionnetworkcanbearbitraryin
somescenarios,whichmeansthebatchsizeofsegmentationnetworkinput
isa`dynamic
dimension<https://docs.openvino.ai/2024/openvino-workflow/running-inference/dynamic-shapes.html>`__,
anditshouldbespecifiedas``-1``orthe``ov::Dimension()``instead
ofapositivenumberusedforstaticdimensions.Inthiscase,for
memoryconsumptionoptimization,wecanspecifythelowerand/orupper
boundsofinputbatchsize.

..code::ipython3

img_file=f"{DATA_DIR}/{IMG_FILE_NAME}"
det_model_path=f"{MODEL_DIR}/meter_det_model/model.pdmodel"
det_model_shape={
"image":[1,3,608,608],
"im_shape":[1,2],
"scale_factor":[1,2],
}
seg_model_path=f"{MODEL_DIR}/meter_seg_model/model.pdmodel"
seg_model_shape={"image":[ov.Dimension(1,2),3,512,512]}

erode_kernel=4
score_threshold=0.5
seg_batch_size=2
input_shape=608

#Intializethemodelobjects
detector=Model(det_model_path,det_model_shape,device.value)
segmenter=Model(seg_model_path,seg_model_shape,device.value)

#Visulizeaoriginalinputphoto
image=cv2.imread(img_file)
rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)




..parsed-literal::

<matplotlib.image.AxesImageat0x7f2d2dd77130>




..image::meter-reader-with-output_files/meter-reader-with-output_16_1.png


Runmeterdetectionmodel
~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__Detectthelocationofthe
meterandpreparetheROIimagesforsegmentation.

..code::ipython3

#Preparetheinputdataformeterdetectionmodel
im_shape=np.array([[input_shape,input_shape]]).astype("float32")
scale_factor=np.array([[1,2]]).astype("float32")
input_image=det_preprocess(image,input_shape)
inputs_dict={"image":input_image,"im_shape":im_shape,"scale_factor":scale_factor}

#Runmeterdetectionmodel
det_results=detector.predict(inputs_dict)

#Filterouttheboundingboxwithlowconfidence
filtered_results=filter_bboxes(det_results,score_threshold)

#Preparetheinputdataformetersegmentationmodel
scale_x=image.shape[1]/input_shape*2
scale_y=image.shape[0]/input_shape

#Createtheindividualpictureforeachdetectedmeter
roi_imgs,loc=roi_crop(image,filtered_results,scale_x,scale_y)
roi_imgs,resize_imgs=roi_process(roi_imgs,METER_SHAPE)

#Createthepicturesofdetectionresults
roi_stack=np.hstack(resize_imgs)

ifcv2.imwrite(f"{DATA_DIR}/detection_results.jpg",roi_stack):
print('Thedetectionresultimagehasbeensavedas"detection_results.jpg"indata')
plt.imshow(cv2.cvtColor(roi_stack,cv2.COLOR_BGR2RGB))


..parsed-literal::

Thedetectionresultimagehasbeensavedas"detection_results.jpg"indata



..image::meter-reader-with-output_files/meter-reader-with-output_18_1.png


Runmetersegmentationmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__Gettheresultsofsegmentation
taskondetectedROI.

..code::ipython3

seg_results=list()
mask_list=list()
num_imgs=len(roi_imgs)

#Runmetersegmentationmodelonalldetectedmeters
foriinrange(0,num_imgs,seg_batch_size):
batch=roi_imgs[i:min(num_imgs,i+seg_batch_size)]
seg_result=segmenter.predict({"image":np.array(batch)})
seg_results.extend(seg_result)
results=[]
foriinrange(len(seg_results)):
results.append(np.argmax(seg_results[i],axis=0))
seg_results=erode(results,erode_kernel)

#Createthepicturesofsegmentationresults
foriinrange(len(seg_results)):
mask_list.append(segmentation_map_to_image(seg_results[i],COLORMAP))
mask_stack=np.hstack(mask_list)

ifcv2.imwrite(f"{DATA_DIR}/segmentation_results.jpg",cv2.cvtColor(mask_stack,cv2.COLOR_RGB2BGR)):
print('Thesegmentationresultimagehasbeensavedas"segmentation_results.jpg"indata')
plt.imshow(mask_stack)


..parsed-literal::

Thesegmentationresultimagehasbeensavedas"segmentation_results.jpg"indata



..image::meter-reader-with-output_files/meter-reader-with-output_20_1.png


Postprocessthemodelsresultandcalculatethefinalreadings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__UseOpenCVfunctiontofindthe
locationofthepointerinascalemap.

..code::ipython3

#Findthepointerlocationinscalemapandcalculatethemetersreading
rectangle_meters=circle_to_rectangle(seg_results)
line_scales,line_pointers=rectangle_to_line(rectangle_meters)
binaried_scales=mean_binarization(line_scales)
binaried_pointers=mean_binarization(line_pointers)
scale_locations=locate_scale(binaried_scales)
pointer_locations=locate_pointer(binaried_pointers)
pointed_scales=get_relative_location(scale_locations,pointer_locations)
meter_readings=calculate_reading(pointed_scales)

rectangle_list=list()
#Plottherectanglemeters
foriinrange(len(rectangle_meters)):
rectangle_list.append(segmentation_map_to_image(rectangle_meters[i],COLORMAP))
rectangle_meters_stack=np.hstack(rectangle_list)

ifcv2.imwrite(
f"{DATA_DIR}/rectangle_meters.jpg",
cv2.cvtColor(rectangle_meters_stack,cv2.COLOR_RGB2BGR),
):
print('Therectangle_metersresultimagehasbeensavedas"rectangle_meters.jpg"indata')
plt.imshow(rectangle_meters_stack)


..parsed-literal::

Therectangle_metersresultimagehasbeensavedas"rectangle_meters.jpg"indata



..image::meter-reader-with-output_files/meter-reader-with-output_22_1.png


Getthereadingresultonthemeterpicture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Createafinalresultphotowithreading
foriinrange(len(meter_readings)):
print("Meter{}:{:.3f}".format(i+1,meter_readings[i]))

result_image=image.copy()
foriinrange(len(loc)):
cv2.rectangle(result_image,(loc[i][0],loc[i][1]),(loc[i][2],loc[i][3]),(0,150,0),3)
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.rectangle(
result_image,
(loc[i][0],loc[i][1]),
(loc[i][0]+100,loc[i][1]+40),
(0,150,0),
-1,
)
cv2.putText(
result_image,
"#{:.3f}".format(meter_readings[i]),
(loc[i][0],loc[i][1]+25),
font,
0.8,
(255,255,255),
2,
cv2.LINE_AA,
)
ifcv2.imwrite(f"{DATA_DIR}/reading_results.jpg",result_image):
print('Thereadingresultsimagehasbeensavedas"reading_results.jpg"indata')
plt.imshow(cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB))


..parsed-literal::

Meter1:1.100
Meter2:6.185
Thereadingresultsimagehasbeensavedas"reading_results.jpg"indata



..image::meter-reader-with-output_files/meter-reader-with-output_24_1.png


Tryitwithyourmeterphotos!
------------------------------

`backtotop⬆️<#table-of-contents>`__
