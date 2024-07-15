HelloImageSegmentation
========================

AverybasicintroductiontousingsegmentationmodelswithOpenVINO™.

Inthistutorial,apre-trained
`road-segmentation-adas-0001<https://docs.openvino.ai/2024/omz_models_model_road_segmentation_adas_0001.html>`__
modelfromthe`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__isused.
ADASstandsforAdvancedDriverAssistanceServices.Themodel
recognizesfourclasses:background,road,curbandmark.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Downloadmodelweights<#download-model-weights>`__
-`Selectinferencedevice<#select-inference-device>`__
-`LoadtheModel<#load-the-model>`__
-`LoadanImage<#load-an-image>`__
-`DoInference<#do-inference>`__
-`PrepareDataforVisualization<#prepare-data-for-visualization>`__
-`Visualizedata<#visualize-data>`__

..code::ipython3

importplatform

#Installrequiredpackages
%pipinstall-q"openvino>=2023.1.0"opencv-pythontqdm

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcv2
importmatplotlib.pyplotasplt
importnumpyasnp
importopenvinoasov

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportsegmentation_map_to_image,download_file

Downloadmodelweights
----------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frompathlibimportPath

base_model_dir=Path("./model").expanduser()

model_name="road-segmentation-adas-0001"
model_xml_name=f"{model_name}.xml"
model_bin_name=f"{model_name}.bin"

model_xml_path=base_model_dir/model_xml_name

ifnotmodel_xml_path.exists():
model_xml_url=(
"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.xml"
)
model_bin_url=(
"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.bin"
)

download_file(model_xml_url,model_xml_name,base_model_dir)
download_file(model_bin_url,model_bin_name,base_model_dir)
else:
print(f"{model_name}alreadydownloadedto{base_model_dir}")



..parsed-literal::

model/road-segmentation-adas-0001.xml:0%||0.00/389k[00:00<?,?B/s]



..parsed-literal::

model/road-segmentation-adas-0001.bin:0%||0.00/720k[00:00<?,?B/s]


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
--------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

core=ov.Core()

model=core.read_model(model=model_xml_path)
compiled_model=core.compile_model(model=model,device_name=device.value)

input_layer_ir=compiled_model.input(0)
output_layer_ir=compiled_model.output(0)

LoadanImage
-------------

`backtotop⬆️<#table-of-contents>`__Asampleimagefromthe
`MapillaryVistas<https://www.mapillary.com/dataset/vistas>`__dataset
isprovided.

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstorage
image_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/empty_road_mapillary.jpg",
directory="data",
)

#ThesegmentationnetworkexpectsimagesinBGRformat.
image=cv2.imread(str(image_filename))

rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_h,image_w,_=image.shape

#N,C,H,W=batchsize,numberofchannels,height,width.
N,C,H,W=input_layer_ir.shape

#OpenCVresizeexpectsthedestinationsizeas(width,height).
resized_image=cv2.resize(image,(W,H))

#Reshapetothenetworkinputshape.
input_image=np.expand_dims(resized_image.transpose(2,0,1),0)
plt.imshow(rgb_image)



..parsed-literal::

data/empty_road_mapillary.jpg:0%||0.00/227k[00:00<?,?B/s]




..parsed-literal::

<matplotlib.image.AxesImageat0x7f866f7dbac0>




..image::hello-segmentation-with-output_files/hello-segmentation-with-output_11_2.png


DoInference
------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Runtheinference.
result=compiled_model([input_image])[output_layer_ir]

#Preparedataforvisualization.
segmentation_mask=np.argmax(result,axis=1)
plt.imshow(segmentation_mask.transpose(1,2,0))




..parsed-literal::

<matplotlib.image.AxesImageat0x7f86340753a0>




..image::hello-segmentation-with-output_files/hello-segmentation-with-output_13_1.png


PrepareDataforVisualization
------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Definecolormap,eachcolorrepresentsaclass.
colormap=np.array([[68,1,84],[48,103,141],[53,183,120],[199,216,52]])

#Definethetransparencyofthesegmentationmaskonthephoto.
alpha=0.3

#Usefunctionfromnotebook_utils.pytotransformmasktoanRGBimage.
mask=segmentation_map_to_image(segmentation_mask,colormap)
resized_mask=cv2.resize(mask,(image_w,image_h))

#Createanimagewithmask.
image_with_mask=cv2.addWeighted(resized_mask,alpha,rgb_image,1-alpha,0)

Visualizedata
--------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Definetitleswithimages.
data={"BasePhoto":rgb_image,"Segmentation":mask,"MaskedPhoto":image_with_mask}

#Createasubplottovisualizeimages.
fig,axs=plt.subplots(1,len(data.items()),figsize=(15,10))

#Fillthesubplot.
forax,(name,image)inzip(axs,data.items()):
ax.axis("off")
ax.set_title(name)
ax.imshow(image)

#Displayanimage.
plt.show(fig)



..image::hello-segmentation-with-output_files/hello-segmentation-with-output_17_0.png

