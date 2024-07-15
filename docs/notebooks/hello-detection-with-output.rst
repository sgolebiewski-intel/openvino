HelloObjectDetection
======================

Averybasicintroductiontousingobjectdetectionmodelswith
OpenVINO™.

The
`horizontal-text-detection-0001<https://docs.openvino.ai/2024/omz_models_model_horizontal_text_detection_0001.html>`__
modelfrom`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__isused.It
detectshorizontaltextinimagesandreturnsablobofdatainthe
shapeof``[100,5]``.Eachdetectedtextboxisstoredinthe
``[x_min,y_min,x_max,y_max,conf]``format,wherethe
``(x_min,y_min)``arethecoordinatesofthetopleftboundingbox
corner,``(x_max,y_max)``arethecoordinatesofthebottomright
boundingboxcornerand``conf``istheconfidenceforthepredicted
class.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Downloadmodelweights<#download-model-weights>`__
-`Selectinferencedevice<#select-inference-device>`__
-`LoadtheModel<#load-the-model>`__
-`LoadanImage<#load-an-image>`__
-`DoInference<#do-inference>`__
-`VisualizeResults<#visualize-results>`__

..code::ipython3

#Installopenvinopackage
%pipinstall-q"openvino>=2023.1.0"opencv-pythontqdm


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcv2
importmatplotlib.pyplotasplt
importnumpyasnp
importopenvinoasov
frompathlibimportPath

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)

fromnotebook_utilsimportdownload_file

Downloadmodelweights
----------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

base_model_dir=Path("./model").expanduser()

model_name="horizontal-text-detection-0001"
model_xml_name=f"{model_name}.xml"
model_bin_name=f"{model_name}.bin"

model_xml_path=base_model_dir/model_xml_name
model_bin_path=base_model_dir/model_bin_name

ifnotmodel_xml_path.exists():
model_xml_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.xml"
model_bin_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.bin"

download_file(model_xml_url,model_xml_name,base_model_dir)
download_file(model_bin_url,model_bin_name,base_model_dir)
else:
print(f"{model_name}alreadydownloadedto{base_model_dir}")



..parsed-literal::

model/horizontal-text-detection-0001.xml:0%||0.00/680k[00:00<?,?B/s]



..parsed-literal::

model/horizontal-text-detection-0001.bin:0%||0.00/7.39M[00:00<?,?B/s]


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
output_layer_ir=compiled_model.output("boxes")

LoadanImage
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstorage
image_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/intel_rnb.jpg",
directory="data",
)

#TextdetectionmodelsexpectanimageinBGRformat.
image=cv2.imread(str(image_filename))

#N,C,H,W=batchsize,numberofchannels,height,width.
N,C,H,W=input_layer_ir.shape

#Resizetheimagetomeetnetworkexpectedinputsizes.
resized_image=cv2.resize(image,(W,H))

#Reshapetothenetworkinputshape.
input_image=np.expand_dims(resized_image.transpose(2,0,1),0)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));



..parsed-literal::

data/intel_rnb.jpg:0%||0.00/288k[00:00<?,?B/s]



..image::hello-detection-with-output_files/hello-detection-with-output_11_1.png


DoInference
------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Createaninferencerequest.
boxes=compiled_model([input_image])[output_layer_ir]

#Removezeroonlyboxes.
boxes=boxes[~np.all(boxes==0,axis=1)]

VisualizeResults
-----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Foreachdetection,thedescriptionisinthe[x_min,y_min,x_max,y_max,conf]format:
#TheimagepassedhereisinBGRformatwithchangedwidthandheight.Todisplayitincolorsexpectedbymatplotlib,usecvtColorfunction
defconvert_result_to_image(bgr_image,resized_image,boxes,threshold=0.3,conf_labels=True):
#Definecolorsforboxesanddescriptions.
colors={"red":(255,0,0),"green":(0,255,0)}

#Fetchtheimageshapestocalculatearatio.
(real_y,real_x),(resized_y,resized_x)=(
bgr_image.shape[:2],
resized_image.shape[:2],
)
ratio_x,ratio_y=real_x/resized_x,real_y/resized_y

#ConvertthebaseimagefromBGRtoRGBformat.
rgb_image=cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)

#Iteratethroughnon-zeroboxes.
forboxinboxes:
#Pickaconfidencefactorfromthelastplaceinanarray.
conf=box[-1]
ifconf>threshold:
#Convertfloattointandmultiplycornerpositionofeachboxbyxandyratio.
#Iftheboundingboxisfoundatthetopoftheimage,
#positiontheupperboxbarlittlelowertomakeitvisibleontheimage.
(x_min,y_min,x_max,y_max)=[
(int(max(corner_position*ratio_y,10))ifidx%2elseint(corner_position*ratio_x))foridx,corner_positioninenumerate(box[:-1])
]

#Drawaboxbasedontheposition,parametersinrectanglefunctionare:image,start_point,end_point,color,thickness.
rgb_image=cv2.rectangle(rgb_image,(x_min,y_min),(x_max,y_max),colors["green"],3)

#Addtexttotheimagebasedonpositionandconfidence.
#Parametersintextfunctionare:image,text,bottom-left_corner_textfield,font,font_scale,color,thickness,line_type.
ifconf_labels:
rgb_image=cv2.putText(
rgb_image,
f"{conf:.2f}",
(x_min,y_min-10),
cv2.FONT_HERSHEY_SIMPLEX,
0.8,
colors["red"],
1,
cv2.LINE_AA,
)

returnrgb_image

..code::ipython3

plt.figure(figsize=(10,6))
plt.axis("off")
plt.imshow(convert_result_to_image(image,resized_image,boxes,conf_labels=False));



..image::hello-detection-with-output_files/hello-detection-with-output_16_0.png

