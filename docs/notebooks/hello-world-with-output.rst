HelloImageClassification
==========================

ThisbasicintroductiontoOpenVINO™showshowtodoinferencewithan
imageclassificationmodel.

Apre-trained`MobileNetV3
model<https://docs.openvino.ai/2024/omz_models_model_mobilenet_v3_small_1_0_224_tf.html>`__
from`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo/>`__isusedin
thistutorial.FormoreinformationabouthowOpenVINOIRmodelsare
created,refertothe`TensorFlowto
OpenVINO<tensorflow-classification-to-openvino-with-output.html>`__
tutorial.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`DownloadtheModelanddata
samples<#download-the-model-and-data-samples>`__
-`Selectinferencedevice<#select-inference-device>`__
-`LoadtheModel<#load-the-model>`__
-`LoadanImage<#load-an-image>`__
-`DoInference<#do-inference>`__

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


Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frompathlibimportPath

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

fromnotebook_utilsimportdownload_file

DownloadtheModelanddatasamples
-----------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

base_artifacts_dir=Path("./artifacts").expanduser()

model_name="v3-small_224_1.0_float"
model_xml_name=f"{model_name}.xml"
model_bin_name=f"{model_name}.bin"

model_xml_path=base_artifacts_dir/model_xml_name

base_url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/"

ifnotmodel_xml_path.exists():
download_file(base_url+model_xml_name,model_xml_name,base_artifacts_dir)
download_file(base_url+model_bin_name,model_bin_name,base_artifacts_dir)
else:
print(f"{model_name}alreadydownloadedto{base_artifacts_dir}")



..parsed-literal::

artifacts/v3-small_224_1.0_float.xml:0%||0.00/294k[00:00<?,?B/s]



..parsed-literal::

artifacts/v3-small_224_1.0_float.bin:0%||0.00/4.84M[00:00<?,?B/s]


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

output_layer=compiled_model.output(0)

LoadanImage
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Downloadtheimagefromtheopenvino_notebooksstorage
image_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
directory="data",
)

#TheMobileNetmodelexpectsimagesinRGBformat.
image=cv2.cvtColor(cv2.imread(filename=str(image_filename)),code=cv2.COLOR_BGR2RGB)

#ResizetoMobileNetimageshape.
input_image=cv2.resize(src=image,dsize=(224,224))

#Reshapetomodelinputshape.
input_image=np.expand_dims(input_image,0)
plt.imshow(image);



..parsed-literal::

data/coco.jpg:0%||0.00/202k[00:00<?,?B/s]



..image::hello-world-with-output_files/hello-world-with-output_11_1.png


DoInference
------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

result_infer=compiled_model([input_image])[output_layer]
result_index=np.argmax(result_infer)

..code::ipython3

imagenet_filename=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
directory="data",
)

imagenet_classes=imagenet_filename.read_text().splitlines()



..parsed-literal::

data/imagenet_2012.txt:0%||0.00/30.9k[00:00<?,?B/s]


..code::ipython3

#Themodeldescriptionstatesthatforthismodel,class0isabackground.
#Therefore,abackgroundmustbeaddedatthebeginningofimagenet_classes.
imagenet_classes=["background"]+imagenet_classes

imagenet_classes[result_index]




..parsed-literal::

'n02099267flat-coatedretriever'


