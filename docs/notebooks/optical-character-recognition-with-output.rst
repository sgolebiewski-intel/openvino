OpticalCharacterRecognition(OCR)withOpenVINO™
==================================================

Thistutorialdemonstrateshowtoperformopticalcharacterrecognition
(OCR)withOpenVINOmodels.Itisacontinuationofthe
`hello-detection<hello-detection-with-output.html>`__tutorial,
whichshowsonlytextdetection.

The
`horizontal-text-detection-0001<https://docs.openvino.ai/2024/omz_models_model_horizontal_text_detection_0001.html>`__
and
`text-recognition-resnet<https://docs.openvino.ai/2024/omz_models_model_text_recognition_resnet_fc.html>`__
modelsareusedtogetherfortextdetectionandthentextrecognition.

Inthistutorial,OpenModelZootoolsincludingModelDownloader,Model
ConverterandInfoDumperareusedtodownloadandconvertthemodels
from`OpenModel
Zoo<https://github.com/openvinotoolkit/open_model_zoo>`__.Formore
information,refertothe
`model-tools<model-tools-with-output.html>`__tutorial.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`Settings<#settings>`__
-`DownloadModels<#download-models>`__
-`ConvertModels<#convert-models>`__
-`Selectinferencedevice<#select-inference-device>`__
-`ObjectDetection<#object-detection>`__

-`LoadaDetectionModel<#load-a-detection-model>`__
-`LoadanImage<#load-an-image>`__
-`DoInference<#do-inference>`__
-`GetDetectionResults<#get-detection-results>`__

-`TextRecognition<#text-recognition>`__

-`LoadTextRecognitionModel<#load-text-recognition-model>`__
-`DoInference<#do-inference>`__

-`ShowResults<#show-results>`__

-`ShowDetectedTextBoxesandOCRResultsforthe
Image<#show-detected-text-boxes-and-ocr-results-for-the-image>`__
-`ShowtheOCRResultperBounding
Box<#show-the-ocr-result-per-bounding-box>`__
-`PrintAnnotationsinPlainText
Format<#print-annotations-in-plain-text-format>`__

..code::ipython3

importplatform

#Installopenvino-devpackage
%pipinstall-q"openvino-dev>=2024.0.0"onnxtorchpillowopencv-python--extra-index-urlhttps://download.pytorch.org/whl/cpu

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
openvino-tokenizers2024.4.0.0.dev20240712requiresopenvino~=2024.4.0.0.dev,butyouhaveopenvino2024.2.0whichisincompatible.
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
fromIPython.displayimportMarkdown,display
fromPILimportImage

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportload_image

Settings
--------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

core=ov.Core()

model_dir=Path("model")
precision="FP16"
detection_model="horizontal-text-detection-0001"
recognition_model="text-recognition-resnet-fc"

model_dir.mkdir(exist_ok=True)

DownloadModels
---------------

`backtotop⬆️<#table-of-contents>`__

ThenextcellswillrunModelDownloadertodownloadthedetectionand
recognitionmodels.Ifthemodelshavebeendownloadedbefore,theywill
notbedownloadedagain.

..code::ipython3

download_command=(
f"omz_downloader--name{detection_model},{recognition_model}--output_dir{model_dir}--cache_dir{model_dir}--precision{precision}--num_attempts5"
)
display(Markdown(f"Downloadcommand:`{download_command}`"))
display(Markdown(f"Downloading{detection_model},{recognition_model}..."))
!$download_command
display(Markdown(f"Finisheddownloading{detection_model},{recognition_model}."))

detection_model_path=(model_dir/"intel/horizontal-text-detection-0001"/precision/detection_model).with_suffix(".xml")
recognition_model_path=(model_dir/"public/text-recognition-resnet-fc"/precision/recognition_model).with_suffix(".xml")



Downloadcommand:
``omz_downloader--namehorizontal-text-detection-0001,text-recognition-resnet-fc--output_dirmodel--cache_dirmodel--precisionFP16--num_attempts5``



Downloadinghorizontal-text-detection-0001,text-recognition-resnet-fc…


..parsed-literal::

################||Downloadinghorizontal-text-detection-0001||################

==========Downloadingmodel/intel/horizontal-text-detection-0001/FP16/horizontal-text-detection-0001.xml


==========Downloadingmodel/intel/horizontal-text-detection-0001/FP16/horizontal-text-detection-0001.bin


################||Downloadingtext-recognition-resnet-fc||################

==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/model.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/weight_init.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/heads/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/heads/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/heads/fc_head.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/heads/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/body.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/component.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/sequences/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/sequences/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/sequences/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/bricks.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/resnet.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/enhance_modules/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/enhance_modules/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/enhance_modules/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/utils/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/utils/builder.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/utils/conv_module.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/utils/fc_module.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/utils/norm.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/models/utils/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/utils/__init__.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/utils/common.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/utils/registry.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/utils/config.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/configs/resnet_fc.py


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/ckpt/resnet_fc.pth


==========Downloadingmodel/public/text-recognition-resnet-fc/vedastr/addict-2.4.0-py3-none-any.whl


==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/heads/__init__.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/__init__.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/sequences/__init__.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/component.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/__init__.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/__init__.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/__init__.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/enhance_modules/__init__.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/utils/__init__.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/utils/__init__.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/utils/config.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/utils/config.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/utils/config.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/utils/config.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/utils/config.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/resnet.py
==========Replacingtextinmodel/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/resnet.py
==========Unpackingmodel/public/text-recognition-resnet-fc/vedastr/addict-2.4.0-py3-none-any.whl




Finisheddownloadinghorizontal-text-detection-0001,
text-recognition-resnet-fc.


..code::ipython3

###Thetext-recognition-resnet-fcmodelconsistsofmanyfiles.Allfilenamesareprintedin
###theoutputofModelDownloader.Uncommentthenexttwolinestoshowthisoutput.

#forlineindownload_result:
#print(line)

ConvertModels
--------------

`backtotop⬆️<#table-of-contents>`__

ThedownloadeddetectionmodelisanIntelmodel,whichisalreadyin
OpenVINOIntermediateRepresentation(OpenVINOIR)format.Thetext
recognitionmodelisapublicmodelwhichneedstobeconvertedto
OpenVINOIR.SincethismodelwasdownloadedfromOpenModelZoo,use
ModelConvertertoconvertthemodeltoOpenVINOIRformat.

TheoutputofModelConverterwillbedisplayed.Whentheconversionis
successful,thelastlinesofoutputwillinclude
``[SUCCESS]GeneratedIRversion11model.``

..code::ipython3

convert_command=f"omz_converter--name{recognition_model}--precisions{precision}--download_dir{model_dir}--output_dir{model_dir}"
display(Markdown(f"Convertcommand:`{convert_command}`"))
display(Markdown(f"Converting{recognition_model}..."))
!$convert_command



Convertcommand:
``omz_converter--nametext-recognition-resnet-fc--precisionsFP16--download_dirmodel--output_dirmodel``



Convertingtext-recognition-resnet-fc…


..parsed-literal::

==========Convertingtext-recognition-resnet-fctoONNX
ConversiontoONNXcommand:/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/bin/python--/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py--model-path=/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/models/public/text-recognition-resnet-fc--model-path=model/public/text-recognition-resnet-fc--model-name=get_model--import-module=model'--model-param=file_config=r"model/public/text-recognition-resnet-fc/vedastr/configs/resnet_fc.py"''--model-param=weights=r"model/public/text-recognition-resnet-fc/vedastr/ckpt/resnet_fc.pth"'--input-shape=1,1,32,100--input-names=input--output-names=output--output-file=model/public/text-recognition-resnet-fc/resnet_fc.onnx

ONNXcheckpassedsuccessfully.

==========Convertingtext-recognition-resnet-fctoIR(FP16)
Conversioncommand:/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/bin/python--/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/bin/mo--framework=onnx--output_dir=model/public/text-recognition-resnet-fc/FP16--model_name=text-recognition-resnet-fc--input=input'--mean_values=input[127.5]''--scale_values=input[127.5]'--output=output--input_model=model/public/text-recognition-resnet-fc/resnet_fc.onnx'--layout=input(NCHW)''--input_shape=[1,1,32,100]'--compress_to_fp16=True

[INFO]MOcommandlinetoolisconsideredasthelegacyconversionAPIasofOpenVINO2023.2release.
In2025.0MOcommandlinetoolandopenvino.tools.mo.convert_model()willberemoved.PleaseuseOpenVINOModelConverter(OVC)oropenvino.convert_model().OVCrepresentsalightweightalternativeofMOandprovidessimplifiedmodelconversionAPI.
FindmoreinformationabouttransitionfromMOtoOVCathttps://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
[INFO]GeneratedIRwillbecompressedtoFP16.Ifyougetloweraccuracy,pleaseconsiderdisablingcompressionexplicitlybyaddingargument--compress_to_fp16=False.
FindmoreinformationaboutcompressiontoFP16athttps://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
[SUCCESS]GeneratedIRversion11model.
[SUCCESS]XMLfile:/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/optical-character-recognition/model/public/text-recognition-resnet-fc/FP16/text-recognition-resnet-fc.xml
[SUCCESS]BINfile:/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/optical-character-recognition/model/public/text-recognition-resnet-fc/FP16/text-recognition-resnet-fc.bin



Selectinferencedevice
-----------------------

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



ObjectDetection
----------------

`backtotop⬆️<#table-of-contents>`__

Loadadetectionmodel,loadanimage,doinferenceandgetthe
detectioninferenceresult.

LoadaDetectionModel
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

detection_model=core.read_model(model=detection_model_path,weights=detection_model_path.with_suffix(".bin"))
detection_compiled_model=core.compile_model(model=detection_model,device_name=device.value)

detection_input_layer=detection_compiled_model.input(0)

LoadanImage
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#The`image_file`variablecanpointtoaURLoralocalimage.
image_file="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/intel_rnb.jpg"

image=load_image(image_file)

#N,C,H,W=batchsize,numberofchannels,height,width.
N,C,H,W=detection_input_layer.shape

#Resizetheimagetomeetnetworkexpectedinputsizes.
resized_image=cv2.resize(image,(W,H))

#Reshapetothenetworkinputshape.
input_image=np.expand_dims(resized_image.transpose(2,0,1),0)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));



..image::optical-character-recognition-with-output_files/optical-character-recognition-with-output_16_0.png


DoInference
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Textboxesaredetectedintheimagesandreturnedasblobsofdatain
theshapeof``[100,5]``.Eachdescriptionofdetectionhasthe
``[x_min,y_min,x_max,y_max,conf]``format.

..code::ipython3

output_key=detection_compiled_model.output("boxes")
boxes=detection_compiled_model([input_image])[output_key]

#Removezeroonlyboxes.
boxes=boxes[~np.all(boxes==0,axis=1)]

GetDetectionResults
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defmultiply_by_ratio(ratio_x,ratio_y,box):
return[max(shape*ratio_y,10)ifidx%2elseshape*ratio_xforidx,shapeinenumerate(box[:-1])]


defrun_preprocesing_on_crop(crop,net_shape):
temp_img=cv2.resize(crop,net_shape)
temp_img=temp_img.reshape((1,)*2+temp_img.shape)
returntemp_img


defconvert_result_to_image(bgr_image,resized_image,boxes,threshold=0.3,conf_labels=True):
#Definecolorsforboxesanddescriptions.
colors={"red":(255,0,0),"green":(0,255,0),"white":(255,255,255)}

#Fetchimageshapestocalculatearatio.
(real_y,real_x),(resized_y,resized_x)=image.shape[:2],resized_image.shape[:2]
ratio_x,ratio_y=real_x/resized_x,real_y/resized_y

#ConvertthebaseimagefromBGRtoRGBformat.
rgb_image=cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)

#Iteratethroughnon-zeroboxes.
forbox,annotationinboxes:
#Pickaconfidencefactorfromthelastplaceinanarray.
conf=box[-1]
ifconf>threshold:
#Convertfloattointandmultiplypositionofeachboxbyxandyratio.
(x_min,y_min,x_max,y_max)=map(int,multiply_by_ratio(ratio_x,ratio_y,box))

#Drawaboxbasedontheposition.Parametersinthe`rectangle`functionare:image,start_point,end_point,color,thickness.
cv2.rectangle(rgb_image,(x_min,y_min),(x_max,y_max),colors["green"],3)

#Addatexttoanimagebasedonthepositionandconfidence.Parametersinthe`putText`functionare:image,text,bottomleft_corner_textfield,font,font_scale,color,thickness,line_type
ifconf_labels:
#Createabackgroundboxbasedonannotationlength.
(text_w,text_h),_=cv2.getTextSize(f"{annotation}",cv2.FONT_HERSHEY_TRIPLEX,0.8,1)
image_copy=rgb_image.copy()
cv2.rectangle(
image_copy,
(x_min,y_min-text_h-10),
(x_min+text_w,y_min-10),
colors["white"],
-1,
)
#Addweightedimagecopywithwhiteboxesunderatext.
cv2.addWeighted(image_copy,0.4,rgb_image,0.6,0,rgb_image)
cv2.putText(
rgb_image,
f"{annotation}",
(x_min,y_min-10),
cv2.FONT_HERSHEY_SIMPLEX,
0.8,
colors["red"],
1,
cv2.LINE_AA,
)

returnrgb_image

TextRecognition
----------------

`backtotop⬆️<#table-of-contents>`__

Loadthetextrecognitionmodelanddoinferenceonthedetectedboxes
fromthedetectionmodel.

LoadTextRecognitionModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

recognition_model=core.read_model(model=recognition_model_path,weights=recognition_model_path.with_suffix(".bin"))

recognition_compiled_model=core.compile_model(model=recognition_model,device_name=device.value)

recognition_output_layer=recognition_compiled_model.output(0)
recognition_input_layer=recognition_compiled_model.input(0)

#Gettheheightandwidthoftheinputlayer.
_,_,H,W=recognition_input_layer.shape

DoInference
~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Calculatescaleforimageresizing.
(real_y,real_x),(resized_y,resized_x)=image.shape[:2],resized_image.shape[:2]
ratio_x,ratio_y=real_x/resized_x,real_y/resized_y

#Converttheimagetograyscaleforthetextrecognitionmodel.
grayscale_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Getadictionarytoencodeoutput,basedonthemodeldocumentation.
letters="~0123456789abcdefghijklmnopqrstuvwxyz"

#Prepareanemptylistforannotations.
annotations=list()
cropped_images=list()
#fig,ax=plt.subplots(len(boxes),1,figsize=(5,15),sharex=True,sharey=True)
#Getannotationsforeachcrop,basedonboxesgivenbythedetectionmodel.
fori,cropinenumerate(boxes):
#Getcoordinatesoncornersofacrop.
(x_min,y_min,x_max,y_max)=map(int,multiply_by_ratio(ratio_x,ratio_y,crop))
image_crop=run_preprocesing_on_crop(grayscale_image[y_min:y_max,x_min:x_max],(W,H))

#Runinferencewiththerecognitionmodel.
result=recognition_compiled_model([image_crop])[recognition_output_layer]

#Squeezetheoutputtoremoveunnecessarydimension.
recognition_results_test=np.squeeze(result)

#Readanannotationbasedonprobabilitiesfromtheoutputlayer.
annotation=list()
forletterinrecognition_results_test:
parsed_letter=letters[letter.argmax()]

#Returning0indexfrom`argmax`signalizesanendofastring.
ifparsed_letter==letters[0]:
break
annotation.append(parsed_letter)
annotations.append("".join(annotation))
cropped_image=Image.fromarray(image[y_min:y_max,x_min:x_max])
cropped_images.append(cropped_image)

boxes_with_annotations=list(zip(boxes,annotations))

ShowResults
------------

`backtotop⬆️<#table-of-contents>`__

ShowDetectedTextBoxesandOCRResultsfortheImage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Visualizetheresultbydrawingboxesaroundrecognizedtextandshowing
theOCRresultfromthetextrecognitionmodel.

..code::ipython3

plt.figure(figsize=(12,12))
plt.imshow(convert_result_to_image(image,resized_image,boxes_with_annotations,conf_labels=True));



..image::optical-character-recognition-with-output_files/optical-character-recognition-with-output_26_0.png


ShowtheOCRResultperBoundingBox
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Dependingontheimage,theOCRresultmaynotbereadableintheimage
withboxes,asdisplayedinthecellabove.Usethecodebelowto
displaytheextractedboxesandtheOCRresultperbox.

..code::ipython3

forcropped_image,annotationinzip(cropped_images,annotations):
display(cropped_image,Markdown("".join(annotation)))



..image::optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_0.png



building



..image::optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_2.png



noyce



..image::optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_4.png



2200



..image::optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_6.png



n



..image::optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_8.png



center



..image::optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_10.png



robert


PrintAnnotationsinPlainTextFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Printannotationsfordetectedtextbasedontheirpositionintheinput
image,startingfromtheupperleftcorner.

..code::ipython3

[annotationfor_,annotationinsorted(zip(boxes,annotations),key=lambdax:x[0][0]**2+x[0][1]**2)]




..parsed-literal::

['robert','n','noyce','building','2200','center']


