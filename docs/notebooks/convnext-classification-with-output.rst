ClassificationwithConvNeXtandOpenVINO
=========================================

The
`torchvision.models<https://pytorch.org/vision/stable/models.html>`__
subpackagecontainsdefinitionsofmodelsforaddressingdifferent
tasks,including:imageclassification,pixelwisesemanticsegmentation,
objectdetection,instancesegmentation,personkeypointdetection,
videoclassification,andopticalflow.Throughoutthisnotebookwewill
showhowtouseoneofthem.

TheConvNeXtmodelisbasedonthe`AConvNetforthe
2020s<https://arxiv.org/abs/2201.03545>`__paper.Theoutcomeofthis
explorationisafamilyofpureConvNetmodelsdubbedConvNeXt.
ConstructedentirelyfromstandardConvNetmodules,ConvNeXtscompete
favorablywithTransformersintermsofaccuracyandscalability,
achieving87.8%ImageNettop-1accuracyandoutperformingSwin
TransformersonCOCOdetectionandADE20Ksegmentation,while
maintainingthesimplicityandefficiencyofstandardConvNets.The
``torchvision.models``subpackage
`contains<https://pytorch.org/vision/main/models/convnext.html>`__
severalpretrainedConvNeXtmodel.InthistutorialwewilluseConvNeXt
Tinymodel.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Getatestimage<#get-a-test-image>`__
-`Getapretrainedmodel<#get-a-pretrained-model>`__
-`Defineapreprocessingandprepareaninput
data<#define-a-preprocessing-and-prepare-an-input-data>`__
-`Usetheoriginalmodeltorunan
inference<#use-the-original-model-to-run-an-inference>`__
-`ConvertthemodeltoOpenVINOIntermediaterepresentation
format<#convert-the-model-to-openvino-intermediate-representation-format>`__
-`UsetheOpenVINOIRmodeltorunan
inference<#use-the-openvino-ir-model-to-run-an-inference>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cputorchtorchvision
%pipinstall-q"openvino>=2023.1.0"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


Getatestimage
----------------

`backtotop⬆️<#table-of-contents>`__Firstofallletsgetatest
imagefromanopendataset.

..code::ipython3

importrequests

fromtorchvision.ioimportread_image
importtorchvision.transformsastransforms


img_path="cats_image.jpeg"
r=requests.get("https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg")

withopen(img_path,"wb")asf:
f.write(r.content)
image=read_image(img_path)
display(transforms.ToPILImage()(image))



..image::convnext-classification-with-output_files/convnext-classification-with-output_4_0.png


Getapretrainedmodel
----------------------

`backtotop⬆️<#table-of-contents>`__Torchvisionprovidesa
mechanismof`listingandretrievingavailable
models<https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models>`__.

..code::ipython3

importtorchvision.modelsasmodels

#Listavailablemodels
all_models=models.list_models()
#Listofmodelsbytype.Classificationmodelsareintheparentmodule.
classification_models=models.list_models(module=models)

print(classification_models)


..parsed-literal::

['alexnet','convnext_base','convnext_large','convnext_small','convnext_tiny','densenet121','densenet161','densenet169','densenet201','efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7','efficientnet_v2_l','efficientnet_v2_m','efficientnet_v2_s','googlenet','inception_v3','maxvit_t','mnasnet0_5','mnasnet0_75','mnasnet1_0','mnasnet1_3','mobilenet_v2','mobilenet_v3_large','mobilenet_v3_small','regnet_x_16gf','regnet_x_1_6gf','regnet_x_32gf','regnet_x_3_2gf','regnet_x_400mf','regnet_x_800mf','regnet_x_8gf','regnet_y_128gf','regnet_y_16gf','regnet_y_1_6gf','regnet_y_32gf','regnet_y_3_2gf','regnet_y_400mf','regnet_y_800mf','regnet_y_8gf','resnet101','resnet152','resnet18','resnet34','resnet50','resnext101_32x8d','resnext101_64x4d','resnext50_32x4d','shufflenet_v2_x0_5','shufflenet_v2_x1_0','shufflenet_v2_x1_5','shufflenet_v2_x2_0','squeezenet1_0','squeezenet1_1','swin_b','swin_s','swin_t','swin_v2_b','swin_v2_s','swin_v2_t','vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn','vit_b_16','vit_b_32','vit_h_14','vit_l_16','vit_l_32','wide_resnet101_2','wide_resnet50_2']


Wewilluse``convnext_tiny``.Togetapretrainedmodeljustuse
``models.get_model("convnext_tiny",weights='DEFAULT')``oraspecific
methodof``torchvision.models``forthismodelusing`default
weights<https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html#torchvision.models.ConvNeXt_Tiny_Weights>`__
thatisequivalentto``ConvNeXt_Tiny_Weights.IMAGENET1K_V1``.Ifyou
don’tspecify``weight``orspecify``weights=None``itwillbearandom
initialization.Togetallavailableweightsforthemodelyoucancall
``weights_enum=models.get_model_weights("convnext_tiny")``,butthere
isonlyoneforthismodel.Youcanfindmoreinformationhowto
initializepre-trainedmodels
`here<https://pytorch.org/vision/stable/models.html#initializing-pre-trained-models>`__.

..code::ipython3

model=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

Defineapreprocessingandprepareaninputdata
------------------------------------------------

`backtotop⬆️<#table-of-contents>`__Youcanuse
``torchvision.transforms``tomakeapreprocessingor
use\`preprocessingtransformsfromthemodel
wight<https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models>`__.

..code::ipython3

importtorch


preprocess=models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()

input_data=preprocess(image)
input_data=torch.stack([input_data],dim=0)

Usetheoriginalmodeltorunaninference
------------------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

outputs=model(input_data)

Andprintresults

..code::ipython3

#downloadclassnumbertoclasslabelmapping
imagenet_classes_file_path="imagenet_2012.txt"
r=requests.get(
url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
)

withopen(imagenet_classes_file_path,"w")asf:
f.write(r.text)

imagenet_classes=open(imagenet_classes_file_path).read().splitlines()


defprint_results(outputs:torch.Tensor):
_,predicted_class=outputs.max(1)
predicted_probability=torch.softmax(outputs,dim=1)[0,predicted_class].item()

print(f"PredictedClass:{predicted_class.item()}")
print(f"PredictedLabel:{imagenet_classes[predicted_class.item()]}")
print(f"PredictedProbability:{predicted_probability}")

..code::ipython3

print_results(outputs)


..parsed-literal::

PredictedClass:281
PredictedLabel:n02123045tabby,tabbycat
PredictedProbability:0.5351971983909607


ConvertthemodeltoOpenVINOIntermediaterepresentationformat
----------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchthroughconversiontoOpenVINOIntermediate
Representation(IR)format.TotaketheadvantageofOpenVINO
optimizationtoolsandfeatures,themodelshouldbeconvertedusingthe
OpenVINOConvertertool(OVC).The``openvino.convert_model``function
providesPythonAPIforOVCusage.Thefunctionreturnstheinstanceof
theOpenVINOModelclass,whichisreadyforuseinthePython
interface.However,itcanalsobesavedondiskusing
``openvino.save_model``forfutureexecution.

..code::ipython3

frompathlibimportPath

importopenvinoasov


ov_model_xml_path=Path("models/ov_convnext_model.xml")

ifnotov_model_xml_path.exists():
ov_model_xml_path.parent.mkdir(parents=True,exist_ok=True)
converted_model=ov.convert_model(model,example_input=torch.randn(1,3,224,224))
#addtransformtoOpenVINOpreprocessingconverting
ov.save_model(converted_model,ov_model_xml_path)
else:
print(f"IRmodel{ov_model_xml_path}alreadyexists.")


..parsed-literal::

['x']


Whenthe``openvino.save_model``functionisused,anOpenVINOmodelis
serializedinthefilesystemastwofileswith``.xml``and``.bin``
extensions.ThispairoffilesiscalledOpenVINOIntermediate
Representationformat(OpenVINOIR,orjustIR)andusefulforefficient
modeldeployment.OpenVINOIRcanbeloadedintoanotherapplicationfor
inferenceusingthe``openvino.Core.read_model``function.

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



..code::ipython3

core=ov.Core()

compiled_model=core.compile_model(ov_model_xml_path,device_name=device.value)

UsetheOpenVINOIRmodeltorunaninference
---------------------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

outputs=compiled_model(input_data)[0]
print_results(torch.from_numpy(outputs))


..parsed-literal::

PredictedClass:281
PredictedLabel:n02123045tabby,tabbycat
PredictedProbability:0.5664422512054443

