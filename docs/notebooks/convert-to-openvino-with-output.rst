OpenVINO™Modelconversion
==========================

Thisnotebookshowshowtoconvertamodelfromoriginalframework
formattoOpenVINOIntermediateRepresentation(IR).

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`OpenVINOIRformat<#openvino-ir-format>`__
-`Fetchingexamplemodels<#fetching-example-models>`__
-`Conversion<#conversion>`__

-`SettingInputShapes<#setting-input-shapes>`__
-`CompressingaModeltoFP16<#compressing-a-model-to-fp16>`__
-`ConvertModelsfrommemory<#convert-models-from-memory>`__

-`MigrationfromLegacyconversion
API<#migration-from-legacy-conversion-api>`__

-`SpecifyingLayout<#specifying-layout>`__
-`ChangingModelLayout<#changing-model-layout>`__
-`SpecifyingMeanandScale
Values<#specifying-mean-and-scale-values>`__
-`ReversingInputChannels<#reversing-input-channels>`__
-`CuttingOffPartsofaModel<#cutting-off-parts-of-a-model>`__

..code::ipython3

#Requiredimports.Pleaseexecutethiscellfirst.
%pipinstall--upgradepip
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu\
"openvino-dev>=2024.0.0""requests""tqdm""transformers[onnx]>=4.31""torch>=2.1""torchvision""tensorflow_hub""tensorflow"


..parsed-literal::

Requirementalreadysatisfied:pipin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(24.1.2)
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


OpenVINOIRformat
------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINO`IntermediateRepresentation
(IR)<https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__
istheproprietarymodelformatofOpenVINO.Itisproducedafter
convertingamodelwithmodelconversionAPI.ModelconversionAPI
translatesthefrequentlyuseddeeplearningoperationstotheir
respectivesimilarrepresentationinOpenVINOandtunesthemwiththe
associatedweightsandbiasesfromthetrainedmodel.TheresultingIR
containstwofiles:an``.xml``file,containinginformationabout
networktopology,anda``.bin``file,containingtheweightsandbiases
binarydata.

Therearetwowaystoconvertamodelfromtheoriginalframeworkformat
toOpenVINOIR:PythonconversionAPIandOVCcommand-linetool.Youcan
chooseoneofthembasedonwhicheverismostconvenientforyou.

OpenVINOconversionAPIsupportsnextmodelformats:``PyTorch``,
``TensorFlow``,``TensorFlowLite``,``ONNX``,and``PaddlePaddle``.
Thesemodelformatscanberead,compiled,andconvertedtoOpenVINOIR,
eitherautomaticallyorexplicitly.

Formoredetails,referto`Model
Preparation<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
documentation.

..code::ipython3

#OVCCLItoolparametersdescription

!ovc--help


..parsed-literal::

usage:ovcINPUT_MODEL...[-h][--output_modelOUTPUT_MODEL]
[--compress_to_fp16[True|False]][--version][--inputINPUT]
[--outputOUTPUT][--extensionEXTENSION][--verbose]

positionalarguments:
INPUT_MODELInputmodelfile(s)fromTensorFlow,ONNX,
PaddlePaddle.Useopenvino.convert_modelinPythonto
convertmodelsfromPyTorch.

optionalarguments:
-h,--helpshowthishelpmessageandexit
--output_modelOUTPUT_MODEL
Thisparameterisusedtonameoutput.xml/.binfiles
ofconvertedmodel.Modelnameoroutputdirectorycan
bepassed.Ifoutputdirectoryispassed,the
resulting.xml/.binfilesarenamedbyoriginalmodel
name.
--compress_to_fp16[True|False]
CompressweightsinoutputOpenVINOmodeltoFP16.To
turnoffcompressionuse"--compress_to_fp16=False"
commandlineparameter.DefaultvalueisTrue.
--versionPrintovcversionandexit.
--inputINPUTInformationofmodelinputrequiredformodel
conversion.Thisisacommaseparatedlistwith
optionalinputnamesandshapes.Theorderofinputs
inconvertedmodelwillmatchtheorderofspecified
inputs.Theshapeisspecifiedascomma-separated
list.Example,toset`input_1`inputwithshape
[1,100]and`sequence_len`inputwithshape[1,?]:
"input_1[1,100],sequence_len[1,?]",where"?"isa
dynamicdimension,whichmeansthatsuchadimension
canbespecifiedlaterintheruntime.Ifthe
dimensionissetasaninteger(like100in[1,100]),
suchadimensionisnotsupposedtobechangedlater,
duringamodelconversionitistreatedasastatic
value.Examplewithunnamedinputs:"[1,100],[1,?]".
--outputOUTPUTOneormorecomma-separatedmodeloutputstobe
preservedintheconvertedmodel.Otheroutputsare
removed.If`output`parameterisnotspecifiedthen
alloutputsfromtheoriginalmodelarepreserved.Do
notadd:0tothenamesforTensorFlow.Theorderof
outputsintheconvertedmodelisthesameasthe
orderofspecifiednames.Example:ovcmodel.onnx
output=out_1,out_2
--extensionEXTENSION
Pathsoracomma-separatedlistofpathstolibraries
(.soor.dll)withextensions.
--verbosePrintdetailedinformationaboutconversion.


Fetchingexamplemodels
-----------------------

`backtotop⬆️<#table-of-contents>`__

Thisnotebookusestwomodelsforconversionexamples:

-`Distilbert<https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
NLPmodelfromHuggingFace
-`Resnet50<https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights>`__
CVclassificationmodelfromtorchvision

..code::ipython3

frompathlibimportPath

#createadirectoryformodelsfiles
MODEL_DIRECTORY_PATH=Path("model")
MODEL_DIRECTORY_PATH.mkdir(exist_ok=True)

Fetch
`distilbert<https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
NLPmodelfromHuggingFaceandexportitinONNXformat:

..code::ipython3

fromtransformersimportAutoModelForSequenceClassification,AutoTokenizer
fromtransformers.onnximportexport,FeaturesManager

ONNX_NLP_MODEL_PATH=MODEL_DIRECTORY_PATH/"distilbert.onnx"

#downloadmodel
hf_model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
#initializetokenizer
tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

#getmodelonnxconfigfunctionforoutputfeatureformatsequence-classification
model_kind,model_onnx_config=FeaturesManager.check_supported_model_or_raise(hf_model,feature="sequence-classification")
#fillonnxconfigbasedonpytorchmodelconfig
onnx_config=model_onnx_config(hf_model.config)

#exporttoonnxformat
export(
preprocessor=tokenizer,
model=hf_model,
config=onnx_config,
opset=onnx_config.default_onnx_opset,
output=ONNX_NLP_MODEL_PATH,
)


..parsed-literal::

2024-07-1223:47:37.510126:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1223:47:37.545461:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1223:47:38.151630:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py:230:TracerWarning:torch.tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
mask,torch.tensor(torch.finfo(scores.dtype).min)




..parsed-literal::

(['input_ids','attention_mask'],['logits'])



Fetch
`Resnet50<https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights>`__
CVclassificationmodelfromtorchvision:

..code::ipython3

fromtorchvision.modelsimportresnet50,ResNet50_Weights

#createmodelobject
pytorch_model=resnet50(weights=ResNet50_Weights.DEFAULT)
#switchmodelfromtrainingtoinferencemode
pytorch_model.eval()




..parsed-literal::

ResNet(
(conv1):Conv2d(3,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
(bn1):BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
(maxpool):MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False)
(layer1):Sequential(
(0):Bottleneck(
(conv1):Conv2d(64,64,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(64,64,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(64,256,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
(downsample):Sequential(
(0):Conv2d(64,256,kernel_size=(1,1),stride=(1,1),bias=False)
(1):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
)
)
(1):Bottleneck(
(conv1):Conv2d(256,64,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(64,64,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(64,256,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
(2):Bottleneck(
(conv1):Conv2d(256,64,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(64,64,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(64,256,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
)
(layer2):Sequential(
(0):Bottleneck(
(conv1):Conv2d(256,128,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(128,128,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
(bn2):BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(128,512,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
(downsample):Sequential(
(0):Conv2d(256,512,kernel_size=(1,1),stride=(2,2),bias=False)
(1):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
)
)
(1):Bottleneck(
(conv1):Conv2d(512,128,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(128,128,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(128,512,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
(2):Bottleneck(
(conv1):Conv2d(512,128,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(128,128,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(128,512,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
(3):Bottleneck(
(conv1):Conv2d(512,128,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(128,128,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(128,512,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
)
(layer3):Sequential(
(0):Bottleneck(
(conv1):Conv2d(512,256,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(256,256,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
(bn2):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(256,1024,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(1024,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
(downsample):Sequential(
(0):Conv2d(512,1024,kernel_size=(1,1),stride=(2,2),bias=False)
(1):BatchNorm2d(1024,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
)
)
(1):Bottleneck(
(conv1):Conv2d(1024,256,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(256,1024,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(1024,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
(2):Bottleneck(
(conv1):Conv2d(1024,256,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(256,1024,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(1024,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
(3):Bottleneck(
(conv1):Conv2d(1024,256,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(256,1024,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(1024,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
(4):Bottleneck(
(conv1):Conv2d(1024,256,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(256,1024,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(1024,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
(5):Bottleneck(
(conv1):Conv2d(1024,256,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(256,1024,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(1024,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
)
(layer4):Sequential(
(0):Bottleneck(
(conv1):Conv2d(1024,512,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(512,512,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
(bn2):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(512,2048,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(2048,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
(downsample):Sequential(
(0):Conv2d(1024,2048,kernel_size=(1,1),stride=(2,2),bias=False)
(1):BatchNorm2d(2048,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
)
)
(1):Bottleneck(
(conv1):Conv2d(2048,512,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(512,2048,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(2048,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
(2):Bottleneck(
(conv1):Conv2d(2048,512,kernel_size=(1,1),stride=(1,1),bias=False)
(bn1):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv2):Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
(bn2):BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(conv3):Conv2d(512,2048,kernel_size=(1,1),stride=(1,1),bias=False)
(bn3):BatchNorm2d(2048,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
(relu):ReLU(inplace=True)
)
)
(avgpool):AdaptiveAvgPool2d(output_size=(1,1))
(fc):Linear(in_features=2048,out_features=1000,bias=True)
)



ConvertPyTorchmodeltoONNXformat:

..code::ipython3

importtorch
importwarnings

ONNX_CV_MODEL_PATH=MODEL_DIRECTORY_PATH/"resnet.onnx"

ifONNX_CV_MODEL_PATH.exists():
print(f"ONNXmodel{ONNX_CV_MODEL_PATH}alreadyexists.")
else:
withwarnings.catch_warnings():
warnings.filterwarnings("ignore")
torch.onnx.export(model=pytorch_model,args=torch.randn(1,3,224,224),f=ONNX_CV_MODEL_PATH)
print(f"ONNXmodelexportedto{ONNX_CV_MODEL_PATH}")


..parsed-literal::

ONNXmodelexportedtomodel/resnet.onnx


Conversion
----------

`backtotop⬆️<#table-of-contents>`__

ToconvertamodeltoOpenVINOIR,usethefollowingAPI:

..code::ipython3

importopenvinoasov

#ov.convert_modelreturnsanopenvino.runtime.Modelobject
print(ONNX_NLP_MODEL_PATH)
ov_model=ov.convert_model(ONNX_NLP_MODEL_PATH)

#thenmodelcanbeserializedto*.xml&*.binfiles
ov.save_model(ov_model,MODEL_DIRECTORY_PATH/"distilbert.xml")


..parsed-literal::

model/distilbert.onnx


..code::ipython3

!ovcmodel/distilbert.onnx--output_modelmodel/distilbert.xml


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

[INFO]GeneratedIRwillbecompressedtoFP16.Ifyougetloweraccuracy,pleaseconsiderdisablingcompressionbyremovingargument"compress_to_fp16"orsetittofalse"compress_to_fp16=False".
FindmoreinformationaboutcompressiontoFP16athttps://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
[SUCCESS]XMLfile:model/distilbert.xml
[SUCCESS]BINfile:model/distilbert.bin


SettingInputShapes
^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Modelconversionissupportedformodelswithdynamicinputshapesthat
containundefineddimensions.However,iftheshapeofdataisnotgoing
tochangefromoneinferencerequesttoanother,itisrecommendedto
setupstaticshapes(whenalldimensionsarefullydefined)forthe
inputs.Doingsoatthemodelpreparationstage,notatruntime,canbe
beneficialintermsofperformanceandmemoryconsumption.

Formoreinformationreferto`SettingInput
Shapes<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/setting-input-shapes.html>`__
documentation.

..code::ipython3

importopenvinoasov

ov_model=ov.convert_model(ONNX_NLP_MODEL_PATH,input=[("input_ids",[1,128]),("attention_mask",[1,128])])

..code::ipython3

!ovcmodel/distilbert.onnx--inputinput_ids[1,128],attention_mask[1,128]--output_modelmodel/distilbert.xml


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

[INFO]GeneratedIRwillbecompressedtoFP16.Ifyougetloweraccuracy,pleaseconsiderdisablingcompressionbyremovingargument"compress_to_fp16"orsetittofalse"compress_to_fp16=False".
FindmoreinformationaboutcompressiontoFP16athttps://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
[SUCCESS]XMLfile:model/distilbert.xml
[SUCCESS]BINfile:model/distilbert.bin


The``input``parameterallowsoverridingoriginalinputshapesifitis
supportedbythemodeltopology.Shapeswithdynamicdimensionsinthe
originalmodelcanbereplacedwithstaticshapesfortheconverted
model,andviceversa.Thedynamicdimensioncanbemarkedinmodel
conversionAPIparameteras``-1``or``?``whenusing``ovc``:

..code::ipython3

importopenvinoasov

ov_model=ov.convert_model(ONNX_NLP_MODEL_PATH,input=[("input_ids",[1,-1]),("attention_mask",[1,-1])])

..code::ipython3

!ovcmodel/distilbert.onnx--input"input_ids[1,?],attention_mask[1,?]"--output_modelmodel/distilbert.xml


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

[INFO]GeneratedIRwillbecompressedtoFP16.Ifyougetloweraccuracy,pleaseconsiderdisablingcompressionbyremovingargument"compress_to_fp16"orsetittofalse"compress_to_fp16=False".
FindmoreinformationaboutcompressiontoFP16athttps://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
[SUCCESS]XMLfile:model/distilbert.xml
[SUCCESS]BINfile:model/distilbert.bin


Tooptimizememoryconsumptionformodelswithundefineddimensionsin
runtime,modelconversionAPIprovidesthecapabilitytodefine
boundariesofdimensions.Theboundariesofundefineddimensioncanbe
specifiedwithellipsisinthecommandlineorwith
``openvino.Dimension``classinPython.Forexample,launchmodel
conversionfortheONNXBertmodelandspecifyaboundaryforthe
sequencelengthdimension:

..code::ipython3

importopenvinoasov


sequence_length_dim=ov.Dimension(10,128)

ov_model=ov.convert_model(
ONNX_NLP_MODEL_PATH,
input=[
("input_ids",[1,sequence_length_dim]),
("attention_mask",[1,sequence_length_dim]),
],
)

..code::ipython3

!ovcmodel/distilbert.onnx--inputinput_ids[1,10..128],attention_mask[1,10..128]--output_modelmodel/distilbert.xml


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

[INFO]GeneratedIRwillbecompressedtoFP16.Ifyougetloweraccuracy,pleaseconsiderdisablingcompressionbyremovingargument"compress_to_fp16"orsetittofalse"compress_to_fp16=False".
FindmoreinformationaboutcompressiontoFP16athttps://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
[SUCCESS]XMLfile:model/distilbert.xml
[SUCCESS]BINfile:model/distilbert.bin


CompressingaModeltoFP16
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

BydefaultmodelweightscompressedtoFP16formatwhensavingOpenVINO
modeltoIR.Thissavesupto2xstoragespaceforthemodelfileandin
mostcasesdoesn’tsacrificemodelaccuracy.Weightcompressioncanbe
disabledbysetting``compress_to_fp16``flagto``False``:

..code::ipython3

importopenvinoasov

ov_model=ov.convert_model(ONNX_NLP_MODEL_PATH)
ov.save_model(ov_model,MODEL_DIRECTORY_PATH/"distilbert.xml",compress_to_fp16=False)

..code::ipython3

!ovcmodel/distilbert.onnx--output_modelmodel/distilbert.xml--compress_to_fp16=False


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

[SUCCESS]XMLfile:model/distilbert.xml
[SUCCESS]BINfile:model/distilbert.bin


ConvertModelsfrommemory
^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

ModelconversionAPIsupportspassingoriginalframeworkPythonobject
directly.Moredetailscanbefoundin
`PyTorch<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-pytorch.html>`__,
`TensorFlow<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-tensorflow.html>`__,
`PaddlePaddle<https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-paddle.html>`__
frameworksconversionguides.

..code::ipython3

importopenvinoasov
importtorch

example_input=torch.rand(1,3,224,224)

ov_model=ov.convert_model(pytorch_model,example_input=example_input,input=example_input.shape)


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..code::ipython3

importos

importopenvinoasov
importtensorflow_hubashub

os.environ["TFHUB_CACHE_DIR"]=str(Path("./tfhub_modules").resolve())

model=hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-lightning/versions/4")
movenet=model.signatures["serving_default"]

ov_model=ov.convert_model(movenet)


..parsed-literal::

2024-07-1223:47:58.665238:Etensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266]failedcalltocuInit:CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:forwardcompatibilitywasattemptedonnonsupportedHW
2024-07-1223:47:58.665270:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168]retrievingCUDAdiagnosticinformationforhost:iotg-dev-workstation-07
2024-07-1223:47:58.665275:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175]hostname:iotg-dev-workstation-07
2024-07-1223:47:58.665487:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199]libcudareportedversionis:470.223.2
2024-07-1223:47:58.665503:Itensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203]kernelreportedversionis:470.182.3
2024-07-1223:47:58.665507:Etensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312]kernelversion470.182.3doesnotmatchDSOversion470.223.2--cannotfindworkingdevicesinthisconfiguration


MigrationfromLegacyconversionAPI
------------------------------------

`backtotop⬆️<#table-of-contents>`__

Inthe2023.1OpenVINOreleaseOpenVINOModelConversionAPIwas
introducedwiththecorrespondingPythonAPI:``openvino.convert_model``
method.``ovc``and``openvino.convert_model``representalightweight
alternativeof``mo``and``openvino.tools.mo.convert_model``whichare
consideredlegacyAPInow.``mo.convert_model()``providesawiderange
ofpreprocessingparameters.Mostoftheseparametershaveanalogsin
OVCorcanbereplacedwithfunctionalityfrom``ov.PrePostProcessor``
class.Referto`OptimizePreprocessing
notebook<optimize-preprocessing-with-output.html>`__for
moreinformationabout`Preprocessing
API<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html>`__.
Hereisthemigrationguidefromlegacymodelpreprocessingto
PreprocessingAPI.

SpecifyingLayout
^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Layoutdefinesthemeaningofdimensionsinashapeandcanbespecified
forbothinputsandoutputs.Somepreprocessingrequirestosetinput
layouts,forexample,settingabatch,applyingmeanorscales,and
reversinginputchannels(BGR<->RGB).Forthelayoutsyntax,checkthe
`LayoutAPI
overview<https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/layout-api-overview.html>`__.
Tospecifythelayout,youcanusethelayoutoptionfollowedbythe
layoutvalue.

Thefollowingexamplespecifiesthe``NCHW``layoutforaPytorch
Resnet50modelthatwasexportedtotheONNXformat:

..code::ipython3

#ConverterAPI
importopenvinoasov

ov_model=ov.convert_model(ONNX_CV_MODEL_PATH)

prep=ov.preprocess.PrePostProcessor(ov_model)
prep.input("input.1").model().set_layout(ov.Layout("nchw"))
ov_model=prep.build()

..code::ipython3

#LegacyModelOptimizerAPI
fromopenvino.toolsimportmo

ov_model=mo.convert_model(ONNX_CV_MODEL_PATH,layout="nchw")


..parsed-literal::

[INFO]MOcommandlinetoolisconsideredasthelegacyconversionAPIasofOpenVINO2023.2release.
In2025.0MOcommandlinetoolandopenvino.tools.mo.convert_model()willberemoved.PleaseuseOpenVINOModelConverter(OVC)oropenvino.convert_model().OVCrepresentsalightweightalternativeofMOandprovidessimplifiedmodelconversionAPI.
FindmoreinformationabouttransitionfromMOtoOVCathttps://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


ChangingModelLayout
^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Transposingofmatrices/tensorsisatypicaloperationinDeepLearning
-youmayhaveaBMPimage``640x480``,whichisanarrayof
``{480,640,3}``elements,butDeepLearningmodelcanrequireinput
withshape``{1,3,480,640}``.

Conversioncanbedoneimplicitly,usingthelayoutofauser’stensor
andthelayoutofanoriginalmodel:

..code::ipython3

#ConverterAPI
importopenvinoasov

ov_model=ov.convert_model(ONNX_CV_MODEL_PATH)

prep=ov.preprocess.PrePostProcessor(ov_model)
prep.input("input.1").tensor().set_layout(ov.Layout("nhwc"))
prep.input("input.1").model().set_layout(ov.Layout("nchw"))
ov_model=prep.build()

..code::ipython3

#LegacyModelOptimizerAPI
fromopenvino.toolsimportmo

ov_model=mo.convert_model(ONNX_CV_MODEL_PATH,layout="nchw->nhwc")

#alternativelyusesource_layoutandtarget_layoutparameters
ov_model=mo.convert_model(ONNX_CV_MODEL_PATH,source_layout="nchw",target_layout="nhwc")


..parsed-literal::

[INFO]MOcommandlinetoolisconsideredasthelegacyconversionAPIasofOpenVINO2023.2release.
In2025.0MOcommandlinetoolandopenvino.tools.mo.convert_model()willberemoved.PleaseuseOpenVINOModelConverter(OVC)oropenvino.convert_model().OVCrepresentsalightweightalternativeofMOandprovidessimplifiedmodelconversionAPI.
FindmoreinformationabouttransitionfromMOtoOVCathttps://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
[INFO]MOcommandlinetoolisconsideredasthelegacyconversionAPIasofOpenVINO2023.2release.
In2025.0MOcommandlinetoolandopenvino.tools.mo.convert_model()willberemoved.PleaseuseOpenVINOModelConverter(OVC)oropenvino.convert_model().OVCrepresentsalightweightalternativeofMOandprovidessimplifiedmodelconversionAPI.
FindmoreinformationabouttransitionfromMOtoOVCathttps://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


SpecifyingMeanandScaleValues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

UsingPreprocessingAPI``mean``and``scale``valuescanbeset.Using
theseAPI,modelembedsthecorrespondingpreprocessingblockfor
mean-valuenormalizationoftheinputdataandoptimizesthisblock.
Referto`OptimizePreprocessing
notebook<optimize-preprocessing-with-output.html>`__for
moreexamples.

..code::ipython3

#ConverterAPI
importopenvinoasov

ov_model=ov.convert_model(ONNX_CV_MODEL_PATH)

prep=ov.preprocess.PrePostProcessor(ov_model)
prep.input("input.1").tensor().set_layout(ov.Layout("nchw"))
prep.input("input.1").preprocess().mean([255*xforxin[0.485,0.456,0.406]])
prep.input("input.1").preprocess().scale([255*xforxin[0.229,0.224,0.225]])

ov_model=prep.build()

..code::ipython3

#LegacyModelOptimizerAPI
fromopenvino.toolsimportmo


ov_model=mo.convert_model(
ONNX_CV_MODEL_PATH,
mean_values=[255*xforxin[0.485,0.456,0.406]],
scale_values=[255*xforxin[0.229,0.224,0.225]],
)


..parsed-literal::

[INFO]MOcommandlinetoolisconsideredasthelegacyconversionAPIasofOpenVINO2023.2release.
In2025.0MOcommandlinetoolandopenvino.tools.mo.convert_model()willberemoved.PleaseuseOpenVINOModelConverter(OVC)oropenvino.convert_model().OVCrepresentsalightweightalternativeofMOandprovidessimplifiedmodelconversionAPI.
FindmoreinformationabouttransitionfromMOtoOVCathttps://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


ReversingInputChannels
^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Sometimes,inputimagesforyourapplicationcanbeofthe``RGB``(or
``BGR``)format,andthemodelistrainedonimagesofthe``BGR``(or
``RGB``)format,whichisintheoppositeorderofcolorchannels.In
thiscase,itisimportanttopreprocesstheinputimagesbyreverting
thecolorchannelsbeforeinference.

..code::ipython3

#ConverterAPI
importopenvinoasov

ov_model=ov.convert_model(ONNX_CV_MODEL_PATH)

prep=ov.preprocess.PrePostProcessor(ov_model)
prep.input("input.1").tensor().set_layout(ov.Layout("nchw"))
prep.input("input.1").preprocess().reverse_channels()
ov_model=prep.build()

..code::ipython3

#LegacyModelOptimizerAPI
fromopenvino.toolsimportmo

ov_model=mo.convert_model(ONNX_CV_MODEL_PATH,reverse_input_channels=True)


..parsed-literal::

[INFO]MOcommandlinetoolisconsideredasthelegacyconversionAPIasofOpenVINO2023.2release.
In2025.0MOcommandlinetoolandopenvino.tools.mo.convert_model()willberemoved.PleaseuseOpenVINOModelConverter(OVC)oropenvino.convert_model().OVCrepresentsalightweightalternativeofMOandprovidessimplifiedmodelconversionAPI.
FindmoreinformationabouttransitionfromMOtoOVCathttps://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


CuttingOffPartsofaModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`backtotop⬆️<#table-of-contents>`__

Cuttingmodelinputsandoutputsfromamodelisnolongeravailablein
thenewconversionAPI.Instead,werecommendperformingthecutinthe
originalframework.ExamplesofmodelcuttingofTensorFlowprotobuf,
TensorFlowSavedModel,andONNXformatswithtoolsprovidedbythe
TensorflowandONNXframeworkscanbefoundin`documentation
guide<https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api.html#cutting-off-parts-of-a-model>`__.
ForPyTorch,TensorFlow2Keras,andPaddlePaddle,werecommendchanging
theoriginalmodelcodetoperformthemodelcut.
