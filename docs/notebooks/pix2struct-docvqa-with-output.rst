DocumentVisualQuestionAnsweringUsingPix2StructandOpenVINO™
=================================================================

DocVQA(DocumentVisualQuestionAnswering)isaresearchfieldin
computervisionandnaturallanguageprocessingthatfocuseson
developingalgorithmstoanswerquestionsrelatedtothecontentofa
documentrepresentedinimageformat,likeascanneddocument,
screenshots,oranimageofatextdocument.Unlikeothertypesof
visualquestionanswering,wherethefocusisonansweringquestions
relatedtoimagesorvideos,DocVQAisfocusedonunderstandingand
answeringquestionsbasedonthetextandlayoutofadocument.The
questionscanbeaboutanyaspectofthedocumenttext.DocVQArequires
understandingthedocument’svisualcontentandtheabilitytoreadand
comprehendthetextinit.

DocVQAoffersseveralbenefitscomparedtoOCR(OpticalCharacter
Recognition)technology:\*Firstly,DocVQAcannotonlyrecognizeand
extracttextfromadocument,butitcanalsounderstandthecontextin
whichthetextappears.Thismeansitcananswerquestionsaboutthe
document’scontentratherthansimplyprovideadigitalversion.\*
Secondly,DocVQAcanhandledocumentswithcomplexlayoutsand
structures,liketablesanddiagrams,whichcanbechallengingfor
traditionalOCRsystems.\*Finally,DocVQAcanautomatemany
document-basedworkflows,likedocumentroutingandapprovalprocesses,
tomakeemployeesfocusonmoremeaningfulwork.Thepotential
applicationsofDocVQAincludeautomatingtaskslikeinformation
retrieval,documentanalysis,anddocumentsummarization.

`Pix2Struct<https://arxiv.org/pdf/2210.03347.pdf>`__isamultimodal
modelforunderstandingvisuallysituatedlanguagethateasilycopes
withextractinginformationfromimages.Themodelistrainedusingthe
novellearningtechniquetoparsemaskedscreenshotsofwebpagesinto
simplifiedHTML,providingasignificantlywell-suitedpretrainingdata
sourcefortherangeofdownstreamactivitiessuchasOCR,visual
questionanswering,andimagecaptioning.

Inthistutorial,weconsiderhowtorunthePix2Structmodelusing
OpenVINOforsolvingdocumentvisualquestionansweringtask.Wewill
useapre-trainedmodelfromthe`HuggingFace
Transformers<https://huggingface.co/docs/transformers/index>`__
library.Tosimplifytheuserexperience,the`HuggingFace
Optimum<https://huggingface.co/docs/optimum>`__libraryisusedto
convertthemodeltoOpenVINO™IRformat.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`AboutPix2Struct<#about-pix2struct>`__
-`Prerequisites<#prerequisites>`__
-`DownloadandConvertModel<#download-and-convert-model>`__
-`Selectinferencedevice<#select-inference-device>`__
-`Testmodelinference<#test-model-inference>`__
-`Interactivedemo<#interactive-demo>`__

AboutPix2Struct
----------------

`backtotop⬆️<#table-of-contents>`__

Pix2Structisanimageencoder-textdecodermodelthatistrainedon
image-textpairsforvarioustasks,includingimagecaptioningand
visualquestionanswering.Themodelcombinesthesimplicityofpurely
pixel-levelinputswiththegeneralityandscalabilityprovidedby
self-supervisedpretrainingfromdiverseandabundantwebdata.The
modeldoesthisbyrecommendingascreenshotparsingobjectivethat
needspredictinganHTML-basedparsefromascreenshotofawebpage
thathasbeenpartiallymasked.Withthediversityandcomplexityof
textualandvisualelementsfoundontheweb,Pix2Structlearnsrich
representationsoftheunderlyingstructureofwebpages,whichcan
effectivelytransfertovariousdownstreamvisuallanguageunderstanding
tasks.

Pix2StructisbasedontheVisionTransformer(ViT),an
image-encoder-text-decodermodelwithchangesininputrepresentationto
makethemodelmorerobusttoprocessingimageswithvariousaspect
ratios.StandardViTextractsfixed-sizepatchesafterscalinginput
imagestoapredeterminedresolution.Thisdistortstheproperaspect
ratiooftheimage,whichcanbehighlyvariablefordocuments,mobile
UIs,andfigures.Pix2Structproposestoscaletheinputimageupor
downtoextractthemaximumnumberofpatchesthatfitwithinthegiven
sequencelength.Thisapproachismorerobusttoextremeaspectratios,
commoninthedomainsPix2Structexperimentswith.Additionally,the
modelcanhandleon-the-flychangestothesequencelengthand
resolution.Tohandlevariableresolutionsunambiguously,2-dimensional
absolutepositionalembeddingsareusedfortheinputpatches.

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

First,weneedtoinstallthe`HuggingFace
Optimum<https://huggingface.co/docs/transformers/index>`__library
acceleratedbyOpenVINOintegration.TheHuggingFaceOptimumAPIisa
high-levelAPIthatenablesustoconvertandquantizemodelsfromthe
HuggingFaceTransformerslibrarytotheOpenVINO™IRformat.Formore
details,refertothe`HuggingFaceOptimum
documentation<https://huggingface.co/docs/optimum/intel/inference>`__.

..code::ipython3

%pipinstall-q"torch>=2.1"torchvisiontorchaudio--index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"git+https://github.com/huggingface/optimum-intel.git""openvino>=2023.1.0""transformers>=4.33.0""peft==0.6.2"onnx"gradio>=4.19"--extra-index-urlhttps://download.pytorch.org/whl/cpu

DownloadandConvertModel
--------------------------

`backtotop⬆️<#table-of-contents>`__

OptimumIntelcanbeusedtoloadoptimizedmodelsfromthe`Hugging
FaceHub<https://huggingface.co/docs/optimum/intel/hf.co/models>`__and
createpipelinestorunaninferencewithOpenVINORuntimeusingHugging
FaceAPIs.TheOptimumInferencemodelsareAPIcompatiblewithHugging
FaceTransformersmodels.Thismeanswejustneedtoreplacethe
``AutoModelForXxx``classwiththecorresponding``OVModelForXxx``
class.

Modelclassinitializationstartswithcallingthe``from_pretrained``
method.WhendownloadingandconvertingtheTransformersmodel,the
parameter``export=True``shouldbeadded.Wecansavetheconverted
modelforthenextusagewiththe``save_pretrained``method.After
modelsavingusingthe``save_pretrained``method,youcanloadyour
convertedmodelwithoutthe``export``parameter,avoidingmodel
conversionforthenexttime.Forreducingmemoryconsumption,wecan
compressmodeltofloat16using``half()``method.

Inthistutorial,weseparatemodelexportandloadingfora
demonstrationofhowtoworkwiththemodelinbothmodes.Wewilluse
the
`pix2struct-docvqa-base<https://huggingface.co/google/pix2struct-docvqa-base>`__
modelasanexampleinthistutorial,butthesamestepsforrunningare
applicableforothermodelsfrompix2structfamily.

..code::ipython3

importgc
frompathlibimportPath
fromoptimum.intel.openvinoimportOVModelForPix2Struct

model_id="google/pix2struct-docvqa-base"
model_dir=Path(model_id.split("/")[-1])

ifnotmodel_dir.exists():
ov_model=OVModelForPix2Struct.from_pretrained(model_id,export=True,compile=False)
ov_model.half()
ov_model.save_pretrained(model_dir)
delov_model
gc.collect();


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..parsed-literal::

NoCUDAruntimeisfound,usingCUDA_HOME='/usr/local/cuda'
2023-10-2013:49:09.525682:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2023-10-2013:49:09.565139:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2023-10-2013:49:10.397504:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/deepspeed.py:23:FutureWarning:transformers.deepspeedmoduleisdeprecatedandwillberemovedinafutureversion.Pleaseimportdeepspeedmodulesdirectlyfromtransformers.integrations
warnings.warn(


Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

selectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importipywidgetsaswidgets
importopenvinoasov

core=ov.Core()

device=widgets.Dropdown(
options=[dfordincore.available_devicesif"GPU"notind]+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



Testmodelinference
--------------------

`backtotop⬆️<#table-of-contents>`__

Thediagrambelowdemonstrateshowthemodelworks:
|pix2struct_diagram.png|

Forrunningmodelinferenceweshouldpreprocessdatafirst.
``Pix2StructProcessor``isresponsibleforpreparinginputdataand
decodingoutputfortheoriginalPyTorchmodelandeasilycanbereused
forrunningwiththeOptimumIntelmodel.Then
``OVModelForPix2Struct.generate``methodwilllaunchanswergeneration.
Finally,generatedanswertokenindicesshouldbedecodedintextformat
by``Pix2StructProcessor.decode``

..|pix2struct_diagram.png|image::https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/c7456b17-0687-4aa9-851b-267bff3dac79

..code::ipython3

fromtransformersimportPix2StructProcessor

processor=Pix2StructProcessor.from_pretrained(model_id)
ov_model=OVModelForPix2Struct.from_pretrained(model_dir,device=device.value)


..parsed-literal::

CompilingtheencodertoAUTO...
CompilingthedecodertoAUTO...
CompilingthedecodertoAUTO...


Let’sseethemodelinaction.Fortestingthemodel,wewillusea
screenshotfrom`OpenVINO
documentation<https://docs.openvino.ai/2024/get-started.html#openvino-advanced-features>`__

..code::ipython3

importrequests
fromPILimportImage
fromioimportBytesIO


defload_image(image_file):
response=requests.get(image_file)
image=Image.open(BytesIO(response.content)).convert("RGB")
returnimage


test_image_url="https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/aa46ef0c-c14d-4bab-8bb7-3b22fe73f6bc"

image=load_image(test_image_url)
text="Whatperformancehintsdo?"

inputs=processor(images=image,text=text,return_tensors="pt")
display(image)



..image::pix2struct-docvqa-with-output_files/pix2struct-docvqa-with-output_11_0.png


..code::ipython3

answer_tokens=ov_model.generate(**inputs)
answer=processor.decode(answer_tokens[0],skip_special_tokens=True)
print(f"Question:{text}")
print(f"Answer:{answer}")


..parsed-literal::

/home/ea/work/ov_venv/lib/python3.8/site-packages/optimum/intel/openvino/modeling_seq2seq.py:395:FutureWarning:`shared_memory`isdeprecatedandwillberemovedin2024.0.Valueof`shared_memory`isgoingtooverride`share_inputs`value.Pleaseuseonly`share_inputs`explicitly.
last_hidden_state=torch.from_numpy(self.request(inputs,shared_memory=True)["last_hidden_state"]).to(
/home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/generation/utils.py:1260:UserWarning:Usingthemodel-agnosticdefault`max_length`(=20)tocontrolthegenerationlength.Werecommendsetting`max_new_tokens`tocontrolthemaximumlengthofthegeneration.
warnings.warn(
/home/ea/work/ov_venv/lib/python3.8/site-packages/optimum/intel/openvino/modeling_seq2seq.py:476:FutureWarning:`shared_memory`isdeprecatedandwillberemovedin2024.0.Valueof`shared_memory`isgoingtooverride`share_inputs`value.Pleaseuseonly`share_inputs`explicitly.
self.request.start_async(inputs,shared_memory=True)


..parsed-literal::

Question:Whatperformancehintsdo?
Answer:automaticallyadjustruntimeparameterstoprioritizeforlowlatencyorhighthroughput


Interactivedemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importgradioasgr

example_images_urls=[
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/94ef687c-aebb-452b-93fe-c7f29ce19503",
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/70b2271c-9295-493b-8a5c-2f2027dcb653",
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1e2be134-0d45-4878-8e6c-08cfc9c8ea3d",
]

file_names=["eiffel_tower.png","exsibition.jpeg","population_table.jpeg"]

forimg_url,image_fileinzip(example_images_urls,file_names):
load_image(img_url).save(image_file)

questions=[
"WhatisEiffeltowertall?",
"Whenisthecoffeebreak?",
"WhatthepopulationofStoddard?",
]

examples=[list(pair)forpairinzip(file_names,questions)]


defgenerate(img,question):
inputs=processor(images=img,text=question,return_tensors="pt")
predictions=ov_model.generate(**inputs,max_new_tokens=256)
returnprocessor.decode(predictions[0],skip_special_tokens=True)


demo=gr.Interface(
fn=generate,
inputs=["image","text"],
outputs="text",
title="Pix2StructforDocVQA",
examples=examples,
cache_examples=False,
allow_flagging="never",
)

try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/
