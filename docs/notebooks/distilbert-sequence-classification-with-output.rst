SentimentAnalysiswithOpenVINO™
=================================

**Sentimentanalysis**istheuseofnaturallanguageprocessing,text
analysis,computationallinguistics,andbiometricstosystematically
identify,extract,quantify,andstudyaffectivestatesandsubjective
information.Thisnotebookdemonstrateshowtoconvertandruna
sequenceclassificationmodelusingOpenVINO.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Imports<#imports>`__
-`InitializingtheModel<#initializing-the-model>`__
-`InitializingtheTokenizer<#initializing-the-tokenizer>`__
-`ConvertModeltoOpenVINOIntermediateRepresentation
format<#convert-model-to-openvino-intermediate-representation-format>`__

-`Selectinferencedevice<#select-inference-device>`__

-`Inference<#inference>`__

-`Forasingleinputsentence<#for-a-single-input-sentence>`__
-`Readfromatextfile<#read-from-a-text-file>`__

Imports
-------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall"openvino>=2023.1.0"transformers"torch>=2.1"tqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Lookinginindexes:https://pypi.org/simple,https://download.pytorch.org/whl/cpu
Requirementalreadysatisfied:openvino>=2023.1.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(2024.4.0.dev20240712)
Requirementalreadysatisfied:transformersin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(4.42.4)
Requirementalreadysatisfied:torch>=2.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(2.3.1+cpu)
Requirementalreadysatisfied:tqdmin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(4.66.4)
Requirementalreadysatisfied:numpy<2.0.0,>=1.16.6in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino>=2023.1.0)(1.23.5)
Requirementalreadysatisfied:openvino-telemetry>=2023.2.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino>=2023.1.0)(2024.1.0)
Requirementalreadysatisfied:packagingin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromopenvino>=2023.1.0)(24.1)
Requirementalreadysatisfied:filelockin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtransformers)(3.15.4)
Requirementalreadysatisfied:huggingface-hub<1.0,>=0.23.2in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtransformers)(0.23.4)
Requirementalreadysatisfied:pyyaml>=5.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtransformers)(6.0.1)
Requirementalreadysatisfied:regex!=2019.12.17in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtransformers)(2024.5.15)
Requirementalreadysatisfied:requestsin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtransformers)(2.32.3)
Requirementalreadysatisfied:safetensors>=0.4.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtransformers)(0.4.3)
Requirementalreadysatisfied:tokenizers<0.20,>=0.19in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtransformers)(0.19.1)
Requirementalreadysatisfied:typing-extensions>=4.8.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(4.12.2)
Requirementalreadysatisfied:sympyin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(1.13.0)
Requirementalreadysatisfied:networkxin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(3.1)
Requirementalreadysatisfied:jinja2in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(3.1.4)
Requirementalreadysatisfied:fsspecin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(2024.5.0)
Requirementalreadysatisfied:MarkupSafe>=2.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromjinja2->torch>=2.1)(2.1.5)
Requirementalreadysatisfied:charset-normalizer<4,>=2in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromrequests->transformers)(3.3.2)
Requirementalreadysatisfied:idna<4,>=2.5in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromrequests->transformers)(3.7)
Requirementalreadysatisfied:urllib3<3,>=1.21.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromrequests->transformers)(2.2.2)
Requirementalreadysatisfied:certifi>=2017.4.17in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromrequests->transformers)(2024.7.4)
Requirementalreadysatisfied:mpmath<1.4,>=1.1.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromsympy->torch>=2.1)(1.3.0)
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importwarnings
frompathlibimportPath
importtime
fromtransformersimportAutoModelForSequenceClassification,AutoTokenizer
importnumpyasnp
importopenvinoasov

InitializingtheModel
----------------------

`backtotop⬆️<#table-of-contents>`__

Wewillusethetransformer-based`DistilBERTbaseuncasedfinetuned
SST-2<https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
modelfromHuggingFace.

..code::ipython3

checkpoint="distilbert-base-uncased-finetuned-sst-2-english"
model=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=checkpoint)

InitializingtheTokenizer
--------------------------

`backtotop⬆️<#table-of-contents>`__

TextPreprocessingcleansthetext-basedinputdatasoitcanbefed
intothemodel.
`Tokenization<https://towardsdatascience.com/tokenization-for-natural-language-processing-a179a891bad4>`__
splitsparagraphsandsentencesintosmallerunitsthatcanbemore
easilyassignedmeaning.Itinvolvescleaningthedataandassigning
tokensorIDstothewords,sotheyarerepresentedinavectorspace
wheresimilarwordshavesimilarvectors.Thishelpsthemodel
understandthecontextofasentence.Here,wewilluse
`AutoTokenizer<https://huggingface.co/docs/transformers/main_classes/tokenizer>`__
-apre-trainedtokenizerfromHuggingFace:

..code::ipython3

tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint)

ConvertModeltoOpenVINOIntermediateRepresentationformat
------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

`Modelconversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
facilitatesthetransitionbetweentraininganddeploymentenvironments,
performsstaticmodelanalysis,andadjustsdeeplearningmodelsfor
optimalexecutiononend-pointtargetdevices.

..code::ipython3

importtorch

ir_xml_name=checkpoint+".xml"
MODEL_DIR="model/"
ir_xml_path=Path(MODEL_DIR)/ir_xml_name

MAX_SEQ_LENGTH=128
input_info=[
(ov.PartialShape([1,-1]),ov.Type.i64),
(ov.PartialShape([1,-1]),ov.Type.i64),
]
default_input=torch.ones(1,MAX_SEQ_LENGTH,dtype=torch.int64)
inputs={
"input_ids":default_input,
"attention_mask":default_input,
}

ov_model=ov.convert_model(model,input=input_info,example_input=inputs)
ov.save_model(ov_model,ir_xml_path)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4565:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py:230:TracerWarning:torch.tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
mask,torch.tensor(torch.finfo(scores.dtype).min)


..parsed-literal::

['input_ids','attention_mask']


OpenVINO™Runtimeusesthe`Infer
Request<https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/inference-request.html>`__
mechanismwhichenablesrunningmodelsondifferentdevicesin
asynchronousorsynchronousmanners.Themodelgraphissentasan
argumenttotheOpenVINOAPIandaninferencerequestiscreated.The
defaultinferencemodeisAUTObutitcanbechangedaccordingto
requirementsandhardwareavailable.Youcanexplorethedifferent
inferencemodesandtheirusage`in
documentation.<https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.html>`__

..code::ipython3

core=ov.Core()

Selectinferencedevice
~~~~~~~~~~~~~~~~~~~~~~~

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



..code::ipython3

warnings.filterwarnings("ignore")
compiled_model=core.compile_model(ov_model,device.value)
infer_request=compiled_model.create_infer_request()

..code::ipython3

defsoftmax(x):
"""
Definingasoftmaxfunctiontoextract
thepredictionfromtheoutputoftheIRformat
Parameters:Logitsarray
Returns:Probabilities
"""

e_x=np.exp(x-np.max(x))
returne_x/e_x.sum()

Inference
---------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

definfer(input_text):
"""
Creatingagenericinferencefunction
toreadtheinputandinfertheresult
into2classes:PositiveorNegative.
Parameters:Texttobeprocessed
Returns:Label:PositiveorNegative.
"""

input_text=tokenizer(
input_text,
truncation=True,
return_tensors="np",
)
inputs=dict(input_text)
label={0:"NEGATIVE",1:"POSITIVE"}
result=infer_request.infer(inputs=inputs)
foriinresult.values():
probability=np.argmax(softmax(i))
returnlabel[probability]

Forasingleinputsentence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

input_text="Ihadawonderfulday"
start_time=time.perf_counter()
result=infer(input_text)
end_time=time.perf_counter()
total_time=end_time-start_time
print("Label:",result)
print("TotalTime:","%.2f"%total_time,"seconds")


..parsed-literal::

Label:POSITIVE
TotalTime:0.03seconds


Readfromatextfile
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

#Downloadthetextfromtheopenvino_notebooksstorage
vocab_file_path=download_file(
"https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/text/food_reviews.txt",
directory="data",
)



..parsed-literal::

data/food_reviews.txt:0%||0.00/71.0[00:00<?,?B/s]


..code::ipython3

start_time=time.perf_counter()
withvocab_file_path.open(mode="r")asf:
input_text=f.readlines()
forlinesininput_text:
print("UserInput:",lines)
result=infer(lines)
print("Label:",result,"\n")
end_time=time.perf_counter()
total_time=end_time-start_time
print("TotalTime:","%.2f"%total_time,"seconds")


..parsed-literal::

UserInput:Thefoodwashorrible.

Label:NEGATIVE

UserInput:Wewentbecausetherestauranthadgoodreviews.
Label:POSITIVE

TotalTime:0.03seconds

