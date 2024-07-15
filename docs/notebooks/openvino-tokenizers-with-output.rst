OpenVINOTokenizers:IncorporateTextProcessingIntoOpenVINOPipelines
========================================================================

..raw::html

<center>

..raw::html

</center>

OpenVINOTokenizersisanOpenVINOextensionandaPythonlibrary
designedtostreamlinetokenizerconversionforseamlessintegration
intoyourprojects.ItsupportsPythonandC++environmentsandis
compatiblewithallmajorplatforms:Linux,Windows,andMacOS.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`TokenizationBasics<#tokenization-basics>`__
-`AcquiringOpenVINOTokenizers<#acquiring-openvino-tokenizers>`__

-`ConvertTokenizerfromHuggingFaceHubwithCLI
Tool<#convert-tokenizer-from_huggingface-hub-with-cli-tool>`__
-`ConvertTokenizerfromHuggingFaceHubwithPython
API<#convert-tokenizer-from-huggingface-hub-with-python-api>`__

-`TextGenerationPipelinewithOpenVINO
Tokenizers<#text-generation-pipeline-with-openvino-tokenizers>`__
-`MergeTokenizerintoaModel<#merge-tokenizer-into-a-model>`__
-`Conclusion<#conclusion>`__
-`Links<#links>`__

TokenizationBasics
-------------------

`backtotop⬆️<#table-of-contents>`__

Onedoesnotsimplyputtextintoaneuralnetwork,onlynumbers.The
processoftransformingtextintoasequenceofnumbersiscalled
**tokenization**.Itusuallycontainsseveralstepsthattransformthe
originalstring,splittingitintoparts-tokens-withanassociated
numberinadictionary.Youcancheckthe`interactiveGPT-4
tokenizer<https://platform.openai.com/tokenizer>`__togainan
intuitiveunderstandingoftheprinciplesoftokenizerwork.

..raw::html

<center>

..raw::html

</center>

Therearetwoimportantpointsinthetokenizer-modelrelation:1.Every
neuralnetworkwithtextinputispairedwithatokenizerand*cannotbe
usedwithoutit*.2.Toreproducethemodel’saccuracyonaspecific
task,itisessentialto*utilizethesametokenizeremployedduringthe
modeltraining*.

Thatiswhyalmostallmodelrepositorieson`HuggingFace
Hub<https://HuggingFace.co/models>`__alsocontaintokenizerfiles
(``tokenizer.json``,``vocab.txt``,``merges.txt``,etc.).

Theprocessoftransformingasequenceofnumbersintoastringis
called**detokenization**.Detokenizercansharethetokendictionary
withatokenizer,likeanyLLMchatmodel,oroperatewithanentirely
distinctdictionary.Forinstance,translationmodelsdealingwith
differentsourceandtargetlanguagesoftennecessitateseparate
dictionaries.

..raw::html

<center>

..raw::html

</center>

Sometasksonlyneedatokenizer,liketextclassification,namedentity
recognition,questionanswering,andfeatureextraction.Ontheother
hand,fortaskssuchastextgeneration,chat,translation,and
abstractivesummarization,bothatokenizerandadetokenizerare
required.

AcquiringOpenVINOTokenizers
-----------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOTokenizersPythonlibraryallowsyoutoconvertHuggingFace
tokenizersintoOpenVINOmodels.Toinstallallrequireddependencies
use``pipinstallopenvino-tokenizers[transformers]``.

..code::ipython3

%pipinstall-Uqpip
%pipinstall--pre-Uqopenvino-tokenizers[transformers]--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly
%pipinstall"torch>=2.1"--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
openvino-dev2024.2.0requiresopenvino==2024.2.0,butyouhaveopenvino2024.4.0.dev20240712whichisincompatible.
openvino-genai2024.3.0.0.dev20240712requiresopenvino_tokenizers~=2024.3.0.0.dev,butyouhaveopenvino-tokenizers2024.4.0.0.dev20240712whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Lookinginindexes:https://pypi.org/simple,https://download.pytorch.org/whl/cpu
Requirementalreadysatisfied:torch>=2.1in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(2.3.1+cpu)
Requirementalreadysatisfied:filelockin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(3.15.4)
Requirementalreadysatisfied:typing-extensions>=4.8.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(4.12.2)
Requirementalreadysatisfied:sympyin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(1.13.0)
Requirementalreadysatisfied:networkxin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(3.1)
Requirementalreadysatisfied:jinja2in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(3.1.4)
Requirementalreadysatisfied:fsspecin/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromtorch>=2.1)(2024.5.0)
Requirementalreadysatisfied:MarkupSafe>=2.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromjinja2->torch>=2.1)(2.1.5)
Requirementalreadysatisfied:mpmath<1.4,>=1.1.0in/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages(fromsympy->torch>=2.1)(1.3.0)
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath


tokenizer_dir=Path("tokenizer/")
model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

ConvertTokenizerfromHuggingFaceHubwithCLITool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ThefirstwayistousetheCLIutility,bundledwithOpenVINO
Tokenizers.Use``--with-detokenizer``flagtoaddthedetokenizermodel
totheoutput.Bysetting``--clean-up-tokenization-spaces=False``we
ensurethatthedetokenizercorrectlydecodesacode-generationmodel
output.``--trust-remote-code``flagworksthesamewayaspassing
``trust_remote_code=True``to``AutoTokenizer.from_pretrained``
constructor.

..code::ipython3

!convert_tokenizer$model_id--with-detokenizer-o$tokenizer_dir


..parsed-literal::

LoadingHuggingfaceTokenizer...
ConvertingHuggingfaceTokenizertoOpenVINO...
SavedOpenVINOTokenizer:tokenizer/openvino_tokenizer.xml,tokenizer/openvino_tokenizer.bin
SavedOpenVINODetokenizer:tokenizer/openvino_detokenizer.xml,tokenizer/openvino_detokenizer.bin


⚠️IfyouhaveanyproblemswiththecommandaboveonMacOS,tryto
`installtbb<https://formulae.brew.sh/formula/tbb#default>`__.

TheresultistwoOpenVINOmodels:``openvino_tokenizer``and
``openvino_detokenizer``.Bothcanbeinteractedwithusing
``read_model``,``compile_model``and``save_model``,similartoany
otherOpenVINOmodel.

ConvertTokenizerfromHuggingFaceHubwithPythonAPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheothermethodistopassHuggingFace``hf_tokenizer``objectto
``convert_tokenizer``function:

..code::ipython3

fromtransformersimportAutoTokenizer
fromopenvino_tokenizersimportconvert_tokenizer


hf_tokenizer=AutoTokenizer.from_pretrained(model_id)
ov_tokenizer,ov_detokenizer=convert_tokenizer(hf_tokenizer,with_detokenizer=True)
ov_tokenizer,ov_detokenizer




..parsed-literal::

(<Model:'tokenizer'
inputs[
<ConstOutput:names[string_input]shape[?]type:string>
]
outputs[
<ConstOutput:names[input_ids]shape[?,?]type:i64>,
<ConstOutput:names[attention_mask]shape[?,?]type:i64>
]>,
<Model:'detokenizer'
inputs[
<ConstOutput:names[Parameter_22]shape[?,?]type:i64>
]
outputs[
<ConstOutput:names[string_output]shape[?]type:string>
]>)



ThatwayyougetOpenVINOmodelobjects.Use``save_model``function
fromOpenVINOtoreuseconvertedtokenizerslater:

..code::ipython3

fromopenvinoimportsave_model


save_model(ov_tokenizer,tokenizer_dir/"openvino_tokenizer.xml")
save_model(ov_detokenizer,tokenizer_dir/"openvino_detokenizer.xml")

Tousethetokenizer,compiletheconvertedmodelandinputalistof
strings.It’sessentialtobeawarethatnotalloriginaltokenizers
supportmultiplestrings(alsocalledbatches)asinput.Thislimitation
arisesfromtherequirementforallresultingnumbersequencesto
maintainthesamelength.Toaddressthis,apaddingtokenmustbe
specified,whichwillbeappendedtoshortertokenizedstrings.Incases
wherenopaddingtokenisdeterminedintheoriginaltokenizer,OpenVINO
Tokenizersdefaultstousing:math:`0`forpadding.Presently,*only
right-sidepaddingissupported*,typicallyusedforclassification
tasks,butnotsuitablefortextgeneration.

..code::ipython3

fromopenvinoimportcompile_model


tokenizer,detokenizer=compile_model(ov_tokenizer),compile_model(ov_detokenizer)
test_strings=["Test","strings"]

token_ids=tokenizer(test_strings)["input_ids"]
print(f"Tokenids:{token_ids}")

detokenized_text=detokenizer(token_ids)["string_output"]
print(f"Detokenizedtext:{detokenized_text}")


..parsed-literal::

Tokenids:[[14321]
[16031]]
Detokenizedtext:['Test''strings']


Wecancomparetheresultofconverted(de)tokenizerwiththeoriginal
one:

..code::ipython3

hf_token_ids=hf_tokenizer(test_strings).input_ids
print(f"Tokenids:{hf_token_ids}")

hf_detokenized_text=hf_tokenizer.batch_decode(hf_token_ids)
print(f"Detokenizedtext:{hf_detokenized_text}")


..parsed-literal::

Tokenids:[[1,4321],[1,6031]]


..parsed-literal::

2024-07-1301:17:56.121802:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:17:56.157863:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:17:56.747281:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


..parsed-literal::

Detokenizedtext:['<s>Test','<s>strings']


TextGenerationPipelinewithOpenVINOTokenizers
-------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Let’sbuildatextgenerationpipelinewithOpenVINOTokenizersand
minimaldependencies.ToobtainanOpenVINOmodelwewillusethe
Optimumlibrary.Thelatestversionallowsyoutogetaso-called
`stateful
model<https://docs.openvino.ai/2024/openvino-workflow/running-inference/stateful-models.html>`__.

Theoriginal``TinyLlama-1.1B-intermediate-step-1431k-3T``modelis
4.4Gb.Toreducenetworkanddiskusagewewillloadaconvertedmodel
whichhasalsobeencompressedto``int8``.Theoriginalconversion
commandiscommented.

..code::ipython3

model_dir=Path(Path(model_id).name)

ifnotmodel_dir.exists():
#convertingtheoriginalmodel
#%pipinstall-U"git+https://github.com/huggingface/optimum-intel.git""nncf>=2.8.0"onnx
#%optimum-cliexportopenvino-m$model_id--tasktext-generation-with-past$model_dir

#loadalreadyconvertedmodel
fromhuggingface_hubimporthf_hub_download

hf_hub_download(
"chgk13/TinyLlama-1.1B-intermediate-step-1431k-3T",
filename="openvino_model.xml",
local_dir=model_dir,
)
hf_hub_download(
"chgk13/TinyLlama-1.1B-intermediate-step-1431k-3T",
filename="openvino_model.bin",
local_dir=model_dir,
)



..parsed-literal::

openvino_model.xml:0%||0.00/2.93M[00:00<?,?B/s]



..parsed-literal::

openvino_model.bin:0%||0.00/1.10G[00:00<?,?B/s]


..code::ipython3

importnumpyasnp
fromtqdm.notebookimporttrange
frompathlibimportPath
fromopenvino_tokenizersimportadd_greedy_decoding
fromopenvino_tokenizers.constantsimportEOS_TOKEN_ID_NAME
fromopenvinoimportCore


core=Core()

#addthegreedydecodingsubgraphontopofLLMtogetthemostprobabletokenasanoutput
ov_model=add_greedy_decoding(core.read_model(model_dir/"openvino_model.xml"))
compiled_model=core.compile_model(ov_model)
infer_request=compiled_model.create_infer_request()

The``infer_request``objectprovidescontroloverthemodel’sstate-a
Key-Valuecachethatspeedsupinferencebyreducingcomputations
Multipleinferencerequestscanbecreated,andeachrequestmaintainsa
distinctandseparatestate..

..code::ipython3

text_input=["Quickbrownfoxjumped"]

model_input={name.any_name:outputforname,outputintokenizer(text_input).items()}

if"position_ids"in(input.any_nameforinputininfer_request.model_inputs):
model_input["position_ids"]=np.arange(model_input["input_ids"].shape[1],dtype=np.int64)[np.newaxis,:]

#nobeamsearch,setidxto0
model_input["beam_idx"]=np.array([0],dtype=np.int32)
#endofsentencetokenisthatmodelsignifiestheendoftextgeneration
#readEOStokenIDfromrt_infooftokenizer/detokenizerov.Modelobject
eos_token=ov_tokenizer.get_rt_info(EOS_TOKEN_ID_NAME).value

tokens_result=np.array([[]],dtype=np.int64)

#resetKVcacheinsidethemodelbeforeinference
infer_request.reset_state()
max_infer=10

for_intrange(max_infer):
infer_request.start_async(model_input)
infer_request.wait()

#getapredictionforthelasttokenonthefirstinference
output_token=infer_request.get_output_tensor().data[:,-1:]
tokens_result=np.hstack((tokens_result,output_token))
ifoutput_token[0,0]==eos_token:
break

#prepareinputfornewinference
model_input["input_ids"]=output_token
model_input["attention_mask"]=np.hstack((model_input["attention_mask"].data,[[1]]))
model_input["position_ids"]=np.hstack(
(
model_input["position_ids"].data,
[[model_input["position_ids"].data.shape[-1]]],
)
)

text_result=detokenizer(tokens_result)["string_output"]
print(f"Prompt:\n{text_input[0]}")
print(f"Generated:\n{text_result[0]}")



..parsed-literal::

0%||0/10[00:00<?,?it/s]


..parsed-literal::

Prompt:
Quickbrownfoxjumped
Generated:
overthefence.







MergeTokenizerintoaModel
----------------------------

`backtotop⬆️<#table-of-contents>`__

Packageslike``tensorflow-text``offertheconvenienceofintegrating
textprocessingdirectlyintothemodel,streamliningbothdistribution
andusage.Similarly,withOpenVINOTokenizers,youcancreatemodels
thatcombineaconvertedtokenizerandamodel.It’simportanttonote
thatnotallscenariosbenefitfromthismerge.Incaseswherea
tokenizerisusedonceandamodelisinferredmultipletimes,asseen
intheearliertextgenerationexample,maintainingaseparate
(de)tokenizerandmodelisadvisabletopreventunnecessary
tokenization-detokenizationcyclesduringinference.Conversely,ifboth
atokenizerandamodelareusedonceineachpipelineinference,
mergingsimplifiestheworkflowandaidsinavoidingthecreationof
intermediateobjects:

..raw::html

<center>

..raw::html

</center>

TheOpenVINOPythonAPIallowsyoutoavoidthisbyusingthe
``share_inputs``optionduringinference,butitrequiresadditional
inputfromadevelopereverytimethemodelisinferred.Combiningthe
modelsandtokenizerssimplifiesmemorymanagement.

..code::ipython3

model_id="mrm8488/bert-tiny-finetuned-sms-spam-detection"
model_dir=Path(Path(model_id).name)

ifnotmodel_dir.exists():
%pipinstall-qUgit+https://github.com/huggingface/optimum-intel.gitonnx
!optimum-cliexportopenvino--model$model_id--tasktext-classification$model_dir
!convert_tokenizer$model_id-o$model_dir


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

2024-07-1301:18:19.229824:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
Frameworknotspecified.Usingpttoexportthemodel.
UsingframeworkPyTorch:2.3.1+cpu
Overriding1configurationitem(s)
	-use_cache->False
['input_ids','attention_mask','token_type_ids']
Detokenizerisnotsupported,converttokenizeronly.


..parsed-literal::

huggingface/tokenizers:Thecurrentprocessjustgotforked,afterparallelismhasalreadybeenused.Disablingparallelismtoavoiddeadlocks...
Todisablethiswarning,youcaneither:
	-Avoidusing`tokenizers`beforetheforkifpossible
	-ExplicitlysettheenvironmentvariableTOKENIZERS_PARALLELISM=(true|false)


..parsed-literal::

LoadingHuggingfaceTokenizer...
ConvertingHuggingfaceTokenizertoOpenVINO...
SavedOpenVINOTokenizer:bert-tiny-finetuned-sms-spam-detection/openvino_tokenizer.xml,bert-tiny-finetuned-sms-spam-detection/openvino_tokenizer.bin


..code::ipython3

fromopenvinoimportCore,save_model
fromopenvino_tokenizersimportconnect_models


core=Core()
text_input=["Freemoney!!!"]

ov_tokenizer=core.read_model(model_dir/"openvino_tokenizer.xml")
ov_model=core.read_model(model_dir/"openvino_model.xml")
combined_model=connect_models(ov_tokenizer,ov_model)
save_model(combined_model,model_dir/"combined_openvino_model.xml")

compiled_combined_model=core.compile_model(combined_model)
openvino_output=compiled_combined_model(text_input)

print(f"Logits:{openvino_output['logits']}")


..parsed-literal::

Logits:[[1.2007061-1.4698029]]


Conclusion
----------

`backtotop⬆️<#table-of-contents>`__

TheOpenVINOTokenizersintegratetextprocessingoperationsintothe
OpenVINOecosystem.EnablingtheconversionofHuggingFacetokenizers
intoOpenVINOmodels,thelibraryallowsefficientdeploymentofdeep
learningpipelinesacrossvariedenvironments.Thefeatureofcombining
tokenizersandmodelsnotonlysimplifiesmemorymanagementbutalso
helpstostreamlinemodelusageanddeployment.

Links
-----

`backtotop⬆️<#table-of-contents>`__

-`Installationinstructionsfordifferent
environments<https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#installation>`__
-`SupportedTokenizer
Types<https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#supported-tokenizer-types>`__
-`OpenVINO.GenAIrepositorywiththeC++exampleofOpenVINO
Tokenizers
usage<https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/cpp/greedy_causal_lm>`__
-`HuggingFaceTokenizersComparison
Table<https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#output-match-by-model>`__
