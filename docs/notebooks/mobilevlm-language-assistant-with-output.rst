MobilelanguageassistantwithMobileVLMandOpenVINO
=====================================================

`MobileVLM<https://arxiv.org/abs/2312.16886>`__isacompetent
multimodalvisionlanguagemodel(MMVLM)targetedtorunonmobile
devices.Itisanamalgamationofamyriadofarchitecturaldesignsand
techniquesthataremobile-oriented,whichcomprisesasetoflanguage
modelsatthescaleof1.4Band2.7Bparameters,trainedfromscratch,a
multimodalvisionmodelthatispre-trainedintheCLIPfashion,
cross-modalityinteractionviaanefficientprojector.

|image0|

TheMobileVLMarchitecture(right)utilizes
`MobileLLaMA<https://huggingface.co/mtgv/MobileLLaMA-1.4B-Base>`__as
itslanguagemodel,intakes:math:`\mathbf{X}_v`and
:math:`\mathbf{X}_q`whichareimageandlanguageinstructionsas
respectiveinputsandgives:math:`\mathbf{Y}_a`astheoutputlanguage
response.LDPreferstoalightweightdownsampleprojector(left).

Seemoreinformationonofficial
`GitHub<https://github.com/Meituan-AutoML/MobileVLM>`__projectpage
and`paper<https://arxiv.org/abs/2312.16886>`__.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Installrequirements<#install-requirements>`__
-`CloneMobileVLMrepository<#clone-mobilevlm-repository>`__
-`Importrequiredpackages<#import-required-packages>`__
-`Loadthemodel<#load-the-model>`__
-`ConvertmodeltoOpenVINOIntermediateRepresentation
(IR)<#convert-model-to-openvino-intermediate-representation-ir>`__
-`Inference<#inference>`__

-`LoadOpenVINOmodel<#load-openvino-model>`__
-`Prepareinputdata<#prepare-input-data>`__
-`Rungenerationprocess<#run-generation-process>`__

-`Interactiveinference<#interactive-inference>`__

..|image0|image::https://github.com/Meituan-AutoML/MobileVLM/raw/main/assets/mobilevlm_arch.png

Installrequirements
--------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%pipinstall-q"torch>=2.1.0""timm>=0.9.12"--extra-index-url"https://download.pytorch.org/whl/cpu"
%pipinstall-q"transformers>=4.33.1,<4.35.0"accelerate"sentencepiece>=0.1.99""openvino>=2023.2.0""nncf>=2.7.0"ipywidgetsnumpy"gradio>=4.19"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
ERROR:pip'sdependencyresolverdoesnotcurrentlytakeintoaccountallthepackagesthatareinstalled.Thisbehaviouristhesourceofthefollowingdependencyconflicts.
mobileclip0.1.0requirestorch==1.13.1,butyouhavetorch2.3.1+cpuwhichisincompatible.
mobileclip0.1.0requirestorchvision==0.14.1,butyouhavetorchvision0.18.1+cpuwhichisincompatible.
optimum-intel1.19.0.dev0+9ef6766requirestransformers<4.43.0,>=4.36.0,butyouhavetransformers4.33.3whichisincompatible.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


CloneMobileVLMrepository
--------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

frompathlibimportPath
importsys

MOBILEVLM_REPO_DIR=Path("./MobileVLM")
ifnotMOBILEVLM_REPO_DIR.exists():
!gitclone-q"https://github.com/Meituan-AutoML/MobileVLM.git"
sys.path.insert(0,str(MOBILEVLM_REPO_DIR))

Importrequiredpackages
------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importwarnings
importitertools
importgc
fromtypingimportOptional,List,Tuple

frommobilevlm.model.mobilevlmimportload_pretrained_model
frommobilevlm.conversationimportconv_templates,SeparatorStyle
frommobilevlm.utilsimport(
disable_torch_init,
process_images,
tokenizer_image_token,
KeywordsStoppingCriteria,
)
frommobilevlm.constantsimportIMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN
importPIL
importtorch
importtransformers
importnumpyasnp
importgradioasgr
importopenvinoasov
importnncf
importipywidgetsaswidgets


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/utils/generic.py:311:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(
2024-07-1301:08:05.768809:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1301:08:05.803780:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1301:08:06.435873:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/utils/generic.py:311:UserWarning:torch.utils._pytree._register_pytree_nodeisdeprecated.Pleaseusetorch.utils._pytree.register_pytree_nodeinstead.
torch.utils._pytree._register_pytree_node(


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..code::ipython3

MODELS_DIR=Path("./models")
MODEL_PATH="mtgv/MobileVLM-1.7B"

TEMPERATURE=0.2
TOP_P=None
NUM_BEAMS=1
MAX_NEW_TOKENS=512

IMAGE_PATH=MOBILEVLM_REPO_DIR/"assets"/"samples"/"demo.jpg"
PROMPT_STR="Whoistheauthorofthisbook?\nAnswerthequestionusingasinglewordorphrase."

Loadthemodel
--------------

`backtotop⬆️<#table-of-contents>`__

Toloadthemodel,weusepre-defined``load_pretrained_model``function
in``mobilevlm``module.Itreturnsthemodelitself,tokenizer,and
imageprocessortoconvertimagestoappropriatetensors.

..code::ipython3

model_name=MODEL_PATH.split("/")[-1]
disable_torch_init()
withwarnings.catch_warnings():
warnings.simplefilter("ignore")
tokenizer,model,image_processor,_=load_pretrained_model(MODEL_PATH,device="cpu")
model=model.to(dtype=torch.float32)


..parsed-literal::

Youareresizingtheembeddinglayerwithoutprovidinga`pad_to_multiple_of`parameter.Thismeansthatthenewembeddingdimensionwillbe32000.Thismightinducesomeperformancereductionas*TensorCores*willnotbeavailable.Formoredetailsaboutthis,orhelponchoosingthecorrectvalueforresizing,refertothisguide:https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc


ConvertmodeltoOpenVINOIntermediateRepresentation(IR)
----------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defcleanup_torchscript_cache():
"""
Helperforremovingcachedmodelrepresentation
"""
torch._C._jit_clear_class_registry()
torch.jit._recursive.concrete_type_store=torch.jit._recursive.ConcreteTypeStore()
torch.jit._state._clear_class_state()

Forreducingmemoryconsumption,weightscompressionoptimizationcanbe
appliedusing`NNCF<https://github.com/openvinotoolkit/nncf>`__.Weight
compressionaimstoreducethememoryfootprintofamodel.Itcanalso
leadtosignificantperformanceimprovementforlargememory-bound
models,suchasLargeLanguageModels(LLMs).LLMsandothermodels,
whichrequireextensivememorytostoretheweightsduringinference,
canbenefitfromweightcompressioninthefollowingways:

-enablingtheinferenceofexceptionallylargemodelsthatcannotbe
accommodatedinthememoryofthedevice;

-improvingtheinferenceperformanceofthemodelsbyreducingthe
latencyofthememoryaccesswhencomputingtheoperationswith
weights,forexample,Linearlayers.

`NeuralNetworkCompressionFramework
(NNCF)<https://github.com/openvinotoolkit/nncf>`__provides4-bit/
8-bitmixedweightquantizationasacompressionmethodprimarily
designedtooptimizeLLMs.Themaindifferencebetweenweights
compressionandfullmodelquantization(post-trainingquantization)is
thatactivationsremainfloating-pointinthecaseofweights
compressionwhichleadstoabetteraccuracy.Weightcompressionfor
LLMsprovidesasolidinferenceperformanceimprovementwhichisonpar
withtheperformanceofthefullmodelquantization.Inaddition,weight
compressionisdata-freeanddoesnotrequireacalibrationdataset,
makingiteasytouse.

``nncf.compress_weights``functioncanbeusedforperformingweights
compression.ThefunctionacceptsanOpenVINOmodelandother
compressionparameters.ComparedtoINT8compression,INT4compression
improvesperformanceevenmore,butintroducesaminordropin
predictionquality.

Moredetailsaboutweightscompression,canbefoundin`OpenVINO
documentation<https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__.

PleaseselectbelowwhetheryouwouldliketorunINT4weight
compressioninsteadofINT8weightcompression.

..code::ipython3

compression_mode=widgets.Dropdown(
options=["INT4","INT8"],
value="INT4",
description="Compressionmode:",
disabled=False,
)

compression_mode




..parsed-literal::

Dropdown(description='Compressionmode:',options=('INT4','INT8'),value='INT4')



..code::ipython3

stage1_xml_path=MODELS_DIR/f"stage1_{compression_mode.value}.xml"
stage2_xml_path=MODELS_DIR/f"stage2_{compression_mode.value}.xml"

..code::ipython3

ifcompression_mode.value=="INT4":
wc_parameters=dict(mode=nncf.CompressWeightsMode.INT4_ASYM,group_size=128,ratio=0.8)
else:
wc_parameters=dict(mode=nncf.CompressWeightsMode.INT8)

..code::ipython3

classModelWrapper(torch.nn.Module):
def__init__(self,model):
super().__init__()
self.model=model

defforward(
self,
input_ids:torch.LongTensor=None,
attention_mask:Optional[torch.Tensor]=None,
past_key_values:Optional[List[torch.FloatTensor]]=None,
inputs_embeds:Optional[torch.FloatTensor]=None,
):
outputs=self.model.model(
input_ids=input_ids,
attention_mask=attention_mask,
past_key_values=past_key_values,
inputs_embeds=inputs_embeds,
)
hidden_states=outputs[0]
logits=self.model.lm_head(hidden_states)

return(logits,)+outputs[1:]

..code::ipython3

defset_input_names(model,past_key_values):
input_names=[
"input_ids",
"attention_mask",
*itertools.chain.from_iterable([f"past_key_values.{idx}.key",f"past_key_values.{idx}.value"]foridx,_inenumerate(past_key_values)),
]
assertlen(input_names)==len(model.inputs)
for_input,input_nameinzip(model.inputs,input_names):
_input.get_tensor().set_names({input_name})

..code::ipython3

defset_output_names(model,past_key_values):
output_names=[
"logits",
*itertools.chain.from_iterable([f"present.{idx}.key",f"present.{idx}.value"]foridx,_inenumerate(past_key_values)),
]
assertlen(output_names)==len(model.outputs)
forout,out_nameinzip(ov_model.outputs,output_names):
out.get_tensor().set_names({out_name})

..code::ipython3

example_input={
"inputs_embeds":torch.zeros((1,205,2048)),
"attention_mask":torch.ones((1,205),dtype=torch.long),
}

wrapped=ModelWrapper(model)
past_key_values=wrapped(**example_input)[1]

ifnotstage1_xml_path.exists():
ov_model=ov.convert_model(wrapped,example_input=example_input)
set_output_names(ov_model,past_key_values)
ov_model=nncf.compress_weights(ov_model,**wc_parameters)
ov.save_model(ov_model,stage1_xml_path)
cleanup_torchscript_cache()
delov_model
gc.collect()


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py:595:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifinput_shape[-1]>1:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py:119:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifseq_len>self.max_seq_len_cached:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py:348:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_weights.size()!=(bsz,self.num_heads,q_len,kv_seq_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py:355:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattention_mask.size()!=(bsz,1,q_len,kv_seq_len):
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py:365:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifattn_output.size()!=(bsz,self.num_heads,q_len,self.head_dim):


..parsed-literal::

['attention_mask','inputs_embeds']



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│24%(43/169)│20%(42/168)│
├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│4│76%(126/169)│80%(126/168)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..code::ipython3

example_input={
"input_ids":torch.ones((1,1),dtype=torch.long),
"past_key_values":past_key_values,
"attention_mask":torch.ones((1,past_key_values[-1][-1].shape[-2]+1),dtype=torch.long),
}

ifnotstage2_xml_path.exists():
ov_model=ov.convert_model(
wrapped,
example_input=example_input,
)
set_input_names(ov_model,past_key_values)
set_output_names(ov_model,past_key_values)
ov_model=nncf.compress_weights(ov_model,**wc_parameters)
ov.save_model(ov_model,stage2_xml_path)
cleanup_torchscript_cache()
delov_model
gc.collect()


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:165:UserWarning:The.gradattributeofaTensorthatisnotaleafTensorisbeingaccessed.Its.gradattributewon'tbepopulatedduringautograd.backward().Ifyouindeedwantthe.gradfieldtobepopulatedforanon-leafTensor,use.retain_grad()onthenon-leafTensor.Ifyouaccessthenon-leafTensorbymistake,makesureyouaccesstheleafTensorinstead.Seegithub.com/pytorch/pytorch/pull/30531formoreinformations.(Triggeredinternallyataten/src/ATen/core/TensorBody.h:489.)
ifa.gradisnotNone:


..parsed-literal::

['input_ids','attention_mask','past_key_values']



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:Statisticsofthebitwidthdistribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│Numbits(N)│%allparameters(layers)│%ratio-definingparameters(layers)│
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│8│28%(44/170)│20%(42/168)│
├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│4│72%(126/170)│80%(126/168)│
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..code::ipython3

prepare_inputs_labels_for_multimodal=model.prepare_inputs_labels_for_multimodal
prepare_inputs_for_generation=model.prepare_inputs_for_generation
config=model.config
config.save_pretrained(MODELS_DIR)

..code::ipython3

delwrapped
delmodel
gc.collect();

Inference
---------

`backtotop⬆️<#table-of-contents>`__

``OVMobileLlamaForCausalLM``classprovidesease-to-useinterfacefor
usingmodelingenerationscenario.Itisbasedon
``transformers.generation.GenerationMixin``thatgivesusopportunityto
reuseallreachcapabilitiesforgenerationimplementedinHuggingFace
Transformerslibrary.Moredetailsaboutthisinterfacecanbefoundin
`HuggingFace
documentation<https://huggingface.co/docs/transformers/main_classes/text_generation>`__.

..code::ipython3

classOVMobileLlamaForCausalLM(transformers.GenerationMixin):
def__init__(self,stage1_path,stage2_path,device):
self.stage1=core.compile_model(stage1_path,device)
self.stage2=core.read_model(stage2_path)

self.generation_config=transformers.GenerationConfig.from_model_config(config)
self.config=transformers.AutoConfig.from_pretrained(MODELS_DIR)
self.main_input_name="input_ids"
self.device=torch.device("cpu")
self.prepare_inputs_for_generation=prepare_inputs_for_generation
self.num_pkv=2
self.input_names={key.get_any_name():idxforidx,keyinenumerate(self.stage2.inputs)}
self.output_names={key.get_any_name():idxforidx,keyinenumerate(self.stage2.outputs)}
self.key_value_input_names=[keyforkeyinself.input_namesif"key_values"inkey]
self.key_value_output_names=[keyforkeyinself.output_namesif"present"inkey]
stage2=core.compile_model(self.stage2,device)
self.request=stage2.create_infer_request()
self._supports_cache_class=False

defcan_generate(self):
"""ReturnsTruetovalidatethecheckthatthemodelusing`GenerationMixin.generate()`canindeedgenerate."""
returnTrue

def__call__(
self,
input_ids:torch.LongTensor,
images:torch.Tensor,
attention_mask:Optional[torch.LongTensor]=None,
prefix_mask:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
**kwargs,
)->transformers.modeling_outputs.CausalLMOutputWithPast:
returnself.forward(input_ids,images,attention_mask,prefix_mask,past_key_values)

defforward(
self,
input_ids:torch.LongTensor,
images:torch.Tensor,
attention_mask:Optional[torch.LongTensor]=None,
prefix_mask:Optional[torch.LongTensor]=None,
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]]=None,
**kwargs,
)->transformers.modeling_outputs.CausalLMOutputWithPast:
"""Generalinferencemethod"""
inputs={}
ifpast_key_valuesisnotNone:
#Flattenthepast_key_values
attention_mask=torch.ones(
(input_ids.shape[0],past_key_values[-1][-1].shape[-2]+1),
dtype=input_ids.dtype,
)
past_key_values=tuple(past_key_valueforpkv_per_layerinpast_key_valuesforpast_key_valueinpkv_per_layer)
#Addthepast_key_valuestothedecoderinputs
inputs=dict(zip(self.key_value_input_names,past_key_values))

else:
returnself.forward_with_image(input_ids,images,attention_mask)
inputs["input_ids"]=np.array(input_ids)

if"attention_mask"inself.input_names:
inputs["attention_mask"]=np.array(attention_mask)

#Runinference
self.request.start_async(inputs,share_inputs=True)
self.request.wait()

logits=torch.from_numpy(self.request.get_tensor("logits").data)

#Tupleoflengthequalto:numberoflayer*numberofpast_key_valueperdecoderlayer(2correspondstotheself-attentionlayer)
past_key_values=tuple(self.request.get_tensor(key).dataforkeyinself.key_value_output_names)
#Tupleoftupleoflength`n_layers`,witheachtupleoflengthequalto2(k/vofself-attention)

past_key_values=tuple(past_key_values[i:i+self.num_pkv]foriinrange(0,len(past_key_values),self.num_pkv))

returntransformers.modeling_outputs.CausalLMOutputWithPast(logits=logits,past_key_values=past_key_values)

defforward_with_image(self,input_ids,images,attention_mask):
"""Firststepinferencemethod,thatresolvesmultimodaldata"""
_,attention_mask,_,input_embed,_=prepare_inputs_labels_for_multimodal(input_ids,attention_mask,images=images,past_key_values=None,labels=None)
outs=self.stage1({"inputs_embeds":input_embed,"attention_mask":attention_mask})
logits=outs[0]
pkv=list(outs.values())[1:]
pkv=tuple(pkv[i:i+self.num_pkv]foriinrange(0,len(pkv),self.num_pkv))
returntransformers.modeling_outputs.CausalLMOutputWithPast(logits=torch.from_numpy(logits),past_key_values=pkv)

Now,whenwehavemodelanddefinedgenerationpipeline,wecanrun
modelinference.

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO.

..code::ipython3

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



LoadOpenVINOmodel
~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ov_model=OVMobileLlamaForCausalLM(stage1_xml_path,stage2_xml_path,device.value)

Prepareinputdata
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

images=[PIL.Image.open(IMAGE_PATH).convert("RGB")]
images_tensor=process_images(images,image_processor,transformers.AutoConfig.from_pretrained(MODELS_DIR))

..code::ipython3

conv=conv_templates["v1"].copy()
conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+"\n"+PROMPT_STR)
conv.append_message(conv.roles[1],None)
prompt=conv.get_prompt()
stop_str=conv.sepifconv.sep_style!=SeparatorStyle.TWOelseconv.sep2
input_ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0)
stopping_criteria=KeywordsStoppingCriteria([stop_str],tokenizer,input_ids)

..code::ipython3

print(PROMPT_STR)
images[0]


..parsed-literal::

Whoistheauthorofthisbook?
Answerthequestionusingasinglewordorphrase.




..image::mobilevlm-language-assistant-with-output_files/mobilevlm-language-assistant-with-output_32_1.png



Rungenerationprocess
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

output_ids=ov_model.generate(
input_ids,
images=images_tensor,
do_sample=TrueifTEMPERATURE>0elseFalse,
temperature=TEMPERATURE,
top_p=TOP_P,
num_beams=NUM_BEAMS,
max_new_tokens=MAX_NEW_TOKENS,
use_cache=True,
stopping_criteria=[stopping_criteria],
)
input_token_len=input_ids.shape[1]
n_diff_input_output=(input_ids!=output_ids[:,:input_token_len]).sum().item()
ifn_diff_input_output>0:
print(f"[Warning]{n_diff_input_output}output_idsarenotthesameastheinput_ids")
outputs=tokenizer.batch_decode(output_ids[:,input_token_len:],skip_special_tokens=True)[0]
outputs=outputs.strip()
ifoutputs.endswith(stop_str):
outputs=outputs[:-len(stop_str)]
print(f"🚀{model_name}withOpenVINO:{outputs.strip()}\n")


..parsed-literal::

🚀MobileVLM-1.7BwithOpenVINO:SusanWiseBauer



Interactiveinference
---------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

defgenerate(img,prompt):
images_tensor=process_images([img],image_processor,transformers.AutoConfig.from_pretrained(MODELS_DIR))
prompt=DEFAULT_IMAGE_TOKEN+"\n"+prompt
conv=conv_templates["v1"].copy()
conv.append_message(conv.roles[0],prompt)
conv.append_message(conv.roles[1],None)
prompt=conv.get_prompt()
stop_str=conv.sepifconv.sep_style!=SeparatorStyle.TWOelseconv.sep2
input_ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0)
stopping_criteria=KeywordsStoppingCriteria([stop_str],tokenizer,input_ids)

output_ids=ov_model.generate(
input_ids,
images=images_tensor,
do_sample=TrueifTEMPERATURE>0elseFalse,
temperature=TEMPERATURE,
top_p=TOP_P,
num_beams=NUM_BEAMS,
max_new_tokens=MAX_NEW_TOKENS,
use_cache=True,
stopping_criteria=[stopping_criteria],
)
input_token_len=input_ids.shape[1]
outputs=tokenizer.batch_decode(output_ids[:,input_token_len:],skip_special_tokens=True)[0]
outputs=outputs.strip()
ifoutputs.endswith(stop_str):
outputs=outputs[:-len(stop_str)]

returnoutputs.strip()


demo=gr.Interface(
generate,
[gr.Image(label="Image",type="pil"),gr.Textbox(label="Prompt")],
gr.Textbox(),
examples=[
[
str(IMAGE_PATH),
PROMPT_STR,
]
],
allow_flagging="never",
)

try:
demo.launch(debug=False)
exceptException:
demo.launch(debug=False,share=True)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>

