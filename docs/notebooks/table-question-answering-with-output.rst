TableQuestionAnsweringusingTAPASandOpenVINO™
==================================================

TableQuestionAnswering(TableQA)istheansweringaquestionaboutan
informationonagiventable.YoucanusetheTableQuestionAnswering
modelstosimulateSQLexecutionbyinputtingatable.

Inthistutorialwedemonstratehowtoperformtablequestionanswering
usingOpenVINO.Thisexamplebasedon`TAPASbasemodelfine-tunedon
WikiTableQuestions
(WTQ)<https://huggingface.co/google/tapas-base-finetuned-wtq>`__that
isbasedonthepaper`TAPAS:WeaklySupervisedTableParsingvia
Pre-training<https://arxiv.org/abs/2004.02349#:~:text=Answering%20natural%20language%20questions%20over,denotations%20instead%20of%20logical%20forms>`__.

Answeringnaturallanguagequestionsovertablesisusuallyseenasa
semanticparsingtask.Toalleviatethecollectioncostoffulllogical
forms,onepopularapproachfocusesonweaksupervisionconsistingof
denotationsinsteadoflogicalforms.However,trainingsemanticparsers
fromweaksupervisionposesdifficulties,andinaddition,thegenerated
logicalformsareonlyusedasanintermediatesteppriortoretrieving
thedenotation.In`this
paper<https://arxiv.org/pdf/2004.02349.pdf>`__,itispresentedTAPAS,
anapproachtoquestionansweringovertableswithoutgeneratinglogical
forms.TAPAStrainsfromweaksupervision,andpredictsthedenotation
byselectingtablecellsandoptionallyapplyingacorresponding
aggregationoperatortosuchselection.TAPASextendsBERT’s
architecturetoencodetablesasinput,initializesfromaneffective
jointpre-trainingoftextsegmentsandtablescrawledfromWikipedia,
andistrainedend-to-end.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Usetheoriginalmodeltorunan
inference<#use-the-original-model-to-run-an-inference>`__
-`ConverttheoriginalmodeltoOpenVINOIntermediateRepresentation
(IR)
format<#convert-the-original-model-to-openvino-intermediate-representation-ir-format>`__
-`RuntheOpenVINOmodel<#run-the-openvino-model>`__
-`Interactiveinference<#interactive-inference>`__

Prerequisites
~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3


%pipinstall-qtorch"transformers>=4.31.0""torch>=2.1"--extra-index-urlhttps://download.pytorch.org/whl/cpu
%pipinstall-q"openvino>=2023.2.0""gradio>=4.0.2"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importtorch
fromtransformersimportTapasForQuestionAnswering
fromtransformersimportTapasTokenizer
fromtransformersimportpipeline
importpandasaspd


..parsed-literal::

2024-07-1304:03:47.892996:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1304:03:47.927787:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1304:03:48.586538:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT


Use``TapasForQuestionAnswering.from_pretrained``todownloada
pretrainedmodeland``TapasTokenizer.from_pretrained``togeta
tokenizer.

..code::ipython3

model=TapasForQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")
tokenizer=TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")

data={
"Actors":["BradPitt","LeonardoDiCaprio","GeorgeClooney"],
"Numberofmovies":["87","53","69"],
}
table=pd.DataFrame.from_dict(data)
question="howmanymoviesdoesLeonardoDiCapriohave?"
table


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1132:FutureWarning:`resume_download`isdeprecatedandwillberemovedinversion1.0.0.Downloadsalwaysresumewhenpossible.Ifyouwanttoforceanewdownload,use`force_download=True`.
warnings.warn(




..raw::html

<div>
<stylescoped>
.dataframetbodytrth:only-of-type{
vertical-align:middle;
}

.dataframetbodytrth{
vertical-align:top;
}

.dataframetheadth{
text-align:right;
}
</style>
<tableborder="1"class="dataframe">
<thead>
<trstyle="text-align:right;">
<th></th>
<th>Actors</th>
<th>Numberofmovies</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>BradPitt</td>
<td>87</td>
</tr>
<tr>
<th>1</th>
<td>LeonardoDiCaprio</td>
<td>53</td>
</tr>
<tr>
<th>2</th>
<td>GeorgeClooney</td>
<td>69</td>
</tr>
</tbody>
</table>
</div>



Usetheoriginalmodeltorunaninference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuse`this
example<https://huggingface.co/tasks/table-question-answering>`__to
demonstratehowtomakeaninference.Youcanuse``pipeline``from
``transformer``libraryforthispurpose.

..code::ipython3

tqa=pipeline(task="table-question-answering",model=model,tokenizer=tokenizer)
result=tqa(table=table,query=question)
print(f"Theansweris{result['cells'][0]}")


..parsed-literal::

Theansweris53


Youcanreadmoreabouttheinferenceoutputstructurein`this
documentation<https://huggingface.co/docs/transformers/model_doc/tapas>`__.

ConverttheoriginalmodeltoOpenVINOIntermediateRepresentation(IR)format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TheoriginalmodelisaPyTorchmodule,thatcanbeconvertedwith
``ov.convert_model``functiondirectly.Wealsouse``ov.save_model``
functiontoserializetheresultofconversion.

..code::ipython3

importopenvinoasov
frompathlibimportPath


#Definetheinputshape
batch_size=1
sequence_length=29

#Modifytheinputshapeofthedummy_inputdictionary
dummy_input={
"input_ids":torch.zeros((batch_size,sequence_length),dtype=torch.long),
"attention_mask":torch.zeros((batch_size,sequence_length),dtype=torch.long),
"token_type_ids":torch.zeros((batch_size,sequence_length,7),dtype=torch.long),
}


ov_model_xml_path=Path("models/ov_model.xml")

ifnotov_model_xml_path.exists():
ov_model=ov.convert_model(model,example_input=dummy_input)
ov.save_model(ov_model,ov_model_xml_path)


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371:FutureWarning:`_is_quantized_training_enabled`isgoingtobedeprecatedintransformers4.39.0.Pleaseuse`model.hf_quantizer.is_trainable`instead
warnings.warn(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1570:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
self.indices=torch.as_tensor(indices)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1571:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
self.num_segments=torch.as_tensor(num_segments,device=indices.device)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1673:TracerWarning:torch.tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
batch_size=torch.prod(torch.tensor(list(index.batch_shape())))
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1749:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
[torch.as_tensor([-1],dtype=torch.long),torch.as_tensor(vector_shape,dtype=torch.long)],dim=0
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1752:TracerWarning:ConvertingatensortoaPythonlistmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
flat_values=values.reshape(flattened_shape.tolist())
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1754:TracerWarning:ConvertingatensortoaPythonintegermightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
out=torch.zeros(int(flat_index.num_segments),dtype=torch.float,device=flat_values.device)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1762:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
torch.as_tensor(index.batch_shape(),dtype=torch.long),
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1763:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
torch.as_tensor([index.num_segments],dtype=torch.long),
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1764:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
torch.as_tensor(vector_shape,dtype=torch.long),
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1769:TracerWarning:ConvertingatensortoaPythonlistmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
output_values=segment_means.clone().view(new_shape.tolist()).to(values.dtype)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1700:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
batch_shape=torch.as_tensor(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1704:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
num_segments=torch.as_tensor(num_segments)#createarank0tensor(scalar)containingnum_segments(e.g.64)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1715:TracerWarning:ConvertingatensortoaPythonlistmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
new_shape=[int(x)forxinnew_tensor.tolist()]
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1718:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
multiples=torch.cat([batch_shape,torch.as_tensor([1])],dim=0)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1719:TracerWarning:ConvertingatensortoaPythonlistmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
indices=indices.repeat(multiples.tolist())
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:286:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
torch.as_tensor(self.config.max_position_embeddings-1,device=device),position-first_position
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1230:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
indices=torch.min(row_ids,torch.as_tensor(self.config.max_num_rows-1,device=row_ids.device)),
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1235:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
indices=torch.min(column_ids,torch.as_tensor(self.config.max_num_columns-1,device=column_ids.device)),
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1927:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
column_logits+=CLOSE_ENOUGH_TO_LOG_ZERO*torch.as_tensor(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1932:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
column_logits+=CLOSE_ENOUGH_TO_LOG_ZERO*torch.as_tensor(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1968:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
labels_per_column,_=reduce_sum(torch.as_tensor(labels,dtype=torch.float32,device=labels.device),col_index)
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1991:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
torch.as_tensor(labels,dtype=torch.long,device=labels.device),cell_index
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1998:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
column_mask=torch.as_tensor(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:2023:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
selected_column_id=torch.as_tensor(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:2028:TracerWarning:torch.as_tensorresultsareregisteredasconstantsinthetrace.Youcansafelyignorethiswarningifyouusethisfunctiontocreatetensorsoutofconstantvariablesthatwouldbethesameeverytimeyoucallthisfunction.Inanyothercase,thismightcausethetracetobeincorrect.
selected_column_mask=torch.as_tensor(


..parsed-literal::

['input_ids','attention_mask','token_type_ids']


RuntheOpenVINOmodel
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

SelectadevicefromdropdownlistforrunninginferenceusingOpenVINO.

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



Weuse``ov.compile_model``tomakeitreadytouseforloadingona
device.Toprepareinputsusetheoriginal``tokenizer``.

..code::ipython3

inputs=tokenizer(table=table,queries=question,padding="max_length",return_tensors="pt")

compiled_model=core.compile_model(ov_model_xml_path,device.value)
result=compiled_model((inputs["input_ids"],inputs["attention_mask"],inputs["token_type_ids"]))

Nowweshouldpostprocessresults.Forthis,wecanusetheappropriate
partofthecodefrom
`postprocess<https://github.com/huggingface/transformers/blob/fe2877ce21eb75d34d30664757e2727d7eab817e/src/transformers/pipelines/table_question_answering.py#L393>`__
methodof``TableQuestionAnsweringPipeline``.

..code::ipython3

logits=result[0]
logits_aggregation=result[1]


predictions=tokenizer.convert_logits_to_predictions(inputs,torch.from_numpy(result[0]))
answer_coordinates_batch=predictions[0]
aggregators={}
aggregators_prefix={}
answers=[]
forindex,coordinatesinenumerate(answer_coordinates_batch):
cells=[table.iat[coordinate]forcoordinateincoordinates]
aggregator=aggregators.get(index,"")
aggregator_prefix=aggregators_prefix.get(index,"")
answer={
"answer":aggregator_prefix+",".join(cells),
"coordinates":coordinates,
"cells":[table.iat[coordinate]forcoordinateincoordinates],
}
ifaggregator:
answer["aggregator"]=aggregator

answers.append(answer)

print(answers[0]["cells"][0])


..parsed-literal::

53


Also,wecanusetheoriginalpipeline.Forthis,weshouldcreatea
wrapperfor``TapasForQuestionAnswering``classreplacing``forward``
methodtousetheOpenVINOmodelforinferenceandmethodsand
attributesoforiginalmodelclasstobeintegratedintothepipeline.

..code::ipython3

fromtransformersimportTapasConfig


#getconfigforpretrainedmodel
config=TapasConfig.from_pretrained("google/tapas-large-finetuned-wtq")


classTapasForQuestionAnswering(TapasForQuestionAnswering):#itisbettertokeeptheclassnametoavoidwarnings
def__init__(self,ov_model_path):
super().__init__(config)#passconfigfromthepretrainedmodel
self.tqa_model=core.compile_model(ov_model_path,device.value)

defforward(self,input_ids,*,attention_mask,token_type_ids):
results=self.tqa_model((input_ids,attention_mask,token_type_ids))

returntorch.from_numpy(results[0]),torch.from_numpy(results[1])


compiled_model=TapasForQuestionAnswering(ov_model_xml_path)
tqa=pipeline(task="table-question-answering",model=compiled_model,tokenizer=tokenizer)
print(tqa(table=table,query=question)["cells"][0])


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1132:FutureWarning:`resume_download`isdeprecatedandwillberemovedinversion1.0.0.Downloadsalwaysresumewhenpossible.Ifyouwanttoforceanewdownload,use`force_download=True`.
warnings.warn(


..parsed-literal::

53


Interactiveinference
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importrequests

importgradioasgr
importpandasaspd

r=requests.get("https://github.com/openvinotoolkit/openvino_notebooks/files/13215688/eu_city_population_top10.csv")

withopen("eu_city_population_top10.csv","w")asf:
f.write(r.text)


defdisplay_table(csv_file_name):
table=pd.read_csv(csv_file_name.name,delimiter=",")
table=table.astype(str)

returntable


defhighlight_answers(x,coordinates):
highlighted_table=pd.DataFrame("",index=x.index,columns=x.columns)
forcoordinates_iincoordinates:
highlighted_table.iloc[coordinates_i[0],coordinates_i[1]]="background-color:lightgreen"

returnhighlighted_table


definfer(query,csv_file_name):
table=pd.read_csv(csv_file_name.name,delimiter=",")
table=table.astype(str)

result=tqa(table=table,query=query)
table=table.style.apply(highlight_answers,axis=None,coordinates=result["coordinates"])

returnresult["answer"],table


withgr.Blocks(title="TAPASTableQuestionAnswering")asdemo:
withgr.Row():
withgr.Column():
search_query=gr.Textbox(label="Searchquery")
csv_file=gr.File(label="CSVfile")
infer_button=gr.Button("Submit",variant="primary")
withgr.Column():
answer=gr.Textbox(label="Result")
result_csv_file=gr.Dataframe(label="Alldata")

examples=[
[
"Whatisthecitywiththehighestpopulationthatisnotacapital?",
"eu_city_population_top10.csv",
],
["InwhichcountryisMadrid?","eu_city_population_top10.csv"],
[
"Inwhichcitiesisthepopulationgreaterthan2,000,000?",
"eu_city_population_top10.csv",
],
]
gr.Examples(examples,inputs=[search_query,csv_file])

#Callbacks
csv_file.upload(display_table,inputs=csv_file,outputs=result_csv_file)
csv_file.select(display_table,inputs=csv_file,outputs=result_csv_file)
csv_file.change(display_table,inputs=csv_file,outputs=result_csv_file)
infer_button.click(infer,inputs=[search_query,csv_file],outputs=[answer,result_csv_file])

try:
demo.queue().launch(debug=False)
exceptException:
demo.queue().launch(share=True,debug=False)


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>

