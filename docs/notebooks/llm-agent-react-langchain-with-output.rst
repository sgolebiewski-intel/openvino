CreateReActAgentusingOpenVINOandLangChain
===============================================

LLMarelimitedtotheknowledgeonwhichtheyhavebeentrainedandthe
additionalknowledgeprovidedascontext,asaresult,ifausefulpiece
ofinformationismissingtheprovidedknowledge,themodelcannot“go
around”andtrytofinditinothersources.Thisisthereasonwhywe
needtointroducetheconceptofAgents.

Thecoreideaofagentsistousealanguagemodeltochooseasequence
ofactionstotake.Inagents,alanguagemodelisusedasareasoning
enginetodeterminewhichactionstotakeandinwhichorder.Agentscan
beseenasapplicationspoweredbyLLMsandintegratedwithasetof
toolslikesearchengines,databases,websites,andsoon.Withinan
agent,theLLMisthereasoningenginethat,basedontheuserinput,is
abletoplanandexecuteasetofactionsthatareneededtofulfillthe
request.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/22fa5396-8381-400f-a78f-97e25d57d807
:alt:agent

agent

`LangChain<https://python.langchain.com/docs/get_started/introduction>`__
isaframeworkfordevelopingapplicationspoweredbylanguagemodels.
LangChaincomeswithanumberofbuilt-inagentsthatareoptimizedfor
differentusecases.

ThisnotebookexploreshowtocreateanAIAgentstepbystepusing
OpenVINOandLangChain.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`Createtools<#create-tools>`__
-`Createprompttemplate<#create-prompt-template>`__
-`CreateLLM<#create-llm>`__

-`Downloadmodel<#select-model>`__
-`Selectinferencedevicefor
LLM<#select-inference-device-for-llm>`__

-`Createagent<#create-agent>`__
-`Runtheagent<#run-agent>`__
-`InteractiveDemo<#interactive-demo>`__

-`Usebuilt-intool<#use-built-in-tool>`__
-`Createcustomizedtools<#create-customized-tools>`__
-`CreateAIagentdemowithGradio
UI<#create-ai-agent-demo-with-gradio-ui>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importos

os.environ["GIT_CLONE_PROTECTION_ACTIVE"]="false"

%pipinstall-Uqpip
%pipuninstall-q-yoptimumoptimum-intel
%pipinstall--pre-Uqopenvinoopenvino-tokenizers[transformers]--extra-index-urlhttps://storage.openvinotoolkit.org/simple/wheels/nightly
%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu\
"git+https://github.com/huggingface/optimum-intel.git"\
"git+https://github.com/openvinotoolkit/nncf.git"\
"torch>=2.1"\
"datasets"\
"accelerate"\
"gradio>=4.19"\
"transformers>=4.38.1""langchain>=0.2.3""langchain-community>=0.2.4""wikipedia"

Createatools
--------------

`backtotop⬆️<#table-of-contents>`__

First,weneedtocreatesometoolstocall.Inthisexample,wewill
create3customfunctionstodobasiccalculation.For`more
information<https://python.langchain.com/docs/modules/tools/>`__on
creatingcustomtools.

..code::ipython3

fromlangchain_core.toolsimporttool


@tool
defmultiply(first_int:int,second_int:int)->int:
"""Multiplytwointegerstogether."""
returnfirst_int*second_int


@tool
defadd(first_int:int,second_int:int)->int:
"Addtwointegers."
returnfirst_int+second_int


@tool
defexponentiate(base:int,exponent:int)->int:
"Exponentiatethebasetotheexponentpower."
returnbase**exponent

..code::ipython3

print(f"nameof`multiply`tool:{multiply.name}")
print(f"descriptionof`multiply`tool:{multiply.description}")


..parsed-literal::

nameof`multiply`tool:multiply
descriptionof`multiply`tool:Multiplytwointegerstogether.


Toolsareinterfacesthatanagent,chain,orLLMcanusetointeract
withtheworld.Theycombineafewthings:

1.Thenameofthetool
2.Adescriptionofwhatthetoolis
3.JSONschemaofwhattheinputstothetoolare
4.Thefunctiontocall
5.Whethertheresultofatoolshouldbereturneddirectlytotheuser

Nowthatwehavecreatedallofthem,andwecancreatealistoftools
thatwewillusedownstream.

..code::ipython3

tools=[multiply,add,exponentiate]

Createprompttemplate
----------------------

`backtotop⬆️<#table-of-contents>`__

Apromptforalanguagemodelisasetofinstructionsorinputprovided
byausertoguidethemodel’sresponse,helpingitunderstandthe
contextandgeneraterelevantandcoherentlanguage-basedoutput,such
asansweringquestions,completingsentences,orengagingina
conversation.

Differentagentshavedifferentpromptingstylesforreasoning.Inthis
example,wewilluse`ReActagent<https://react-lm.github.io/>`__with
itstypicalprompttemplate.Forafulllistofbuilt-inagentssee
`agent
types<https://python.langchain.com/docs/modules/agents/agent_types/>`__.

..figure::https://github.com/openvinotoolkit/openvino_notebooks/assets/91237924/a83bdf7f-bb9d-4b1f-9a0a-3fe4a76ba1ae
:alt:react

react

AReActpromptconsistsoffew-shottask-solvingtrajectories,with
human-writtentextreasoningtracesandactions,aswellasenvironment
observationsinresponsetoactions.ReActpromptingisintuitiveand
flexibletodesign,andachievesstate-of-the-artfew-shotperformances
acrossavarietyoftasks,fromquestionansweringtoonlineshopping!

Inanprompttemplateforagent,``input``isuser’squeryand
``agent_scratchpad``shouldbeasequenceofmessagesthatcontainsthe
previousagenttoolinvocationsandthecorrespondingtooloutputs.

..code::ipython3

PREFIX="""[INST]Respondtothehumanashelpfullyandaccuratelyaspossible.Youhaveaccesstothefollowingtools:"""

FORMAT_INSTRUCTIONS="""Useajsonblobtospecifyatoolbyprovidinganactionkey(toolname)andanaction_inputkey(toolinput).

Valid"action"values:"FinalAnswer"or{tool_names}

ProvideonlyONEactionper$JSON_BLOB,asshown:

```
{{{{
"action":$TOOL_NAME,
"action_input":$INPUT
}}}}
```

Followthisformat:

Question:inputquestiontoanswer
Thought:considerpreviousandsubsequentsteps
Action:
```
$JSON_BLOB
```
Observation:actionresult
...(repeatThought/Action/ObservationNtimes)
Thought:Iknowwhattorespond
Action:
```
{{{{
"action":"FinalAnswer",
"action_input":"Finalresponsetohuman"
}}}}
```[/INST]"""

SUFFIX="""Begin!RemindertoALWAYSrespondwithavalidjsonblobofasingleaction.Usetoolsifnecessary.Responddirectlyifappropriate.FormatisAction:```$JSON_BLOB```thenObservation:.
Thought:[INST]"""

HUMAN_MESSAGE_TEMPLATE="{input}\n\n{agent_scratchpad}"

CreateLLM
----------

`backtotop⬆️<#table-of-contents>`__

LargeLanguageModels(LLMs)areacorecomponentofLangChain.
LangChaindoesnotserveitsownLLMs,butratherprovidesastandard
interfaceforinteractingwithmanydifferentLLMs.Inthisexample,we
select``Mistral-7B-Instruct-v0.3``asLLMinagentpipeline.

-**Mistral-7B-Instruct-v0.3**-TheMistral-7B-Instruct-v0.3Large
LanguageModel(LLM)isaninstructfine-tunedversionofthe
Mistral-7B-v0.3.Youcanfindmoredetailsaboutmodelinthe`model
card<https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>`__,
`paper<https://arxiv.org/abs/2310.06825>`__and`releaseblog
post<https://mistral.ai/news/announcing-mistral-7b/>`__.
>\**Note**:runmodelwithdemo,youwillneedtoacceptlicense
agreement.>Youmustbearegistereduserin🤗HuggingFaceHub.
Pleasevisit`HuggingFacemodel
card<https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>`__,
carefullyreadtermsofusageandclickacceptbutton.Youwillneed
touseanaccesstokenforthecodebelowtorun.Formore
informationonaccesstokens,referto`thissectionofthe
documentation<https://huggingface.co/docs/hub/security-tokens>`__.
>YoucanloginonHuggingFaceHubinnotebookenvironment,using
followingcode:

..code::python

##logintohuggingfacehubtogetaccesstopretrainedmodel

fromhuggingface_hubimportnotebook_login,whoami

try:
whoami()
print('Authorizationtokenalreadyprovided')
exceptOSError:
notebook_login()

Downloadmodel
~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

TorunLLMlocally,wehavetodownloadthemodelinthefirststep.It
ispossibleto`exportyour
model<https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export>`__
totheOpenVINOIRformatwiththeCLI,andloadthemodelfromlocal
folder.

..code::ipython3

frompathlibimportPath

model_id="mistralai/Mistral-7B-Instruct-v0.3"
model_path="Mistral-7B-Instruct-v0.3-ov-int4"

ifnotPath(model_path).exists():
!optimum-cliexportopenvino--model{model_id}--tasktext-generation-with-past--trust-remote-code--weight-formatint4{model_path}

SelectinferencedeviceforLLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importopenvinoasov
importipywidgetsaswidgets

core=ov.Core()

support_devices=core.available_devices
if"NPU"insupport_devices:
support_devices.remove("NPU")

device=widgets.Dropdown(
options=support_devices+["AUTO"],
value="CPU",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',options=('CPU','GPU','AUTO'),value='CPU')



OpenVINOmodelscanberunlocallythroughthe``HuggingFacePipeline``
classinLangChain.TodeployamodelwithOpenVINO,youcanspecifythe
``backend="openvino"``parametertotriggerOpenVINOasbackend
inferenceframework.For`more
information<https://python.langchain.com/docs/integrations/llms/openvino/>`__.

..code::ipython3

fromlangchain_community.llms.huggingface_pipelineimportHuggingFacePipeline
fromtransformers.generation.stopping_criteriaimportStoppingCriteriaList,StoppingCriteria


classStopSequenceCriteria(StoppingCriteria):
"""
Thisclasscanbeusedtostopgenerationwheneverasequenceoftokensisencountered.

Args:
stop_sequences(`str`or`List[str]`):
Thesequence(orlistofsequences)onwhichtostopexecution.
tokenizer:
Thetokenizerusedtodecodethemodeloutputs.
"""

def__init__(self,stop_sequences,tokenizer):
ifisinstance(stop_sequences,str):
stop_sequences=[stop_sequences]
self.stop_sequences=stop_sequences
self.tokenizer=tokenizer

def__call__(self,input_ids,scores,**kwargs)->bool:
decoded_output=self.tokenizer.decode(input_ids.tolist()[0])
returnany(decoded_output.endswith(stop_sequence)forstop_sequenceinself.stop_sequences)


ov_config={"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1","CACHE_DIR":""}
stop_tokens=["Observation:"]

ov_llm=HuggingFacePipeline.from_model_id(
model_id=model_path,
task="text-generation",
backend="openvino",
model_kwargs={
"device":device.value,
"ov_config":ov_config,
"trust_remote_code":True,
},
pipeline_kwargs={"max_new_tokens":2048},
)
ov_llm=ov_llm.bind(skip_prompt=True,stop=["Observation:"])

tokenizer=ov_llm.pipeline.tokenizer
ov_llm.pipeline._forward_params["stopping_criteria"]=StoppingCriteriaList([StopSequenceCriteria(stop_tokens,tokenizer)])


..parsed-literal::

2024-06-0723:17:16.804739:Itensorflow/core/util/port.cc:111]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-06-0723:17:16.807973:Itensorflow/tsl/cuda/cudart_stub.cc:28]Couldnotfindcudadriversonyourmachine,GPUwillnotbeused.
2024-06-0723:17:16.850235:Etensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342]UnabletoregistercuDNNfactory:AttemptingtoregisterfactoryforplugincuDNNwhenonehasalreadybeenregistered
2024-06-0723:17:16.850258:Etensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609]UnabletoregistercuFFTfactory:AttemptingtoregisterfactoryforplugincuFFTwhenonehasalreadybeenregistered
2024-06-0723:17:16.850290:Etensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518]UnabletoregistercuBLASfactory:AttemptingtoregisterfactoryforplugincuBLASwhenonehasalreadybeenregistered
2024-06-0723:17:16.859334:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-06-0723:17:17.692415:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
Youset`add_prefix_space`.Thetokenizerneedstobeconvertedfromtheslowtokenizers
Theargument`trust_remote_code`istobeusedalongwithexport=True.Itwillbeignored.
CompilingthemodeltoGPU...


Youcangetadditionalinferencespeedimprovementwith`Dynamic
QuantizationofactivationsandKV-cachequantizationon
CPU<https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-hf.html#enabling-openvino-runtime-optimizations>`__.
Theseoptionscanbeenabledwith``ov_config``asfollows:

..code::ipython3

ov_config={
"KV_CACHE_PRECISION":"u8",
"DYNAMIC_QUANTIZATION_GROUP_SIZE":"32",
"PERFORMANCE_HINT":"LATENCY",
"NUM_STREAMS":"1",
"CACHE_DIR":"",
}

Createagent
------------

`backtotop⬆️<#table-of-contents>`__

Nowthatwehavedefinedthetools,prompttemplateandLLM,wecan
createtheagent_executor.

Theagentexecutoristheruntimeforanagent.Thisiswhatactually
callstheagent,executestheactionsitchooses,passestheaction
outputsbacktotheagent,andrepeats.

..code::ipython3

fromlangchain.agentsimportAgentExecutor,StructuredChatAgent

agent=StructuredChatAgent.from_llm_and_tools(
ov_llm,
tools,
prefix=PREFIX,
suffix=SUFFIX,
human_message_template=HUMAN_MESSAGE_TEMPLATE,
format_instructions=FORMAT_INSTRUCTIONS,
)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

Runtheagent
-------------

`backtotop⬆️<#table-of-contents>`__

Wecannowruntheagentwithamathquery.Beforegettingthefinal
answer,aagentexecutorwillalsoproduceintermediatestepsof
reasoningandactions.Theformatofthesemessageswillfollowyour
prompttemplate.

..code::ipython3

agent_executor.invoke({"input":"Take3tothefifthpowerandmultiplythatbythesumoftwelveandthree,thensquarethewholeresult"})


..parsed-literal::



>EnteringnewAgentExecutorchain...
Thought:Icanusetheexponentiateandaddtoolstosolvethefirstpart,andthenusethemultiplytoolforthesecondpart,andfinallytheexponentiatetoolagaintosquaretheresult.

Action:
```
{
"action":"exponentiate",
"action_input":{"base":3,"exponent":5}
}
```
Observation:
Observation:243
Thought:NowIneedtoaddtwelveandthree

Action:
```
{
"action":"add",
"action_input":{"first_int":12,"second_int":3}
}
```
Observation:
Observation:15
Thought:NowIneedtomultiplytheresultby243

Action:
```
{
"action":"multiply",
"action_input":{"first_int":243,"second_int":15}
}
```
Observation:
Observation:3645
Thought:Finally,Ineedtosquaretheresult

Action:
```
{
"action":"exponentiate",
"action_input":{"base":3645,"exponent":2}
}
```
Observation:
Observation:13286025
Thought:Iknowwhattorespond

Action:
```
{
"action":"FinalAnswer",
"action_input":"Thefinalansweris13286025"
}
```

>Finishedchain.




..parsed-literal::

{'input':'Take3tothefifthpowerandmultiplythatbythesumoftwelveandthree,thensquarethewholeresult',
'output':'Thefinalansweris13286025'}



InteractiveDemo
----------------

`backtotop⬆️<#table-of-contents>`__

Let’screateainteractiveagentusing
`Gradio<https://www.gradio.app/>`__.

Usebuilt-intools
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

LangChainhasprovidedalistofall`built-in
tools<https://python.langchain.com/docs/integrations/tools/>`__.In
thisexample,wewilluse``Wikipedia``pythonpackagetoquerykey
wordsgeneratedbyagent.

..code::ipython3

fromlangchain_community.toolsimportWikipediaQueryRun
fromlangchain_community.utilitiesimportWikipediaAPIWrapper
fromlangchain_core.pydantic_v1importBaseModel,Field
fromlangchain_core.callbacksimportCallbackManagerForToolRun
fromtypingimportOptional


classWikipediaQueryRunWrapper(WikipediaQueryRun):
def_run(
self,
text:str,
run_manager:Optional[CallbackManagerForToolRun]=None,
)->str:
"""UsetheWikipediatool."""
returnself.api_wrapper.run(text)


api_wrapper=WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=1000)


classWikiInputs(BaseModel):
"""inputstothewikipediatool."""

text:str=Field(description="querytolookuponwikipedia.")


wikipedia=WikipediaQueryRunWrapper(
description="AwrapperaroundWikipedia.Usefulforwhenyouneedtoanswergeneralquestionsaboutpeople,places,companies,facts,historicalevents,orothersubjects.Inputshouldbeasearchquery.",
args_schema=WikiInputs,
api_wrapper=api_wrapper,
)

..code::ipython3

wikipedia.invoke({"text":"OpenVINO"})




..parsed-literal::

'Page:OpenVINO\nSummary:OpenVINOisanopen-sourcesoftwaretoolkitforoptimizinganddeployingdeeplearningmodels.ItenablesprogrammerstodevelopscalableandefficientAIsolutionswithrelativelyfewlinesofcode.Itsupportsseveralpopularmodelformatsandcategories,suchaslargelanguagemodels,computervision,andgenerativeAI.\nActivelydevelopedbyIntel,itprioritizeshigh-performanceinferenceonIntelhardwarebutalsosupportsARM/ARM64processorsandencouragescontributorstoaddnewdevicestotheportfolio.\nBasedinC++,itoffersthefollowingAPIs:C/C++,Python,andNode.js(anearlypreview).\nOpenVINOiscross-platformandfreeforuseunderApacheLicense2.0.\n\nPage:StableDiffusion\nSummary:StableDiffusionisadeeplearning,text-to-imagemodelreleasedin2022basedondiffusiontechniques.Itisconsideredtobeapartoftheongoingartificialintelligenceboom.\nItisprimarilyusedtogeneratedetailedimagesconditionedontextdescriptions,t'



Createcustomizedtools
~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Inthisexamples,wewillcreate2customizedtoolsfor
``imagegeneration``and``weatherqurey``.

..code::ipython3

importurllib.parse
importjson5


@tool
defpainting(prompt:str)->str:
"""
AIpainting(imagegeneration)service,inputtextdescription,andreturntheimageURLdrawnbasedontextinformation.
"""
prompt=urllib.parse.quote(prompt)
returnjson5.dumps({"image_url":f"https://image.pollinations.ai/prompt/{prompt}"},ensure_ascii=False)


painting.invoke({"prompt":"acat"})




..parsed-literal::

'{image_url:"https://image.pollinations.ai/prompt/a%20cat"}'



..code::ipython3

@tool
defweather(
city_name:str,
)->str:
"""
Getthecurrentweatherfor`city_name`
"""

ifnotisinstance(city_name,str):
raiseTypeError("Citynamemustbeastring")

key_selection={
"current_condition":[
"temp_C",
"FeelsLikeC",
"humidity",
"weatherDesc",
"observation_time",
],
}
importrequests

resp=requests.get(f"https://wttr.in/{city_name}?format=j1")
resp.raise_for_status()
resp=resp.json()
ret={k:{_v:resp[k][0][_v]for_vinv}fork,vinkey_selection.items()}

returnstr(ret)


weather.invoke({"city_name":"London"})




..parsed-literal::

"{'current_condition':{'temp_C':'9','FeelsLikeC':'8','humidity':'93','weatherDesc':[{'value':'Sunny'}],'observation_time':'04:39AM'}}"



CreateAIagentdemowithGradioUI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

tools=[wikipedia,painting,weather]

agent=StructuredChatAgent.from_llm_and_tools(
ov_llm,
tools,
prefix=PREFIX,
suffix=SUFFIX,
human_message_template=HUMAN_MESSAGE_TEMPLATE,
format_instructions=FORMAT_INSTRUCTIONS,
)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

..code::ipython3

importgradioasgr

examples=[
["BasedoncurrentweatherinLondon,showmeapictureofBigBenthroughitsURL"],
["WhatisOpenVINO?"],
["CreateanimageofpinkcatandreturnitsURL"],
["HowmanypeopleliveinCanada?"],
["WhatistheweatherlikeinNewYorknow?"],
]


defpartial_text_processor(partial_text,new_text):
"""
helperforupdatingpartiallygeneratedanswer,usedbydefault

Params:
partial_text:textbufferforstoringprevioslygeneratedtext
new_text:textupdateforthecurrentstep
Returns:
updatedtextstring

"""
partial_text+=new_text
returnpartial_text


defuser(message,history):
"""
callbackfunctionforupdatingusermessagesininterfaceonsubmitbuttonclick

Params:
message:currentmessage
history:conversationhistory
Returns:
None
"""
#Appendtheuser'smessagetotheconversationhistory
return"",history+[[message,""]]


defbot(history):
"""
callbackfunctionforrunningchatbotonsubmitbuttonclick

Params:
history:conversationhistory

"""
partial_text=""

fornew_textinagent_executor.stream(
{"input":history[-1][0]},
):
if"output"innew_text.keys():
partial_text=partial_text_processor(partial_text,new_text["output"])
history[-1][1]=partial_text
yieldhistory


defrequest_cancel():
ov_llm.pipeline.model.request.cancel()


withgr.Blocks(
theme=gr.themes.Soft(),
css=".disclaimer{font-variant-caps:all-small-caps;}",
)asdemo:
names=[tool.namefortoolintools]
gr.Markdown(f"""<h1><center>OpenVINOAgentfor{str(names)}</center></h1>""")
chatbot=gr.Chatbot(height=500)
withgr.Row():
withgr.Column():
msg=gr.Textbox(
label="ChatMessageBox",
placeholder="ChatMessageBox",
show_label=False,
container=False,
)
withgr.Column():
withgr.Row():
submit=gr.Button("Submit")
stop=gr.Button("Stop")
clear=gr.Button("Clear")
gr.Examples(examples,inputs=msg,label="Clickonanyexampleandpressthe'Submit'button")

submit_event=msg.submit(
fn=user,
inputs=[msg,chatbot],
outputs=[msg,chatbot],
queue=False,
).then(
fn=bot,
inputs=[
chatbot,
],
outputs=chatbot,
queue=True,
)
submit_click_event=submit.click(
fn=user,
inputs=[msg,chatbot],
outputs=[msg,chatbot],
queue=False,
).then(
fn=bot,
inputs=[
chatbot,
],
outputs=chatbot,
queue=True,
)
stop.click(
fn=request_cancel,
inputs=None,
outputs=None,
cancels=[submit_event,submit_click_event],
queue=False,
)
clear.click(lambda:None,None,chatbot,queue=False)

#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#ifyouhaveanyissuetolaunchonyourplatform,youcanpassshare=Truetolaunchmethod:
#demo.launch(share=True)
#itcreatesapubliclyshareablelinkfortheinterface.Readmoreinthedocs:https://gradio.app/docs/
demo.launch()


..parsed-literal::



>EnteringnewAgentExecutorchain...
Thought:IneedtousetheweathertooltogetthecurrentweatherinLondon,thenusethepaintingtooltogenerateapictureofBigBenbasedontheweatherinformation.

Action:
```
{
"action":"weather",
"action_input":"London"
}
```

Observation:
Observation:{'current_condition':{'temp_C':'9','FeelsLikeC':'8','humidity':'93','weatherDesc':[{'value':'Sunny'}],'observation_time':'04:39AM'}}
Thought:IhavethecurrentweatherinLondon.NowIcanusethepaintingtooltogenerateapictureofBigBenbasedontheweatherinformation.

Action:
```
{
"action":"painting",
"action_input":"BigBen,sunnyday"
}
```

Observation:
Observation:{image_url:"https://image.pollinations.ai/prompt/Big%20Ben%2C%20sunny%20day"}
Thought:IhavetheimageURLofBigBenonasunnyday.NowIcanrespondtothehumanwiththeimageURL.

Action:
```
{
"action":"FinalAnswer",
"action_input":"HereistheimageofBigBenonasunnyday:https://image.pollinations.ai/prompt/Big%20Ben%2C%20sunny%20day"
}
```
Observation:

>Finishedchain.


..code::ipython3

#pleaserunthiscellforstoppinggradiointerface
demo.close()
