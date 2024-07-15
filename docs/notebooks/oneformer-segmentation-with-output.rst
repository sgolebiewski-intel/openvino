UniversalSegmentationwithOneFormerandOpenVINO
==================================================

Thistutorialdemonstrateshowtousethe
`OneFormer<https://arxiv.org/abs/2211.06220>`__modelfromHuggingFace
withOpenVINO.ItdescribeshowtodownloadweightsandcreatePyTorch
modelusingHuggingFacetransformerslibrary,thenconvertmodelto
OpenVINOIntermediateRepresentationformat(IR)usingOpenVINOModel
OptimizerAPIandrunmodelinference.Additionally,
`NNCF<https://github.com/openvinotoolkit/nncf/>`__quantizationis
appliedtoimproveOneFormersegmentationspeed.

|image0|

OneFormerisafollow-upworkof
`Mask2Former<https://arxiv.org/abs/2112.01527>`__.Thelatterstill
requirestrainingoninstance/semantic/panopticdatasetsseparatelyto
getstate-of-the-artresults.

OneFormerincorporatesatextmoduleintheMask2Formerframework,to
conditionthemodelontherespectivesubtask(instance,semanticor
panoptic).Thisgivesevenmoreaccurateresults,butcomeswithacost
ofincreasedlatency,however.

..|image0|image::https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_architecture.png

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Installrequiredlibraries<#install-required-libraries>`__
-`Preparetheenvironment<#prepare-the-environment>`__
-`LoadOneFormerfine-tunedonCOCOforuniversal
segmentation<#load-oneformer-fine-tuned-on-coco-for-universal-segmentation>`__
-`ConvertthemodeltoOpenVINOIR
format<#convert-the-model-to-openvino-ir-format>`__
-`Selectinferencedevice<#select-inference-device>`__
-`Chooseasegmentationtask<#choose-a-segmentation-task>`__
-`Inference<#inference>`__
-`Quantization<#quantization>`__

-`Preparingcalibrationdataset<#preparing-calibration-dataset>`__
-`Runquantization<#run-quantization>`__
-`Comparemodelsizeand
performance<#compare-model-size-and-performance>`__

-`InteractiveDemo<#interactive-demo>`__

Installrequiredlibraries
--------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

%pipinstall-q--extra-index-urlhttps://download.pytorch.org/whl/cpu"transformers>=4.26.0""openvino>=2023.1.0""nncf>=2.7.0""gradio>=4.19""torch>=2.1"scipyipywidgetsPillowtqdm

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.


Preparetheenvironment
-----------------------

`backtotop⬆️<#table-of-contents>`__

Importallrequiredpackagesandsetpathsformodelsandconstant
variables.

..code::ipython3

importwarnings
fromcollectionsimportdefaultdict
frompathlibimportPath

fromtransformersimportOneFormerProcessor,OneFormerForUniversalSegmentation
fromtransformers.models.oneformer.modeling_oneformerimport(
OneFormerForUniversalSegmentationOutput,
)
importtorch
importmatplotlib.pyplotasplt
importmatplotlib.patchesasmpatches
fromPILimportImage
fromPILimportImageOps

importopenvino

#Fetch`notebook_utils`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py","w").write(r.text)
fromnotebook_utilsimportdownload_file

..code::ipython3

IR_PATH=Path("oneformer.xml")
OUTPUT_NAMES=["class_queries_logits","masks_queries_logits"]

LoadOneFormerfine-tunedonCOCOforuniversalsegmentation
------------------------------------------------------------

`backtotop⬆️<#table-of-contents>`__

Hereweusethe``from_pretrained``methodof
``OneFormerForUniversalSegmentation``toloadthe`HuggingFaceOneFormer
model<https://huggingface.co/docs/transformers/model_doc/oneformer>`__
basedonSwin-Lbackboneandtrainedon
`COCO<https://cocodataset.org/>`__dataset.

Also,weuseHuggingFaceprocessortopreparethemodelinputsfrom
imagesandpost-processmodeloutputsforvisualization.

..code::ipython3

processor=OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
model=OneFormerForUniversalSegmentation.from_pretrained(
"shi-labs/oneformer_coco_swin_large",
)
id2label=model.config.id2label


..parsed-literal::

2023-10-0614:00:53.306851:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2023-10-0614:00:53.342792:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2023-10-0614:00:53.913248:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT
/home/nsavel/venvs/ov_notebooks_tmp/lib/python3.8/site-packages/transformers/models/oneformer/image_processing_oneformer.py:427:FutureWarning:The`reduce_labels`argumentisdeprecatedandwillberemovedinv4.27.Pleaseuse`do_reduce_labels`instead.
warnings.warn(


..code::ipython3

task_seq_length=processor.task_seq_length
shape=(800,800)
dummy_input={
"pixel_values":torch.randn(1,3,*shape),
"task_inputs":torch.randn(1,task_seq_length),
}

ConvertthemodeltoOpenVINOIRformat
---------------------------------------

`backtotop⬆️<#table-of-contents>`__

ConvertthePyTorchmodeltoIRformattotakeadvantageofOpenVINO
optimizationtoolsandfeatures.The``openvino.convert_model``python
functioninOpenVINOConvertercanconvertthemodel.Thefunction
returnsinstanceofOpenVINOModelclass,whichisreadytousein
Pythoninterface.However,itcanalsobeserializedtoOpenVINOIR
formatforfutureexecutionusing``save_model``function.PyTorchto
OpenVINOconversionisbasedonTorchScripttracing.HuggingFacemodels
havespecificconfigurationparameter``torchscript``,whichcanbeused
formakingthemodelmoresuitablefortracing.Forpreparingmodel.we
shouldprovidePyTorchmodelinstanceandexampleinputto
``openvino.convert_model``.

..code::ipython3

model.config.torchscript=True

ifnotIR_PATH.exists():
withwarnings.catch_warnings():
warnings.simplefilter("ignore")
model=openvino.convert_model(model,example_input=dummy_input)
openvino.save_model(model,IR_PATH,compress_to_fp16=False)


..parsed-literal::

WARNING:tensorflow:Pleasefixyourimports.Moduletensorflow.python.training.tracking.basehasbeenmovedtotensorflow.python.trackable.base.Theoldmodulewillbedeletedinversion2.11.


..parsed-literal::

[WARNING]Pleasefixyourimports.Module%shasbeenmovedto%s.Theoldmodulewillbedeletedinversion%s.


Selectinferencedevice
-----------------------

`backtotop⬆️<#table-of-contents>`__

SelectdevicefromdropdownlistforrunninginferenceusingOpenVINO

..code::ipython3

importipywidgetsaswidgets

core=openvino.Core()

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



WecanpreparetheimageusingtheHuggingFaceprocessor.OneFormer
leveragesaprocessorwhichinternallyconsistsofanimageprocessor
(fortheimagemodality)andatokenizer(forthetextmodality).
OneFormerisactuallyamultimodalmodel,sinceitincorporatesboth
imagesandtexttosolveimagesegmentation.

..code::ipython3

defprepare_inputs(image:Image.Image,task:str):
"""Convertimagetomodelinput"""
image=ImageOps.pad(image,shape)
inputs=processor(image,[task],return_tensors="pt")
converted={
"pixel_values":inputs["pixel_values"],
"task_inputs":inputs["task_inputs"],
}
returnconverted

..code::ipython3

defprocess_output(d):
"""ConvertOpenVINOmodeloutputtoHuggingFacerepresentationforvisualization"""
hf_kwargs={output_name:torch.tensor(d[output_name])foroutput_nameinOUTPUT_NAMES}

returnOneFormerForUniversalSegmentationOutput(**hf_kwargs)

..code::ipython3

#Readthemodelfromfiles.
model=core.read_model(model=IR_PATH)
#Compilethemodel.
compiled_model=core.compile_model(model=model,device_name=device.value)

Modelpredicts``class_queries_logits``ofshape
``(batch_size,num_queries)``and``masks_queries_logits``ofshape
``(batch_size,num_queries,height,width)``.

Herewedefinefunctionsforvisualizationofnetworkoutputstoshow
theinferenceresults.

..code::ipython3

classVisualizer:
@staticmethod
defextract_legend(handles):
fig=plt.figure()
fig.legend(handles=handles,ncol=len(handles)//20+1,loc="center")
fig.tight_layout()
returnfig

@staticmethod
defpredicted_semantic_map_to_figure(predicted_map):
segmentation=predicted_map[0]
#gettheusedcolormap
viridis=plt.get_cmap("viridis",max(1,torch.max(segmentation)))
#getalltheuniquenumbers
labels_ids=torch.unique(segmentation).tolist()
fig,ax=plt.subplots()
ax.imshow(segmentation)
ax.set_axis_off()
handles=[]
forlabel_idinlabels_ids:
label=id2label[label_id]
color=viridis(label_id)
handles.append(mpatches.Patch(color=color,label=label))
fig_legend=Visualizer.extract_legend(handles=handles)
fig.tight_layout()
returnfig,fig_legend

@staticmethod
defpredicted_instance_map_to_figure(predicted_map):
segmentation=predicted_map[0]["segmentation"]
segments_info=predicted_map[0]["segments_info"]
#gettheusedcolormap
viridis=plt.get_cmap("viridis",max(torch.max(segmentation),1))
fig,ax=plt.subplots()
ax.imshow(segmentation)
ax.set_axis_off()
instances_counter=defaultdict(int)
handles=[]
#foreachsegment,drawitslegend
forsegmentinsegments_info:
segment_id=segment["id"]
segment_label_id=segment["label_id"]
segment_label=id2label[segment_label_id]
label=f"{segment_label}-{instances_counter[segment_label_id]}"
instances_counter[segment_label_id]+=1
color=viridis(segment_id)
handles.append(mpatches.Patch(color=color,label=label))

fig_legend=Visualizer.extract_legend(handles)
fig.tight_layout()
returnfig,fig_legend

@staticmethod
defpredicted_panoptic_map_to_figure(predicted_map):
segmentation=predicted_map[0]["segmentation"]
segments_info=predicted_map[0]["segments_info"]
#gettheusedcolormap
viridis=plt.get_cmap("viridis",max(torch.max(segmentation),1))
fig,ax=plt.subplots()
ax.imshow(segmentation)
ax.set_axis_off()
instances_counter=defaultdict(int)
handles=[]
#foreachsegment,drawitslegend
forsegmentinsegments_info:
segment_id=segment["id"]
segment_label_id=segment["label_id"]
segment_label=id2label[segment_label_id]
label=f"{segment_label}-{instances_counter[segment_label_id]}"
instances_counter[segment_label_id]+=1
color=viridis(segment_id)
handles.append(mpatches.Patch(color=color,label=label))

fig_legend=Visualizer.extract_legend(handles)
fig.tight_layout()
returnfig,fig_legend

@staticmethod
deffigures_to_images(fig,fig_legend,name_suffix=""):
seg_filename,leg_filename=(
f"segmentation{name_suffix}.png",
f"legend{name_suffix}.png",
)
fig.savefig(seg_filename,bbox_inches="tight")
fig_legend.savefig(leg_filename,bbox_inches="tight")
segmentation=Image.open(seg_filename)
legend=Image.open(leg_filename)
returnsegmentation,legend

..code::ipython3

defsegment(model,img:Image.Image,task:str):
"""
Applysegmentationonanimage.

Args:
img:Inputimage.Itwillberesizedto800x800.
task:Stringdescribingthesegmentationtask.Supportedvaluesare:"semantic","instance"and"panoptic".
Returns:
Tuple[Figure,Figure]:Segmentationmapandlegendcharts.
"""
ifimgisNone:
raisegr.Error("Pleaseloadtheimageoruseonefromtheexampleslist")
inputs=prepare_inputs(img,task)
outputs=model(inputs)
hf_output=process_output(outputs)
predicted_map=getattr(processor,f"post_process_{task}_segmentation")(hf_output,target_sizes=[img.size[::-1]])
returngetattr(Visualizer,f"predicted_{task}_map_to_figure")(predicted_map)

..code::ipython3

image=download_file("http://images.cocodataset.org/val2017/000000439180.jpg","sample.jpg")
image=Image.open("sample.jpg")
image



..parsed-literal::

sample.jpg:0%||0.00/194k[00:00<?,?B/s]




..image::oneformer-segmentation-with-output_files/oneformer-segmentation-with-output_23_1.png



Chooseasegmentationtask
--------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

fromipywidgetsimportDropdown

task=Dropdown(options=["semantic","instance","panoptic"],value="semantic")
task




..parsed-literal::

Dropdown(options=('semantic','instance','panoptic'),value='semantic')



Inference
---------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importmatplotlib

matplotlib.use("Agg")#disableshowingfigures


defstack_images_horizontally(img1:Image,img2:Image):
res=Image.new("RGB",(img1.width+img2.width,max(img1.height,img2.height)),(255,255,255))
res.paste(img1,(0,0))
res.paste(img2,(img1.width,0))
returnres


segmentation_fig,legend_fig=segment(compiled_model,image,task.value)
segmentation_image,legend_image=Visualizer.figures_to_images(segmentation_fig,legend_fig)
plt.close("all")
prediction=stack_images_horizontally(segmentation_image,legend_image)
prediction




..image::oneformer-segmentation-with-output_files/oneformer-segmentation-with-output_27_0.png



Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingquantizationlayersintomodel
graphandthenusingasubsetofthetrainingdatasettoinitializethe
parametersoftheseadditionalquantizationlayers.Quantizedoperations
areexecutedin``INT8``insteadof``FP32``/``FP16``makingmodel
inferencefaster.

Theoptimizationprocesscontainsthefollowingsteps:1.Createa
calibrationdatasetforquantization.2.Run``nncf.quantize()``to
obtainquantizedmodel.3.Serializethe``INT8``modelusing
``openvino.save_model()``function.

Note:Quantizationistimeandmemoryconsumingoperation.Running
quantizationcodebelowmaytakesometime.

Pleaseselectbelowwhetheryouwouldliketorunquantizationto
improvemodelinferencespeed.

..code::ipython3

compiled_quantized_model=None

to_quantize=widgets.Checkbox(
value=False,
description="Quantization",
disabled=False,
)

to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



Let’sloadskipmagicextensiontoskipquantizationifto_quantizeis
notselected

..code::ipython3

#Fetch`skip_kernel_extension`module
r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

%load_extskip_kernel_extension

Preparingcalibrationdataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Weuseimagesfrom
`COCO128<https://www.kaggle.com/datasets/ultralytics/coco128>`__
datasetascalibrationsamples.

..code::ipython3

%%skipnot$to_quantize.value

importnncf
importtorch.utils.dataasdata

fromzipfileimportZipFile

DATA_URL="https://ultralytics.com/assets/coco128.zip"
OUT_DIR=Path('.')


classCOCOLoader(data.Dataset):
def__init__(self,images_path):
self.images=list(Path(images_path).iterdir())

def__getitem__(self,index):
image=Image.open(self.images[index])
ifimage.mode=='L':
rgb_image=Image.new("RGB",image.size)
rgb_image.paste(image)
image=rgb_image
returnimage

def__len__(self):
returnlen(self.images)


defdownload_coco128_dataset():
download_file(DATA_URL,directory=OUT_DIR,show_progress=True)
ifnot(OUT_DIR/"coco128/images/train2017").exists():
withZipFile('coco128.zip',"r")aszip_ref:
zip_ref.extractall(OUT_DIR)
coco_dataset=COCOLoader(OUT_DIR/'coco128/images/train2017')
returncoco_dataset


deftransform_fn(image):
#Wequantizemodelinpanopticmodebecauseitproducesoptimalresultsforbothsemanticandinstancesegmentationtasks
inputs=prepare_inputs(image,"panoptic")
returninputs


coco_dataset=download_coco128_dataset()
calibration_dataset=nncf.Dataset(coco_dataset,transform_fn)


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino



..parsed-literal::

coco128.zip:0%||0.00/6.66M[00:00<?,?B/s]


Runquantization
~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Belowwecall``nncf.quantize()``inordertoapplyquantizationto
OneFormermodel.

..code::ipython3

%%skipnot$to_quantize.value

INT8_IR_PATH=Path(str(IR_PATH).replace(".xml","_int8.xml"))

ifnotINT8_IR_PATH.exists():
quantized_model=nncf.quantize(
model,
calibration_dataset,
model_type=nncf.parameters.ModelType.TRANSFORMER,
subset_size=len(coco_dataset),
#smooth_quant_alphavalueof0.5wasselectedbasedonpredictionqualityvisualexamination
advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.5))
openvino.save_model(quantized_model,INT8_IR_PATH)
else:
quantized_model=core.read_model(INT8_IR_PATH)
compiled_quantized_model=core.compile_model(model=quantized_model,device_name=device.value)


..parsed-literal::

Statisticscollection:100%|██████████████████████████████████████████████████████████████████████████████████████████████|128/128[03:55<00:00,1.84s/it]
ApplyingSmoothQuant:100%|██████████████████████████████████████████████████████████████████████████████████████████████|216/216[00:18<00:00,11.89it/s]


..parsed-literal::

INFO:nncf:105ignorednodeswasfoundbynameintheNNCFGraph


..parsed-literal::

Statisticscollection:100%|██████████████████████████████████████████████████████████████████████████████████████████████|128/128[09:24<00:00,4.41s/it]
ApplyingFastBiascorrection:100%|██████████████████████████████████████████████████████████████████████████████████████|338/338[03:20<00:00,1.68it/s]


Let’sseequantizedmodelpredictionnexttooriginalmodelprediction.

..code::ipython3

%%skipnot$to_quantize.value

fromIPython.displayimportdisplay

image=Image.open("sample.jpg")
segmentation_fig,legend_fig=segment(compiled_quantized_model,image,task.value)
segmentation_image,legend_image=Visualizer.figures_to_images(segmentation_fig,legend_fig,name_suffix="_int8")
plt.close("all")
prediction_int8=stack_images_horizontally(segmentation_image,legend_image)
print("Originalmodelprediction:")
display(prediction)
print("Quantizedmodelprediction:")
display(prediction_int8)


..parsed-literal::

Originalmodelprediction:



..image::oneformer-segmentation-with-output_files/oneformer-segmentation-with-output_39_1.png


..parsed-literal::

Quantizedmodelprediction:



..image::oneformer-segmentation-with-output_files/oneformer-segmentation-with-output_39_3.png


Comparemodelsizeandperformance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Belowwecompareoriginalandquantizedmodelfootprintandinference
speed.

..code::ipython3

%%skipnot$to_quantize.value

importtime
importnumpyasnp
fromtqdm.autoimporttqdm

INFERENCE_TIME_DATASET_SIZE=30

defcalculate_compression_rate(model_path_ov,model_path_ov_int8):
model_size_fp32=model_path_ov.with_suffix(".bin").stat().st_size/1024
model_size_int8=model_path_ov_int8.with_suffix(".bin").stat().st_size/1024
print("Modelfootprintcomparison:")
print(f"*FP32IRmodelsize:{model_size_fp32:.2f}KB")
print(f"*INT8IRmodelsize:{model_size_int8:.2f}KB")
returnmodel_size_fp32,model_size_int8


defcalculate_call_inference_time(model):
inference_time=[]
foriintqdm(range(INFERENCE_TIME_DATASET_SIZE),desc="Measuringperformance"):
image=coco_dataset[i]
start=time.perf_counter()
segment(model,image,task.value)
end=time.perf_counter()
delta=end-start
inference_time.append(delta)
returnnp.median(inference_time)


time_fp32=calculate_call_inference_time(compiled_model)
time_int8=calculate_call_inference_time(compiled_quantized_model)

model_size_fp32,model_size_int8=calculate_compression_rate(IR_PATH,INT8_IR_PATH)

print(f"Modelfootprintreduction:{model_size_fp32/model_size_int8:.3f}")
print(f"Performancespeedup:{time_fp32/time_int8:.3f}")



..parsed-literal::

Measuringperformance:0%||0/30[00:00<?,?it/s]



..parsed-literal::

Measuringperformance:0%||0/30[00:00<?,?it/s]


..parsed-literal::

Modelfootprintcomparison:
*FP32IRmodelsize:899385.45KB
*INT8IRmodelsize:237545.83KB
Modelfootprintreduction:3.786
Performancespeedup:1.260


InteractiveDemo
----------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importtime
importgradioasgr

quantized_model_present=compiled_quantized_modelisnotNone


defcompile_model(device):
globalcompiled_model
globalcompiled_quantized_model
compiled_model=core.compile_model(model=model,device_name=device)
ifquantized_model_present:
compiled_quantized_model=core.compile_model(model=quantized_model,device_name=device)


defsegment_wrapper(image,task,run_quantized=False):
current_model=compiled_quantized_modelifrun_quantizedelsecompiled_model

start_time=time.perf_counter()
segmentation_fig,legend_fig=segment(current_model,image,task)
end_time=time.perf_counter()

name_suffix=""ifnotquantized_model_presentelse"_int8"ifrun_quantizedelse"_fp32"
segmentation_image,legend_image=Visualizer.figures_to_images(segmentation_fig,legend_fig,name_suffix=name_suffix)
plt.close("all")
result=stack_images_horizontally(segmentation_image,legend_image)
returnresult,f"{end_time-start_time:.2f}"


withgr.Blocks()asdemo:
withgr.Row():
withgr.Column():
inp_img=gr.Image(label="Image",type="pil")
inp_task=gr.Radio(["semantic","instance","panoptic"],label="Task",value="semantic")
inp_device=gr.Dropdown(label="Device",choices=core.available_devices+["AUTO"],value="AUTO")
withgr.Column():
out_result=gr.Image(label="Result(Original)"ifquantized_model_presentelse"Result")
inference_time=gr.Textbox(label="Time(seconds)")
out_result_quantized=gr.Image(label="Result(Quantized)",visible=quantized_model_present)
inference_time_quantized=gr.Textbox(label="Time(seconds)",visible=quantized_model_present)
run_button=gr.Button(value="Run")
run_button.click(
segment_wrapper,
[inp_img,inp_task,gr.Number(0,visible=False)],
[out_result,inference_time],
)
run_quantized_button=gr.Button(value="Runquantized",visible=quantized_model_present)
run_quantized_button.click(
segment_wrapper,
[inp_img,inp_task,gr.Number(1,visible=False)],
[out_result_quantized,inference_time_quantized],
)
gr.Examples(examples=[["sample.jpg","semantic"]],inputs=[inp_img,inp_task])

defon_device_change_begin():
return(
run_button.update(value="Changingdevice...",interactive=False),
run_quantized_button.update(value="Changingdevice...",interactive=False),
inp_device.update(interactive=False),
)

defon_device_change_end():
return(
run_button.update(value="Run",interactive=True),
run_quantized_button.update(value="Runquantized",interactive=True),
inp_device.update(interactive=True),
)

inp_device.change(on_device_change_begin,outputs=[run_button,run_quantized_button,inp_device]).then(compile_model,inp_device).then(
on_device_change_end,outputs=[run_button,run_quantized_button,inp_device]
)

try:
demo.launch(debug=False)
exceptException:
demo.launch(share=True,debug=False)
#ifyouarelaunchingremotely,specifyserver_nameandserver_port
#demo.launch(server_name='yourservername',server_port='serverportinint')
#Readmoreinthedocs:https://gradio.app/docs/


..parsed-literal::

RunningonlocalURL:http://127.0.0.1:7860

Tocreateapubliclink,set`share=True`in`launch()`.



..raw::html

<div><iframesrc="http://127.0.0.1:7860/"width="100%"height="500"allow="autoplay;camera;microphone;clipboard-read;clipboard-write;"frameborder="0"allowfullscreen></iframe></div>

