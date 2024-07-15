ObjectsegmentationswithEfficientSAMandOpenVINO
===================================================

`SegmentAnythingModel(SAM)<https://segment-anything.com/>`__has
emergedasapowerfultoolfornumerousvisionapplications.Akey
componentthatdrivestheimpressiveperformanceforzero-shottransfer
andhighversatilityisasuperlargeTransformermodeltrainedonthe
extensivehigh-qualitySA-1Bdataset.Whilebeneficial,thehuge
computationcostofSAMmodelhaslimiteditsapplicationstowider
real-worldapplications.Toaddressthislimitation,EfficientSAMs,
light-weightSAMmodelsthatexhibitdecentperformancewithlargely
reducedcomplexity,wereproposed.TheideabehindEfficientSAMisbased
onleveragingmaskedimagepretraining,SAMI,whichlearnsto
reconstructfeaturesfromSAMimageencoderforeffectivevisual
representationlearning.

..figure::https://yformer.github.io/efficient-sam/EfficientSAM_files/overview.png
:alt:overview.png

overview.png

Moredetailsaboutmodelcanbefoundin
`paper<https://arxiv.org/pdf/2312.00863.pdf>`__,`modelweb
page<https://yformer.github.io/efficient-sam/>`__and`original
repository<https://github.com/yformer/EfficientSAM>`__

InthistutorialweconsiderhowtoconvertandrunEfficientSAMusing
OpenVINO.Wealsodemonstratehowtoquantizemodelusing
`NNCF<https://github.com/openvinotoolkit/nncf.git>`__

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`Prerequisites<#prerequisites>`__
-`LoadPyTorchmodel<#load-pytorch-model>`__
-`RunPyTorchmodelinference<#run-pytorch-model-inference>`__

-`Prepareinputdata<#prepare-input-data>`__
-`Definehelpersforinputandoutput
processing<#define-helpers-for-input-and-output-processing>`__

-`ConvertmodeltoOpenVINOIR
format<#convert-model-to-openvino-ir-format>`__
-`RunOpenVINOmodelinference<#run-openvino-model-inference>`__

-`Selectinferencedevicefromdropdown
list<#select-inference-device-from-dropdown-list>`__
-`CompileOpenVINOmodel<#compile-openvino-model>`__
-`Inferenceandvisualize
result<#inference-and-visualize-result>`__

-`Quantization<#quantization>`__

-`Preparecalibrationdatasets<#prepare-calibration-datasets>`__
-`RunModelQuantization<#run-model-quantization>`__

-`Verifyquantizedmodel
inference<#verify-quantized-model-inference>`__

-`Savequantizemodelondisk<#save-quantize-model-on-disk>`__
-`Comparequantizedmodelsize<#compare-quantized-model-size>`__
-`CompareinferencetimeoftheFP16andINT8
models<#compare-inference-time-of-the-fp16-and-int8-models>`__

-`Interactivesegmentationdemo<#interactive-segmentation-demo>`__

Prerequisites
-------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importplatform

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4"
else:
%pipinstall-q"matplotlib>=3.4,<3.7"

%pipinstall-q"openvino>=2023.3.0""nncf>=2.7.0"opencv-python"gradio>=4.13"torchtorchvisiontqdm--extra-index-urlhttps://download.pytorch.org/whl/cpu


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

frompathlibimportPath

repo_dir=Path("EfficientSAM")

ifnotrepo_dir.exists():
!gitclonehttps://github.com/yformer/EfficientSAM.git
%cd$repo_dir


..parsed-literal::

Cloninginto'EfficientSAM'...
remote:Enumeratingobjects:424,done.[K
remote:Countingobjects:100%(85/85),done.[K
remote:Compressingobjects:100%(33/33),done.[K
remote:Total424(delta76),reused52(delta52),pack-reused339[K
Receivingobjects:100%(424/424),262.14MiB|28.43MiB/s,done.
Resolvingdeltas:100%(246/246),done.
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM


LoadPyTorchmodel
------------------

`backtotop⬆️<#table-of-contents>`__

Thereareseveralmodelsavailableintherepository:

-**efficient-sam-vitt**-EfficientSAMwithVisionTransformerTiny
(VIT-T)asimageencoder.Thesmallestandfastestmodelfrom
EfficientSAMfamily.
-**efficient-sam-vits**-EfficientSAMwithVisionTransformerSmall
(VIT-S)asimageencoder.Heavierthanefficient-sam-vitt,butmore
accuratemodel.

EfficientSAMprovidesaunifiedinterfaceforinteractionwithmodels.
Itmeansthatallprovidedstepsinthenotebookforconversionand
runningthemodelwillbethesameforallmodels.Below,youcanselect
oneofthemasexample.

..code::ipython3

fromefficient_sam.build_efficient_samimport(
build_efficient_sam_vitt,
build_efficient_sam_vits,
)
importzipfile

MODELS_LIST={
"efficient-sam-vitt":build_efficient_sam_vitt,
"efficient-sam-vits":build_efficient_sam_vits,
}

#SinceEfficientSAM-Scheckpointfileis>100MB,westorethezipfile.
withzipfile.ZipFile("weights/efficient_sam_vits.pt.zip","r")aszip_ref:
zip_ref.extractall("weights")

Selectonefromsupportedmodels:

..code::ipython3

importipywidgetsaswidgets

model_ids=list(MODELS_LIST)

model_id=widgets.Dropdown(
options=model_ids,
value=model_ids[0],
description="Model:",
disabled=False,
)

model_id




..parsed-literal::

Dropdown(description='Model:',options=('efficient-sam-vitt','efficient-sam-vits'),value='efficient-sam-vitt…



buildPyTorchmodel

..code::ipython3

pt_model=MODELS_LIST[model_id.value]()

pt_model.eval();

RunPyTorchmodelinference
---------------------------

`backtotop⬆️<#table-of-contents>`__Now,whenweselectedand
loadedPyTorchmodel,wecancheckitsresult

Prepareinputdata
~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Firstofall,weshouldprepareinputdataformodel.Modelhas3
inputs:\*imagetensor-tensorwithnormalizedinputimage.\*input
points-tensorwithuserprovidedpoints.Itmaybejustsomespecific
pointsontheimage(e.g. providedbyuserclicksonthescreen)or
boundingboxcoordinatesinformatleft-topanglepointandright-bottom
anglepint.\*inputlabels-tensorwithdefinitionofpointtypefor
eachprovidedpoint,1-forregularpoint,2-left-toppointof
boundingbox,3-right-bottompointofboundingbox.

..code::ipython3

fromPILimportImage

image_path="figs/examples/dogs.jpg"

image=Image.open(image_path)
image




..image::efficient-sam-with-output_files/efficient-sam-with-output_11_0.png



Definehelpersforinputandoutputprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thecodebelowdefineshelpersforpreparingmodelinputandpostprocess
inferenceresults.Theinputformatisacceptedbythemodeldescribed
above.Themodelpredictsmasklogitsforeachpixelontheimageand
intersectionoverunionscoreforeacharea,howcloseitistoprovided
points.Wealsoprovidedsomehelperfunctionforresultsvisualization.

..code::ipython3

importtorch
importmatplotlib.pyplotasplt
importnumpyasnp


defprepare_input(input_image,points,labels,torch_tensor=True):
img_tensor=np.ascontiguousarray(input_image)[None,...].astype(np.float32)/255
img_tensor=np.transpose(img_tensor,(0,3,1,2))
pts_sampled=np.reshape(np.ascontiguousarray(points),[1,1,-1,2])
pts_labels=np.reshape(np.ascontiguousarray(labels),[1,1,-1])
iftorch_tensor:
img_tensor=torch.from_numpy(img_tensor)
pts_sampled=torch.from_numpy(pts_sampled)
pts_labels=torch.from_numpy(pts_labels)
returnimg_tensor,pts_sampled,pts_labels


defpostprocess_results(predicted_iou,predicted_logits):
sorted_ids=np.argsort(-predicted_iou,axis=-1)
predicted_iou=np.take_along_axis(predicted_iou,sorted_ids,axis=2)
predicted_logits=np.take_along_axis(predicted_logits,sorted_ids[...,None,None],axis=2)

returnpredicted_logits[0,0,0,:,:]>=0


defshow_points(coords,labels,ax,marker_size=375):
pos_points=coords[labels==1]
neg_points=coords[labels==0]
ax.scatter(
pos_points[:,0],
pos_points[:,1],
color="green",
marker="*",
s=marker_size,
edgecolor="white",
linewidth=1.25,
)
ax.scatter(
neg_points[:,0],
neg_points[:,1],
color="red",
marker="*",
s=marker_size,
edgecolor="white",
linewidth=1.25,
)


defshow_box(box,ax):
x0,y0=box[0],box[1]
w,h=box[2]-box[0],box[3]-box[1]
ax.add_patch(plt.Rectangle((x0,y0),w,h,edgecolor="yellow",facecolor=(0,0,0,0),lw=5))


defshow_anns(mask,ax):
ax.set_autoscale_on(False)
img=np.ones((mask.shape[0],mask.shape[1],4))
img[:,:,3]=0
#foranninmask:
#m=ann
color_mask=np.concatenate([np.random.random(3),[0.5]])
img[mask]=color_mask
ax.imshow(img)

Thecompletemodelinferenceexampledemonstratedbelow

..code::ipython3

input_points=[[580,350],[650,350]]
input_labels=[1,1]

example_input=prepare_input(image,input_points,input_labels)

predicted_logits,predicted_iou=pt_model(*example_input)

predicted_mask=postprocess_results(predicted_iou.detach().numpy(),predicted_logits.detach().numpy())

..code::ipython3

image=Image.open(image_path)

plt.figure(figsize=(20,20))
plt.axis("off")
plt.imshow(image)
show_points(np.array(input_points),np.array(input_labels),plt.gca())
plt.figure(figsize=(20,20))
plt.axis("off")
plt.imshow(image)
show_anns(predicted_mask,plt.gca())
plt.title(f"PyTorch{model_id.value}",fontsize=18)
plt.show()



..image::efficient-sam-with-output_files/efficient-sam-with-output_16_0.png



..image::efficient-sam-with-output_files/efficient-sam-with-output_16_1.png


ConvertmodeltoOpenVINOIRformat
-----------------------------------

`backtotop⬆️<#table-of-contents>`__

OpenVINOsupportsPyTorchmodelsviaconversioninIntermediate
Representation(IR)formatusingOpenVINO`ModelConversion
API<https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
``openvino.convert_model``functionacceptsinstanceofPyTorchmodel
andexampleinput(thathelpsincorrectmodeloperationtracingand
shapeinference)andreturns``openvino.Model``objectthatrepresents
modelinOpenVINOframework.This``openvino.Model``isreadyfor
loadingonthedeviceusing``ov.Core.compile_model``orcanbesavedon
diskusing``openvino.save_model``.

..code::ipython3

importopenvinoasov

core=ov.Core()

ov_model_path=Path(f"{model_id.value}.xml")

ifnotov_model_path.exists():
ov_model=ov.convert_model(pt_model,example_input=example_input)
ov.save_model(ov_model,ov_model_path)
else:
ov_model=core.read_model(ov_model_path)


..parsed-literal::

/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM/efficient_sam/efficient_sam.py:220:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
if(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM/efficient_sam/efficient_sam_encoder.py:241:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
assert(
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM/efficient_sam/efficient_sam_encoder.py:163:TracerWarning:ConvertingatensortoaPythonfloatmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
size=int(math.sqrt(xy_num))
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM/efficient_sam/efficient_sam_encoder.py:164:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
assertsize*size==xy_num
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM/efficient_sam/efficient_sam_encoder.py:166:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifsize!=horsize!=w:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM/efficient_sam/efficient_sam_encoder.py:251:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
assertx.shape[2]==num_patches
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM/efficient_sam/efficient_sam.py:85:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifnum_pts>self.decoder_max_num_input_points:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM/efficient_sam/efficient_sam.py:92:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
elifnum_pts<self.decoder_max_num_input_points:
/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-727/.workspace/scm/ov-notebook/notebooks/efficient-sam/EfficientSAM/efficient_sam/efficient_sam.py:126:TracerWarning:ConvertingatensortoaPythonbooleanmightcausethetracetobeincorrect.Wecan'trecordthedataflowofPythonvalues,sothisvaluewillbetreatedasaconstantinthefuture.Thismeansthatthetracemightnotgeneralizetootherinputs!
ifoutput_w>0andoutput_h>0:


..parsed-literal::

['batched_images','batched_points','batched_point_labels']


RunOpenVINOmodelinference
----------------------------

`backtotop⬆️<#table-of-contents>`__

Selectinferencedevicefromdropdownlist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

device=widgets.Dropdown(
options=core.available_devices+["AUTO"],
value="AUTO",
description="Device:",
disabled=False,
)

device




..parsed-literal::

Dropdown(description='Device:',index=1,options=('CPU','AUTO'),value='AUTO')



CompileOpenVINOmodel
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

compiled_model=core.compile_model(ov_model,device.value)

Inferenceandvisualizeresult
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Now,wecantakealookonOpenVINOmodelprediction

..code::ipython3

example_input=prepare_input(image,input_points,input_labels,torch_tensor=False)
result=compiled_model(example_input)

predicted_logits,predicted_iou=result[0],result[1]

predicted_mask=postprocess_results(predicted_iou,predicted_logits)

plt.figure(figsize=(20,20))
plt.axis("off")
plt.imshow(image)
show_points(np.array(input_points),np.array(input_labels),plt.gca())
plt.figure(figsize=(20,20))
plt.axis("off")
plt.imshow(image)
show_anns(predicted_mask,plt.gca())
plt.title(f"OpenVINO{model_id.value}",fontsize=18)
plt.show()



..image::efficient-sam-with-output_files/efficient-sam-with-output_24_0.png



..image::efficient-sam-with-output_files/efficient-sam-with-output_24_1.png


Quantization
------------

`backtotop⬆️<#table-of-contents>`__

`NNCF<https://github.com/openvinotoolkit/nncf/>`__enables
post-trainingquantizationbyaddingthequantizationlayersintothe
modelgraphandthenusingasubsetofthetrainingdatasetto
initializetheparametersoftheseadditionalquantizationlayers.The
frameworkisdesignedsothatmodificationstoyouroriginaltraining
codeareminor.

Theoptimizationprocesscontainsthefollowingsteps:

1.Createacalibrationdatasetforquantization.
2.Run``nncf.quantize``toobtainquantizedencoderanddecodermodels.
3.Serializethe``INT8``modelusing``openvino.save_model``function.

..

**Note**:Quantizationistimeandmemoryconsumingoperation.
Runningquantizationcodebelowmaytakesometime.

PleaseselectbelowwhetheryouwouldliketorunEfficientSAM
quantization.

..code::ipython3

to_quantize=widgets.Checkbox(
value=True,
description="Quantization",
disabled=False,
)

to_quantize




..parsed-literal::

Checkbox(value=True,description='Quantization')



..code::ipython3

#Fetch`skip_kernel_extension`module
importrequests

r=requests.get(
url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py","w").write(r.text)

%load_extskip_kernel_extension

Preparecalibrationdatasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Thefirststepistopreparecalibrationdatasetsforquantization.We
willusecoco128datasetforquantization.Usually,thisdatasetused
forsolvingobjectdetectiontaskanditsannotationprovidesbox
coordinatesforimages.Inourcase,boxcoordinateswillserveasinput
pointsforobjectsegmentation,thecodebelowdownloadsdatasetand
createsDataLoaderforpreparinginputsforEfficientSAMmodel.

..code::ipython3

%%skipnot$to_quantize.value

fromzipfileimportZipFile

r=requests.get(
url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py',
)

open('notebook_utils.py','w').write(r.text)

fromnotebook_utilsimportdownload_file

DATA_URL="https://ultralytics.com/assets/coco128.zip"
OUT_DIR=Path('.')

download_file(DATA_URL,directory=OUT_DIR,show_progress=True)

ifnot(OUT_DIR/"coco128/images/train2017").exists():
withZipFile('coco128.zip',"r")aszip_ref:
zip_ref.extractall(OUT_DIR)



..parsed-literal::

coco128.zip:0%||0.00/6.66M[00:00<?,?B/s]


..code::ipython3

%%skipnot$to_quantize.value

importtorch.utils.dataasdata

classCOCOLoader(data.Dataset):
def__init__(self,images_path):
self.images=list(Path(images_path).iterdir())
self.labels_dir=images_path.parents[1]/'labels'/images_path.name

defget_points(self,image_path,image_width,image_height):
file_name=image_path.name.replace('.jpg','.txt')
label_file=self.labels_dir/file_name
ifnotlabel_file.exists():
x1,x2=np.random.randint(low=0,high=image_width,size=(2,))
y1,y2=np.random.randint(low=0,high=image_height,size=(2,))
else:
withlabel_file.open("r")asf:
box_line=f.readline()
_,x1,y1,x2,y2=box_line.split()
x1=int(float(x1)*image_width)
y1=int(float(y1)*image_height)
x2=int(float(x2)*image_width)
y2=int(float(y2)*image_height)
return[[x1,y1],[x2,y2]]

def__getitem__(self,index):
image_path=self.images[index]
image=Image.open(image_path)
image=image.convert('RGB')
w,h=image.size
points=self.get_points(image_path,w,h)
labels=[1,1]ifindex%2==0else[2,3]
batched_images,batched_points,batched_point_labels=prepare_input(image,points,labels,torch_tensor=False)
return{'batched_images':np.ascontiguousarray(batched_images)[0],'batched_points':np.ascontiguousarray(batched_points)[0],'batched_point_labels':np.ascontiguousarray(batched_point_labels)[0]}

def__len__(self):
returnlen(self.images)

..code::ipython3

%%skipnot$to_quantize.value

coco_dataset=COCOLoader(OUT_DIR/'coco128/images/train2017')
calibration_loader=torch.utils.data.DataLoader(coco_dataset)

RunModelQuantization
~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

The``nncf.quantize``functionprovidesaninterfaceformodel
quantization.ItrequiresaninstanceoftheOpenVINOModeland
quantizationdataset.Optionally,someadditionalparametersforthe
configurationquantizationprocess(numberofsamplesforquantization,
preset,ignoredscope,etc.)canbeprovided.EfficientSAMcontains
non-ReLUactivationfunctions,whichrequireasymmetricquantizationof
activations.Toachieveabetterresult,wewillusea``mixed``
quantization``preset``.ModelencoderpartisbasedonVision
Transformerarchitectureforactivatingspecialoptimizationsforthis
architecturetype,weshouldspecify``transformer``in``model_type``.

..code::ipython3

%%skipnot$to_quantize.value

importnncf

calibration_dataset=nncf.Dataset(calibration_loader)

model=core.read_model(ov_model_path)
quantized_model=nncf.quantize(model,
calibration_dataset,
model_type=nncf.parameters.ModelType.TRANSFORMER,
subset_size=128)
print("modelquantizationfinished")


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..parsed-literal::

2024-07-1300:20:24.222824:Itensorflow/core/util/port.cc:110]oneDNNcustomoperationsareon.Youmayseeslightlydifferentnumericalresultsduetofloating-pointround-offerrorsfromdifferentcomputationorders.Toturnthemoff,settheenvironmentvariable`TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-1300:20:24.255951:Itensorflow/core/platform/cpu_feature_guard.cc:182]ThisTensorFlowbinaryisoptimizedtouseavailableCPUinstructionsinperformance-criticaloperations.
Toenablethefollowinginstructions:AVX2AVX512FAVX512_VNNIFMA,inotheroperations,rebuildTensorFlowwiththeappropriatecompilerflags.
2024-07-1300:20:24.882804:Wtensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]TF-TRTWarning:CouldnotfindTensorRT



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>




..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

INFO:nncf:57ignorednodeswerefoundbynameintheNNCFGraph
INFO:nncf:88ignorednodeswerefoundbynameintheNNCFGraph



..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>




..parsed-literal::

Output()



..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace"></pre>




..raw::html

<prestyle="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVuSansMono',consolas,'CourierNew',monospace">
</pre>



..parsed-literal::

modelquantizationfinished


Verifyquantizedmodelinference
--------------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

compiled_model=core.compile_model(quantized_model,device.value)

result=compiled_model(example_input)

predicted_logits,predicted_iou=result[0],result[1]

predicted_mask=postprocess_results(predicted_iou,predicted_logits)

plt.figure(figsize=(20,20))
plt.axis("off")
plt.imshow(image)
show_points(np.array(input_points),np.array(input_labels),plt.gca())
plt.figure(figsize=(20,20))
plt.axis("off")
plt.imshow(image)
show_anns(predicted_mask,plt.gca())
plt.title(f"OpenVINOINT8{model_id.value}",fontsize=18)
plt.show()



..image::efficient-sam-with-output_files/efficient-sam-with-output_35_0.png



..image::efficient-sam-with-output_files/efficient-sam-with-output_35_1.png


Savequantizemodelondisk
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

quantized_model_path=Path(f"{model_id.value}_int8.xml")
ov.save_model(quantized_model,quantized_model_path)

Comparequantizedmodelsize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

%%skipnot$to_quantize.value

fp16_weights=ov_model_path.with_suffix('.bin')
quantized_weights=quantized_model_path.with_suffix('.bin')

print(f"SizeofFP16modelis{fp16_weights.stat().st_size/1024/1024:.2f}MB")
print(f"SizeofINT8quantizedmodelis{quantized_weights.stat().st_size/1024/1024:.2f}MB")
print(f"CompressionrateforINT8model:{fp16_weights.stat().st_size/quantized_weights.stat().st_size:.3f}")


..parsed-literal::

SizeofFP16modelis21.50MB
SizeofINT8quantizedmodelis11.08MB
CompressionrateforINT8model:1.941


CompareinferencetimeoftheFP16andINT8models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Tomeasuretheinferenceperformanceofthe``FP16``and``INT8``
models,weuse``bencmark_app``.

**NOTE**:Forthemostaccurateperformanceestimation,itis
recommendedtorun``benchmark_app``inaterminal/commandprompt
afterclosingotherapplications.

..code::ipython3

!benchmark_app-m$ov_model_path-d$device.value-data_shape"batched_images[1,3,512,512],batched_points[1,1,2,2],batched_point_labels[1,1,2]"-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook29.66ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]batched_images(node:batched_images):f32/[...]/[?,?,?,?]
[INFO]batched_points(node:batched_points):i64/[...]/[?,?,?,?]
[INFO]batched_point_labels(node:batched_point_labels):i64/[...]/[?,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:aten::reshape/Reshape_3):f32/[...]/[?,?,3,?,?]
[INFO]***NO_NAME***(node:aten::reshape/Reshape_2):f32/[...]/[?,?,3]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]batched_images(node:batched_images):f32/[...]/[?,?,?,?]
[INFO]batched_points(node:batched_points):i64/[...]/[?,?,?,?]
[INFO]batched_point_labels(node:batched_point_labels):i64/[...]/[?,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:aten::reshape/Reshape_3):f32/[...]/[?,?,3,?,?]
[INFO]***NO_NAME***(node:aten::reshape/Reshape_2):f32/[...]/[?,?,3]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook1312.01ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:32
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:24
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:6
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'batched_images'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'batched_points'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'batched_point_labels'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'batched_images'withrandomvalues
[INFO]Fillinput'batched_points'withrandomvalues
[INFO]Fillinput'batched_point_labels'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,6inferencerequests,limits:15000msduration)
[INFO]Benchmarkinginfullmode(inputsfillingareincludedinmeasurementloop).
[INFO]Firstinferencetook666.28ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:49iterations
[INFO]Duration:15804.58ms
[INFO]Latency:
[INFO]Median:1903.83ms
[INFO]Average:1881.69ms
[INFO]Min:626.71ms
[INFO]Max:1969.71ms
[INFO]Throughput:3.10FPS


..code::ipython3

ifto_quantize.value:
!benchmark_app-m$quantized_model_path-d$device.value-data_shape"batched_images[1,3,512,512],batched_points[1,1,2,2],batched_point_labels[1,1,2]"-t15


..parsed-literal::

[Step1/11]Parsingandvalidatinginputarguments
[INFO]Parsinginputparameters
[Step2/11]LoadingOpenVINORuntime
[INFO]OpenVINO:
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]Deviceinfo:
[INFO]AUTO
[INFO]Build.................................2024.4.0-16028-fe423b97163
[INFO]
[INFO]
[Step3/11]Settingdeviceconfiguration
[WARNING]Performancehintwasnotexplicitlyspecifiedincommandline.Device(AUTO)performancehintwillbesettoPerformanceMode.THROUGHPUT.
[Step4/11]Readingmodelfiles
[INFO]Loadingmodelfiles
[INFO]Readmodeltook43.30ms
[INFO]OriginalmodelI/Oparameters:
[INFO]Modelinputs:
[INFO]batched_images(node:batched_images):f32/[...]/[?,?,?,?]
[INFO]batched_points(node:batched_points):i64/[...]/[?,?,?,?]
[INFO]batched_point_labels(node:batched_point_labels):i64/[...]/[?,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:aten::reshape/Reshape_3):f32/[...]/[?,?,3,?,?]
[INFO]***NO_NAME***(node:aten::reshape/Reshape_2):f32/[...]/[?,?,3]
[Step5/11]Resizingmodeltomatchimagesizesandgivenbatch
[INFO]Modelbatchsize:1
[Step6/11]Configuringinputofthemodel
[INFO]Modelinputs:
[INFO]batched_images(node:batched_images):f32/[...]/[?,?,?,?]
[INFO]batched_points(node:batched_points):i64/[...]/[?,?,?,?]
[INFO]batched_point_labels(node:batched_point_labels):i64/[...]/[?,?,?]
[INFO]Modeloutputs:
[INFO]***NO_NAME***(node:aten::reshape/Reshape_3):f32/[...]/[?,?,3,?,?]
[INFO]***NO_NAME***(node:aten::reshape/Reshape_2):f32/[...]/[?,?,3]
[Step7/11]Loadingthemodeltothedevice
[INFO]Compilemodeltook1679.82ms
[Step8/11]Queryingoptimalruntimeparameters
[INFO]Model:
[INFO]NETWORK_NAME:Model0
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]PERFORMANCE_HINT:PerformanceMode.THROUGHPUT
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]MULTI_DEVICE_PRIORITIES:CPU
[INFO]CPU:
[INFO]AFFINITY:Affinity.CORE
[INFO]CPU_DENORMALS_OPTIMIZATION:False
[INFO]CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE:1.0
[INFO]DYNAMIC_QUANTIZATION_GROUP_SIZE:32
[INFO]ENABLE_CPU_PINNING:True
[INFO]ENABLE_HYPER_THREADING:True
[INFO]EXECUTION_DEVICES:['CPU']
[INFO]EXECUTION_MODE_HINT:ExecutionMode.PERFORMANCE
[INFO]INFERENCE_NUM_THREADS:24
[INFO]INFERENCE_PRECISION_HINT:<Type:'float32'>
[INFO]KV_CACHE_PRECISION:<Type:'float16'>
[INFO]LOG_LEVEL:Level.NO
[INFO]MODEL_DISTRIBUTION_POLICY:set()
[INFO]NETWORK_NAME:Model0
[INFO]NUM_STREAMS:6
[INFO]OPTIMAL_NUMBER_OF_INFER_REQUESTS:6
[INFO]PERFORMANCE_HINT:THROUGHPUT
[INFO]PERFORMANCE_HINT_NUM_REQUESTS:0
[INFO]PERF_COUNT:NO
[INFO]SCHEDULING_CORE_TYPE:SchedulingCoreType.ANY_CORE
[INFO]MODEL_PRIORITY:Priority.MEDIUM
[INFO]LOADED_FROM_CACHE:False
[INFO]PERF_COUNT:False
[Step9/11]Creatinginferrequestsandpreparinginputtensors
[WARNING]Noinputfilesweregivenforinput'batched_images'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'batched_points'!.Thisinputwillbefilledwithrandomvalues!
[WARNING]Noinputfilesweregivenforinput'batched_point_labels'!.Thisinputwillbefilledwithrandomvalues!
[INFO]Fillinput'batched_images'withrandomvalues
[INFO]Fillinput'batched_points'withrandomvalues
[INFO]Fillinput'batched_point_labels'withrandomvalues
[Step10/11]Measuringperformance(Startinferenceasynchronously,6inferencerequests,limits:15000msduration)
[INFO]Benchmarkinginfullmode(inputsfillingareincludedinmeasurementloop).
[INFO]Firstinferencetook604.97ms
[Step11/11]Dumpingstatisticsreport
[INFO]ExecutionDevices:['CPU']
[INFO]Count:55iterations
[INFO]Duration:16291.10ms
[INFO]Latency:
[INFO]Median:1758.14ms
[INFO]Average:1740.52ms
[INFO]Min:625.06ms
[INFO]Max:1830.61ms
[INFO]Throughput:3.38FPS


Interactivesegmentationdemo
-----------------------------

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importcopy
importgradioasgr
importnumpyasnp
fromPILimportImageDraw,Image
importcv2
importmatplotlib.pyplotasplt

example_images=[
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/b8083dd5-1ce7-43bf-8b09-a2ebc280c86e",
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/9a90595d-70e7-469b-bdaf-469ef4f56fa2",
"https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/b626c123-9fa2-4aa6-9929-30565991bf0c",
]

examples_dir=Path("examples")
examples_dir.mkdir(exist_ok=True)

forimg_id,image_urlinenumerate(example_images):
r=requests.get(image_url)
img_path=examples_dir/f"example_{img_id}.jpg"
withimg_path.open("wb")asf:
f.write(r.content)


defsigmoid(x):
return1/(1+np.exp(-x))


defclear():
returnNone,None,[],[]


defformat_results(masks,scores,logits,filter=0):
annotations=[]
n=len(scores)
foriinrange(n):
annotation={}

mask=masks[i]
tmp=np.where(mask!=0)
ifnp.sum(mask)<filter:
continue
annotation["id"]=i
annotation["segmentation"]=mask
annotation["bbox"]=[
np.min(tmp[0]),
np.min(tmp[1]),
np.max(tmp[1]),
np.max(tmp[0]),
]
annotation["score"]=scores[i]
annotation["area"]=annotation["segmentation"].sum()
annotations.append(annotation)
returnannotations


defpoint_prompt(masks,points,point_label,target_height,target_width):#numpy
h=masks[0]["segmentation"].shape[0]
w=masks[0]["segmentation"].shape[1]
ifh!=target_heightorw!=target_width:
points=[[int(point[0]*w/target_width),int(point[1]*h/target_height)]forpointinpoints]
onemask=np.zeros((h,w))
fori,annotationinenumerate(masks):
ifisinstance(annotation,dict):
mask=annotation["segmentation"]
else:
mask=annotation
fori,pointinenumerate(points):
ifpoint[1]<mask.shape[0]andpoint[0]<mask.shape[1]:
ifmask[point[1],point[0]]==1andpoint_label[i]==1:
onemask+=mask
ifmask[point[1],point[0]]==1andpoint_label[i]==0:
onemask-=mask
onemask=onemask>=1
returnonemask,0


defshow_mask(
annotation,
ax,
random_color=False,
bbox=None,
retinamask=True,
target_height=960,
target_width=960,
):
mask_sum=annotation.shape[0]
height=annotation.shape[1]
weight=annotation.shape[2]
#annotationissortedbyarea
areas=np.sum(annotation,axis=(1,2))
sorted_indices=np.argsort(areas)[::1]
annotation=annotation[sorted_indices]

index=(annotation!=0).argmax(axis=0)
ifrandom_color:
color=np.random.random((mask_sum,1,1,3))
else:
color=np.ones((mask_sum,1,1,3))*np.array([30/255,144/255,255/255])
transparency=np.ones((mask_sum,1,1,1))*0.6
visual=np.concatenate([color,transparency],axis=-1)
mask_image=np.expand_dims(annotation,-1)*visual

mask=np.zeros((height,weight,4))

h_indices,w_indices=np.meshgrid(np.arange(height),np.arange(weight),indexing="ij")
indices=(index[h_indices,w_indices],h_indices,w_indices,slice(None))

mask[h_indices,w_indices,:]=mask_image[indices]
ifbboxisnotNone:
x1,y1,x2,y2=bbox
ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="b",linewidth=1))

ifnotretinamask:
mask=cv2.resize(mask,(target_width,target_height),interpolation=cv2.INTER_NEAREST)

returnmask


defprocess(
annotations,
image,
scale,
better_quality=False,
mask_random_color=True,
bbox=None,
points=None,
use_retina=True,
withContours=True,
):
ifisinstance(annotations[0],dict):
annotations=[annotation["segmentation"]forannotationinannotations]

original_h=image.height
original_w=image.width
ifbetter_quality:
ifisinstance(annotations[0],torch.Tensor):
annotations=np.array(annotations)
fori,maskinenumerate(annotations):
mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
annotations[i]=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,np.ones((8,8),np.uint8))
annotations=np.array(annotations)
inner_mask=show_mask(
annotations,
plt.gca(),
random_color=mask_random_color,
bbox=bbox,
retinamask=use_retina,
target_height=original_h,
target_width=original_w,
)

ifisinstance(annotations,torch.Tensor):
annotations=annotations.cpu().numpy()

ifwithContours:
contour_all=[]
temp=np.zeros((original_h,original_w,1))
fori,maskinenumerate(annotations):
ifisinstance(mask,dict):
mask=mask["segmentation"]
annotation=mask.astype(np.uint8)
ifnotuse_retina:
annotation=cv2.resize(
annotation,
(original_w,original_h),
interpolation=cv2.INTER_NEAREST,
)
contours,_=cv2.findContours(annotation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
forcontourincontours:
contour_all.append(contour)
cv2.drawContours(temp,contour_all,-1,(255,255,255),2//scale)
color=np.array([0/255,0/255,255/255,0.9])
contour_mask=temp/255*color.reshape(1,1,-1)

image=image.convert("RGBA")
overlay_inner=Image.fromarray((inner_mask*255).astype(np.uint8),"RGBA")
image.paste(overlay_inner,(0,0),overlay_inner)

ifwithContours:
overlay_contour=Image.fromarray((contour_mask*255).astype(np.uint8),"RGBA")
image.paste(overlay_contour,(0,0),overlay_contour)

returnimage


#Description
title="<center><strong><fontsize='8'>EfficientSegmentAnythingwithOpenVINOandEfficientSAM<font></strong></center>"


description_p="""#InteractiveInstanceSegmentation
-Point-promptinstruction
<ol>
<li>Clickontheleftimage(pointinput),visualizingthepointontherightimage</li>
<li>ClickthebuttonofSegmentwithPointPrompt</li>
</ol>
-Box-promptinstruction
<ol>
<li>Clickontheleftimage(onepointinput),visualizingthepointontherightimage</li>
<li>Clickontheleftimage(anotherpointinput),visualizingthepointandtheboxontherightimage</li>
<li>ClickthebuttonofSegmentwithBoxPrompt</li>
</ol>
"""

#examples
examples=[[img]forimginexamples_dir.glob("*.jpg")]

default_example=examples[0]

css="h1{text-align:center}.about{text-align:justify;padding-left:10%;padding-right:10%;}"


defsegment_with_boxs(
image,
seg_image,
global_points,
global_point_label,
input_size=1024,
better_quality=False,
withContours=True,
use_retina=True,
mask_random_color=True,
):
ifglobal_pointsisNoneorlen(global_points)<2orglobal_points[0]isNone:
returnimage,global_points,global_point_label

input_size=int(input_size)
w,h=image.size
scale=input_size/max(w,h)
new_w=int(w*scale)
new_h=int(h*scale)
image=image.resize((new_w,new_h))

scaled_points=np.array([[int(x*scale)forxinpoint]forpointinglobal_points])
scaled_points=scaled_points[:2]
scaled_point_label=np.array(global_point_label)[:2]

ifscaled_points.size==0andscaled_point_label.size==0:
returnimage,global_points,global_point_label

nd_image=np.array(image)
img_tensor=nd_image.astype(np.float32)/255
img_tensor=np.transpose(img_tensor,(2,0,1))

pts_sampled=np.reshape(scaled_points,[1,1,-1,2])
pts_sampled=pts_sampled[:,:,:2,:]
pts_labels=np.reshape(np.array([2,3]),[1,1,2])

results=compiled_model([img_tensor[None,...],pts_sampled,pts_labels])
predicted_logits=results[0]
predicted_iou=results[1]
all_masks=sigmoid(predicted_logits[0,0,:,:,:])>=0.5
predicted_iou=predicted_iou[0,0,...]

max_predicted_iou=-1
selected_mask_using_predicted_iou=None
selected_predicted_iou=None

forminrange(all_masks.shape[0]):
curr_predicted_iou=predicted_iou[m]
ifcurr_predicted_iou>max_predicted_iouorselected_mask_using_predicted_iouisNone:
max_predicted_iou=curr_predicted_iou
selected_mask_using_predicted_iou=all_masks[m:m+1]
selected_predicted_iou=predicted_iou[m:m+1]

results=format_results(selected_mask_using_predicted_iou,selected_predicted_iou,predicted_logits,0)

annotations=results[0]["segmentation"]
annotations=np.array([annotations])
fig=process(
annotations=annotations,
image=image,
scale=(1024//input_size),
better_quality=better_quality,
mask_random_color=mask_random_color,
use_retina=use_retina,
bbox=scaled_points.reshape([4]),
withContours=withContours,
)

global_points=[]
global_point_label=[]
returnfig,global_points,global_point_label


defsegment_with_points(
image,
global_points,
global_point_label,
input_size=1024,
better_quality=False,
withContours=True,
use_retina=True,
mask_random_color=True,
):
input_size=int(input_size)
w,h=image.size
scale=input_size/max(w,h)
new_w=int(w*scale)
new_h=int(h*scale)
image=image.resize((new_w,new_h))

ifglobal_pointsisNoneorlen(global_points)<1orglobal_points[0]isNone:
returnimage,global_points,global_point_label
scaled_points=np.array([[int(x*scale)forxinpoint]forpointinglobal_points])
scaled_point_label=np.array(global_point_label)

ifscaled_points.size==0andscaled_point_label.size==0:
returnimage,global_points,global_point_label

nd_image=np.array(image)
img_tensor=(nd_image).astype(np.float32)/255
img_tensor=np.transpose(img_tensor,(2,0,1))

pts_sampled=np.reshape(scaled_points,[1,1,-1,2])
pts_labels=np.reshape(np.array(global_point_label),[1,1,-1])

results=compiled_model([img_tensor[None,...],pts_sampled,pts_labels])
predicted_logits=results[0]
predicted_iou=results[1]
all_masks=sigmoid(predicted_logits[0,0,:,:,:])>=0.5
predicted_iou=predicted_iou[0,0,...]

results=format_results(all_masks,predicted_iou,predicted_logits,0)
annotations,_=point_prompt(results,scaled_points,scaled_point_label,new_h,new_w)
annotations=np.array([annotations])

fig=process(
annotations=annotations,
image=image,
scale=(1024//input_size),
better_quality=better_quality,
mask_random_color=mask_random_color,
points=scaled_points,
bbox=None,
use_retina=use_retina,
withContours=withContours,
)

global_points=[]
global_point_label=[]
#returnfig,None
returnfig,global_points,global_point_label


defget_points_with_draw(image,cond_image,global_points,global_point_label,evt:gr.SelectData):
print(global_points)
iflen(global_points)==0:
image=copy.deepcopy(cond_image)
x,y=evt.index[0],evt.index[1]
label="AddMask"
point_radius,point_color=15,(
(255,255,0)
iflabel=="AddMask"
else(
255,
0,
255,
)
)
global_points.append([x,y])
global_point_label.append(1iflabel=="AddMask"else0)

ifimageisnotNone:
draw=ImageDraw.Draw(image)

draw.ellipse(
[
(x-point_radius,y-point_radius),
(x+point_radius,y+point_radius),
],
fill=point_color,
)

returnimage,global_points,global_point_label


defget_points_with_draw_(image,cond_image,global_points,global_point_label,evt:gr.SelectData):
iflen(global_points)==0:
image=copy.deepcopy(cond_image)
iflen(global_points)>2:
returnimage,global_points,global_point_label
x,y=evt.index[0],evt.index[1]
label="AddMask"
point_radius,point_color=15,(
(255,255,0)
iflabel=="AddMask"
else(
255,
0,
255,
)
)
global_points.append([x,y])
global_point_label.append(1iflabel=="AddMask"else0)

ifimageisnotNone:
draw=ImageDraw.Draw(image)
draw.ellipse(
[
(x-point_radius,y-point_radius),
(x+point_radius,y+point_radius),
],
fill=point_color,
)

iflen(global_points)==2:
x1,y1=global_points[0]
x2,y2=global_points[1]
ifx1<x2andy1<y2:
draw.rectangle([x1,y1,x2,y2],outline="red",width=5)
elifx1<x2andy1>=y2:
draw.rectangle([x1,y2,x2,y1],outline="red",width=5)
global_points[0][0]=x1
global_points[0][1]=y2
global_points[1][0]=x2
global_points[1][1]=y1
elifx1>=x2andy1<y2:
draw.rectangle([x2,y1,x1,y2],outline="red",width=5)
global_points[0][0]=x2
global_points[0][1]=y1
global_points[1][0]=x1
global_points[1][1]=y2
elifx1>=x2andy1>=y2:
draw.rectangle([x2,y2,x1,y1],outline="red",width=5)
global_points[0][0]=x2
global_points[0][1]=y2
global_points[1][0]=x1
global_points[1][1]=y1

returnimage,global_points,global_point_label


cond_img_p=gr.Image(label="InputwithPoint",value=default_example[0],type="pil")
cond_img_b=gr.Image(label="InputwithBox",value=default_example[0],type="pil")

segm_img_p=gr.Image(label="SegmentedImagewithPoint-Prompt",interactive=False,type="pil")
segm_img_b=gr.Image(label="SegmentedImagewithBox-Prompt",interactive=False,type="pil")


withgr.Blocks(css=css,title="EfficientSAM")asdemo:
global_points=gr.State([])
global_point_label=gr.State([])
withgr.Row():
withgr.Column(scale=1):
#Title
gr.Markdown(title)

withgr.Tab("Pointmode"):
#Images
withgr.Row(variant="panel"):
withgr.Column(scale=1):
cond_img_p.render()

withgr.Column(scale=1):
segm_img_p.render()

#Submit&Clear
####
withgr.Row():
withgr.Column():
withgr.Column():
segment_btn_p=gr.Button("SegmentwithPointPrompt",variant="primary")
clear_btn_p=gr.Button("Clear",variant="secondary")

gr.Markdown("Trysomeoftheexamplesbelow⬇️")
gr.Examples(
examples=examples,
inputs=[cond_img_p],
examples_per_page=4,
)

withgr.Column():
#Description
gr.Markdown(description_p)

withgr.Tab("Boxmode"):
#Images
withgr.Row(variant="panel"):
withgr.Column(scale=1):
cond_img_b.render()

withgr.Column(scale=1):
segm_img_b.render()

#Submit&Clear
withgr.Row():
withgr.Column():
withgr.Column():
segment_btn_b=gr.Button("SegmentwithBoxPrompt",variant="primary")
clear_btn_b=gr.Button("Clear",variant="secondary")

gr.Markdown("Trysomeoftheexamplesbelow⬇️")
gr.Examples(
examples=examples,
inputs=[cond_img_b],
examples_per_page=4,
)

withgr.Column():
#Description
gr.Markdown(description_p)

cond_img_p.select(
get_points_with_draw,
inputs=[segm_img_p,cond_img_p,global_points,global_point_label],
outputs=[segm_img_p,global_points,global_point_label],
)

cond_img_b.select(
get_points_with_draw_,
[segm_img_b,cond_img_b,global_points,global_point_label],
[segm_img_b,global_points,global_point_label],
)

segment_btn_p.click(
segment_with_points,
inputs=[cond_img_p,global_points,global_point_label],
outputs=[segm_img_p,global_points,global_point_label],
)

segment_btn_b.click(
segment_with_boxs,
inputs=[cond_img_b,segm_img_b,global_points,global_point_label],
outputs=[segm_img_b,global_points,global_point_label],
)

clear_btn_p.click(clear,outputs=[cond_img_p,segm_img_p,global_points,global_point_label])
clear_btn_b.click(clear,outputs=[cond_img_b,segm_img_b,global_points,global_point_label])

demo.queue()
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

