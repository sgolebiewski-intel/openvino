BigTransferImageClassificationModelQuantizationpipelinewithNNCF
=======================================================================

ThistutorialdemonstratestheQuantizationoftheBigTransferImage
Classificationmodel,whichisfine-tunedonthesub-setofImageNet
datasetwith10classlabelswith
`NNCF<https://github.com/openvinotoolkit/nncf>`__.Ituses
`BiT-M-R50x1/1<https://www.kaggle.com/models/google/bit/frameworks/tensorFlow2/variations/m-r50x1/versions/1?tfhub-redirect=true>`__
model,whichistrainedonImageNet-21k.BigTransferisarecipefor
pre-trainingimageclassificationmodelsonlargesuperviseddatasets
andefficientlyfine-tuningthemonanygiventargettask.Therecipe
achievesexcellentperformanceonawidevarietyoftasks,evenwhen
usingveryfewlabeledexamplesfromthetargetdataset.Thistutorial
usesOpenVINObackendforperformingmodelquantizationinNNCF.

Tableofcontents:
^^^^^^^^^^^^^^^^^^

-`PrepareDataset<#prepare-dataset>`__
-`Plottingdatasamples<#plotting-data-samples>`__
-`ModelFine-tuning<#model-fine-tuning>`__
-`Performmodeloptimization(IR)
step<#perform-model-optimization-ir-step>`__
-`ComputeaccuracyoftheTF
model<#compute-accuracy-of-the-tf-model>`__
-`ComputeaccuracyoftheOpenVINO
model<#compute-accuracy-of-the-openvino-model>`__
-`QuantizeOpenVINOmodelusing
NNCF<#quantize-openvino-model-using-nncf>`__
-`Computeaccuracyofthequantized
model<#compute-accuracy-of-the-quantized-model>`__
-`CompareFP32andINT8accuracy<#compare-fp32-and-int8-accuracy>`__
-`Compareinferenceresultsonone
picture<#compare-inference-results-on-one-picture>`__

..code::ipython3

importplatform

%pipinstall-q"tensorflow-macos>=2.5;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version>'3.8'"#macOSM1andM2
%pipinstall-q"tensorflow-macos>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine=='arm64'andpython_version<='3.8'"#macOSM1andM2
%pipinstall-q"tensorflow>=2.5;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version>'3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform=='darwin'andplatform_machine!='arm64'andpython_version<='3.8'"#macOSx86
%pipinstall-q"tensorflow>=2.5;sys_platform!='darwin'andpython_version>'3.8'"
%pipinstall-q"tensorflow>=2.5,<=2.12.0;sys_platform!='darwin'andpython_version<='3.8'"

%pipinstall-q"openvino>=2024.0.0""nncf>=2.7.0""tensorflow-hub>=0.15.0"tf_keras
%pipinstall-q"scikit-learn>=1.3.2"

ifplatform.system()!="Windows":
%pipinstall-q"matplotlib>=3.4""tensorflow_datasets>=4.9.0"
else:
%pipinstall-q"matplotlib>=3.4,<3.7""tensorflow_datasets>=4.9.0<4.9.3"


..parsed-literal::

Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.
Note:youmayneedtorestartthekerneltouseupdatedpackages.


..code::ipython3

importos
importnumpyasnp
frompathlibimportPath

fromopenvino.runtimeimportCore
importopenvinoasov
importnncf
importlogging

fromnncf.common.logging.loggerimportset_log_level

set_log_level(logging.ERROR)

fromsklearn.metricsimportaccuracy_score

os.environ["TF_USE_LEGACY_KERAS"]="1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["TFHUB_CACHE_DIR"]=str(Path("./tfhub_modules").resolve())

importtensorflowastf
importtensorflow_datasetsastfds
importtensorflow_hubashub

tfds.core.utils.gcs_utils._is_gcs_disabled=True
os.environ["NO_GCE_CHECK"]="true"


..parsed-literal::

INFO:nncf:NNCFinitializedsuccessfully.Supportedframeworksdetected:torch,tensorflow,onnx,openvino


..code::ipython3

core=Core()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#Fortop5labels.
MAX_PREDS=1
TRAINING_BATCH_SIZE=128
BATCH_SIZE=1
IMG_SIZE=(256,256)#DefaultImagenetimagesize
NUM_CLASSES=10#ForImagenettedataset
FINE_TUNING_STEPS=1
LR=1e-5

MEAN_RGB=(0.485*255,0.456*255,0.406*255)#FromImagenetdataset
STDDEV_RGB=(0.229*255,0.224*255,0.225*255)#FromImagenetdataset

PrepareDataset
~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

datasets,datasets_info=tfds.load(
"imagenette/160px",
shuffle_files=True,
as_supervised=True,
with_info=True,
read_config=tfds.ReadConfig(shuffle_seed=0),
)
train_ds,validation_ds=datasets["train"],datasets["validation"]


..parsed-literal::

2024-07-1223:29:23.376213:Etensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266]failedcalltocuInit:CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:forwardcompatibilitywasattemptedonnonsupportedHW
2024-07-1223:29:23.376437:Etensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312]kernelversion470.182.3doesnotmatchDSOversion470.223.2--cannotfindworkingdevicesinthisconfiguration


..code::ipython3

defpreprocessing(image,label):
image=tf.image.resize(image,IMG_SIZE)
image=tf.cast(image,tf.float32)/255.0
label=tf.one_hot(label,NUM_CLASSES)
returnimage,label


train_dataset=train_ds.map(preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
validation_dataset=(
validation_ds.map(preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
)

..code::ipython3

#Classlabelsdictionarywithimagenettesamplenamesandclasses
lbl_dict=dict(
n01440764="tench",
n02102040="Englishspringer",
n02979186="cassetteplayer",
n03000684="chainsaw",
n03028079="church",
n03394916="Frenchhorn",
n03417042="garbagetruck",
n03425413="gaspump",
n03445777="golfball",
n03888257="parachute",
)

#Imagenettesamplesnameindex
class_idx_dict=[
"n01440764",
"n02102040",
"n02979186",
"n03000684",
"n03028079",
"n03394916",
"n03417042",
"n03425413",
"n03445777",
"n03888257",
]


deflabel_func(key):
returnlbl_dict[key]

Plottingdatasamples
~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

importmatplotlib.pyplotasplt

#Gettheclasslabelsfromthedatasetinfo
class_labels=datasets_info.features["label"].names

#Displaylabelsalongwiththeexamples
num_examples_to_display=4
fig,axes=plt.subplots(nrows=1,ncols=num_examples_to_display,figsize=(10,5))

fori,(image,label_index)inenumerate(train_ds.take(num_examples_to_display)):
label_name=class_labels[label_index.numpy()]

axes[i].imshow(image.numpy())
axes[i].set_title(f"{label_func(label_name)}")
axes[i].axis("off")
plt.tight_layout()
plt.show()



..image::tensorflow-bit-image-classification-nncf-quantization-with-output_files/tensorflow-bit-image-classification-nncf-quantization-with-output_9_0.png


..code::ipython3

#Gettheclasslabelsfromthedatasetinfo
class_labels=datasets_info.features["label"].names

#Displaylabelsalongwiththeexamples
num_examples_to_display=4
fig,axes=plt.subplots(nrows=1,ncols=num_examples_to_display,figsize=(10,5))

fori,(image,label_index)inenumerate(validation_ds.take(num_examples_to_display)):
label_name=class_labels[label_index.numpy()]

axes[i].imshow(image.numpy())
axes[i].set_title(f"{label_func(label_name)}")
axes[i].axis("off")
plt.tight_layout()
plt.show()



..image::tensorflow-bit-image-classification-nncf-quantization-with-output_files/tensorflow-bit-image-classification-nncf-quantization-with-output_10_0.png


ModelFine-tuning
~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#LoadtheBigTransfermodel
bit_model_url="https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1/versions/1"
bit_m=hub.KerasLayer(bit_model_url,trainable=True)

#Customizethemodelforthenewtask
model=tf.keras.Sequential([bit_m,tf.keras.layers.Dense(NUM_CLASSES,activation="softmax")])

#Compilethemodel
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
loss="categorical_crossentropy",
metrics=["accuracy"],
)

#Fine-tunethemodel
model.fit(
train_dataset.take(3000),
epochs=FINE_TUNING_STEPS,
validation_data=validation_dataset.take(1000),
)
model.save("./bit_tf_model/",save_format="tf")


..parsed-literal::

101/101[==============================]-968s9s/step-loss:0.5046-accuracy:0.8758-val_loss:0.0804-val_accuracy:0.9660


..parsed-literal::

WARNING:absl:Founduntracedfunctionssuchas_update_step_xlawhilesaving(showing1of1).Thesefunctionswillnotbedirectlycallableafterloading.


Performmodeloptimization(IR)step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

ir_path=Path("./bit_ov_model/bit_m_r50x1_1.xml")
ifnotir_path.exists():
print("Initiatingmodeloptimization..!!!")
ov_model=ov.convert_model("./bit_tf_model")
ov.save_model(ov_model,ir_path)
else:
print(f"IRmodel{ir_path}alreadyexists.")


..parsed-literal::

Initiatingmodeloptimization..!!!


ComputeaccuracyoftheTFmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

tf_model=tf.keras.models.load_model("./bit_tf_model/")

tf_predictions=[]
gt_label=[]

for_,labelinvalidation_dataset:
forcls_labelinlabel:
l_list=cls_label.numpy().tolist()
gt_label.append(l_list.index(1))

forimg_batch,label_batchinvalidation_dataset:
tf_result_batch=tf_model.predict(img_batch,verbose=0)
foriinrange(len(img_batch)):
tf_result=tf_result_batch[i]
tf_result=tf.reshape(tf_result,[-1])
top5_label_idx=np.argsort(tf_result)[-MAX_PREDS::][::-1]
tf_predictions.append(top5_label_idx)

#ConverttheliststoNumPyarraysforaccuracycalculation
tf_predictions=np.array(tf_predictions)
gt_label=np.array(gt_label)

tf_acc_score=accuracy_score(tf_predictions,gt_label)

ComputeaccuracyoftheOpenVINOmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

Selectdeviceforinference:

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

ov_fp32_model=core.read_model("./bit_ov_model/bit_m_r50x1_1.xml")
ov_fp32_model.reshape([1,IMG_SIZE[0],IMG_SIZE[1],3])

#TargetdevicesettoCPU(OtheroptionsEx:AUTO/GPU/dGPU/)
compiled_model=ov.compile_model(ov_fp32_model,device.value)
output=compiled_model.outputs[0]

ov_predictions=[]
forimg_batch,_invalidation_dataset:
forimageinimg_batch:
image=tf.expand_dims(image,axis=0)
pred=compiled_model(image)[output]
ov_result=tf.reshape(pred,[-1])
top_label_idx=np.argsort(ov_result)[-MAX_PREDS::][::-1]
ov_predictions.append(top_label_idx)

fp32_acc_score=accuracy_score(ov_predictions,gt_label)

QuantizeOpenVINOmodelusingNNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

ModelQuantizationusingNNCF

1.PreprocessingandpreparingvalidationsamplesforNNCFcalibration
2.PerformNNCFQuantizationonOpenVINOFP32model
3.SerializeQuantizedOpenVINOINT8model

..code::ipython3

defnncf_preprocessing(image,label):
image=tf.image.resize(image,IMG_SIZE)
image=image-MEAN_RGB
image=image/STDDEV_RGB
returnimage


val_ds=validation_ds.map(nncf_preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

calibration_dataset=nncf.Dataset(val_ds)

ov_fp32_model=core.read_model("./bit_ov_model/bit_m_r50x1_1.xml")

ov_int8_model=nncf.quantize(ov_fp32_model,calibration_dataset,fast_bias_correction=False)

ov.save_model(ov_int8_model,"./bit_ov_int8_model/bit_m_r50x1_1_ov_int8.xml")



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



Computeaccuracyofthequantizedmodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

nncf_quantized_model=core.read_model("./bit_ov_int8_model/bit_m_r50x1_1_ov_int8.xml")
nncf_quantized_model.reshape([1,IMG_SIZE[0],IMG_SIZE[1],3])

#TargetdevicesettoCPUbydefault
compiled_model=ov.compile_model(nncf_quantized_model,device.value)
output=compiled_model.outputs[0]

ov_predictions=[]
inp_tensor=nncf_quantized_model.inputs[0]
out_tensor=nncf_quantized_model.outputs[0]

forimg_batch,_invalidation_dataset:
forimageinimg_batch:
image=tf.expand_dims(image,axis=0)
pred=compiled_model(image)[output]
ov_result=tf.reshape(pred,[-1])
top_label_idx=np.argsort(ov_result)[-MAX_PREDS::][::-1]
ov_predictions.append(top_label_idx)

int8_acc_score=accuracy_score(ov_predictions,gt_label)

CompareFP32andINT8accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

print(f"Accuracyofthetensorflowmodel(fp32):{tf_acc_score*100:.2f}%")
print(f"AccuracyoftheOpenVINOoptimizedmodel(fp32):{fp32_acc_score*100:.2f}%")
print(f"AccuracyoftheOpenVINOquantizedmodel(int8):{int8_acc_score*100:.2f}%")
accuracy_drop=fp32_acc_score-int8_acc_score
print(f"AccuracydropbetweenOVFP32andINT8model:{accuracy_drop*100:.1f}%")


..parsed-literal::

Accuracyofthetensorflowmodel(fp32):96.60%
AccuracyoftheOpenVINOoptimizedmodel(fp32):96.60%
AccuracyoftheOpenVINOquantizedmodel(int8):96.20%
AccuracydropbetweenOVFP32andINT8model:0.4%


Compareinferenceresultsononepicture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`backtotop⬆️<#table-of-contents>`__

..code::ipython3

#Accessingvalidationsample
sample_idx=50
vds=datasets["validation"]

iflen(vds)>sample_idx:
sample=vds.take(sample_idx+1).skip(sample_idx).as_numpy_iterator().next()
else:
print("Datasetdoesnothaveenoughsamples...!!!")

#Imagedata
sample_data=sample[0]

#Labelinfo
sample_label=sample[1]

#Imagedatapre-processing
image=tf.image.resize(sample_data,IMG_SIZE)
image=tf.expand_dims(image,axis=0)
image=tf.cast(image,tf.float32)/255.0


#OpenVINOinference
defov_inference(model:ov.Model,image)->str:
compiled_model=ov.compile_model(model,device.value)
output=compiled_model.outputs[0]
pred=compiled_model(image)[output]
ov_result=tf.reshape(pred,[-1])
pred_label=np.argsort(ov_result)[-MAX_PREDS::][::-1]
returnpred_label


#OpenVINOFP32model
ov_fp32_model=core.read_model("./bit_ov_model/bit_m_r50x1_1.xml")
ov_fp32_model.reshape([1,IMG_SIZE[0],IMG_SIZE[1],3])

#OpenVINOINT8model
ov_int8_model=core.read_model("./bit_ov_int8_model/bit_m_r50x1_1_ov_int8.xml")
ov_int8_model.reshape([1,IMG_SIZE[0],IMG_SIZE[1],3])

#OpenVINOFP32modelinference
ov_fp32_pred_label=ov_inference(ov_fp32_model,image)

print(f"Predictedlabelforthesamplepicturebyfloat(fp32)model:{label_func(class_idx_dict[int(ov_fp32_pred_label)])}\n")

#OpenVINOFP32modelinference
ov_int8_pred_label=ov_inference(ov_int8_model,image)
print(f"Predictedlabelforthesamplepicturebyqunatized(int8)model:{label_func(class_idx_dict[int(ov_int8_pred_label)])}\n")

#Plottingtheimagesamplewithgroundtruth
plt.figure()
plt.imshow(sample_data)
plt.title(f"Groundtruth:{label_func(class_idx_dict[sample_label])}")
plt.axis("off")
plt.show()


..parsed-literal::

Predictedlabelforthesamplepicturebyfloat(fp32)model:gaspump

Predictedlabelforthesamplepicturebyqunatized(int8)model:gaspump




..image::tensorflow-bit-image-classification-nncf-quantization-with-output_files/tensorflow-bit-image-classification-nncf-quantization-with-output_27_1.png

