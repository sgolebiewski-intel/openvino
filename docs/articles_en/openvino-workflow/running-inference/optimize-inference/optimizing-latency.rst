Optimizing for Latency
======================


.. toctree::
   :maxdepth: 1
   :hidden:

   optimizing-latency/model-caching-overview

.. meta::
   :description: OpenVINO provides methods that help to preserve minimal
                 latency despite the number of inference requests and
                 improve throughput without degrading latency.


An application that loads a single model and uses a single input at a time is
a widespread use case in deep learning. Surely, more requests can be created if
needed, for example to support :ref:`asynchronous inputs population <async_api>`.
However, **the number of parallel requests affects inference performance**
of the application.

Also, inferring multiple models on the same device, for example in the inference
pipeline, relies on whether the models are executed simultaneously or in a chain.
Running only one inference at a time on one device results in low latency, while
running multiple in high.

However, some conventional "root" devices (that is, CPU or GPU) can be in fact
internally composed of several "sub-devices". For instance, while serving multiple
clients simultaneously you can use OpenVINO to handle the "sub-devices"
transparently. It will improve application's throughput without impairing latency.
What is more, multi-socket CPUs can deliver as many requests at the same minimal
latency as there are NUMA nodes in the system. Similarly, a multi-tile GPU,
which is essentially multiple GPUs in a single package, can deliver a multi-tile
scalability with the number of inference requests, while preserving the
single-tile latency.

To achieve more "throughput", even in the typical latency-oriented cases, user's
expertise in inference devices is required. Naturally, OpenVINO can help you with
such a configuration via :doc:`high-level performance hints <high-level-performance-hints>`,
namely the `ov::hint::PerformanceMode::LATENCY <https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.properties.hint.PerformanceMode.html#openvino.properties.hint.PerformanceMode.LATENCY>`__
specified for the ``compile_model``.

.. note::

   :doc:`OpenVINO performance hints <high-level-performance-hints>` is a
   recommended way for performance configuration, which is both device-agnostic
   and future-proof.


**When multiple models are to be used simultaneously**, consider running
inference on separate devices for each of them. Finally, when multiple models
are executed in parallel on a device, using additional ``ov::hint::model_priority``
may help to define relative priorities of the models. Refer to the documentation
on the :doc:`OpenVINO feature support for devices <../../../../about-openvino/compatibility-and-support/supported-devices>`
to check if your device supports the feature.

**First-Inference Latency and Model Load/Compile Time**

In some cases, model loading and compilation contribute to the "end-to-end"
latency more than usual. For example, when the model is used exactly once, or
when it is unloaded and reloaded in a cycle, to free the memory for another
inference due to on-device memory limitations.

Such a "first-inference latency" scenario may pose an additional limitation on
the model load\compilation time, as inference accelerators (other than the CPU)
usually require a certain level of model compilation upon loading.
The :doc:`model caching <optimizing-latency/model-caching-overview>` option is
a way to lessen the impact over multiple application runs. If model caching is
not possible, for example, it may require write permissions for the application,
the CPU offers the fastest model load time almost every time.

To improve common "first-inference latency" scenario, model reading was replaced
with model mapping (using `mmap`) into a memory. But in some use cases (first of
all, if model is located on removable or network drive) mapping may lead to
latency increase. To switch mapping to reading, specify ``ov::enable_mmap(false)``
property for the ``ov::Core``.

Another way of dealing with first-inference latency is using the
:doc:`AUTO device selection inference mode <../inference-devices-and-modes/auto-device-selection>`.
It starts inference on the CPU, while waiting for the actual accelerator to load
the model. At that point, it shifts to the new device seamlessly.

Finally, note that any :doc:`throughput-oriented options <optimizing-throughput>`
 may significantly increase the model uptime.
