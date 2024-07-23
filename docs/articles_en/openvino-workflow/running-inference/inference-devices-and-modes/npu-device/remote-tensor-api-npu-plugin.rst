.. {#openvino_docs_OV_UG_supported_plugins_NPU_RemoteTensor_API}

Remote Tensor API of NPU Plugin
===============================


.. meta::
   :description: The Remote Tensor API of NPU plugin in OpenVINO™ supports
                 interoperability with existing native APIs, such as
                 Microsoft DirectX, or DMA-BUF System Heap.


The NPU plugin implementation of the ``ov::RemoteContext`` and ``ov::RemoteTensor`` interface supports NPU
pipeline developers who need memory sharing with existing native APIs, such as Microsoft DirectX,
or DMA-BUF System Heap. They allow you to avoid any memory copy overhead when plugging
OpenVINO™ inference into an existing NPU pipeline.

Supported scenario by the Remote Tensor API:

* The NPU plugin context and memory objects can be constructed from low-level device, display, or memory handles and used to create the OpenVINO™ ``ov::CompiledModel`` or ``ov::Tensor`` objects.

Class and function declarations for the API are defined in the following file: ``openvino/runtime/intel_npu/level_zero/level_zero.hpp``

The most common way to enable the interaction of your application with the Remote Tensor API is to use user-side utility classes
and functions that consume or produce native handles directly.

Context Sharing Between Application and NPU Plugin
###########################################################

NPU plugin classes that implement the ``ov::RemoteContext`` interface are responsible for context sharing.
Obtaining a context object is the first step in sharing pipeline objects.
The context object of the NPU plugin directly wraps Level Zero context, setting a scope for sharing the
``ov::RemoteTensor`` objects. The ``ov::RemoteContext`` object is retrieved from the NPU plugin.

Once you have obtained the context, you can use it to create ``ov::RemoteTensor`` objects.

Getting RemoteContext from the Plugin
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To request the current default context of the plugin, use one of the following methods:

.. tab-set::

   .. tab-item:: Get context from Core
      :sync: get-context-core

      .. doxygensnippet:: docs/articles_en/assets/snippets/npu_remote_objects_creation.cpp
          :language: cpp
          :fragment: [default_context_from_core]

   .. tab-item:: Get context from compiled model
      :sync: get-context-compiled-model

      .. doxygensnippet:: docs/articles_en/assets/snippets/npu_remote_objects_creation.cpp
          :language: cpp
          :fragment: [default_context_from_model]

Memory Sharing Between Application and NPU Plugin
###########################################################

The classes that implement the ``ov::RemoteTensor`` interface are the wrappers for native API
memory handles (which can be obtained from them at any time).

To create a shared tensor from a native memory handle, use dedicated ``create_tensor``, ``create_l0_host_tensor``, or ``create_host_tensor``
methods of the ``ov::RemoteContext`` sub-classes.
``ov::intel_npu::level_zero::LevelZero`` has multiple overloads methods which allow to wrap pre-allocated native handles with the ``ov::RemoteTensor``
object or request plugin to allocate specific device memory.
For more details, see the code snippets below:


.. tab-set::

   .. tab-item:: Wrap native handles
      :sync: wrap-native-handles

      .. tab-set::

         .. tab-item:: Direct3D12 resource
            :sync: d3d12

            .. doxygensnippet:: docs/articles_en/assets/snippets/npu_remote_objects_creation.cpp
               :language: cpp
               :fragment: [wrap_d3d12_resource]

         .. tab-item:: DMA-BUF System Heap resource
            :sync: dma-buf

            .. doxygensnippet:: docs/articles_en/assets/snippets/npu_remote_objects_creation.cpp
               :language: cpp
               :fragment: [wrap_dmabuf_resource]

   .. tab-item:: Allocate device memory
      :sync: allocate-device-memory

      .. tab-set::

         .. tab-item:: Remote Tensor - Level Zero host memory
            :sync: remote-level-zero-host-memory

            .. doxygensnippet:: docs/articles_en/assets/snippets/npu_remote_objects_creation.cpp
               :language: cpp
               :fragment: [allocate_remote_level_zero_host]

         .. tab-item:: Tensor - Level Zero host memory
            :sync: level-zero-host-memory

            .. doxygensnippet:: docs/articles_en/assets/snippets/npu_remote_objects_creation.cpp
               :language: cpp
               :fragment: [allocate_level_zero_host]


Limitations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Allocation of the Direct3D12 or DMA-BUF System Heap resources is user's responsibility.

Low-Level Methods for RemoteContext and RemoteTensor Creation
#####################################################################

The high-level wrappers mentioned above bring a direct dependency on native APIs to the user program.
If you want to avoid the dependency, you still can directly use the ``ov::Core::create_context()``,
``ov::RemoteContext::create_tensor()``, and ``ov::RemoteContext::get_params()`` methods.
On this level, native handles are re-interpreted as void pointers and all arguments are passed
using ``ov::AnyMap`` containers that are filled with ``std::string, ov::Any`` pairs.
Two types of map entries are possible: descriptor and container.
Descriptor sets the expected structure and possible parameter values of the map.

For possible low-level properties and their description, refer to the header file:
`remote_properties.hpp <https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/intel_npu/remote_properties.hpp>`__.

See Also
#######################################

* `ov::Core <https://docs.openvino.ai/2024/api/c_cpp_api/classov_1_1_core.html>`__
* `ov::RemoteTensor <https://docs.openvino.ai/2024/api/c_cpp_api/classov_1_1_remote_tensor.html>`__

