GenAI Use Cases
=====================

This article provides several use case scenarios for Generative AI model
inference. The applications presented in the code samples below
only require minimal configuration, like setting an inference device. Feel free
to explore and modify the source code as you need.


Using GenAI for Text-to-Image Generation
########################################

Examples below demonstrate inference on text-to-image models, like Stable Diffusion
1.5, 2.1, and LCM, with a text prompt as input. The :ref:`main.cpp <maincpp>`
sample shows basic usage of the ``Text2ImagePipeline`` pipeline.
:ref:`lora.cpp <loracpp>` shows how to apply LoRA adapters to the pipeline.


.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. tab-set::

         .. tab-item:: main.cpp
            :name: maincpp

            .. code-block:: cpp
               :emphasize-lines: 8-19

               // #include "openvino/genai/text2image/pipeline.hpp"

               // #include "imwrite.hpp"

               // int32_t main(int32_t argc, char* argv[]) {
               //     OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

                   const std::string models_path = argv[1], prompt = argv[2];
                   const std::string device = "CPU";  // GPU, NPU can be used as well.

                   ov::genai::Text2ImagePipeline pipe(models_path, device);
                   ov::Tensor image = pipe.generate(prompt,
                       ov::genai::width(512),
                       ov::genai::height(512),
                       ov::genai::num_inference_steps(20),
                       ov::genai::num_images_per_prompt(1));

                   // Saves images with a `num_images_per_prompt` name pattern.
                   imwrite("image_%d.bmp", image, true);

               //     return EXIT_SUCCESS;
               // }

         .. tab-item:: LoRA.cpp
            :name: loracpp

            .. literalinclude:: https://raw.githubusercontent.com/openvinotoolkit/openvino.genai/refs/heads/master/samples/cpp/text2image/lora.cpp
               :emphasize-lines: 8-38
               :language: cpp



      For more information, refer to the
      `C++ sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/text2image/README.md>`__

Using GenAI in Speech Recognition
#################################


The application, shown in code samples below, performs inference on speech
recognition Whisper Models. The samples include the ``WhisperPipeline`` class
and use audio files in WAV format at a sampling rate of 16 kHz as input.

.. tab-set::

   .. tab-item:: Python
      :sync: cpp

      .. code-block:: python

         import argparse
         import openvino_genai
         import librosa


         def read_wav(filepath):
             raw_speech, samplerate = librosa.load(filepath, sr=16000)
             return raw_speech.tolist()


         def main():
             parser = argparse.ArgumentParser()
             parser.add_argument("model_dir")
             parser.add_argument("wav_file_path")
             args = parser.parse_args()

             raw_speech = read_wav(args.wav_file_path)

             pipe = openvino_genai.WhisperPipeline(args.model_dir)

             def streamer(word: str) -> bool:
                 print(word, end="")
                 return False

             result = pipe.generate(
                 raw_speech,
                 max_new_tokens=100,
                 # The 'task' and 'language' parameters are supported for multilingual models only.
                 language="<|en|>",
                 task="transcribe",
                 return_timestamps=True,
                 streamer=streamer,
             )

             print()

             for chunk in result.chunks:
                 print(f"timestamps: [{chunk.start_ts}, {chunk.end_ts}] text: {chunk.text}")


      For more information, refer to the
      `Python sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/whisper_speech_recognition/README.md>`__.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp
         :emphasize-lines: 6-31

         // int main(int argc, char* argv[]) {
         //    if (3 > argc) {
         //        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\"");
         //    }

             std::string model_path = argv[1];
             std::string wav_file_path = argv[2];

             ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

             ov::genai::WhisperPipeline pipeline{model_path};

             ov::genai::WhisperGenerationConfig config{model_path + "/generation_config.json"};
             config.max_new_tokens = 100;
             // 'task' and 'language' parameters are supported for multilingual models only
             config.language = "<|en|>";
             config.task = "transcribe";
             config.return_timestamps = true;

             auto streamer = [](std::string word) {
                 std::cout << word;
                 return false;
             };

             auto result = pipeline.generate(raw_speech, config, streamer);

             std::cout << "\n";

             for (auto& chunk : *result.chunks) {
                 std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text << "\n";
             }
         }


      For more information, refer to the
      `C++ sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/whisper_speech_recognition/README.md>`__.


Using GenAI in Chat Scenario
############################

For chat scenarios where inputs and outputs represent a conversation, maintaining KVCache across inputs
may prove beneficial. The ``start_chat`` and ``finish_chat`` chat-specific methods are used to
mark a conversation session, as shown in the samples below:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         import argparse
         import openvino_genai


         def streamer(subword):
             print(subword, end='', flush=True)
             # The return flag corresponds to whether generation should be stopped or not.
             # False means continue generation.
             return False


         def main():
             parser = argparse.ArgumentParser()
             parser.add_argument('model_dir')
             args = parser.parse_args()

             device = 'CPU'  # GPU can be used as well.
             pipe = openvino_genai.LLMPipeline(args.model_dir, device)

             config = openvino_genai.GenerationConfig()
             config.max_new_tokens = 100

             pipe.start_chat()
             while True:
                 try:
                     prompt = input('question:\n')
                 except EOFError:
                     break
                 pipe.generate(prompt, config, streamer)
                 print('\n----------')
             pipe.finish_chat()


         if '__main__' == __name__:
             main()


      For more information, refer to the
      `Python sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/chat_sample/README.md>`__.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp
        :emphasize-lines: 10-29

         #include "openvino/genai/llm_pipeline.hpp"

         int main(int argc, char* argv[]) {
             if (2 != argc) {
                 throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR>");
             }
             std::string prompt;
             std::string model_path = argv[1];

             std::string device = "CPU";  // GPU, NPU can be used as well
             ov::genai::LLMPipeline pipe(model_path, device);

             ov::genai::GenerationConfig config;
             config.max_new_tokens = 100;
             std::function<bool(std::string)> streamer = [](std::string word) {
                 std::cout << word << std::flush;
                 // Return flag corresponds whether generation should be stopped.
                 // false means continue generation.
                 return false;
             };

             pipe.start_chat();
             std::cout << "question:\n";
             while (std::getline(std::cin, prompt)) {
                 pipe.generate(prompt, config, streamer);
                 std::cout << "\n----------\n"
                     "question:\n";
             }
             pipe.finish_chat();
         }


      For more information, refer to the
      `C++ sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/chat_sample/README.md>`__

Additional Resources
#####################

* :doc:`Install OpenVINO GenAI <../../../get-started/install-openvino/install-openvino-genai>`
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO GenAI Samples <https://github.com/openvinotoolkit/openvino.genai/tree/master/samples>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
