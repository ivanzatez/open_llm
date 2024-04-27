from llama_cpp import Llama
from decouple import config


# Model used for inference --> # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
# https://github.com/ggerganov/llama.cpp

class Config:
    LLM_PATH = config("LLM_PATH", default="")
    LLM = Llama(
            model_path= LLM_PATH,  # path to GGUF file
            n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=12, # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=35, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
            )
print("llm is initialized and ready to be imported")