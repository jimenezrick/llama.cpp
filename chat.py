import pprint

from llama_cpp import Llama
from llama_cpp.llama_cache import LlamaDiskCache

MISTRAL_7B_32K_MODEL = "./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"  # https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
LLAMA3_8B_32K_MODEL = "./models/Llama-3-8B-Instruct-32k-v0.1.Q5_K_M.gguf"  # https://huggingface.co/MaziyarPanahi/Llama-3-8B-Instruct-32k-v0.1-GGUF

ALL_GPU_LAYERS = -1

llm = Llama(
    # model_path=MISTRAL_7B_32K_MODEL,
    model_path=LLAMA3_8B_32K_MODEL,
    n_gpu_layers=ALL_GPU_LAYERS,
    n_ctx=22 * 1024,  # `0` for context length from model
)
llm.set_cache(LlamaDiskCache())
output = llm.create_completion(
    prompt="""Q: Combine the JSON objects `{"aa": 123}` and `{"bb": [1,2,3]}`. Response just with a valid JSON. Don't add markdown markup. A: """,
    max_tokens=None,
    stop=["Q:"],
)
pprint.pp(output)
print("---")
print(output["choices"][0]["text"].strip(" \n"))


prompt = """Q: Write a single 1-5 words label that represents the best the next email.

Dear Ricardo,

Thank you for using the Magna Groups Enterprises Limited online booking system to book your child onto one or more sessions.

A payment of £ 560.00 has been taken from your credit / debit card.

If you have any queries please contact us on 0333 012 4984 quoting bookings reference 040514

To view the session(s) booked, please log into your account.Â 

We look forward to welcoming your child to Magna Groups Enterprises Limited and if you have any queries or concerns, please do not hesitate to speak to the Club Manager or call direct on 0333 012 4984.

A: 
"""
output = llm.create_completion(
    prompt=prompt,
    max_tokens=None,
    stop=["Q:"],
)
pprint.pp(output)
print("---")
print(output["choices"][0]["text"].strip(" \n"))
