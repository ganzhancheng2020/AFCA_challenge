import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2_5-7b-chat')

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

def get_respond(prompt):
    response, history = model.chat(tokenizer, prompt, history=[])
    return response