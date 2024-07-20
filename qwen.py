#模型下载
from modelscope import snapshot_download
from modelscope import AutoModelForCausalLM, AutoTokenizer
from prompt_template import get_sys_prompt
from peft import PeftModel
import logging
import torch
logger = logging.getLogger(__name__)


model_dir = snapshot_download('qwen/Qwen2-7B-Instruct')
lora_path = './output/Qwen2_instruct_lora/20240720/checkpoint-500'

device = "cuda"
# the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(model, model_id=lora_path)

device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")

#model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model.to(device)


def qwen_response(prompt):

    messages = [
        {"role": "system", "content": get_sys_prompt()},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response