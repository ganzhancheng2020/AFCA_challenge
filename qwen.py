#模型下载
from modelscope import snapshot_download
from modelscope import AutoModelForCausalLM, AutoTokenizer
from prompt_template import get_sys_prompt
model_dir = snapshot_download('qwen/Qwen2-7B-Instruct')
lora_path = './output/Qwen2_instruct_lora/checkpoint-600'

device = "cuda" 
# the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(model, model_id=lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_dir)


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
