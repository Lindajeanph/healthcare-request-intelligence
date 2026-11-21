import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = 'Qwen/Qwen3-0.6B'
CATEGORIES = ['records_request', 'billing', 'insurance', 'appointment', 'technical_support']
SYSTEM = ("You are a healthcare administrative request triage classifier. "
          "Use only information in the request. Output a single JSON object with exactly these keys: "
          "category, priority, action. "
          "Allowed categories: records_request, billing, insurance, appointment, technical_support. "
          "Allowed priorities: normal, urgent. "
          "Allowed actions: route_to_roi_specialist, route_to_billing, route_to_insurance, "
          "route_to_scheduling, route_to_it. Do not release records or make clinical decisions. "
          "Use urgent only for an explicitly time-sensitive administrative request. No markdown.")

def prompt_for(request):
    return [{'role':'system','content':SYSTEM}, {'role':'user','content':request}]

def load_model(adapter=None):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not detected. Check WSL NVIDIA driver / CUDA-enabled PyTorch.')
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to('cuda')
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model

def predict(tok, model, request):
    text = tok.apply_chat_template(prompt_for(request), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                pad_token_id=tok.eos_token_id)
    result = tok.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return result

def parse_result(raw):
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or set(obj) != {'category','priority','action'}:
            return None
        return obj
    except (ValueError, TypeError):
        return None
