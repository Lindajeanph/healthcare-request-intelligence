import json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from common import MODEL_ID, prompt_for

if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Verify WSL/PyTorch installation first.')
with open('data/train.jsonl',encoding='utf-8') as f:
    rows = [json.loads(line) for line in f]
# Prompt-completion format computes loss on responses only.
dataset = Dataset.from_list([
    {'prompt':prompt_for(row['request']),
     'completion':[{'role':'assistant','content':json.dumps(row['answer'],separators=(',',':'))}]}
    for row in rows
])
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
args = SFTConfig(
    output_dir='outputs/training', num_train_epochs=3,
    per_device_train_batch_size=1, gradient_accumulation_steps=8,
    learning_rate=2e-4, max_length=512,
    fp16=True, bf16=False, gradient_checkpointing=True,
    logging_steps=10, save_strategy='epoch',
    completion_only_loss=True, report_to='none',
)
trainer = SFTTrainer(
    model=model, args=args, train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
                           target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
)
trainer.train()
trainer.save_model('outputs/hri-lora')
tokenizer.save_pretrained('outputs/hri-lora')
print('Saved LoRA adapter to outputs/hri-lora')
