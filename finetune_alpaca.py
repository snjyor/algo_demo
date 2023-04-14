import torch
import transformers
import datasets
tokenizer = transformers.AutoTokenizer.from_pretrained('cerebras/Cerebras-GPT-2.7B')
tokenizer.pad_token_id = 0
dataset = datasets.load_dataset('json', data_files='/content/drive/MyDrive/Colab Notebooks/data/alpaca_data_cleaned.json')


cutoff_len = 512

def generate_prompt(entry):
    if entry['input']:
        return f"User: {entry['instruction']}: {entry['input']}\n\nAssistant: {entry['output']}"
    else:
        return f"User: {entry['instruction']}\n\nAssistant: {entry['output']}"

def tokenize(item, add_eos_token=True):
    result = tokenizer(
        generate_prompt(item),
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

train_val = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
train_data = train_val["train"].shuffle().map(tokenize)
val_data = train_val["test"].shuffle().map(tokenize)

if 'model' in globals():
    del model
    torch.cuda.empty_cache()

model = transformers.AutoModelForCausalLM.from_pretrained(
    'cerebras/Cerebras-GPT-2.7B',
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={'': 0}
)


import peft

model = peft.prepare_model_for_int8_training(model)

model = peft.get_peft_model(model, peft.LoraConfig(
    r=8,
    lora_alpha=16,
    # target_modules=["q_proj", "v_proj"],
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
))

import os
import wandb

output_dir = 'lora-cerebras-gpt2.7b-alpaca'

use_wandb = True,
wandb_run_name = f"{output_dir}-{wandb.util.generate_id()}"

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]=output_dir

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    optim="adamw_torch",
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    output_dir=output_dir,
    save_total_limit=3,

    report_to="wandb" if use_wandb else None,
    run_name=wandb_run_name if use_wandb else None,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False
result = trainer.train('lora-cerebras-gpt2.7b-alpaca/checkpoint-800')
model.save_pretrained(output_dir)

wandb.finish()


model.config
print(model.dtype)

model.half()

text = "Human: Can I run inference on my local machine?\nAssistant:"

inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)

generation_config = transformers.GenerationConfig(
    max_new_tokens=100,
    temperature=0.2,
    top_p=0.75,
    top_k=50,
    repetition_penalty=1.2,
    do_sample=True,
    early_stopping=True,
    #     num_beams=5,

    pad_token_id=model.config.pad_token_id,
    eos_token_id=model.config.eos_token_id,
)

with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        generation_config=generation_config
    )[0].cuda()

result = tokenizer.decode(output, skip_special_tokens=True).strip()
print(result)

