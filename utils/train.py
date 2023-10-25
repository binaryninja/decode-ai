from datetime import datetime
import os
import sys

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

from datasets import load_dataset

# pip install git+https://github.com/huggingface/transformers.git@main bitsandbytes  # we need latest transformers for this
# pip install git+https://github.com/huggingface/peft.git@4c611f4
# pip install datasets==2.10.1
# pip install wandb
# pip install scipy

#TODO change this to my huggingface dataset

# dataset1 = load_dataset("b-mc2/sql-create-context", split="train")
# train_dataset = dataset1.train_test_split(test_size=0.1)["train"]
# eval_dataset = dataset1.train_test_split(test_size=0.1)["test"]



# dataset2 = load_dataset("dyngnosis/function_names_v2", split="train")
# train_dataset = dataset2.train_test_split(test_size=0.1)["train"]
# eval_dataset = dataset2.train_test_split(test_size=0.1)["test"]

dataset2 = load_dataset("dyngnosis/function_names_v2", split="train")

# Split the dataset into training and evaluation sets
split_data = dataset2.train_test_split(test_size=0.01)
train_dataset = split_data["train"]
eval_dataset = split_data["test"]

print(train_dataset[3])
print(len(train_dataset))
print(len(eval_dataset))



base_model = "codellama/CodeLlama-34b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-Instruct-hf")

eval_prompt = """You are an advanced malware reverse engineer capable of understanding decompiled C code and identifying malicious functionality please create a ### New Function Name from the decompiled code in the ### Context.

You must output a descriptive ### Function name for the decompiled code provided in ### Context.

### Context:
void fcn.004014f0(void)\n\n{\n    uint uVar1;\n    \n    uVar1 = (*_sym.imp.KERNEL32.dll_LoadLibraryA)(\"advapi32\", \"RegOpenKeyExW\");\n    *0x4410e8 = (*_sym.imp.KERNEL32.dll_GetProcAddress)(uVar1);\n    return;\n}

### New Function Name:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=4096,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""You are an advanced malware reverse engineer capable of understanding decompiled C code and identifying malicious functionality please create a ### New Function Name from the decompiled code in the ### Context.

You must output a descriptive ### Function name for the decompiled code provided in ### Context.

### Context:
{data_point["input"]}

### New Function Name:
{data_point["output"]}

"""
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.train() # put model back into training mode
model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
resume_from_checkpoint = ""

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")


if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

batch_size =2
per_device_train_batch_size = 1
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "instruct-decode-llama-4096-34b"

training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        #this was defaulting to 8 and causing OOM errors
        per_device_eval_batch_size=per_device_train_batch_size,
        warmup_steps=50,
        max_steps=20000,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"decodellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)


trainer.train()
