import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers
from datasets import load_dataset
import os
from datetime import datetime

def main():
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    logs_dir = os.path.join(base_dir, "logs")
    models_dir = os.path.join(base_dir, "models")

    # 1. Load the dataset
    train_dataset = load_dataset('json', data_files=os.path.join(data_dir, 'notes.jsonl'), split='train')
    eval_dataset = load_dataset('json', data_files=os.path.join(data_dir, 'notes_validation.jsonl'), split='train')

    # 2. Initialize tokenizer
    base_model_id = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples["note"], truncation=True, padding="max_length", max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # 4. Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 5. Prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 6. Set up LoRA configuration
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # 7. Get PEFT model
    model = get_peft_model(model, config)

    # 8. Set up training arguments
    output_dir = os.path.join(models_dir, f"mistral-journal-finetune-{datetime.now().strftime('%Y-%m-%d-%H-%M')}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=500,
        learning_rate=2.5e-5,
        fp16=True,
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        optim="paged_adamw_8bit",
        logging_dir=logs_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    # 9. Create Trainer instance
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 10. Train the model
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # 11. Save the final model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
