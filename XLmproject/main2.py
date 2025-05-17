import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_dataset
import sacrebleu
import numpy as np
import os

# --- Cell 1: Imports, Installations, and GPU Check ---

print("Setting up environment...")

# !!! Note: Library installations should be done via pip or conda in your terminal environment !!!
# !!! The !pip install lines from the Colab version are removed here. !!!

# Add this to help with memory fragmentation on CUDA (optional but can help with OOM)
# This environment variable needs to be set *before* PyTorch initializes CUDA memory management.
# Setting it here might be sufficient, but setting it in your terminal session before running
# the script (e.g., `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on Linux/macOS
# or `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on Windows Command Prompt) is more reliable.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Debug line to confirm device availability
print(f"\nDEBUG: torch.backends.mps.is_available() at script start: {torch.backends.mps.is_available()}")
print(f"DEBUG: torch.cuda.is_available() at script start: {torch.cuda.is_available()}")

# >>>>>>>>>>>>>> DEVICE DETECTION FOR GENERAL PC (CUDA > MPS > CPU) <<<<<<<<<<<<<<
if torch.cuda.is_available():
    print(f"GPU (CUDA) is available and will be used: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # For Apple Silicon (if you transfer back)
    print(f"MPS (Apple Silicon GPU) is available and will be used.")
    device = torch.device("mps")
else:
    print("No GPU (CUDA or MPS) found. Training will run on CPU, which will be significantly slower.")
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"DEBUG: Device determined by script is: {device}") # Keep this debug line

# --- Cell 2: Model and Tokenizer Loading ---

model_name = "Helsinki-NLP/opus-mt-hi-en"
print(f"\nLoading tokenizer and model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device) # Move model to GPU/CPU

# Define source and target languages for translation
source_lang = "Hinglish"
target_lang = "English"

# --- Cell 3: Dataset Loading and Preparation ---

print("\nLoading custom dataset from CSV...")

# >>>>>>>>>>>>>> Reverted: Loading from a local file path <<<<<<<<<<<<<<
# Assume my_hinglish_dataset.csv is in the same directory as the script
csv_file_name = "my_hinglish_dataset.csv"
# Construct the full path relative to the script's directory
csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_file_name)

print(f"Attempting to load dataset from: {csv_file_path}")

# Load the dataset from CSV
try:
    full_dataset = load_dataset("csv", data_files=csv_file_path)
except FileNotFoundError:
    print(f"ERROR: CSV file not found at {csv_file_path}.")
    print("Please ensure 'my_hinglish_dataset.csv' is in the same directory as the script.")
    exit()

split_key = list(full_dataset.keys())[0]
split_dataset = full_dataset[split_key].train_test_split(test_size=0.2, seed=42)

raw_datasets = DatasetDict({
    'train': split_dataset['train'].rename_columns({"hi_en": "Hinglish", "en": "English"}),
    'validation': split_dataset['test'].rename_columns({"hi_en": "Hinglish", "en": "English"})
})

print(f"Raw dataset structure: {raw_datasets}")
print(f"Train examples: {len(raw_datasets['train'])}")
print(f"Validation examples: {len(raw_datasets['validation'])}")
print(f"Example from raw dataset (train split):\n{raw_datasets['train'][0]}")

# --- Cell 4: Preprocessing the Dataset ---

print("\nPreprocessing dataset...")

# Keep the reduced sequence lengths for memory
max_input_length = 64
max_target_length = 64

def preprocess_function(examples):
    inputs = [ex for ex in examples[source_lang]]
    targets = [ex for ex in examples[target_lang]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
print(f"Tokenized dataset structure: {tokenized_datasets}")
print("Example from tokenized dataset (train split):")
print(tokenized_datasets['train'][0])

# --- Cell 5: Data Collator ---

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- Cell 6: Training Arguments ---

output_dir = "./results/hinglish_translator_model" # Results will be saved in this directory
logging_dir = "./logs" # TensorBoard logs will be saved here

# Start with a small batch size due to potential memory limits on unknown GPU
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8 # Effective batch size: 1 * 8 = 8

num_train_epochs = 3 # Start with 3, adjust later

print(f"\nConfiguring Training Arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=logging_dir,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    report_to="tensorboard",
    # >>>>>>>>>>>>>> UPDATED: Mixed Precision for CUDA GPUs <<<<<<<<<<<<<<
    # Set fp16 to True if CUDA is available (recommended for speed/memory on most NVIDIA GPUs)
    fp16=torch.cuda.is_available(),
    bf16=False, # Set bf16 to False unless you know your GPU supports it natively (e.g., A100)
    seed=42,
    # Disable pin_memory as it's not needed for CUDA and can sometimes cause issues
    # pin_memory=False, # This is often handled by DataCollator, but explicitly set if needed elsewhere
)

# --- Cell 7: Metrics Function ---

print("\nDefining compute_metrics function (for BLEU score calculation)...")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    try:
        preds = np.asarray(preds, dtype=np.int64)
        preds = np.squeeze(preds)
        if preds.ndim == 0:
            preds = preds.reshape(1)
    except Exception as e:
        print(f"ERROR (compute_metrics): Failed to convert/squeeze preds array. Error: {e}")
        print(f"DEBUG (compute_metrics): Raw preds data: {preds[:5] if isinstance(preds, list) or isinstance(preds, np.ndarray) else preds}")
        raise

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
    return {"bleu": result.score}

# --- Cell 8: Trainer Initialization and Training ---

print("\nInitializing Trainer and starting training...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting training... This might take a while.")
try:
    trainer.train()
    print("\nTraining complete!")
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    print("Common errors: Out of memory (OOM). Try reducing 'per_device_train_batch_size', 'gradient_accumulation_steps', or 'max_input_length/max_target_length'.")
    # Include CUDA-specific OOM troubleshooting for PC
    if device.type == 'cuda':
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Attempted to clear CUDA cache. If OOM persists, consider reducing memory settings further or using a more powerful GPU.")


print(f"\nSaving the fine-tuned model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model and tokenizer saved successfully!")

# --- Cell 9: Evaluation and Inference ---

print("\n--- Evaluation ---")
# Evaluate the model on the validation set
# Pass generation parameters directly to trainer.evaluate() for OOM mitigation
# This requires a recent version of transformers (installed via pip/conda in terminal setup)
eval_results = trainer.evaluate(
    gen_kwargs={"max_length": 128, "num_beams": 2} # Keep these reasonable, adjust if eval OOM occurs
)
print(f"Evaluation results: {eval_results}")
print(f"BLEU score on validation set: {eval_results.get('eval_bleu', 'N/A')}")

print("\n--- Inference Examples ---")
# Load the fine-tuned model for inference (optional, if you restarted kernel or want to verify load)
# You can load from the saved output_dir:
# model_for_inference = AutoModelForSeq2SeqLM.from_pretrained(output_dir).to(device)
# tokenizer_for_inference = AutoTokenizer.from_pretrained(output_dir)


def translate_hinglish_to_english(text, model_to_use, tokenizer_to_use):
    inputs = tokenizer_to_use(text, return_tensors="pt", max_length=max_input_length, truncation=True)
    # Move inputs to the same device as the model (GPU if available)
    inputs = {k: v.to(model_to_use.device) for k, v in inputs.items()}

    outputs = model_to_use.generate(
        **inputs,
        max_new_tokens=max_target_length, # Use max_new_tokens instead of max_length for generation control
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    translated_text = tokenizer_to_use.decode(outputs[0], skip_special_tokens=True)
    return translated_text

test_sentences_hinglish = [
    "Hello, kya haal hai?",
    "Mera naam Rohan hai.",
    "Yeh kitne ka hai?",
    "Mujhe bhookh lagi hai, kuch khaane ko milega?",
    "Aaj ka mausam kaisa hai?",
    "Main office jaa raha hoon, wahan milte hain.",
    "Aapki madad ke liye dhanyawaad.",
    "Please jaldi aao.",
    "Yeh bohot accha hai.",
    "Kya aap Mumbai se ho?",
    "Train late hai.",
    "Party shuru ho gayi hai."
]

print("\nTranslating example sentences:")
for sentence in test_sentences_hinglish:
    # Use the 'model' and 'tokenizer' variables which hold the fine-tuned model loaded by the Trainer
    translated = translate_hinglish_to_english(sentence, model, tokenizer)
    print(f"Hinglish: {sentence}")
    print(f"English:   {translated}\n")

