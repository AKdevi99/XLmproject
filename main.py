import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import sacrebleu
import numpy as np
import os

# --- Cell 1: Imports, Installations, and GPU Check ---

# Ensure required libraries are installed. Run this cell first.
# Note: Paperspace Gradient's PyTorch containers often have torch pre-installed.
# If you encounter issues with PyTorch not being found or not being compatible,
# you might need to uncomment and run the specific torch installation line.
print("Installing required libraries...")

# Debug line to confirm MPS availability at script start (keep this)
print(f"DEBUG: torch.backends.mps.is_available() at script start: {torch.backends.mps.is_available()}")

# >>>>>>>>>>>>>> IMPORTANT: REPLACE YOUR ENTIRE DEVICE DETECTION BLOCK WITH THIS <<<<<<<<<<<<<<
# Force device to MPS if available, as auto-detection seems to be misbehaving in the script's context
if torch.backends.mps.is_available():
    print(f"MPS (Apple Silicon GPU) is available and will be used.")
    device = torch.device("mps")
elif torch.cuda.is_available(): # This is for NVIDIA GPUs, generally not on Mac M3
    print(f"GPU (CUDA) is available: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("No GPU or MPS found. Training will run on CPU, which will be significantly slower.")
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"DEBUG: Device determined by script is: {device}") # Keep this debug line

# --- Cell 2: Model and Tokenizer Loading ---

# This model (Helsinki-NLP/opus-mt-hi-en) translates Hindi to English.
# For actual Hinglish-to-English, you would ideally:
# 1. Fine-tune this model on a Hinglish-English parallel corpus.
# 2. Or, choose a larger multilingual model (like mBART or mT5) and fine-tune it.
# This serves as a good starting point to demonstrate the pipeline.
model_name = "Helsinki-NLP/opus-mt-hi-en"
print(f"\nLoading tokenizer and model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device) # Move model to GPU/CPU


# Define source and target languages for translation
source_lang = "Hinglish" # This should match the column name you renamed your Hinglish data to
target_lang = "English"  # This should match the column name you renamed your English data to

from datasets import load_dataset, DatasetDict
import os # Import os module to join paths

# --- Cell 3: Dataset Loading ---
print("\nLoading custom dataset from CSV...")

# Define the path to your CSV file
# >>> IMPORTANT: Ensure 'my_hinglish_dataset.csv' is the exact name of your CSV file <<<
# >>>          and that it is in the same directory as your main.py script. <<<
csv_file_name = "my_hinglish_dataset.csv" # You can change this name if your file is named differently
csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_file_name)

# Load the dataset from CSV
try:
    full_dataset = load_dataset("csv", data_files=csv_file_path)
except FileNotFoundError:
    print(f"ERROR: CSV file not found at {csv_file_path}. Please ensure the file is in the correct directory.")
    exit() # Exit the script if the file is not found

# If the CSV only has one split (e.g., 'train'), you'll need to split it into train/validation.
split_key = list(full_dataset.keys())[0] # Get the key of the loaded split (e.g., 'train')

# Split the dataset into training and validation sets (80% train, 20% validation)
split_dataset = full_dataset[split_key].train_test_split(test_size=0.2, seed=42)

# Rename the columns to 'Hinglish' and 'English' for consistency with the rest of your code.
# These are the exact column names you provided: 'hi_en' and 'en'.
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

# Set max sequence lengths. Adjust based on your data's sentence lengths.
# Longer sequences require more memory.
max_input_length = 96
max_target_length = 96

def preprocess_function(examples):
    inputs = [ex for ex in examples[source_lang]]
    targets = [ex for ex in examples[target_lang]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
print(f"Tokenized dataset structure: {tokenized_datasets}")
print("Example from tokenized dataset (train split):")
print(tokenized_datasets['train'][0])

# --- Cell 5: Data Collator ---

# Data collator that will be used for batching examples to the model.
# It dynamically pads the input and target sequences to the maximum length in the batch.
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- Cell 6: Training Arguments ---

output_dir = "./results/hinglish_translator_model" # Where model checkpoints will be saved
logging_dir = "./logs" # Directory for TensorBoard logs

# Adjust batch size based on your GPU memory (M4000 has 8GB).
# For a small model and max_length 128, 8-16 might be feasible. Start low and increase if possible.
per_device_train_batch_size = 1
per_device_eval_batch_size = 1

# Number of training epochs. For demonstration, a small number.
# For proper fine-tuning on a larger dataset, you might need 3-10 epochs or more.
num_train_epochs = 3

print(f"\nConfiguring Training Arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=8,
    warmup_steps=500, # Number of warmup steps for learning rate scheduler
    weight_decay=0.01, # Strength of weight decay
    logging_dir=logging_dir, # Directory for storing logs (for TensorBoard)
    logging_steps=10, # Log every N updates steps
    eval_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch", # Save model at the end of each epoch
    load_best_model_at_end=True, # Load the best model saved during training
    metric_for_best_model="bleu", # Metric to use for loading the best model
    greater_is_better=True, # Higher BLEU is better
    report_to="tensorboard", # Integrates with TensorBoard for visualization
    fp16=False, # For CUDA GPUs, set to False if MPS is available
    bf16=torch.backends.mps.is_available(), # Enable bf16 for MPS if available # Use mixed precision training if GPU is available (recommended for faster training)
    seed=42, # For reproducibility
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

    # Debugging prints (keep these to verify the new shape)
    print(f"DEBUG (compute_metrics): Type of preds: {type(preds)}")
    if hasattr(preds, 'shape'):
        print(f"DEBUG (compute_metrics): Shape of preds: {preds.shape}")
        if preds.ndim == 3 and preds.shape[-1] == tokenizer.vocab_size:
            # >>>>>>>>>>>>>> IMPORTANT: THIS IS THE KEY FIX FOR THE LOGITS ISSUE <<<<<<<<<<<<<<
            # If preds is (batch_size, num_beams, vocab_size) or (batch_size, sequence_length, vocab_size),
            # it's likely logits. We need to take the argmax to get the predicted token IDs.
            preds = np.argmax(preds, axis=-1)
            print(f"DEBUG (compute_metrics): After argmax, New Shape of preds: {preds.shape}")
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Convert preds to a numpy array of integer type and remove any singleton dimensions
    try:
        preds = np.asarray(preds, dtype=np.int64) # Ensure it's a numpy array of integers
        preds = np.squeeze(preds) # Remove dimensions of size 1 (e.g., from (batch, seq_len, 1) to (batch, seq_len))
        print(f"DEBUG (compute_metrics): After squeeze, Final Shape of preds: {preds.shape}")
    except Exception as e:
        print(f"ERROR (compute_metrics): Failed to convert/squeeze preds array. Error: {e}")
        print(f"DEBUG (compute_metrics): Raw preds data: {preds[:5] if isinstance(preds, list) or isinstance(preds, np.ndarray) else preds}")
        raise # Re-raise the exception to see the original problem


    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-process the predictions and labels for BLEU calculation
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
    tokenizer=tokenizer, # Pass tokenizer to Trainer for data collator and other internal uses
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
# This will take time depending on dataset size, model size, and GPU.
# On Paperspace free tier (M4000), expect it to be slow for even small datasets.
print("\nStarting training... This might take a while.")
try:
    trainer.train()
    print("\nTraining complete!")
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    print("Common errors: Out of memory (OOM). Try reducing 'per_device_train_batch_size' or 'max_input_length'.")
    print("If 'cuda out of memory' try: import gc; gc.collect(); torch.cuda.empty_cache()")

# Save the fine-tuned model
# This will save to the 'output_dir' specified in TrainingArguments.
# The best model (based on eval_bleu) will be loaded at the end of training
# and saved to the output directory.
print(f"\nSaving the fine-tuned model to {output_dir}...")
trainer.save_model(output_dir)
# You can optionally also save the tokenizer with the model
tokenizer.save_pretrained(output_dir)
print("Model and tokenizer saved successfully!")

# --- Cell 9: Evaluation and Inference ---

print("\n--- Evaluation ---")
# Evaluate the model on the validation set
# Note: This will use the best model checkpoint saved during training
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
print(f"BLEU score on validation set: {eval_results.get('eval_bleu', 'N/A')}")

print("\n--- Inference Examples ---")
# Load the fine-tuned model for inference (optional, if you restarted kernel or want to verify load)
# If you are in the same session and ran trainer.train(), trainer.model is already the best model.
# model_for_inference = AutoModelForSeq2SeqLM.from_pretrained(output_dir).to(device)
# tokenizer_for_inference = AutoTokenizer.from_pretrained(output_dir)

def translate_hinglish_to_english(text, model_to_use, tokenizer_to_use):
    inputs = tokenizer_to_use(text, return_tensors="pt", max_length=max_input_length, truncation=True)
    # Move inputs to the same device as the model (GPU if available)
    inputs = {k: v.to(model_to_use.device) for k, v in inputs.items()}

    outputs = model_to_use.generate(
        **inputs,
        max_new_tokens=max_target_length,
        num_beams=5, # Use beam search for potentially better quality translations
        early_stopping=True, # Stop generation when all beam hypotheses have finished
        no_repeat_ngram_size=2 # Avoid repeating n-grams
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
    translated = translate_hinglish_to_english(sentence, model, tokenizer)
    print(f"Hinglish: {sentence}")
    print(f"English:   {translated}\n")

print("\n--- Next Steps ---")
print("1. **Replace the dummy dataset in 'Cell 3'** with your actual Hinglish-English parallel corpus. This is critical for a useful model.")
print("2. Adjust 'num_train_epochs' and 'per_device_train_batch_size' based on your dataset size and GPU memory.")
print("3. Monitor training progress using TensorBoard:")
print(f"   In your Paperspace JupyterLab terminal, run: `tensorboard --logdir {logging_dir}`")
print("   Then click on the provided URL (usually a localhost URL) in your browser or use Paperspace's 'Port' tab to forward that port.")
print("4. For better performance and larger models/datasets, consider upgrading to a paid Paperspace tier with a more powerful GPU.")
print("5. Explore different pre-trained models (e.g., mBART, mT5) that might be better suited for multilingual tasks, then fine-tune them.")
print("6. Implement more sophisticated data loading and preprocessing if your dataset is very large.")
print("\nProject setup complete! Happy translating!")