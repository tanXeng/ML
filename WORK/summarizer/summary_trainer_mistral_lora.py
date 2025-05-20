from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from sklearn.model_selection import train_test_split
import pandas as pd
from trl import SFTTrainer
import logging
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model, PeftModel
import torch
from datasets import Dataset
import os


EXCEL_FILE = "/home/dsta/saic/datasets/A/A_2024.xlsx"
MODEL_PATH = "/home/dsta/pipeline-b/backend/models/sumgen/mistral_7b_model"
TOKENIZER_PATH = "/home/dsta/pipeline-b/backend/models/sumgen/mistral_7b_tokenizer"
OUTPUT_DIR = "./models/A_Coy/mistral_with_lora_fine_tuned"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()

class SummaryDataset(Dataset):
    # Initialize the dataset with a tokenizer, data, and maximum token length
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer  # Tokenizer for encoding text
        self.data = data            # Data containing dialogues and summaries
        self.max_length = max_length # Maximum length of tokens

    # Return the number of items in the dataset
    def __len__(self):
        return len(self.data)

    # Retrieve an item from the dataset by index
    def __getitem__(self, idx):
        item = self.data.iloc[idx]  # Get the row at the specified index
        article_content = item[0] # Extract dialogue from the row
        summary = item[2]   # Extract summary from the row

        if pd.isna(article_content) or pd.isna(summary):
            raise ValueError(f"Found NaN at index {idx} in article_content or summary")
        
        article_content = str(article_content)
        summary = str(summary)

        # Encode the dialogue as input data for the model
        source = self.tokenizer.encode_plus(
            article_content, 
            max_length=self.max_length, 
            padding='max_length', 
            return_tensors='pt', 
            truncation=True
        )

        # Encode the summary as target data for the model
        target = self.tokenizer.encode_plus(
            summary, 
            max_length=self.max_length, 
            padding='max_length', 
            return_tensors='pt', 
            truncation=True
        )

        # Return a dictionary containing input_ids, attention_mask, labels, and the original summary text
        return {
            'input_ids': source['input_ids'].flatten(),
            'attention_mask': source['attention_mask'].flatten(),
            'labels': target['input_ids'].flatten(),
            'summary': summary 
        }
    
def print_trainable_parameters(model):
    """
        Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable (%): {100 * trainable_params / all_param}")

# Set tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    start_prompt = "Summarize the following article:\n"
    end_prompt = '\n\nSummary: '
    
    # Create input prompts
    inputs = [start_prompt + str(text) + end_prompt for text in examples["article_content"]]    
    
    # Tokenize inputs
    input_encoding = tokenizer(
        inputs, 
        padding="max_length", 
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    
    # Tokenize summaries (labels)
    label_encoding = tokenizer(
        examples["human_summary"],
        padding="max_length",
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    )
    
    # Combine encodings
    examples["input_ids"] = input_encoding["input_ids"]
    examples["attention_mask"] = input_encoding["attention_mask"]
    examples["labels"] = label_encoding["input_ids"]
    
    return examples

# Load your Excel data
df = pd.read_excel(EXCEL_FILE, engine='openpyxl', header=None)
df[[0, 2]] = df[[0, 2]].replace([r'^\s*$', 'nan', '', None, 'NaN', 'NAN'], pd.NA, regex=True)
df = df.dropna(subset=[0, 2])
# df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
df_train = df
df_train.columns = ['article_content', 'article_title', 'human_summary']

dataset_train = Dataset.from_pandas(df_train[['article_content', 'human_summary']])
tokenized_dataset = dataset_train.map(
    tokenize_function, 
    batched=True,
    remove_columns=["article_content", "human_summary"]
)

config = LoraConfig( # weight is scaled with (alpha / rank)
    r=32, # Sets the rank of the LoRA adaptation, which controls the size of low-rank updates.
    lora_alpha=16, # Scaling factor for the LoRA weights.
    bias="none", 
    lora_dropout=0.1, 
    target_modules=[ # Specifies the layers to which LoRA will be applied (e.g., projection layers in the attention and feedforward network).
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj", "lm_head"
    ],
    task_type="CAUSAL_LM"
)

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Set model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map="auto")
model = model.to(device)

model = get_peft_model(model, config)
# print_trainable_parameters(model)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# for param in model.parameters():
#     param.requires_grad = True


training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR, 
    num_train_epochs=3, 
    per_device_train_batch_size=1, # batch size for training is 1 per GPU
    gradient_accumulation_steps=2, # gradients will be accumulated over 2 steps
    gradient_checkpointing = True, 
    save_steps=500, # save model every {save_steps} steps
    # logging_steps=50, 
    max_steps=-1, # -1 indicates "no step limit"
    optim="paged_adamw_32bit", # Uses the 32-bit Paged AdamW optimizer, an optimized version of the AdamW optimizer, suited for training in lower precision
    learning_rate=2e-4, 
    weight_decay=0.001, 
    fp16=False, bf16=True, # Training is done in bfloat16 precision (bf16=True), which is more stable than FP16 in terms of numerical accuracy but still provides the speed and memory efficiency advantages. 
    max_grad_norm=0.3, # limit grad's norm to 0.3 to prevent exploding gradient
    warmup_ratio=0.03, #  Specifies a warmup period where the learning rate gradually increases during the first {warmup_ratio}% of training
    group_by_length=True, 
    lr_scheduler_type="cosine" # Uses a cosine learning rate scheduler, which reduces the learning rate following a cosine curve after the warmup phase.
)


# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model, 
    train_dataset=tokenized_dataset, 
    peft_config=config, 
    tokenizer=tokenizer, 
    args=training_arguments, 
    )



# Starting the training process
trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
