import os
from dotenv import load_dotenv
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    RagTokenizer, 
    RagRetriever, 
    RagSequenceForGeneration
)
import logging, json, spacy
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load Spacy model for POS and DEP tagging
nlp = spacy.load("en_core_web_sm")

# Function for POS and DEP tagging
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    dep_tags = [token.dep_ for token in doc]
    return tokens, pos_tags, dep_tags

# Load and preprocess the dataset
def load_and_preprocess_dataset(dataset_name="squad", split="train[:10%]"):
    dataset = load_dataset(dataset_name, split=split)
    preprocessed_data = []
    
    for entry in tqdm(dataset):
        question = entry['question']
        context = entry['context']
        
        # POS and DEP tagging
        context_tokens, context_pos, context_dep = preprocess_text(context)
        question_tokens, question_pos, question_dep = preprocess_text(question)
        
        # Tokenize and encode the inputs
        input_ids = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True).input_ids
        
        preprocessed_data.append({
            "input_ids": input_ids,
            "context_tokens": context_tokens,
            "context_pos": context_pos,
            "context_dep": context_dep,
            "question_tokens": question_tokens,
            "question_pos": question_pos,
            "question_dep": question_dep,
        })
    
    return Dataset.from_dict(preprocessed_data)

# Load JSON files and extract documents
def load_json_files(dataset_dir):
    json_files = [
        os.path.join(dataset_dir, os.getenv('DEP_MAPPING_FILE')),
        os.path.join(dataset_dir, os.getenv('POS_MAPPING_FILE')),
        os.path.join(dataset_dir, os.getenv('TEST_FILE')),
        os.path.join(dataset_dir, os.getenv('TRAIN_FILE')),
        os.path.join(dataset_dir, os.getenv('VAL_FILE'))
    ]

    documents = []
    for file in json_files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found. Skipping.")
            continue
        
        with open(file, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]

                for item in data:
                    if 'question' in item:
                        document = {
                            'text': item['question'],
                            'pos_tags': item.get('question_pos_tokens', []),
                            'dep_tags': item.get('question_dep_ids', [])
                        }
                        documents.append(document)
                    else:
                        print(f"Warning: No suitable field found in {file}. Skipping this item.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {file}. Skipping.")
    return documents

# Define paths to your datasets
dataset_dirs = [
    os.getenv('DATASET_DIR_1'), 
    os.getenv('DATASET_DIR_2'), 
    os.getenv('DATASET_DIR_3')
]
all_documents = []

# Loop through each dataset and load all the documents
for dataset_dir in dataset_dirs:
    documents = load_json_files(dataset_dir)
    all_documents.extend(documents)

print(f"Total documents loaded: {len(all_documents)}")

# Load a pre-trained RAG model and tokenizer
rag_tokenizer = RagTokenizer.from_pretrained(os.getenv('RAG_MODEL_NAME'))
rag_model = RagSequenceForGeneration.from_pretrained(os.getenv('RAG_MODEL_NAME'))

# Initialize the retriever with the combined documents
retriever = RagRetriever.from_pretrained(
    os.getenv('RAG_MODEL_NAME'),
    index_name="custom",
    passages=all_documents  
)

# Save the retriever for later use
retriever.save_pretrained("retriever")

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=os.getenv('OUTPUT_DIR'),
    evaluation_strategy=os.getenv('EVAL_STRATEGY'),
    learning_rate=float(os.getenv('LEARNING_RATE')),
    per_device_train_batch_size=int(os.getenv('TRAIN_BATCH_SIZE')),
    per_device_eval_batch_size=int(os.getenv('EVAL_BATCH_SIZE')),
    weight_decay=float(os.getenv('WEIGHT_DECAY')),
    save_total_limit=int(os.getenv('SAVE_TOTAL_LIMIT')),
    num_train_epochs=int(os.getenv('NUM_TRAIN_EPOCHS')),
    predict_with_generate=bool(os.getenv('PREDICT_WITH_GENERATE')),
)

# Load and preprocess the dataset
dataset = load_and_preprocess_dataset()

# Define the trainer
trainer = Seq2SeqTrainer(
    model=rag_model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

# Fine-tune the model
trainer.train()
