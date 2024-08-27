from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
import torch

# Function to evaluate the model
def evaluate_model(model, tokenizer, dataset, device):
    model.eval()
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    exact_matches = 0
    total = len(dataset)

    for data in tqdm(dataset, desc="Evaluating"):
        input_ids = data['input_ids'].to(device)
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, max_length=50, num_beams=5)
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        reference = data['question']

        # Compare generated output with reference
        bleu.add(prediction=output.split(), reference=[reference.split()])
        rouge.add(prediction=output, reference=reference)

        # Exact match calculation
        if output.strip() == reference.strip():
            exact_matches += 1

    bleu_score = bleu.compute()
    rouge_score = rouge.compute()
    exact_match_score = exact_matches / total

    print(f"BLEU Score: {bleu_score['bleu']}")
    print(f"ROUGE Score: {rouge_score}")
    print(f"Exact Match Score: {exact_match_score}")

# Load the dataset and tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your validation/test dataset here
validation_dataset = load_and_preprocess_dataset(dataset_name="squad", split="validation[:10%]")

# Evaluate the model
evaluate_model(rag_model, tokenizer, validation_dataset, device)
