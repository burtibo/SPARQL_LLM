# Example questions for testing
questions = [
    "What is the capital of France?",
    "Who is the president of the United States?",
]

for question in questions:
    # Tokenize the input question
    input_ids = rag_tokenizer(question, return_tensors="pt").input_ids.to(device)

    # Generate the answer using the RAG model
    generated = rag_model.generate(input_ids=input_ids, max_length=50, num_beams=5)
    output = rag_tokenizer.decode(generated[0], skip_special_tokens=True)

    # Print the results
    print(f"Question: {question}")
    print(f"Generated Answer: {output}\n")

    # Retrieve documents (optional)
    retrieved_docs = retriever.retrieve(input_ids=input_ids)
    print(f"Retrieved Documents: {retrieved_docs}\n")
