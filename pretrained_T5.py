from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pre-trained T5 model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Provide the large dataset as the context
with open("IndianMythology.txt", "r") as file:
    context = file.read()

while True:
    # Accept user input for the question
    question = input("Ask a question (type 'exit' to quit): ")

    # Check if the user wants to exit
    if question.lower() == "exit":
        break

    # Encode the input
    input_text = "answer: " + question + " context: " + context

    # Tokenize and generate the answer
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

    # Decode and print the answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Answer:", answer)