from transformers import pipeline

# Load the pre-trained question-answering model
qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Provide the large dataset as the context
with open("IndianMythology.txt", "r") as file:
    context = file.read()

while True:
    # Accept user input for the question
    question = input("Ask a question (type 'exit' to quit): ")

    # Check if the user wants to exit
    if question.lower() == "exit":
        break

    # Use the model to answer the question
    answer = qa_model(question=question, context=context)

    # Print the answer
    print("Answer:", answer["answer"])
