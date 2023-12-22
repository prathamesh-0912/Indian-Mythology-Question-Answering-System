from transformers import pipeline, BartForQuestionAnswering, BartTokenizer

# Load the pre-trained BART question-answering model and tokenizer
model_name = "facebook/bart-large-cnn"
qa_model = pipeline("question-answering", model=BartForQuestionAnswering.from_pretrained(model_name), tokenizer=BartTokenizer.from_pretrained(model_name))

# Provide the large dataset as the context
with open("IndianMythology.txt", "r") as file:
    context = file.read()

while True:
    # Accept user input for the question
    question = input("Ask a question (type 'exit' to quit): ")

    # Check if the user wants to exit
    if question.lower() == "exit":
        break

    # Use the pre-trained model to answer the question
    answer = qa_model(question=question, context=context)

    # Print the answer
    print("Answer:", answer["answer"])
