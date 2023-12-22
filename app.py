import streamlit as st
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import RobertaTokenizer, RobertaForQuestionAnswering

import torch

# Load the BERT model for question answering
bert_qa = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Load the T5 model and tokenizer
# model_name = "t5-small"
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = T5Tokenizer.from_pretrained(model_name)

# Streamlit UI
st.title("Mythology QA System")

# Model selection
model_option = st.selectbox("Choose a model", ("BERT", "RoBERTa"))

# User input
user_question = st.text_input("Enter your question")

def load_model_and_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
    model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
    return model, tokenizer


# Load context from file
@st.cache_data
def load_context():
    with open("IndianMythology.txt", "r") as file:
        return file.read()

context = load_context()

# Function to get answer using BERT
def get_answer_bert(question, context):
    return bert_qa(question=question, context=context)["answer"]

def answer_question(question, context, model, tokenizer):
    # Tokenize the question and context separately
    question_tokens = tokenizer(question, add_special_tokens=False)["input_ids"]
    context_tokens = tokenizer(context, add_special_tokens=False)["input_ids"]

    # Define the maximum sequence length for RoBERTa
    max_seq_length = model.config.max_position_embeddings

    # Initialize variables to store the best answer and its confidence score
    best_answer = ""
    best_confidence = float('-inf')

    # Split the context into chunks that fit within the maximum sequence length
    for i in range(0, len(context_tokens), max_seq_length - len(question_tokens) - 2):
        # Combine the question tokens and the context chunk
        input_ids = question_tokens + context_tokens[i:i+max_seq_length-len(question_tokens)-2]

        # Perform QA
        inputs = torch.tensor(input_ids).unsqueeze(0)
        with torch.no_grad():
            outputs = model(inputs)
            start_scores, end_scores = outputs.start_logits, outputs.end_logits

        # Find the answer span in this chunk
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        # Convert token IDs to answer text
        answer = tokenizer.decode(input_ids[answer_start:answer_end+1])

        # Calculate the confidence score
        confidence = start_scores[0][answer_start] + end_scores[0][answer_end]

        # Update the best answer if the current answer has higher confidence
        if confidence > best_confidence:
            best_answer = answer
            best_confidence = confidence

    return best_answer 

# Function to get answer using T5
# def get_answer_t5(question, context):
#     input_text = "answer: " + question + " context: " + context
#     input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
#     output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# Answering the question
if st.button("Get Answer"):
    if model_option == "BERT":
        answer = get_answer_bert(user_question, context)
    elif model_option == "RoBERTa":
        model, tokenizer = load_model_and_tokenizer()
        answer = answer_question(user_question, context, model, tokenizer)
    else:
        answer = "Invalid model selection"

    st.text("Answer: " + answer)
