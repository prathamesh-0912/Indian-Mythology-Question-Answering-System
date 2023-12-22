from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch

def load_model_and_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
    model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
    return model, tokenizer

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

    return best_answer  # Return the best answer based on confidence score

def read_context_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Example usage
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()

    context = read_context_from_file("IndianMythology.txt")  # Replace with your context file path
    question = input("Ask a question (type 'exit' to quit): ")
    
    if question.lower() == "exit":
        print("Come Back ......")
    else:
        answer = answer_question(question, context, model, tokenizer)
        print(f"Answer: {answer}")
