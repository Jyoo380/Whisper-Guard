from flask import Flask, jsonify, request, render_template
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from phe import paillier
import json
import re
import random
from math import gcd
from sympy import nextprime, mod_inverse

def generate_keypair(bit_length=1024):
    p = nextprime(random.getrandbits(bit_length // 2))
    q = nextprime(random.getrandbits(bit_length // 2))
    
    n = p * q
    phi = (p - 1) * (q - 1)
    
    g = n + 1
    while gcd(g, n**2) != 1:
        g += n
    
    lambda_ = phi // gcd(p - 1, q - 1)
    mu = mod_inverse((pow(g, lambda_, n**2) - 1) // n, n)
    
    public_key = (n, g)
    private_key = (lambda_, mu)
    
    return public_key, private_key

def generate_evaluation_key(public_key):
    n, _ = public_key
    r = random.randint(1, n)
    r_inv = mod_inverse(r, n)
    
    evaluation_key = (r, r_inv)
    
    return evaluation_key


app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Generate keys for homomorphic encryption
public_key, private_key = paillier.generate_paillier_keypair()

def string_to_int(message):
    int_message = int.from_bytes(message.encode(), 'big')
    return int_message

def int_to_string(int_message):
    string_message = int_message.to_bytes((int_message.bit_length() + 7) // 8, 'big').decode()
    return string_message

def encrypt_message(message):
    int_message = string_to_int(message)
    encrypted_message = public_key.encrypt(int_message)
    return encrypted_message

def decrypt_message(encrypted_message):
    decrypted_int_message = private_key.decrypt(encrypted_message)
    decrypted_message = int_to_string(decrypted_int_message)
    return decrypted_message

def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    if not isinstance(start_logits, torch.Tensor) or not isinstance(end_logits, torch.Tensor):
        print(f"start_logits type: {type(start_logits)}, value: {start_logits}")
        print(f"end_logits type: {type(end_logits)}, value: {end_logits}")
        return "Error: start_logits or end_logits is not a tensor"

    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    
    answer_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer, answer_tokens

def calculate_accuracy(answer_tokens, context, question):
    # Tokenize the context and question
    context_sentences = re.split(r'[.!?]', context.strip())
    question_tokens = tokenizer.tokenize(question.lower())
    
    # Identify the relevant part of the context
    relevant_sentence = None
    for sentence in context_sentences:
        if all(token.lower() in sentence.lower() for token in answer_tokens):
            relevant_sentence = sentence.strip()
            break
    
    # If no relevant sentence found, return 0 accuracy
    if not relevant_sentence:
        return 0
    
    # Tokenize the relevant sentence
    relevant_tokens = tokenizer.tokenize(relevant_sentence.lower())
    
    # Remove special characters from tokens
    answer_tokens_clean = [re.sub(r'[^\w\s]', '', token) for token in answer_tokens]
    relevant_tokens_clean = [re.sub(r'[^\w\s]', '', token.lower()) for token in relevant_tokens]
    
    # Calculate overlap
    overlap = len(set(answer_tokens_clean) & set(relevant_tokens_clean))
    
    # Calculate accuracy
    accuracy = (overlap / len(set(relevant_tokens_clean))) * 100 if relevant_tokens_clean else 0
    return accuracy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/Chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_question = request.form['user_question']
        result = ask(user_question)
        return render_template('Webpage.html', answer=result['answer'], accuracy=result['accuracy'])
    return render_template('Webpage.html', answer="", accuracy=0)

def ask(user_question):
    encrypted_question = encrypt_message(user_question)
    
    context = """
        Cloud computing is the delivery of various services through the Internet. These services include servers, storage, databases, networking, software, analytics, and intelligence. Cloud computing allows companies to access and store data in third-party data centers. This eliminates the need for owning and maintaining physical servers and other infrastructure.
        
        There are different types of cloud computing services, including Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). IaaS provides virtualized computing resources over the internet, PaaS offers a platform allowing customers to develop, run, and manage applications without dealing with the infrastructure, and SaaS delivers software applications over the internet on a subscription basis.
        
        Cloud computing offers several advantages such as cost-efficiency, scalability, flexibility, and accessibility. Companies can scale their resources based on demand, reducing costs associated with maintaining physical hardware. It also allows businesses to access their data and applications from anywhere with an internet connection, promoting collaboration and remote work.
        
        However, cloud computing also comes with challenges such as security concerns, data privacy issues, and dependency on internet connectivity. Companies need to implement proper security measures and compliance protocols to protect their data and ensure privacy.
        
        Major cloud service providers include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform (GCP). These platforms offer a wide range of cloud computing services and tools to support businesses in their digital transformation journey.
        """
    
    decrypted_question = decrypt_message(encrypted_question)
    answer, answer_tokens = get_answer(decrypted_question, context)
    
    accuracy = calculate_accuracy(answer_tokens, context, decrypted_question)
    
    return {'answer': answer, 'accuracy': accuracy}

if __name__ == '__main__':
    app.run(debug=True)
