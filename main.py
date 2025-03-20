import numpy as np
import pickle
from tensorflow.keras.models import load_model
import ssl
from flask import jsonify, Flask, request
from flask_cors import CORS
import nltk
import json
import random
from datetime import datetime
import logging
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


try:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    lemmatizer = nltk.WordNetLemmatizer()
    model = load_model('models/chatbot_model.h5')
    words = pickle.load(open('models/words.pkl', 'rb'))
    classes = pickle.load(open('models/classes.pkl', 'rb'))

    with open('data.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)

    logger.info("Đã tải xong mô hình và dữ liệu chatbot.")
except Exception as e:
    logger.error(f"Lỗi khi tải mô hình và dữ liệu: {str(e)}")
    raise
def clean_up_sentence(sentence):
    """Tiền xử lý câu đầu vào"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    """Chuyển đổi câu thành bag of words"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Dự đoán intent của câu đầu vào"""
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list):
    """Lấy câu trả lời dựa trên intent dự đoán được"""
    if not intents_list:
        return "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể diễn đạt lại không?"
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Xin lỗi, tôi không thể trả lời câu hỏi này."

def log_conversation(user_message, bot_response, intent=None):
    """Ghi log cuộc hội thoại"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | User: {user_message} | Bot: {bot_response} | Intent: {intent}\n"
    with open('conversation_logs.txt', 'a', encoding='utf-8') as f:
        f.write(log_entry)
        
        
        
        
@app.route('/')
def home():
    return "Welcome to Flask Chatbot API"

@app.route('/api/get_response', methods=['POST'])
def get_bot_response():
    """API xử lý tin nhắn từ ReactJS"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"response": "Vui lòng nhập tin nhắn"}), 400
        intents_list = predict_class(user_message)
        response = get_response(intents_list)
        intent = intents_list[0]['intent'] if intents_list else 'unknown'
        log_conversation(user_message, response, intent)
        return jsonify({"response": response, "intent": intent}), 200
    except Exception as e:
        logger.error(f"Lỗi khi xử lý tin nhắn: {str(e)}")
        return jsonify({"response": "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau."}), 500

if __name__ == '__main__':
    app.run(debug=True)
