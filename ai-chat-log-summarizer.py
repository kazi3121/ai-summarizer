import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

def parse_chat_log(file_path):
    user_messages, ai_messages, current_speaker, current_message = [], [], None, []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('User:'):
                    if current_message and current_speaker:
                        msg = ' '.join(current_message).strip()
                        if msg:
                            (user_messages if current_speaker == 'User' else ai_messages).append(msg)
                    current_speaker, current_message = 'User', [line[5:].strip()]
                elif line.startswith('AI:'):
                    if current_message and current_speaker:
                        msg = ' '.join(current_message).strip()
                        if msg:
                            (user_messages if current_speaker == 'User' else ai_messages).append(msg)
                    current_speaker, current_message = 'AI', [line[3:].strip()]
                elif current_speaker:
                    current_message.append(line)
            if current_message and current_speaker:
                msg = ' '.join(current_message).strip()
                if msg:
                    (user_messages if current_speaker == 'User' else ai_messages).append(msg)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    return user_messages, ai_messages

def get_message_statistics(user_messages, ai_messages):
    total_user, total_ai= len(user_messages), len(ai_messages)
    return total_user+total_ai, total_user, total_ai

