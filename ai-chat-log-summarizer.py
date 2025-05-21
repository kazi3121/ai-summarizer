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

def extract_keywords(messages, top_n=5, use_tfidf=True):
    stop_words = set(stopwords.words("english")).union({'like', 'yeah', 'hey', 'really', 'that', 'what'})
    all_text= ' '.join(messages).lower()
    
    if use_tfidf:
        try:
            vectorizer= TfidfVectorizer(stop_words=list(stop_words), max_features= top_n, token_pattern=r'(?u)\b\w{4,}\b')
            tfidf_matrix= vectorizer.fit_transform([all_text])
            keywords= [k for k, _ in sorted(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]), key= lambda x:x[1], reverse=True)[:top_n]]
            return keywords if keywords else['general', 'topics', 'detected', 'no', 'specific']
        except ValueError:
            print("Warning: Empty vocabulary in TF-IDF. Using word frequency.")

    words= [w for w in word_tokenize(all_text) if w.isalnum() and w not in stop_words and len(w)>=4]
    return [w for w, _ in Counter(words).most_common(top_n)]

def infer_conversation_nature(keywords):
    return f"The user asked mainly about {', '.join(k.capitalize() for k in keywords[:3])}." if keywords else "The conversation covered general topics."


def generate_summary(file_path, use_tfidf=True, output_file=None):
    user_msgs, ai_msgs = parse_chat_log(file_path)
    total, user_count, ai_count = get_message_statistics(user_msgs, ai_msgs)
    if not total:
        error_msg = "Error: No messages found in the chat log."
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(error_msg)
            print(f"Error message saved to {output_file}")
        return error_msg
    keywords = extract_keywords(user_msgs + ai_msgs, top_n=5, use_tfidf=use_tfidf)
    summary = (f"Summary:\n"
               f"- The conversation had {total} exchanges.\n"
               f"- User messages: {user_count}, AI messages: {ai_count}.\n"
               f"- {infer_conversation_nature(keywords)}\n"
               f"- Most common keywords: {', '.join(keywords)}.")
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary saved to {output_file}")
    return summary
