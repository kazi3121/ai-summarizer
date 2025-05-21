AI Chat Log Summarizer

Overview

It's is a Python-based tool that processes .txt chat logs between a user and an AI, parsing messages, counting exchanges, extracting keywords, and generating a summary. It uses TF-IDF for keyword extraction and dynamically infers conversation topics, making it suitable for diverse topics like cybersecurity or travel without manual keyword updates.


Features:

Parses Chat Logs: Separates User and AI messages from .txt files formatted with User: and AI: prefixes.

Message Statistics: Counts total exchanges and messages per speaker.

Keyword Extraction: Identifies top 5 keywords using TF-IDF, excluding stop words and prioritizing words with 4+ characters.

Dynamic Topic Inference: Describes conversation topics based on top keywords (e.g., "Cybersecurity, Antarctica, Data").

Summary Generation: Outputs a summary with exchange count, message counts, topic, and keywords, saved to a .txt file.

Multi-File Support: Summarizes multiple chat logs in a folder.


Prerequisites:

Python 3.6+
Libraries: nltk, scikit-learn
Install dependencies: pip install nltk scikit-learn