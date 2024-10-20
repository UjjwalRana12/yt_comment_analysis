import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

class SentimentAnalyzer:
    def __init__(self):
       
        self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    def analyze_sentiment(self, comment):
       
        tokens = self.tokenizer.encode(comment, return_tensors='pt', truncation=True, padding=True)
        result = self.model(tokens)
        sentiment_score = int(torch.argmax(result.logits)) + 1  # Sentiment score from 1 to 5
        return sentiment_score
    
    def get_sentiment_distribution(self, comments):
       
        sentiment_count = defaultdict(int)

        for comment in comments:
            sentiment = self.analyze_sentiment(comment)
            sentiment_count[sentiment] += 1

        return sentiment_count
    
    def print_sentiment_distribution(self, sentiment_count):
       
        print("\nSentiment Analysis Results:")
        for score in range(1, 6):
            print(f"Comments with sentiment {score}: {sentiment_count[score]}")
