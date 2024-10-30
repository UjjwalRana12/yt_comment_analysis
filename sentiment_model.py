import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

    def analyze_sentiment(self, comment):
        tokens = self.tokenizer.encode(comment, return_tensors='pt', truncation=True, padding=True)
        result = self.model(tokens)
        sentiment_score = int(torch.argmax(result.logits))  # Adjusted for binary sentiment
        return sentiment_score


    def get_comments_by_sentiment(self, comments):
        # """Groups comments by sentiment score (1 to 5)."""
        sentiment_groups = defaultdict(list)

        for comment in comments:
            sentiment = self.analyze_sentiment(comment)
            sentiment_groups[sentiment].append(comment)

        return sentiment_groups

    def print_comments_by_sentiment(self, sentiment_groups):
        # """Prints the comments grouped by their sentiment score."""
        for score in range(1, 6):
            print(f"\nComments with sentiment {score}:")
            for comment in sentiment_groups[score]:
                print(f"- {comment}")
