import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    def analyze_sentiment(self, comment):
        # Returns the sentiment score of a given comment.
        tokens = self.tokenizer.encode(comment, return_tensors='pt', truncation=True, padding=True)
        result = self.model(tokens)
        sentiment_score = int(torch.argmax(result.logits)) + 1  # Sentiment is between 1 and 5
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
