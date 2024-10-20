from sentiment_analysis.youtube_comments import YouTubeCommentsFetcher
from sentiment_analysis.sentiment_model import SentimentAnalyzer
from sentiment_analysis.utils import extract_video_id
from api_key import api_key

def main():
    
    fetcher = YouTubeCommentsFetcher(api_key)
    analyzer = SentimentAnalyzer()

    
    youtube_url = input("Enter the YouTube video URL: ")
    video_id = extract_video_id(youtube_url)

    if video_id:
        print("Fetching comments for video ID:", video_id)
        comments = fetcher.get_youtube_comments(video_id)

        
        sentiment_distribution = analyzer.get_sentiment_distribution(comments)

        
        analyzer.print_sentiment_distribution(sentiment_distribution)
    else:
        print("Invalid YouTube URL")

if __name__ == "__main__":
    main()
