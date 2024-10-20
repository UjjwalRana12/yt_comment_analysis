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

        # Group comments by sentiment score
        sentiment_groups = analyzer.get_comments_by_sentiment(comments)

        # Print comments by their sentiment scores (1 to 5)
        analyzer.print_comments_by_sentiment(sentiment_groups)
    else:
        print("Invalid YouTube URL")

if __name__ == "__main__":
    main()
