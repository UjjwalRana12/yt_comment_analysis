from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from youtube_comments import YouTubeCommentsFetcher
from sentiment_model import SentimentAnalyzer
from utils import extract_video_id
from api_key import api_key

app = FastAPI()

# Initialize fetcher and analyzer
fetcher = YouTubeCommentsFetcher(api_key)
analyzer = SentimentAnalyzer()

# Pydantic model for input
class YouTubeURLInput(BaseModel):
    youtube_url: str

@app.post("/analyze-comments")
async def analyze_comments(data: YouTubeURLInput):
    video_id = extract_video_id(data.youtube_url)

    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    try:
        # Fetch comments
        comments = fetcher.get_youtube_comments(video_id)
        if not comments:
            raise HTTPException(status_code=404, detail="No comments found for this video.")
        
        # Analyze sentiment
        sentiment_groups = analyzer.get_comments_by_sentiment(comments)

        return {
            "video_id": video_id,
            "comments_by_sentiment": sentiment_groups
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
