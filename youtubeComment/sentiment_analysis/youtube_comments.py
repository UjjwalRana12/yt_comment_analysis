import googleapiclient.discovery

class YouTubeCommentsFetcher:
    def __init__(self, api_key):
        
        self.youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    def get_youtube_comments(self, video_id):
        """Fetches comments from a YouTube video using the video ID."""
        comments = []
        request = self.youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        response = request.execute()

        while response:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            if 'nextPageToken' in response:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100
                )
                response = request.execute()
            else:
                break

        return comments
