import re

def extract_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return None
