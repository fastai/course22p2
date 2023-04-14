from pyyoutube import Api
from youtube_transcript_downloader import get_transcript
import os

api = Api(api_key=os.environ['GOOGLE_API_KEY'])
playlist_id = "PLfYUBJiXbdtRUvTUYpLdfHHp9a58nWVXP"
items = api.get_playlist_items(playlist_id=playlist_id)

next_page_token = None
video_ids = []
while True:
    items = api.get_playlist_items(playlist_id=playlist_id, count=None, limit=50, page_token=next_page_token)
    video_ids += [item.snippet.resourceId.videoId for item in items.items]
    next_page_token = items.nextPageToken
    if not next_page_token: break

for video_id in video_ids:
    transcript = get_transcript(f'https://youtu.be/{video_id}')
    transcript = ' '.join(transcript.values()).replace(u'\xa0', u' ').replace('\n', ' ').replace('  ',' ')
    with open(video_id + ".txt", "w") as f: f.write(str(transcript))
