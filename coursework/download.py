# import requests

# video_url = "https://elearning.iirs.gov.in/courses/IIRS_NCERT_Remote_Sensing/story_content/video_5ik2NqpNJoo_22_48_1000x650.mp4"
# video_file = "video.mp4"

# with requests.get(video_url, stream=True, verify=False) as r:  # <- disable SSL verify
#     r.raise_for_status()
#     with open(video_file, "wb") as f:
#         for chunk in r.iter_content(chunk_size=8192):
#             f.write(chunk)

# import subprocess
# subprocess.run(["ffmpeg", "-i", r"coursework\video.mp4", "-q:a", "0", "-map", "a", "audio.mp3"])

