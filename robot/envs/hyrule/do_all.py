import os
for i in range(1000):
    os.system(f"xvfb-run python3 search.py --env_name scenes/{i}.json --output_path solution/{i}.json --video_path solution/{i}.avi")