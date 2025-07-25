import torch
import os
# Download the video
url = 'https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4'

import imageio.v3 as iio
frames = iio.imread(url, plugin="FFMPEG")  # plugin="pyav"

device = 'cuda'
grid_size = 10

video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
print('input video shape:', video.shape)

# Run Offline CoTracker:
cotracker = torch.hub.load(os.path.join(os.path.expanduser("~"), ".cache/torch/hub/facebookresearch_co-tracker_main/"), "cotracker3_offline", source="local").to(device)
#cotracker = torch.hub.load(os.path.join(os.path.expanduser("~"), ".cache/torch/hub/facebookresearch_co-tracker_main/"), "cotracker2", source="local")
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1
print('pred_tracks.shape: ', pred_tracks.shape)