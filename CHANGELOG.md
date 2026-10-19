# Changelog

## 0.1.0a1

- First release: `detect_image_async` finds the 17 COCO body landmarks of the most prominent person in a photo,
  and `detect_video_async` does it for frames sampled from a video (`fps`, `max_duration`, `max_size`, a progress
  callback). Android runs ML Kit pose detection (bundled model, offline); iOS runs Vision
  (`VNDetectHumanBodyPoseRequest`). `is_available_async` answers False on the web, the desktop and the iOS simulator (where Vision cannot run
  the request).
  `mean_score` and `best_frame` help pick a clear frame.
