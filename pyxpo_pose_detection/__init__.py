"""On-device human pose detection for pyxpo apps: 17 body landmarks from a photo or from every frame of a video.

    import pyxpo_pose_detection as PoseDetection

    if await PoseDetection.is_available_async():
        pose = await PoseDetection.detect_image_async(photo_uri)       # or detectImageAsync
        track = await PoseDetection.detect_video_async(video_uri, fps=15, on_progress=print)

Android runs Google ML Kit pose detection (bundled model, no download, works offline); iOS runs Apple Vision
(`VNDetectHumanBodyPoseRequest`). Both report the same 17 COCO landmarks in pixels of the upright picture
(x to the right, y down) with a 0..1 score each. A landmark the detector did not find is left out.
The web and the desktop have no detector: `is_available_async()` answers False there and the detect calls
raise `NotSupportedError`, so an app can offer something else.
"""
from __future__ import annotations

import itertools
import os
from typing import Any, Callable, Dict, List, Optional

from pyxpo.module_api import NotSupportedError, call as _call, on_event, platform, require_linked

__version__ = "0.1.0a1"

__all__ = ["LANDMARKS", "NotSupportedError", "is_available_async", "detect_image_async", "detect_video_async",
           "isAvailableAsync", "detectImageAsync", "detectVideoAsync", "best_frame", "mean_score"]

# COCO keypoint order, the one most pose tools use.
LANDMARKS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
             "right_knee", "left_ankle", "right_ankle")

Pose = Dict[str, Any]
_EVENT = "PoseDetection.progress"
_ids = itertools.count(1)
_progress: Dict[int, Callable[[float], Any]] = {}
_subscribed = [False]


def _native() -> bool:
    return platform() in ("ios", "android")


def call(method: str, *args: Any) -> Any:
    require_linked("PoseDetection", "pose-detection", package="pyxpo-pose-detection")
    return _call("PoseDetection", method, *args, reply=True)


def _path(uri: Any) -> str:
    if not isinstance(uri, str) or not uri:
        raise ValueError("uri must be a file path or a file:// URL, got %r" % (uri,))
    return uri


def _check_local(uri: str) -> None:
    if uri.startswith(("http://", "https://")):
        raise ValueError("pose detection reads local files only; download %s first" % uri)
    path = uri[7:] if uri.startswith("file://") else uri
    if not uri.startswith("content://") and not os.path.isfile(path):
        raise FileNotFoundError("no file %s" % path)


def _landmarks(raw: Any) -> Dict[str, List[float]]:
    """Keeps the 17 known names with finite [x, y, score] values; drops anything else."""
    out: Dict[str, List[float]] = {}
    if not isinstance(raw, dict):
        return out
    for name in LANDMARKS:
        v = raw.get(name)
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            continue
        try:
            x, y, s = float(v[0]), float(v[1]), float(v[2])
        except (TypeError, ValueError):
            continue
        if x == x and y == y and s == s:          # not NaN
            out[name] = [x, y, max(0.0, min(1.0, s))]
    return out


def _on_progress(data: Any) -> None:
    fn = _progress.get(int((data or {}).get("id", 0)))
    if fn is not None:
        fn(float(data.get("progress", 0)))


async def is_available_async() -> bool:
    """True on Android and iOS when this module is linked into the build and the detector can run; False on the web,
    the desktop and the iOS simulator (recent simulators cannot run Vision's body pose request)."""
    if not _native():
        return False
    try:
        return bool(await call("available"))
    except Exception:  # noqa: BLE001 - not linked, or the detector failed to start
        return False


async def detect_image_async(uri: str) -> Pose:
    """Finds the most prominent person in a photo.

    Returns `{"width", "height", "landmarks": {name: [x, y, score]}}`; `landmarks` is empty when nobody was found.
    The photo's EXIF orientation is applied first, so the numbers match the picture as people see it."""
    uri = _path(uri)
    if not _native():
        raise NotSupportedError("pose detection needs Android or iOS (this is %s)" % platform())
    _check_local(uri)
    res = await call("detectImage", uri) or {}
    return {"width": int(res.get("width", 0)), "height": int(res.get("height", 0)),
            "landmarks": _landmarks(res.get("landmarks"))}


async def detect_video_async(uri: str, *, fps: float = 15, max_duration: float = 30, max_size: int = 640,
                             on_progress: Optional[Callable[[float], Any]] = None) -> Pose:
    """Samples the video `fps` times a second (up to `max_duration` seconds) and finds the person in each frame.

    Frames are scaled so their longest side is at most `max_size` pixels before detection (faster; the
    coordinates are in that scaled frame). `on_progress(p)` gets 0 to 1 while it works.

    Returns `{"width", "height", "duration", "fps", "frames": [{"t", "landmarks"}]}`, where `t` is seconds from
    the start and `duration` is the length that was read. Frames where nobody was found have empty landmarks."""
    uri = _path(uri)
    fps, max_duration, max_size = float(fps), float(max_duration), int(max_size)
    if not 1 <= fps <= 60:
        raise ValueError("fps must be between 1 and 60, got %r" % fps)
    if not 0 < max_duration <= 600:
        raise ValueError("max_duration must be between 0 and 600 seconds, got %r" % max_duration)
    if not 64 <= max_size <= 4096:
        raise ValueError("max_size must be between 64 and 4096, got %r" % max_size)
    if not _native():
        raise NotSupportedError("pose detection needs Android or iOS (this is %s)" % platform())
    _check_local(uri)
    require_linked("PoseDetection", "pose-detection", package="pyxpo-pose-detection")
    if not _subscribed[0]:
        on_event(_EVENT, _on_progress)
        _subscribed[0] = True
    rid = next(_ids)
    if on_progress is not None:
        _progress[rid] = on_progress
    try:
        res = await call("detectVideo", uri, {"id": rid, "fps": fps, "maxDuration": max_duration,
                                              "maxSize": max_size}) or {}
    finally:
        _progress.pop(rid, None)
    frames = []
    for f in res.get("frames") or []:
        if isinstance(f, dict):
            frames.append({"t": float(f.get("t", 0)), "landmarks": _landmarks(f.get("landmarks"))})
    frames.sort(key=lambda f: f["t"])
    return {"width": int(res.get("width", 0)), "height": int(res.get("height", 0)),
            "duration": float(res.get("duration", frames[-1]["t"] if frames else 0)), "fps": fps, "frames": frames}


def mean_score(landmarks: Dict[str, List[float]]) -> float:
    """Average score over all 17 landmarks, counting missing ones as 0. A rough "how clearly was the body seen"."""
    return sum(v[2] for v in landmarks.values()) / len(LANDMARKS)


def best_frame(track: Pose) -> Optional[Dict[str, Any]]:
    """The frame of a `detect_video_async` result whose body was seen most clearly, or None when there are none."""
    frames = track.get("frames") or []
    return max(frames, key=lambda f: mean_score(f["landmarks"])) if frames else None


# Expo-style names.
isAvailableAsync = is_available_async
detectImageAsync = detect_image_async
detectVideoAsync = detect_video_async
