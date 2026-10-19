# pyxpo-pose-detection

On-device human pose detection for [pyxpo](https://github.com/pyxpo/pyxpo) apps, written in Python. It finds 17 body
landmarks (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) in a photo, or in every sampled frame of
a video. Android runs Google ML Kit pose detection and iOS runs Apple Vision. Both work offline, and the picture or
video never leaves the phone.

<p align="center">
  <img src="https://raw.githubusercontent.com/pyxpo/pyxpo/main/packages/pose-detection/docs/images/hero.png" width="820" alt="Doodle Puppet reading a dance video on Android, the stick-figure filmstrip it found, and a drawing dancing those moves">
</p>

```python
import pyxpo_pose_detection as PoseDetection

if await PoseDetection.is_available_async():
    pose = await PoseDetection.detect_image_async(photo_uri)       # {"width", "height", "landmarks"}
    track = await PoseDetection.detect_video_async(video_uri, fps=15, on_progress=print)
    for frame in track["frames"]:
        print(frame["t"], frame["landmarks"].get("left_wrist"))    # [x, y, score] or None
```

## Features

- **Photos and videos:**
  - `detect_image_async(uri)` finds the most prominent person in a photo.
  - `detect_video_async(uri)` samples a video at a frame rate you choose and finds the person in each frame, with
    a progress callback.
- **The same 17 landmarks everywhere:** COCO names and order, in pixels of the upright picture (x to the right,
  y down), each with a 0..1 score. Android and iOS answer in the same shape, so app code has no platform branches.
- **Private and offline:**
  - Android bundles the ML Kit model in the app, so nothing is downloaded at run time.
  - iOS uses the Vision framework that ships with the system.
  - No network calls, no account, no API key.
- **Easy to get right:**
  - EXIF orientation is applied, so phone photos come out upright.
  - Landmarks the detector did not find are left out, never guessed.
  - Arguments are checked in Python before any native work starts.
  - `is_available_async()` answers False on the web and the desktop, so an app can offer something else there.
- **Helpers:** `mean_score(landmarks)` says how clearly a body was seen, and `best_frame(track)` picks the clearest
  frame of a video.
- **Typed:** ships `py.typed`. Expo-style camelCase names (`detectImageAsync`, ...) work too.

## Contents

| If you are... | Start here |
| --- | --- |
| New to pose detection | [Quick start](#quick-start) → [Landmarks](#landmarks) |
| Reading a photo | [Photos](#photos) |
| Reading a video | [Videos](#videos) → [Speed](#speed) |
| Seeing it in a real app | [Recipe: copy a dance into stick figures](#recipe-copy-a-dance-into-stick-figures) |
| Shipping to the stores | [Production checklist](#production-checklist) |
| Looking for exact signatures | [API reference](#api-reference) · [Errors](#errors) |
| Stuck | [Testing](#testing) · [Troubleshooting](#troubleshooting) · [FAQ](#faq) |

## Platform support

| Target | Pose detection |
| --- | --- |
| Android, API 24+ | Photos and videos. Google ML Kit pose detection (`com.google.mlkit:pose-detection`, base model, bundled). |
| iOS 15+ | Photos and videos on an iPhone or iPad. Apple Vision (`VNDetectHumanBodyPoseRequest`). Recent iOS simulators cannot run this Vision request, so `is_available_async()` returns False there; test on a real device. |
| Web page and desktop window (`pyxpo run web` / `pyxpo run desktop`), Mac Catalyst, Windows | None. `is_available_async()` returns False and the detect calls raise `NotSupportedError`. |

Requires Python 3.9+ and `pyxpo>=0.1.0a1`.

## Install

```sh
pip install pyxpo-pose-detection
```

Add the module to your app's `pyproject.toml`:

```toml
[tool.pyxpo]
modules = ["pose-detection", "image-picker"]   # image-picker to let people choose or record a photo or video
```

The package has no config plugin and asks for no permissions itself. It reads files your app already has; the
camera and photo library permissions belong to whatever gets the file (for example `pyxpo-image-picker`).

The module has native code, so rebuild the app after adding it (`pyxpo run android` or `pyxpo run ios`). The first
Android build downloads ML Kit through Gradle.

## Quick start

A screen that lets the user choose a photo and says how many body landmarks it found:

```python
import pyxpo_pose_detection as PoseDetection
from pyxpo import Pressable, Text, View, component, use_async, use_state
from pyxpo_image_picker import ImagePicker


@component
def PoseCheck():
    available = use_async(PoseDetection.is_available_async, [])
    message, set_message = use_state("Choose a photo of a person")

    async def choose():
        res = await ImagePicker.launchImageLibraryAsync(mediaTypes="images")
        if res.get("canceled") or not res.get("assets"):
            return
        set_message("Looking...")
        pose = await PoseDetection.detect_image_async(res["assets"][0]["uri"])
        found = pose["landmarks"]
        if not found:
            set_message("No person found")
            return
        clear = PoseDetection.mean_score(found)
        set_message("Found %d of 17 landmarks (clarity %.0f%%)" % (len(found), clear * 100))

    if available.loading:
        return Text("Getting ready...")
    if not available.value:
        return Text("Pose detection needs the Android or iPhone app.")
    return View(
        Text(message),
        Pressable(Text("Choose a photo"), on_press=choose),
        style={"flex": 1, "alignItems": "center", "justifyContent": "center", "gap": 16},
    )
```

## Landmarks

Every result reports landmarks as a dict from name to `[x, y, score]`:

- `x`, `y` are pixels in the upright picture. `x` grows to the right, `y` grows down. For a video they are pixels
  of the scaled frame (see `max_size`), and `width` / `height` in the result say how big that frame was.
- `score` is 0..1. On Android it is ML Kit's in-frame likelihood; on iOS it is Vision's confidence. Treat it as
  "how sure", not as an exact probability, and do not compare scores across platforms.
- A landmark the detector did not find is **left out**. Always use `landmarks.get(name)`.

`PoseDetection.LANDMARKS` lists the 17 names in COCO order:

| # | Name | # | Name | # | Name |
| --- | --- | --- | --- | --- | --- |
| 0 | `nose` | 6 | `right_shoulder` | 12 | `right_hip` |
| 1 | `left_eye` | 7 | `left_elbow` | 13 | `left_knee` |
| 2 | `right_eye` | 8 | `right_elbow` | 14 | `right_knee` |
| 3 | `left_ear` | 9 | `left_wrist` | 15 | `left_ankle` |
| 4 | `right_ear` | 10 | `right_wrist` | 16 | `right_ankle` |
| 5 | `left_shoulder` | 11 | `left_hip` | | |

"Left" and "right" are the person's own sides. In a selfie video the person's left hand is on the right of the
picture. Swap the sides yourself if you want a mirror image.

> **Tip:** to draw a skeleton, connect these pairs: shoulders, hips, each shoulder to its hip, shoulder → elbow →
> wrist, and hip → knee → ankle on both sides.

## Photos

```python
pose = await PoseDetection.detect_image_async(uri)
# {"width": 3024, "height": 4032, "landmarks": {"nose": [1490.2, 980.7, 0.99], ...}}
```

| Argument | Type | Description |
| --- | --- | --- |
| `uri` | `str` | A local file path, a `file://` URL, or an Android `content://` URI (what the image picker returns). |

It returns the most prominent person: the one ML Kit tracks on Android, the most confident one on iOS. `landmarks`
is empty when nobody was found. `width` and `height` are the upright size, after EXIF orientation.

## Videos

```python
track = await PoseDetection.detect_video_async(uri, fps=15, max_duration=30, max_size=640,
                                               on_progress=lambda p: print("%d%%" % (p * 100)))
# {"width": 360, "height": 640, "duration": 14.9, "fps": 15.0,
#  "frames": [{"t": 0.0, "landmarks": {...}}, {"t": 0.0667, "landmarks": {...}}, ...]}
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `uri` | `str` | | A local file path, a `file://` URL, or an Android `content://` URI. |
| `fps` | `float` | `15` | Frames read per second of video, 1 to 60. |
| `max_duration` | `float` | `30` | Seconds read from the start, more than 0 and at most 600. Longer videos are cut there. |
| `max_size` | `int` | `640` | Frames are scaled so the longest side is at most this many pixels before detection, 64 to 4096. Smaller is faster. |
| `on_progress` | `callable` | `None` | Called with 0..1 while it works, on the app's event loop. Safe to pass a state setter. |

`frames` is sorted by `t` (seconds from the start). A frame where nobody was found has empty `landmarks`; it is
kept so the timing stays regular. `duration` is the length that was actually read.

Pick the clearest frame, for example for a thumbnail:

```python
frame = PoseDetection.best_frame(track)          # None when there are no frames
if frame is not None:
    print(frame["t"], PoseDetection.mean_score(frame["landmarks"]))
```

### Speed

Detection runs on a background thread, so the app stays responsive, but it is not instant. The time grows with the
number of frames (`fps` × seconds), so a 15 second clip at 15 fps is 225 detections. Show progress and tell people
it can take a minute.

On Android each frame is fetched from the video by time. Videos from a phone camera have a key frame about every
second and read quickly. Videos with very few key frames (some screen recordings and web downloads) can take
several minutes, because each fetch decodes from the last key frame. Lower `fps` or `max_duration` for those, or
re-encode them.

## Errors

| Error | When | What to do |
| --- | --- | --- |
| `NotSupportedError` | A detect call on the web, the desktop, Mac Catalyst or Windows. | Check `is_available_async()` first and offer something else. |
| `ValueError` | `uri` is empty or not a string, is an `http(s)://` URL, or an argument is out of range. | Download remote files first; keep `fps`, `max_duration` and `max_size` in range. |
| `FileNotFoundError` | A local path or `file://` URL that does not exist. | Check the path. A picker result is a real file until the app deletes it. |
| `ModuleNotLinkedError` (from `pyxpo.module_api`) | The app was built without the module. | Add `"pose-detection"` to `[tool.pyxpo] modules` and rebuild. |
| Any other exception | The file could not be decoded (not an image or video, unsupported codec), or the detector failed. | Show a short message and let the user pick another file. |

`is_available_async()` never raises: it returns False when the module is missing or the detector cannot start.

## Web and desktop

There is no detector on the web or the desktop. `is_available_async()` returns False, and `detect_image_async` /
`detect_video_async` raise `NotSupportedError`. Argument errors are still raised first, so a bad `fps` shows up in
the browser too. Apps usually show a demo instead: Doodle Puppet offers built-in demo moves there.

## Recipe: copy a dance into stick figures

This is the core of the Doodle Puppet "My moves" screen: the user picks a dance video, the app reads it with a
progress bar, and shows five stick figures from across the video.

```python
import pyxpo_pose_detection as PoseDetection
from pyxpo import Pressable, StyleSheet, Text, View, component, use_state
from pyxpo.vector_icons import Path
from pyxpo_image_picker import ImagePicker

BONES = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle")]


def stick_figure(landmarks, width, height, min_score=0.3):
    """An SVG path in a 100 x 100 box for the bones whose two ends were seen clearly."""
    scale = 100.0 / max(width, height, 1)
    seen = {n: v for n, v in landmarks.items() if v[2] >= min_score}
    parts = []
    for a, b in BONES:
        if a in seen and b in seen:
            parts.append("M%.1f %.1f L%.1f %.1f" % (seen[a][0] * scale, seen[a][1] * scale,
                                                    seen[b][0] * scale, seen[b][1] * scale))
    return " ".join(parts)


@component
def DanceReader():
    progress, set_progress = use_state(None)       # None: not reading
    figures, set_figures = use_state([])
    error, set_error = use_state(None)

    async def choose():
        res = await ImagePicker.launchImageLibraryAsync(mediaTypes="videos")
        if res.get("canceled") or not res.get("assets"):
            return
        set_error(None)
        set_progress(0.0)
        try:
            track = await PoseDetection.detect_video_async(res["assets"][0]["uri"], fps=10,
                                                           on_progress=set_progress)
        except Exception as e:  # noqa: BLE001 - shown to the user
            set_error("Could not read this video (%s)" % (str(e) or type(e).__name__))
            return
        finally:
            set_progress(None)
        frames = [f for f in track["frames"] if PoseDetection.mean_score(f["landmarks"]) > 0.5]
        if not frames:
            set_error("Nobody was seen clearly. Stand back so the camera sees you from head to feet.")
            return
        picks = [frames[i * (len(frames) - 1) // 4] for i in range(5)]
        set_figures([stick_figure(f["landmarks"], track["width"], track["height"]) for f in picks])

    if progress is not None:
        return View(Text("Reading the video... %d%%" % (progress * 100)), style=styles.screen)
    return View(
        View(*[Path(d=d, stroke="#221B33", stroke_width=5, viewbox=100, style=styles.figure, key=str(i))
               for i, d in enumerate(figures)], style=styles.strip),
        Text(error or "Choose a video of someone dancing"),
        Pressable(Text("Choose a video"), on_press=choose),
        style=styles.screen,
    )


styles = StyleSheet.create({
    "screen": {"flex": 1, "alignItems": "center", "justifyContent": "center", "gap": 16, "padding": 24},
    "strip": {"flexDirection": "row", "gap": 4},
    "figure": {"width": 64, "height": 64},
})
```

Doodle Puppet goes further: it fills short gaps where a joint was missed, turns the track into a joint motion,
mirrors it if asked, and makes a drawing dance it. It keeps only the landmark track, never the video.

## Production checklist

- Check `is_available_async()` before showing pose features, and give the web and desktop builds a fallback.
- Show progress for videos and keep `fps` and `max_duration` as low as your feature allows. 10 to 15 fps is enough
  for body motion.
- Tell people how to film: the whole body in the picture, good light, one person.
- Check `mean_score` (or the scores of the joints you need) before trusting a result, and explain what to fix when
  it is low.
- Delete copies of picked videos you no longer need. This package does not copy or keep anything.
- Say in your privacy policy (and in the App Store / Play data safety forms) that photos and videos are processed
  on the device. No data is sent anywhere by this package.
- Test on a real Android phone and a real iPhone. The Android emulator is much slower, and the iOS simulator has no
  detector at all.

## Testing

**On a phone or emulator**, record or choose a short video of one person dancing, full body in the picture. On the
Android emulator you can push a file to the gallery with
`adb push dance.mp4 /sdcard/Movies/ && adb shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Movies/dance.mp4`.
The iOS simulator cannot run the detector (see [Troubleshooting](#troubleshooting)), so test iOS on a real device.

**Unit tests** run without a device. The module is a set of plain async functions, so swap in fakes with
`monkeypatch`:

```python
import pyxpo_pose_detection as PoseDetection

STANDING = {name: [100.0 + i, 50.0 + 20 * i, 0.9] for i, name in enumerate(PoseDetection.LANDMARKS)}


async def fake_available():
    return True


async def fake_video(uri, *, fps=15, max_duration=30, max_size=640, on_progress=None):
    if on_progress is not None:
        on_progress(1.0)
    frames = [{"t": i / fps, "landmarks": dict(STANDING)} for i in range(int(fps * 2))]
    return {"width": 360, "height": 640, "duration": 2.0, "fps": float(fps), "frames": frames}


def test_reads_a_dance(monkeypatch):
    monkeypatch.setattr(PoseDetection, "is_available_async", fake_available)
    monkeypatch.setattr(PoseDetection, "detect_video_async", fake_video)
    ...
```

`mean_score`, `best_frame` and `LANDMARKS` are pure Python and can be used in tests as they are.

## Troubleshooting

| Problem | Cause and fix |
| --- | --- |
| `landmarks` is empty for a photo with a person | The body is too small, cut off, or turned away. Use a photo where the person fills a good part of the picture. Drawings and cartoons are often not seen as people. |
| Hands or feet are missing in many frames | They leave the picture or blur in fast moves. Film from further away, in better light. Fill short gaps by interpolating between frames. |
| Video reading takes several minutes on Android | The video has very few key frames (see [Speed](#speed)). Lower `fps` / `max_duration`, or use a video recorded with the phone's camera. |
| Coordinates do not match the original video size | Video frames are scaled to `max_size`. Use the result's `width` and `height`, not the video's own size. |
| Left and right look swapped | Landmarks use the person's own sides. Mirror them yourself for selfie-style display. |
| `ValueError: pose detection reads local files only` | You passed an `http(s)://` URL. Download it with `pyxpo-file-system` first. |
| `is_available_async()` is False on the iOS simulator | Recent simulators fail every Vision body pose request (`Code=9 "Unable to setup request"`), even on the CPU. The module checks this once and reports no detector. Use a real iPhone or iPad. |
| `NotSupportedError` in the browser | There is no detector on the web or the desktop. Check `is_available_async()`. |
| `ModuleNotLinkedError` | The build does not include the module. Add it to `[tool.pyxpo] modules` and rebuild. |
| The Android build hangs downloading ML Kit | Gradle cannot reach `dl.google.com`. Check the network; if IPv4 to Google is blocked, `JAVA_TOOL_OPTIONS=-Djava.net.preferIPv6Addresses=true` can help. |
| `RuntimeWarning: coroutine ... was never awaited` | The check was started with `use_effect(lambda: coro())`. Use `use_async(PoseDetection.is_available_async, [])`. |

## FAQ

**Does the photo or video leave the phone?**
No. Both detectors run on the device, and the package makes no network calls.

**What if there are several people?**
You get one person: the most prominent one on Android, the most confident one on iOS. Multi-person results are not
supported.

**Is there a live camera mode?**
Not yet. Record a video and read it, or take photos.

**Are there 3D coordinates, hand or face landmarks?**
No. This package reports the 17 COCO body landmarks in 2D, the set both platforms support.

**Why ML Kit's base model on Android?**
It is bundled and fast. ML Kit's model is marked beta by Google; the API in this package will stay the same if the
model changes.

## API reference

**Functions**

| Name | Description |
| --- | --- |
| `is_available_async() -> bool` | True on Android and iOS when the module is linked and the detector can run (False on the iOS simulator). Never raises. |
| `detect_image_async(uri) -> dict` | `{"width", "height", "landmarks"}` for the most prominent person in a photo. |
| `detect_video_async(uri, *, fps=15, max_duration=30, max_size=640, on_progress=None) -> dict` | `{"width", "height", "duration", "fps", "frames": [{"t", "landmarks"}]}`. |
| `mean_score(landmarks) -> float` | Average score over all 17 landmarks, missing ones counted as 0. |
| `best_frame(track) -> dict or None` | The frame of a video result with the highest `mean_score`. |
| `isAvailableAsync`, `detectImageAsync`, `detectVideoAsync` | Expo-style names for the functions above. |

**Constants and types**

| Name | Description |
| --- | --- |
| `LANDMARKS` | The 17 landmark names, in COCO order. |
| `NotSupportedError` | Raised by the detect calls where there is no detector (re-exported from `pyxpo.module_api`). |

`landmarks` is always `{name: [x, y, score]}` with names from `LANDMARKS`.

## License

MIT. The hero image shows Doodle Puppet reading a dance video; the stick figures were found in
["Man playing Dance Dance revolution"](https://commons.wikimedia.org/wiki/File:Man_playing_Dance_Dance_revolution.webm)
from Wikimedia Commons (CC BY 3.0). ML Kit is a Google product and Vision is an Apple framework; this package is not
made or endorsed by Google or Apple, and using ML Kit means agreeing to Google's ML Kit terms.
