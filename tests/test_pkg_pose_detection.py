"""pyxpo-pose-detection: calls over a fake bridge, landmark cleaning, progress events, argument checks, manifest."""
import asyncio
import json
from pathlib import Path

import pytest

from pyxpo.build import autolink
from pyxpo.module_api import ModuleNotLinkedError, NotSupportedError, dispatch
from pyxpo.module_testing import FakeBridge

import pyxpo_pose_detection as PD

PKG = Path(PD.__file__).parent


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "me.jpg"
    p.write_bytes(b"\xff\xd8\xff")
    return str(p)


def test_detect_image(photo):
    raw = {"nose": [10, 20, 0.9], "left_wrist": [1.5, 2.5, 1.7], "right_knee": [3, 4, -1],
           "tail": [0, 0, 1], "left_ankle": [float("nan"), 1, 1], "right_ear": [1, 2], "left_eye": "x"}
    for os_name in ("ios", "android"):
        with FakeBridge(os_name) as fb:
            fb.on("PoseDetection", "detectImage", lambda uri: {"width": 640, "height": 480, "landmarks": raw})
            res = run(PD.detect_image_async(photo))
            assert fb.calls == [("PoseDetection", "detectImage", (photo,))]
        assert res == {"width": 640, "height": 480, "landmarks": {
            "nose": [10.0, 20.0, 0.9], "left_wrist": [1.5, 2.5, 1.0], "right_knee": [3.0, 4.0, 0.0]}}


def test_detect_image_nobody_and_file_url(photo):
    with FakeBridge("ios") as fb:
        fb.on("PoseDetection", "detectImage", lambda uri: {"width": 1, "height": 1, "landmarks": {}})
        assert run(PD.detect_image_async("file://" + photo))["landmarks"] == {}
        fb.on("PoseDetection", "detectImage", lambda uri: None)
        assert run(PD.detect_image_async(photo)) == {"width": 0, "height": 0, "landmarks": {}}


def test_detect_image_rejects_bad_uris(tmp_path):
    with FakeBridge("android") as fb:
        fb.on("PoseDetection", "detectImage", lambda uri: {})
        with pytest.raises(ValueError):
            run(PD.detect_image_async("https://example.com/a.jpg"))
        with pytest.raises(ValueError):
            run(PD.detect_image_async(""))
        with pytest.raises(FileNotFoundError):
            run(PD.detect_image_async(str(tmp_path / "none.jpg")))
        run(PD.detect_image_async("content://media/external/images/1"))   # resolved by the native side
        assert len(fb.calls) == 1


def test_detect_video_with_progress(photo):
    seen = []

    def detect(uri, opts):
        for p in (0.5, 1.0):
            dispatch({"name": "PoseDetection.progress", "data": {"id": opts["id"], "progress": p}})
        dispatch({"name": "PoseDetection.progress", "data": {"id": opts["id"] + 1000, "progress": 0.1}})  # another call
        return {"width": 360, "height": 640, "duration": 0.2, "frames": [
            {"t": 0.1, "landmarks": {"nose": [1, 2, 0.5]}}, {"t": 0.0, "landmarks": {}}, "junk"]}

    with FakeBridge("ios") as fb:
        fb.on("PoseDetection", "detectVideo", detect)
        res = run(PD.detect_video_async(photo, fps=10, max_duration=5, max_size=320, on_progress=seen.append))
        (mod, method, (uri, opts)), = fb.calls
        assert (mod, method, uri) == ("PoseDetection", "detectVideo", photo)
        assert opts["fps"] == 10.0 and opts["maxDuration"] == 5.0 and opts["maxSize"] == 320
    assert seen == [0.5, 1.0]
    assert res == {"width": 360, "height": 640, "duration": 0.2, "fps": 10.0, "frames": [
        {"t": 0.0, "landmarks": {}}, {"t": 0.1, "landmarks": {"nose": [1.0, 2.0, 0.5]}}]}
    assert PD._progress == {}                       # the callback is dropped once the call ends


def test_detect_video_without_callback_and_duration_fallback(photo):
    with FakeBridge("android") as fb:
        fb.on("PoseDetection", "detectVideo", lambda uri, o: {"frames": [{"t": 1.5, "landmarks": {}}]})
        res = run(PD.detect_video_async(photo))
        assert fb.calls[0][2][1]["fps"] == 15.0 and fb.calls[0][2][1]["maxSize"] == 640
    assert res["duration"] == 1.5 and res["width"] == 0


def test_progress_callback_dropped_on_error(photo):
    def fail(uri, o):
        raise RuntimeError("could not read any frame of the video")

    with FakeBridge("android") as fb:
        fb.on("PoseDetection", "detectVideo", fail)
        with pytest.raises(RuntimeError):
            run(PD.detect_video_async(photo, on_progress=print))
    assert PD._progress == {}


@pytest.mark.parametrize("kw", [{"fps": 0}, {"fps": 61}, {"max_duration": 0}, {"max_duration": 601},
                                {"max_size": 63}, {"max_size": 5000}])
def test_detect_video_bad_args(photo, kw):
    with FakeBridge("ios") as fb:
        fb.on("PoseDetection", "detectVideo", lambda uri, o: {})
        with pytest.raises(ValueError):
            run(PD.detect_video_async(photo, **kw))
        assert fb.calls == []


@pytest.mark.parametrize("os_name", ["web", "desktop"])
def test_no_detector_off_device(photo, os_name):
    with FakeBridge(os_name, modules=("PoseDetection",)) as fb:
        assert run(PD.is_available_async()) is False
        with pytest.raises(NotSupportedError):
            run(PD.detect_image_async(photo))
        with pytest.raises(NotSupportedError):
            run(PD.detect_video_async(photo))
        assert fb.calls == []


def test_is_available():
    with FakeBridge("ios") as fb:
        fb.on("PoseDetection", "available", lambda: True)
        assert run(PD.isAvailableAsync()) is True

        def broken():
            raise RuntimeError("no model")

        fb.on("PoseDetection", "available", broken)
        assert run(PD.is_available_async()) is False


def test_not_linked(photo):
    with FakeBridge("android"):
        assert run(PD.is_available_async()) is False
        with pytest.raises(ModuleNotLinkedError):
            run(PD.detect_image_async(photo))
        with pytest.raises(ModuleNotLinkedError):
            run(PD.detect_video_async(photo))


def test_best_frame_and_mean_score():
    clear = {n: [0, 0, 1.0] for n in PD.LANDMARKS}
    track = {"frames": [{"t": 0, "landmarks": {"nose": [0, 0, 1.0]}}, {"t": 1, "landmarks": clear}]}
    assert PD.mean_score(clear) == 1.0 and PD.mean_score({}) == 0
    assert PD.best_frame(track)["t"] == 1 and PD.best_frame({"frames": []}) is None
    assert len(PD.LANDMARKS) == 17 and PD.detectVideoAsync is PD.detect_video_async


def test_manifest_and_native_sources():
    meta = json.loads((PKG / "pyxpo-module.json").read_text())
    autolink.validate(meta, "pyxpo-module.json")
    assert meta["native"] == "PoseDetection" and "pose-detection" in autolink.OFFICIAL
    kt = (PKG / "android/src/main/java/dev/pyxpo/modules/posedetection/PoseDetectionModule.kt").read_text()
    swift = (PKG / "ios/PoseDetectionModule.swift").read_text()
    for src in (kt, swift):
        for name in PD.LANDMARKS:
            assert '"%s"' % name in src, name
        for method in ("available", "detectImage", "detectVideo", "PoseDetection.progress"):
            assert '"%s"' % method in src, method
