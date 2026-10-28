import cv2
import numpy as np
import pytest

from video_analytics import annotate, lane_segments, motion_boxes, process


def test_static_scene_has_no_motion_and_no_lane_crash():
    frame = np.zeros((180, 320, 3), np.uint8)
    assert motion_boxes(frame, frame) == []
    assert lane_segments(frame) == []


def test_moving_object_and_annotations_do_not_mutate_input():
    before = np.zeros((180, 320, 3), np.uint8)
    after = before.copy()
    after[60:120, 100:160] = 255
    original = after.copy()
    output, count = annotate(after, before, "privacy-blur", 100)
    assert count > 0
    assert np.array_equal(after, original)
    assert not np.array_equal(output, after)


def test_region_ignores_motion_outside_center():
    before = np.zeros((240, 400, 3), np.uint8)
    after = before.copy()
    after[5:30, 5:30] = 255
    assert annotate(after, before, "region", 20)[1] == 0


@pytest.mark.parametrize("mode", ["motion", "privacy-blur", "lanes", "region"])
def test_file_stream_finishes_at_eof(tmp_path, mode):
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    assert writer.isOpened()
    for offset in range(6):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[30:80, 20 + offset * 5 : 60 + offset * 5] = 255
        writer.write(frame)
    writer.release()
    output = tmp_path / "output.mp4"
    report = process(str(source), mode, str(output), min_area=20)
    assert report["frames"] == 6
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 6


def test_missing_source_reports_error(tmp_path):
    with pytest.raises(ValueError, match="Source"):
        process(str(tmp_path / "missing.mp4"), "motion")
