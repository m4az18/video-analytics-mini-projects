"""Classical video mini projects: motion, motion blur, lanes, and a region alarm."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def motion_boxes(previous, current, min_area=1000, threshold=30):
    """Compare untouched frames so annotations never become apparent movement."""
    delta = cv2.absdiff(previous, current)
    gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.dilate(mask, None, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def lane_segments(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    edges[: int(edges.shape[0] * 0.6)] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)
    return [] if lines is None else [tuple(map(int, line[0])) for line in lines]


def annotate(frame, previous, mode, min_area):
    result = frame.copy()
    if mode == "lanes":
        lines = lane_segments(frame)
        for x1, y1, x2, y2 in lines:
            cv2.line(result, (x1, y1), (x2, y2), (0, 200, 120), 2)
        return result, len(lines)
    if previous is None:
        return result, 0
    boxes = motion_boxes(previous, frame, min_area)
    if mode == "region":
        # A normalized central region remains usable across input resolutions.
        height, width = frame.shape[:2]
        rx, ry, rw, rh = int(width * 0.25), int(height * 0.25), int(width * 0.5), int(height * 0.5)
        boxes = [
            (x, y, w, h) for x, y, w, h in boxes if x < rx + rw and x + w > rx and y < ry + rh and y + h > ry
        ]
        cv2.rectangle(result, (rx, ry), (rx + rw, ry + rh), (0, 0, 255) if boxes else (0, 200, 0), 2)
    for x, y, width, height in boxes:
        if mode == "privacy-blur":
            crop = result[y : y + height, x : x + width]
            result[y : y + height, x : x + width] = cv2.blur(crop, (51, 51))
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 180, 255), 2)
    return result, len(boxes)


def process(source, mode, output=None, show=False, min_area=1000):
    if not str(source).isdecimal() and not Path(source).is_file():
        raise ValueError("Source must be a local video path or an explicit camera index")
    if output and Path(output).exists():
        raise ValueError("Output exists; choose a new filename")
    capture = cv2.VideoCapture(int(source) if str(source).isdecimal() else str(source))
    writer = None
    if not capture.isOpened():
        capture.release()
        raise ValueError("Could not open video source")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 25.0
    previous, frames, events = None, 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if previous is not None and previous.shape != frame.shape:
                previous = None
            rendered, count = annotate(frame, previous, mode, min_area)
            previous = frame
            frames += 1
            events += bool(count)
            if output:
                if writer is None:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                    height, width = rendered.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise OSError("Could not initialize output writer")
                writer.write(rendered)
            if show:
                cv2.imshow(mode, rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    if not frames:
        raise ValueError("Input has no decodable frames")
    return {"mode": mode, "frames": frames, "frames_with_candidates": events, "output_fps": fps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["motion", "privacy-blur", "lanes", "region"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", help="Optional local .mp4 file; no audio is retained")
    parser.add_argument("--min-area", type=float, default=1000)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.min_area) or args.min_area <= 0:
        parser.error("min-area must be finite and positive")
    if args.output and Path(args.output).suffix.lower() != ".mp4":
        parser.error("output must use the .mp4 extension")
    try:
        print(json.dumps(process(args.source, args.mode, args.output, args.show, args.min_area), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
