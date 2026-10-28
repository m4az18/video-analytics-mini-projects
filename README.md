# Video analytics mini projects

Four small OpenCV applications built from the supplied motion-monitoring,
privacy-blur, lane-detection and restricted-region prototypes. Each runs on
an explicit local video source and can export annotated video without a GUI.

| Mini project | Method | Main limitation |
| --- | --- | --- |
| Motion events | Frame difference, threshold, dilation and contour area | Camera movement also triggers events |
| Motion privacy blur | Blur regions with detected movement | Stationary people can remain visible |
| Lane candidates | Canny edges and probabilistic Hough lines | Road markings are not semantically verified |
| Region monitoring | Motion intersecting a central region | Fixed region and pixel-area threshold |

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements.txt
python video_analytics.py motion --source data/example.mp4 --show
python video_analytics.py privacy-blur --source data/example.mp4 --output outputs/blurred.mp4
python video_analytics.py lanes --source data/road.mp4 --show
python video_analytics.py region --source data/example.mp4 --min-area 500
```

Pass `--source 0` explicitly to use a camera. `--show` opens a desktop
preview; press **q** to exit. Input media and outputs remain local and are
ignored by Git. Export contains video frames only and drops audio.

## Engineering details

Raw consecutive frames are compared before annotation, preventing boxes and
text from feeding back into motion detection. EOF is checked before image
processing. Capture and writer objects are released on exceptions and early
exit. Missing Hough lines produce an empty result, and output overwrite is refused.

## Scope

These are heuristic demonstrations. Motion blur does not guarantee anonymity;
the lane project is unsuitable for driving decisions. No cat or object identity
classifier is present: motion events report movement only. The original raw
footage, personal imagery and course exercises are excluded.

## Checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Tests use generated video and arrays to verify static scenes, moving regions,
non-mutating annotations, empty lane results, region filtering and complete
video export at end-of-file. No external footage or model downloads are needed.
