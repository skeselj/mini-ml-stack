"""
Pipeline that detects people in videos.


Example usage
-------------

1. With GPU.

PROFILE_RESOURCES=1 python person_detection.py


2. Without GPU.

CUDA_VISIBLE_DEVICES="" PROFILE_RESOURCES=1 python person_detection.py

"""

import itertools
import logging
import os
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from resource_profiler import profiler
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

YOLO_PERSON_CLASS = 0  # COCO class index for "person".
_EXHAUSTED = object()  # Sentinel for an exhausted iterator.

YOLO_WEIGHTS = os.environ.get(
    "YOLO_WEIGHTS",
    str(Path(__file__).resolve().parent.parent / "models" / "yolo26n.pt"),
)


def _timed_iter(iterable: Iterable, phase: str) -> Iterator:
    """
    Yield from `iterable`, charging time to `phase`.
    """

    iterator = iter(iterable)

    while True:
        with profiler.phase(phase):
            item = next(iterator, _EXHAUSTED)

        if item is _EXHAUSTED:
            return

        yield item


def get_bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def get_output_path(input_path: str, batch_idx: int) -> str:
    path = Path(input_path)
    return str(path.parent / f"{path.stem}_output_{batch_idx:05d}{path.suffix}")


def get_merged_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.parent / f"{path.stem}_output{path.suffix}")


@cache
def load_yolo_model(warmup: bool = True) -> YOLO:
    """
    Return the YOLO model, moved onto its inference device.
    """

    model = YOLO(YOLO_WEIGHTS)

    if warmup:
        with profiler.phase("warmup", on_gpu=True) as span:
            model(
                [np.zeros((1080, 1920, 3), dtype=np.uint8)],
                classes=[YOLO_PERSON_CLASS],
                imgsz=960,
                verbose=False,
            )
        if profiler.enabled:
            logger.info(f"loaded model; profile: {span.summary()}")

    return model


def get_video_fps(video_path: str) -> float:
    with cv2.VideoCapture(video_path) as capture:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}.")

        return capture.get(cv2.CAP_PROP_FPS)


def get_frame_batches(
    video_path: str, batch_size: int
) -> Iterator[tuple[np.ndarray, ...]]:
    """
    Yield the frames of `video_path` as batched NumPy arrays.

    Yields
    ------
        batch: batch of up to `batch_size` frames, as BGR arrays.
    """

    with cv2.VideoCapture(video_path) as capture:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}.")

        batch: list[np.ndarray] = []
        frame_count = 0

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            batch.append(frame)
            frame_count += 1
            if len(batch) == batch_size:
                yield tuple(batch)
                batch = []

        if batch:
            yield tuple(batch)

        if frame_count == 0:
            raise ValueError(f"No frames in {video_path}.")


def detect_people(
    frames: tuple[np.ndarray, ...],
    conf: float = 0.33,
    top_k: int = 1,
) -> list[list[dict]]:
    """
    Run a YOLO model on `frames` and return the person detections.
    """

    model = load_yolo_model()

    with profiler.phase("detect", on_gpu=True) as span:
        results = model(
            list(frames),
            classes=[YOLO_PERSON_CLASS],
            conf=conf,
            imgsz=960,
            verbose=False,
        )
    if profiler.enabled:
        logger.info(
            f"ran detection on {len(frames)} frames, "
            f"{span.wall_s * 1e3 / len(frames):.1f} ms/frame; "
            f"profile: {span.summary()}"
        )

    detections = []

    for frame_result in results:
        frame_detections = [
            {"bbox": box.xyxy[0].tolist(), "confidence": float(box.conf[0])}
            for box in frame_result.boxes
        ]
        frame_detections.sort(
            key=lambda d: get_bbox_area(d["bbox"]), reverse=True
        )

        detections.append(frame_detections[:top_k])

    return detections


def render_detections(
    frame: np.ndarray,
    detections: list[dict],
    color: tuple[int] = (255, 180, 111),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Return a copy of `frame` that has `detections` rendered on top of it.

    Parameters
    ----------
        frame: BGR array.
    """

    rendered = frame.copy()

    height, width = rendered.shape[:2]
    color_ary = np.array(color, dtype=np.float64)

    for detection in detections:
        x1, y1, x2, y2 = (round(coord) for coord in detection["bbox"])

        x1, x2 = max(x1, 0), min(x2, width)
        y1, y2 = max(y1, 0), min(y2, height)
        if x1 >= x2 or y1 >= y2:
            logger.warning(
                "Encountered invalid detection (x1, x2, y1, y2) = "
                f"({x1}, {x2}, {y1}, {y2})."
            )
            continue

        box = rendered[y1:y2, x1:x2]
        blended = box * (1 - alpha) + color_ary * alpha
        rendered[y1:y2, x1:x2] = blended.round().astype(np.uint8)

    return rendered


def detect_in_batches(
    frame_batches: Iterable[tuple[np.ndarray, ...]],
) -> Iterator[list[dict]]:
    """
    Yield the detections for each frame of `frame_batches`.

    Detections are small enough to hand between pipeline stages, which frames
    are not.
    """

    for frames in _timed_iter(frame_batches, "decode"):
        yield from detect_people(frames)


def render_frames(
    frame_batches: Iterable[tuple[np.ndarray, ...]],
    detections: Iterable[list[dict]],
) -> Iterator[np.ndarray]:
    """
    Render `detections` onto the frames of `frame_batches`, in order.

    The counterpart to `detect_in_batches`: whatever decoded the frames for
    detection is long gone by now, so they are decoded a second time.
    """

    frames = (
        frame
        for batch in _timed_iter(frame_batches, "decode")
        for frame in batch
    )

    for frame, frame_detections in zip(frames, detections, strict=True):
        with profiler.phase("render"):
            yield render_detections(frame, frame_detections)


def write_video(
    frames: Iterable[np.ndarray], output_path: str, fps: float
) -> None:
    """
    Write `frames` to `output_path` as an MP4.

    Parameters
    ----------
        frames: BGR arrays.
    """

    frames = iter(frames)
    first_frame = next(frames, None)
    if first_frame is None:
        raise ValueError(f"No frames to write to {output_path}.")

    height, width = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {output_path} for writing.")

    written = 0
    try:
        with profiler.phase("encode"):
            writer.write(first_frame)
        written += 1

        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame {written} is {frame.shape[:2]}, expected {(height, width)}."
                )
            with profiler.phase("encode"):
                writer.write(frame)
            written += 1
    finally:
        writer.release()

    logger.info(f"Wrote {written:,} frame(s) to {output_path}")


def concatenate_videos(video_paths: list[str], output_path: str) -> None:
    if not video_paths:
        raise ValueError(f"No videos to concatenate into {output_path}.")

    ffmpeg_file_list = "".join(
        "file '{}'\n".format(str(Path(path).resolve()).replace("'", "'\\''"))
        for path in video_paths
    )

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False
    ) as list_file:
        list_file.write(ffmpeg_file_list)
        list_path = list_file.name

    try:
        # fmt: off
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                output_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # fmt: on
    finally:
        Path(list_path).unlink()

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to concatenate into {output_path}:\n{result.stderr[-2000:]}"
        )

    logger.info(
        f"Concatenated {len(video_paths):,} chunk(s) into {output_path}"
    )


def detection_pipeline(
    video_path: str,
    output_fps: float | None = None,
    video_chunk_size_s: int = 30,
    detection_batch_size: int = 64,
) -> None:
    """
    Do person detection on `video_path` and render output to file.
    """

    if output_fps is None:
        output_fps = get_video_fps(video_path)

    video_chunk_size_frames = round(video_chunk_size_s * output_fps)
    batches_per_chunk = max(1, video_chunk_size_frames // detection_batch_size)

    batches = get_frame_batches(video_path, batch_size=detection_batch_size)

    chunk_paths = []

    for chunk_idx in itertools.count():
        chunk_batches = itertools.islice(batches, batches_per_chunk)

        with profiler.phase("decode"):
            first_batch = next(chunk_batches, None)
        if first_batch is None:
            break

        with profiler.phase("decode"):
            chunk = [first_batch, *chunk_batches]

        detections = list(detect_in_batches(chunk))

        output_path = get_output_path(
            input_path=video_path, batch_idx=chunk_idx
        )
        write_video(
            render_frames(chunk, detections), output_path, fps=output_fps
        )

        chunk_paths.append(output_path)

    concatenate_videos(chunk_paths, get_merged_output_path(video_path))

    for chunk_path in chunk_paths:
        Path(chunk_path).unlink()

    logger.info(f"Removed {len(chunk_paths):,} intermediate chunk(s)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    detection_pipeline("../data/sports/womens_marathon_record_2023.mp4")
