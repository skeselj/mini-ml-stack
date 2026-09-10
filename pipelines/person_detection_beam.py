"""
Pipeline that detects people in videos. Uses Beam.


Example usage
-------------

1. DirectRunner (local, no cluster).

python person_detection_beam.py ../data/sports/womens_marathon_record_2023.mp4


2. SparkRunner with a local master.

python person_detection_beam.py ../data/sports/womens_marathon_record_2023.mp4 \
    --runner=SparkRunner \
    --spark_master_url='local[*]' \
    --environment_type=LOOPBACK


3. Spark on the kind (Kubernetes in Docker) cluster.

python person_detection_beam.py ../data/sports/womens_marathon_record_2023.mp4 \
    --video-chunk-size-s 10 \
    --runner=PortableRunner \
    --worker-root /mnt/project \
    --job_endpoint=localhost:8099 \
    --artifact_endpoint=localhost:8098 \
    --environment_type=EXTERNAL \
    --environment_config=localhost:50000

"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from person_detection import (
    concatenate_videos,
    detect_in_batches,
    get_frame_batches,
    get_video_fps,
    load_yolo_model,
    render_frames,
    write_video,
)

logger = logging.getLogger(__name__)

HOST_ROOT = Path(__file__).resolve().parent.parent
WORKER_ROOT = Path("/mnt/project")


def to_worker_path(path: str | Path, worker_root: Path | None) -> str:
    """
    Rewrite a host path so a worker can open it.
    """

    if worker_root is None:
        return str(path)

    return str(worker_root / Path(path).resolve().relative_to(HOST_ROOT))


def split_video(
    video_path: str, chunk_dir: Path, chunk_size_s: int
) -> list[tuple[int, str]]:
    """
    Split `video_path` into independently decodable chunks under `chunk_dir`.

    Returns
    -------
        chunks: (index, path) pairs, ordered by index.
    """

    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)

    # fmt: off
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-i", video_path,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(chunk_size_s),
            "-reset_timestamps", "1",
            str(chunk_dir / "chunk_%05d.mp4"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # fmt: on
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to split {video_path}:\n{result.stderr[-2000:]}"
        )

    chunks = [
        (idx, str(path))
        for idx, path in enumerate(sorted(chunk_dir.glob("chunk_*.mp4")))
    ]
    if not chunks:
        raise ValueError(f"No chunks produced from {video_path}.")

    logger.info(f"Split {video_path} into {len(chunks):,} chunk(s)")

    return chunks


class DetectChunk(beam.DoFn):
    """
    Detect people in one chunk.
    """

    def __init__(self, detection_batch_size: int = 32):
        self._detection_batch_size = detection_batch_size

    def setup(self) -> None:
        load_yolo_model()

    def process(self, element: tuple[int, str]):
        chunk_idx, chunk_path = element

        batches = get_frame_batches(
            chunk_path, batch_size=self._detection_batch_size
        )

        yield (chunk_idx, chunk_path, list(detect_in_batches(batches)))


class RenderChunk(beam.DoFn):
    """
    Render detections onto one chunk.
    """

    def __init__(self, detection_batch_size: int = 32):
        self._detection_batch_size = detection_batch_size

    def process(self, element: tuple[int, str, list[list[dict]]]):
        chunk_idx, chunk_path, detections = element

        path = Path(chunk_path)
        output_path = str(path.parent / f"{path.stem}_output{path.suffix}")

        batches = get_frame_batches(
            chunk_path, batch_size=self._detection_batch_size
        )
        write_video(
            render_frames(batches, detections),
            output_path,
            fps=get_video_fps(chunk_path),
        )

        yield (chunk_idx, output_path)


def concatenate_in_order(
    indexed_paths: list[tuple[int, str]], output_path: str
) -> str:
    paths = [path for _, path in sorted(indexed_paths)]
    concatenate_videos(paths, output_path)
    return output_path


def run(
    video_path: str,
    video_chunk_size_s: int = 30,
    detection_batch_size: int = 32,
    worker_root: Path | None = None,
    beam_args: list[str] | None = None,
) -> None:
    """
    Do person detection on `video_path` and render output to file.
    """

    path = Path(video_path)
    chunk_dir = path.parent / f"{path.stem}_chunks"
    output_path = str(path.parent / f"{path.stem}_output{path.suffix}")

    chunks = split_video(video_path, chunk_dir, video_chunk_size_s)
    chunks = [(idx, to_worker_path(path, worker_root)) for idx, path in chunks]

    options = PipelineOptions(beam_args)
    with beam.Pipeline(options=options) as pipeline:
        chunk_paths = pipeline.apply(
            beam.Create(chunks), beam.pvalue.PBegin(pipeline), "CreateChunks"
        )
        detected = chunk_paths.apply(
            beam.ParDo(DetectChunk(detection_batch_size)), "Detect"
        )
        rendered = detected.apply(
            beam.ParDo(RenderChunk(detection_batch_size)), "Render"
        )
        gathered = rendered.apply(beam.combiners.ToList(), "GatherChunks")
        gathered.apply(
            beam.Map(
                concatenate_in_order,
                output_path=to_worker_path(output_path, worker_root),
            ),
            "Concatenate",
        )

    shutil.rmtree(chunk_dir)
    logger.info(f"Wrote {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("--video-chunk-size-s", type=int, default=30)
    parser.add_argument("--detection-batch-size", type=int, default=32)
    parser.add_argument("--worker-root", type=Path, default=None)
    args, beam_args = parser.parse_known_args()

    run(
        args.video_path,
        video_chunk_size_s=args.video_chunk_size_s,
        detection_batch_size=args.detection_batch_size,
        worker_root=args.worker_root,
        beam_args=beam_args,
    )
