"""Benchmarking YOLO26 vs YOLO11 for hand pose estimation."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from src.utils.constants import OUTPUTS_DIR
from src.utils.device import get_device


@dataclass
class BenchmarkResult:
    """Single benchmark result."""

    model_name: str
    device: str
    batch_size: int
    image_size: int
    # Timing
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    fps: float
    # Model info
    parameters: int
    flops: Optional[float] = None
    # Accuracy (if evaluated)
    map50: Optional[float] = None
    map50_95: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class YOLOBenchmark:
    """
    Benchmark YOLO26 vs YOLO11 for hand pose estimation.

    Compares:
    - Inference latency (CPU and GPU)
    - Throughput (FPS)
    - Model size and parameters
    - Accuracy on hand keypoints dataset

    Example:
        >>> benchmark = YOLOBenchmark()
        >>> results = benchmark.run_full_benchmark()
        >>> benchmark.save_results(results)
    """

    def __init__(
        self,
        models: Optional[list[str]] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize benchmark.

        Args:
            models: List of model names to benchmark.
            output_dir: Output directory for results.
        """
        self.models = models or [
            "yolo26n-pose.pt",
            "yolo26s-pose.pt",
            "yolo11n-pose.pt",
            "yolo11s-pose.pt",
        ]
        self.output_dir = output_dir or OUTPUTS_DIR / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def benchmark_latency(
        self,
        model_name: str,
        device: str = "cpu",
        image_size: int = 640,
        batch_size: int = 1,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ) -> BenchmarkResult:
        """
        Benchmark inference latency.

        Args:
            model_name: Model to benchmark.
            device: Device to run on.
            image_size: Input image size.
            batch_size: Batch size.
            warmup_iterations: Warmup iterations.
            benchmark_iterations: Benchmark iterations.

        Returns:
            BenchmarkResult with timing data.
        """
        from ultralytics import YOLO

        logger.info(f"Benchmarking {model_name} on {device}")

        # Load model
        model = YOLO(model_name)

        # Create dummy input
        dummy_input = np.random.randint(
            0, 255, (image_size, image_size, 3), dtype=np.uint8
        )

        # Warmup
        logger.debug(f"Warming up for {warmup_iterations} iterations")
        for _ in range(warmup_iterations):
            model(dummy_input, device=device, verbose=False)

        # Benchmark
        latencies = []
        logger.debug(f"Benchmarking for {benchmark_iterations} iterations")

        for _ in range(benchmark_iterations):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            model(dummy_input, device=device, verbose=False)

            if device == "cuda":
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies = np.array(latencies)

        # Get model info
        params = sum(p.numel() for p in model.model.parameters())

        result = BenchmarkResult(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            image_size=image_size,
            mean_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            fps=float(1000 / np.mean(latencies)),
            parameters=params,
        )

        logger.info(
            f"{model_name} ({device}): "
            f"{result.mean_latency_ms:.2f} ms ({result.fps:.1f} FPS)"
        )

        return result

    def benchmark_accuracy(
        self,
        model_name: str,
        data_yaml: str = "hand-keypoints.yaml",
        device: str = "cpu",
    ) -> dict:
        """
        Benchmark model accuracy on hand keypoints dataset.

        Args:
            model_name: Model to evaluate.
            data_yaml: Dataset configuration.
            device: Device to run on.

        Returns:
            Dictionary with accuracy metrics.
        """
        from ultralytics import YOLO

        logger.info(f"Evaluating {model_name} accuracy")

        model = YOLO(model_name)
        results = model.val(data=data_yaml, device=device, verbose=False)

        metrics = {
            "mAP50": float(results.pose.map50),
            "mAP50-95": float(results.pose.map),
            "precision": float(results.pose.mp),
            "recall": float(results.pose.mr),
        }

        logger.info(f"{model_name}: mAP50={metrics['mAP50']:.3f}")

        return metrics

    def run_full_benchmark(
        self,
        devices: Optional[list[str]] = None,
        image_sizes: list[int] = [640],
        include_accuracy: bool = False,
    ) -> list[BenchmarkResult]:
        """
        Run full benchmark suite.

        Args:
            devices: Devices to benchmark on.
            image_sizes: Image sizes to test.
            include_accuracy: Include accuracy evaluation.

        Returns:
            List of benchmark results.
        """
        if devices is None:
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda")
            if torch.backends.mps.is_available():
                devices.append("mps")

        results = []

        for model_name in self.models:
            for device in devices:
                for imgsz in image_sizes:
                    try:
                        result = self.benchmark_latency(
                            model_name=model_name,
                            device=device,
                            image_size=imgsz,
                        )

                        if include_accuracy and device == devices[0]:
                            accuracy = self.benchmark_accuracy(model_name, device=device)
                            result.map50 = accuracy["mAP50"]
                            result.map50_95 = accuracy["mAP50-95"]

                        results.append(result)

                    except Exception as e:
                        logger.warning(f"Failed to benchmark {model_name} on {device}: {e}")

        return results

    def compare_yolo26_vs_yolo11(
        self,
        device: str = "cpu",
    ) -> dict:
        """
        Direct comparison of YOLO26 vs YOLO11.

        Args:
            device: Device to benchmark on.

        Returns:
            Comparison dictionary.
        """
        logger.info("Comparing YOLO26 vs YOLO11 for hand pose estimation")

        # Benchmark both models
        yolo26 = self.benchmark_latency("yolo26n-pose.pt", device=device)
        yolo11 = self.benchmark_latency("yolo11n-pose.pt", device=device)

        # Calculate improvements
        speedup = yolo11.mean_latency_ms / yolo26.mean_latency_ms
        speedup_pct = (speedup - 1) * 100

        comparison = {
            "yolo26": yolo26.to_dict(),
            "yolo11": yolo11.to_dict(),
            "speedup": speedup,
            "speedup_percent": speedup_pct,
            "device": device,
            "conclusion": (
                f"YOLO26 is {speedup_pct:.1f}% faster than YOLO11 on {device}"
                if speedup > 1
                else f"YOLO11 is {-speedup_pct:.1f}% faster than YOLO26 on {device}"
            ),
        }

        logger.info(comparison["conclusion"])

        return comparison

    def save_results(
        self,
        results: list[BenchmarkResult],
        filename: str = "benchmark_results.json",
    ) -> Path:
        """Save benchmark results to JSON."""
        output_path = self.output_dir / filename

        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {output_path}")
        return output_path

    def generate_report(
        self,
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate markdown benchmark report."""
        import pandas as pd

        df = pd.DataFrame([r.to_dict() for r in results])

        report = "# YOLO26 vs YOLO11 Benchmark Report\n\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += "## Latency Results\n\n"
        report += df.to_markdown(index=False)
        report += "\n\n"

        # Calculate speedups
        if len(df) >= 2:
            report += "## Key Findings\n\n"

            yolo26_cpu = df[df["model_name"].str.contains("yolo26") & (df["device"] == "cpu")]
            yolo11_cpu = df[df["model_name"].str.contains("yolo11") & (df["device"] == "cpu")]

            if len(yolo26_cpu) > 0 and len(yolo11_cpu) > 0:
                speedup = yolo11_cpu["mean_latency_ms"].values[0] / yolo26_cpu["mean_latency_ms"].values[0]
                report += f"- **CPU Speedup**: YOLO26 is {(speedup-1)*100:.1f}% faster\n"

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report


def main():
    """CLI entry point for benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Benchmark")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compare", action="store_true", help="Compare YOLO26 vs YOLO11")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")

    args = parser.parse_args()

    benchmark = YOLOBenchmark(models=args.models)

    if args.compare:
        comparison = benchmark.compare_yolo26_vs_yolo11(device=args.device)
        print(json.dumps(comparison, indent=2))
    elif args.full:
        results = benchmark.run_full_benchmark()
        benchmark.save_results(results)
        print(benchmark.generate_report(results))
    else:
        result = benchmark.benchmark_latency("yolo26n-pose.pt", device=args.device)
        print(result)


if __name__ == "__main__":
    main()
