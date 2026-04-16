import asyncio
import time
import random
import numpy as np
import os
import json
import datetime
from typing import Callable, Any, List, Optional, Awaitable, Dict
from dataclasses import dataclass
from .hardware import profile_hardware

@dataclass
class QuerySample:
    """A single query sample in a loadgen scenario."""
    id: int
    index: int
    arrival_time: float

@dataclass
class QuerySampleResponse:
    """A response to a query sample."""
    id: int
    response_data: Any
    arrival_time: float
    completion_time: float
    latency: float

class LoadGenProxy:
    """
    An asynchronous LoadGen proxy for MLPerf EDU.
    
    This class simulates 'Offline', 'Server', and 'SingleStream' scenarios using 
    asyncio to manage high-concurrency and precise timing without GIL bottlenecks.
    """
    def __init__(self, scenario: str = 'Offline', qps: float = 10.0, min_query_count: int = 100, min_duration_seconds: float = 10.0, division: str = 'closed', config: Dict[str, Any] = None):
        self.scenario = scenario
        self.qps = qps
        self.min_query_count = min_query_count
        self.min_duration_seconds = min_duration_seconds
        self.division = division
        self.config = config
        self.responses: List[QuerySampleResponse] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        from .power import PowerProfiler
        self.power_profiler = PowerProfiler()
        self.power_report: Dict[str, Any] = {}

    async def run(self, query_handler: Callable[[List[QuerySample]], Awaitable[Any]]) -> Dict[str, Any]:
        """
        Executes the benchmark scenario.
        
        Args:
            query_handler: An asynchronous function that processes a batch of QuerySamples.
            
        Returns:
            A dictionary containing benchmark metrics.
        """
        self.responses = []
        self.start_time = time.perf_counter()
        
        # Introspect Hardware before generating workload load
        self.hardware_signature = profile_hardware()

        # Spin up Power Telemetry Loop!
        await self.power_profiler.start()
        
        if self.scenario == 'Offline':
            await self._run_offline(query_handler)
        elif self.scenario == 'Server':
            await self._run_server(query_handler)
        elif self.scenario == 'SingleStream':
            await self._run_single_stream(query_handler)
        elif self.scenario == 'MultiStream':
            await self._run_multi_stream(query_handler)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")
            
        self.end_time = time.perf_counter()
        
        # Halt Power Telemetry & gather Watts
        self.power_report = await self.power_profiler.stop()
        
        report = self._report()
        
        # Enforce Submission Bounds
        self._evaluate_submission_tolerances(report)
        
        # Secure the Payload
        self._generate_submission_artifact(report)
        return report

    def _evaluate_submission_tolerances(self, report: Dict[str, Any]):
        """
        Referee grading hook: Prevents students from faking performance by ruining accuracy.
        Extracts student self-reported 'accuracy' (or computes it internally),
        and tests it against the exact yaml division threshold for this SUT.
        """
        from rich.console import Console
        import sys
        cn = Console()
        
        if not self.config or "divisions" not in self.config:
            return  # Skip if testing pure monolithic modules

        student_accuracy = report.get("accuracy_avg", None)
        if student_accuracy is None:
            # Fallback for testing/pedagogy mock if student plugin didn't explicitly return accuracy
            student_accuracy = self.config['baseline_accuracy_fp32']
            cn.print("[yellow]⚠️ Warning: SUT returned no accuracy metric. Falling back to FP32 Golden assumed value for grading.[/yellow]")
            
        target_threshold = self.config['divisions'][self.division]['threshold']
        
        cn.print(f"[bold cyan]🔍 Grading the {self.division.upper()} Division...[/bold cyan]")
        cn.print(f"Goal: {target_threshold:.4f} | Achieved: {student_accuracy:.4f}")
        
        if student_accuracy < target_threshold:
            cn.print(f"[bold red]❌ DISQUALIFIED: Submission failed the '{self.division.upper()}' rule.[/bold red]")
            cn.print(f"[red]The SUT optimization degraded the model below the mathematical {self.division} threshold.[/red]")
            sys.exit(1)
            
        cn.print(f"[bold green]✅ Submission passed {self.division.upper()} division constraints![/bold green]")
        report["division_passed"] = self.division

    def _generate_submission_artifact(self, report: Dict[str, Any]):
        """Build a real provenance manifest binding the run to its inputs.

        Iter-5 (Dean's spec): replaces the iter-1 era `str(report)`
        self-hash with manifest.build_provd(), which computes a Merkle
        root over leaves for source_tree/weights/dataset/rng/hardware/
        roofline_sidecar/measurement. Verification is via
        scripts/verify_submission.py.
        """
        from .manifest import build_provd

        os.makedirs("submissions", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"submissions/mlperf_submission_{timestamp}_report.json"
        manifest_filename = f"submissions/mlperf_submission_{timestamp}.provd.json"

        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)

        # Roofline sidecar (if a measure_roofline() block ran in this process).
        sidecar_path = os.environ.get("MLPERF_EDU_LAST_SIDECAR")

        # Optional inputs the workload's run() can stash on self for binding.
        manifest = build_provd(
            workload=getattr(self, "workload_name", "unknown"),
            scenario=self.scenario,
            division=self.division,
            hardware_fingerprint=self.hardware_signature,
            report=report,
            report_path=report_filename,
            weights_path=getattr(self, "weights_path", None),
            weights_n_params=getattr(self, "weights_n_params", None),
            dataset_name=getattr(self, "dataset_name", "unknown"),
            dataset_files=getattr(self, "dataset_files", None),
            rng_seed=getattr(self, "rng_seed", None),
            torch_state_bytes=getattr(self, "torch_state_bytes", None),
            roofline_sidecar_path=sidecar_path,
        )
        with open(manifest_filename, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)

    async def _run_offline(self, query_handler: Callable[[List[QuerySample]], Awaitable[Any]]):
        """
        Offline scenario: All samples are available at once.
        Usually processes one gigantic batch, but we must sustain load for min_duration!
        For pedagogical simulation, we loop sending the max query count until time expires.
        """
        queries_sent = 0
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Offline LoadGen"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TextColumn("•"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task_id = progress.add_task("Inferencing...", total=self.min_duration_seconds)
            
            while True:
                samples = [
                    QuerySample(id=i + queries_sent, index=i + queries_sent, arrival_time=self.start_time) 
                    for i in range(self.min_query_count)
                ]
                await self._dispatch(samples, query_handler)
                queries_sent += len(samples)
                
                elapsed = time.perf_counter() - self.start_time
                progress.update(task_id, completed=min(elapsed, self.min_duration_seconds))
                
                if elapsed >= self.min_duration_seconds and queries_sent >= self.min_query_count:
                    break

    async def _run_server(self, query_handler: Callable[[List[QuerySample]], Awaitable[Any]]):
        """
        Server scenario: Samples arrive according to a Poisson process.
        Tests system responsiveness under random load.
        """
        tasks = []
        target_relative_time = 0.0
        queries_sent = 0
        
        from rich.console import Console
        cn = Console()
        cn.print(f"[bold cyan]⏳ Initiating Poisson LoadGen: Target Duration {self.min_duration_seconds}s | Min Queries {self.min_query_count}[/bold cyan]")
        
        while True:
            inter_arrival = random.expovariate(self.qps)
            target_relative_time += inter_arrival
            
            sample = QuerySample(id=queries_sent, index=queries_sent, arrival_time=self.start_time + target_relative_time)
            
            now_relative = time.perf_counter() - self.start_time
            delay = target_relative_time - now_relative
            if delay > 0:
                await asyncio.sleep(delay)
            
            tasks.append(asyncio.create_task(self._dispatch([sample], query_handler)))
            queries_sent += 1
            
            # Check Statistical limits
            elapsed = time.perf_counter() - self.start_time
            if elapsed >= self.min_duration_seconds and queries_sent >= self.min_query_count:
                break
            
        if tasks:
            await asyncio.gather(*tasks)

    async def _run_single_stream(self, query_handler: Callable[[List[QuerySample]], Awaitable[Any]]):
        """
        SingleStream scenario: One sample at a time.
        Sends next sample only after previous one is finished.
        Tests minimum latency.
        """
        queries_sent = 0
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]SingleStream LoadGen"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total} Queries"),
            TextColumn("•"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task_id = progress.add_task("Inferencing...", total=self.min_query_count)
            
            while True:
                arrival_time = time.perf_counter()
                sample = QuerySample(id=queries_sent, index=queries_sent, arrival_time=arrival_time)
                await self._dispatch([sample], query_handler)
                queries_sent += 1
                
                elapsed = time.perf_counter() - self.start_time
                progress.update(task_id, completed=queries_sent)
                
                if elapsed >= self.min_duration_seconds and queries_sent >= self.min_query_count:
                    break

    async def _run_multi_stream(self, query_handler: Callable[[List[QuerySample]], Awaitable[Any]]):
        """
        MultiStream scenario: N concurrent fixed streams issuing queries continuously.
        Tests parallel concurrency capacity limits automatically bounded.
        """
        queries_sent = 0
        concurrency_limit = self.config.get('multi_stream_concurrency', 4) if self.config else 4
        
        from rich.console import Console
        cn = Console()
        cn.print(f"[bold cyan]⏳ Initiating MultiStream ({concurrency_limit} concurrent streams) LoadGen: Target Duration {self.min_duration_seconds}s | Min Queries {self.min_query_count}[/bold cyan]")
        
        async def stream_worker():
            nonlocal queries_sent
            while True:
                elapsed = time.perf_counter() - self.start_time
                if elapsed >= self.min_duration_seconds and queries_sent >= self.min_query_count:
                    break
                
                # We dynamically grab the next ID across all streams naturally
                query_id = queries_sent
                queries_sent += 1
                
                arrival_time = time.perf_counter()
                sample = QuerySample(id=query_id, index=query_id, arrival_time=arrival_time)
                await self._dispatch([sample], query_handler)

        workers = [asyncio.create_task(stream_worker()) for _ in range(concurrency_limit)]
        await asyncio.gather(*workers)

    async def _dispatch(self, samples: List[QuerySample], query_handler: Callable[[List[QuerySample]], Awaitable[Any]]):
        """Dispatches samples to the handler and records timing metrics."""
        try:
            # query_handler is an async function.
            result = await query_handler(samples)
        except Exception as e:
            # Log error but don't stop the whole benchmark
            print(f"[LoadGenProxy] Error in query_handler: {e}")
            result = None
            
        completion_time = time.perf_counter()
        
        for i, sample in enumerate(samples):
            latency = completion_time - sample.arrival_time
            # If query_handler returned a list of results, map them to samples
            sample_result = result[i] if isinstance(result, list) and len(result) == len(samples) else result
            
            self.responses.append(QuerySampleResponse(
                id=sample.id,
                response_data=sample_result,
                arrival_time=sample.arrival_time,
                completion_time=completion_time,
                latency=latency
            ))

    def _report(self) -> Dict[str, Any]:
        """Calculates and returns performance metrics."""
        duration = self.end_time - self.start_time
        latencies = [r.latency for r in self.responses]
        
        # Sort responses by id to ensure they are in order if needed
        self.responses.sort(key=lambda r: r.id)
        
        report = {
            "scenario": self.scenario,
            "duration": float(duration),
            "total_samples": len(self.responses),
            "queries_per_second": len(self.responses) / duration if duration > 0 else 0,
            "latency_avg": float(np.mean(latencies)) if latencies else 0.0,
            "latency_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "latency_p90": float(np.percentile(latencies, 90)) if latencies else 0.0,
            "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "latency_p99": float(np.percentile(latencies, 99)) if latencies else 0.0,
        }

        # Add Power Metrics mathematically
        report["average_watts"] = self.power_report.get("average_watts", 0.0)
        report["estimated_joules"] = self.power_report.get("estimated_joules", 0.0)

        # Handle arbitrary dictionary metrics (e.g. NMS vs Backbone time, TTFT)
        if self.responses and isinstance(self.responses[0].response_data, dict):
            keys = self.responses[0].response_data.keys()
            for key in keys:
                # We skip non-numeric keys like output_ids manually here or infer safely
                vals = [r.response_data.get(key) for r in self.responses if isinstance(r.response_data.get(key), (int, float))]
                if vals:
                    report[f"{key}_avg"] = float(np.mean(vals))
                    report[f"{key}_p90"] = float(np.percentile(vals, 90))

        return report
