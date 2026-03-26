from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class UIExecutionDefaults:
    execution_backend: str = "local"
    slurm_cpus: int = 8
    slurm_mem: str = "50G"
    slurm_gres: str = "gpu:8"
    slurm_partition: str = ""
    slurm_queue: str = ""
    slurm_nproc_per_node: int = 8
    slurm_python: str = sys.executable


def format_shell_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(token) for token in command])
    return shlex.join([str(token) for token in command])


def build_greymodel_job_command(
    task_tokens: Sequence[str],
    *,
    python_executable: str | Path = sys.executable,
    nproc_per_node: int = 1,
) -> list[str]:
    command = [str(python_executable)]
    if int(nproc_per_node) > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=%d" % int(nproc_per_node),
                "-m",
                "greymodel",
            ]
        )
    else:
        command.extend(["-m", "greymodel"])
    command.extend([str(token) for token in task_tokens])
    return command


def build_slurm_submission_command(
    *,
    inner_command: Sequence[str],
    repo_root: Path | str,
    cpus: int = 8,
    mem: str = "50G",
    gres: str = "gpu:8",
    partition: str | None = None,
    queue: str | None = None,
    job_name: str | None = None,
    log_path: Path | str | None = None,
) -> list[str]:
    wrap_command = "cd %s && %s" % (
        shlex.quote(str(repo_root)),
        format_shell_command(inner_command),
    )
    command = ["sbatch", "--parsable", "-c", str(int(cpus)), "--mem=%s" % mem, "--gres=%s" % gres]
    if partition:
        command.extend(["-p", str(partition)])
    if queue:
        command.extend(["-q", str(queue)])
    if job_name:
        command.extend(["--job-name", str(job_name)])
    if log_path is not None:
        command.extend(["--output", str(log_path), "--error", str(log_path)])
    command.extend(["--wrap", wrap_command])
    return command


def build_streamlit_command(
    *,
    run_root: Path | str = "artifacts",
    data_root: Path | str = "data",
    host: str = "127.0.0.1",
    port: int = 8501,
    headless: bool = True,
    default_execution_backend: str = "local",
    slurm_cpus: int = 8,
    slurm_mem: str = "50G",
    slurm_gres: str = "gpu:8",
    slurm_partition: str = "",
    slurm_queue: str = "",
    slurm_nproc_per_node: int = 8,
    slurm_python: str | Path | None = sys.executable,
) -> list[str]:
    resolved_slurm_python = sys.executable if slurm_python is None else slurm_python
    app_path = Path(__file__).with_name("ui_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address=%s" % host,
        "--server.port=%d" % int(port),
        "--",
        "--run-root",
        str(run_root),
        "--data-root",
        str(data_root),
        "--default-execution-backend",
        str(default_execution_backend),
        "--slurm-cpus",
        str(int(slurm_cpus)),
        "--slurm-mem",
        str(slurm_mem),
        "--slurm-gres",
        str(slurm_gres),
        "--slurm-nproc-per-node",
        str(int(slurm_nproc_per_node)),
        "--slurm-python",
        str(resolved_slurm_python),
    ]
    if slurm_partition:
        command.extend(["--slurm-partition", str(slurm_partition)])
    if slurm_queue:
        command.extend(["--slurm-queue", str(slurm_queue)])
    command.insert(5, "--server.headless=%s" % ("true" if headless else "false"))
    return command


def launch_ui(
    *,
    run_root: Path | str = "artifacts",
    data_root: Path | str = "data",
    host: str = "127.0.0.1",
    port: int = 8501,
    headless: bool = True,
    default_execution_backend: str = "local",
    slurm_cpus: int = 8,
    slurm_mem: str = "50G",
    slurm_gres: str = "gpu:8",
    slurm_partition: str = "",
    slurm_queue: str = "",
    slurm_nproc_per_node: int = 8,
    slurm_python: str | Path | None = sys.executable,
) -> dict[str, object]:
    if importlib.util.find_spec("streamlit") is None:
        raise ImportError("Streamlit is required for `greymodel ui`. Install `greymodel[framework]` or add `streamlit`.")
    command = build_streamlit_command(
        run_root=run_root,
        data_root=data_root,
        host=host,
        port=port,
        headless=headless,
        default_execution_backend=default_execution_backend,
        slurm_cpus=slurm_cpus,
        slurm_mem=slurm_mem,
        slurm_gres=slurm_gres,
        slurm_partition=slurm_partition,
        slurm_queue=slurm_queue,
        slurm_nproc_per_node=slurm_nproc_per_node,
        slurm_python=slurm_python,
    )
    result = subprocess.run(command, check=False)
    return {
        "command": command,
        "return_code": int(result.returncode),
        "run_root": str(run_root),
        "data_root": str(data_root),
        "host": host,
        "port": int(port),
        "default_execution_backend": str(default_execution_backend),
    }


def launch_streamlit_ui(
    *,
    run_root: Path | str = "artifacts",
    data_root: Path | str = "data",
    host: str = "127.0.0.1",
    port: int = 8501,
    headless: bool = True,
    dry_run: bool = False,
    default_execution_backend: str = "local",
    slurm_cpus: int = 8,
    slurm_mem: str = "50G",
    slurm_gres: str = "gpu:8",
    slurm_partition: str = "",
    slurm_queue: str = "",
    slurm_nproc_per_node: int = 8,
    slurm_python: str | Path | None = sys.executable,
) -> dict[str, object]:
    resolved_slurm_python = sys.executable if slurm_python is None else slurm_python
    command = build_streamlit_command(
        run_root=run_root,
        data_root=data_root,
        host=host,
        port=port,
        headless=headless,
        default_execution_backend=default_execution_backend,
        slurm_cpus=slurm_cpus,
        slurm_mem=slurm_mem,
        slurm_gres=slurm_gres,
        slurm_partition=slurm_partition,
        slurm_queue=slurm_queue,
        slurm_nproc_per_node=slurm_nproc_per_node,
        slurm_python=resolved_slurm_python,
    )
    payload = {
        "command": command,
        "run_root": str(run_root),
        "data_root": str(data_root),
        "host": host,
        "port": int(port),
        "headless": bool(headless),
        "default_execution_backend": str(default_execution_backend),
        "slurm_cpus": int(slurm_cpus),
        "slurm_mem": str(slurm_mem),
        "slurm_gres": str(slurm_gres),
        "slurm_partition": str(slurm_partition),
        "slurm_queue": str(slurm_queue),
        "slurm_nproc_per_node": int(slurm_nproc_per_node),
        "slurm_python": str(resolved_slurm_python),
    }
    if dry_run:
        payload["return_code"] = 0
        return payload
    if importlib.util.find_spec("streamlit") is None:
        raise ImportError("Streamlit is required for `greymodel ui`. Install `greymodel[framework]` or add `streamlit`.")
    result = subprocess.run(command, check=False)
    payload["return_code"] = int(result.returncode)
    return payload
