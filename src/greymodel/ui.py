from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import posixpath
import shlex
import subprocess
import sys
from typing import Mapping, Sequence
from urllib.parse import urlparse, urlunparse


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


@dataclass(frozen=True)
class UIProxyLaunchConfig:
    proxy_mode: str
    bind_address: str
    bind_port: int
    base_url_path: str
    browser_server_address: str | None
    browser_server_port: int | None
    public_base_url: str | None
    local_url: str
    proxy_url: str


_NOTEBOOK_ENV_MARKERS = (
    "JPY_PARENT_PID",
    "JPY_SESSION_NAME",
    "JUPYTERHUB_API_TOKEN",
    "JUPYTERHUB_USER",
    "JUPYTER_SERVER_ROOT",
    "JUPYTER_IMAGE_SPEC",
)


def format_shell_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(token) for token in command])
    return shlex.join([str(token) for token in command])


def _normalized_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_url_path(path: str | None, *, trailing_slash: bool = True) -> str:
    normalized = _normalized_optional_text(path)
    if not normalized:
        return "/" if trailing_slash else ""
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    normalized = posixpath.normpath(normalized.replace("\\", "/"))
    if normalized == ".":
        normalized = "/"
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    if trailing_slash and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _to_streamlit_base_url_path(path: str | None) -> str:
    normalized = _normalize_url_path(path, trailing_slash=False)
    return normalized.strip("/")


def _is_absolute_url(value: str | None) -> bool:
    normalized = _normalized_optional_text(value)
    if normalized is None:
        return False
    parsed = urlparse(normalized)
    return bool(parsed.scheme and parsed.netloc)


def _join_public_location(base: str | None, route_path: str) -> str:
    normalized_route = _normalize_url_path(route_path, trailing_slash=True)
    normalized_base = _normalized_optional_text(base)
    if normalized_base is None:
        return normalized_route
    if _is_absolute_url(normalized_base):
        parsed = urlparse(normalized_base)
        joined_path = posixpath.join(parsed.path.rstrip("/"), normalized_route.lstrip("/"))
        if not joined_path.startswith("/"):
            joined_path = "/" + joined_path
        if not joined_path.endswith("/"):
            joined_path += "/"
        return urlunparse(parsed._replace(path=joined_path, params="", query="", fragment=""))
    return _normalize_url_path(normalized_base, trailing_slash=True) + normalized_route.lstrip("/")


def _extract_browser_target(public_base_url: str | None) -> tuple[str | None, int | None]:
    normalized = _normalized_optional_text(public_base_url)
    if normalized is None or not _is_absolute_url(normalized):
        return None, None
    parsed = urlparse(normalized)
    if parsed.hostname is None:
        return None, None
    if parsed.port is not None:
        return parsed.hostname, int(parsed.port)
    if parsed.scheme == "https":
        return parsed.hostname, 443
    if parsed.scheme == "http":
        return parsed.hostname, 80
    return parsed.hostname, None


def _infer_notebook_base_path(env: Mapping[str, str]) -> str:
    for key in ("NB_PREFIX", "JUPYTERHUB_SERVICE_PREFIX", "JUPYTER_BASE_URL", "JUPYTERHUB_BASE_URL"):
        if _normalized_optional_text(env.get(key)):
            return _normalize_url_path(env.get(key), trailing_slash=True)
    return "/"


def _detect_proxy_mode(proxy_mode: str, env: Mapping[str, str]) -> str:
    if proxy_mode != "auto":
        return proxy_mode
    if _normalized_optional_text(env.get("JUPYTERHUB_SERVICE_URL")):
        return "jupyter_service"
    if _normalized_optional_text(env.get("NB_PREFIX")) or any(_normalized_optional_text(env.get(key)) for key in _NOTEBOOK_ENV_MARKERS):
        return "jupyter_port"
    if _normalized_optional_text(env.get("JUPYTERHUB_SERVICE_PREFIX")):
        return "jupyter_port"
    return "off"


def resolve_ui_proxy_configuration(
    *,
    proxy_mode: str = "auto",
    public_base_url: str | None = None,
    base_url_path: str | None = None,
    bind_address: str | None = None,
    bind_port: int | None = None,
    browser_server_address: str | None = None,
    browser_server_port: int | None = None,
    env: Mapping[str, str] | None = None,
) -> UIProxyLaunchConfig:
    resolved_env = dict(os.environ if env is None else env)
    detected_mode = _detect_proxy_mode(str(proxy_mode), resolved_env)

    service_url = _normalized_optional_text(resolved_env.get("JUPYTERHUB_SERVICE_URL"))
    parsed_service_url = urlparse(service_url) if service_url else None
    env_service_prefix = _normalized_optional_text(resolved_env.get("JUPYTERHUB_SERVICE_PREFIX"))

    default_bind_address = "127.0.0.1"
    default_bind_port = 8501
    resolved_base_url_path = _normalized_optional_text(base_url_path)
    display_path = "/"

    if detected_mode == "jupyter_service":
        default_bind_address = parsed_service_url.hostname if parsed_service_url and parsed_service_url.hostname else "127.0.0.1"
        default_bind_port = parsed_service_url.port if parsed_service_url and parsed_service_url.port else 8501
        resolved_base_url_path = resolved_base_url_path or env_service_prefix or (parsed_service_url.path if parsed_service_url else None)
        display_path = _normalize_url_path(resolved_base_url_path, trailing_slash=True)
    elif detected_mode == "jupyter_port":
        default_bind_address = "0.0.0.0"
        default_bind_port = 8501
        display_path = _normalize_url_path(resolved_base_url_path, trailing_slash=True) if resolved_base_url_path else "/"
    else:
        display_path = _normalize_url_path(resolved_base_url_path, trailing_slash=True) if resolved_base_url_path else "/"

    final_bind_address = _normalized_optional_text(bind_address) or default_bind_address
    final_bind_port = int(bind_port if bind_port is not None else default_bind_port)
    streamlit_base_url_path = _to_streamlit_base_url_path(resolved_base_url_path)

    browser_host, browser_port = _extract_browser_target(public_base_url)
    final_browser_server_address = _normalized_optional_text(browser_server_address) or browser_host
    final_browser_server_port = int(browser_server_port) if browser_server_port is not None else browser_port

    local_display_host = "127.0.0.1" if final_bind_address in {"0.0.0.0", "::"} else final_bind_address
    local_url = "http://%s:%d/" % (local_display_host, final_bind_port)
    if streamlit_base_url_path:
        local_url += streamlit_base_url_path.rstrip("/") + "/"

    normalized_public_base_url = _normalized_optional_text(public_base_url)
    if detected_mode == "jupyter_service":
        proxy_url = _join_public_location(normalized_public_base_url, display_path)
    elif detected_mode == "jupyter_port":
        proxy_url = _join_public_location(normalized_public_base_url or _infer_notebook_base_path(resolved_env), "proxy/%d/" % final_bind_port)
    else:
        proxy_url = _join_public_location(normalized_public_base_url, display_path) if normalized_public_base_url else local_url

    return UIProxyLaunchConfig(
        proxy_mode=detected_mode,
        bind_address=final_bind_address,
        bind_port=final_bind_port,
        base_url_path=streamlit_base_url_path,
        browser_server_address=final_browser_server_address,
        browser_server_port=final_browser_server_port,
        public_base_url=normalized_public_base_url,
        local_url=local_url,
        proxy_url=proxy_url,
    )


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
    workspace_path: Path | str | None = None,
    host: str | None = None,
    port: int | None = None,
    bind_address: str | None = None,
    bind_port: int | None = None,
    headless: bool = True,
    proxy_mode: str = "auto",
    public_base_url: str | None = None,
    base_url_path: str | None = None,
    browser_server_address: str | None = None,
    browser_server_port: int | None = None,
    default_execution_backend: str = "local",
    slurm_cpus: int = 8,
    slurm_mem: str = "50G",
    slurm_gres: str = "gpu:8",
    slurm_partition: str = "",
    slurm_queue: str = "",
    slurm_nproc_per_node: int = 8,
    slurm_python: str | Path | None = sys.executable,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    resolved_slurm_python = sys.executable if slurm_python is None else slurm_python
    resolved_proxy = resolve_ui_proxy_configuration(
        proxy_mode=proxy_mode,
        public_base_url=public_base_url,
        base_url_path=base_url_path,
        bind_address=bind_address or host,
        bind_port=bind_port if bind_port is not None else port,
        browser_server_address=browser_server_address,
        browser_server_port=browser_server_port,
        env=env,
    )
    app_path = Path(__file__).with_name("ui_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=%s" % ("true" if headless else "false"),
        "--server.address=%s" % resolved_proxy.bind_address,
        "--server.port=%d" % int(resolved_proxy.bind_port),
    ]
    if resolved_proxy.base_url_path:
        command.append("--server.baseUrlPath=%s" % resolved_proxy.base_url_path)
    if resolved_proxy.browser_server_address:
        command.append("--browser.serverAddress=%s" % resolved_proxy.browser_server_address)
    if resolved_proxy.browser_server_port is not None:
        command.append("--browser.serverPort=%d" % int(resolved_proxy.browser_server_port))
    command.extend(
        [
            "--",
            "--run-root",
            str(run_root),
            "--data-root",
            str(data_root),
        ]
    )
    if workspace_path is not None:
        command.extend(["--workspace-path", str(workspace_path)])
    command.extend(
        [
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
    )
    if slurm_partition:
        command.extend(["--slurm-partition", str(slurm_partition)])
    if slurm_queue:
        command.extend(["--slurm-queue", str(slurm_queue)])
    return command


def launch_ui(
    *,
    run_root: Path | str = "artifacts",
    data_root: Path | str = "data",
    workspace_path: Path | str | None = None,
    host: str | None = None,
    port: int | None = None,
    bind_address: str | None = None,
    bind_port: int | None = None,
    headless: bool = True,
    proxy_mode: str = "auto",
    public_base_url: str | None = None,
    base_url_path: str | None = None,
    browser_server_address: str | None = None,
    browser_server_port: int | None = None,
    print_url: bool = False,
    default_execution_backend: str = "local",
    slurm_cpus: int = 8,
    slurm_mem: str = "50G",
    slurm_gres: str = "gpu:8",
    slurm_partition: str = "",
    slurm_queue: str = "",
    slurm_nproc_per_node: int = 8,
    slurm_python: str | Path | None = sys.executable,
) -> dict[str, object]:
    return launch_streamlit_ui(
        run_root=run_root,
        data_root=data_root,
        workspace_path=workspace_path,
        host=host,
        port=port,
        bind_address=bind_address,
        bind_port=bind_port,
        headless=headless,
        proxy_mode=proxy_mode,
        public_base_url=public_base_url,
        base_url_path=base_url_path,
        browser_server_address=browser_server_address,
        browser_server_port=browser_server_port,
        print_url=print_url,
        default_execution_backend=default_execution_backend,
        slurm_cpus=slurm_cpus,
        slurm_mem=slurm_mem,
        slurm_gres=slurm_gres,
        slurm_partition=slurm_partition,
        slurm_queue=slurm_queue,
        slurm_nproc_per_node=slurm_nproc_per_node,
        slurm_python=slurm_python,
        dry_run=False,
    )


def launch_streamlit_ui(
    *,
    run_root: Path | str = "artifacts",
    data_root: Path | str = "data",
    workspace_path: Path | str | None = None,
    host: str | None = None,
    port: int | None = None,
    bind_address: str | None = None,
    bind_port: int | None = None,
    headless: bool = True,
    dry_run: bool = False,
    proxy_mode: str = "auto",
    public_base_url: str | None = None,
    base_url_path: str | None = None,
    browser_server_address: str | None = None,
    browser_server_port: int | None = None,
    print_url: bool = False,
    default_execution_backend: str = "local",
    slurm_cpus: int = 8,
    slurm_mem: str = "50G",
    slurm_gres: str = "gpu:8",
    slurm_partition: str = "",
    slurm_queue: str = "",
    slurm_nproc_per_node: int = 8,
    slurm_python: str | Path | None = sys.executable,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    resolved_slurm_python = sys.executable if slurm_python is None else slurm_python
    resolved_proxy = resolve_ui_proxy_configuration(
        proxy_mode=proxy_mode,
        public_base_url=public_base_url,
        base_url_path=base_url_path,
        bind_address=bind_address or host,
        bind_port=bind_port if bind_port is not None else port,
        browser_server_address=browser_server_address,
        browser_server_port=browser_server_port,
        env=env,
    )
    command = build_streamlit_command(
        run_root=run_root,
        data_root=data_root,
        workspace_path=workspace_path,
        bind_address=resolved_proxy.bind_address,
        bind_port=resolved_proxy.bind_port,
        headless=headless,
        proxy_mode=resolved_proxy.proxy_mode,
        public_base_url=resolved_proxy.public_base_url,
        base_url_path=resolved_proxy.base_url_path,
        browser_server_address=resolved_proxy.browser_server_address,
        browser_server_port=resolved_proxy.browser_server_port,
        default_execution_backend=default_execution_backend,
        slurm_cpus=slurm_cpus,
        slurm_mem=slurm_mem,
        slurm_gres=slurm_gres,
        slurm_partition=slurm_partition,
        slurm_queue=slurm_queue,
        slurm_nproc_per_node=slurm_nproc_per_node,
        slurm_python=resolved_slurm_python,
        env=env,
    )
    payload = {
        "command": command,
        "streamlit_command": command,
        "run_root": str(run_root),
        "data_root": str(data_root),
        "workspace_path": str(workspace_path) if workspace_path is not None else None,
        "host": resolved_proxy.bind_address,
        "port": int(resolved_proxy.bind_port),
        "bind_address": resolved_proxy.bind_address,
        "bind_port": int(resolved_proxy.bind_port),
        "headless": bool(headless),
        "default_execution_backend": str(default_execution_backend),
        "slurm_cpus": int(slurm_cpus),
        "slurm_mem": str(slurm_mem),
        "slurm_gres": str(slurm_gres),
        "slurm_partition": str(slurm_partition),
        "slurm_queue": str(slurm_queue),
        "slurm_nproc_per_node": int(slurm_nproc_per_node),
        "slurm_python": str(resolved_slurm_python),
        "proxy_mode": resolved_proxy.proxy_mode,
        "public_base_url": resolved_proxy.public_base_url,
        "base_url_path": resolved_proxy.base_url_path,
        "browser_server_address": resolved_proxy.browser_server_address,
        "browser_server_port": resolved_proxy.browser_server_port,
        "local_url": resolved_proxy.local_url,
        "proxy_url": resolved_proxy.proxy_url,
    }
    if print_url:
        if resolved_proxy.proxy_url:
            print(resolved_proxy.proxy_url)
        if resolved_proxy.local_url and resolved_proxy.local_url != resolved_proxy.proxy_url:
            print(resolved_proxy.local_url)
    if dry_run:
        payload["return_code"] = 0
        return payload
    if importlib.util.find_spec("streamlit") is None:
        raise ImportError("Streamlit is required for `greymodel ui`. Install `greymodel[framework]` or add `streamlit`.")
    result = subprocess.run(command, check=False)
    payload["return_code"] = int(result.returncode)
    return payload
