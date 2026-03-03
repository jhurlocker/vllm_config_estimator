import os
import shlex
import subprocess
import time

import psutil
import requests

from llm_optimizer.exceptions import ServerNotReadyError
from llm_optimizer.logging import get_logger

logger = get_logger("server_utils")


def wait_for_server(
    url: str,
    max_retries: int = 60,
    delay: int = 5,
    request_timeout: int = 5,
) -> bool:
    """Repeatedly try to access the server until it's available."""
    logger.info(f"Waiting for server to be ready at {url}...")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=request_timeout)
            if response.status_code == 200:
                logger.info(f"Server is up after {attempt + 1} attempts.")
                return True
        except requests.RequestException:
            pass

        logger.debug(
            f"Server not ready yet (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s."
        )
        time.sleep(delay)
    return False


def start_server(
    server_cmd: str, server_envs: dict[str, str], ready_url: str, mute: bool
) -> psutil.Process:
    """Starts the server and waits for it to become ready."""
    cmd = shlex.split(server_cmd)
    logger.info(f"Starting server with command: {server_cmd}")
    server_process = None
    envs = os.environ.copy()
    envs.update(server_envs)

    try:
        stdout_pipe = subprocess.DEVNULL if mute else None
        server_process = psutil.Popen(
            cmd, text=True, stdout=stdout_pipe, stderr=subprocess.STDOUT, env=envs
        )
        logger.debug(f"Started server with PID: {server_process.pid}")

        if not wait_for_server(ready_url):
            raise ServerNotReadyError(
                "Server did not become ready in the allotted time."
            )

        return server_process
    except Exception:
        if server_process:
            terminate_process_top_down(server_process)
        raise


def terminate_process_top_down(process: psutil.Process, timeout: int = 10):
    """Gracefully terminates a process and its children, escalating to KILL if necessary."""
    logger.info(f"Terminating server process {process.pid} and its children.")
    try:
        children = process.children(recursive=True)
        # Terminate children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Terminate parent
        process.terminate()

        # Wait for all to die
        _, alive = psutil.wait_procs([process] + children, timeout=timeout)

        # Force kill any remaining processes
        if alive:
            logger.warning(
                "Some processes did not terminate gracefully. Escalating to SIGKILL."
            )
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=timeout)
    except psutil.NoSuchProcess:
        logger.debug(f"Process {process.pid} was already gone.")
    except Exception as e:
        logger.error(f"Error during process termination: {e}")
