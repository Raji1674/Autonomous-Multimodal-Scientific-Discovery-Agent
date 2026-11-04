
"""Docker-based isolated Python execution."""

import docker
from docker.errors import DockerException
import requests
from urllib3.exceptions import ReadTimeoutError

from src.sandbox.restrictions import (
    DOCKER_IMAGE,
    MEMORY_LIMIT,
    CPU_LIMIT,
    PIDS_LIMIT,
    EXECUTION_TIMEOUT,
    NETWORK_MODE,
    READ_ONLY_ROOTFS,
    DROP_ALL_CAPABILITIES,
    NO_NEW_PRIVILEGES,
    TMPFS_SIZE,
    MAX_OUTPUT_BYTES,
)


class SandboxExecutionError(Exception):
    """Raised when sandbox execution fails."""


class SandboxTimeoutError(SandboxExecutionError):
    """Raised when sandbox execution exceeds its time limit."""


def execute_python(code: str) -> dict:
    """Execute Python code inside a restricted Docker container."""

    if not isinstance(code, str) or not code.strip():
        raise ValueError("Python code must be a non-empty string.")

    if len(code.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ValueError("Python code exceeds the permitted size.")

    client = None
    container = None

    try:
        client = docker.from_env()

        container = client.containers.run(
            image=DOCKER_IMAGE,
            command=["python", "-I", "-c", code],
            detach=True,
            network_mode=NETWORK_MODE,
            mem_limit=MEMORY_LIMIT,
            nano_cpus=int(CPU_LIMIT * 1_000_000_000),
            pids_limit=PIDS_LIMIT,
            read_only=READ_ONLY_ROOTFS,
            cap_drop=["ALL"] if DROP_ALL_CAPABILITIES else [],
            security_opt=(
                ["no-new-privileges:true"]
                if NO_NEW_PRIVILEGES else []
            ),
            tmpfs={
                "/tmp": f"rw,noexec,nosuid,nodev,size={TMPFS_SIZE}"
            },
            user="65534:65534",
            stdin_open=False,
            tty=False,
            auto_remove=False,
        )

        # Wait for container execution to finish.
        try:
            result = container.wait(timeout=EXECUTION_TIMEOUT)

        except (
            TimeoutError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            ReadTimeoutError,
        ) as exc:
            # Stop the container when execution exceeds the time limit.
            try:
                container.kill()
            except DockerException:
                pass

            raise SandboxTimeoutError(
                f"Execution exceeded {EXECUTION_TIMEOUT} seconds."
            ) from exc

        except DockerException as exc:
            raise SandboxExecutionError(
                f"Docker wait failed: {exc}"
            ) from exc

        # Read only a bounded amount of output.
        output_stream = container.logs(
            stdout=True,
            stderr=True,
            stream=True,
            follow=False,
        )

        chunks = []
        total_bytes = 0
        truncated = False

        for chunk in output_stream:
            remaining = MAX_OUTPUT_BYTES - total_bytes

            if remaining <= 0:
                truncated = True
                break

            chunks.append(chunk[:remaining])
            total_bytes += min(len(chunk), remaining)

            if len(chunk) > remaining:
                truncated = True
                break

        output = b"".join(chunks).decode(
            "utf-8",
            errors="replace",
        )

        exit_code = result.get("StatusCode", 1)

        return {
            "result_label": (
                "executed" if exit_code == 0
                else "execution_failed"
            ),
            "exit_code": exit_code,
            "output": output,
            "output_truncated": truncated,
        }

    except SandboxExecutionError:
        raise

    except DockerException as exc:
        raise SandboxExecutionError(
            f"Docker execution error: {exc}"
        ) from exc

    finally:
        # Always remove the container and close the Docker client.
        if container is not None:
            try:
                container.remove(force=True)
            except DockerException:
                pass

        if client is not None:
            try:
                client.close()
            except DockerException:
                pass
