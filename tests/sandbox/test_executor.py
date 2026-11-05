
import pytest

from src.sandbox.executor import (
    execute_python,
    SandboxExecutionError,
    SandboxTimeoutError,
)


def test_successful_python_execution():
    result = execute_python("print('Hello from sandbox')")

    assert result["result_label"] == "executed"
    assert result["exit_code"] == 0
    assert "Hello from sandbox" in result["output"]


def test_invalid_empty_code():
    with pytest.raises(ValueError):
        execute_python("")


def test_invalid_code_type():
    with pytest.raises(ValueError):
        execute_python(None)


def test_python_execution_error():
    result = execute_python("raise ValueError('Test error')")

    assert result["result_label"] == "execution_failed"
    assert result["exit_code"] != 0
    assert "Test error" in result["output"]


def test_execution_timeout(monkeypatch):
    import src.sandbox.executor as sandbox

    monkeypatch.setattr(
        sandbox,
        "EXECUTION_TIMEOUT",
        2,
    )

    with pytest.raises(SandboxTimeoutError):
        execute_python("import time; time.sleep(10)")



def test_output_is_truncated(monkeypatch):
    import src.sandbox.executor as sandbox

    monkeypatch.setattr(sandbox, "MAX_OUTPUT_BYTES", 1024)

    result = execute_python(
        "print('A' * 10000)"
    )

    assert result["result_label"] == "executed"
    assert result["output_truncated"] is True
    assert len(result["output"].encode("utf-8")) <= 1024


def test_container_removed_after_execution():
    import docker

    client = docker.from_env()
    before = {
        container.id
        for container in client.containers.list(all=True)
    }

    try:
        execute_python("print('cleanup test')")

        after = {
            container.id
            for container in client.containers.list(all=True)
        }

        assert after == before
    finally:
        client.close()



def test_network_is_disabled():
    result = execute_python(
        """
import socket

try:
    socket.create_connection(("1.1.1.1", 80), timeout=2)
    print("NETWORK_AVAILABLE")
except OSError:
    print("NETWORK_BLOCKED")
"""
    )

    assert result["result_label"] == "executed"
    assert "NETWORK_BLOCKED" in result["output"]



def test_container_uses_expected_working_directory():
    result = execute_python(
        """
import os

print("WORKDIR:", os.getcwd())
print("ROOT_ENTRIES:", os.listdir("/"))
"""
    )

    assert result["result_label"] == "executed"
    assert "WORKDIR: /workspace" in result["output"]


def test_csv_is_mounted_read_only(tmp_path):
    csv_file = tmp_path / "students.csv"
    csv_file.write_text(
        "hours,passed\n5,1\n2,0\n",
        encoding="utf-8",
    )

    result = execute_python(
        """
from pathlib import Path
import pandas as pd

path = Path("/dataset.csv")

print("EXISTS:", path.is_file())
print("DATA:", pd.read_csv(path).shape)

try:
    path.write_text("modified")
    print("WRITE_ALLOWED")
except OSError:
    print("WRITE_BLOCKED")
""",
        dataset_path=str(csv_file),
    )

    assert result["result_label"] == "executed"
    assert "EXISTS: True" in result["output"]
    assert "DATA: (2, 2)" in result["output"]
    assert "WRITE_BLOCKED" in result["output"]
    assert "WRITE_ALLOWED" not in result["output"]

    assert csv_file.read_text(encoding="utf-8") == (
        "hours,passed\n5,1\n2,0\n"
    )


def test_csv_mount_rejects_non_csv_file(tmp_path):
    file_path = tmp_path / "students.txt"
    file_path.write_text("not a CSV", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV"):
        execute_python(
            "print('test')",
            dataset_path=str(file_path),
        )
