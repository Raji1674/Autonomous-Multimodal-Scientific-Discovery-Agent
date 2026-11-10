
import docker
import pytest

import src.sandbox.executor as sandbox


class FakeContainer:
    def __init__(self):
        self.removed = False
        self.killed = False

    def wait(self, timeout=None):
        return {"StatusCode": 0}

    def logs(self, **kwargs):
        return iter([b"Mock execution successful"])

    def remove(self, force=False):
        self.removed = True

    def kill(self):
        self.killed = True


class FakeContainers:
    def __init__(self, container=None, error=None):
        self.container = container or FakeContainer()
        self.error = error
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        if self.error:
            raise self.error
        return self.container


class FakeDockerClient:
    def __init__(self, containers):
        self.containers = containers
        self.closed = False

    def close(self):
        self.closed = True


def test_reject_oversized_code(monkeypatch):
    monkeypatch.setattr(sandbox, "MAX_OUTPUT_BYTES", 10)

    with pytest.raises(ValueError, match="exceeds"):
        sandbox.execute_python("print('This code is too long')")


def test_reject_missing_dataset():
    with pytest.raises(FileNotFoundError):
        sandbox.execute_python(
            "print('test')",
            dataset_path="missing_dataset.csv",
        )


def test_reject_non_csv_dataset(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("sample data")

    with pytest.raises(ValueError, match="Only CSV"):
        sandbox.execute_python("print('test')", file_path)


def test_reject_dataset_directory(tmp_path):
    with pytest.raises(ValueError, match="file"):
        sandbox.execute_python("print('test')", tmp_path)


def test_reject_dataset_over_100mb(tmp_path):
    file_path = tmp_path / "large.csv"

    with file_path.open("wb") as file:
        file.truncate(100 * 1024 * 1024 + 1)

    with pytest.raises(ValueError, match="100 MB"):
        sandbox.execute_python("print('test')", file_path)


def test_docker_unavailable(monkeypatch):
    def fail_connection():
        raise docker.errors.DockerException("Docker unavailable")

    monkeypatch.setattr(sandbox.docker, "from_env", fail_connection)

    with pytest.raises(sandbox.SandboxExecutionError, match="Docker"):
        sandbox.execute_python("print('test')")


def test_container_start_failure_closes_client(monkeypatch):
    containers = FakeContainers(
        error=docker.errors.DockerException("Container startup failed")
    )
    client = FakeDockerClient(containers)

    monkeypatch.setattr(sandbox.docker, "from_env", lambda: client)

    with pytest.raises(sandbox.SandboxExecutionError):
        sandbox.execute_python("print('test')")

    assert client.closed is True


def test_security_options_are_passed(monkeypatch):
    container = FakeContainer()
    containers = FakeContainers(container=container)
    client = FakeDockerClient(containers)

    monkeypatch.setattr(sandbox.docker, "from_env", lambda: client)

    result = sandbox.execute_python("print('test')")

    options = containers.run_kwargs

    assert options["network_mode"] == "none"
    assert options["mem_limit"] == sandbox.MEMORY_LIMIT
    assert options["pids_limit"] == sandbox.PIDS_LIMIT
    assert options["read_only"] is True
    assert options["cap_drop"] == ["ALL"]
    assert options["security_opt"] == ["no-new-privileges:true"]
    assert options["user"] == "65534:65534"
    assert result["result_label"] == "executed"


def test_container_removed_and_client_closed(monkeypatch):
    container = FakeContainer()
    client = FakeDockerClient(FakeContainers(container=container))

    monkeypatch.setattr(sandbox.docker, "from_env", lambda: client)

    sandbox.execute_python("print('test')")

    assert container.removed is True
    assert client.closed is True
