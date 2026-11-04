
"""Security configuration for Docker-based experiment execution."""

# Container resource limits
MEMORY_LIMIT = "512m"
CPU_LIMIT = 0.5
PIDS_LIMIT = 64

# Execution timeout in seconds
EXECUTION_TIMEOUT = 90

# Docker security settings
NETWORK_MODE = "none"
READ_ONLY_ROOTFS = True
RUN_AS_NON_ROOT = True
DROP_ALL_CAPABILITIES = True
NO_NEW_PRIVILEGES = True

# Only use the pre-downloaded Python image.
DOCKER_IMAGE = "scientific-ml-sandbox:latest"

# Restrict container temporary storage.
TMPFS_SIZE = "64m"

# Maximum captured output size (bytes)
MAX_OUTPUT_BYTES = 100_000
