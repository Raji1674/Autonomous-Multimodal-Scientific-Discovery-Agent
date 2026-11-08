"""Load YAML config files plus .env; secrets are read only from the environment."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    provider: str = "gemini"
    model: str = "set-me"
    temperature: float = 0.2


class RetrievalConfig(BaseModel):
    top_k: int = 8
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "data/chroma"
    download_pdfs: bool = False
    max_pdf_papers: int = Field(default=3, ge=0)


class AgentsConfig(BaseModel):
    max_hypothesis_retries: int = 2
    max_revisions: int = 2
    checkpoint_path: str = "data/checkpoints/research.sqlite"


class AppConfig(BaseModel):
    models: ModelConfig = ModelConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    agents: AgentsConfig = AgentsConfig()


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_config(config_dir: str | Path = "configs", load_env: bool = True) -> AppConfig:
    """Read configs/*.yaml (missing files fall back to defaults)."""
    if load_env:
        load_dotenv()
    d = Path(config_dir)
    return AppConfig(
        models=ModelConfig(**_read(d / "models.yaml")),
        retrieval=RetrievalConfig(**_read(d / "retrieval.yaml")),
        agents=AgentsConfig(**_read(d / "agents.yaml")),
    )


def get_secret(name: str) -> str:
    """Read a secret from the environment; never from YAML."""
    value = os.environ.get(name, "")
    if not value:
        raise KeyError(f"Missing environment variable {name}; set it in your .env file")
    return value
