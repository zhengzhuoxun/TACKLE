"""
Central configuration for the TableQA agentic pipeline.

Loads settings from three sources (highest priority last):
  1. Default values (hard-coded in dataclasses).
  2. YAML pipeline config  (config/pipeline.yaml).
  3. .env file             (LLM credentials, provider, model, etc.).
  4. Programmatic overrides (via init_config(**overrides)).

Usage:
    from config.settings import init_config, get_config

    init_config()             # load from .env + YAML
    cfg = get_config()
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Load .env (must happen before anything else that reads os.environ)
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    """Load .env into os.environ.  Silently skips if python-dotenv or the
    file is missing."""
    try:
        from dotenv import load_dotenv as _ld
        # Look for .env in the project root (parent of config/)
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            _ld(dotenv_path=str(env_path), override=False)
    except ImportError:
        pass  # python-dotenv not installed — rely on real env vars

_load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _env(key: str, default: str = "") -> str:
    """Read a string from the environment with a TABLEQA_ prefix."""
    val = os.environ.get(f"TABLEQA_{key}", "")
    return val if val else default


def _env_first(keys: list[str], default: str = "") -> str:
    """Read the first non-empty value from a list of env keys."""
    for key in keys:
        val = os.environ.get(key, "")
        if val:
            return val
    return default


def _env_float(key: str, default: float = 0.0) -> float:
    val = _env(key, str(default))
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(key: str, default: int = 0) -> int:
    val = _env(key, str(default))
    try:
        return int(val)
    except ValueError:
        return default


_DEFAULT_YAML_PATH = Path(__file__).resolve().parent / "pipeline.yaml"


def _yaml_path() -> Path:
    """Resolve the pipeline YAML path (TABLEQA_CONFIG overrides the default)."""
    override = os.environ.get("TABLEQA_CONFIG", "").strip()
    if override:
        return Path(override).resolve()
    return _DEFAULT_YAML_PATH


def _load_yaml() -> dict[str, Any]:
    """Load the pipeline YAML config and return its contents as a dict."""
    path = _yaml_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """LLM provider configuration — populated from .env."""
    provider: str = "openai"
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.0
    max_completion_tokens: int = 4096
    # Embeddings can come from a different provider than chat (e.g. chat via
    # DeepSeek, which has no embeddings endpoint, embeddings via OpenAI).
    # Left blank ("") to mean "let LLMClient pick a sensible provider/model
    # default"; explicit values here always win.
    embedding_provider: str = ""
    embedding_api_key: str = ""
    embedding_api_base: str = ""

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = _env("LLM_PROVIDER", "openai").strip().lower() or "openai"
        return cls(
            provider=provider,
            model=_env("LLM_MODEL", "gpt-4o"),
            embedding_model=_env("EMBEDDING_MODEL", ""),
            api_key=_env_first(
                [
                    "TABLEQA_LLM_API_KEY",
                    f"TABLEQA_{provider.upper()}_API_KEY",
                    f"{provider.upper()}_API_KEY",
                ],
                "",
            ),
            api_base=_env_first(
                [
                    "TABLEQA_LLM_API_BASE",
                    f"TABLEQA_{provider.upper()}_API_BASE",
                    f"{provider.upper()}_API_BASE",
                ],
                "",
            ),
            temperature=_env_float("LLM_TEMPERATURE", 0.0),
            max_completion_tokens=_env_int("LLM_MAX_TOKENS", 4096),
            embedding_provider=_env("EMBEDDING_PROVIDER", "").strip().lower(),
            embedding_api_key=_env("EMBEDDING_API_KEY", ""),
            embedding_api_base=_env("EMBEDDING_API_BASE", ""),
        )


@dataclass
class DataConfig:
    """Data paths configuration."""
    data_dir: str = "data/MMQA"
    two_table_file: str = "Synthesized_two_table.json"
    three_table_file: str = "Synthesized_three_table.json"
    merged_three_table_file: str = "merged_three_table.json"


@dataclass
class PipelineConfig:
    """Pipeline behaviour — populated from config/pipeline.yaml."""
    dataset: str = "three_table"  # two_table | three_table | merged_three_table | dirty_three_table | dirty_merged_three_table | self | merged_self
    max_samples: int = 10                # -1 for all
    output_dir: str = "outputs"
    save_intermediate: bool = True
    verbose: bool = False
    item_id: int = 0                     # specific item to run in single-item mode
    start_item: int | None = None        # inclusive item.id_ start value
    end_item: int | None = None          # inclusive item.id_ end value
    start_item_id: int | None = None     # legacy alias for inclusive item.id_ start value
    end_item_id: int | None = None       # legacy alias for inclusive item.id_ end value
    random_num_items: int | None = None  # if > 0, randomly sample this many items (overrides start/end)
    phase: str = "schema"                 # "cluster" | "merge" | "relation" | "schema" | "instance" | "execute"
    direct_cypher: bool = False          # use direct NL→Cypher translation (single LLM call)
    cypher_repair: bool = False          # enable the error-repair loop in the two-step Cypher pipeline
    cypher_repair_attempts: int = 3      # max repair attempts for the two-step Cypher pipeline

    @classmethod
    def from_yaml(cls, yaml_data: dict[str, Any] | None = None) -> "PipelineConfig":
        """Create PipelineConfig from YAML dict (or load it)."""
        if yaml_data is None:
            yaml_data = _load_yaml()
        return cls(
            dataset=yaml_data.get("dataset", "three_table"),
            max_samples=int(yaml_data.get("max_samples", 10)),
            output_dir=yaml_data.get("output_dir", "outputs"),
            save_intermediate=bool(yaml_data.get("save_intermediate", True)),
            verbose=bool(yaml_data.get("verbose", False)),
            item_id=int(yaml_data.get("item_id", 0)),
            start_item=(
                int(yaml_data["start_item"])
                if yaml_data.get("start_item") is not None
                else None
            ),
            end_item=(
                int(yaml_data["end_item"])
                if yaml_data.get("end_item") is not None
                else None
            ),
            start_item_id=(
                int(yaml_data["start_item_id"])
                if yaml_data.get("start_item_id") is not None
                else None
            ),
            end_item_id=(
                int(yaml_data["end_item_id"])
                if yaml_data.get("end_item_id") is not None
                else None
            ),
            random_num_items=(
                int(yaml_data["random_num_items"])
                if yaml_data.get("random_num_items") is not None
                else None
            ),
            phase=yaml_data.get("phase", "schema"),
            direct_cypher=bool(
                yaml_data.get("direct_cypher", False)
            ),
            cypher_repair=bool(
                yaml_data.get("cypher_repair", False)
            ),
            cypher_repair_attempts=int(
                yaml_data.get("cypher_repair_attempts", 3)
            ),
        )


@dataclass
class Config:
    """Root configuration aggregator."""
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    data: DataConfig = field(default_factory=DataConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig.from_yaml)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the global config singleton.  Call init_config() first."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def init_config(**overrides: Any) -> Config:
    """
    Initialise the global config from .env and YAML, then apply any
    programmatic overrides.

    Override keys can be nested, e.g.:
        init_config(pipeline={"max_samples": 50}, llm={"model": "gpt-3.5-turbo"})
    """
    global _config
    _config = Config()

    for section, values in overrides.items():
        if hasattr(_config, section):
            target = getattr(_config, section)
            for k, v in values.items():
                if hasattr(target, k):
                    setattr(target, k, v)
    return _config


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
KGQA_PHASES: dict[str, int] = {
    "cluster": 1, "clusters": 1, "c": 1,
    "merge": 2, "m": 2,
    "relation": 3, "relations": 3, "rel": 3, "r": 3,
    "schema": 4, "s": 4,
    "instance": 5, "ins": 5, "a": 5,
    "execute": 6, "full": 6, "f": 6,
}


def _resolve_phase(phase: str | int) -> int:
    """Resolve a phase name or int to a step number (1-6).

    Used by SemanticGraphBuilder and PipelineOrchestrator.
    """
    if isinstance(phase, int):
        return max(1, min(6, phase))
    key = str(phase).lower().strip()
    return KGQA_PHASES.get(key, 5)
