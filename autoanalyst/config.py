from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AUTOANALYST_", extra="ignore", populate_by_name=True
    )

    max_file_mb: float = 50.0
    max_rows: int = 200_000
    min_rows: int = 20
    seed: int = 42
    runs_dir: Path = Path("runs")
    retention_days: int = 7

    # profiling and fallback plan
    onehot_max_levels: int = 15
    ordinal_max_levels: int = 50
    id_unique_ratio: float = 0.95
    mostly_missing_frac: float = 0.6
    detect_ratio: float = 0.9
    outlier_iqr_factor: float = 3.0

    # validators
    max_row_drop_frac: float = 0.10
    max_mean_shift_std: float = 0.25
    min_std_ratio: float = 0.6
    max_std_ratio: float = 1.5
    max_category_tvd: float = 0.2
    leakage_corr: float = 0.98
    holdout_frac: float = 0.25

    # training
    test_size: float = 0.2
    cv_folds: int = 5
    search_iter: int = 10
    n_jobs: int = -1
    importance_repeats: int = 5
    max_classes: int = 50
    overfit_gap: float = 0.15

    # report
    convert_pdf: bool = True
    soffice_path: str | None = None
    pdf_timeout: int = 180

    # llm
    llm_provider: Literal["hosted", "ollama", "mock"] = "hosted"
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "AUTOANALYST_GEMINI_API_KEY")
    )
    hosted_model: str = "gemini-flash-latest"

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _empty_key_is_missing(cls, value):
        return value or None

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_num_ctx: int = 8192
    llm_temperature: float = 0.1
    llm_max_retries: int = 2
    llm_timeout: int = 180
    debate_max_rounds: int = 3
    fix_max_rounds: int = 3
    writer_max_attempts: int = 3
    cache_dir: Path | None = None  # defaults to <runs_dir>/_cache
    use_critic: bool = True
    max_workers: int = 2
    inline_jobs: bool = False  # run steps in the request thread (tests, debugging)

    # eda
    strong_corr: float = 0.9
    minority_share: float = 0.1


@lru_cache
def get_settings() -> Settings:
    return Settings()
