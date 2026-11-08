"""Standard-library logging setup for application modules."""

import logging

from lowparllm.core.config import AppConfig


def configure_logging(config: AppConfig) -> None:
    """Configure the root logger using the selected application log level."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
