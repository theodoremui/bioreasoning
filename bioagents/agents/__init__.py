from loguru import logger
import sys
import os

# Configure logger with reduced verbosity to avoid JSON serialization issues
logger.remove()
logger.add(
    sink=sys.stderr,
    format=("<green>{time:HH:mm:ss}</green>|" "<cyan>{name}</cyan>:<bold><yellow>{function}</yellow></bold>:" "<cyan>{line}</cyan>-<white>{message}</white>"),
    level=os.environ.get("LOG_LEVEL", "INFO")  # Default to INFO level to avoid debug issues
)

