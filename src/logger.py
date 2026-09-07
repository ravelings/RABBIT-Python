import logging
from pathlib import Path
from datetime import datetime

LOG_TO_FILE = False

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger.propagate = False # avoids duplicate messages

if LOG_TO_FILE:
    current_dir = Path(__file__).resolve().parent 
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    now = (datetime.now()).strftime("%H-%M-%S")
    log_path = log_dir / f"{logger.name}_log_{now}.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

stream_handler= logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(stream_handler)