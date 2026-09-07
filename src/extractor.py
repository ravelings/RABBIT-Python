import pinocchio
from src.logger import logger
from pathlib import Path

def pinocchio_extractor(urdf_path: Path) -> pinocchio.Model | None:
    try: 
        model = pinocchio.buildModelFromUrdf(urdf_path)
        logger.info(f"Model name: {model.name}")
        return model
    except Exception as e:
        logger.info(f"Error when Extracting URDF: {str(e)}")
    

