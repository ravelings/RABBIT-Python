from logger import logger
import pinocchio

class Robot:

    def __init__(self, model: pinocchio.Model) -> None:
        self.model = model
        self.data = model.createData()

    def step(self):
        

    """
    def __init__(self, **kwargs):
        try:
            self.mass = kwargs.pop("mass")
            self.inertial = kwargs.pop("inertial")
            self.coriolis = kwargs.pop("coriolis")
            self.gravity = kwargs.pop("gravity")
        except KeyError as e:
            logger.info(f"KeyWordError: {str(e)}")

        for key, value in kwargs.items():
            setattr(self, str(key), value)
    """