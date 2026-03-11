




class BaseEvaluator:
    def __init__(self, **kwargs):
        pass

    def preload(self):
        pass

    def run(self):
        raise NotImplementedError("The run method should be implemented in the subclass.")