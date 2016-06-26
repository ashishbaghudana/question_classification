from abc import abstractmethod

class FeatureGenerator(object):
    """
    Abstract Class for Feature Generator
    Each feature generator will be [Name]FeatureGenerator and will extend
    from this class
    """
    @abstractmethod
    def generate(self, dataset):
        pass
