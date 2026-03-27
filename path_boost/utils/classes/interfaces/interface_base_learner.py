import abc


class BaseLearnerClassInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        if len(args) < 2:
            raise TypeError("fit() method requires at least two positional arguments (X and y)")

        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        if len(args) < 1:
            raise TypeError("fit() method requires at least one positional argument (X)")

        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is BaseLearnerClassInterface:
            if any("fit" in B.__dict__ for B in C.__mro__) and any("predict" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
