import abc


class SelectorClassInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @property
    @abc.abstractmethod
    def feature_importances_(self):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is SelectorClassInterface:
            if any("fit" in B.__dict__ for B in C.__mro__) and any("predict" in B.__dict__ for B in C.__mro__) and any(
                    "feature_importances_" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
