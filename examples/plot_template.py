"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`extended_path_boost.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt

from extended_path_boost import PathBoost

X = np.arange(100).reshape(100, 1)
y = np.zeros((100,))
estimator = PathBoost()
estimator.fit(X, y)
plt.plot(estimator.predict(X))
plt.show()
