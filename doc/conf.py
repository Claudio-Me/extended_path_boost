extended_path_boost# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as get_version

project = "Scikit-learn Project Template"
copyright = "2016, V. Birodkar"
author = "V. Birodkar"
release = get_version('extended_path_boost')
version = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx-prompt",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_style = "css/extended_path_boost.css"
html_logo = "_static/img/logo.png"
# html_favicon = "_static/img/favicon.ico"
html_css_files = [
    "css/extended_path_boost.css",
]
html_sidebars = {
    "quick_start": [],
    "user_guide": [],
    "auto_examples/index": [],
}

html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/scikit-learn-contrib/extended_path_boost",
    # "twitter_url": "https://twitter.com/pandas_dev",
    "use_edit_page_button": True,
    "show_toc_level": 1,
    # "navbar_align": "right",  # For testing that the navbar items align properly
}

html_context = {
    "github_user": "scikit-learn-contrib",
    "github_repo": "extended_path_boost",
    "github_version": "master",
    "doc_path": "doc",
}

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# -- Options for intersphinx --------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "scikit-learn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
}

# -- Options for sphinx-gallery -----------------------------------------------

# Generate the plot for the gallery
plot_gallery = True

sphinx_gallery_conf = {
    "doc_module": "extended_path_boost",
    "backreferences_dir": os.path.join("generated"),
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "reference_url": {"extended_path_boost": None},
}
