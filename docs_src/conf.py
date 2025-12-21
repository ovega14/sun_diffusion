# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sun_diffusion'
copyright = '2025, Gurtej Kanwar, Octavio Vega'
author = 'Gurtej Kanwar, Octavio Vega'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx_copybutton',
]

myst_enable_extensions = [
    'dollarmath'
]

autosummary_generate = True
napoleon_numpy_docstring = False
napoleon_google_docstring = True
#napoleon_custom_sections = [('Returns', 'params_style')]


autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = 'sun_diffusion'
html_theme = 'furo'

html_theme_options = {
    'source_repository': 'https://github.com/ovega14/sun_diffusion',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
    'top_of_page_buttons': ['view', 'edit'],
}

html_static_path = ['_static']
