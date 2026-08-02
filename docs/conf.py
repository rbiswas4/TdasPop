import os
import sys

sys.path.insert(0, os.path.abspath('..'))

from tdaspop.version import __VERSION__

project = 'TdasPop'
copyright = '2026, R. Biswas'
author = 'R. Biswas'
version = __VERSION__
release = __VERSION__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Napoleon settings: this codebase uses numpy-style docstrings
# (Parameters/Returns sections), not Google-style.
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_mock_imports = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
