from datetime import date

import audeer

from docs.utils import (
    main,
)


# Project -----------------------------------------------------------------
author = 'H. Wierstorf, A. Derington'
copyright = f'{date.today().year} audEERING GmbH'
project = 'SER Tests'
# The x.y.z version read from tags
try:
    version = audeer.git_repo_version()
except Exception:
    version = '<unknown>'
title = project

# General -----------------------------------------------------------------
master_doc = 'index'
extensions = []
source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = None
extensions = [
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.katex',
    'matplotlib.sphinxext.plot_directive',
    'linuxdoc.rstFlatTable',
]

intersphinx_mapping = {
    'audbackend': ('https://audeering.github.io/audbackend/', None),
    'audeer': ('https://audeering.github.io/audeer/', None),
    'audformat': ('http://audeering.github.io/audformat/', None),
    'audiofile': ('https://audeering.github.io/audiofile/', None),
    'audobject': ('https://audeering.github.io/audobject/', None),
    'audresample': ('https://audeering.github.io/audresample/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'python': ('https://docs.python.org/3/', None),
}
linkcheck_ignore = [
    'https://gitlab.audeering.com',
]
html_static_path = ['_static']
html_extra_path = ['extra']
html_css_files = ['colors.css', 'math.css']
bibtex_bibfiles = ['refs.bib']

# Matplot plot_directive settings
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ['png']

# HTML --------------------------------------------------------------------
html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': False,
    'wide_pages': [],
    'footer_links': True,
}
html_context = {
    'display_gitlab': True,
}
html_title = title


# MAIN --------------------------------------------------------------------
main(html_theme_options)
