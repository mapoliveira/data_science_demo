"""Creates a Jupyter notebook from a python driver.
$  python convert2iPynb.py the_driver.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import sys

notebook = new_notebook()
with open(sys.argv[1]) as f:
    code = f.read()

notebook.cells.append(new_code_cell(code))
nbformat.write(notebook, sys.argv[1]+'.ipynb')
