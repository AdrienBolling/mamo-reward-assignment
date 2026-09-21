"""Statistics and figures computed from run records.

This layer reads run records and writes tables and figures. The code stays in
this repository. The artifacts go to the paper repository under ``paper/``.

This layer imports the record schema only. It never imports an agent or an
environment, so the analysis runs without a GPU.
"""
