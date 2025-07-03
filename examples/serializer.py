"""
This program tests the ability of the serializer along with the integration of latex parser
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_constructor.database import Database

from paper_collector.latex_parser import clean_latex_code, clean_latex_format



db = Database()

tables = ["sections"]

series = db.serialize_all(tables=tables, parser=clean_latex_code)

print("Paper parsed by the clean_latex_code")
print(series)

print()

print("--------------")
print()

series = db.serialize_all(tables=tables, parser=clean_latex_format)
print("Paper parsed by the clean_latex_format")
print(series)