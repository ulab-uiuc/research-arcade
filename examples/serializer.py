"""
This program tests the ability of the serializer along with the integration of latex parser
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_constructor.database import Database



db = Database()

tables = ["papers"]

series = db.serialize_all(tables=tables)

print(series)


