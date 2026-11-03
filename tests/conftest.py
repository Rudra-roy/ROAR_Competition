import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("competition_code", "tools"):
    sys.path.insert(0, os.path.join(root, d))
