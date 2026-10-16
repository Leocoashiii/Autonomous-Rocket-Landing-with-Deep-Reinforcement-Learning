"""Visualization entrypoint.

This module delegates plotting to utils.plotting so the evaluation package owns
result presentation while utils only provides reusable helpers.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting import main


if __name__ == "__main__":
    main()
