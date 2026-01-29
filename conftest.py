from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser
import pytest

# Initialize Sybil with PythonCodeBlockParser
pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(),
    ],
    patterns=["*.md"],
).pytest()
