try:
    from sybil import Sybil
    from sybil.parsers.markdown import PythonCodeBlockParser

    # Initialize Sybil with PythonCodeBlockParser
    pytest_collect_file = Sybil(
        parsers=[
            PythonCodeBlockParser(),
        ],
        patterns=["*.md"],
    ).pytest()
except ImportError:
    pass
