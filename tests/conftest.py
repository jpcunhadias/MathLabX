import sys
from unittest.mock import MagicMock

# Mock heavy UI/plotting modules so lab imports stay lightweight in tests.
MOCK_MODULES = {
    "streamlit": MagicMock(),
    "matplotlib": MagicMock(),
    "matplotlib.pyplot": MagicMock(),
    "matplotlib.patches": MagicMock(),
    "mpl_toolkits": MagicMock(),
    "mpl_toolkits.mplot3d": MagicMock(),
    "plotly": MagicMock(),
    "plotly.graph_objects": MagicMock(),
}

for name, mock in MOCK_MODULES.items():
    sys.modules.setdefault(name, mock)
