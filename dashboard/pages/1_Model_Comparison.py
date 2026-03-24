"""Model Comparison — existing dashboard view."""
import sys
from pathlib import Path

# Add dashboard directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import main
main()
