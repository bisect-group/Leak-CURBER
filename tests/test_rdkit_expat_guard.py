import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

ROOT = Path(__file__).resolve().parents[1]


def test_xml_parse_after_rdkit_draw_does_not_crash():
    code = (
        "import src; from rdkit.Chem import Draw; "
        "import xml.dom.minidom as m; m.parseString('<a/>')"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
