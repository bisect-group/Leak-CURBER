import sys
from types import ModuleType

import pytest

from tests.helpers.builders import load_script


class _FakeProgres:
    """Counts C-alpha rows as the number of lines in the structure file."""

    @staticmethod
    def read_coords(path, fileformat, res_range):
        text = open(path).read()
        if text == "corrupt":
            raise ValueError("cannot parse")
        return text.split()

    @staticmethod
    def coords_to_graph(coords):
        if coords[0] == "nograph":
            raise RuntimeError("no graph")


@pytest.fixture
def script(monkeypatch):
    package = ModuleType("progres")
    package.progres = _FakeProgres
    monkeypatch.setitem(sys.modules, "progres", package)
    monkeypatch.setitem(sys.modules, "progres.progres", _FakeProgres)
    monkeypatch.delitem(sys.modules, "filter_progres_structurelist", raising=False)
    module = load_script("filter_progres_structurelist")
    yield module
    sys.modules.pop("filter_progres_structurelist", None)


def _task(line, min_ca=3, skip_graph=False):
    return 7, line, "guess", min_ca, skip_graph


def test_split_input_line_keeps_the_note_whole(script):
    assert script.split_input_line("a.pdb dom some note\n") == ("a.pdb", "dom", "some note")
    assert script.split_input_line("a.pdb dom\n") == ("a.pdb", "dom", "")
    assert script.split_input_line("a.pdb\n") is None


def test_reject_to_tsv_line_flattens_tabs_and_newlines(script):
    reject = script.RejectRecord(3, "f", "d", "progres_error", "a\tb\nc")
    assert script.reject_to_tsv_line(reject) == "3\tf\td\tprogres_error\ta b c\n"


def test_count_lines_counts_newlines(tmp_path, script):
    path = tmp_path / "x.txt"
    path.write_text("a\nb\nc\n")
    assert script.count_lines(str(path)) == 3


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("  \n", "malformed_line"),
        ("only_one_column\n", "malformed_line"),
        ("nope.pdb d\n", "missing_file"),
    ],
)
def test_evaluate_line_rejects_unusable_rows(script, line, reason):
    _, keep, reject = script.evaluate_line(_task(line))
    assert keep is None
    assert reject.reason == reason
    assert reject.line_no == 7


def test_evaluate_line_keeps_a_valid_row_with_its_note(script, tmp_path):
    path = tmp_path / "a.pdb"
    path.write_text("x y z")
    _, keep, reject = script.evaluate_line(_task(f"{path}  dom   a note\n"))
    assert keep == f"{path} dom a note"
    assert reject is None


def test_evaluate_line_rejects_structures_with_too_few_ca(script, tmp_path):
    path = tmp_path / "a.pdb"
    path.write_text("x y")
    _, keep, reject = script.evaluate_line(_task(f"{path} dom\n"))
    assert keep is None
    assert (reject.reason, reject.details) == ("too_few_ca", "n_ca=2")


def test_evaluate_line_reports_parse_errors(script, tmp_path):
    path = tmp_path / "a.pdb"
    path.write_text("corrupt")
    _, _, reject = script.evaluate_line(_task(f"{path} dom\n"))
    assert reject.reason == "progres_error"
    assert "cannot parse" in reject.details


def test_evaluate_line_runs_the_graph_check_unless_skipped(script, tmp_path):
    path = tmp_path / "a.pdb"
    path.write_text("nograph b c")
    assert script.evaluate_line(_task(f"{path} dom\n"))[2].reason == "progres_error"
    assert script.evaluate_line(_task(f"{path} dom\n", skip_graph=True))[1] == f"{path} dom"


def test_main_splits_rows_into_filtered_and_excluded_lists(script, tmp_path, monkeypatch):
    good = tmp_path / "good.pdb"
    good.write_text("x y z")
    short = tmp_path / "short.pdb"
    short.write_text("x")
    listing = tmp_path / "in.txt"
    listing.write_text(f"{good} g\n{short} s\nmissing.pdb m\n")
    out = tmp_path / "out" / "kept.txt"
    argv = ["x", "-i", str(listing), "-o", str(out), "--workers", "1", "--no-progress"]
    monkeypatch.setattr(sys, "argv", argv)
    assert script.main() == 0
    assert out.read_text() == f"{good} g\n"
    bad = (tmp_path / "out" / "kept.txt.excluded.tsv").read_text().splitlines()
    assert bad[0] == "line_no\tfile_path\tdomain_id\treason\tdetails"
    assert [row.split("\t")[3] for row in bad[1:]] == ["too_few_ca", "missing_file"]


@pytest.mark.parametrize(("flag", "value"), [("--workers", "0"), ("--chunksize", "0")])
def test_main_rejects_non_positive_pool_settings(script, tmp_path, monkeypatch, flag, value):
    listing = tmp_path / "in.txt"
    listing.write_text("")
    argv = ["x", "-i", str(listing), "-o", str(tmp_path / "o.txt"), flag, value]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match=">= 1"):
        script.main()
