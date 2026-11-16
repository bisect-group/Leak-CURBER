from __future__ import annotations

import logging

from src.utils.rich_progress import get_component_logger, progress, track


def test_progress_iterates_and_manual_mode_is_safe_when_disabled() -> None:
    assert list(track(["a", "b"], description="items", disable=True)) == ["a", "b"]

    with progress(total=2, desc="manual", disable=True) as task:
        task.update()
        task.set_postfix(state="running")
        task.update()


def test_component_logger_keeps_plain_independent_files(tmp_path) -> None:
    left_path = tmp_path / "left.log"
    right_path = tmp_path / "right.log"
    left = get_component_logger("left", left_path, logging.INFO)
    right = get_component_logger("right", right_path, logging.INFO)
    left.info("left message")
    right.info("right message")
    get_component_logger("left", left_path).info("left once")

    assert "left message" in left_path.read_text()
    assert left_path.read_text().count("left once") == 1
    assert "right message" not in left_path.read_text()
    assert "right message" in right_path.read_text()
