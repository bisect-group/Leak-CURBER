from __future__ import annotations

from typing import Any

__all__ = [
    "RankedLogger",
    "create_exp_name",
    "enforce_tags",
    "extras",
    "get_metric_value",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "print_config_tree",
    "task_wrapper",
]


def __getattr__(name: str) -> Any:
    if name in {"instantiate_callbacks", "instantiate_loggers"}:
        from src.utils.instantiators import instantiate_callbacks, instantiate_loggers

        return {
            "instantiate_callbacks": instantiate_callbacks,
            "instantiate_loggers": instantiate_loggers,
        }[name]
    if name in {"create_exp_name", "log_hyperparameters"}:
        from src.utils.logging_utils import create_exp_name, log_hyperparameters

        return {
            "create_exp_name": create_exp_name,
            "log_hyperparameters": log_hyperparameters,
        }[name]
    if name == "RankedLogger":
        from src.utils.pylogger import RankedLogger

        return RankedLogger
    if name in {"enforce_tags", "print_config_tree"}:
        from src.utils.rich_utils import enforce_tags, print_config_tree

        return {
            "enforce_tags": enforce_tags,
            "print_config_tree": print_config_tree,
        }[name]
    if name in {"extras", "get_metric_value", "task_wrapper"}:
        from src.utils.utils import extras, get_metric_value, task_wrapper

        return {
            "extras": extras,
            "get_metric_value": get_metric_value,
            "task_wrapper": task_wrapper,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
