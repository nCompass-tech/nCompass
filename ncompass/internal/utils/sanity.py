from typing import TypeVar, List
import ncompass.internal.logging as nclog

T = TypeVar("T")
def validate_arg(arg: T, valid_args: List[T]) -> None:
    if arg not in valid_args:
        nclog.ERROR(\
                  f"Invalid argument {arg}. Valid arguments are {valid_args}"\
                , ValueError)

