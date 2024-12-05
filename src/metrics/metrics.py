from typing import TypedDict


class Metrics(TypedDict):
    loss: float
    accuracy: float
    f1: float
    recall: float
    precision: float