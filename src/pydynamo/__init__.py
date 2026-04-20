"""pydynamo — minimal Python port of DYNAM-O.

Exposes:
    run_dynamo(data, fs, stage_times, stage_vals, ...) -> DynamoOutput

Stage convention (DYNAM-O): 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake — reversed from
most EDF stagers. Histograms default to NREM (stages 1, 2, 3).
"""

from pydynamo.defaults import DetectionOpts, BaselineOpts, SOPHOpts
from pydynamo.pipeline import run_dynamo, DynamoOutput

__all__ = ["run_dynamo", "DynamoOutput", "DetectionOpts", "BaselineOpts", "SOPHOpts"]
__version__ = "0.1.0"
