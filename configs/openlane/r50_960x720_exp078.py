from __future__ import annotations

from copy import deepcopy

from configs.openlane.r50_960x720_exp076 import CONFIG as BASE_CONFIG


CONFIG = deepcopy(BASE_CONFIG)
CONFIG["runtime"]["data_probe"] = True
CONFIG["runtime"]["log_interval"] = 10
