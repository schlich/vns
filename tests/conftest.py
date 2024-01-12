import json
from pathlib import Path


def pytest_sessionfinish(exitstatus: int) -> None:
    workspace = Path("vns.code-workspace")

    test_result = "Material Theme DeepForest High Contrast" if exitstatus == 0 else "Red"

    with Path.open(workspace) as file:
        data = json.load(file)

    data["settings"]["workbench.colorTheme"] = test_result
    with Path.open(workspace) as file:
        json.dump(data, file)
