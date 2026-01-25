from pathlib import Path, PosixPath
import yaml
import sys

from smart_investigator.foundation.tools.tool_set_utils import resolve_endpoint_path
from smart_investigator.foundation.tools.tool_factory import ToolFactory
from smart_investigator.foundation.utils.configs_utils import get_nginx_configs
from langchain_core.tools import StructuredTool
from mlflow.models.resources import DatabricksServingEndpoint

# CONFIG_DIR = (Path(__file__).resolve().parent / "tools_config").resolve()
# print(CONFIG_DIR)

def construct_tools(path: PosixPath) -> StructuredTool:
    tools, errors = [], []

    for yml in path.glob("*.y*ml"):
        print(yml)
        try:
            with open(yml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                agent_cfg = cfg["agent"]
                agent_name = agent_cfg["name"]
                endpoint = agent_cfg["endpoint"]
                tool = ToolFactory(None, agent_cfg).create_tool()
                tools.append(tool)

        except Exception as e:
            errors.append((str(yml), repr(e)))
            print(f"[tool_set] ERROR in {yml}: {e}")
    return tools


def get_endpoint_resources(path: PosixPath):
    endpoint_resources, errors = [], []
    for yml in path.glob("*.y*ml"):
        print(yml)
        try:
            with open(yml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                agent_cfg = cfg["agent"]
                endpoint_name = agent_cfg["endpoint_name"]
                endpoint_resources.append(
                    DatabricksServingEndpoint(endpoint_name=endpoint_name)
                )
        except Exception as e:
            errors.append((str(yml), repr(e)))
            print(f"[tool_set] ERROR in {yml}: {e}")

    return endpoint_resources
