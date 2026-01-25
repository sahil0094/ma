from smart_investigator.foundation.tools.tool_set import construct_tools, get_endpoint_resources
from pathlib import Path
tools_config_path = (Path(file).resolve().parent / “tools_config”).resolve()
tools = construct_tools(tools_config_path)
endpoint_resources = get_endpoint_resources(tools_config_path)
