from smart_investigator.foundation.agents.response_agent import LanggraphResponsesAgent, CheckpointerConfig
from smart_investigator.foundation.llm.ppo2a_azure_openai import get_llm
from smart_investigator.foundation.llm.human_in_the_loop import human_in_the_loop, prepare_hitl_task
from smart_investigator.foundation.settings import MASTER_AGENT_NAME
from smart_investigator.foundation.schemas import ResponseAgentCustomOutput, FrontendOutput
from agents.orchestrator.master_agent_graph import get_master_agent_graph
from agents.orchestrator.master_agent_utils import MasterAgentContext
from agents.orchestrator.tools_set import tools as agent_tools
from langgraph_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.graph import StateGraph
from mlflow.pyfunc import PythonModelContext
from typing import List, TypedDict, Optional, Annotated, Union, Any, NamedTuple, Generator
import mlflow
import uuid
from pathlib import Path


class MasterAgentResponsesAgent(LanggraphResponsesAgent):
    AgentName = MASTER_AGENT_NAME

    def get_graph(self) -> StateGraph:
        return get_master_agent_graph(self.llm, tools=agent_tools + [human_in_the_loop])

    # def _prepare_custom_response_agent_stream_event(self, input_dict: dict) -> ResponseAgentCustomOutput:
    #     return input_dict


agent = MasterAgentResponsesAgent()
# TODO: Register agent as a model in databricks
mlflow.langchain.autolog()
mlflow.models.set_model(agent)
