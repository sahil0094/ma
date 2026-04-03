import uuid
import mlflow
from smart_investigator.foundation.agents.feature_response_agent import FeatureLanggraphResponsesAgent
from agents.external_agent.external_agent_graph import get_graph as get_external_agent_graph
from smart_investigator.foundation.tools.tool_names import EXTERNAL_AGENT_NAME
# TODO: replace with PPOA
from smart_investigator.foundation.llm.ppoa_azure_openai import get_llm
from smart_investigator.foundation.schemas.schemas import ResponseAgentCustomOutput, FrontendOutput
from smart_investigator.foundation.agents.agent_helpers import databricks_trace_input_output_filter,
databricks_trace_chat_model_stream_series_filter


class ExternalAgentResponsesAgent(FeatureLanggraphResponsesAgent):
    AgentName = EXTERNAL_AGENT_NAME

    def get_graph(self):
        return get_external_agent_graph(self.llm)


agent = ExternalAgentResponsesAgent()
# TODO: Register agent as an endpoint to databricks
mlflow.autolog(disable=False)
mlflow.openai.autolog()
mlflow.langchain.autolog()
mlflow.tracing.configure(span_processors=[
                         databricks_trace_input_output_filter, databricks_trace_chat_model_stream_series_filter])
mlflow.models.set_model(agent)
