from typing import Literal, Dict, Any
from langchain_core.tools import tool
import requests
import json
import os
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from requests.exceptions import HTTPError
from json import JSONDecodeError
from openai import RateLimitError  # or wherever you're Importing this from
import httpx
from smart_investigator.foundation.utils.configs_utils import namedtuple_to_dict, get_nginx_configs, SecretNames
# from smart_investigator.foundation.utils.utils import generate_sp_token
from databricks.sdk import WorkspaceClient
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from langgraph.config import get_stream_writer

ERROR_SENTINELS = {
    "an error occurred",
    "scale to zero"
}


def _is_downstream_error(answer: str) -> bool:
    """
    Heuristic detection of semantic (soft) downstream errors.
    """
    if not answer or not isinstance(answer, str):
        return True

    answer_l = answer.lower().strip()

    if any(s in answer_l for s in ERROR_SENTINELS):
        return True

    return False


def _extract_text(response):
    texts = []
    for msg in response.output or []:
        for item in msg.content or []:
            if hasattr(item, "text"):
                texts.append(item.text)
    return "\n".join(texts)


max_retries = 3


@tool("query_investigation_processes",
      description="Retrieve fraud-type specific investigation processes: aims, objectives, and example interview questions.")
def query_investigation_processes(endpoint_name: str, query: str, filters: dict):
    w = WorkspaceClient()

    client = w.serving_endpoints.get_open_ai_client()

    custom_inputs = {
        "artifact": filters,
        "is_direct": True,
        "agent_name": "InterviewPlanAgent",
        "top_k": 10
    }

    payload = [{"type": "message",
                "id": "InterviewPlanAgent",
                "content": [
                    {
                        "type": "output_text",
                        "text": query,
                        "custom_inputs": custom_inputs
                    }
                ]}]

    @retry(
        retry=retry_if_exception_type((RuntimeError, ValueError)),
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential_jitter(initial=10, max=60, jitter=1)
    )
    def _get_response():
        try:
            resp = client.responses.create(
                model=endpoint_name,
                input=payload,
                stream=False,
                extra_body={
                    "user": "ip_agent",
                    "custom_inputs": {
                        "state": {}
                    },
                    "databricks_options": dict(return_trace=True),
                }
            )
        except Exception as e:
            raise RuntimeError(
                f"[query_investigation_processes] Unexpected error calling downstream endpoint"
            ) from e

        try:
            answer = resp.output[0].content[0].text
        except Exception as e:
            raise ValueError(f"Failed to parse response object: {resp}")

        if _is_downstream_error(answer):
            raise RuntimeError(
                f"[query_investigation_processes] Downstream endpoint returned an error response: {resp}. query: {query}, filters: {filters}")

        return answer

    answer = _get_response()

    tool_output: Dict[str, Any] = {
        "knowledge": [
            {
                "query": query,
                "answer": answer
            }
        ]
    }

    return tool_output
