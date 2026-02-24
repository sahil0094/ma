from typing import Any
from mlflow.genai.judges import make_judge


def create_conversation_judge(
    name: str,
    guidelines: str,
    feedback_value_type: Any,
    model: str = "azure:/gpt-4o"
):
    """
    Rubric-following conversation judge.

    Evaluates based on the full conversation context.
    Use for metrics like conversation_coherence, task_completion.
    """
    return make_judge(
        name=name,
        instructions=(
            f"{guidelines}\n\n"
            f"Evaluate the {name} based on the below:\n"
            "Conversation: {{conversation}}"
        ),
        feedback_value_type=feedback_value_type,
        model=model
    )


def create_input_output_judge(
    name: str,
    guidelines: str,
    feedback_value_type: Any,
    model: str = "azure:/gpt-4o"
):
    """
    Rubric-following input/output judge.

    Evaluates based on input/output pairs.
    Use for metrics like response_quality, tone_compliance, accuracy.
    """
    return make_judge(
        name=name,
        instructions=(
            f"{guidelines}\n\n"
            f"Evaluate the {name} based on the below:\n"
            "Inputs: {{inputs}}\n"
            "Outputs: {{outputs}}"
        ),
        feedback_value_type=feedback_value_type,
        model=model
    )


def create_tool_call_judge(
    name: str,
    guidelines: str,
    feedback_value_type: Any,
    model: str = "azure:/gpt-4o"
):
    """
    Rubric-following tool call/routing judge.

    Evaluates tool selection decisions specifically.
    Use for metrics like routing_plausibility, error_root_cause.

    Expects inputs to contain embedded tool context:
    {
        "request": <original user request>,
        "tools_called": [<list of tool names>],
        "tool_details": [<list of tool span dicts with name, inputs, outputs, status>]
    }
    """
    return make_judge(
        name=name,
        instructions=(
            f"{guidelines}\n\n"
            f"Evaluate the {name} based on the following:\n\n"
            "Input Context (contains request, tools_called, and tool_details): {{ inputs }}\n\n"
            "Agent Response: {{ outputs }}"
        ),
        feedback_value_type=feedback_value_type,
        model=model
    )
