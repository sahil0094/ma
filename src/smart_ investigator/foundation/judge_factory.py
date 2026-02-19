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
    Use for metrics like routing_plausibility.

    Expected template variables:
    - inputs: User's message/request
    - available_tools: Description of available tools
    - selected_tool: The tool that was selected
    - tool_arguments: Arguments passed to the tool
    """
    return make_judge(
        name=name,
        instructions=(
            f"{guidelines}\n\n"
            f"Evaluate the {name} based on the tool selection:\n"
            "User Input: {{inputs}}\n"
            "Available Tools:\n{{available_tools}}\n"
            "Selected Tool: {{selected_tool}}\n"
            "Tool Arguments: {{tool_arguments}}"
        ),
        feedback_value_type=feedback_value_type,
        model=model
    )
