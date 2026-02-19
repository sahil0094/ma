from typing import Tuple, Type, Dict, List
from smart_investigator.foundation.tools.interview_plan.evaluation import (
    create_input_output_judge
)
from smart_investigator.foundation.tools.interview_plan.evaluation.prompt_manager.llm_judge_prompts import (
    ACCURACY_GUIDELINES,
    COMPLETENESS_GUIDELINES,
    LANGUAGE_GUIDELINES,
    COMPLIANCE_UNDERWRITING_GUIDELINES,
    COMPLIANCE_FINANCIAL_HISTORY_GUIDELINES,
)

_METRIC_REGISTRY: Dict[str, Tuple[str, Type]] = {
    "accuracy": (ACCURACY_GUIDELINES, int),
    "completeness": (COMPLETENESS_GUIDELINES, int),
    "language": (LANGUAGE_GUIDELINES, int),
    "compliance_underwriting": (COMPLIANCE_UNDERWRITING_GUIDELINES, bool),
    "compliance_financial_history": (COMPLIANCE_FINANCIAL_HISTORY_GUIDELINES, bool),
}


def build_judges(metric_ids: List[str], model: str = "azure:/gpt-4o"):
    """
    Build MLflow LLM judges from metric IDs.
    """

    judges = []
    unknown = [m for m in metric_ids if m not in _METRIC_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown metric ids: {unknown}. Known: {list(_METRIC_REGISTRY)}")

    for metric_id in metric_ids:
        guidelines, value_type = _METRIC_REGISTRY[metric_id]
        judges.append(
            create_input_output_judge(
                name=metric_id,
                guidelines=guidelines,
                feedback_value_type=value_type,
                model=model,
            )
        )
    return judges
