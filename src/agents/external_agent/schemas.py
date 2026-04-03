from pydantic import BaseModel, Field
from typing import List, TypedDict, Optional, Annotated, Union, Any, Literal, Tuple, Dict
from smart_investigator.foundation.schemas.schemas import SmartInvestigatorAgentState, SIErrorCode
from datetime import datetime
from langgraph.graph import StateGraph, START, END, MessagesState


class SmartInvestigatorAgentState(MessagesState):
    artifact: dict[str, Any]
    resume: bool


class Reference(BaseModel):
    reference: str
    source: str


class InterviewObjective(BaseModel):
    objective_title: str = Field(
        description="Brief 1-4 word label"
    )
    objective: str
    supporting_evidence: Optional[List[Reference]] = None


class InterviewStrategy(BaseModel):
    aim: str
    objectives: List[InterviewObjective]
    update_notes: Optional[str] = None


class InterviewQuestion(BaseModel):
    question_id: int
    category: str
    question_text: str


class InterviewQuestionSets(BaseModel):
    question_sets: List[InterviewQuestion]


class KeyConcern(BaseModel):
    concern: str
    rationale: str


class KeyConcernSet(BaseModel):
    concern_set: List[KeyConcern]


class DocRequest(BaseModel):
    doc_type: str
    doc_details: str


class DocRequestSet(BaseModel):
    document_set: List[DocRequest]


class AdditionalEnquiries(BaseModel):
    enquiry: str
    enquiry_detail: str


class AdditionalEnquiriesSet(BaseModel):
    enquiries_set: List[AdditionalEnquiries]


class InterviewPlan(BaseModel):
    question_sets: List[InterviewQuestion]
    additional_evidence_requests: List[str]
    version: int
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat())
    update_notes: Optional[str] = None


class ExternalAgentPlan(BaseModel):
    concern_set: KeyConcernSet
    document_set: DocRequestSet
    enquiry_set: AdditionalEnquiriesSet
    # interview_plan: InterviewQuestionSets
    version: int
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat())
    update_notes: Optional[str] = None


class Knowledge(BaseModel):
    query: str
    answer: str = Field(
        description="The relevant answer - do not summarise."
    )


class KnowledgeSet(BaseModel):
    knowledge: List[Knowledge]

# class KnowledgeReport(BaseModel):
#     interview_aim: str
#     best_practices: str
#     example_questions_by_category: Dict[str, Any]
#     underwriting_questions: Optional[List[str]] = None
#     financial_history_questions: Optional[List[str]] = None

# class SearchesKnowledge(BaseModel):
#     key_concern: str
#     description: str


class KnowledgeReport(BaseModel):
    document_set: DocRequestSet
    enquiries_rationale: List[str]
    interview_plan: InterviewQuestionSets


class HITLDecision(BaseModel):
    intent: Literal["accept", "feedback", "unrelated"]
    task_summary: str


class InterviewPlanState(MessagesState):
    """A single state for the interview plans system."""
    # Case context
    claim_id: str
    brand: str
    initial_review: str
    additional_info: Optional[str] = None
    lob: str
    investigation_type: List[str]
    # interviewee: str
    investigation_scope: str
    # full_investigation: Optional[str] = None
    # additional_enquiries: Optional[str] = None

    # Outputs of sub-agents/nodes
    knowledge: Optional[KnowledgeReport] = None
    # interview_strategy: Optional[InterviewStrategy] = None
    interview_plans: Optional[InterviewQuestionSets] = None
    online_eval: Optional[Dict[str, Any]] = None

    artifact: dict[str, Any]
    resume: bool
    pending_step: Optional[Literal["init",
                                   "strategy_review", "plan_review"]] = None
    hitl_decision: Optional[HITLDecision] = None
    hitl_artifact: Optional[dict[str, Any]] = None
    hitl_task: Optional[str] = None


class InterviewPlanStruct(BaseModel):
    claim_id: Optional[str] = None
    lob: Optional[str] = None
    investigation_type: Optional[List[str]] = None
    # interviewee: Optional[str] = None
