EXTERNAL_AGENT_SYSTEM_PROMPT = """
You are a senior insurance fraud investigator. Your task is to create **comprehensive instructions** for an external investigation.
"""

KEY_CONCERNS_DRAFT_PROMPT = """
<TASK_DEFINITION>
Key concerns are critical issues identified from INITIAL REVIEW and not just a call out. They can not be specific, factual observations or anomalies in INITIAL REVIEW. Key concerns would be material risks, uncertainties or issues that could impact coverage, liability, admissibility, fraud exposure or settlement.
</TASK_DEFINITION>

<TASK>
**YOUR TASK**
Draft key concerns and their description for external investigation :

1. Assess the INITIAL REVIEW and understand all the claim details mentioned like reason for claim, important dates, past history and all possible events and details mentioned in INITIAL REVIEW.

2. Draft the key concerns and description using the **INITIAL REVIEW**. Key Concerns should be unbiased, non opinionated and non accusatory

3. Guidelines for drafting the key concerns:
    a. START with analyzing the INITIAL REVIEW to understand the sequence of events and their nature.
    b. List down all key concerns from INITIAL REVIEW that require attention with a short rationale.
    c. Rationale must be factual and comprehensive - include:
       - The specific evidence/data from INITIAL REVIEW supporting the concern
       - Relevance to the investigation (why this matters for coverage/liability/fraud exposure)
       - Financial implications or motive considerations where applicable
    d. Concern and rationale should be unbiased, non opinionated and non accusatory.
    e. If there are multiple concerns, each must be explicitly stated as a separate concern.
    f. Ensure all concerns are clear and avoid using any jargons.
    g. Re-frame any opinion-based language from INITIAL REVIEW into neutral, factual statements. Do not echo subjective terms like "grossly", "suspicious", or "concerning" from the source material.

4. CRITICAL REVIEW STEP - Before outputting, review each concern:
    a. Does this concern ONLY involve parties listed on the claim? If NO → REMOVE
    b. Can this concern be substantiated through investigation? If NO → REMOVE
    c. Does this concern overlap with another? If YES → CONSOLIDATE into one
    d. Does this concern use neutral, factual language? If NO → REFRAME
    e. Does the rationale include specific evidence AND financial/valuation implications? If NO → ENHANCE

5. Aim for 3-5 distinct, well-supported concerns. Quality over quantity.
</TASK>

<LANGUAGE_GUIDELINES>
Use neutral, factual language. Replace accusatory or opinion-based terms:

AVOID: "fraudulent", "fraud", "suspicious", "concerning", "red flags", "warning signs", "collusion", "conspiracy", "motive", "intentional", "grossly", "extremely"

USE INSTEAD: "requires verification", "determine whether X is consistent with insured's version", "discrepancy between X and Y", "pattern of similar claims", "warrants investigation"

CRITICAL: Avoiding prejudicial language does NOT mean excluding legitimate investigation areas. Verifying incident circumstances, checking for patterns in claims history, and assessing valuation discrepancies are all valid concerns - they just need neutral framing.

Do not echo opinion-based language from INITIAL REVIEW. If the input says "grossly overinsured", reframe as "insured value exceeds market valuation".
</LANGUAGE_GUIDELINES>

<RELEVANCE_CRITERIA>
Only include KEY CONCERNS - material issues impacting coverage, liability, or claim validity. Exclude observations that are merely "call outs".

STRICTLY apply each criterion - if a concern fails ANY criterion, EXCLUDE it:

1. **Party Scope**: ONLY include concerns about parties explicitly listed on the current claim. Background information (criminal history, address history, associations) of individuals NOT named as claim parties is NOT relevant - exclude it entirely.

2. **Actionable**: The concern must be verifiable through investigation. Ask: "Is there a legal requirement, documentary evidence, or practical method to substantiate this?" If NO, exclude it.

3. **Non-Duplicative**: Each concern must cover a DISTINCT issue. Before adding a concern, check if it overlaps with another - if so, consolidate into ONE comprehensive concern rather than listing separately.

4. **Material**: The concern must directly impact claim outcome. Behavioral observations (timing of calls, how claim was lodged) are NOT standalone concerns unless they directly affect validity.

5. **Evidence-Based**: Must reference specific evidence from INITIAL REVIEW. Do not include concerns based on assumptions about intent or inferences from general patterns.
</RELEVANCE_CRITERIA>

<CONSTRAINTS>
The key concerns must:
- Reference specific evidence from the INITIAL REVIEW.
- Ensure concerns are factual and do not lead the investigation to a predetermined outcome. For example, instead of "investigate possible staged accident," use "determine how the accident and damage to the vehicle occurred and whether this is consistent with the Insured's version of events."
- Comply with the General Insurance Code of Practice and relevant privacy and fairness obligations.
</CONSTRAINTS>

<EXAMPLES>
The following examples illustrate proper key concern formatting:

INCORRECT (Accusatory Language):
{{
    "concern": "Suspected fraudulent claim due to suspicious behavior",
    "rationale": "The claimant's behavior suggests potential fraud and collusion."
}}

CORRECT (Factual/Neutral):
{{
    "concern": "Claim circumstances require verification",
    "rationale": "The timeline between incident and claim lodgement, along with the reported sequence of events, warrants verification to confirm consistency with the insured's version."
}}

INCORRECT (Non-Actionable, Opinion-Based):
{{
    "concern": "Insured is grossly overinsured, indicating financial motive",
    "rationale": "The insured value is significantly above market value, raising red flags about potential fraud."
}}

CORRECT (Actionable, Evidence-Based):
{{
    "concern": "Insured value exceeds market valuation",
    "rationale": "The insured value exceeds the assessed market valuation. This discrepancy requires assessment to determine appropriate settlement value and whether the sum insured reflects the asset's actual worth at inception."
}}
</EXAMPLES>

<OUTPUT>
{format}
</OUTPUT>

<CONTEXT>
These are the relevant materials for your case:

The INITIAL REVIEW includes notes on the claim, policy and relevant details from searches conducted for the case being investigated. Use this information to inform your question set:
<INITIAL REVIEW>
{initial_review}
</INITIAL REVIEW>
</CONTEXT>
"""

DOC_REQUEST_DRAFT_PROMPT = """
<TASK>
**YOUR TASK**
List down all the document types and document details required for external investigation for provided investigation type :

1. Assess the INITIAL REVIEW and understand all the claim details mentioned like reason for claim, important dates, past history and all possible events and details mentioned in INITIAL REVIEW.

2. List down the document types and document details required using the **INITIAL REVIEW** and **INVESTIGATION PROCESSES**.

3. Guidelines for listing down the document types and details:
    a. START with analyzing the INITIAL REVIEW to understand the sequence of events and their nature.
    b. Analyse the knowledge from INVESTIGATION PROCESSES to understand what all documents are requested for given investigation type
    c. List down all the document types using INITIAL REVIEW and INVESTIGATION PROCESSES that are required with details of what all documents are required.
    d. As per initial review, you can mention the detailed list of documents in the "document details".

4. Review the document types along with their details and ensure that you have included all possible documents required.

</TASK>

<CONTEXT>
These are the relevant materials for your case:

The INITIAL REVIEW includes notes on the claim, policy and relevant details from searches conducted for the case being investigated. Use this information to inform your question set:
<INITIAL REVIEW>
{initial_review}
</INITIAL REVIEW>

Here is the INVESTIGATION PROCESSES to guide you:
<INVESTIGATION PROCESSES>
{knowledge}
</INVESTIGATION PROCESSES>
</CONTEXT>

<OUTPUT>
{format}
</OUTPUT>

Here are some output examples
<EXAMPLES>
Example 1
Output:
{{
    "doc_type": "Bank Statement ",
    "doc_details": "All financial statements for any and all accounts held in your name or jointly with somebody else for the period TBA. Please ensure this includes savings, current and credit card accounts and that the records CONFIDENTIAL appear on the letterhead of the relevant financial institution and ensure the bank details are redacted"
}}
Example 2
Output:
{{
    "doc_type": "Vehicle Photo (incident)",
    "doc_details": "A copy of any photos taken from incident scene this includes, other parties details/ licence, damages to yours and their vehicles. these photos are in the original format and size, please do not rename the photo and attach the photo to the email itself"
}}
</EXAMPLES>
"""

ADDITIONAL_ENQUIRIES_DRAFT_PROMPT = """
<TASK_DEFINITION>
Additional Enquiries are the additional responsibilities which the external investigator is required to perform in addition to their core responsibilitites for provided investigation type.
</TASK_DEFINITION>

<TASK>
**YOUR TASK**
Determine the ADDITIONAL ENQUIRIES required for provided investigation type :

1. Assess the INITIAL REVIEW and understand all the claim details mentioned like reason for claim, important dates, past history and all possible events and details mentioned in INITIAL REVIEW.

2. Determine the ADDITIONAL ENQUIRIES using **INITIAL REVIEW** and **INVESTIGATION PROCESSES**.

3. Guidelines for drafting the key concerns:
    a. Analyse the INITIAL REVIEW to understand the sequence of events and their nature.
    b. Analyse the knowledge from INVESTIGATION PROCESSES to understand what all additional enquiries are generally raised for given investigation type.
    c. Determine the ADDITIONAL ENQUIRIES using INITIAL REVIEW and INVESTIGATION PROCESSES.
    d. Include details about what needs to be done in the additional enquiries.
    e. If there are multiple enquiries, details must be explicitly stated for each.
    f. Ensure enquiries and details are clear and avoid using any jargons.

4. Review the enquiries generated and ensure that you have included all details for all enquiries.
</TASK>

<CONTEXT>
These are the relevant materials for your case:

The INITIAL REVIEW includes notes on the claim, policy and relevant details from searches conducted for the case being investigated. Use this information to inform your question set:
<INITIAL REVIEW>
{initial_review}
</INITIAL REVIEW>

Here is the INVESTIGATION PROCESSES to guide you:
<INVESTIGATION PROCESSES>
{knowledge}
</INVESTIGATION PROCESSES>
</CONTEXT>

<OUTPUT>
{format}
</OUTPUT>

<EXAMPLES>
Example 1:
Output:
{{
    "enquiry":"Please canvas loss location",
    "enquiry_details":"Please canvas loss location to confirm exactly where accident occurred, the barricade IO hit, any witnesses, CCTV etc, road conditions "
}}
Example 2:
Output:
{{
    "enquiry":"Please speak to Towie",
    "enquiry_details":"Please speak to Towie if identified and confirm observations, when contacted for tow, any other details they can provide"
}}
</EXAMPLES>
"""

INTERVIEW_PLAN_DRAFT_PROMPT = """
<TASK>
**YOUR TASK**
Draft an interview plan for the provided investigation type:

1. Assess the INITIAL REVIEW and understand all the claim details mentioned like reason for claim, important dates, past history and all possible events and details mentioned in INITIAL REVIEW.

2. List down interview categories and questions under each category using the **INVESTIGATION PROCESSES** and **INITIAL REVIEW**.

3. Guidelines for drafting the key concerns:
    a. START with analyzing the INITIAL REVIEW to understand the sequence of events and their nature.
    b. USE the information from the INVESTIGATION PROCESS to understand all the categories of questions and each question within that category.
    c. Now analyze the questions from INVESTIGATION PROCESS which are relevant to ask as per the INITIAL REVIEW
    d. Each object must include:
        - "question_id" -> the number question.
        - "category" -> a category label. Do not jump back to a previous category later in the interview.
        - "question_text" -> the interview question.
    e. Make sure there are no duplicate questions across categories

4. Review your plan and ensure that you have included all possible questions. Ensure it is following the order of questions in the INVESTIGATION PROCESSES. If you are unsure, progress from incident details --> claim-specific --> reports/documents/evidence --> underwriting/policy disclosure --> financial history. Any underwriting and/or financial history questions must be at the end.
</TASK>

<OUTPUT>
{format}
</OUTPUT>

<CONTEXT>
These are the relevant materials for your case:

Here is the INVESTIGATION PROCESSES to guide you:
<INVESTIGATION PROCESSES>
{knowledge}
</INVESTIGATION PROCESSES>

The INITIAL REVIEW includes notes on the claim, policy and relevant details from searches conducted for the case being investigated. Use this information to inform your question set:
<INITIAL REVIEW>
{initial_review}
</INITIAL REVIEW>
</CONTEXT>
"""

INTERVIEW_DOC_REQUEST_PROMPT = """
Your task is to provide a list of additional evidence to obtain from the interviewee based on the interview plan. This may include phone records, bank records, witness details or receipts, depending on the type of claim under investigation.

<INTERVIEW PLAN>
{interview_plan}
</INTERVIEW PLAN>

<OUTPUT>
{format}
</OUTPUT>
"""

INTERVIEW_PLAN_FEEDBACK_PROMPT = """
<TASK>
**YOUR TASK**
Revise the PREVIOUS VERSION of the interview plan by:

1. Prioritising and applying the FEEDBACK exactly as provided.
2. Making the **minimum necessary changes** to address the FEEDBACK.
3. Preserving structure, tone, and compliant PEACE-model sequencing unless FEEDBACK requires otherwise.
4. Populate the 'update_notes' with a user-friendly message, summarising what has changed due to the FEEDBACK.

If FEEDBACK is ambiguous, interpret it conservatively and document the intent through improved clarity rather than added scope.
</TASK>

<OUTPUT>
{format}
</OUTPUT>

<CONTEXT>
You are revising an existing interview plan based on reviewer FEEDBACK.

<PREVIOUS VERSION>
{prev_version}
</PREVIOUS VERSION>

<FEEDBACK>
{feedback}
</FEEDBACK>

Here is the supporting context for the case (for reference only - do not re-interpret unless required by feedback):

The INITIAL REVIEW includes notes on the claim, policy and relevant details from searches conducted for the case being investigated.
<INITIAL REVIEW>
{initial_review}
</INITIAL REVIEW>

The ADDITIONAL INFORMATION includes additional notes on the claim, which can include police reports, engineer reports, incident reports, or other evidence.
<ADDITIONAL INFORMATION>
{additional_info}
</ADDITIONAL INFORMATION>

Here is the INVESTIGATION PROCESSES to guide you:
<INVESTIGATION PROCESSES>
{knowledge}
</INVESTIGATION PROCESSES>
</CONTEXT>
"""
