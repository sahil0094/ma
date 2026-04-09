EXTERNAL_AGENT_SYSTEM_PROMPT = """
You are a senior insurance fraud investigator. Your task is to create **comprehensive instructions** for an external investigation.
"""

KEY_CONCERNS_DRAFT_PROMPT = """
<CRITICAL_RULES>
BEFORE drafting any concerns, you MUST understand these rules. Violating these rules is a critical error.

**RULE 1 - PARTY SCOPE**: Only raise concerns about parties NAMED on this claim (insured, claimant, listed drivers). If someone is mentioned in INITIAL REVIEW but is NOT a party to this claim, do NOT include concerns about their background, criminal history, or associations.

**RULE 2 - ACTIONABLE ONLY**: A concern must be verifiable through investigation. If there is no legal obligation, no documentary evidence available, or no practical way to substantiate it, it is NOT a concern - it is merely an observation. Exclude it.

**RULE 3 - NO DUPLICATES**: Each concern must address a unique issue. If two concerns cover the same underlying issue (e.g., prior claims, valuation, timing), CONSOLIDATE them into ONE concern. Do not list the same issue multiple times with different wording.

**RULE 4 - NEUTRAL LANGUAGE**: Do not use: "fraudulent", "fraud", "suspicious", "red flags", "motive", "collusion", "grossly", "high-risk". Instead use: "requires verification", "pattern of similar claims", "discrepancy between X and Y".

**RULE 5 - 3-5 CONCERNS MAX**: Output exactly 3-5 distinct concerns. If you have more, consolidate or remove the weakest.
</CRITICAL_RULES>

<TASK>
Draft key concerns for external investigation based on INITIAL REVIEW.

Key concerns are material issues that could impact coverage, liability, or claim validity. They are NOT general observations or call outs from the INITIAL REVIEW.

**IMPORTANT**: The INITIAL REVIEW contains both relevant concerns AND irrelevant observations. Your job is to FILTER and identify only the material, actionable concerns that comply with CRITICAL_RULES above.

Steps:
1. Read INITIAL REVIEW and identify potential issues
2. For EACH potential issue, check against CRITICAL_RULES - if it fails ANY rule, exclude it
3. Consolidate overlapping issues into single concerns
4. Draft 3-5 concerns with factual rationales that include specific evidence and financial/valuation implications
</TASK>

<RATIONALE_REQUIREMENTS>
Each rationale must include:
- Specific evidence/data from INITIAL REVIEW (cite facts, dates, values)
- Why this matters for coverage, liability, or claim validity
- Financial or valuation implications where relevant

Use neutral framing. Frame as "determine whether X is consistent with insured's version" not "investigate fraud".
</RATIONALE_REQUIREMENTS>

<OUTPUT>
{format}
</OUTPUT>

<CONTEXT>
<INITIAL REVIEW>
{initial_review}
</INITIAL REVIEW>
</CONTEXT>
"""

DOC_REQUEST_DRAFT_PROMPT = """
<CRITICAL_RULES>
BEFORE listing any documents, you MUST understand these rules. Violating these rules is a critical error.

**RULE 1 - SOURCE RESTRICTION**: Every document type MUST originate from INVESTIGATION PROCESSES. If a document type cannot be traced back to a specific entry in INVESTIGATION PROCESSES, it MUST be excluded — regardless of how relevant it seems based on INITIAL REVIEW.

**RULE 2 - PARTY SCOPE**: Only request documents from parties directly involved in the claim. Use INITIAL REVIEW to identify who the direct parties are. Do not request documents from associated individuals who are not direct parties to the claim. Replace generic references in INVESTIGATION PROCESSES with the specific individuals identified from INITIAL REVIEW.

**RULE 3 - RELEVANCE FILTER**: For each document type from INVESTIGATION PROCESSES, assess whether it is applicable based on the facts in INITIAL REVIEW. If INVESTIGATION PROCESSES includes a conditional qualifier (e.g., "only if there are concerns"), apply that condition against INITIAL REVIEW — if the condition is not met, exclude that document or the irrelevant sub-item from the output.

**RULE 4 - NO DUPLICATES**: Each piece of information must appear under exactly one document type. If the same information could fall under multiple document types, place it under the most specific one and exclude it from the others.
</CRITICAL_RULES>

<TASK>
**YOUR TASK**
List down all the document types and document details required for external investigation for provided investigation type:

Steps:
1. Read INVESTIGATION PROCESSES first. Identify all document types specified for the given investigation type. These are your ONLY permitted document types.

2. Read INITIAL REVIEW to extract case-specific details (names of relevant parties, dates, locations, incident specifics).

3. For each document type identified in Step 1:
    a. Assess whether it is relevant to this case based on INITIAL REVIEW (apply RULE 3).
    b. If relevant, contextualise the document details with case-specific information from INITIAL REVIEW — include specific names, dates, vehicle details, and locations where applicable.

4. **Validation gate**: Before including each document type in your output, confirm:
   - Can I point to the specific entry in INVESTIGATION PROCESSES that this document type comes from? If NO → exclude it.
   - Am I only requesting this from parties directly involved in the claim? If NO → remove irrelevant parties.
   - Is this document applicable based on the facts in INITIAL REVIEW? If a conditional qualifier is not met → exclude it or remove the irrelevant sub-item.
   - Does this information already appear under another document type? If YES → keep it only under the most specific type.

5. Review the final list and ensure all document types pass the validation gate.
</TASK>

<CONTEXT>
These are the relevant materials for your case:

Here is the INVESTIGATION PROCESSES — this is your ONLY source for document types:
<INVESTIGATION PROCESSES>
{knowledge}
</INVESTIGATION PROCESSES>

The INITIAL REVIEW provides case-specific details for contextualisation and relevance assessment. Do NOT derive new document types from this section:
<INITIAL REVIEW>
{initial_review}
</INITIAL REVIEW>
</CONTEXT>

<OUTPUT>
{format}
</OUTPUT>

Here are some output examples
<EXAMPLES>
Example 1
Output:
{{
  "doc_type": "Bank Statement",
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

# ADDITIONAL_ENQUIRIES_DRAFT_PROMPT = """
# <TASK_DEFINITION>
# Additional Enquiries are the additional responsibilities which the external investigator is required to perform in addition to their core responsibilitites for provided investigation type.
# </TASK_DEFINITION>

# <TASK>
# **YOUR TASK**
# Determine the ADDITIONAL ENQUIRIES required for provided investigation type :

# 1. Assess the INITIAL REVIEW and understand all the claim details mentioned like reason for claim, important dates, past history and all possible events and details mentioned in INITIAL REVIEW.

# 2. Determine the ADDITIONAL ENQUIRIES using **INITIAL REVIEW** and **INVESTIGATION PROCESSES**.

# 3. Guidelines for drafting the key concerns:
#     a. Analyse the INITIAL REVIEW to understand the sequence of events and their nature.
#     b. Analyse the knowledge from INVESTIGATION PROCESSES to understand what all additional enquiries are generally raised for given investigation type.
#     c. Determine the ADDITIONAL ENQUIRIES using INITIAL REVIEW and INVESTIGATION PROCESSES.
#     d. Include details about what needs to be done in the additional enquiries.
#     e. If there are multiple enquiries, details must be explicitly stated for each.
#     f. Ensure enquiries and details are clear and avoid using any jargons.

# 4. Review the enquiries generated and ensure that you have included all details for all enquiries.
# </TASK>

# <CONTEXT>
# These are the relevant materials for your case:

# The INITIAL REVIEW includes notes on the claim, policy and relevant details from searches conducted for the case being investigated. Use this information to inform your question set:
# <INITIAL REVIEW>
# {initial_review}
# </INITIAL REVIEW>

# Here is the INVESTIGATION PROCESSES to guide you:
# <INVESTIGATION PROCESSES>
# {knowledge}
# </INVESTIGATION PROCESSES>
# </CONTEXT>

# <OUTPUT>
# {format}
# </OUTPUT>

# <EXAMPLES>
# Example 1:
# Output:
# {{
#     "enquiry":"Please canvas loss location",
#     "enquiry_details":"Please canvas loss location to confirm exactly where accident occurred, the barricade IO hit, any witnesses, CCTV etc, road conditions "
# }}
# Example 2:
# Output:
# {{
#     "enquiry":"Please speak to Towie",
#     "enquiry_details":"Please speak to Towie if identified and confirm observations, when contacted for tow, any other details they can provide"
# }}
# </EXAMPLES>
# """

ADDITIONAL_ENQUIRIES_DRAFT_PROMPT = """

<TASK_DEFINITION>
Additional Enquiries are the additional responsibilities which the external investigator is required to perform in addition to their core responsibilities for provided investigation type.
</TASK_DEFINITION>

<CRITICAL_RULES>
BEFORE drafting any enquiries, you MUST understand these rules. Violating these rules is a critical error.

**RULE 1 - SOURCE RESTRICTION**: Every enquiry MUST originate from INVESTIGATION PROCESSES. If an enquiry cannot be traced back to a specific section or requirement in INVESTIGATION PROCESSES, it MUST be excluded — regardless of how relevant it seems based on INITIAL REVIEW.

**RULE 2 - CONTEXTUALISE AND DECOMPOSE**: You must rewrite each enquiry from INVESTIGATION PROCESSES using case-specific details from INITIAL REVIEW. This includes:
  a. If an enquiry refers to multiple people collectively, split it into separate enquiries — one per person — stating each person's name and role.
  b. Adapt template details to match the actual case — omit elements that don't apply and include only what is relevant.
  c. The output must never read like a generic template. Every enquiry must reference specific names, dates, locations, or details from INITIAL REVIEW.
INITIAL REVIEW must NEVER be used to generate new enquiry topics.

**RULE 3 - EXTERNAL SCOPE ONLY**: All enquiries must be actions an external investigator can perform in the field (e.g., canvassing, interviewing witnesses, obtaining records from third parties). Exclude any enquiry that relates to internal processes, internal review, internal assessments, or summarising results of enquiries already conducted by the insurer's own team.
</CRITICAL_RULES>

<TASK>
**YOUR TASK**
Determine the ADDITIONAL ENQUIRIES required for provided investigation type:

Steps:
1. Read INVESTIGATION PROCESSES first. Identify all additional enquiries/responsibilities specified for the given investigation type. These are your ONLY permitted enquiry topics.

2. Read INITIAL REVIEW to extract case-specific details (names, dates, locations, incident specifics).

3. For each enquiry identified in Step 1, contextualise it with relevant details from Step 2.

4. **Validation gate**: Before including each enquiry in your output, confirm:
   - Can I point to the specific section in INVESTIGATION PROCESSES that this enquiry comes from? If NO → exclude it.
   - Is this an action for an external investigator (not internal)? If NO → exclude it.
   - Does this enquiry reference specific people, places, dates, or details from INITIAL REVIEW? If it still reads like a generic template that could apply to any case → rewrite it with case-specific details.
   - Does this enquiry cover multiple people? If YES → split it into one enquiry per person.

5. Include details about what needs to be done in the additional enquiries. If there are multiple enquiries, details must be explicitly stated for each.

6. Ensure enquiries and details are clear and avoid using any jargons.

Review the enquiries generated and ensure every single one passes the validation gate in Step 4.
</TASK>

<CONTEXT>
These are the relevant materials for your case:

Here is the INVESTIGATION PROCESSES — this is your ONLY source for enquiry topics:
<INVESTIGATION PROCESSES>
{knowledge}
</INVESTIGATION PROCESSES>

The INITIAL REVIEW provides case-specific details for contextualisation only. Do NOT derive new enquiry topics from this section:
<INITIAL REVIEW>
{initial_review}
</INITIAL REVIEW>
</CONTEXT>

<OUTPUT>
{format}
</OUTPUT>

<EXAMPLES>
Example 1:
Output:


{{
  "enquiry": "Please canvas loss location",
  "enquiry_details": "Please canvas loss location to confirm exactly where accident occurred, the barricade IO hit, any witnesses, CCTV etc, road conditions"
}}
Example 2:
Output:

{{
  "enquiry": "Please speak to Towie",
  "enquiry_details": "Please speak to Towie if identified and confirm observations, when contacted for tow, any other details they can provide"
}}
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
