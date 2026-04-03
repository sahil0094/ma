RETRIEVAL_TASKS = {
    "doc_requests": {
        "task": "Determine what are the documents to be requested given the provided investigation type",
        "stopping_criteria": "You have clearly put down all the documents required for external investigation for the provided investigation type"
    },
    "additional_enquiries": {
        "task": "Determine the additional enquiries required for external investigation given the provided investigation type",
        "stopping_criteria": "You have clearly put down all the additional enquiries required for external investigation for the provided investigation type"
    },
    # "question_categories": {
    #     "task": """
    # Determine interview question categories for the given investigation type and extract all questions per category except for underwriting and financial history rela
    # categories.
    #     For example, if the investigation type is 'Policy Exclusion | DUI investigation', you should call 'query_investigation_processes' to first assess the categories:
    #     - "List the category names of interview questions for a Policy Exclusion - DUI"
    #     Then, review the response and call 'query_investigation_processes' again with the following queries in parallel:
    #     - "Extract a list of all suggested interview questions in the category <insert category name> associated with Policy Exclusion DUI-related investigation, verbatim
    # stated in the context."
    # """
    # "stopping_criteria": "You have clearly stated all interview question categories(except underwriting and financial history) and all discovered categories have had a
    # questions extracted verbatim"
    # },
    # "underwriting_financial": {
    #     "task": """
    # Determine whether underwriting matters and/or financial history questions should be asked for the given investigation type, and if so, extract.
    #     For example, if the investigation type is 'Policy Exclusion | DUI investigation', you should call 'query_investigation_processes' with the following queries in par
    #     - Extract a list of all suggested interview questions for Underwriting Matters/Canvassing associated with Policy Exclusion DUI-related investigation, verbatim as i
    # stated in the context.
    #     - Extract a list of all suggested interview questions for Financial History associated with Policy Exclusion DUI-related investigation, verbatim as it is stated in
    # context.

    #  - There can be investigation types for which there will be no questions related to Underwriting Matters/Canvassing and Financial History so do not create any catego

}


KNOWLEDGE_RETRIEVAL_TASK_PROMPT = """
<TASK>
You are retrieving knowledge for a particular task.
Your goal is COMPLETE whwen you have extracted the required information for the task.
Here is your task:
{task}
</TASK>

<TOOLS>
You have access to three main tools:
1. 'query_investigation_processes' -> fraud-specific textbooks with investigation methodologies
2. 'search_complete' -> indicate that search is complete and present results
3. 'think_tool' -> for reflection and strategic planning during search
**CRITICAL: Use think_tool after each search to reflect on results and plan next steps.**
</TOOLS>

<INSTRUCTIONS>
1. **Be specific** - Phrase your searches specifically to the particular task
2. **After each search, use the think_tool to pause and assess** - Do I have enough to information?
</INSTRUCTIONS>

<STOP_WHEN>
{stopping_criteria}
</STOP_WHEN>

<OUTPUT>
Compile all search queries and answers as follows:
{format}
Do NOT output any extra commentary outside this JSON. Only return this.
</OUTPUT>
-----------------------------------------------------------
The identified investigation type is:
Note: it will be presented as <investigation type> | <sub-type if present>
<INVESTIGATION_TYPE>
{investigation_type}
</INVESTIGATION_TYPE>
"""


KNOWLEDGE_RETRIEVAL_SYSTEM_PROMPT = """
You are a fraud knowledge-retrieval assistant. You will receive an investigation type and your goal is to retrieve relevant knowledge using your tools.
"""

# KNOWLEDGE_RETRIEVAL_SYSTEM_PROMPT = """
# You are an insurance knowledge-retrieval assistant. Your goal is to retrieve all relevant searches to be done in preliminary review for the provided investigation type using
# your tools.
# """

KNOWLEDGE_REPORT_SYSTEM_PROMPT = """
You are an insurance report drafting assistant. Your task is to synthesise retrieved knowledge into a structured report which contains all the searches for the provided investigation
type .
"""

KNOWLEDGE_REPORT_PROMPT = """
<TASK>
Using the provided retrieved knowledge, produce a structured report for the given investigation type.

You MUST:
- Use ONLY the provided knowledge.
- Not introduce new content
- Not paraphrase questions
- Preserve wording of concerns raised.
</TASK>

<INPUTS>
<INVESTIGATION_TYPE>
{investigation_type}
</INVESTIGATION_TYPE>

<RETRIEVED_KNOWLEDGE>
{knowledge_set}
</RETRIEVED_KNOWLEDGE>
</INPUTS>

<OUTPUT>
Compile your report into the following format:
{format}
Do NOT output any extra commentary outside this JSON. Only return this.

</OUTPUT>
"""
