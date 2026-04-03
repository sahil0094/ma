from langchain.tools import tool


@tool("search_complete",
      description="Decision-making tool to decide research is complete")
def search_complete(complete: str) -> str:
    """Tool to decide that search is complete.

    Args:
        complete: indication that the research is complete

    Returns:
        Confirmation that research complete was recorded
    """
    return f"Research complete: {complete}"
