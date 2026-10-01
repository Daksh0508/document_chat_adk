from google.adk.agents import Agent
from .tools import search_documents

root_agent = Agent(
    name="doc_chat_agent",
    model="gemini-2.5-flash",   # check the ADK docs for the current model name
    description="Answers questions using the user's uploaded documents.",
    instruction=(
        "You answer questions ONLY using the search_documents tool. "
        "Always call it before answering. If results look weak, try again "
        "with a rephrased query. Cite every fact as [source, page]. "
        "If the best score is below 0.5 or the passages don't answer the "
        "question, say you couldn't find it in the documents. "
        "Never use outside knowledge."
    ),
    tools=[search_documents],
)