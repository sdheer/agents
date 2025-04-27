import os

from multi_tool_agent.agent import call_agent_async
async def run_conversation():
    await call_agent_async("C:\\Users\\User\\Downloads\\air_ticket.pdf")
    # await call_agent_async("How about Paris?") # Expecting the tool's error message
    # await call_agent_async("Tell me the weather in New York")

# Execute the conversation using await in an async context (like Colab/Jupyter)
async def main():
    await run_conversation()