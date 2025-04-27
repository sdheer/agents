import datetime
import os
from typing import Literal
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.genai import types # For creating message Content/Parts
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
import asyncio
from google.genai import types # For creating message Content/Parts
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
import pdfplumber
import warnings
import dotenv
import requests
# Ignore all warnings
warnings.filterwarnings("ignore")
dotenv.load_dotenv()
import logging
logging.basicConfig(level=logging.ERROR)

print("Libraries imported.")

# --- IMPORTANT: Replace placeholders with your real API keys ---

# Gemini API Key (Get from Google AI Studio: https://aistudio.google.com/app/apikey)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# OpenAI API Key (Get from OpenAI Platform: https://platform.openai.com/api-keys)

os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_AI")

# Anthropic API Key (Get from Anthropic Console: https://console.anthropic.com/settings/keys)
os.environ['ANTHROPIC_API_KEY'] = "Your Anthropic API Key"


# --- Verify Keys (Optional Check) ---
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"OpenAI API Key set: {'Yes' if os.environ.get('OPENAI_API_KEY') and os.environ['OPENAI_API_KEY'] != 'YOUR_OPENAI_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"Anthropic API Key set: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') and os.environ['ANTHROPIC_API_KEY'] != 'YOUR_ANTHROPIC_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")

# Configure ADK to use API keys directly (not Vertex AI for this multi-model setup)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# @markdown **Security Note:** It's best practice to manage API keys securely (e.g., using Colab Secrets or environment variables) rather than hardcoding them directly in the notebook. Replace the placeholder strings above.
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash-exp"

# Note: Specific model names might change. Refer to LiteLLM/Provider documentation.
MODEL_GPT_4O = "openai/gpt-4.1-mini"
MODEL_CLAUDE_SONNET = "anthropic/claude-3-sonnet-20240229"


print("\nEnvironment configured.")

# @title Define the Weather Agent
# Use one of the model constants defined earlier
AGENT_MODEL = MODEL_GEMINI_2_0_FLASH # Starting with a powerful Gemini model
def analyze_document(file_path: str) -> str:
    """
    Analyzes the content of a document (PDF) and returns either a summary, sentiment, or keywords.
    `file_path` is the path to a local PDF document.
    """
    model = AGENT_MODEL
    analysis_type: Literal["summary", "sentiment", "keywords"] = "summary"
    try:
        # Extract text from PDF

        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        
        if not text:
            return "Could not extract text from the document."

        # Use LLM to analyze the text

        prompt = f"Here is a document:\n{text[:6000]}\n\nPlease provide a {analysis_type} of it."
        from google.generativeai import GenerativeModel # Make sure this import is present

        # Instantiate the model using the model name string
        llm = GenerativeModel(AGENT_MODEL) # Use AGENT_MODEL defined earlier

        # Now call generate_content on the model instance
        result = llm.generate_content(prompt)
        return result.text
    except Exception as e:
        return f"Error during document analysis: {e}"
    
def plan_journey(ticket_info: str) -> str:
    #analysis_type: Literal["summary", "sentiment", "keywords"] = "summary"
    try:
        # Extract text from PDF
        if not ticket_info:
            return "Could not extract text from the document."

        # Use LLM to analyze the text

        prompt = f"Here is a summary of flight ticket information:\n{ticket_info[:6000]}\n\n, analyze and provide a journey plan of it."
        from agents.extensions.models.litellm_model import LitellmModel
        from openai import OpenAI
        client = OpenAI()
        model=LitellmModel(model="o4-mini", api_key = os.environ['OPENAI_API_KEY']),
        # Now call generate_content on the model instance
        response = client.responses.create(
            model=model,
            reasoning={"effort": "medium"},
            input=[
            {
                "role": "user", 
                "content": prompt
            }
        ])
        result = response.output_text
        return result
    except Exception as e:
        return f"Error during document analysis: {e}"
    
planner_agent = Agent(
    name="planner_agent",
    model= AGENT_MODEL,
    description="Provide joruney plan",
    instruction= "anlyse the information and provide a journey plan to reach the destination on time taking into consideration weather, traffic, peak hours.",
    tools=[plan_journey],
)
document_analyser_agent = Agent(
    name="analyzer_agent",
    model=AGENT_MODEL, # Specifies the underlying LLM
    description="Analyzes travel documents (PDFs) and coordinates journey planning.", # Updated description
    instruction="You are an assistant that analyzes travel document PDFs. First, use the 'analyze_document' tool to understand the document content (like flight details). Then, delegate the task of creating a detailed journey plan based on this information to the 'planner_agent'.", # Clearer instruction about delegation
    tools=[analyze_document], # Make the tool available to this agent
    sub_agents=[planner_agent]
)

print(f"Agent '{document_analyser_agent.name}' created using model '{AGENT_MODEL}'.")

# @title Setup Session Service and Runner

# --- Session Management ---
# Key Concept: SessionService stores conversation history & state.
# InMemorySessionService is simple, non-persistent storage for this tutorial.
session_service = InMemorySessionService()

# Define constants for identifying the interaction context
APP_NAME = "document analyser app"
USER_ID_G = "user_1"
RUNNER = 'runner'
SESSION_ID_G = "session_001" # Using a fixed ID for simplicity

# Create the specific session where the conversation will happen
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID_G,
    session_id=SESSION_ID_G
)
print(f"Session created: App='{APP_NAME}', User='{USER_ID_G}', Session='{SESSION_ID_G}'")

# --- Runner ---
# Key Concept: Runner orchestrates the agent execution loop.
runner = Runner(
    agent=document_analyser_agent, # The agent we want to run
    app_name=APP_NAME,   # Associates runs with our app
    session_service=session_service # Uses our session manager
)
print(f"Runner created for agent '{runner.agent.name}'.")

# @title Define Agent Interaction Function


async def call_agent_async(query: str,
                           runner,
                           USER_ID,
                           SESSION_ID
                          ):
  """Sends a query to the agent and prints the final response."""
  print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  
  try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            # You can uncomment the line below to see *all* events during execution
            print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

            # Key Concept: is_final_response() marks the concluding message for the turn.
            if event.is_final_response():
                if event.content and event.content.parts:
                    # Assuming text response in the first part
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                # Add more checks here if needed (e.g., specific error codes)
                break # Stop processing events once the final response is found

        print(f"<<< Agent Response: {final_response_text}")
  except Exception as e:
        print(e)

async def test_gpt_agent():
        # --- Test the GPT Agent ---
        print("\n--- Testing GPT Agent ---")
        # Ensure call_agent_async uses the correct runner, user_id, session_id
        await call_agent_async(query = "C:\\Users\\User\\Downloads\\TICKET.pdf",
                               runner= runner,
                               USER_ID=USER_ID_G,
                               SESSION_ID=SESSION_ID_G)
asyncio.run(test_gpt_agent())