"""Provider Workforce Query Agent

This agent provides natural language querying capabilities for a provider workforce database.
"""

import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from agno.os.interfaces.agui import AGUI
from agno.tools.toolkit import Toolkit
from agno.tools.function import Function


# Load environment variables
import dotenv
dotenv.load_dotenv()


# Load instructions from prompt file
def load_prompt(filename: str) -> str:
    """Load prompt from file"""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "You are an expert in analyzing provider workforce related data."

instructions = load_prompt("syphilis_query_elucidation_prompt.txt")

# Define chart display tool that calls the frontend action
def display_chart(chart_type: str, title: str, labels: str, data: str, dataset_label: str = None) -> str:
    """Display interactive charts inline in the chat

    Args:
        chart_type: Type of chart (bar, line, pie, doughnut)
        title: Chart title
        labels: Comma-separated list of chart labels
        data: Comma-separated list of numeric data values
        dataset_label: Optional label for the data series
    """
    # This function will be called by the agent and should trigger the frontend action
    return f"Displaying {chart_type} chart: {title}"

# Create the agent
agent = Agent(
    name="Provider Workforce Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="""You are an expert in analyzing provider workforce related data.

When users ask for charts, visualizations, or data displays:
1. Use the display_chart tool to show the interactive chart
2. Provide a brief, insightful interpretation of what the chart reveals
3. Do NOT list the raw data values - let the chart visualization speak for itself

Choose the appropriate chart type:
- Bar charts for comparing categories (e.g., provider counts by specialty)
- Line charts for trends over time
- Pie/Doughnut charts for proportions or distributions

Generate realistic healthcare workforce sample data. Keep responses focused and concise.""",
    tools=[display_chart],
    markdown=True,
)

# Set up AgentOS
agent_os = AgentOS(agents=[agent], interfaces=[AGUI(agent=agent)])
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="agent_workforce:app", port=8001, reload=True)
