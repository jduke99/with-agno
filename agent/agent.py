"""Example: Agno Agent with Finance tools

This example shows how to create an Agno Agent with tools (YFinanceTools) and expose it in an AG-UI compatible way.
"""

import dotenv
from agno.agent.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from agno.os.interfaces.agui import AGUI
from agno.tools.yfinance import YFinanceTools
from frontend_tools import add_proverb, set_theme_color

dotenv.load_dotenv()

agent = Agent(
  name="TestAgent",
  model=OpenAIChat(id="gpt-4o-mini"),
  tools=[
    # Example of a backend tool, defined and handled in your agno agent
    YFinanceTools(),
    # Example of frontend tools, handled in the frontend Next.js app
    add_proverb,
    set_theme_color,
  ],
  description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
  instructions="Format your response using markdown and use tables to display data where possible.",
)

agent_os = AgentOS(agents=[agent], interfaces=[AGUI(agent=agent)])
app = agent_os.get_app()

if __name__ == "__main__":
  agent_os.serve(app="agent:app", port=8000, reload=True)
