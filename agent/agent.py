"""Example: Agno Agent with Finance tools

This example shows how to create an Agno Agent with tools (YFinanceTools) and expose it in an AG-UI compatible way.
"""

import dotenv
from agno.agent.agent import Agent
from agno.app.agui.app import AGUIApp
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from frontend_tools import add_proverb, set_theme_color

dotenv.load_dotenv()

agent = Agent(
  model=OpenAIChat(id="gpt-4o"),
  tools=[
    # Example of a backend tool, defined and handled in your agno agent
    YFinanceTools(stock_price=True, historical_prices=True),
    # Example of frontend tools, handled in the frontend Next.js app
    add_proverb,
    set_theme_color,
  ],
  description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
  instructions="Format your response using markdown and use tables to display data where possible.",
)

agui_app = AGUIApp(
  agent=agent,
  name="Investment Analyst",
  app_id="investment_analyst",
  description="An investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
)

app = agui_app.get_app()

if __name__ == "__main__":
  agui_app.serve(app="agent:app", port=8000, reload=True)
