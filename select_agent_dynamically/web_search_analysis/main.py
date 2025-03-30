import asyncio

from autogen_agentchat.ui import Console
from dotenv import load_dotenv

from select_agent_dynamically.web_search_analysis.models import TeamAgent

if __name__ == "__main__":
    load_dotenv()
    task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"
    team_manager = TeamAgent()
    print(asyncio.run(Console(team_manager.run_stream(task=task))))
