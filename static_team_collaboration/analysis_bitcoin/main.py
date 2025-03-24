import asyncio

from static_team_collaboration.analysis_bitcoin.models import TeamAgent

if __name__ == "__main__":
    asyncio.run(TeamAgent().run())
