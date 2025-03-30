import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from select_agent_dynamically.web_search_analysis.utils import (
    search_web_tool,
    percentage_change_tool,
)


class PlanningAgent(AssistantAgent):
    def __init__(
        self,
    ):
        super().__init__(
            name="PlanningAgent",
            description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("PLANNING_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            system_message="""
                You are a planning agent.
                Your job is to break down complex tasks into smaller, manageable subtasks.
                Your team members are:
                    WebSearchAgent: Searches for information
                    DataAnalystAgent: Performs calculations

                You only plan and delegate tasks - you do not execute them yourself.

                When assigning tasks, use this format:
                1. <agent> : <task>

                After all tasks are complete, summarize the findings and end with "TERMINATE".
            """,
        )


class WebSearchAgent(AssistantAgent):
    def __init__(
        self,
    ):
        super().__init__(
            name="WebSearchAgent",
            description="An agent for searching information on the web.",
            tools=[search_web_tool],
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("WEB_SEARCH_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            system_message="""
                You are a web search agent.
                Your only tool is search_tool - use it to find information.
                You make only one search call at a time.
                Once you have the results, you never do calculations based on them.
            """,
        )


class DataAnalystAgent(AssistantAgent):
    def __init__(
        self,
    ):
        super().__init__(
            name="DataAnalystAgent",
            description="An agent for performing calculations.",
            tools=[percentage_change_tool],
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("DATA_ANALYST_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            system_message="""
                You are a data analyst.
                Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
                If you have not seen the data, ask for it.
            """,
        )


class TeamAgent(SelectorGroupChat):
    def __init__(self):
        super().__init__(
            participants=[
                PlanningAgent(),
                WebSearchAgent(),
                DataAnalystAgent(),
            ],
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("SELECTOR_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            selector_prompt="""Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
""",
            termination_condition=(
                TextMentionTermination("TERMINATE")
                | MaxMessageTermination(max_messages=25)
            ),
            allow_repeated_speaker=True,
        )
