import asyncio
import os
from typing import Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient

from dotenv import load_dotenv

model_type = Union[OllamaChatCompletionClient, OpenAIChatCompletionClient]

async def main(selector_model_client: model_type, model_client: model_type, keyword: str) -> None:
    creator = AssistantAgent(
        name="CreatorAgent",
        description="An agent that generates content based on a given topic.",
        model_client=model_client,
        system_message=(
            """
            You are an agent that generates content on a given topic.
            When a 'topic' is provided, generate content based on that topic.
            """
        ),
    )

    student = AssistantAgent(
        name="StudentAgent",
        description="A student with poor report writing skills. Revises based on feedback to improve.",
        model_client=model_client,
        system_message=(
            """
            You are a student with poor report writing skills.
            Your role is to write a report based on the content created by 'CreatorAgent'.
            You might struggle initially, but use feedback from 'ProfessorAgent' to improve.
            """
        ),
    )

    professor = AssistantAgent(
        name="ProfessorAgent",
        description="A strict professor who evaluates reports critically and provides constructive feedback.",
        model_client=model_client,
        system_message=(
            """
            You are a strict professor evaluating the report written by 'StudentAgent'.
            Read the report critically and provide constructive feedback, grade.
            Point out any weaknesses or areas for improvement clearly.
            If Grade is A+, then say 'Excellent!'.
            """
        ),
    )

    text_mention_termination = TextMentionTermination("Excellent!")

    selector_prompt = """Select an agent to perform the next task.
    
    Below is the Agents'roles:
    - CreatorAgent: Generates content based on a given topic.
    - StudentAgent: Writes a report based on the content created by 'CreatorAgent'.
    - ProfessorAgent: Evaluates the report written by 'StudentAgent'.
    
    Current conversation context:
    {history}

    Read the roles, conversation above and select an agent from {participants} to perform the next task.
    Only one agent can be selected per task.
    """

    team = SelectorGroupChat(
        participants=[creator, student, professor],
        model_client=selector_model_client,
        termination_condition=text_mention_termination,
        selector_prompt=selector_prompt,
        max_turns=10,
        allow_repeated_speaker=False,
    )

    task = f"Topic: '{keyword}'"

    print(await Console(team.run_stream(task=task)))


if __name__ == "__main__":
    keyword = input("Enter a topic:  ")

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    # model_client = OllamaChatCompletionClient(model="llama3.2")
    model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)
    selector_model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    asyncio.run(main(selector_model_client=selector_model_client, model_client=model_client, keyword=keyword))
