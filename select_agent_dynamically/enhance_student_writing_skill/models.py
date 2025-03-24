import asyncio
import os
from typing import Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


class CreatorAgent(AssistantAgent):
    def __init__(
        self,
        model_client: Union[OpenAIChatCompletionClient, OllamaChatCompletionClient],
    ):
        super().__init__(
            name="CreatorAgent",
            description="An agent that generates content based on a given topic.",
            model_client=model_client,
            system_message=(
                """
                You are an agent that generates content based on a given <keyword>.
                When a 'keyword' are provided, generate content based on that 'keyword'.
                """
            ),
        )


class StudentAgent(AssistantAgent):
    def __init__(
        self,
        model_client: Union[OpenAIChatCompletionClient, OllamaChatCompletionClient],
    ):
        super().__init__(
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


class ProfessorAgent(AssistantAgent):
    def __init__(
        self,
        model_client: Union[OpenAIChatCompletionClient, OllamaChatCompletionClient],
    ):
        super().__init__(
            name="ProfessorAgent",
            description="A strict professor who evaluates reports critically and provides constructive feedback.",
            model_client=model_client,
            system_message=(
                """
                You are a strict professor evaluating the report written by 'StudentAgent'.
                Read the report critically and provide constructive feedback, grade.
                Point out any weaknesses or areas for improvement clearly.
                If grade is greater than or equal to 'A', then you should say "Excellent!".
                """
            ),
        )


class TutorTeam(RoundRobinGroupChat):
    def __init__(
        self,
        participants: list[AssistantAgent],
        termination_condition: TextMentionTermination,
    ):
        super().__init__(
            participants=participants,
            termination_condition=termination_condition,
            max_turns=10,
        )


class TeamAgent:
    def __init__(self, keyword: str):
        self.keyword = keyword

    async def get_content(self):
        creator = CreatorAgent(
            model_client=OllamaChatCompletionClient(model="llama3.2")
        )
        result: Response = await creator.on_messages(
            [
                TextMessage(
                    content=f"Generate content based on a {self.keyword}", source="user"
                )
            ],
            cancellation_token=CancellationToken(),
        )
        return result.chat_message.content

    async def run(self):
        content = await self.get_content()

        tutor_team = TutorTeam(
            participants=[
                StudentAgent(
                    model_client=OpenAIChatCompletionClient(
                        model=os.getenv("STUDENT_MODEL"),
                        api_key=os.getenv("OPENAI_API_KEY"),
                    )
                ),
                ProfessorAgent(
                    model_client=OpenAIChatCompletionClient(
                        model=os.getenv("PROFESSOR_MODEL"),
                        api_key=os.getenv("OPENAI_API_KEY"),
                    )
                ),
            ],
            termination_condition=TextMentionTermination("Excellent!"),
        )

        task = f"Generate report based on follow content:\n\n{content}"

        print(await Console(tutor_team.run_stream(task=task)))


if __name__ == "__main__":
    load_dotenv()

    keyword = input("Enter a topic:  ")
    team_manager = TeamAgent(keyword)
    asyncio.run(team_manager.run())
