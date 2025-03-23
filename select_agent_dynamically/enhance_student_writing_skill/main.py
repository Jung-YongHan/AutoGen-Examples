import asyncio
import os

from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from select_agent_dynamically.enhance_student_writing_skill.models import TeamAgent, CreatorAgent, StudentAgent, ProfessorAgent


async def main():
    keyword = input("Enter a topic:  ")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    creator = CreatorAgent(
        model_client=OllamaChatCompletionClient(
            model="llama3.2"
        )
    )
    student = StudentAgent(
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=api_key
        )
    )
    professor = ProfessorAgent(
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=api_key
        )
    )

    team_manager = TeamAgent(
        participants=[creator, student, professor],
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=api_key),
        termination_condition=TextMentionTermination("Excellent!"))

    print(await Console(team_manager.run_stream(task=f"Topic: '{keyword}'")))

if __name__ == '__main__':
    asyncio.run(main())