from typing import Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient


class CreatorAgent(AssistantAgent):
    def __init__(self, model_client: Union[OpenAIChatCompletionClient, OllamaChatCompletionClient]):
        super().__init__(
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

class StudentAgent(AssistantAgent):
    def __init__(self, model_client: Union[OpenAIChatCompletionClient, OllamaChatCompletionClient]):
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
    def __init__(self, model_client: Union[OpenAIChatCompletionClient, OllamaChatCompletionClient]):
        super().__init__(
            name="ProfessorAgent",
            description="A strict professor who evaluates reports critically and provides constructive feedback.",
            model_client=model_client,
            system_message=(
                """
                You are a strict professor evaluating the report written by 'StudentAgent'.
                Read the report critically and provide constructive feedback, grade.
                Point out any weaknesses or areas for improvement clearly.
                If grade is A+, then say 'Excellent!'.
                """
            ),
        )

class TeamAgent(SelectorGroupChat):
    def __init__(self, participants: list[AssistantAgent], model_client: Union[OpenAIChatCompletionClient, OllamaChatCompletionClient], termination_condition: TextMentionTermination):
        super().__init__(
            participants=participants,
            model_client=model_client,
            termination_condition=termination_condition,
            selector_prompt="""Select an agent to perform the next task.
                
                Below is the Agents'roles:
                    - CreatorAgent: Generates content based on a given topic.
                    - StudentAgent: Writes a report based on the content created by 'CreatorAgent'.
                    - ProfessorAgent: Evaluates the report written by 'StudentAgent'.

                Current conversation context:
                {history}

                Read the roles, conversation above and select an agent from {participants} to perform the next task.
                Only one agent can be selected per task.
                """,
            max_turns = 10,
            allow_repeated_speaker=False
        )