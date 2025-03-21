import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import RoutedAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient

RoutedAgent

async def main() -> None:
    keyword = input("Enter a keyword: ")

    model_client = OllamaChatCompletionClient(model="llama3.2")

    poet = AssistantAgent(
        name="PoetAgent",
        model_client=model_client,
        system_message="당신은 창의적인 시인입니다. 주어진 키워드로 10줄짜리 시를 창작하세요."
    )

    praise = AssistantAgent(
        name="PraiseAgent",
        model_client=model_client,
        system_message=(
            "당신은 긍정적인 시 감상가입니다. 시를 읽고 칭찬을 해주세요. "
            "시가 매우 훌륭하다면 '훌륭한 시야!'라고 말하세요."
        )
    )

    critic = AssistantAgent(
        name="CriticAgent",
        model_client=model_client,
        system_message=(
            "당신은 비판적인 시 감상가입니다. 시를 읽고 건설적인 비판을 해주세요. "
            "시가 매우 훌륭하다면 '훌륭한 시야!'라고 말하세요."
        )
    )

    groupchat = RoundRobinGroupChat(
        participants=[poet, praise, critic],
    )

    class PoetryManager(GroupChatManager):
        def _check_termination(self, messages):
            praise_comment = None
            critic_comment = None
            for msg in reversed(messages):
                if msg["name"] == "PraiseAgent" and praise_comment is None:
                    praise_comment = msg["content"]
                if msg["name"] == "CriticAgent" and critic_comment is None:
                    critic_comment = msg["content"]
                if praise_comment and critic_comment:
                    break

            if praise_comment == "훌륭한 시야!" and critic_comment == "훌륭한 시야!":
                return True, "칭찬과 비판 에이전트 모두 시를 칭찬했습니다. 창작 완료!"
            return False, None

# Create the agents.
    model_client = OllamaChatCompletionClient(model="llama3.2")
    poet_agent = AssistantAgent("poet", model_client=model_client,
                                system_message="Create a poem based on a specific keyword provided by the user. Also, incorporate comments from {receptive_critic} and {critical_critic} and refine the poem according to the requirements.")
    receptive_agent = AssistantAgent("receptive_critic", model_client=model_client, system_message="Evaluate the poem created by {poet} in a receptive manner and suggest areas for improvement. If you like the poem, say 'Excellent poem!'")
    critical_agent = AssistantAgent("critical_critic", model_client=model_client, system_message="Evaluate the poem created by {poet} in a critical manner and suggest areas for improvement. If you like the poem, say 'Excellent poem!'")

    # Create the termination condition which will end the conversation when the user says "APPROVE".
    termination = TextMentionTermination("Excellent poem!")

    # Create the team.
    team = SelectorGroupChat(
        [receptive_agent, critical_agent, poet_agent],
        model_client=model_client,
                             termination_condition=termination)

    await Console(team.run_stream(task=f"this poem's keyword is : {keyword}"))

# Use asyncio.run(...) when running in a script.
asyncio.run(main())