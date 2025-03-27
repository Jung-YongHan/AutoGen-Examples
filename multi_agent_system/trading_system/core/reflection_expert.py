import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


class ReflectionExpert(AssistantAgent):
    def __init__(self) -> None:
        super().__init__(
            name="ReflectionExpert",
            description="성찰 전문가",
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("REFLECTION_EXPERT_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            system_message="""
                당신은 투자 성과 및 행동에 대한 성찰 전문가이다.
                과거 투자 기록을 분석하여 투자 전문가가 다음 투자에 대한 성과 및 행동을 개선할 수 있도록 도와야 한다.
                성찰 보고서는 다음과 같은 형식으로 생성된다:
                - [투자 성과 및 행동에 대한 성찰 내용]
                - [성찰 내용을 바탕으로 한 향후 투자 전략]
            """,
        )
