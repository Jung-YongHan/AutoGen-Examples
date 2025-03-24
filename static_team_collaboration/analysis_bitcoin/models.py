import asyncio

from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
from autogen_agentchat.agents import UserProxyAgent, AssistantAgent

from static_team_collaboration.analysis_bitcoin.utils import (
    search_price_data,
    get_recent_day,
)


class UserAgent(UserProxyAgent):
    def __init__(self):
        super().__init__(
            name="UserAgent",
            description="시스템과 상호작용하는 사용자 프록시 에이전트.",
            input_func=input,
        )


class DataCollectingAgent(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="DataCollectingAgent",
            description="사용자가 제시한 시작, 끝 날짜에 맞는 암호화폐 데이터를 수집하는 에이전트",
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("DATA_COLLECTING_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            tools=[get_recent_day, search_price_data],
            reflect_on_tool_use=True,
            system_message=(
                """
                당신은 비트코인 가격 데이터를 수집하는 데 특화된 데이터 수집 에이전트입니다.
                도구를 활용하여 주어진 작업을 해결해야 합니다.
                당신의 주요 책임은 다음과 같습니다:
                
                ### 데이터 수집:
                    지정된 시작일과 종료일을 기준으로 비트코인 가격 데이터를 수집합니다.
                    정확하고 포괄적인 가격 데이터를 가져오기 위해 사용 가능한 도구(get_recent_day, search_price_data)의 기능을 효과적으로 활용합니다.
                ### 파라미터 식별:
                    제공된 문자에서 시작일과 종료일을 명확히 식별하고 유효성을 검증합니다.
                    ex) 20250201 ~ 20250305 => 시작일: 2025년 2월 1일, 종료일: 2025년 3월 5일
                    250201-250305 => 시작일: 2025년 2월 1일, 종료일: 2025년 3월 5일
                    2025-02-01 ~ 2025-03-05 => 시작일: 2025년 2월 1일, 종료일: 2025년 3월 5일
                    
                    수집된 데이터가 지정된 기간을 정확히 따르도록 보장합니다.
                    
                당신의 주요 목표는 비트코인 가격 데이터를 효율적이고 정확하게 수집하여 이후 분석을 원활히 지원하는 것입니다.
                """
            ),
        )


class DataAnalysisAgent(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="DataAnalysisAgent",
            description="주어진 암호화폐 가격 데이터를 분석하여 보고서를 작성하는 에이전트",
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("DATA_ANALYSIS_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            system_message=(
                """
                당신은 비트코인 가격 데이터를 분석하고 보고서를 작성하는 에이전트입니다.
                주요 책임은 다음과 같습니다:
                
                ### 데이터 분석
                    1. 평균 가격, 변동성, 최대 및 최소 가격 등 주요 지표를 포함한 통계 분석을 수행합니다.
                    2. 주어진 데이터 내에서 상승, 하락, 횡보 등 추세와 주목할 만한 패턴을 식별합니다.
                    3. 가격 변동 중 이상 징후나 비정상적인 이벤트를 탐지합니다.
                ### 시각화:
                    가격 움직임, 변동성, 분포 등을 설명하기 위한 선 그래프, 히스토그램, 박스 플롯 등 명확하고 정보 전달력이 높은 시각화를 생성합니다.
                    시각화 내에서 중요한 사건이나 변화가 드러나도록 강조합니다.
                    이때, 실제 파이썬 코드를 사용하여 데이터 시각화를 수행하세요.
                ### 보고서 작성:
                    분석 결과를 구조화되고 읽기 쉬운 형식으로 명확하고 간결하게 요약합니다.
                    시각화 및 통계 분석 결과를 바탕으로 분석 기간 동안의 시장 상황에 대한 통찰을 제공합니다.
                    분석을 통해 도출된 내용을 기반으로 트레이더나 투자자에게 고려할 만한 시사점이나 전략적 제안을 제시합니다.
                    
                당신의 보고서는 철저하면서도 이해하기 쉬워야 하며, 의사결정을 지원할 수 있는 통찰을 명확하게 전달해야 합니다.
                """
            ),
        )


class DataTeam(SelectorGroupChat):
    def __init__(self):
        super().__init__(
            participants=[
                UserAgent(),
                DataCollectingAgent(),
                DataAnalysisAgent(),
            ],
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("DATA_TEAM_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
            ),
            termination_condition=TextMentionTermination("APPROVE"),
            allow_repeated_speaker=False,
            selector_prompt="""Select an agent to perform task.

Agents' Roles:
1. UserAgent: Sends dates (start, end) to the system. And interacts with the system.
2. DataCollectingAgent: Collects the crypto data based on the provided dates. And it can use the search_price_data tool.
3. DataAnalysisAgent: Analyzes the collected crypto data and returns the report.

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
""",
        )


class TeamAgent:
    def __init__(self):
        load_dotenv()
        self.team = DataTeam()

    async def run(self):
        task = "비트코인 가격 데이터를 분석하세요."
        print(await Console(self.team.run_stream(task=task)))
