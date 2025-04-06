import os
import time
from typing import List, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from multi_agent_system.trading_system.utils.time_utils import calculate_elapsed_time


class PriceAnalysisExpert(AssistantAgent):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            name="PriceAnalysisExpert",
            description="암호화폐 가격 데이터 분석 전문가",
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("PRICE_ANALYSIS_EXPERT_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            system_message="""
                You are an expert in cryptocurrency price data analysis. 
                You need to analyze the given cryptocurrency price data to determine the short-term price trend. 
                The explanation should be concise and clear, based on evidence, so that investment experts can easily understand it.

                You can use various methods for analysis, including technical analysis indicators.
                Additionally, do not summarize data by candle (tick, e.g., date), volatility summary, or trend summary, as the data can become lengthy. 
                Only summarize the key points that can be used as evidence.
                
                Please respond to all prompts in Korean.          
            """,
        )

    async def analyze_trend(self, price_data: List[Dict]) -> str:
        """
        수집된 가격 데이터를 통해 단기적인 가격 추세를 분석 및 요약 리포트 반환.

        Args:
            price_data (List[Dict]): 수집된 가격 데이터

        Returns:
            str: 가격 추세에 대한 요약 리포트 (예: "단기적으로 상승 추세가 예상됩니다.")
        """
        start_time = time.time()

        content = f"""
        {price_data}
        """

        response = await self.on_messages(
            [TextMessage(content=content, source="DataCollector")],
            CancellationToken(),
        )

        analysis_report = response.chat_message.content

        end_time = time.time()
        elapsed_day, elapsed_hour, elapsed_minute, elapsed_second = (
            calculate_elapsed_time(start_time, end_time)
        )
        print("-------------- 가격 분석 전문가 (PriceAnalysisExpert) ---------------")
        print(f"\n{analysis_report}\n")
        print(
            f"응답 소요 시간: {elapsed_day}일 {elapsed_hour}시간 {elapsed_minute}분 {elapsed_second}초"
        )
        print("-------------------------------------------------------------------")
        return analysis_report
