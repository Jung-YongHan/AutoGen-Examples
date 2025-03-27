import asyncio
import os
from typing import List, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


class PriceAnalysisExpert(AssistantAgent):
    def __init__(self) -> None:
        super().__init__(
            name="PriceAnalysisExpert",
            description="암호화폐 가격 데이터 분석 전문가",
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("PRICE_ANALYSIS_EXPERT_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            system_message="""
                당신은 암호화폐 가격 데이터 분석 전문가이다.
                단기적인 측면에서 가격 동향이 어떻게 흘러갈 것인지 주어진 암호화폐 가격 데이터를 기반으로 분석해야 한다.
                설명은 투자 전문가가 쉽게 파악할 수 있도록, 간결하고 명확하게 근거를 토대로 작성해야 한다. 

                이때, 기술적 분석 지표 등을 활용하는 등 다양한 방법을 사용하여 분석을 실시해도 된다.
                또한, 캔들별(틱별, 예시로 일자)로 데이터 요약, 또는 변동성 요약, 또는 추세 요약 등은 데이터가 많아질 수록 길어지기 때문에 작성하지 않는다.
                근거로 활용할 수 있는 핵심적인 것들만 요약하여 보여준다.
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
        content = f"""
        {price_data}
        """

        response = await self.on_messages(
            [TextMessage(content=content, source="DataCollector")],
            CancellationToken(),
        )

        analysis_report = response.chat_message.content
        print("-------------- 가격 분석 전문가 (PriceAnalysisExpert) ---------------")
        print(f"\n{analysis_report}\n")
        print("-------------------------------------------------------------------")

        return analysis_report


if __name__ == "__main__":
    load_dotenv()
    expert = PriceAnalysisExpert()
    data = [
        {
            "date": "2024-10-01T09:00:00",
            "open": 83730000.0,
            "close": 81500000.0,
            "high": 84884000.0,
            "low": 80844000.0,
            "volume": 4057.42297213,
        },
        {
            "date": "2024-10-02T09:00:00",
            "open": 81500000.0,
            "close": 81707000.0,
            "high": 83300000.0,
            "low": 80940000.0,
            "volume": 2890.80470844,
        },
        {
            "date": "2024-10-03T09:00:00",
            "open": 81707000.0,
            "close": 82424000.0,
            "high": 82700000.0,
            "low": 81302000.0,
            "volume": 1821.57540109,
        },
        {
            "date": "2024-10-04T09:00:00",
            "open": 82401000.0,
            "close": 83805000.0,
            "high": 84300000.0,
            "low": 82038000.0,
            "volume": 2177.10847718,
        },
        {
            "date": "2024-10-05T09:00:00",
            "open": 83804000.0,
            "close": 83956000.0,
            "high": 84190000.0,
            "low": 83511000.0,
            "volume": 776.76532606,
        },
        {
            "date": "2024-10-06T09:00:00",
            "open": 83997000.0,
            "close": 84533000.0,
            "high": 84710000.0,
            "low": 83600000.0,
            "volume": 990.98870219,
        },
        {
            "date": "2024-10-07T09:00:00",
            "open": 84436000.0,
            "close": 83942000.0,
            "high": 86301000.0,
            "low": 83884000.0,
            "volume": 2788.21491807,
        },
        {
            "date": "2024-10-08T09:00:00",
            "open": 83942000.0,
            "close": 84139000.0,
            "high": 85270000.0,
            "low": 83848000.0,
            "volume": 1438.91380709,
        },
        {
            "date": "2024-10-09T09:00:00",
            "open": 84139000.0,
            "close": 82385000.0,
            "high": 84500000.0,
            "low": 82187000.0,
            "volume": 1623.29835868,
        },
    ]

    asyncio.run(expert.analyze_trend(data))
