import os
import re
import time

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from multi_agent_system.trading_system.utils.time_utils import calculate_elapsed_time


class TradingExpert(AssistantAgent):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            name="TradingExpert",
            description="투자 전문가",
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("PRICE_ANALYSIS_EXPERT_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            system_message="""
                You are an expert in generating trading signals. 
                You are responsible for generating the next tick (candle) trading signal based on the summary text of cryptocurrency price data analysis provided by the PriceAnalysisExpert. 
                You need to provide the rationale for the generated trading signal. 
                The signals are generated as follows:

                - Buy(매수): 1,
                - Hold(보유): 0
                - Sell(매도): -1,
                
                In the final output, print the trading signal (1, 0, -1) on the first line, and then summarize the rationale clearly on the following lines. 
                Each summary point should start with '- '. 
                (Write across multiple lines, not just the second line.)   
                
                Please respond to all prompts in Korean.          
            """,
        )

    async def generate_signal(self, analysis_report: str) -> int:
        """
        PriceAnalysisExpert의 리포트를 고려하여
        매매 신호를 생성합니다.

        Args:
        {analysis_report (str): PriceAnalysisExpert가 생성한 요약 리포트

        Returns:
            int: 매매신호
            매매 신호는 1(매수), 0(보유), -1(매도)
        """
        start_time = time.time()

        reason = f"""
        가격 추세 분석 리포트: 
            {analysis_report}
        """

        response = await self.on_messages(
            [TextMessage(content=reason, source="PriceAnalysisExpert")],
            CancellationToken(),
        )

        content = response.chat_message.content
        lines = content.splitlines()
        match = re.match(r"-?\d+", lines[0])
        if match:
            signal = int(match.group())
            reasons = "\n    ".join(lines[1:])
        if signal == 1:
            reason = f"""
# Signal: 
    - 매수
# Reason: 
    {reasons}
"""
        elif signal == 0:
            reason = f"""
# Signal: 
    - 보유
# Reason: 
    {reasons}
"""
        else:
            reason = f"""
# Signal: 
    - 매도
# Reason: 
    {reasons}
"""

        end_time = time.time()
        elapsed_day, elapsed_hour, elapsed_minute, elapsed_second = (
            calculate_elapsed_time(start_time, end_time)
        )
        print("-------------------- 투자 전문가 (TradingExpert) --------------------")
        print(f"\n{reason}\n\n")
        print(
            f"응답 소요 시간: {elapsed_day}일 {elapsed_hour}시간 {elapsed_minute}분 {elapsed_second}초"
        )
        print("-------------------------------------------------------------------")
        return signal
