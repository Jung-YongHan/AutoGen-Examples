import asyncio
import re
import time

from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from multi_agent_system.trading_system.utils.time_utils import calculate_elapsed_time


class TradingExpert(AssistantAgent):
    def __init__(self) -> None:
        super().__init__(
            name="TradingExpert",
            description="투자 전문가",
            # model_client=OpenAIChatCompletionClient(
            #     model=os.getenv("TRADING_EXPERT_MODEL"),
            #     api_key=os.getenv("OPENAI_API_KEY"),
            # ),
            model_client=OllamaChatCompletionClient(model="deepseek-r1:32b"),
            system_message="""
                'Say Korean for all responses'
                당신은 투자 매매 신호 생성 전문가이다.
                당신은 PriceAnalysisExpert가 암호화페 가격 데이터를 분석한 요약 지문을 기반으로 다음 틱(캔들)의 매매 신호를 생성하는 역할을 수행한다.
                매매 신호 생성에 대한 근거를 작성해야 한다.
                이때, 신호는 다음과 같이 생성한다:
                - 매수: 1,
                - 보유: 0
                - 매도: -1,

                최종 출력 시에는 첫 줄에 매매 신호(1,0,-1)를 출력하고, 다음 줄부터 근거를 잘 요약하여 서술한다.
                각 요약 사항은 '- '로 시작하여 작성한다.
                (두번째 줄에만 작성하는 것이 아니라, 여러 줄에 걸쳐서 작성한다.)                
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


if __name__ == "__main__":
    load_dotenv()
    trading_expert = TradingExpert()
    print(asyncio.run(trading_expert.generate_signal(analysis_report="test")))

# TODO : 출력 형식에 따른 signal, reason 분리 필요
