import asyncio
from datetime import datetime, timedelta
from typing import Tuple

from dotenv import load_dotenv

from multi_agent_system.trading_system.core.constants import (
    TIME_DELTA_MAP,
    DEFAULT_UNIT,
)
from multi_agent_system.trading_system.core.data_collector import DataCollector
from multi_agent_system.trading_system.core.portfolio_manager import (
    PortfolioManager,
)
from multi_agent_system.trading_system.core.price_analysis_expert import (
    PriceAnalysisExpert,
)
from multi_agent_system.trading_system.core.trading_expert import TradingExpert


class CryptoTradingSystem:
    def __init__(
        self,
        initial_cash: int,
        fee_rate: float,
        coin: str,
        start_date: str,
        end_date: str,
        candle_unit: str,
    ):
        self.data_collector = DataCollector()
        self.price_analysis_expert = PriceAnalysisExpert()
        self.trading_expert = TradingExpert()
        self.portfolio_manager = PortfolioManager(
            initial_cash=initial_cash, fee_rate=fee_rate
        )
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.coin = coin
        self.start_date = start_date
        self.end_date = end_date
        self.candle_unit = candle_unit

        # 설정한 투자 기간 동안 동적으로 바뀔 변수들
        self.tmp_start_date = start_date
        self.tmp_end_date = end_date

    async def run(self):
        """
        전체 투자 프로세스를 순차적으로 실행

        1. 지정된 기간의 일부(예: 5%)만 우선 수집
        2. PriceAnalysisExpert로 가격 추세 리포트 생성
        3. 가격 추세 리포트 및 ReflectionExpert의 보고서를 참고하여 TradingExpert 매매 신호 생성
        4. 매매 실행 후 성찰 보고서를 갱신
        5. 반복 수행
        """
        print("-------------- 투자 시스템 시작 --------------")
        print("투자 시작 캔들:", self.start_date)
        print("투자 종료 캔들:", self.end_date)
        print("투자 대상 코인:", self.coin)
        print("캔들 단위:", self.candle_unit)
        print("초기 현금:", self.portfolio_manager.current_cash)

        # 1) 첫 수행 시, 지정된 기간의 일부(예: 5%)만 우선 수집
        self.tmp_end_date = await self._calculate_partial_end_date(
            start_date=self.start_date,
            end_date=self.end_date,
            portion=0.05,
            candle_unit=self.candle_unit,
        )

        while True:
            if self.tmp_end_date >= self.end_date:
                break

            data = await self.data_collector.collect_price_data(
                coin=self.coin,
                start_date=self.start_date,
                end_date=self.tmp_end_date,
                candle_unit=self.candle_unit,
            )

            # 2) 수집된 데이터 기반 가격 분석 리포트 생성
            analysis_report = await self.price_analysis_expert.analyze_trend(
                price_data=data
            )

            # 3) 분석 리포트 기반 매매 신호를 생성
            signal = await self.trading_expert.generate_signal(
                analysis_report=analysis_report
            )

            # 4) 다음 틱 데이터 수집
            self.tmp_start_date, self.tmp_end_date = await self.set_dates(
                self.tmp_end_date, self.candle_unit
            )
            price_data = await self.data_collector.collect_price_data(
                self.coin, self.tmp_start_date, self.tmp_end_date, self.candle_unit
            )

            # 5) 매매 실행
            if price_data:
                latest_open = price_data[-1]["open"]
                self.portfolio_manager.record_trade(
                    date=price_data[-1]["date"],
                    action=signal,
                    open_price=latest_open,
                )

            print(
                f"-------------- {self.tmp_end_date} 기준 포트폴리오 현황 --------------\n"
            )
            if signal == 1:
                print("Position: 매수")
            elif signal == 0:
                print("Position: 보유")
            else:
                print("Position: 매도")
            print(
                f"\n현금: {self.portfolio_manager.current_cash}, 코인 수량: {self.portfolio_manager.current_position}"
            )
            print("-------------------------------------------------------------------")

        if self.portfolio_manager.current_position > 0:
            # 전체 코인을 팔아서 현금화
            print(self.tmp_end_date)
            print(price_data[-1]["close"])
            self.portfolio_manager.record_trade(
                date=self.tmp_end_date,
                action=-1,
                open_price=price_data[-1]["close"],
            )

        # 수익률 계산
        total_profit = (
            (self.portfolio_manager.current_cash - self.initial_cash)
            / self.initial_cash
            * 100
        )
        print(f"멀티 에이전트 시스템 전략 수익률: {total_profit:.2f}")
        await self.backtest_buy_and_hold()

    async def _calculate_partial_end_date(
        self,
        start_date: str,
        end_date: str,
        portion: float,
        candle_unit: str,
    ) -> str:
        """
        시작일과 종료일을 바탕으로, candle 단위에 맞게 portion 비율만큼의 캔들을 포함하는 시점 계산, 문자열 반환

        Args:
            start_date (str): 시작 날짜 (예: "2024-10-01 09:00:00")
            end_date (str): 종료 날짜 (예: "2024-10-09 09:00:00")
            portion (float): 비율 (예: 0.05 -> 5%)
            candle_unit (str): 캔들 단위 (예: "1d", "1h", "15m", "5m", "1m")

        Returns:
            str: 부분 종료 날짜 (캔들 간격에 맞게 정렬됨)
        """
        fmt = "%Y-%m-%d %H:%M:%S"
        start_dt = datetime.strptime(start_date, fmt)
        end_dt = datetime.strptime(end_date, fmt)

        total_delta = end_dt - start_dt

        # candle 단위의 timedelta 생성 (TIME_DELTA_MAP 활용)
        candle_delta_args = TIME_DELTA_MAP.get(
            candle_unit, TIME_DELTA_MAP[DEFAULT_UNIT]
        )
        candle_interval = timedelta(**candle_delta_args)

        # 전체 캔들 수 (실수 값)
        total_candles = total_delta / candle_interval

        # portion에 해당하는 캔들 수 (정수로 내림)
        partial_candles = int(total_candles * portion)
        # 최소 1개의 캔들을 포함하도록 설정
        if partial_candles < 1:
            partial_candles = 1

        partial_end_dt = start_dt + candle_interval * partial_candles
        return partial_end_dt.strftime(fmt)

    async def set_dates(self, end_date: str, candle_unit: str) -> Tuple[str, str]:
        """
        시작일과 종료일을 설정합니다.

        Args:
            end_date (str): 종료 날짜
            candle_unit(str): 캔들 단위

        Returns:
            Tuple[str, str]: 시작일, 종료일
        """
        fmt = "%Y-%m-%d %H:%M:%S"

        start_dt = datetime.strptime(end_date, fmt)
        end_dt = datetime.strptime(end_date, fmt)

        delta_args = TIME_DELTA_MAP.get(candle_unit, TIME_DELTA_MAP[DEFAULT_UNIT])
        new_end_dt = end_dt + timedelta(**delta_args)
        return start_dt.strftime(fmt), new_end_dt.strftime(fmt)

    async def backtest_buy_and_hold(self):
        all_data = await self.data_collector.collect_price_data(
            coin=self.coin,
            start_date=self.start_date,
            end_date=self.end_date,
            candle_unit=self.candle_unit,
        )

        buy_amount = all_data[0]["open"]
        sell_amount = all_data[-1]["close"]
        # 수익률 계산
        profit = (sell_amount - buy_amount) / buy_amount * 100
        print(f"Buy and Hold 전략 수익률: {profit:.2f}%")


class AsyncCryptoTradingSystem(CryptoTradingSystem):
    def __init__(
        self,
        initial_cash: int,
        fee_rate: float,
        coin: str,
        start_date: str,
        end_date: str,
        candle_unit: str,
    ):
        super().__init__(
            initial_cash, fee_rate, coin, start_date, end_date, candle_unit
        )

    def run(self):
        asyncio.run(super().run())


def create_system(
    initial_cash: int,
    fee_rate: float,
    coin: str,
    start_date: str,
    end_date: str,
    candle_unit: str,
):
    load_dotenv()

    return AsyncCryptoTradingSystem(
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        coin=coin,
        start_date=start_date,
        end_date=end_date,
        candle_unit=candle_unit,
    )
