import asyncio
from datetime import datetime, timedelta
from typing import List, Dict

import requests

from multi_agent_system.trading_system.core.constants import (
    UNIT_MAP,
    DEFAULT_UNIT,
    TIME_DELTA_MAP,
)


class DataCollector:

    def __init__(self):
        self.collected_data: List[Dict] = []

    async def collect_price_data(
        self, coin: str, start_date: str, end_date: str, candle_unit: str
    ) -> List[Dict]:
        """
        지정된 기간 동안 특정 코인 가격 데이터를 외부 거래소(예: Upbit)에서 수집.

        Args:
            coin (str): 예) "KRW-BTC"
            start_date (str): 시작 날짜 (예: "2020-10-10")
            end_date (str): 종료 날짜 (예: "2024-10-09")
            candle_unit (str): 캔들 단위 (예: "1d", "1h" 등)

        Returns:
            List[Dict]: 수집된 가격 데이터가 담긴 리스트
        """

        base_url = "https://api.upbit.com/v1/candles"

        candle_type = UNIT_MAP.get(candle_unit, "days")
        delta_kwargs = TIME_DELTA_MAP.get(candle_unit, TIME_DELTA_MAP[DEFAULT_UNIT])

        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

        now_dt = datetime.now()
        if end_dt > now_dt:
            end_dt = now_dt

        # 만약 start_dt가 end_dt보다 미래라면, 데이터가 없음
        if start_dt > end_dt:
            print("[Warning] 시작일이 종료일보다 미래입니다. 수집할 데이터가 없습니다.")
            return self.collected_data

        # 중복 데이터 제거를 위한 집합
        existing_dates = {d["date"] for d in self.collected_data}

        current_dt = end_dt
        while True:
            # 루프 탈출 조건
            if current_dt < start_dt:
                break

            to_param = current_dt.strftime("%Y-%m-%d %H:%M:%S")

            # 남은 일수(또는 시간, 분)를 구해서 count 결정
            # (예: 1일봉일 때 남은 일이 5일 뿐이면, count=5로 요청)
            days_diff = (current_dt.date() - start_dt.date()).days + 1
            # 200보다 작으면 그대로, 크면 200
            n_candles_to_fetch = min(days_diff, 200)

            # 최대 200개씩 받을 수 있으므로 count=200
            url = (
                f"{base_url}/{candle_type}"
                f"?market={coin}"
                f"&to={to_param}"
                f"&count={n_candles_to_fetch}"
            )
            response = requests.get(url)
            if response.status_code != 200:
                print(f"[Error] {response.status_code} / {response.text}")
                break

            candles = response.json()
            if not candles:
                break
            candles.reverse()

            for c in candles:
                kst_time = c["candle_date_time_kst"]  # 예) "2024-10-09T09:00:00"
                if kst_time not in existing_dates:
                    self.collected_data.append(
                        {
                            "date": kst_time,
                            "open": c["opening_price"],
                            "close": c["trade_price"],
                            "high": c["high_price"],
                            "low": c["low_price"],
                            "volume": c["candle_acc_trade_volume"],
                        }
                    )
                    existing_dates.add(kst_time)

            oldest_kst_str = candles[0]["candle_date_time_kst"]
            oldest_kst_dt = datetime.strptime(oldest_kst_str, "%Y-%m-%dT%H:%M:%S")
            current_dt = oldest_kst_dt - timedelta(**delta_kwargs)

        self.collected_data.sort(
            key=lambda x: datetime.strptime(x["date"], "%Y-%m-%dT%H:%M:%S")
        )
        return self.collected_data


if __name__ == "__main__":
    collector = DataCollector()
    data = asyncio.run(
        collector.collect_price_data(
            "KRW-BTC", "2024-10-01 09:00:00", "2024-10-09 09:00:00", "1d"
        )
    )
    print(data)
