import requests
from datetime import datetime, timedelta, date


def get_recent_day():
    """오늘 날짜를 반환합니다."""
    return date.today()


def search_price_data(start: str, end: str, market: str = "KRW-BTC"):
    """
    업비트 API를 사용하여 특정 기간의 일봉 가격 데이터를 가져옵니다.

    Parameters:
        start (str): 시작 시각 ("YYYY-MM-DD HH:MM:SS")
        end (str): 종료 시각 ("YYYY-MM-DD HH:MM:SS")
        market (str): 조회할 시장 코드 (예: "KRW-BTC")

    Returns:
        List[dict]: 기간 내의 일봉 데이터 리스트 (시간 오름차순)
    """
    url = "https://api.upbit.com/v1/candles/days"
    count = 200  # API에서 한 번에 가져올 최대 캔들 수
    data = []

    # 문자열을 datetime 객체로 변환
    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")

    # 초기 'to' 파라미터는 종료 시각으로 설정
    to = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    while True:
        params = {"market": market, "to": to, "count": count}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("API 요청 실패:", response.status_code)
            break

        candles = response.json()
        if not candles:
            break

        for candle in candles:
            # API에서 반환하는 시간은 ISO 8601 포맷 (예: "2025-03-23T00:00:00")
            candle_time = datetime.strptime(
                candle["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S"
            )
            # 지정한 기간 내의 데이터만 추가
            if start_dt <= candle_time <= end_dt:
                data.append(candle)

        # 마지막(가장 오래된) 캔들의 시간 확인
        oldest_candle = candles[-1]
        oldest_time = datetime.strptime(
            oldest_candle["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S"
        )

        # 시작일보다 이전 데이터까지 도달한 경우 종료
        if oldest_time <= start_dt:
            break

        # 중복 데이터를 방지하기 위해, 마지막 캔들보다 1초 빠른 시간으로 'to' 업데이트
        new_to = oldest_time - timedelta(seconds=1)
        to = new_to.strftime("%Y-%m-%d %H:%M:%S")

    # 시간 순서가 내림차순이므로, 오름차순으로 정렬 후 반환
    data.sort(key=lambda x: x["candle_date_time_utc"])
    return data


# 사용 예시
if __name__ == "__main__":
    start = "2023-12-31 00:00:00"
    end = "2024-12-31 00:00:00"

    price_data = search_price_data(start, end)
    print(price_data)
