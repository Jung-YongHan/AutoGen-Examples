UNIT_MAP = {
    "1d": "days",
    "1h": "minutes/60",
    "15m": "minutes/15",
    "5m": "minutes/5",
    "1m": "minutes/1",
}

# 캔들 단위별 timedelta 생성용 매핑
TIME_DELTA_MAP = {
    "1d": dict(days=1),
    "1h": dict(hours=1),
    "15m": dict(minutes=15),
    "5m": dict(minutes=5),
    "1m": dict(minutes=1),
}

# 필요하다면 추가적인 상수나 매핑, 기본값 등도 같이 정의...
DEFAULT_UNIT = "1d"
