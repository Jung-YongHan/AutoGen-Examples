import os

from autogen_ext.models.openai import OpenAIChatCompletionClient

from multi_agent_system.trading_system.system.crypto_trading_system import (
    create_system,
)

app = create_system(
    system_name="trading_system",
    initial_cash=10_000_000,
    fee_rate=0.08,
    coin="KRW-BTC",
    start_date="2024-09-01 09:00:00",
    end_date="2024-12-31 09:00:00",
    candle_unit="1d",
    price_analysis_expert_model=OpenAIChatCompletionClient(
        model=os.getenv("PRICE_ANALYSIS_EXPERT_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    trading_expert_model=OpenAIChatCompletionClient(
        model=os.getenv("TRADING_EXPERT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    ),
)

if __name__ == "__main__":
    app.run()
