class PortfolioManager:
    """
    매매 기록 및 현재 보유 포지션을 관리 및 기록 업데이트.

    Attributes:
        trade_history (List[Dict]): 매매 기록
        current_position (int): 현재 보유 수량(코인 단위)
        current_cash (float): 가용 현금
    """

    def __init__(self, initial_cash: float = 10_000_000, fee_rate: float = 0.08):
        """
        PortfolioManager 클래스의 초기화 메서드입니다.

        Args:
            initial_cash (float): 초기 가용 현금
        """
        self.trade_history = []
        self.current_position = 0
        self.current_cash = initial_cash
        self.fee_rate = fee_rate / 100  # 수수료율 (예: 0.08% -> 0.0008)

    def record_trade(self, date: str, action: int, open_price: float):
        """
        매매 액션(매수, 보유, 매도)에 따라 포트폴리오 상태와 매매 기록을 업데이트합니다.

        매수(BUY) 로직:
            - 현재 현금(all-in)으로 살 수 있는 만큼 코인을 구매
            - 단, 매수 금액에 수수료 0.08%를 포함
            - 예: 현재 잔액 = 1,000,000, 수수료율 = 0.0008
              -> 실제 코인 매수금 cost_of_coins = current_cash / (1 + fee_rate)
              -> 매수 가능한 코인 개수 = cost_of_coins / open_price
              -> 수수료 = cost_of_coins * fee_rate
              -> 매수 후 current_cash = 0 (부족한 잔액이 남을 수도 있으나, 여기서는 전액 매수로 가정)

        매도(SELL) 로직:
            - 보유 코인을 전부 매도(100%)
            - 매도 받은 금액에서 수수료 0.08% 공제
            - proceeds = current_position * open_price
            - fee = proceeds * fee_rate
            - net = proceeds - fee
            - current_position = 0
            - current_cash += net

        보유(HOLD) 로직:
            - 아무것도 하지 않음.

        Args:
            date (str): 매매 일자
            action (int): 1(매수), 0(보유), -1(매도)
            open_price (float): 매매 시점의 코인 1개당 가격
        """
        if action == 1:  # BUY
            if self.current_cash > 0:
                # (1) 총 투자 금액에서 수수료를 고려해, 실제 코인 매수에 쓸 수 있는 금액을 계산
                cost_of_coins = self.current_cash / (1 + self.fee_rate)
                # (2) 매수 가능한 코인 수량 (소수점까지)
                coins_bought = cost_of_coins / open_price
                # (3) 수수료 = cost_of_coins * fee_rate
                fee_amount = cost_of_coins * self.fee_rate
                # (4) 총 지불 금액
                total_spent = cost_of_coins + fee_amount  # == self.current_cash

                # (5) 포트폴리오 업데이트
                self.current_position += coins_bought
                self.current_cash -= total_spent  # 여기서는 전액 소진(0이 됨)

                # (6) 거래 기록
                self.trade_history.append(
                    {
                        "date": date,
                        "action": "BUY",
                        "trade_price": open_price,
                        "coins_traded": coins_bought,
                        "fee_rate": self.fee_rate,
                        "fee_amount": fee_amount,
                        "spent_amount": total_spent,
                        "current_position": self.current_position,
                        "current_cash": self.current_cash,
                    }
                )

        elif action == -1:  # SELL
            if self.current_position > 0:
                # (1) 전량 매도한다고 가정
                proceeds = self.current_position * open_price
                fee_amount = proceeds * self.fee_rate
                net_after_fee = proceeds - fee_amount

                # (2) 포트폴리오 업데이트
                coins_sold = self.current_position
                self.current_position = 0.0
                self.current_cash += net_after_fee

                # (3) 거래 기록
                self.trade_history.append(
                    {
                        "date": date,
                        "action": "SELL",
                        "trade_price": open_price,
                        "coins_traded": coins_sold,
                        "fee_rate": self.fee_rate,
                        "fee_amount": fee_amount,
                        "gross_received": proceeds,
                        "net_after_fee": net_after_fee,
                        "current_position": self.current_position,
                        "current_cash": self.current_cash,
                    }
                )

        else:  # 0 == HOLD
            # 아무것도 하지 않고 기록만 남김
            self.trade_history.append(
                {
                    "date": date,
                    "action": "HOLD",
                    "open_price": open_price,
                    "current_position": self.current_position,
                    "current_cash": self.current_cash,
                }
            )


if __name__ == "__main__":
    # 포트폴리오 매니저 객체 생성
    pm = PortfolioManager()

    # 매수 시나리오
    pm.record_trade(date="2021-01-01 09:00:00", action=1, open_price=1000)

    # 매도 시나리오
    pm.record_trade(date="2021-01-02 09:00:00", action=-1, open_price=1100)

    # 보유 시나리오
    pm.record_trade(date="2021-01-03 09:00:00", action=0, open_price=1100)

    # 매수 시나리오
    pm.record_trade(date="2021-01-04 09:00:00", action=1, open_price=1200)

    # 매도 시나리오
    pm.record_trade(date="2021-01-05 09:00:00", action=-1, open_price=1300)

    # 보유 시나리오
    pm.record_trade(date="2021-01-06 09:00:00", action=0, open_price=1300)

    print("\n=== Trade History ===")
    for record in pm.trade_history:
        print(record)
