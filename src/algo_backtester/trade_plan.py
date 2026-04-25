from algo_backtester.options_engine import OptionsRecommendation


def build_trade_plan(
        signal: str,
        entry_price: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        trailing_stop_pct: float,
) -> dict:
    if signal == "BUY":
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        take_profit_price = entry_price * (1 + take_profit_pct)
        initial_trailing_stop = entry_price * (1 - trailing_stop_pct)

        risk_per_share = entry_price - stop_loss_price
        reward_per_share = take_profit_price - entry_price
        reward_risk = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

        return {
            "Signal": signal,
            "HasActionableTrade": True,
            "EstimatedEntryReference": round(entry_price, 2),
            "ActualEntry": "Next market open",
            "StopLoss": round(stop_loss_price, 2),
            "TakeProfit": round(take_profit_price, 2),
            "InitialTrailingStop": round(initial_trailing_stop, 2),
            "RiskPerShare": round(risk_per_share, 2),
            "RewardPerShare": round(reward_per_share, 2),
            "RewardRisk": round(reward_risk, 2),
        }

    if signal == "BEARISH_ENTRY":
        stop_loss_price = entry_price * (1 + stop_loss_pct)
        take_profit_price = entry_price * (1 - take_profit_pct)
        initial_trailing_stop = entry_price * (1 + trailing_stop_pct)

        risk_per_share = stop_loss_price - entry_price
        reward_per_share = entry_price - take_profit_price
        reward_risk = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

        return {
            "Signal": signal,
            "HasActionableTrade": True,
            "EstimatedEntryReference": round(entry_price, 2),
            "ActualEntry": "Next market open",
            "StopLoss": round(stop_loss_price, 2),
            "TakeProfit": round(take_profit_price, 2),
            "InitialTrailingStop": round(initial_trailing_stop, 2),
            "RiskPerShare": round(risk_per_share, 2),
            "RewardPerShare": round(reward_per_share, 2),
            "RewardRisk": round(reward_risk, 2),
        }

    return {
        "Signal": signal,
        "HasActionableTrade": False,
        "TradePlan": "No new entry trade plan because signal is not BUY or BEARISH_ENTRY.",
    }


def build_signal_interpretation(
        latest_signal: dict,
        trade_plan: dict,
        options_rec: OptionsRecommendation,
) -> dict:
    signal = str(latest_signal.get("Signal", "NO_DATA"))
    reason = str(latest_signal.get("Reason", "No signal reason available."))

    interpretation = {
        "Signal": signal,
        "Meaning": "",
        "SystemSees": "",
        "BacktesterSimulates": "",
        "WhatToDoToday": "",
        "HowOptionsFit": "",
        "PlannedTradeLevels": [],
        "RiskReward": [],
        "OptionsDetails": [],
    }

    if signal == "BUY":
        interpretation["Meaning"] = "A bullish pullback setup has been confirmed."
        interpretation["SystemSees"] = (
            "The stock is in an uptrend, pulled back in a controlled way, "
            f"and resumed upward momentum with confirmation. Trigger reason: {reason}."
        )
        interpretation["BacktesterSimulates"] = (
            "Buy shares of stock at the next session open and hold until an exit condition triggers."
        )
        interpretation["WhatToDoToday"] = "Prepare a long trade for the next session open."
        interpretation["HowOptionsFit"] = (
            "Instead of buying shares, this bullish thesis can be expressed with a call debit spread."
        )
        interpretation["PlannedTradeLevels"] = [
            f"Estimated Buy Reference: ${trade_plan['EstimatedEntryReference']:.2f}",
            f"Actual Buy Price: {trade_plan['ActualEntry']}",
            f"Stop Loss: ${trade_plan['StopLoss']:.2f}",
            f"Take Profit: ${trade_plan['TakeProfit']:.2f}",
            f"Initial Trailing Stop: ${trade_plan['InitialTrailingStop']:.2f}",
        ]
        interpretation["RiskReward"] = [
            f"Risk per Share: ${trade_plan['RiskPerShare']:.2f}",
            f"Reward per Share: ${trade_plan['RewardPerShare']:.2f}",
            f"Reward/Risk: {trade_plan['RewardRisk']:.2f}",
        ]
    elif signal == "EXIT_LONG":
        interpretation["Meaning"] = "The bullish trade is no longer valid."
        interpretation["SystemSees"] = f"The existing long setup weakened enough to trigger an exit. Trigger reason: {reason}."
        interpretation["BacktesterSimulates"] = "Sell shares at the next session open."
        interpretation["WhatToDoToday"] = "Exit long stock next session open and close bullish options exposure."
        interpretation["HowOptionsFit"] = (
            "Close bullish call exposure. Do not open bearish options unless a separate BEARISH_ENTRY appears."
        )
        interpretation["PlannedTradeLevels"] = [
            "Exit long next session open.",
            "Do not add new bullish exposure.",
        ]
    elif signal == "BEARISH_ENTRY":
        interpretation["Meaning"] = "A bearish setup has been confirmed."
        interpretation["SystemSees"] = f"Trend and momentum conditions now favor downside. Trigger reason: {reason}."
        interpretation["BacktesterSimulates"] = (
            "The backtester flags a bearish setup, but it does not yet simulate short stock positions."
        )
        interpretation["WhatToDoToday"] = "Prepare a bearish thesis for the next session open if you trade this setup."
        interpretation["HowOptionsFit"] = "This bearish thesis can be expressed with a put debit spread."
        if trade_plan.get("HasActionableTrade"):
            interpretation["PlannedTradeLevels"] = [
                f"Estimated Short Reference: ${trade_plan['EstimatedEntryReference']:.2f}",
                f"Actual Entry: {trade_plan['ActualEntry']}",
                f"Stop Loss: ${trade_plan['StopLoss']:.2f}",
                f"Take Profit: ${trade_plan['TakeProfit']:.2f}",
                f"Initial Trailing Stop: ${trade_plan['InitialTrailingStop']:.2f}",
            ]
            interpretation["RiskReward"] = [
                f"Risk per Share: ${trade_plan['RiskPerShare']:.2f}",
                f"Reward per Share: ${trade_plan['RewardPerShare']:.2f}",
                f"Reward/Risk: {trade_plan['RewardRisk']:.2f}",
            ]
        else:
            interpretation["PlannedTradeLevels"] = [
                "No short-stock stop or profit target is printed yet.",
                "Bearish stock execution rules are not backtested in the current engine.",
            ]
    elif signal == "HOLD_POSITION":
        interpretation["Meaning"] = "The existing long position remains valid."
        interpretation["SystemSees"] = f"No exit condition fired, so the long trade is still active. Current state: {reason}."
        interpretation["BacktesterSimulates"] = "Continue holding the existing long stock position."
        interpretation["WhatToDoToday"] = "Do not open a new position. Keep managing the active long using the exit rules."
        interpretation["HowOptionsFit"] = "No new options entry is created from HOLD_POSITION."
        entry_price = float(latest_signal.get("EntryPrice", 0.0))
        initial_stop = float(latest_signal.get("InitialStopPrice", 0.0))
        if entry_price > 0:
            interpretation["PlannedTradeLevels"] = [
                f"Current Entry Price: ${entry_price:.2f}",
                f"Initial Stop Reference: ${initial_stop:.2f}",
                "Next action: continue holding until an exit trigger appears.",
            ]
    else:
        interpretation["Meaning"] = "No actionable trade setup today."
        interpretation["SystemSees"] = (
            "The current conditions are not fully confirmed for a new entry or exit. "
            f"Current state: {reason}."
        )
        interpretation["BacktesterSimulates"] = "No new stock position is opened from this signal."
        interpretation["WhatToDoToday"] = "Do nothing. Wait for a confirmed BUY or BEARISH_ENTRY signal."
        interpretation["HowOptionsFit"] = "No options trade. Options are only used after a valid directional signal appears."
        interpretation["PlannedTradeLevels"] = [
            "No active trade setup.",
            "No stop loss.",
            "No take profit.",
        ]

    if options_rec.options_action != "NO_OPTIONS_TRADE":
        interpretation["OptionsDetails"] = [
            f"Structure: {options_rec.structure}",
            f"Expiration: {options_rec.expiration} ({options_rec.dte} DTE)",
            f"Long Strike: {options_rec.long_strike}",
            f"Short Strike: {options_rec.short_strike}",
            f"Estimated Debit: ${options_rec.estimated_debit:.2f}",
            f"Max Loss: ${options_rec.max_loss:.2f}",
            f"Max Profit: ${options_rec.max_profit:.2f}",
            f"Reason: {options_rec.reason}",
        ]
    elif signal in {"BUY", "BEARISH_ENTRY"}:
        interpretation["OptionsDetails"] = [f"No options trade should be placed yet. Reason: {options_rec.reason}"]

    return interpretation


def print_signal_interpretation(interpretation: dict) -> None:
    print("\nSignal Interpretation")
    print("---------------------")
    print(f"Current signal: {interpretation['Signal']}")
    print("\nWhat this means:")
    print(interpretation["Meaning"])
    print("\nWhat the system sees:")
    print(interpretation["SystemSees"])
    print("\nWhat the backtester simulates:")
    print(interpretation["BacktesterSimulates"])
    print("\nWhat to do today:")
    print(interpretation["WhatToDoToday"])

    if interpretation.get("PlannedTradeLevels"):
        print("\nPlanned Trade Levels")
        print("--------------------")
        for line in interpretation["PlannedTradeLevels"]:
            print(line)

    if interpretation.get("RiskReward"):
        print("\nRisk/Reward")
        print("-----------")
        for line in interpretation["RiskReward"]:
            print(line)

    print("\nHow options fit:")
    print(interpretation["HowOptionsFit"])

    if interpretation.get("OptionsDetails"):
        print("\nOptions Details")
        print("---------------")
        for line in interpretation["OptionsDetails"]:
            print(line)
