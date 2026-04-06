from app.bootstrap import load_runtime_config
from trading.exchange.bybit_adapter import BybitAdapter


def main():
    cfg = load_runtime_config()
    adapter = BybitAdapter(cfg.adapter)
    try:
        details = adapter.get_account_mode_details()
        wallet = adapter.client.request_private(
            "GET",
            "/v5/account/wallet-balance",
            params={"accountType": "UNIFIED", "coin": "USDT"},
        )
        account = adapter.get_account()
        print(
            {
                "mode": cfg.mode,
                "testnet": cfg.adapter.testnet,
                "demo": cfg.adapter.demo,
                "retCode": details.get("retCode"),
                "retMsg": details.get("retMsg"),
                "unifiedMarginStatus": details.get("unifiedMarginStatus"),
                "wallet_retCode": wallet.get("retCode"),
                "wallet_retMsg": wallet.get("retMsg"),
                "wallet_result": wallet.get("result"),
                "equity": account.equity_usdt,
                "available": account.available_balance_usdt,
                "private_auth_invalid": adapter.private_auth_invalid,
                "private_auth_invalid_reason": adapter.private_auth_invalid_reason,
            }
        )
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
