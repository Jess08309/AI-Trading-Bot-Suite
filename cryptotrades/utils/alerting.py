"""
Discord webhook alerting for the crypto trading bot.

Sends notifications for:
- Circuit breaker triggers
- Large losses
- Bot startup/shutdown
- Error conditions

Set DISCORD_WEBHOOK_URL in .env to enable.
"""
from __future__ import annotations
import os
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Webhook URL from environment
_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")


def set_webhook_url(url: str):
    """Set webhook URL at runtime (e.g., from config)."""
    global _webhook_url
    _webhook_url = url


def _send_discord(content: str, username: str = "CryptoBot"):
    """Send a message to Discord webhook (non-blocking)."""
    if not _webhook_url:
        return

    def _do_send():
        try:
            payload = {
                "username": username,
                "content": content,
            }
            resp = requests.post(_webhook_url, json=payload, timeout=10)
            if resp.status_code not in (200, 204):
                logger.debug(f"Discord webhook returned {resp.status_code}")
        except Exception as e:
            logger.debug(f"Discord webhook error: {e}")

    # Send in background thread so it doesn't block trading
    threading.Thread(target=_do_send, daemon=True).start()


def alert_circuit_breaker(market: str, reason: str):
    """Alert when circuit breaker triggers."""
    _send_discord(
        f"🚨 **CIRCUIT BREAKER TRIGGERED** ({market.upper()})\n"
        f"```{reason}```\n"
        f"Trading paused. Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )


def alert_large_loss(symbol: str, loss_pct: float, loss_usd: float):
    """Alert on a large individual trade loss."""
    _send_discord(
        f"📉 **LARGE LOSS** on {symbol}\n"
        f"Loss: {loss_pct:+.2f}% (${loss_usd:+,.2f})\n"
        f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )


def alert_bot_startup(spot_positions: int, futures_positions: int,
                      spot_balance: float, futures_balance: float):
    """Alert on bot startup."""
    _send_discord(
        f"✅ **Bot Started**\n"
        f"Positions: {spot_positions} spot, {futures_positions} futures\n"
        f"Balance: ${spot_balance:,.2f} spot, ${futures_balance:,.2f} futures\n"
        f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )


def alert_bot_shutdown(cycle_count: int, spot_positions: int,
                       futures_positions: int):
    """Alert on bot shutdown."""
    _send_discord(
        f"🛑 **Bot Stopped** after {cycle_count} cycles\n"
        f"Open positions: {spot_positions} spot, {futures_positions} futures\n"
        f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )


def alert_error(error_msg: str, context: str = ""):
    """Alert on a critical error."""
    _send_discord(
        f"⚠️ **ERROR** {context}\n"
        f"```{error_msg[:500]}```\n"
        f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )


def alert_trade(symbol: str, side: str, price: float, size: float,
                profit_pct: Optional[float] = None):
    """Alert on trade execution (optional, can be noisy)."""
    emoji = "🟢" if side.upper() in ("BUY", "LONG") else "🔴"
    msg = f"{emoji} **{side.upper()}** {symbol} @ ${price:,.2f} (${size:,.2f})"
    if profit_pct is not None:
        msg += f"\nP/L: {profit_pct:+.2f}%"
    _send_discord(msg)


def is_configured() -> bool:
    """Check if Discord alerting is configured."""
    return bool(_webhook_url)
