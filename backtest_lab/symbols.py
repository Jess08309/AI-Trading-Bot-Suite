"""OSI option symbol construction + standard monthly expiry calendar.

OSI format: {ROOT (padded/truncated, no padding needed for <=5 char tickers used here)}
            {YYMMDD}{C or P}{strike * 1000, zero-padded to 8 digits}
"""
from datetime import date, timedelta


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(days=14)


def monthly_expiries_near(target_date: date, count_before: int = 2, count_after: int = 4) -> list:
    """Standard monthly (3rd-Friday) expiries spanning a window around target_date."""
    expiries = []
    y, m = target_date.year, target_date.month
    for offset in range(-count_before, count_after + 1):
        total = (y * 12 + (m - 1)) + offset
        yy, mm = divmod(total, 12)
        expiries.append(third_friday(yy, mm + 1))
    return sorted(set(expiries))


def pick_expiry_for_dte(entry_date: date, target_dte: int, min_dte: int, max_dte: int):
    """Pick the standard monthly expiry whose DTE from entry_date is closest to target_dte,
    constrained to [min_dte, max_dte]."""
    candidates = monthly_expiries_near(entry_date)
    in_range = [e for e in candidates if min_dte <= (e - entry_date).days <= max_dte]
    pool = in_range or candidates
    return min(pool, key=lambda e: abs((e - entry_date).days - target_dte))


def osi_symbol(root: str, expiry: date, right: str, strike: float) -> str:
    yy = expiry.strftime("%y%m%d")
    cp = "C" if right == "call" else "P"
    strike_int = round(strike * 1000)
    return f"{root}{yy}{cp}{strike_int:08d}"
