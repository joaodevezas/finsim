#!/usr/bin/env python3
"""
generate_data.py -- fetches real company data for the `fin` interpreter.

The Rust interpreter shells out to this script whenever it evaluates
`t<TICKER>` in a .fi program (e.g. `company = t<APPL>`). This version
pulls heavily expanded metrics from Yahoo Finance via the `yfinance` package.

The JSON shape is the contract the interpreter relies on:

    {
      "<field>": { "<subset>": <number>, ... },
      ...
    }

WHAT EACH SUBSET MEANS
-----------------------
For standard financial statements (revenue, earnings, cash flows):
    last -> most recent reported quarter
    ttm  -> sum of the last 4 reported quarters (trailing twelve months)
    avg  -> mean of all available reported quarters
    ytd  -> sum of quarters reported within the current calendar year

For balance sheet snapshots (assets, debt, equity) and shares:
    last -> most recent reported quarter
    avg  -> mean over available history
    yoy  -> percentage change from the same quarter 1 year ago

For growth metrics (eps_growth, fcf_growth, revenue_growth, earnings_growth):
    1y   -> (Most Recent Year - Previous Year) / Previous Year
    2y   -> (1 Year Ago - 2 Years Ago) / 2 Years Ago
    ... up to 5y, plus an "avg" of available historical yearly growth rates.

FALLBACK BEHAVIOR
------------------
Every individual number that can't be found or computed is written as 0.0. 
A single missing line item, a bad ticker, or a Yahoo hiccup never stops the 
JSON file from being written.

Usage:
    python3 generate_data.py TICKER [OUTPUT_DIR]

Requires: pip install yfinance pandas
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print(
        "error: the 'yfinance' package is not installed.\n"
        "Install it with: pip install yfinance pandas",
        file=sys.stderr,
    )
    sys.exit(1)

import pandas as pd

SUBSETS_STD = ["last", "ttm", "avg", "ytd"]
SUBSETS_SNAPSHOT = ["last", "avg", "yoy"]
SUBSETS_GROWTH = ["1y", "2y", "3y", "4y", "5y", "avg"]

# Statement Row Spellings across yfinance versions
REVENUE_ROWS = ["Total Revenue", "TotalRevenue", "Operating Revenue"]
EARNINGS_ROWS = ["Net Income", "NetIncome", "Net Income Common Stockholders"]
GROSS_PROFIT_ROWS = ["Gross Profit", "GrossProfit"]
OPERATING_INC_ROWS = ["Operating Income", "OperatingIncome"]
EBITDA_ROWS = ["EBITDA", "Normalized EBITDA"]
EPS_ROWS = ["Basic EPS", "Diluted EPS", "EPS"]

FCF_ROWS = ["Free Cash Flow", "FreeCashFlow"]
OCF_ROWS = ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
CAPEX_ROWS = ["Capital Expenditure", "CapitalExpenditure"]

ASSETS_ROWS = ["Total Assets"]
LIABILITIES_ROWS = ["Total Liabilities Net Minority Interest", "Total Liabilities"]
DEBT_ROWS = ["Total Debt"]
EQUITY_ROWS = ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]


def _safe_float(value, default: float = 0.0) -> float:
    """Coerce to float, falling back to `default` for None/NaN/junk values."""
    try:
        if value is None:
            return default
        f = float(value)
        if f != f:  # NaN != NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


def _sort_desc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensively sort a statement's columns most-recent-first. The code
    below assumes column 0 is "the most recent period" everywhere (that's
    what `last` means) -- rather than trusting yfinance to always hand
    columns back in that order for every ticker/version, sort explicitly.
    """
    if df is None or df.empty:
        return df
    try:
        return df.sort_index(axis=1, ascending=False)
    except Exception:
        return df


def _select_row(df: pd.DataFrame, row_names: list):
    """
    Look up the first matching row name in `df`, guarding against a classic
    pandas gotcha: if a statement has the same line item listed twice
    (duplicate index labels), `df.loc[name]` returns a DataFrame instead of 
    a Series. Take the first occurrence instead.
    """
    for name in row_names:
        if name in df.index:
            row = df.loc[name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row
    return None


def _fetch_shares(ticker: yf.Ticker, info: dict) -> dict:
    """
    Fetches actual shares outstanding history using get_shares_full(), 
    falling back to the sharesOutstanding snapshot in info. 
    Shares don't sum over periods (unlike revenue) - we average them for TTM/YTD.
    """
    try:
        shares = ticker.get_shares_full()
        if shares is not None and not shares.empty:
            last = _safe_float(shares.iloc[-1])
            avg = _safe_float(shares.mean())
            
            now = pd.Timestamp.now(tz=shares.index.tz)
            
            # TTM Average
            one_year_ago = now - pd.Timedelta(days=365)
            ttm_series = shares[shares.index >= one_year_ago]
            ttm = _safe_float(ttm_series.mean()) if not ttm_series.empty else last
            
            # YTD Average
            start_of_year = pd.Timestamp(year=now.year, month=1, day=1, tz=shares.index.tz)
            ytd_series = shares[shares.index >= start_of_year]
            ytd = _safe_float(ytd_series.mean()) if not ytd_series.empty else last
            
            return {
                "last": round(last, 4),
                "ttm": round(ttm, 4),
                "avg": round(avg, 4),
                "ytd": round(ytd, 4),
            }
    except Exception as e:
        print(f"warning: could not fetch full shares history: {e}", file=sys.stderr)
        
    # Fallback to info snapshot if time-series fails
    authoritative = _safe_float(info.get("sharesOutstanding", 0.0))
    return {k: round(authoritative, 4) for k in SUBSETS_STD}


def _price_or_volume_stats(history: pd.DataFrame, column: str) -> dict:
    zeros = {k: 0.0 for k in SUBSETS_STD}
    if history is None or history.empty or column not in history.columns:
        return zeros

    series = history[column].dropna()
    if series.empty:
        return zeros

    last = _safe_float(series.iloc[-1])
    avg = _safe_float(series.mean())

    now = pd.Timestamp.now(tz=series.index.tz)
    one_year_ago = now - pd.Timedelta(days=365)
    ttm_series = series[series.index >= one_year_ago]
    ttm = _safe_float(ttm_series.mean()) if not ttm_series.empty else 0.0

    start_of_year = pd.Timestamp(year=now.year, month=1, day=1, tz=series.index.tz)
    ytd_series = series[series.index >= start_of_year]
    ytd = _safe_float(ytd_series.mean()) if not ytd_series.empty else 0.0

    return {
        "last": round(last, 4),
        "ttm": round(ttm, 4),
        "avg": round(avg, 4),
        "ytd": round(ytd, 4),
    }


def _statement_stats(quarterly: pd.DataFrame, row_names: list, average_instead_of_sum: bool = False) -> dict:
    """Standard stats for income/cash flow statements (sums for TTM/YTD unless average_instead_of_sum=True)"""
    zeros = {k: 0.0 for k in SUBSETS_STD}
    if quarterly is None or quarterly.empty:
        return zeros

    row = _select_row(quarterly, row_names)
    if row is None:
        return zeros

    row = row.dropna()
    if row.empty:
        return zeros

    values = [_safe_float(v) for v in row.tolist()]
    dates = list(row.index)

    last = values[0]
    avg = sum(values) / len(values) if values else 0.0

    ttm_values = values[:4]
    if average_instead_of_sum:
        ttm = sum(ttm_values) / len(ttm_values) if ttm_values else 0.0
    else:
        ttm = sum(ttm_values)

    current_year = datetime.now(timezone.utc).year
    ytd_values = [v for v, d in zip(values, dates) if getattr(d, "year", None) == current_year]
    
    if average_instead_of_sum:
        ytd = sum(ytd_values) / len(ytd_values) if ytd_values else 0.0
    else:
        ytd = sum(ytd_values)

    return {
        "last": round(last, 4),
        "ttm": round(ttm, 4),
        "avg": round(avg, 4),
        "ytd": round(ytd, 4),
    }


def _snapshot_stats(quarterly: pd.DataFrame, row_names: list) -> dict:
    """For Balance Sheet items where summing TTM/YTD makes no sense (they are point-in-time snapshots)"""
    zeros = {k: 0.0 for k in SUBSETS_SNAPSHOT}
    if quarterly is None or quarterly.empty:
        return zeros

    row = _select_row(quarterly, row_names)
    if row is None:
        return zeros

    row = row.dropna()
    if row.empty:
        return zeros

    values = [_safe_float(v) for v in row.tolist()]
    last = values[0]
    avg = sum(values) / len(values) if values else 0.0

    # Year over Year change (most recent quarter vs same quarter 1 year ago, which is index 4 typically)
    yoy = 0.0
    if len(values) >= 5 and values[4] != 0:
        yoy = (values[0] - values[4]) / abs(values[4])

    return {
        "last": round(last, 4),
        "avg": round(avg, 4),
        "yoy": round(yoy, 4),
    }


def _growth_stats(annual: pd.DataFrame, row_names: list) -> dict:
    """Calculates yearly percentage growth steps going back up to 5 years."""
    zeros = {k: 0.0 for k in SUBSETS_GROWTH}
    if annual is None or annual.empty:
        return zeros

    row = _select_row(annual, row_names)
    if row is None:
        return zeros

    row = row.dropna()
    if len(row) < 2:
        return zeros

    # Values are newest-first
    values = [_safe_float(v) for v in row.tolist()]
    
    growths = []
    for i in range(len(values) - 1):
        new_val = values[i]
        old_val = values[i+1]
        if old_val != 0:
            growth = (new_val - old_val) / abs(old_val)
            growths.append(growth)
        else:
            growths.append(0.0)
            
    avg_growth = sum(growths) / len(growths) if growths else 0.0
    
    # Pad out to 5 years if history is shorter
    while len(growths) < 5:
        growths.append(0.0)

    return {
        "1y": round(growths[0], 4),
        "2y": round(growths[1], 4),
        "3y": round(growths[2], 4),
        "avg": round(avg_growth, 4),
    }


def _ratios(info: dict) -> dict:
    """Pull key valuation and profitability ratios straight from Yahoo Finance info"""
    keys = [
        "trailingPE", "forwardPE", "priceToBook", "debtToEquity",
        "returnOnEquity", "returnOnAssets", "grossMargins", "operatingMargins",
        "dividendYield", "payoutRatio", "trailingPegRatio", "marketCap"
    ]
    ratios = {}
    for k in keys:
        ratios[k] = round(_safe_float(info.get(k, 0.0)), 4)
    return ratios


def generate_data(ticker_symbol: str) -> dict:
    """Fetch massively expanded data for `ticker_symbol` from Yahoo Finance."""
    ticker = yf.Ticker(ticker_symbol)

    # Fetch History
    try:
        history = ticker.history(period="5y")
    except Exception as e:
        print(f"warning: could not fetch price history for {ticker_symbol}: {e}", file=sys.stderr)
        history = pd.DataFrame()

    # Fetch Info
    try:
        info = ticker.info
    except Exception:
        info = {}

    # Fetch Financial Statements
    try:
        q_income = _sort_desc(ticker.quarterly_income_stmt)
        a_income = _sort_desc(ticker.income_stmt)
        q_cashflow = _sort_desc(ticker.quarterly_cashflow)
        a_cashflow = _sort_desc(ticker.cashflow)
        q_balance = _sort_desc(ticker.quarterly_balance_sheet)
    except Exception as e:
        print(f"warning: could not fetch financials for {ticker_symbol}: {e}", file=sys.stderr)
        q_income = a_income = q_cashflow = a_cashflow = q_balance = pd.DataFrame()

    # 1. Pre-calculate base metrics BEFORE the return statement
    earnings_stats = _statement_stats(q_income, EARNINGS_ROWS)
    shares_stats = _fetch_shares(ticker, info)

    # 2. Derive EPS directly (Earnings / Shares) for every time period
    eps_stats = {}
    for period in SUBSETS_STD:
        earnings = earnings_stats.get(period, 0.0)
        shares = shares_stats.get(period, 0.0)
        
        # Avoid dividing by zero if shares data is missing
        if shares != 0.0:
            eps_stats[period] = round(earnings / shares, 4)
        else:
            eps_stats[period] = 0.0

    # 3. Return the final dictionary with the pre-calculated variables plugged in
    return {
        "ticker": ticker_symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
        
        # Price & Volume
        "price": _price_or_volume_stats(history, "Close"),
        "volume": _price_or_volume_stats(history, "Volume"),
        
        # Income Statement
        "revenue": _statement_stats(q_income, REVENUE_ROWS),
        "gross_profit": _statement_stats(q_income, GROSS_PROFIT_ROWS),
        "operating_income": _statement_stats(q_income, OPERATING_INC_ROWS),
        "ebitda": _statement_stats(q_income, EBITDA_ROWS),
        "earnings": earnings_stats,  # Plugs in the variable from step 1
        "eps": eps_stats,            # Plugs in the variable from step 2
        "shares": shares_stats,      # Plugs in the variable from step 1

        # Cash Flow Statement
        "operating_cash_flow": _statement_stats(q_cashflow, OCF_ROWS),
        "free_cash_flow": _statement_stats(q_cashflow, FCF_ROWS),
        "capex": _statement_stats(q_cashflow, CAPEX_ROWS),

        # Balance Sheet Snapshots
        "total_assets": _snapshot_stats(q_balance, ASSETS_ROWS),
        "total_liabilities": _snapshot_stats(q_balance, LIABILITIES_ROWS),
        "total_debt": _snapshot_stats(q_balance, DEBT_ROWS),
        "total_equity": _snapshot_stats(q_balance, EQUITY_ROWS),

        # Annual Historical Growth Rates
        "revenue_growth": _growth_stats(a_income, REVENUE_ROWS),
        "earnings_growth": _growth_stats(a_income, EARNINGS_ROWS),
        "eps_growth": _growth_stats(a_income, EPS_ROWS),
        "fcf_growth": _growth_stats(a_cashflow, FCF_ROWS),

        # Extracted Ratios
        "ratios": _ratios(info)
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: generate_data.py TICKER [OUTPUT_DIR]", file=sys.stderr)
        sys.exit(1)

    ticker_symbol = sys.argv[1].upper()
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_data(ticker_symbol)
    out_path = out_dir / f"{ticker_symbol}.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()