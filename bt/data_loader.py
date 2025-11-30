
import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union

import pandas as pd
import yfinance as yf

def load_symbols_from_csv(csv_path: str, symbol_col: str = "Symbol") -> List[str]:
    """
    Read ticker symbols from a CSV and append '.NS' for NSE (Yahoo Finance style).
    """
    df = pd.read_csv(csv_path)

    if symbol_col not in df.columns:
        raise ValueError(f"Column '{symbol_col}' not found in {csv_path}")

    symbols = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    symbols_ns = [s + ".NS" if not s.endswith(".NS") else s for s in symbols]

    return symbols_ns


def fetch_price_data(
    tickers: List[str],
    start: datetime,
    end: datetime,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if not tickers:
        raise ValueError("Ticker list is empty.")

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=progress,
        threads=True
    )

    if raw.empty:
        raise ValueError("Downloaded data is empty. Check tickers or date range.")

    try:
        adj_close = raw["Adj Close"].copy()
        low = raw["Low"].copy()
    except KeyError:
        raise ValueError(
            f"'Adj Close' or 'Low' not found in downloaded data. Columns are: {raw.columns}"
        )

    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])
    if isinstance(low, pd.Series):
        low = low.to_frame(name=tickers[0])

    
    adj_close = adj_close.dropna(axis=1, how="all").dropna(axis=0, how="all")
    low = low.dropna(axis=1, how="all").dropna(axis=0, how="all")
    adj_close.index = pd.to_datetime(adj_close.index)
    low.index = pd.to_datetime(low.index)

    adj_close = adj_close.sort_index()
    low = low.sort_index()

    common_cols = adj_close.columns.intersection(low.columns)
    adj_close = adj_close[common_cols]
    low = low[common_cols]

    return adj_close, low

def get_nse500_data(
    symbols_csv_path: str,
    years: int = 5,
    end: Optional[datetime] = None,
    symbol_col: str = "Symbol",
    save_path: Optional[str] = None,
    return_low: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    
    if end is None:
        end = datetime.today()
    start = end - timedelta(days=years * 365)

    tickers = load_symbols_from_csv(symbols_csv_path, symbol_col=symbol_col)
    print(f"Loaded {len(tickers)} tickers from {symbols_csv_path}.")

    adj_close, low = fetch_price_data(tickers, start=start, end=end, progress=True)
    print(
        f"Fetched data for {adj_close.shape[1]} tickers "
        f"from {start.date()} to {end.date()}."
    )
    if save_path is not None:
        base_dir = os.path.dirname(save_path)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        if save_path.lower().endswith(".feather"):
            adj_close.reset_index().rename(columns={"index": "Date"}).to_feather(
                save_path
            )
            print(f"Saved Adj Close to feather file: {save_path}")
        elif save_path.lower().endswith(".csv"):
            adj_close.to_csv(save_path, index=True)
            print(f"Saved Adj Close to CSV file: {save_path}")
        else:
            print(
                f"save_path '{save_path}' does not end with .feather or .csv; "
                "Adj Close data NOT saved."
            )
        root, ext = os.path.splitext(save_path)
        low_path = root + "_low" + ext

        if ext.lower() == ".feather":
            low.reset_index().rename(columns={"index": "Date"}).to_feather(low_path)
            print(f"Saved Low prices to feather file: {low_path}")
        elif ext.lower() == ".csv":
            low.to_csv(low_path, index=True)
            print(f"Saved Low prices to CSV file: {low_path}")
        else:
            print(
                f"Low data not saved because extension '{ext}' is not .feather or .csv."
            )
    if return_low:
        return adj_close, low
    else:
        return adj_close


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for running this module as a script."""
    parser = argparse.ArgumentParser(description="Fetch NSE 500 price data for bt.")

    parser.add_argument(
        "--symbols_csv",
        type=str,
        required=True,
        help="Path to CSV file containing NSE 500 symbols (with 'Symbol' column).",
    )
    parser.add_argument(
        "--symbol_col",
        type=str,
        default="Symbol",
        help="Name of the column that contains the tickers (default: 'Symbol').",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of data to fetch (default: 5).",
    )
    parser.add_argument(
        "--out",
        type=str,
        choices=["feather", "csv", "none"],
        default="feather",
        help="Output format to save data (feather/csv/none). Default: feather.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/nse500_prices.feather",
        help="Path to save the Adj Close output file. "
             "Low file will get '_low' suffix automatically. "
             "Ignored if --out=none.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    save_path = None
    if args.out != "none":
        save_path = args.output_path
        if args.out == "feather" and not save_path.lower().endswith(".feather"):
            save_path += ".feather"
        elif args.out == "csv" and not save_path.lower().endswith(".csv"):
            save_path += ".csv"
    _ = get_nse500_data(
        symbols_csv_path=args.symbols_csv,
        years=args.years,
        symbol_col=args.symbol_col,
        save_path=save_path,
        return_low=True,
    )
