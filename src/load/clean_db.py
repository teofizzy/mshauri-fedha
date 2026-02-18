# clean_database.py
import pandas as pd
from sqlalchemy import create_engine, text
import logging

# Set up a logger that can be configured by the importer
logger = logging.getLogger("DBCleaner")

def drop_blacklisted_tables(engine):
    """Drops tables matching the blacklist patterns."""
    drop_patterns = [
        "bop_annual",
        "commercial_banks_average_lending_rates",
        "depository_corporation", 
        "exchange_rates_end_period",
        "exchange_rates_period_average",
        "forex_bureau_rates_sheet", 
        "lr_return_template",
        "nsfr_return_template"
    ]
    
    with engine.connect() as conn:
        all_tables = [t[0] for t in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()]
        tables_to_drop = []
        
        for t in all_tables:
            if any(p in t for p in drop_patterns):
                tables_to_drop.append(t)
        
        if not tables_to_drop:
            logger.info("No tables found matching blacklist patterns.")
            return

        logger.info(f"Dropping {len(tables_to_drop)} tables...")
        for t in tables_to_drop:
            conn.execute(text(f'DROP TABLE "{t}"'))
            logger.info(f"   - Dropped: {t}")
        conn.commit()

def clean_table(engine, table_name, drop_top_rows=0, rename_map=None, rename_by_index=None, static_date=None):
    """
    Generic cleaner for specific table fixes.
    """
    try:
        # Check if table exists first
        with engine.connect() as conn:
            exists = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")).scalar()
        if not exists:
            logger.warning(f"   Table '{table_name}' not found. Skipping.")
            return

        df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
        if df.empty: return

        # Drop columns that are completely empty
        df = df.dropna(axis=1, how='all')
        
        # Drop top rows if requested
        if drop_top_rows > 0:
            df = df.iloc[drop_top_rows:].reset_index(drop=True)
            
        # Rename by Index (useful for 'col_1', 'col_2')
        if rename_by_index:
            curr_cols = list(df.columns)
            new_cols = curr_cols.copy()
            for idx, new_name in rename_by_index.items():
                if idx < len(curr_cols):
                    new_cols[idx] = new_name
            df.columns = new_cols
            
        # Rename by Map
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            
        # Inject Static Date if missing
        if static_date:
            if 'date' not in df.columns:
                df.insert(0, 'date', static_date)
            else:
                df['date'] = static_date

        # Save back to DB (Replace mode)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logger.info(f"   Fixed '{table_name}': {len(df)} rows")
        
    except Exception as e:
        logger.error(f"    Error cleaning '{table_name}': {e}")

def run_specific_fixes(engine):
    """Orchestrates the specific cleaning rules."""
    logger.info("Running specific table fixes...")
    
    # 1. Historical Rates
    clean_table(engine, "download_all_historical_rates", 
                rename_by_index={2: "mean_rate", 3: "buy_rate", 4: "sell_rate"})

    # 2. Foreign Trade Summary
    clean_table(engine, "foreign_trade_summary", drop_top_rows=1)

    # 3. Forex Bureau Rates
    clean_table(engine, "forex_bureau_rates", 
                rename_map={"bureau_name": "currency"})

    # 4. Indicative Rates (Indicative Sheet)
    clean_table(engine, "indicative_rates_sheet_indicative", 
                static_date="2017-11-16",
                rename_by_index={0: "currency", 1: "mean_rate", 2: "buy_rate", 3: "sell_rate"})

    # 5. Indicative Rates (Press Sheet)
    clean_table(engine, "indicative_rates_sheet_press", 
                static_date="2017-11-16",
                rename_by_index={
                    0: "bank_name",
                    1: "usd_buy", 2: "usd_sell", 3: "usd_margin",
                    4: "gbp_buy", 5: "gbp_sell", 6: "gbp_margin"
                })

    # 6. Selected Domestic Exports
    clean_table(engine, "value_of_selected_domestic_exports", drop_top_rows=2)

    # 7. Imports by Commodity
    clean_table(engine, "value_of_direct_imports_by_commodities", drop_top_rows=1)

def clean_database_pipeline(db_name):
    """Main entry point for external calls."""
    connection_str = f"sqlite:///{db_name}"
    engine = create_engine(connection_str)
    
    logger.info(f" Starting cleanup on {db_name}...")
    drop_blacklisted_tables(engine)
    run_specific_fixes(engine)
    logger.info(" Cleanup Complete.")

def drop_tables(engine):
    """Drops the specific list of tables requested."""
    tables_to_drop = [
        'forex_bureau_rates',
        'forex_bureaus_rates_sheet_chief_dealers',
        'forex_bureaus_rates_sheet_director',
        'forex_bureaus_rates_sheet_directors',
        'forex_bureaus_rates_sheet_fbx',
        'forex_bureaus_rates_sheet_fbx1',
        'forex_bureaus_rates_sheet_fbx2',
        'forex_bureaus_rates_sheet_fxb1',
        'forex_bureaus_rates_sheet_fxb2',
        'forex_bureaus_rates_sheet_fxb22',
        'forex_bureaus_rates_sheet_market_intelligence',
        'forex_bureaus_rates_sheet_sheet1',
        'forex_bureaus_rates_sheet_sheet2',
        'forex_bureaus_rates_sheet_sheet3',
        'forex_bureaus_rates_sheet_sheet4',
        'issues_of_treasury_bills',
        'issues_of_treasury_bonds'
    ]

    print("Dropping Tables...")
    with engine.connect() as conn:
        for t in tables_to_drop:
            try:
                conn.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
                print(f"   - Dropped: {t}")
            except Exception as e:
                print(f"   Could not drop {t}: {e}")
        conn.commit()

def fix_foreign_trade(engine):
    """Renames first column to 'year'."""
    table_name = "foreign_trade_summary"
    try:
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
        if 'kenyan_shillings_million_year' in df.columns:
            df.rename(columns={'kenyan_shillings_million_year': 'year'}, inplace=True)
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f" Fixed '{table_name}': Renamed 'year' column.")
        else:
            print(f"  '{table_name}': Target column not found.")
    except Exception as e:
        print(f" Error fixing {table_name}: {e}")

def fix_indicative_rates_shift(engine):
    """
    Applies the 'Shift Right + Fixed Date' logic.
    Inserts 2017-11-16 at position 0, shifting existing data to the right.
    """
    targets = [
        "indicative_rates_sheet_indicative",
        "indicative_rates_sheet_press"
    ]
    
    fixed_date = "2017-11-16"

    for table in targets:
        try:
            df = pd.read_sql(f'SELECT * FROM "{table}"', engine)
            if df.empty: continue

            # Logic: Insert new date column at index 0
            # This effectively "shifts" the old col 0 to col 1
            df.insert(0, 'fixed_date', fixed_date)
            
            # Rename columns to reflect the shift clearly
            # We assume the user wants standard names for the shifted data
            # Adjust names based on the table type
            new_columns = list(df.columns)
            new_columns[0] = "date" # The new fixed column
            
            # Assigning generic or specific headers for the shifted data
            if "press" in table:
                # Based on previous prompt instructions for Press sheet:
                # Bank, USD_Buy, USD_Sell, USD_Margin, GBP_Buy...
                expected_headers = ["date", "bank_name", "usd_buy", "usd_sell", "usd_margin", "gbp_buy", "gbp_sell", "gbp_margin", "euro_buy", "euro_sell", "euro_margin"]
            else:
                # Indicative sheet: Currency, Mean, Buy, Sell
                expected_headers = ["date", "currency", "mean_rate", "buy_rate", "sell_rate"]

            # Map headers safely (truncate if df has fewer cols, pad if more)
            final_cols = expected_headers + [f"col_{i}" for i in range(len(df.columns) - len(expected_headers))]
            df.columns = final_cols[:len(df.columns)]
            
            # Clean up: Drop any old 'date' column if it was pushed to the right and is duplicate/garbage
            # (Optional, but safer to keep strictly what we shifted)
            
            df.to_sql(table, engine, if_exists='replace', index=False)
            print(f" Fixed '{table}': Applied Date Shift & Header Rename.")
            
        except Exception as e:
            print(f" Error fixing {table}: {e}")

def fix_cbk_indicative_swap(engine):
    """Swaps 'date' and 'currency' column names."""
    table_name = "cbk_indicative_rates"
    try:
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
        
        rename_map = {}
        if 'date' in df.columns: rename_map['date'] = 'currency'
        if 'currency' in df.columns: rename_map['currency'] = 'date'
        
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f" Fixed '{table_name}': Swapped 'date' <-> 'currency'.")
    except Exception as e:
        print(f" Error fixing {table_name}: {e}")