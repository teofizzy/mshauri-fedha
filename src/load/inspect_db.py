import os
import pandas as pd
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
DB_NAME = "mshauri_fedha.db"
DB_CONNECTION = f"sqlite:///{DB_NAME}"

def list_all_tables(engine):
    print(f"\n --- DATABASE SUMMARY: {DB_NAME} ---")
    try:
        with engine.connect() as conn:
            # Query the master table for all table names
            query = text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = conn.execute(query).fetchall()
            
            if not tables:
                print(" Database is empty.")
                return []

            table_list = [t[0] for t in tables]
            
            print(f"{'ID':<4} | {'Rows':<8} | {'Table Name'}")
            print("-" * 60)
            
            for i, t_name in enumerate(table_list):
                # Count rows for verification
                try:
                    count = conn.execute(text(f'SELECT COUNT(*) FROM "{t_name}"')).scalar()
                    print(f"{i:<4} | {count:<8} | {t_name}")
                except:
                    print(f"{i:<4} | {'ERROR':<8} | {t_name}")
            
            return table_list
    except Exception as e:
        print(f" Connection failed: {e}")
        return []

def inspect_table(engine, table_name):
    print(f"\n🔎 Inspecting Table: '{table_name}'")
    try:
        # Read schema/columns
        query = f'SELECT * FROM "{table_name}" LIMIT 5'
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print(" Table is empty.")
        else:
            print(f"Columns: {list(df.columns)}")
            print("\n--- First 5 Rows ---")
            # to_string() makes it readable in terminal without truncation
            print(df.to_string(index=False)) 
            print("-" * 50)
    except Exception as e:
        print(f" Could not read table: {e}")

def main():
    if not os.path.exists(DB_NAME):
        print(f" Error: Database file '{DB_NAME}' not found in current directory.")
        print(f"Current Directory: {os.getcwd()}")
        return

    engine = create_engine(DB_CONNECTION)
    tables = list_all_tables(engine)
    
    if not tables: return

    while True:
        try:
            user_input = input("\nEnter Table ID (or Name) to inspect, or 'q' to quit: ").strip()
            if user_input.lower() == 'q': break
            
            target_table = None
            
            # Handle numeric ID input
            if user_input.isdigit():
                idx = int(user_input)
                if 0 <= idx < len(tables):
                    target_table = tables[idx]
            # Handle name input
            elif user_input in tables:
                target_table = user_input
            
            if target_table:
                inspect_table(engine, target_table)
            else:
                print(" Invalid selection.")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()