import pandas as pd
import glob
import os
import logging
import sys

# Configure logging to show up in the notebook
logging.basicConfig(
    level=logging.INFO, 
    format='%(message)s', 
    stream=sys.stdout, 
    force=True
)
logger = logging.getLogger("SchemaExplorer")

def analyze_schemas(news_dir: str):
    """
    Scans all CSV files in the given directory and groups them by their column structure.
    """
    if not os.path.exists(news_dir):
        logger.error(f" Directory not found: {news_dir}")
        return

    csv_files = glob.glob(os.path.join(news_dir, "*.csv"))
    logger.info(f"Scanning {len(csv_files)} files in '{news_dir}'...\n")
    
    if not csv_files:
        logger.warning(" No CSV files found.")
        return

    # Dictionary to store unique schemas: { (col1, col2): [file1, file2] }
    schemas = {}
    
    for f in csv_files:
        try:
            # Read only the header (fast)
            df = pd.read_csv(f, nrows=0)
            
            # Sort columns to ensure order doesn't matter for grouping
            cols = tuple(sorted(df.columns.tolist()))
            
            if cols not in schemas:
                schemas[cols] = []
            schemas[cols].append(os.path.basename(f))
            
        except Exception as e:
            logger.error(f" Error reading {os.path.basename(f)}: {e}")

    # Report Findings
    logger.info("--- Schema Report ---")
    for i, (cols, files) in enumerate(schemas.items()):
        logger.info(f"\nTYPE {i+1}: Found in {len(files)} files")
        logger.info(f"Columns: {list(cols)}")
        if len(files) < 5:
            logger.info(f"Examples: {files}")
        else:
            logger.info(f"Examples: {files[:3]} ... (+{len(files)-3} others)")

    # Date Format Check (Random Sample from the first valid file)
    logger.info("\n--- Date Format Sample ---")
    try:
        sample_file = csv_files[0]
        sample = pd.read_csv(sample_file, nrows=5)
        
        # Look for a column containing 'date' or 'time'
        date_col = next((c for c in sample.columns if 'date' in c.lower() or 'time' in c.lower() or 'published' in c.lower()), None)
        
        if date_col:
            logger.info(f"Sample from column '{date_col}' in {os.path.basename(sample_file)}:")
            logger.info(sample[date_col].head().tolist())
        else:
            logger.warning("No obvious 'date' column found in sample.")
    except Exception as e:
        logger.error(f"Could not read sample for date check: {e}")