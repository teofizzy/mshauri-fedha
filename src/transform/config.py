import os
import shutil
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """Central configuration for Data Sources."""
    root_dir: str
    source_name: str  # e.g., 'knbs' or 'cbk'

    # Settings
    batch_size: int = 20
    max_workers: int = 4
    min_image_bytes: int = 3000
    min_image_dim: int = 100
    max_page_objects: int = 500

    def __post_init__(self):
        # Paths setup
        self.base_processed_dir = os.path.join(self.root_dir, 'processed')
        self.source_dir = os.path.join(self.base_processed_dir, self.source_name)
        self.drive_zip_dir = os.path.join(self.source_dir, "zipped_batches")
        self.meta_dir = os.path.join(self.source_dir, f"{self.source_name}_index_metadata")
        
        # Log Files
        self.logs = {
            'docs': os.path.join(self.meta_dir, f'{self.source_name}_docs_metadata.jsonl'),
            'images': os.path.join(self.meta_dir, f'{self.source_name}_images_index.jsonl'),
            'tables': os.path.join(self.meta_dir, f'{self.source_name}_tables_index.jsonl')
        }
        
        # Local Temp Paths
        self.local_work_dir = Path(f"/tmp/temp_work_{self.source_name}")
        self.local_dirs = {
            'texts': self.local_work_dir / "texts",
            'images': self.local_work_dir / "images",
            'tables': self.local_work_dir / "tables",
            'pdfs': self.local_work_dir / "pdfs"
        }

    def setup(self):
        os.makedirs(self.drive_zip_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        if self.local_work_dir.exists():
            shutil.rmtree(self.local_work_dir)
        for d in self.local_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.create_canary()

    def create_canary(self):
        script_content = """
import sys, pymupdf, pdfplumber
if len(sys.argv) < 2: sys.exit(1)
try:
    doc = pymupdf.open(sys.argv[1])
    for p in doc: _, _ = p.get_text(), [doc.extract_image(i[0]) for i in p.get_images(full=True)]
    with pdfplumber.open(sys.argv[1]) as p: _ = [page.objects for page in p.pages]
    print("SAFE")
    sys.exit(0)
except: sys.exit(1)
"""
        with open("pdf_canary.py", "w") as f:
            f.write(script_content.strip())