from app import db
from models import PalmPrint

for palm in PalmPrint.query.all():
    print(f"PalmPrint ID: {palm.id}")
    print(f"  Original:  {palm.original_image_path}")
    print(f"  Processed: {palm.processed_image_path}")
    print('-' * 40) 