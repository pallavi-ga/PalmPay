from app import app, db
from models import PalmPrint

with app.app_context():
    for palm in PalmPrint.query.all():
        print(f"PalmPrint ID: {palm.id}")
        print(f"  Original:  {palm.original_image_path}")
        print(f"  Processed: {palm.processed_image_path}")
        print('-' * 40)

    for palm in PalmPrint.query.all():
        if palm.original_image_path and palm.original_image_path.startswith('static/'):
            while palm.original_image_path.startswith('static/'):
                palm.original_image_path = palm.original_image_path[len('static/'):]
        if palm.processed_image_path and palm.processed_image_path.startswith('static/'):
            while palm.processed_image_path.startswith('static/'):
                palm.processed_image_path = palm.processed_image_path[len('static/'):]
    db.session.commit()
    print("All palm print image paths updated.") 