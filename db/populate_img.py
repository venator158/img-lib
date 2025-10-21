import psycopg2
from tensorflow.keras.datasets import cifar10
from PIL import Image
import io
import json

(x_train, y_train), (_, _) = cifar10.load_data()

def generate_metadata(idx):
    return {"source": "cifar10", "image_index": idx, "description": "32x32 color image"}

conn = psycopg2.connect("dbname=imsrc user=postgres password=14789")
cur = conn.cursor()

for i in range(100):
    image_array = x_train[i]
    category_id = int(y_train[i][0]) + 1

    img = Image.fromarray(image_array)
    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')
    img_bytes = byte_io.getvalue()

    metadata = json.dumps(generate_metadata(i))

    cur.execute("""
        INSERT INTO images (image_data, metadata, category_id)
        VALUES (%s, %s::jsonb, %s)
        RETURNING image_id;
    """, (psycopg2.Binary(img_bytes), metadata, category_id))

    image_id = cur.fetchone()[0]
    print(f"Inserted image {i} with image_id: {image_id}")

conn.commit()
cur.close()
conn.close()
