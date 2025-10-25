-- Triggers and helper functions for img-lib
-- This file creates lightweight trigger-based queues and helpers
-- that the Python service can use to keep prototypes and FAISS in sync.

-- Make sure pgcrypto is enabled (for hashing binary image data)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Add a unique image_hash column if missing
ALTER TABLE images
ADD COLUMN IF NOT EXISTS image_hash TEXT UNIQUE;

-- Automatically compute hash before insert/update if missing
CREATE OR REPLACE FUNCTION set_image_hash()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.image_hash IS NULL THEN
        -- Compute SHA256 hash of image binary data
        NEW.image_hash := encode(digest(NEW.image_data, 'sha256'), 'hex');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_image_hash ON images;
CREATE TRIGGER trg_set_image_hash
BEFORE INSERT OR UPDATE ON images
FOR EACH ROW
EXECUTE FUNCTION set_image_hash();

-- Prevent insertion of duplicate images
CREATE OR REPLACE FUNCTION prevent_duplicate_images()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM images WHERE image_hash = NEW.image_hash) THEN
        RAISE EXCEPTION 'Duplicate image detected (hash: %)', NEW.image_hash;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_prevent_duplicate_images ON images;
CREATE TRIGGER trg_prevent_duplicate_images
BEFORE INSERT ON images
FOR EACH ROW
EXECUTE FUNCTION prevent_duplicate_images();


-- Queue table to mark categories that need prototype recomputation
CREATE TABLE IF NOT EXISTS prototype_recompute_queue (
    category_id INT PRIMARY KEY,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Function to mark a category as needing prototype recompute
CREATE OR REPLACE FUNCTION mark_prototype_recompute() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    cat INT;
BEGIN
    -- Determine category_id from the affected vector's image
    IF (TG_OP = 'DELETE') THEN
        SELECT category_id INTO cat FROM images WHERE image_id = OLD.image_id;
    ELSE
        SELECT category_id INTO cat FROM images WHERE image_id = NEW.image_id;
    END IF;

    IF cat IS NOT NULL THEN
        INSERT INTO prototype_recompute_queue (category_id, last_updated)
        VALUES (cat, now())
        ON CONFLICT (category_id) DO UPDATE SET last_updated = EXCLUDED.last_updated;
    END IF;

    RETURN NULL;
END;
$$;

-- Trigger: mark prototypes stale when vectors change
DROP TRIGGER IF EXISTS trg_vectors_mark_prototype ON vectors;
CREATE TRIGGER trg_vectors_mark_prototype
AFTER INSERT OR UPDATE OR DELETE ON vectors
FOR EACH ROW EXECUTE FUNCTION mark_prototype_recompute();


-- When a new FAISS index row is inserted, populate vectors.index_id for vectors that are NULL.
CREATE OR REPLACE FUNCTION faiss_after_insert() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE vectors SET index_id = NEW.index_id WHERE index_id IS NULL;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_faiss_after_insert ON faiss;
CREATE TRIGGER trg_faiss_after_insert
AFTER INSERT ON faiss
FOR EACH ROW EXECUTE FUNCTION faiss_after_insert();


-- Queue for vectors that need to be removed from FAISS indexes (background worker will process)
CREATE TABLE IF NOT EXISTS vector_deletion_queue (
    image_id INT PRIMARY KEY,
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE OR REPLACE FUNCTION queue_vector_deletion() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    iid INT;
BEGIN
    IF (TG_OP = 'DELETE') THEN
        iid := OLD.image_id;
    ELSE
        RETURN NEW;
    END IF;

    IF iid IS NOT NULL THEN
        INSERT INTO vector_deletion_queue (image_id, queued_at)
        VALUES (iid, now())
        ON CONFLICT (image_id) DO NOTHING;
    END IF;

    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_vectors_queue_deletion ON vectors;
CREATE TRIGGER trg_vectors_queue_deletion
AFTER DELETE ON vectors
FOR EACH ROW EXECUTE FUNCTION queue_vector_deletion();


--maintain denormalized image count per category
ALTER TABLE categories ADD COLUMN IF NOT EXISTS image_count INT DEFAULT 0;

CREATE OR REPLACE FUNCTION update_category_image_count() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
     IF (TG_OP = 'INSERT') THEN
         UPDATE categories SET image_count = COALESCE(image_count,0) + 1 WHERE category_id = NEW.category_id;
     ELSIF (TG_OP = 'DELETE') THEN
         UPDATE categories SET image_count = GREATEST(COALESCE(image_count,0) - 1, 0) WHERE category_id = OLD.category_id;
     END IF;
     RETURN NULL;
 END;
 $$;

 DROP TRIGGER IF EXISTS trg_images_update_count ON images;
 CREATE TRIGGER trg_images_update_count AFTER INSERT OR DELETE ON images FOR EACH ROW EXECUTE FUNCTION update_category_image_count();

 -- Stored procedure to delete vectors by image ID
CREATE OR REPLACE PROCEDURE delete_image_vectors(p_image_id INT)
LANGUAGE plpgsql
AS $$
DECLARE
    v_vector_id INT;
BEGIN
    -- Loop through all vectors for the given image
    FOR v_vector_id IN 
        SELECT vector_id FROM vectors WHERE image_id = p_image_id
    LOOP
        -- Add each vector to the deletion queue
        INSERT INTO vector_deletion_queue (vector_id, queued_at)
        VALUES (v_vector_id, NOW());

        -- Optionally delete the vector immediately from vectors table
        DELETE FROM vectors WHERE vector_id = v_vector_id;
    END LOOP;

    -- If the image affects a category prototype, add it to the prototype recompute queue
    INSERT INTO prototype_recompute_queue (category_id, queued_at)
    SELECT category_id, NOW() 
    FROM images 
    WHERE image_id = p_image_id;

    -- Finally, delete the image itself
    DELETE FROM images WHERE image_id = p_image_id;

END;
$$;


-- End of triggers.sql
