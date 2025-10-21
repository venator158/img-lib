CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE faiss (
    index_id SERIAL PRIMARY KEY,
    index_type VARCHAR(255) NOT NULL,
    index_filepath TEXT NOT NULL
);

CREATE TABLE _category_prototypes (
    prototype_id SERIAL PRIMARY KEY,
    category_id INT UNIQUE NOT NULL,
    prototype_vector vector NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE images (
    image_id SERIAL PRIMARY KEY,
    image_data BYTEA NOT NULL,
    metadata JSONB DEFAULT '{}'::JSONB,
    category_id INT NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE vectors (
    vector_id SERIAL PRIMARY KEY,
    embedding vector NOT NULL,
    index_id INT,
    image_id INT UNIQUE NOT NULL,
    FOREIGN KEY (index_id) REFERENCES faiss(index_id) ON DELETE SET NULL ON UPDATE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE ON UPDATE CASCADE
);