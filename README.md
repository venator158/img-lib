# Image Similarity Search Application

A powerful image similarity search application built with FAISS vector search, PostgreSQL, and deep learning embeddings. The system uses prototype-based filtering to efficiently search through large image datasets.

## Features

- 🔍 **FAISS-powered similarity search** - Ultra-fast vector similarity search
- 🧠 **Deep learning embeddings** - ResNet-based image feature extraction
- 🎯 **Prototype filtering** - Efficient two-stage search using category prototypes
- 🗄️ **PostgreSQL storage** - Reliable storage for images, vectors, and metadata
- 🌐 **REST API** - FastAPI-based backend with comprehensive endpoints
- 🖥️ **Web interface** - Clean, responsive frontend for image upload and search
- 📊 **CIFAR-10 integration** - Pre-configured for CIFAR-10 dataset

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   PostgreSQL    │
│   (HTML/JS)     │◄──►│   Backend        │◄──►│   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   FAISS Index    │
                       │   (Vector Search)│
                       └──────────────────┘
```

### Two-Stage Search Process

1. **Prototype Stage**: Query image is compared against category prototypes (averaged embeddings)
2. **Detailed Stage**: Search within the most similar categories using FAISS index

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- pgvector extension (optional, for advanced vector operations)

### Installation

#### Windows
```bash
# Run the setup script
setup.bat

# Follow the interactive menu to:
# 1. Install dependencies
# 2. Setup database
# 3. Populate with CIFAR-10 data
# 4. Initialize vector search system
# 5. Start the server
```

#### Linux/macOS
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Follow the interactive menu
```

#### Manual Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Setup database**:
```bash
# Create PostgreSQL database named 'imsrc'
# Update connection details in .env file if needed

python db/init_system.py --setup-db
```

3. **Populate with CIFAR-10 data**:
```bash
python db/populate_cifar10.py --num-images 1000
```

4. **Initialize system**:
```bash
python db/init_system.py --all
```

5. **Start server**:
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

### Web Interface
1. Open `http://localhost:8000/app`
2. Upload an image using drag-and-drop or click to select
3. Configure search options:
   - Use prototype filtering (recommended)
   - Number of results (1-50)
   - Top categories to consider (1-10)
4. Click "Search Similar Images" or "Search Prototypes Only"
5. View results with similarity scores

### API Endpoints

#### Upload Image
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_image.jpg"
```

#### Search Similar Images
```bash
curl -X POST "http://localhost:8000/search?limit=10&use_prototypes=true" \
  -F "file=@query_image.jpg"
```

#### Get System Status
```bash
curl "http://localhost:8000/status"
```

#### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## Configuration

### Environment Variables (.env)
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=imsrc
DB_USER=postgres
DB_PASSWORD=your_password

# Model Configuration
MODEL_NAME=resnet50  # or resnet18
DEVICE=auto          # auto, cpu, cuda

# FAISS Configuration
FAISS_INDEX_PATH=data/faiss_index.bin
FAISS_INDEX_TYPE=flatl2  # flatl2, ivfflat, hnsw
USE_PROTOTYPE_FILTERING=true

# API Configuration
MAX_UPLOAD_SIZE=10485760  # 10MB
DEFAULT_SEARCH_LIMIT=10
API_HOST=0.0.0.0
API_PORT=8000
```

## Database Schema

### Tables

- **categories**: CIFAR-10 class labels
- **images**: Binary image data with metadata
- **vectors**: Feature embeddings for each image
- **_category_prototypes**: Averaged embeddings per category
- **faiss**: FAISS index metadata

### Example Queries

```sql
-- Get images by category
SELECT i.*, c.category_name 
FROM images i 
JOIN categories c ON i.category_id = c.category_id 
WHERE c.category_name = 'cat';

-- Find images without vectors
SELECT i.* FROM images i 
LEFT JOIN vectors v ON i.image_id = v.image_id 
WHERE v.image_id IS NULL;
```

## Performance

### Benchmarks (1000 images)
- **Image embedding generation**: ~50 images/second (ResNet50, CPU)
- **FAISS index search**: <10ms per query
- **Prototype filtering**: <1ms per query
- **End-to-end search**: <100ms per query

### Scaling
- **Database**: Supports millions of images with proper indexing
- **FAISS**: Handles millions of vectors efficiently
- **Memory usage**: ~2GB for 50K images with ResNet50 embeddings

## Development

### Project Structure
```
img-lib/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── vector_processor.py  # Image embeddings & FAISS
│   ├── database.py          # Database operations
│   └── config.py           # Configuration management
├── db/
│   ├── create_table.sql     # Database schema
│   ├── populate_cifar10.py  # Data population
│   └── init_system.py       # System initialization
├── frontend/
│   └── index.html          # Web interface
├── requirements.txt        # Python dependencies
└── .env                   # Configuration
```

### Adding New Models

1. **Extend ImageEmbedder class**:
```python
class ImageEmbedder:
    def _load_model(self):
        if self.model_name == 'your_model':
            model = load_your_model()
            # ... configure model
```

2. **Update configuration**:
```bash
MODEL_NAME=your_model
```

### Custom Datasets

1. **Create category mapping**:
```python
CUSTOM_CLASSES = {0: 'class1', 1: 'class2', ...}
```

2. **Modify populate script**:
```python
# Update populate_cifar10.py with your data loading logic
```

## Troubleshooting

### Common Issues

1. **Database connection fails**:
   - Check PostgreSQL is running
   - Verify connection details in `.env`
   - Ensure database 'imsrc' exists

2. **FAISS index not found**:
   - Run initialization: `python db/init_system.py --build-index`

3. **Out of memory during processing**:
   - Reduce batch size in scripts
   - Use smaller model (resnet18 vs resnet50)

4. **Slow search performance**:
   - Enable prototype filtering
   - Use IVF or HNSW index for large datasets
   - Check database indices

### Logs
- Application logs: Console output
- Database logs: PostgreSQL logs
- API logs: FastAPI automatic logging

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Submit pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- FAISS library by Facebook AI Research
- CIFAR-10 dataset by Alex Krizhevsky
- ResNet models from torchvision
- FastAPI framework by Sebastián Ramirez