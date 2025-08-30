# Review Quality Detection System - Deployment Guide

## Quick Start

### Prerequisites

- Python 3.13+
- 8GB RAM minimum
- PostgreSQL (optional for production)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd review-quality-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Basic Deployment

```bash
# Run the system
python src/main.py

# Run demo
python scripts/demo/demo_review_quality_detection.py

# Train models
python scripts/training/train_review_quality_model.py
```

### Production Deployment

#### Docker Deployment

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "src/main.py"]
```

```bash
# Build and run
docker build -t review-quality-system .
docker run -p 8000:8000 review-quality-system
```

#### Configuration

Edit `config/config.yaml` for your environment:

```yaml
# Production settings
system:
  debug: false
  log_level: 'INFO'

performance:
  processing:
    batch_size: 1000
    max_workers: 4

security:
  data_security:
    encryption_enabled: true
```

### Monitoring

```bash
# Check logs
tail -f logs/review_quality_pipeline_*.log

# Monitor performance
python -c "
from src.main import ReviewQualityDetectionPipeline
pipeline = ReviewQualityDetectionPipeline()
results = pipeline.run_complete_pipeline()
print(f'Pipeline status: {results[\"success\"]}')
"
```

### Backup

```bash
# Backup models and data
tar -czf backup_$(date +%Y%m%d).tar.gz models/ data/ config/
```

---

**Version**: 2.0.0  
**Status**: Production Ready
