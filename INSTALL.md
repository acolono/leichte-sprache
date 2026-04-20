# Leichte Sprache API - Installation Guide

**Version:** 0.1.0
**Package:** leichte_sprache_rulez-0.1.0-py3-none-any.whl

## 📋 System Requirements

- **Python:** 3.11 or higher
- **RAM:** Minimum 2GB (for spaCy German model)
- **Disk Space:** ~1GB free (for dependencies and models)
- **Operating System:** Linux, macOS, or Windows

## 🚀 Quick Installation

### Step 1: Verify Python Version

```bash
python --version
# Should show Python 3.11.x or higher
```

If you don't have Python 3.11+, download it from [python.org](https://www.python.org/downloads/)

### Step 2: Install uv (Recommended)

```bash
# Install uv
pip install uv
```

### Step 3: Create Virtual Environment and Install

```bash
# Create virtual environment
uv venv

# Activate it
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate

# Install from wheel file
uv pip install leichte_sprache_rulez-0.1.0-py3-none-any.whl

# Verify installation
uv pip show leichte-sprache-rulez
```

### Step 4: Install German Language Model

```bash
# Download spaCy German model (~500MB)
python -m spacy download de_core_news_lg

# This may take 2-5 minutes depending on your internet connection
```

### Step 5: Start the API Server

```bash
# Start the server
leichte-sprache-api

# You should see:
# 🚀 Leichte Sprache API gestartet
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

The API is now running at **http://localhost:8000**

## ✅ Verify Installation

### Check API Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Leichte Sprache API",
  "version": "1.0.0"
}
```

### Test Analysis Endpoint

```bash
curl -X POST "http://localhost:8000/analyse" \
  -H "Content-Type: application/json" \
  -d '{"text": "Die komplexe Administration evaluiert die Implementation."}'
```

### Open Interactive Documentation

Visit in your browser:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 📖 Basic Usage

### Analyze Text via API

**Request:**
```bash
curl -X POST "http://localhost:8000/analyse" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Der Text soll geprüft werden."
  }'
```

**Response:**
```json
{
  "annotated_text": "Der Text soll geprüft werden.",
  "statistics": {
    "total_violations": 0,
    "unique_violations": 0,
    "violations_by_rule": {}
  },
  "issues": []
}
```

### Use as Python Module

```python
from leichte_sprache import analyse_text

# Analyze text
result = analyse_text("Die komplexe Administration evaluiert die Implementation.")

# Access results
print(result['annotated_text'])
print(f"Found {result['statistics']['total_violations']} violations")

for issue in result['issues']:
    print(f"- {issue['text']}: {issue['message']}")
```

## 🔧 Configuration

### Change Server Port

```bash
# Use uvicorn directly
uvicorn leichte_sprache.api_main:app --host 0.0.0.0 --port 9000
```

### Run in Production Mode

```bash
# With multiple workers
uvicorn leichte_sprache.api_main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level warning
```

### Enable HTTPS (with certificate)

```bash
uvicorn leichte_sprache.api_main:app \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem
```

## 🐳 Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy wheel file
COPY leichte_sprache_rulez-0.1.0-py3-none-any.whl .

# Install package and spaCy model
RUN pip install --no-cache-dir leichte_sprache_rulez-0.1.0-py3-none-any.whl && \
    python -m spacy download de_core_news_lg

# Expose port
EXPOSE 8000

# Run API
CMD ["leichte-sprache-api"]
```

Build and run:
```bash
docker build -t leichte-sprache-api .
docker run -p 8000:8000 leichte-sprache-api
```

## 🔍 Troubleshooting

### Problem: "No module named 'leichte_sprache'"

**Solution:** Make sure the package is installed:
```bash
pip install leichte_sprache_rulez-0.1.0-py3-none-any.whl
```

### Problem: "Can't find model 'de_core_news_lg'"

**Solution:** Download the German spaCy model:
```bash
python -m spacy download de_core_news_lg
```

### Problem: "Port 8000 already in use"

**Solution:** Use a different port or kill the process:
```bash
# Use different port
uvicorn leichte_sprache.api_main:app --port 8001

# Or find and kill the process (Linux/macOS)
lsof -ti:8000 | xargs kill
```

### Problem: API is slow on first request

**Cause:** The German spaCy model loads on first request (~2-3 seconds)
**Solution:** This is normal. Subsequent requests will be fast.

### Problem: High memory usage

**Cause:** spaCy German model requires ~1.5-2GB RAM
**Solution:**
- Ensure your system has sufficient RAM
- Use a single worker in production
- Consider using a smaller spaCy model (not recommended for accuracy)

### Problem: "UnicodeDecodeError"

**Solution:** Ensure your text is UTF-8 encoded:
```python
text = "Dein Text hier".encode('utf-8').decode('utf-8')
```

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyse` | POST | Analyze German text |
| `/health` | GET | Health check |
| `/info` | GET | API information and rules |
| `/docs` | GET | Interactive Swagger UI |
| `/redoc` | GET | Alternative documentation |

## 🔐 Security Notes

- The API currently has **no authentication** - add authentication for production
- Consider using **HTTPS** in production
- Set up **rate limiting** to prevent abuse
- Review **CORS settings** in `api_main.py` for production

## 📊 Performance Tips

1. **Use multiple workers** for high traffic:
   ```bash
   uvicorn leichte_sprache.api_main:app --workers 4
   ```

2. **Add caching** for repeated requests (not included)

3. **Monitor with Prometheus/Grafana** (add metrics endpoint)

4. **Use a reverse proxy** (nginx) for better performance

## 🆘 Getting Help

- **Documentation:** See README.md in the repository
- **API Docs:** http://localhost:8000/docs (when running)
- **Issues:** Report bugs via project issue tracker
- **Support:** Contact your system administrator

## 📝 Next Steps

After successful installation:

1. ✅ Read the full API documentation at `/docs`
2. ✅ Test with your German text samples
3. ✅ Integrate into your application
4. ✅ Set up monitoring and logging
5. ✅ Configure for production (authentication, HTTPS, etc.)

## 🔄 Updating

To update to a newer version:

```bash
# Uninstall old version
pip uninstall leichte-sprache-rulez

# Install new version
pip install leichte_sprache_rulez-X.Y.Z-py3-none-any.whl
```

## 📄 License

See LICENSE file in the project repository.

---

**Package Information:**
- Version: 0.1.0
- Python: 3.11+
- Built with: setuptools + uv
- Date: 2025-09-30

**For more information, visit the project documentation.**
