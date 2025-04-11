# Luna Backend

FastAPI backend service for Luna Brewery Analytics, designed to be deployed on Vercel.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Vercel CLI:
```bash
npm install -g vercel
```

3. Local development:
```bash
uvicorn api.index:app --reload
```

## Deployment

1. Initialize git repository (if not already done):
```bash
git init
git add .
git commit -m "Initial commit"
```

2. Deploy to Vercel:
```bash
vercel
```

## API Endpoints

### Generate SQL Query
`POST /api/generate`

Request body:
```json
{
    "prompt": "string",
    "system": "string (optional)",
    "schema": {},
    "temperature": 0.7,
    "maxTokens": null
}
```

Response:
```json
{
    "queries": [
        {
            "queryName": "string",
            "queryDescription": "string",
            "sql": "string"
        }
    ]
}
```

### Health Check
`GET /api/health`

Response:
```json
{
    "status": "healthy"
}
```

## Project Structure
```
luna-backend/
├── api/
│   └── index.py      # Main FastAPI application
├── requirements.txt  # Python dependencies
├── vercel.json      # Vercel deployment configuration
└── README.md        # This file
```

## Frontend Integration
The API is configured to work with the Luna frontend at https://luna-sable-six.vercel.app/. CORS is enabled for this domain site.