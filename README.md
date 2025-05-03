# Luna Backend

FastAPI backend service for Luna Brewery Analytics, designed to be deployed on Vercel.

## Setup

1. Install dependencies and the Luna package:
```bash
pip install -e .
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
`POST /api/chat/completions`

Request body:
```json
{
    "messages": [
        {"role": "user", "content": "Show me sales for the past month"}
    ],
    "schema": {
        "tables": [
            {
                "name": "sales",
                "columns": ["id", "product_id", "quantity", "price", "date"]
            },
            {
                "name": "products",
                "columns": ["id", "name", "category", "cost"]
            }
        ]
    },
    "temperature": 0.7,
    "maxTokens": null
}
```

Response:
```json
{
    "queries": [
        {
            "queryName": "Monthly Sales Report",
            "queryDescription": "This query retrieves all sales data for the past month",
            "sql": "SELECT * FROM sales WHERE date >= CURRENT_DATE - INTERVAL '1 month'"
        }
    ]
}
```

### Generate Chart
`POST /api/charts/generate`

Request body:
```json
{
    "data": [
        {"product": "Beer A", "sales": 100},
        {"product": "Beer B", "sales": 150},
        {"product": "Beer C", "sales": 80}
    ],
    "prompt": "Create a bar chart showing sales by product"
}
```

Response:
```json
{
    "chart": {
        "chartType": "bar",
        "title": "Sales by Product",
        "xAxis": {
            "type": "category",
            "data": ["Beer A", "Beer B", "Beer C"]
        },
        "yAxis": {
            "type": "value"
        },
        "seriesConfig": {
            "data": [100, 150, 80]
        }
    }
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
│   └── index.py          # Main FastAPI application
├── luna/                 # Luna package
│   ├── __init__.py       # Package initialization
│   ├── routes.py         # API routes
│   └── generator.py      # LunaGenerator for SQL and chart generation
├── setup.py              # Package installation configuration
├── requirements.txt      # Python dependencies
├── vercel.json           # Vercel deployment configuration
└── README.md             # This file
```

## Luna Generator

The Luna Generator is the core class that provides the main functionality of the system:

1. **SQL Query Generation**: Transforms natural language questions into SQL queries based on database schema
2. **Chart Generation**: Creates chart configurations based on data and natural language visualization requests
3. **NewsFeed**: Provide feed items for alerts

The generator uses LangChain with OpenAI models to process natural language prompts and generate appropriate SQL queries or chart configurations. It includes:

- Structured output parsing for consistent response formats
- Comprehensive error handling with fallback responses
- Singleton pattern implementation for efficient resource usage
- Asynchronous methods for improved performance

Example usage:
```python
# Get the generator instance
generator = get_luna_generator()

# Generate SQL query
query_result = await generator.generate_sql_query(
    "Show me total sales by product",
    schema={"tables": [...]}
)

# Generate chart configuration
chart_result = await generator.generate_chart(
    data=[{"product": "Beer A", "sales": 100}, ...],
    prompt="Create a bar chart showing sales by product"
)
```

## Frontend Integration
The API is configured to work with the Luna frontend at https://luna-sable-six.vercel.app/. CORS is enabled for this domain.