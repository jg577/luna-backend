from typing import Dict, Any, List, Optional
import json
import logging
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain import hub
from langchain.chains.sql_database.query import create_sql_query_chain
import ast  # For safely evaluating the string representation of Python literals
import re   # For regex patterns


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LunaGenerator:
    """
    Main class for Luna functionality including SQL query generation and chart visualization.
    """

    def __init__(self, model_name: str = "gpt-4o", model_provider: str = "openai"):
        """
        Initialize the LunaGenerator with a language model.

        Args:
            model_name: The name of the LLM model to use
            model_provider: The provider of the model (e.g., "openai")
        """
        self.llm = ChatOpenAI(model_name=model_name)
        logger.info(
            f"Initialized LunaGenerator with model {model_name} from {model_provider}"
        )
        self.db = SQLDatabase.from_uri(
            database_uri="postgresql://neondb_owner:npg_wphuiy9KT8We@ep-calm-poetry-a8atq3ap-pooler.eastus2.azure.neon.tech/neondb?sslmode=require",
        )
        logger.info(f"Initialized the db {self.db} from {model_provider}")

        # Load prompt templates from JSON file
        self.prompt_options = self._load_prompt_options()
        logger.info(f"Loaded {len(self.prompt_options)} prompt templates")

    def _load_prompt_options(self) -> Dict[str, Dict[str, str]]:
        """
        Load prompt templates from the JSON configuration file.

        Returns:
            Dictionary of prompt templates indexed by name
        """
        # Get the path to the prompts.json file (in the same directory as this file)
        prompts_path = Path(__file__).parent / "prompts.json"

        try:
            with open(prompts_path, "r") as f:
                prompts = json.load(f)
                logger.info(f"Successfully loaded prompts from {prompts_path}")

                # Process the loaded prompts to interpret escape sequences
                processed_prompts = {}
                for prompt_name, prompt_data in prompts.items():
                    processed_prompt = {}
                    for key, value in prompt_data.items():
                        if isinstance(value, str):
                            # Process escape sequences by encoding and decoding
                            processed_value = value.encode("utf-8").decode(
                                "unicode_escape"
                            )
                            processed_prompt[key] = processed_value
                        else:
                            processed_prompt[key] = value
                    processed_prompts[prompt_name] = processed_prompt

                return processed_prompts
        except Exception as e:
            logger.error(f"Error loading prompts from {prompts_path}: {str(e)}")
            # Return empty dict if file can't be loaded
            return {}

    async def generate_sql_query(
        self, user_input: str, schema: Dict[str, Any], prompt_template: str = "prompt1"
    ) -> Dict[str, Any]:
        """
        Generate SQL queries based on a natural language prompt.

        Args:
            user_input: The natural language question from the user
            schema: The database schema structure
            prompt_template: The name of the prompt template to use (defaults to "prompt1")

        Returns:
            A dictionary containing the generated SQL queries with metadata
        """
        logger.info(
            f"Generating SQL query for prompt: {user_input} using template: {prompt_template}"
        )

        # Define output schema for structured response
        response_schemas = [
            ResponseSchema(
                name="queryName",
                description="A short name describing what this query calculates",
            ),
            ResponseSchema(
                name="queryDescription",
                description="A brief description of what this query does and what insights it provides",
            ),
            ResponseSchema(name="sql", description="The SQL query to execute"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions(True)

        # Get the selected prompt template or fallback to default
        template_config = self.prompt_options.get(
            prompt_template, self.prompt_options.get("prompt1", {})
        )
        logger.info(f"template_config: {template_config}")

        # Create a custom SQL generation prompt
        sql_system_template = """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Only use the following tables:
{table_info}

Additional information about the tables:

    Table: item_selection_details: The item_selection_details table is also a daily table and is probably the most important table because it contains the entire sales data. Every row is a sales from POS that is logged into this table. So this table contains information about food/beverage orders, their prices, and details about the dining experience. The datewise join keys are sent_date, and we can have other join key on menu_item for menu_mappings.

    Table: time_entries: The time_entries table is a daily table and primarily a cost table that contains information about employee work shifts, including the employee details, hours worked, wages, and tips. Important columns are out_date that we use for daily hours/shifts etc and also a join key in case we need a datewise join with item_selection_details, or costs tables
        
        
    Table: menu_mappings - This is a supporting mapping table for item_selection_details to map the menu_item from that table, which could have a bunch of permutations on the menu_item name, so we have this to standardize the menu_item to product_name and type.It provides standardized mappings between the adhoc menu item names in item_selection_details.menu_item and standardized product names, essential for accurate analytics.
    
    Table: costs: This is the costs table that is directly imported from a food vendor. This is updated monthly. This mostly contains the ingredients that the brewery/restaurant makes to prepare the food/beverages. The important columns in this table are item_name that give the name of the item ordered, date for when the order was made and sales, weight/quantity for pricess.
    
    Table: costs_groups: The costs_groups table is a supporting table for costs and contains information on the item_name from costs, this item_name could actually have a lot of variations on the name, so we have this table to standardize the names to items, and item_group/item_type that could be used for categorization for anything that requires a groupby.


    The tables can be joined on relevant fields for cross-table analysis:
      - time_entries and item_selection_details can be joined on date fields for date-based analysis.
      - for now costs, time_entries and item_selection_details can only be joined on dates because we don't quite have a mapping from menu_item in item_selection_details table to item_name in costs table. 
      - item_selection_details and menu_mappings should be joined (item_selection_details.menu_item = menu_mappings.item_name) AND coalesce(isd.menu_group, 'Null') = coalesce(mm.menu_group, 'Null') to standardize menu items for accurate analytics.
      - costs and costs_groups should be joined on (costs.item_name = costs_groups.item_name). When there are any queries on costs, do a join with costs_groups and do groupbys after that so that the nomenclature is standard.




Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Division by 0s

Here are some sample queries: 
{few_shot_examples}
Use format:

First draft: <<FIRST_DRAFT_QUERY>>
Final answer: <<FINAL_ANSWER_QUERY>>
"""

        # Create prompt template with all the required variables
        sql_prompt = ChatPromptTemplate.from_messages(
            [("system", sql_system_template), ("human", "Question: {input}")]
        ).partial(
            dialect=template_config.get("dialect", "PostgreSQL"),
            table_info=template_config.get("table_info", ""),
            few_shot_examples=template_config.get("few_shot_examples", ""),
            input=user_input,
            format_instructions=format_instructions,
            top_k=100,
            question=user_input,
        )

        # Create the SQL query chain
        sql_chain = create_sql_query_chain(self.llm, self.db, prompt=sql_prompt)

        # Execute the chain and return the result
        try:
            sql_response = await sql_chain.ainvoke({"question": user_input})

            logger.info(f"Generated SQL response: {sql_response[:100]}...")

            # Log the full response for debugging
            logger.info(f"Full SQL response: {sql_response}")

            # Try to extract just the SQL query if possible
            if "```sql" in sql_response:
                sql_parts = sql_response.split("```sql")
                if len(sql_parts) > 1:
                    sql_code_parts = sql_parts[1].split("```")
                    if sql_code_parts:
                        sql_response = sql_code_parts[0].strip()

            # Try to parse the response with the structured output parser
            try:
                parsed_response = output_parser.parse(sql_response)
                result = {
                    "queries": [
                        {
                            "queryName": parsed_response["queryName"],
                            "queryDescription": parsed_response["queryDescription"],
                            "sql": parsed_response["sql"],
                        }
                    ]
                }
            except Exception as parse_error:
                logger.warning(
                    f"Could not parse structured response: {str(parse_error)}"
                )
                # Fallback if structured parsing fails - use the raw SQL
                result = {
                    "queries": [
                        {
                            "queryName": f"Query for: {user_input[:30]}...",
                            "queryDescription": f"SQL query generated for: {user_input}",
                            "sql": sql_response.strip(),
                        }
                    ]
                }

            return result
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            # Fallback response if generation fails
            return {
                "queries": [
                    {
                        "queryName": "Error Query",
                        "queryDescription": "An error occurred during query generation",
                        "sql": "SELECT 'Error generating query' as error",
                    }
                ]
            }

    async def generate_chart(
        self,
        data: List[Dict[str, Any]],
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Generate chart configuration based on data and a prompt.

        Args:
            data: The data to visualize
            prompt: The natural language description of the desired chart

        Returns:
            A dictionary containing chart configuration
        """
        logger.info(f"Generating chart for prompt: {prompt}")

        # Define output schema for chart configurations
        response_schemas = [
            ResponseSchema(
                name="chartType",
                description="The type of chart to create (bar, line, pie, etc.)",
            ),
            ResponseSchema(name="title", description="The title for the chart"),
            ResponseSchema(name="xAxis", description="Configuration for the X axis"),
            ResponseSchema(name="yAxis", description="Configuration for the Y axis"),
            ResponseSchema(
                name="seriesConfig", description="Configuration for the data series"
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        # Create system message
        system_message = f"""You are a data visualization expert. 
Given a dataset and a request, create a chart configuration that best represents the data.
{format_instructions}
"""

        # Create prompt template
        template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", "Dataset sample: {data}\n\nVisualization request: {prompt}"),
            ]
        )

        # Prepare and send the prompt
        prompt_with_values = template.format(
            data=json.dumps(
                data[:5] if len(data) > 5 else data, indent=2
            ),  # Send sample of data
            prompt=prompt,
        )

        response = await self.llm.apredict(prompt_with_values)

        try:
            # Parse structured output
            parsed_response = output_parser.parse(response)

            # Return the chart configuration
            return {"chart": parsed_response}
        except Exception as e:
            logger.error(f"Error parsing chart generation response: {str(e)}")
            # Fallback response if parsing fails
            return {
                "chart": {
                    "chartType": "bar",
                    "title": "Error generating chart",
                    "xAxis": {"type": "category"},
                    "yAxis": {"type": "value"},
                    "seriesConfig": {},
                }
            }

    async def fetch_news_feed(self) -> List[Dict[str, Any]]:
        """
        Fetch news items from the feed_items table and transform them to the expected frontend format.

        Returns:
            A list of news items with id, title, description, severity, timestamp, and imageUrl
        """
        logger.info("Fetching news items from feed_items table")

        try:
            # SQL query with correct column names for feed_items table
            query = """
            SELECT 
                id::text, 
                name,
                date,
                result,
                TO_CHAR(created_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as timestamp
            FROM 
                feed_items
            ORDER BY 
                created_at DESC
            """
            
            # Execute the query
            result = self.db.run(query)
            logger.info(f"Fetched news items from feed_items table")
            
            # Parse the result into a list of dictionaries
            parsed_items = []
            
            if isinstance(result, str):
                # Replace datetime.date objects with string representations
                # Format: datetime.date(2025, 4, 30) -> "2025-04-30"
                pattern = r'datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)'
                
                def date_replacer(match):
                    year, month, day = match.groups()
                    return f'"{year}-{month.zfill(2)}-{day.zfill(2)}"'
                
                result_fixed = re.sub(pattern, date_replacer, result)
                
                try:
                    # Now we can safely parse the modified string
                    parsed_result = ast.literal_eval(result_fixed)
                    
                    # Process each tuple in the result
                    for row in parsed_result:
                        if len(row) >= 4:  # Make sure we have enough columns
                            # Extract fields
                            id_val = row[0]
                            title = row[1]
                            date_val = row[2]  # Now a string in format "YYYY-MM-DD"
                            result_json = row[3]  # This is a dict
                            timestamp = row[4] if len(row) > 4 else None
                            
                            # Extract message or description from the result
                            description = result_json.get('message', f"Analysis from {date_val}")
                            
                            # Determine severity based on notification type or content
                            severity = "neutral"
                            notification_type = result_json.get('notification_type', '')
                            if "Alert" in notification_type or "Warning" in notification_type:
                                if "suspicious" in description.lower() or "suss" in description.lower():
                                    severity = "bad"
                            elif "Success" in notification_type or "Info" in notification_type:
                                severity = "good"
                            
                            # Create the item in the expected frontend format
                            item = {
                                "id": str(id_val),
                                "title": title,
                                "description": description,
                                "severity": severity,
                                "timestamp": timestamp,
                                "imageUrl": None  # No image URLs in this data
                            }
                            
                            parsed_items.append(item)
                except Exception as e:
                    logger.error(f"Error parsing result: {str(e)}")
            
            return parsed_items
        except Exception as e:
            logger.error(f"Error fetching news feed: {str(e)}")
            # Return empty list if query fails
            return []
    
    def _extract_description(self, item: Dict[str, Any]) -> str:
        """Extract a meaningful description from the item data"""
        # Try to get description from result JSONB if available
        if isinstance(item.get("result"), dict):
            result = item["result"]
            # Look for common fields that might contain useful text
            for field in ["description", "summary", "message", "text", "content"]:
                if field in result and result[field]:
                    return str(result[field])
            
            # If no specific fields found, return a stringified summary
            return f"Analysis results for {item.get('name', 'query')} on {item.get('date', '')}"
        
        # Fallback description
        return f"Analysis from {item.get('date', 'recent date')}"


# Singleton instance
_luna_generator = None


def get_luna_generator(
    model_name: str = "gpt-4o-mini", model_provider: str = "openai"
) -> LunaGenerator:
    """
    Get or create a singleton instance of LunaGenerator.

    Args:
        model_name: The name of the LLM model to use
        model_provider: The provider of the model

    Returns:
        A LunaGenerator instance
    """
    global _luna_generator
    if _luna_generator is None:
        _luna_generator = LunaGenerator(model_name, model_provider)
    return _luna_generator
