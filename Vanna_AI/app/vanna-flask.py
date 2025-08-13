from vanna.flask import VannaFlaskApp
from ai import AIPipeline
import os

CHAT_MODEL = os.environ.get("CHAT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v0.4")
CHAT_MODEL_BASE_URL = os.environ.get("CHAT_MODEL_BASE_URL", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None) # api key for the model endpoint used
USE_INHOUSE_ENDPOINT = os.getenv("USE_INHOUSE_ENDPOINT", "false").lower() == "true" # if set to false, openrouter model endpoints will be used
TEMP = os.environ.get("TEMP", 0.4)
MAX_TOKENS = os.environ.get("MAX_TOKENS", 131072)
DB_PATH = os.environ.get("DB_PATH", "./db")
DATABASE_CONNECTION_STRING = os.environ.get("DATABASE_CONNECTION_STRING")
DB_TYPE = os.environ.get("DB_TYPE", "sqlite")


ai = AIPipeline(
    chat_model=CHAT_MODEL,
    chat_model_url=CHAT_MODEL_BASE_URL,
    open_api_key=OPENAI_API_KEY,
    max_tokens=MAX_TOKENS,
    use_inhouse_endpoints=USE_INHOUSE_ENDPOINT,
    temp=TEMP,
    db_path=DB_PATH,
    db_connection_string=DATABASE_CONNECTION_STRING,
    db_type=DB_TYPE
)

vn = ai.init_vanna()

app = VannaFlaskApp(
    vn,
    allow_llm_to_see_data=True
)

app.run()
