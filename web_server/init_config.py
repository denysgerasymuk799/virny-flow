from fastapi import FastAPI

from database.database_client import DatabaseClient


# Initial configurations
app = FastAPI(title='virny-flow-webserver')

# For local debugging you can use 'localhost'
# host = 'cassandra-node'
host = 'localhost'
port = 8080

db_client = DatabaseClient()
