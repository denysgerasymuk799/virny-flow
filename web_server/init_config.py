from fastapi import FastAPI

from database.database_client import DatabaseClient


# Config objects
app = FastAPI(title='virny-flow-webserver')
db_client = DatabaseClient()

# Global constants
# host = 'cassandra-node'
host = 'localhost'
port = 8080

cors = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Authorization, Content-Type',
    'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, HEAD, OPTIONS'
}
