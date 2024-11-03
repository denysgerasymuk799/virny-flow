from contextlib import asynccontextmanager
from fastapi import FastAPI, status, Query
from fastapi.responses import JSONResponse

from .database.task_manager_db_client import TaskManagerDBClient
from domain_logic.initial_configuration import create_initial_config_state


cors = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Authorization, Content-Type',
    'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, HEAD, OPTIONS',
}

def register_routes(app: FastAPI, db_client: TaskManagerDBClient, logger):
    @app.options("/{full_path:path}")
    async def options():
        return JSONResponse(status_code=status.HTTP_200_OK, headers=cors, content=None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup actions
        print("Starting up...")
        db_client.connect()
        await create_initial_config_state(db_client)

        yield  # Control passes to the application here

        # Shutdown actions
        print("Shutting down...")
        db_client.close()

    @app.get("/get_worker_task", response_class=JSONResponse)
    async def get_worker_task(exp_config_name: str = Query()):
        high_priority_task = await db_client.read_worker_task_from_db(exp_config_name=exp_config_name)
        logger.info(f'New task was retrieved: {high_priority_task["task_name"]}')
        return JSONResponse(content={"task_name": high_priority_task["task_name"],
                                     "task_guid": str(high_priority_task["_id"]),
                                     "stage_id": high_priority_task["stage_id"]},
                            status_code=status.HTTP_200_OK)

    @app.post("/complete_worker_task", response_class=JSONResponse)
    async def complete_worker_task(exp_config_name: str = Query(), task_guid: str = Query(),
                                   task_name: str = Query(), stage_id: int = Query()):
        done_tasks_count, ready_tasks_count = await db_client.complete_worker_task_in_db(exp_config_name=exp_config_name,
                                                                                         task_guid=task_guid,
                                                                                         task_name=task_name,
                                                                                         stage_id=stage_id)
        logger.info(f'Task {task_name} with task_guid = {task_guid} was successfully completed.')
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Marked {done_tasks_count} document(s) as DONE and {ready_tasks_count} document(s) as READY"})
