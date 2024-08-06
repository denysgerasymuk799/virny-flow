from fastapi import status, Query
from fastapi.responses import JSONResponse

from init_config import app, db_client, cors
from domain_logic.custom_logger import logger
from domain_logic.execution_plan import create_execution_plan


@app.options("/{full_path:path}")
async def options():
    return JSONResponse(status_code=status.HTTP_200_OK, headers=cors, content=None)


@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    db_client.connect()

    # Create an optimized execution plan
    await create_execution_plan(db_client)


@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down...")
    db_client.close()


@app.get("/get_worker_task", response_class=JSONResponse)
async def get_worker_task(exp_config_name: str = Query()):
    high_priority_task = await db_client.read_worker_task_from_db(exp_config_name=exp_config_name)
    logger.info(f'New task was retrieved: {high_priority_task["task_name"]}')
    return JSONResponse(content={"task_name": high_priority_task["task_name"],
                                 "task_id": str(high_priority_task["_id"])},
                        status_code=status.HTTP_200_OK)


@app.post("/complete_worker_task", response_class=JSONResponse)
async def complete_worker_task(task_id: str = Query(), task_name: str = Query()):
    modified_count = await db_client.complete_worker_task_in_db(task_id=task_id)
    logger.info(f'Task {task_name} with task_id = {task_id} was successfully completed.')
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Modified {modified_count} document(s)"})


# For local development
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080)
