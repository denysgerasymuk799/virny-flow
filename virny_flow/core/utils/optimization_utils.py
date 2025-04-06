import time
from virny_flow.core.custom_classes.async_counter import AsyncCounter


def has_remaining_time_budget(max_time_budget: int, start_time):
    finish_time = time.perf_counter()
    if max_time_budget > finish_time - start_time:
        return True
    return False


async def has_remaining_pipelines_budget(max_total_pipelines_num: int, total_pipelines_counter: AsyncCounter):
    cur_total_pipelines_num = await total_pipelines_counter.get_value()
    if max_total_pipelines_num > cur_total_pipelines_num:
        return True
    return False
