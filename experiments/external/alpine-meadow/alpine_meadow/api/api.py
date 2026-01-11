"""Alpine Meadow API."""

from flask import Flask, jsonify, request

try:
    from .context import APIContext
except:  # noqa: E722  # pylint: disable=bare-except
    from context import APIContext


# api
api = Flask(__name__)
context = APIContext()


@api.route('/version', methods=['GET'])
def get_version():
    response = jsonify({'version': context.version})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@api.route('/tasks', methods=['GET'])
def get_tasks():
    response = jsonify({'tasks': context.get_tasks()})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@api.route('/task', methods=['POST'])
def register_task():
    task = request.get_json()['task']
    context.register_task(task)
    return jsonify(f"Task {task['id']} registered")


@api.route('/pipelines', methods=['GET'])
def get_pipelines():
    task_id = request.args.get('task_id')
    pipelines = context.get_pipelines(task_id)
    response = jsonify({'pipelines': pipelines})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@api.route('/pipelines', methods=['POST'])
def register_pipelines():
    request_json = request.get_json()
    task_id = request_json['task_id']
    pipelines = request.get_json()['pipelines']
    context.register_pipelines(task_id, pipelines)
    return jsonify(f'{len(pipelines)} pipelines registered')


if __name__ == "__main__":
    api.run()
