set -e
python -m flake8 --config=tools/build-support/flake8 --exclude alpine_meadow/common/proto alpine_meadow tests setup.py
python -m pylint --rcfile=tools/build-support/pylintrc  --output-format=parseable --jobs=4 alpine_meadow tests
