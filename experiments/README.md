# virny-flow-experiments

Run Alpine Meadow:
```shell
virny-flow-experiments/external/alpine-meadow $ python3 -m tools.benchmark.exps.run_exp test_exp diabetes 1 3 none 10
```

Work with auto-sklearn on Mac and Windows:
```shell
# Dockerfile is located in external/auto-sklearn/Dockerfile
docker build --platform=linux/amd64 -t autosklearn-dev .

docker run --platform=linux/amd64 -it -v $(pwd):/auto-sklearn autosklearn-dev /bin/bash

python3 -m examples.exps.run_exp test_exp diabetes 1 ./tmp/exp3 3 3072 none 10
```
