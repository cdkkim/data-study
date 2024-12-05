
# fastapi-pytorch-mnist
Run FastAPI server with pytorch model for mnist dataset

endpoint is http://localhost:8000/mnist

## Prerequisite
- REST API

## Steps

1. make and source virtualenv

```bash
python3 -m venv ~/.venv/app
```

2. Install [FastAPI](https://fastapi.tiangolo.com/#alternative-api-docs)

```bash
pip install "fastapi[standard]"
fastapi dev main.py
```

3. Install torch torchvision

```bash
pip install torch torchvision
```

4. Train [pytorch/examples/mnist](https://github.com/pytorch/examples/blob/main/mnist/main.py)

5. Add a POST request handler where input is an array of integers(mnist image) and output is predicted value

6. Run server

