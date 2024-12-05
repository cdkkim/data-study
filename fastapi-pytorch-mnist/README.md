
# fastapi-pytorch-mnist
Run FastAPI server with pytorch model for mnist dataset

endpoint is http://localhost:8000/mnist

## Prerequisite
- REST API

## Development

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

## Test

0. Install dependencies
```bash
pip install -r requirements.txt
```

1. Build model
```bash
python mnist.py --save-model
```

2. Run server
```bash
fastapi dev main.py
```

3. Run test script
```bash
python test.py
```

## Output

### Test 0utput
![test output](docs/test_output.png "Test output")

### Server log
![server log](docs/server_log.png "Server log")

