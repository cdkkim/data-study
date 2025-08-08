# 채용 인터뷰 질문 생성 에이전트

## Setup

```bash
pip install -r requirements.txt
```

## Run

```
export OPENAI_API_KEY=<openai-api-key>
export TAVILY_API_KEY=<tavily-api-key>
```

```python
python interview_simulator.py --resume <path/to/resume.pdf> --job <job.url>
```

## Result

See [result/](result/) for internal documentation and agent architecture details.

![Question 1](result/q1.png)
![Question 2](result/q2.png)
![Question 3](result/q3.png)
![Question 4](result/q4.png)
![Question 5](result/q5.png)
