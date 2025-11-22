import pandas
import matplotlib.pyplot as plt

metrics_path = "/tmp/tinker-examples/rl-loop/metrics.jsonl"
df = pandas.read_json(metrics_path, lines=True)
plt.plot(df["reward/mean"], label="reward/mean")
plt.legend()
plt.show()
