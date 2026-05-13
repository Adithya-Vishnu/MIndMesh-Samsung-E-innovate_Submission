import pandas as pd
import os
import time
class MetricsLogger:
    def __init__(self):
        self.logs = []
    def log(self,
        command,
        parse_success,
        perception_success,
        motion_success,
        final_success,
        execution_time,
        ik_error):
        self.logs.append({
            "timestamp": time.time(),
            "command": command,
            "parse_success": parse_success,
            "perception_success": perception_success,
            "motion_success": motion_success,
            "final_success": final_success,
            "execution_time": execution_time,
            "ik_error": ik_error,
            })
    def save(self, path="outputs/metrics/results.csv"):
        os.makedirs("outputs/metrics", exist_ok=True)
        df = pd.DataFrame(self.logs)
        df.to_csv(path, index=False)
        print(f"Metrics saved -> {path}")