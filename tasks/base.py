# tasks/base.py
from abc import ABC, abstractmethod

class Task(ABC):
    def __init__(self, db_session, task_id: int, **kwargs):
        self.db = db_session
        self.task_id = task_id
        self.params = kwargs

    @abstractmethod
    def load(self):
        """Fetch any needed inputs from the DB."""
        pass

    @abstractmethod
    def run(self):
        """Run the model or retrieval for this task."""
        pass

    @abstractmethod
    def evaluate(self, prediction):
        """Compute metrics (e.g. accuracy, mrr)."""
        pass

    def save_result(self, prediction, metrics):
        """Persist outputs and scores back to DB or file."""
        # e.g. INSERT INTO results (...)
        pass

    def execute(self):
        data = self.load()
        pred = self.run(data)
        metrics = self.evaluate(pred, data)
        self.save_result(pred, metrics)
        return metrics
