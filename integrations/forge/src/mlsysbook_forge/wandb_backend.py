"""W&B implementation, imported only by an explicitly online session."""

from .evidence import numeric_metrics


class WandbBackend:
    def __init__(self, *, context, session_id, entity, project):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            entity=entity, project=project, id=session_id, resume="allow",
            config=context, job_type=context["result_kind"],
            settings=wandb.Settings(disable_git=True, save_code=False, console="off"),
        )

    def publish(self, event, path):
        metrics = numeric_metrics(event["payload"], "result")
        self.run.log({**metrics, "forge/event_id": event["id"]})
        artifact = self.wandb.Artifact(
            name=f"evidence-{event['id']}", type="lab-evidence",
            metadata={"result_kind": event["context"]["result_kind"], "schema_version": 1},
        )
        artifact.add_file(str(path), name="evidence.json")
        self.run.log_artifact(artifact).wait()

    def finish(self):
        self.run.finish()
