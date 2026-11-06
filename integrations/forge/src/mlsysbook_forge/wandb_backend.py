"""W&B implementation, imported only by an explicitly online session."""

from .evidence import numeric_metrics


class WandbBackend:
    def __init__(self, *, context, session_id, entity, project):
        import wandb

        self.wandb = wandb
        self.project_path = f"{entity}/{project}"
        self.api = wandb.Api(timeout=30)
        self.run = wandb.init(
            entity=entity, project=project, id=session_id, resume="allow",
            config=context, job_type=context["result_kind"],
            settings=wandb.Settings(disable_git=True, save_code=False, console="off",
                                    x_disable_meta=True, x_disable_stats=True),
        )

    def publish(self, event, path):
        # Hosted runtimes can lose their local journal. A committed immutable
        # artifact is the remote receipt, so reopening does not append history.
        if self.api.artifact_exists(f"{self.project_path}/evidence-{event['id']}:latest", type="lab-evidence"):
            return
        artifact = self.wandb.Artifact(
            name=f"evidence-{event['id']}", type="lab-evidence",
            metadata={"result_kind": event["context"]["result_kind"], "schema_version": 1},
        )
        artifact.add_file(str(path), name="evidence.json")
        self.run.log_artifact(artifact).wait()
        metrics = numeric_metrics(event["payload"], "result")
        self.run.log({**metrics, "forge/event_id": event["id"]})

    def finish(self):
        self.run.finish()
