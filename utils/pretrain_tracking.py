"""Optional W&B tracking; callers synchronize rank-zero failures across ranks."""


def start_tracking(config, output, *, resume=False):
    tracking = config.get("tracking")
    if not tracking or not tracking.get("enabled", True):
        return None
    import wandb
    run = wandb.init(entity=tracking.get("entity"), project=tracking["project"],
                     name=tracking["name"], id=tracking["id"], resume="allow" if resume else "never",
                     config=config, dir=str(output))
    run.define_metric("update")
    run.define_metric("*", step_metric="update")
    return run


def log_update(run, row, epoch_steps):
    if run is not None:
        run.log(dict(update=row["step"], epoch=row["step"]/epoch_steps,
                     **{"train/"+key: value for key, value in row.items() if key != "step"}))


def log_panels(run, report, path, epoch_steps):
    if run is not None:
        import wandb
        run.log(dict(update=report["step"], epoch=report["step"]/epoch_steps,
                     **{"validation/"+key: value for key, value in report["metrics"].items()},
                     denoising=wandb.Image(str(path))))
