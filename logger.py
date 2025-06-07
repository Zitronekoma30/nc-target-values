import wandb

def create_run(config):
    conf = vars(config)


    run = wandb.init(
        entity="leon-andrassik-paris-lodron-universit-t-salzburg",
        project="nc-adaptive-tv",

        config = conf
    )

    return run
