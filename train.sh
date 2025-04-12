game_name=$1
seed=$2
cuda=$3
wandb_enabled=True
tcl_enabled=False
type=seed_${seed}_kl_notcl
wandb_exp=${game_name}_${type}
proj_name=STORM-ATARI

python -u train.py --config-name STORM \
BasicSettings.n=${wandb_exp} \
BasicSettings.Seed=${seed} \
BasicSettings.env_name="ALE/${game_name}-v5" \
BasicSettings.device="cuda:${cuda}" \
wandb.exp_name=${wandb_exp} \
wandb.project_name=${proj_name} \
wandb.log=${wandb_enabled} \
tcl.use_tcl_loss=${tcl_enabled} \

