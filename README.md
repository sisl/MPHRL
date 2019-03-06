# Model Primitive Hierarchical Lifelong Reinforcement Learning


## Usage

See [`Makefile`](./Makefile). Run the appropriate command as argument to `make` to run the particular experiment.

## Description of Config Options
`ckpt_path`: the checkpoint to load if `restore_model` is `True`.

`math`: the coupling option, i.e. inclusion of `Pi_k` in posterior target

`stable_old`: given that `math` is `True`, should `Pi_k` be computed using the current policy or an old policy

`l2`: whether the individual posterior probabilities are reweighted using an `l2` norm or a `l1` sum

### 8-P&P

`obstacle`: whether there are two walls in the 8-P&P taskset

`obstacle_height`: given `obstacle` is true, the height of the walls

`repeat`: the one-hot vector has to be repeated a couple times to allow the baseline PPO to learn more quickly; 
this is the number of repeats

`redundant`: whether the observation contains stage information of both boxes in 8-P&P, 
or just the stage information of the current box

`bring_close`: the maximum allowable distance for successful `reach above` actions

`drop_close`: the maximum allowable distance for successful `dropping` actions

`drop_width`: the maximum allowable distance for successful `carry` actions

`split`: whether to normalize one-hot vectors in 8-P&P

`bounded`: whether to use bounded distance

`dist_bound`: if so, what is the bounded distance

`dist_obv`: whether to include relative distance between objects in 8-P&P observation

`above_target`: the multiplier of box size in 8-P&P for distance above the box during `reach above` actions

`stage_obv`: whether stage is observable in 8-P&P

`manhattan`: whether distance is calculated using manhattan or l2 distance

### 10-Maze and 8-P&P

`soft`: whether the gating controller outputs softmax or hardmax selection

`weighted`: whether MPHRL reweights the cross entropy based on ground truth label

`restore_model`: whether to restore a checkpoint

`always_restore`: whether to restore the checkpoint for every new task in lifelong learning

`oracle_master`: whether to use oracle gating controller

`old_policy`: whether to use old policy in posterior calculation

`enforced`: whether to minibatch optimation of gating controller in target tasks

`reset`: whether to reset gating controller for new tasks

`transfer`: whether to transfer subpolicies for new tasks

`paths`: the lifelong learning taskset configuration

`survival_reward`: reward for the ant to be alive per timestep

`record`: record videos during training

`prior`: has no effect on MPHRL

`num_cpus`: number of actors

`num_cores`: same as `num_cpus`

`bs_per_cpu`: train batch size

`max_num_timesteps`: horizon

`bs_per_core`: batch size per core during parallel operations

`total_ts`: total timesteps per lifelong leraning task

`prev_timestep`: skipped timesteps during checkpoint restore for faster convergence

`num_batches`: number of training batches per task

`mov_avg`: moving average calculation of accuracies, etc. 
