import json
import os

from gym.envs.registration import register

from config import c, n
from test_env.envs import *

seed_json = '/tmp/seed.json'


if os.path.exists(seed_json):
    # NOTE: WORKERS
    with open(seed_json, 'r') as f:
        seed = json.load(f)['seed']
    print('WORKER SEED', seed)
else:
    seed = 10
    print('NOISE: %s %s' % (c.Tnoise, c.Fnoise))
    print('SOLVED: %s' % c.solved_threshold)
    print('ABOVE TARGET: %s' % c.above_target)
    print('RUN: %s' % c.xml_name)
    print('pg_sub_lr %s' % c.pg_sub_lr)
    print('pg_master_ce_lr_source %s' % c.pg_master_ce_lr_source)
    print('REPEAT %s' % c.repeat)
    print('update_master_interval %s' % c.update_master_interval)
    print('reset %s' % c.reset)
    # NOTE: MASTER
    with open(seed_json, 'w') as f:
        json.dump({'seed': seed}, f)
    print('MASTER SEED', seed, 'can leave now...')
    print(n(c, seed))

env_id = c.env_name.replace('-', '_')
for i in range(c.num_tasks):
    id = '%s%s' % (i, c.env_name)
    if c.env_type == 'mujoco':
        register(
            id=id,
            entry_point='test_env.envs.%s:%s' % (env_id, env_id),
            max_episode_steps=c.max_num_timesteps,
            reward_threshold=60000.0,
            kwargs={
                'paths': c.paths[i],
                'seed': seed
            }
        )
    else:
        domain, task, _ = c.xml_name.split('-')
        register(id=id,
                 entry_point='test_env.envs.%s:%s' % (env_id, env_id),
                 kwargs={'domain': c.domain, 'task': task, 'task_id': i})

register(
    id='Fourrooms-small-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'small',
    })

register(
    id='Stochastic-Fourrooms-small-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'small',
        'stochastic': True,
        'slip_away': True,
    })

register(
    id='Stochastic-Fourrooms-small-v1',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'small',
        'stochastic': True,
        'slip_away': True,
        'dense_reward': True
    })

register(
    id='Fourrooms-medium-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'medium'
    })

register(
    id='Stochastic-Fourrooms-medium-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'medium',
        'stochastic': True,
        'slip_away': True,
    })

register(
    id='Stochastic-Fourrooms-medium-v1',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'medium',
        'stochastic': True,
        'slip_away': True,
        'dense_reward': True
    })

register(
    id='Fourrooms-large-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'large',
    })

register(
    id='Stochastic-Fourrooms-large-v0',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'large',
        'stochastic': True,
        'slip_away': True,
    })

register(
    id='Stochastic-Fourrooms-large-v1',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': 'large',
        'stochastic': True,
        'slip_away': True,
        'dense_reward': True
    })

register(id='KeyDoor-v1', entry_point='test_env.envs.key_door:KeyDoor', )
