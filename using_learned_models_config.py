import socket

import numpy as np
import tensorflow as tf


def value(dict, k):
    for key in dict:
        if k == key or k in key:
            return dict[key]
    raise NotImplementedError


def n(c, seed):
    if c.env_type == 'mujoco':
        return 'rrw-%s-%s-bs=%s-c=%s-%s-up' \
               '=%s-%s-%s-%s-bs=%s-%s-' \
               'ni=%s-%s-ld=%s-max=%s-mlr-%s' \
               '-%s-k=%s-gl=%s-cw=%s-sf=%s-tf=%s-mv=%s-rt=%s-sl=%s-sof' \
               '=%s-se=%s-om=%s-ol=%s-gr=%s-%s-pos=%s-lr=%s-%s-e=%s-mt=%s' \
               '%s%s-w=%s%s%s-%s%s-%sx%s-%s-%s%s-%s' % (
               c.xml_name, c.algo, c.bs_per_cpu, c.num_cpus, c.short[c.hostname], c.update_master_interval,
               c.pg_pi_hidden_fc_size, c.pg_master_hidden_fc_size, c.pg_bl_hidden_fc_size, c.mini_bs, c.optim_epochs,
               c.Tnoise, c.Fnoise, c.short[c.restore_model], c.max_num_timesteps, c.pg_master_pg_lr, c.stop_pg_ts_perc,
               c.num_models, str(c.goal).replace(' ', '_'), c.cor_w, c.safe_m_loss, c.short[c.transfer], c.mov_avg,
               c.short[c.reset], c.solved_threshold, c.short[c.soft], str(seed), c.short[c.oracle_master],
               c.short[c.old_policy], c.goal_r, c.forward_mult, c.short[c.pos_r], c.pg_master_ce_lr_source,
               c.pg_master_ce_lr_target, c.short[c.enforced], c.short[c.math], c.short[c.stable_old], c.short[c.l2],
               c.short[c.weighted], c.short[c.oracle_master], c.short[c.always_restore],
               c.short[c.learn_model_primitive], c.short[c.use_model_primitive], c.pg_mp_n_layers,
               c.pg_mp_hidden_fc_size, c.pg_mp_lr, c.short[c.model_primitive_delta], c.short[c.model_primitive_pg],
               int(c.success_dist))
    elif c.env_type == 'dm_control':
        return '13-%s-%s-i=%s-bs=%s-c=%s-%s-up' \
               '=%s-%s-%s-%s-en=%s-bs=%s-%s-' \
               'ni=%s-%s-l=%s-m=%s-lr-%s' \
               '-%s-%s-k=%s-sf=%s-tf=%s-mv=%s-rt=%s-%s-sf=%s' \
               '-se=%s-om=%s-ol=%s-lr=%s-%s-e=%s-di=%s-st=%s-b=%s-%s-mh' \
               '=%s-ce=%s%s%s-at=%s-mt=%s%s%s-ob=%s%s-%s%s%s%s' % (
               c.xml_name, c.algo, c.num_batches, c.bs_per_cpu, c.num_cpus, c.short[c.hostname],
               c.update_master_interval, c.pg_pi_hidden_fc_size, c.pg_master_hidden_fc_size, c.pg_bl_hidden_fc_size,
               c.ent_coeff, c.mini_bs, c.optim_epochs, c.Tnoise, c.Fnoise, c.short[c.restore_model],
               c.max_num_timesteps, c.pg_master_pg_lr, c.stop_pg_ts_perc, c.pg_sub_lr, c.num_models, c.safe_m_loss,
               c.short[c.transfer], c.mov_avg, c.short[c.reset], c.solved_threshold, c.short[c.soft], str(seed),
               c.short[c.oracle_master], c.short[c.old_policy], c.pg_master_ce_lr_source, c.pg_master_ce_lr_target,
               c.short[c.enforced], c.short[c.dist_obv], c.short[c.stage_obv], c.short[c.bounded], c.dist_bound,
               c.short[c.manhattan], c.bring_close, c.drop_close, c.drop_width, c.above_target, c.short[c.math],
               c.short[c.stable_old], c.short[c.l2], c.short[c.obstacle], c.obstacle_height, c.short[c.redundant],
               c.short[c.split], c.repeat, c.short[c.weighted])


class c(object):
    task_id = 0
    # NOTE: RECORDING
    xml_name = 'TMaze-Ant-v2'
    env_name = 'Basic%s' % (xml_name[xml_name.find('-'):])
    cor_w = value({'Basic-Humanoid-v2': 7, 'Basic-Ant-v2': 1.5, ('Basic-bring_ball-v0', 'Basic-stack_2-v0'): 1.0},
        env_name)
    cor_wd = {'E': cor_w * 2, 'N': cor_w * 2, 'W': cor_w * 0.85, 'S': cor_w * 0.85, 'U': None}
    paths = np.array(value({'T0-Ant-v2': [[[[0, -8], [0, 0], [8, 0]], [[0, 8], [0, 0], [8, 0]], [[0, 0], [8, 0]]]],
                            'T1-Ant-v2': [[[[8, 0], [0, 0], [0, 8]], [[-8, 0], [0, 0], [0, 8]], [[0, 0], [0, 8]]]],
                            'T2-Ant-v2': [[[[0, -8], [0, 0], [-8, 0]], [[0, 8], [0, 0], [-8, 0]], [[0, 0], [-8, 0]]]],
                            'T3-Ant-v2': [[[[8, 0], [0, 0], [-0, -8]], [[-8, 0], [0, 0], [0, -8]], [[0, 0], [0, -8]]]],
                            ('TS-Ant-v2', 'BS-Ant-v2'): [[[0, 0], [12, 0], [12, 12], [-6, 12], [-6, 0]]],
                            ('BL-Ant-v2', 'TL-Ant-v2', 'TL-Humanoid-v2', 'BL-Humanoid-v2'): [
                                [[0, 0], [12, 0], [12, 12]]],
                            ('TT-Ant-v2', 'LL-Ant-v2'): [[[0, 0], [12, 0], [12, 12], [24, 12], [24, 24]]], (
                            'TMaze-Ant-v2', 'RMaze-Ant-v2', 'BMaze-Ant-v2', 'CMaze-Ant-v2', 'VMaze-Ant-v2',
                            '2Maze-Ant-v2', '5Maze-Ant-v2'): [[[[0, 0], [8, 0], [8, -8], [16, -8], [16, 0], [24, 0]]],
                                                              [[[0, 0], [0, 8], [8, 8], [8, 0], [16, 0], [16, -8]]],
                                                              [[[0, 0], [-8, 0], [-8, 8], [0, 8], [0, 16], [8, 16]]], [
                                                                  [[0, 0], [0, -8], [-8, -8], [-8, -16], [-16, -16],
                                                                   [-16, -8]]], [
                                                                  [[0, 0], [8, 0], [8, -8], [16, -8], [16, -16],
                                                                   [24, -16]]], [
                                                                  [[0, 0], [0, 8], [-8, 8], [-8, 16], [-16, 16],
                                                                   [-16, 24]]],
                                                              [[[0, 0], [8, 0], [8, 8], [0, 8], [0, 16], [-8, 16]]],
                                                              [[[0, 0], [0, 8], [8, 8], [8, 16], [0, 16], [0, 24]]],
                                                              [[[0, 0], [8, 0], [8, -8], [0, -8], [0, -16], [8, -16]]],
                                                              [[[0, 0], [0, -8], [-8, -8], [-8, -16], [-16, -16],
                                                                [-16, -24]]]],
                            # [[[[0, -0.25], [6, -0.25], [6, -4], [13, -4], [13, 3], [6, 3]]],
                            #  [[[0, 0], [0, 6], [6, 6], [6, 0], [10, 0], [10, -5]]],
                            #  [[[0, -0.25], [-5, -0.25], [-5, 7], [1, 7], [1, 3], [8, 3]]],
                            #  [[[0, 0], [0, -5], [-5, -5], [-5, -10], [-11, -10], [-11, -5]]],
                            #  [[[0, 0], [7, 0], [7, -7], [11, -7], [11, -13], [17, -13]]],
                            #  [[[0, 0], [0, 5], [-6, 5], [-6, 12], [-13, 12], [-13, 18]]],
                            #  [[[0, 0], [5, 0], [5, 4], [-2, 4], [-2, 9], [-9, 9]]],
                            #  [[[0, 0], [0, 7], [7, 7], [7, 14], [1, 14], [1, 19]]],
                            #  [[[0, 0], [6, 0], [6, -4], [1, -4], [1, -10], [5, -10]]],
                            #  [[[0, 0], [0, -7], [-7, -7], [-7, -11], [-12, -11], [-12, -18]]]],
                            ('SP-bring_ball-v0', 'SP-stack_2-v0', 'V-stack_2-v0', 'A-stack_2-v0', '2-stack_2-v0'): [
                                [[0, 0], [1, 0]]],
                            ('SPll-stack_2-v0', 'All-stack_2-v0', '2ll-stack_2-v0', 'Vll-stack_2-v0'): [
                                [[0, 0], [1, 0]], [[1, 1], [0, 0], [0, 1]], [[1, 0], [1, 1]], [[0, 0], [0, 1], [1, 1]],
                                [[0, 1], [1, 1], [1, 0]], [[0, 1], [0, 0], [0, 1]], [[1, 0], [0, 0]],
                                [[1, 1], [0, 1], [0, 0]], ]}, xml_name))
    env_type = 'mujoco' if xml_name.endswith('v2') else 'dm_control'

    ckpt_path = 'rr-TMaze-Ant-v2-vppo-bs=2048-c=16-tv-up=1-16-64-64-bs=4096-10-ni=0-0.5-ld=F-max=2048-mlr-0-1-k=4-gl' \
                '=None-cw=1.5-sf=20.0-tf=T-mv=10-rt=T-sl=0.7-sof=T-se=0-om=F-ol=F-gr=200-50-pos=T-lr=0.001-0.003-e=T' \
                '-mt=FFF-w=FFF-FF-2x64-0.0003-TF-3/44'

    math = False
    stable_old = math
    l2 = False
    # NOTE: DM_CONTROL
    obstacle = True
    obstacle_height = 0.09
    repeat = 10
    redundant = True
    bring_close = 1.5
    drop_close = 1.5
    drop_width = 1.5
    split = False
    learn_model_primitive = value(
        {('T0-Ant-v2', 'T1-Ant-v2', 'T2-Ant-v2', 'T3-Ant-v2'): True, ('BMaze-Ant-v2', "TMaze-Ant-v2"): False}, xml_name)
    use_model_primitive = value(
        {('T0-Ant-v2', 'T1-Ant-v2', 'T2-Ant-v2', 'T3-Ant-v2', 'BMaze-Ant-v2'): False, "TMaze-Ant-v2": True}, xml_name)
    model_primitive_checkpoints = [
        'rr-T0-Ant-v2-vppo-bs=2048-c=16-bg-up=1-16-64-64-bs=4096-10-ni=0-0.5-ld=F-max=2048-mlr-0-1-k=1-gl=None-cw=1.5'
        '-sf=20.0-tf=T-mv=10-rt=T-sl=1.5-sof=T-se=40-om=F-ol=F-gr=200-50-pos=T-lr=0.001-0.003-e=T-mt=FFF-w=FFF-TF'
        '-2x64-0.0003-TF-3/70',
        'rr-T1-Ant-v2-vppo-bs=2048-c=16-g1-up=1-16-64-64-bs=4096-10-ni=0-0.5-ld=F-max=2048-mlr-0-1-k=1-gl=None-cw=1.5'
        '-sf=20.0-tf=T-mv=10-rt=T-sl=1.5-sof=T-se=40-om=F-ol=F-gr=200-50-pos=T-lr=0.001-0.003-e=T-mt=FFF-w=FFF-TF'
        '-2x64-0.0003-TF-3/70',
        'rr-T2-Ant-v2-vppo-bs=2048-c=16-oc-up=1-16-64-64-bs=4096-10-ni=0-0.5-ld=F-max=2048-mlr-0-1-k=1-gl=None-cw=1.5'
        '-sf=20.0-tf=T-mv=10-rt=T-sl=1.5-sof=T-se=40-om=F-ol=F-gr=200-50-pos=T-lr=0.001-0.003-e=T-mt=FFF-w=FFF-TF'
        '-2x64-0.0003-TF-3/70',
        'rr-T3-Ant-v2-vppo-bs=2048-c=16-jp-up=1-16-64-64-bs=4096-10-ni=0-0.5-ld=F-max=2048-mlr-0-1-k=1-gl=None-cw=1.5'
        '-sf=20.0-tf=T-mv=10-rt=T-sl=1.5-sof=T-se=40-om=F-ol=F-gr=200-50-pos=T-lr=0.001-0.003-e=T-mt=FFF-w=FFF-TF'
        '-2x64-0.0003-TF-3/70']
    bounded = False
    dist_bound = 0.08
    dist_obv = True
    above_target = 3
    stage_obv = True
    manhattan = False
    model_primitive_pg = False
    model_primitive_delta = True
    # NOTE: MUJOCO
    soft = True
    weighted = False
    restore_model = False
    always_restore = False
    oracle_master = False
    old_policy = False
    enforced = True
    reset = value({('SPll-stack_2-v0', 'SP-stack_2-v0'): True,
        ('All-stack_2-v0', 'A-stack_2-v0', '2ll-stack_2-v0', 'Vll-stack_2-v0'): True, (
        'TMaze-Ant-v2', 'CMaze-Ant-v2', 'TS-Ant-v2', 'BS-Ant-v2', 'VMaze-Ant-v2', '2Maze-Ant-v2', 'TL-Ant-v2',
        'BL-Ant-v2', '5Maze-Ant-v2', 'T0-Ant-v2', 'T1-Ant-v2', 'T2-Ant-v2', 'T3-Ant-v2'): True, 'BMaze-Ant-v2': False},
        xml_name)
    transfer = value({('SPll-stack_2-v0', 'SP-stack_2-v0'): False,
        ('All-stack_2-v0', 'A-stack_2-v0', '2ll-stack_2-v0', 'Vll-stack_2-v0'): True, (
        'TMaze-Ant-v2', 'CMaze-Ant-v2', 'TS-Ant-v2', 'BS-Ant-v2', 'VMaze-Ant-v2', '2Maze-Ant-v2', 'TL-Ant-v2',
        'BL-Ant-v2', '5Maze-Ant-v2', 'T0-Ant-v2', 'T1-Ant-v2', 'T2-Ant-v2', 'T3-Ant-v2'): True, 'BMaze-Ant-v2': False},
        xml_name)
    oracle_turn_on = -1

    num_tasks = paths.shape[0]
    if 'stack_2' in xml_name:
        box_to_target = []
        for path in paths:
            d = {}
            for i in range(len(path)):
                box = path[i][0]
                if box not in d:
                    target = i
                    d[box] = target
            box_to_target.append(d)

        target_height = np.array(value({
            ('All-stack_2-v0', 'SPll-stack_2-v0', '2ll-stack_2-v0', 'Vll-stack_2-v0'): [[0, 1], [0, 0, 1], [0, 0],
                                                                                        [0, 0, 1], [0, 1, 0], [0, 0, 0],
                                                                                        [0, 1], [0, 1, 0]],
            ('SP-stack_2-v0', 'A-stack_2-v0', '2-stack_2-v0'): [[0, 1]]}, xml_name))

        target_to_drop = [np.array([i[1] for i in path]) for path in paths]
        box_of_interest = [np.array([i[0] for i in path]) for path in paths]

    domain = value({'Basic-bring_ball-v0': 'manipulator', 'Basic-stack_2-v0': 'stacker', 'Basic-Ant-v2': 'maze'},
        env_name)
    # NOTE: gamma -> discount factor
    gamma = 0.99
    record_all_lifelong_tasks = False
    stop_pg_ts_perc = 1
    survival_reward = 0
    # NOTE: algorithm
    algo = 'vppo'
    record = False
    prior = 0.25
    # NOTE: # trajectory generation
    num_cpus = value({'mujoco': 16, 'dm_control': 24}, env_type)
    num_cores = num_cpus
    bs_per_cpu = value({'dm_control': 1536, 'mujoco': 2048}, env_type)
    max_num_timesteps = bs_per_cpu
    bs_per_core = bs_per_cpu * num_cpus // num_cores + 1
    total_ts = int(value({'dm_control': 3e7, 'mujoco': 3e7}, env_type))
    prev_timestep = int(restore_model)
    num_batches = (total_ts - prev_timestep) // num_cpus // bs_per_cpu
    decay_steps_factor = 0.5
    decay_steps = int(num_batches * decay_steps_factor)
    # NOTE: alpha -> learning rate of Q's
    alpha_begin = 0.01
    alpha_end = 0.005
    # NOTE: beta -> learning rate of priors
    beta = 0.5
    # NOTE: e-greedy params
    eps_begin = 1.0
    eps_end = 0.01
    lam = 0.95
    # NOTE: multiple runs
    num_models = value(
        {('A-stack_2-v0', 'All-stack_2-v0'): 12, '5Maze-Ant-v2': 5, ('Vll-stack_2-v0', 'V-stack_2-v0'): 6,
            ('VMaze-Ant-v2', '2Maze-Ant-v2', '2ll-stack_2-v0'): 2, (
        'Stochastic-Fourrooms-large-v0', 'Stochastic-Fourrooms-small-v0', 'Stochastic-Fourrooms-small-v1',
        'SquareNoDist-Ant-v2', 'TMaze-Ant-v2', 'RMaze-Ant-v2', 'L-Ant-v2', 'TS-Ant-v2', 'CMaze-Ant-v2'): 4, (
        'TL-Ant-v2', 'L-Swimmer-v2', 'TL-Humanoid-v2', 'T-Humanoid-v2', 'LL-Humanoid-v2', 'L2-Ant-v2',
        'BL-Humanoid-v2'): 2,
            ('BL-Ant-v2', 'BS-Ant-v2', 'BMaze-Ant-v2', 'T0-Ant-v2', 'T1-Ant-v2', 'T2-Ant-v2', 'T3-Ant-v2'): 1,
            ('SP-bring_ball-v0', 'SP-stack_2-v0', 'SPll-stack_2-v0'): 1}, xml_name)
    # NOTE: plot params
    mov_avg = 10
    obscure_factor = 0
    savefig = False
    dpi = 300
    # NOTE: deep neural network for learn_dynamics
    n_layers = 10
    hidden_fc_size = 256

    num_epochs = 100000
    base_lr = 0.0001
    # NOTE: PG baseline network
    pg_bl_n_layers = 2
    pg_bl_hidden_fc_size = value({'dm_control': 64, 'mujoco': 64}, env_type)
    pg_bl_lr = 3e-4
    # NOTE: Model Primitives
    pg_mp_n_layers = 2
    pg_mp_hidden_fc_size = value({'dm_control': 64, 'mujoco': 64}, env_type)
    pg_mp_lr = 3e-4

    # NOTE: PG master network
    pg_master_n_layers = 2
    pg_master_hidden_fc_size = value({'dm_control': 64, 'mujoco': 64}, env_type)
    pg_master_ce_lr_source = value({'dm_control': 3e-2, 'mujoco': 1e-3}, env_type)
    pg_master_ce_lr_target = 3e-3
    pg_master_pg_lr = value(
        {'Basic-Humanoid-v2': 0, 'Basic-Ant-v2': 0, ('Basic-bring_ball-v0', 'Basic-stack_2-v0'): 0, }, env_name)
    # NOTE: PG main network
    pg_pi_n_layers = 2
    pg_pi_hidden_fc_size = value({('SPll-stack_2-v0', 'SP-stack_2-v0', 'BS-Ant-v2', 'BL-Ant-v2', 'BMaze-Ant-v2'): 64,
        ('All-stack_2-v0', 'A-stack_2-v0', 'Vll-stack_2-v0'): 16, (
        'TMaze-Ant-v2', 'CMaze-Ant-v2', 'TS-Ant-v2', 'TL-Ant-v2', 'VMaze-Ant-v2', '2Maze-Ant-v2', '5Maze-Ant-v2',
        'T0-Ant-v2', 'T1-Ant-v2', 'T2-Ant-v2', 'T3-Ant-v2'): 16, '2ll-stack_2-v0': 16}, xml_name)
    pg_sub_lr = 3e-4
    # NOTE: ppo clip
    clip_param = 0.2
    # NOTE: debug
    update_master_interval = 1
    execution_stats_interval = 10
    activation_fn_str = 'tanh'
    activation_fn = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'None': None}[activation_fn_str]
    lr_step_interval = 100
    validate = False  # NOTE: validate: whether to verify logic
    log_interval = 1
    mini_bs = 256 * num_cpus
    ent_coeff = 0
    optim_epochs = 10
    eval_num_rollouts = num_cpus
    hostname = socket.gethostname()
    hostname = 'SUNet' if 'SUNet' in hostname else hostname
    gpu = True if hostname in ['astoria'] else False
    Tnoise = 0
    Fnoise = 0.5
    # NOTE: joint update params
    stop_pg_ts = total_ts * stop_pg_ts_perc
    gradclip = 20000.

    # NOTE: Tver: 4->3, 2->4; bethpage: current
    viewer = True
    view_sub = False
    # NOTE: GIF CONFIG
    line_pad = 15
    thickness = 1
    gif_s = 512
    gif_format = 'gif'
    fontscale = 0.5
    goal_r = 200
    num_eval = 70
    safe_m_loss = 20.0
    # NOTE: ENV
    fix_sub = False
    goal = None
    viewer_dist = 1.2
    success_dist = cor_w * 2
    solved_threshold = value(
        {('T0-Ant-v2', 'T1-Ant-v2', 'T2-Ant-v2', 'T3-Ant-v2'): 1.5, ("BMaze-Ant-v2", "TMaze-Ant-v2"): 0.7},
        xml_name) if env_type == 'mujoco' else 0.75
    corridors = None
    wl_h = 0.8
    wl_w = 0.25
    warmup = 0.1
    e = 1e-20
    max_frames = 1000
    forward_mult = 50
    pos_r = True
    short = {True: 'T', False: 'F', 'SUNet': 'SU', 'sislgc1': 'g1', 'sislgc2': 'g2', 'sislgc3': 'g3', 'sislgc4': 'g4',
        'sislgc5': 'g5', 'sislgc6': 'g6', 'sislgc7': 'g7', 'sislgc8': 'g8', 'sislgc9': 'g9', 'sislgc10': 'g10',
        'sislgc11': 'g11', 'sislgc12': 'g12', 'sislgc13': 'g13', 'sislgc14': 'g14', 'sislgc15': 'g15',
        'sislgc16': 'g16', 'sislgc17': 'g17', 'sislgc18': 'g18', 'sislgc19': 'g19', 'sislgc20': 'g20',
        'sislgc21': 'g21', 'sislgc22': 'g22', 'sislgc23': 'g23', 'sislgc24': 'g24', 'sislgc25': 'g25',
        'sislgc26': 'g26', 'sislgc27': 'g27', 'sislgc28': 'g28', 'sislgc29': 'g29', 'sislgc30': 'g30', 'bethpage': 'bg',
        'cambridge': 'ca', 'tula': 'tl', 'tver': 'tv', 'jodhpur': 'jp', 'astoria': 'as', 'cheonan': 'ch',
        'oceanside': 'oc', 'Bohans-MacBook-Pro.local': 'mac', 'dyn-160-39-154-237.dyn.columbia.edu': 'mac', }

    dir_to_model = {'E': 0, 'N': 1, 'W': 2, 'S': 3, }
    # NOTE: GENERATE MAZES
    num_mazes = 20

    num_corr = [5] * num_mazes
    max_corr_len = 8
    min_corr_len = 4
    finalize = True
    run_stds = [[4.599198, 4.6641273, 1.28923, 2.454618, 0.099999994, 0.099999994, 0.12651926, 0.44238856, 0.24519706,
                 0.27842945, 0.3273297, 0.3899642, 0.2788939, 0.39544523, 0.2628939, 0.37652737, 0.2529815, 0.39808935,
                 0.27402195, 1.2319881, 0.6234127, 0.6694361, 1.0196308, 1.0814291, 1.2173401, 2.8830736, 3.0794158,
                 4.0499864, 1.9910984, 3.6656153, 1.9535313, 4.0345163, 2.2735405, 0.099999994, 0.099999994,
                 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.13444856, 0.13409562, 0.099999994, 0.2144815,
                 0.21478048, 0.2999312, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
                 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.2986342, 0.30273247,
                 0.29329726, 0.30751246, 0.30443984, 0.30402762, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
                 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
                 0.32695666, 0.3283175, 0.32865548, 0.33718628, 0.3334012, 0.3277009, 0.099999994, 0.099999994,
                 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
                 0.099999994, 0.099999994, 0.29600772, 0.29339242, 0.28964946, 0.30079424, 0.30085075, 0.29916844,
                 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
                 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.31006745, 0.31021363, 0.30940148, 0.31883976,
                 0.316916, 0.31288943],
        [1.2312175, 2.5586734, 4.862292, 4.9084835, 0.099999994, 0.099999994, 0.11902989, 0.37661228, 0.19612414,
         0.1970121, 0.3508082, 0.41372088, 0.24030799, 0.37279978, 0.2762219, 0.40963152, 0.25021195, 0.356578,
         0.29620296, 0.68267214, 1.3554753, 0.762895, 1.0119213, 1.1173822, 1.1998409, 4.2042365, 2.1269264, 2.3674908,
         3.1287558, 4.3375616, 2.4620428, 2.987101, 1.4826976, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.11626974, 0.11602408, 0.099999994, 0.17530695, 0.17600684, 0.22305314, 0.099999994,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.099999994, 0.32071027, 0.31924236, 0.3196359, 0.32367983, 0.32657215, 0.31954604,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.3258689, 0.3229741, 0.31455493, 0.3269703, 0.32847512,
         0.32323778, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.28302696, 0.2854431, 0.282894, 0.28799674,
         0.28955868, 0.2865865, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.3557716, 0.3584011, 0.34562215,
         0.36439514, 0.3656799, 0.35458568],
        [0.27116972, 0.2731105, 0.49103335, 0.35940948, 10.0, 10.0, 8.285499, 3.1395023, 5.3231235, 5.5543966, 2.809905,
         2.5080442, 4.685012, 2.5534766, 3.8159177, 2.789227, 3.6269453, 2.5306308, 3.9287062, 0.829414, 1.6191914,
         1.1945486, 0.9497294, 0.88029605, 0.71083105, 0.26208684, 0.6023471, 0.22584441, 0.36695793, 0.33044592,
         0.31234562, 0.23242629, 0.59796476, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.156298, 9.454527, 10.0, 6.133729,
         6.050961, 4.6875067, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 3.3002894,
         3.3518806, 3.370098, 3.2725973, 3.2803993, 3.3199122, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 3.1619124, 3.175854, 3.1968389, 3.0986283, 3.091742, 3.1791952, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 3.7807808, 3.753944, 3.8666487, 3.6851885, 3.694838, 3.7078226, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.9965332, 2.9918234, 2.9949353, 2.9243422,
         2.925929, 3.0298653],
        [1.2362218, 3.4339106, 3.632699, 3.6123943, 0.099999994, 0.099999994, 0.12220171, 0.3551185, 0.23772441,
         0.1933735, 0.347828, 0.35959145, 0.21327308, 0.40731293, 0.2096475, 0.34834144, 0.27511126, 0.38403288,
         0.27539626, 0.5728318, 1.1423793, 0.77614814, 1.1704272, 1.0412856, 1.4850398, 2.3779807, 1.7846571, 4.932697,
         1.9872103, 2.5577078, 3.2989748, 3.4681187, 2.5812807, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.115986496, 0.11946792, 0.099999994, 0.18283401, 0.18193267, 0.24615471,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.2910546, 0.2980322, 0.28877378, 0.29828975, 0.29668167,
         0.29561338, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.33946756, 0.34012014, 0.3420155, 0.3465538,
         0.34908473, 0.3354624, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.2563183, 0.25463828,
         0.24056835, 0.2594071, 0.26107442, 0.26175717, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994,
         0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.099999994, 0.3323459,
         0.33377367, 0.32560465, 0.3441525, 0.34077618, 0.3317802],

    ]
    run_std = value({
        'Basic-Ant-v2': [0.0001, 0.0001, 0.0001, 0.0001, 11926939, 19020905, 0.12928092, 0.6212888, 0.25102094,
                         0.22332077, 0.58209974, 0.42657858, 0.2656426, 0.40442517, 0.27392906, 0.40158844, 0.27972764,
                         0.40863708, 0.30180925, 0.7156728, 0.77419865, 0.65207326, 1.1297017, 1.106495, 1.3923764,
                         2.8920958, 2.2228127, 3.126648, 2.608846, 3.370323, 2.8734233, 4.168823, 2.3023212, 0.09999999,
                         0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.14755759, 0.14737378, 0.09999999,
                         0.21581785, 0.21529765, 0.25864798, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999,
                         0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.3328616,
                         0.33258832, 0.3209721, 0.33333594, 0.33337238, 0.3281808, 0.09999999, 0.09999999, 0.09999999,
                         0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999,
                         0.09999999, 0.36311337, 0.36590517, 0.35446492, 0.36720052, 0.36716765, 0.35422853, 0.09999999,
                         0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999,
                         0.09999999, 0.09999999, 0.09999999, 0.29842412, 0.29977155, 0.29137745, 0.3030151, 0.3025487,
                         0.2990263, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999,
                         0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.36081558, 0.36251754, 0.35274118,
                         0.36504218, 0.3648821, 0.35191768], 'Basic-Humanoid-v2': np.ones(shape=382),
        'Basic-bring_ball-v0': np.ones(shape=37), # 'Basic-stack_2-v0': np.ones(
        #     shape=48 + 14 * dist_obv + (18 if redundant else 8) + repeat * 2)
        'Basic-stack_2-v0': [0.59313864, 0.5122037, 1.5283116, 1.3284554, 1.0007584, 0.2044712, 0.159425, 0.20430137,
                             0.16148166, 0.16000782, 0.09999999, 0.7036024, 0.7099915, 0.19793506, 0.09999999,
                             0.7012213, 0.7117969, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999,
                             0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999, 0.09999999,
                             3.3793998, 2.9180048, 2.562093, 2.7047367, 1.0841713, 0.8094801, 1.0757841, 0.8439478,
                             0.44067168, 0.33905765, 7.5584087, 0.3428601, 0.3224279, 5.8961425, 1.3366703, 1.8276505,
                             1.8320743, 1.1618187, 1.1803689, 0.15215792, 0.10663375, 0.17174725, 0.09999999,
                             0.20094976, 0.10194187, 0.20406862, 0.10068466, 0.25226733, 0.09999999, 0.20357151,
                             0.09999999, 0.2427386, 0.11436407, 0.36003974, 0.23516357, 0.17657563, 0.47387388,
                             0.23171127, 0.09999999, 0.47834542, 0.09999999, 0.32406855, 0.18077546, 0.13010821,
                             0.2439245, 0.31692365, 0.09999999, 0.4783455, 0.09999999, 0.4783455, 0.47834542, 0.4783455,
                             0.47834542, 0.4783455, 0.47834542, 0.4783455, 0.47834542, 0.4783455, 0.47834542, 0.4783455,
                             0.47834542, 0.4783455, 0.47834542, 0.4783455, 0.47834542, 0.4783455, 0.47834542,
                             0.47834548, 0.4783454, 0.47834548, 0.4783454]}, env_name)
    debug = True
    change_color = True
    gold = (0.8 * 255, 0.6 * 255, 0.4 * 255)
