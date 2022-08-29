import os
import traceback
import numpy as np
import logging
import threading
import timeit
import time
import pprint
import torch
from torch import nn
from torch import multiprocessing as mp


from autosmote.rl.utils import Buffers, Environment, FileWriter, from_logits, get_batch, compute_baseline_loss, compute_entropy_loss, compute_policy_gradient_loss
from autosmote.rl.env import Env, GymWrapper
from autosmote.rl.models import CrossInstanceNet, InstanceSpecificNet, LowLevelNet

def train(flags, train_X, train_y, val_X, val_y, test_X, test_y, clf):
    plogger = FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )

    T = flags.instance_specific_unroll_length
    B = flags.batch_size

    torch.set_num_threads(1)

    env = Env(train_X, train_y, val_X, val_y, clf, flags.metric)
    env = GymWrapper(env, flags, neighbors=1)
    flags.neighbors = min(flags.num_max_neighbors, env.length)
    flags.considered_neighbors = flags.neighbors # reserved for tuning the neighbors. It is fixed now

    env = Env(train_X, train_y, val_X, val_y, clf, flags.metric)
    env = GymWrapper(env, flags)
    test_env = Env(train_X, train_y, test_X, test_y, clf, flags.metric)
    test_env = GymWrapper(test_env, flags)

    flags.num_cross_instance_actions = flags.undersample_ratio * flags.cross_instance_scale // flags.num_instance_specific_actions
    cross_instance_model = CrossInstanceNet(env.observation_space, flags.num_cross_instance_actions)
    cross_instance_model.share_memory()
    cross_instance_buffers = create_cross_instance_buffers(flags, env.observation_space, flags.num_cross_instance_actions)

    instance_specific_model = InstanceSpecificNet(env.observation_space, env.num_instance_specific_actions)
    instance_specific_model.share_memory()
    instance_specific_buffers = create_instance_specific_buffers(flags, env.observation_space, env.num_instance_specific_actions)

    low_level_model = LowLevelNet(env.observation_space, len(flags.ratio_map))
    low_level_model.share_memory()
    low_level_buffers = create_low_level_buffers(flags, env.observation_space)

    actor_processes = []
    ctx = mp.get_context("fork")
    cross_instance_free_queue = ctx.SimpleQueue()
    cross_instance_full_queue = ctx.SimpleQueue()
    instance_specific_free_queue = ctx.SimpleQueue()
    instance_specific_full_queue = ctx.SimpleQueue()
    low_level_free_queue = ctx.SimpleQueue()
    low_level_full_queue = ctx.SimpleQueue()
    data_queue = ctx.Queue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                cross_instance_free_queue,
                cross_instance_full_queue,
                instance_specific_free_queue,
                instance_specific_full_queue,
                low_level_free_queue,
                low_level_full_queue,
                data_queue,
                cross_instance_model,
                cross_instance_buffers,
                instance_specific_model,
                instance_specific_buffers,
                low_level_model,
                low_level_buffers,
                train_X,
                train_y,
                val_X,
                val_y,
                clf
            ),
        )
        actor.start()
        actor_processes.append(actor)

    cross_instance_learner_model = CrossInstanceNet(env.observation_space, flags.num_cross_instance_actions)
    cross_instance_learner_model = cross_instance_learner_model.to(flags.device)
    instance_specific_learner_model = InstanceSpecificNet(env.observation_space, env.num_instance_specific_actions)
    instance_specific_learner_model = instance_specific_learner_model.to(flags.device)
    low_level_learner_model = LowLevelNet(env.observation_space, len(flags.ratio_map), device=flags.device)
    low_level_learner_model = low_level_learner_model.to(flags.device)

    cross_instance_optimizer = torch.optim.Adam(
        cross_instance_learner_model.parameters(),
        lr=flags.learning_rate,
    )

    instance_specific_optimizer = torch.optim.Adam(
        instance_specific_learner_model.parameters(),
        lr=flags.learning_rate,
    )

    low_level_optimizer = torch.optim.Adam(
        low_level_learner_model.parameters(),
        lr=flags.learning_rate,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    logger = logging.getLogger("logfile")

    step, cross_instance_stats, stats, low_level_stats = 0, {}, {}, {}

    def cross_instance_batch_and_learn(i):
        """Thread target for the learning process."""
        nonlocal step, cross_instance_stats
        while step < flags.total_steps:
            indices = []

            while len(indices) < flags.batch_size:
                while len(indices) < flags.batch_size and not cross_instance_full_queue.empty():
                    indices.append(cross_instance_full_queue.get())
                time.sleep(1)
                if step >= flags.total_steps:
                    return
            batch = get_batch(
                flags,
                cross_instance_free_queue,
                cross_instance_full_queue,
                cross_instance_buffers,
                indices
            )
            cross_instance_stats = learn(
                flags, cross_instance_model, cross_instance_learner_model, batch, cross_instance_optimizer
            )

    def instance_specific_batch_and_learn(i):
        """Thread target for the learning process."""
        nonlocal step, stats
        while step < flags.total_steps:
            indices = []

            while len(indices) < flags.batch_size:
                while len(indices) < flags.batch_size and not instance_specific_full_queue.empty():
                    indices.append(instance_specific_full_queue.get())
                time.sleep(1)
                if step >= flags.total_steps:
                    return
            batch = get_batch(
                flags,
                instance_specific_free_queue,
                instance_specific_full_queue,
                instance_specific_buffers,
                indices
            )
            stats = learn(
                #flags, model, learner_model, batch, agent_state, optimizer, scheduler
                flags, instance_specific_model, instance_specific_learner_model, batch, instance_specific_optimizer
            )

    def low_level_batch_and_learn(i):
        """Thread target for the learning process."""
        nonlocal step, low_level_stats
        while step < flags.total_steps:
            indices = []

            while len(indices) < flags.batch_size:
                while len(indices) < flags.batch_size and not low_level_full_queue.empty():
                    indices.append(low_level_full_queue.get())
                time.sleep(1)
                if step >= flags.total_steps:
                    return
            batch = get_batch(
                flags,
                low_level_free_queue,
                low_level_full_queue,
                low_level_buffers,
                indices,
            )
            low_level_stats = learn(
                flags, low_level_model, low_level_learner_model, batch, low_level_optimizer
            )

    for m in range(flags.num_buffers):
        cross_instance_free_queue.put(m)
        instance_specific_free_queue.put(m)
        low_level_free_queue.put(m)

    # Cross instance thread
    cross_instance_thread = threading.Thread(
        target=cross_instance_batch_and_learn, name="pre-level-batch-and-learn-%d" % i, args=(i,)
    )
    cross_instance_thread.start()

    # Instance specific thread
    instance_specific_thread = threading.Thread(
        target=instance_specific_batch_and_learn, name="high-level-batch-and-learn-%d" % i, args=(i,)
    )
    instance_specific_thread.start()

    # Low-level thread
    low_level_thread = threading.Thread(
        target=low_level_batch_and_learn, name="low-level-batch-and-learn-%d" % i, args=(i,)
    )
    low_level_thread.start()

    best_val_score = 0
    test_score = 0
    def evaluate():
        nonlocal step, best_val_score, test_score
        while step < flags.total_steps:
            while data_queue.empty():
                time.sleep(1)
                if step >= flags.total_steps:
                    return

            val_score, samples = data_queue.get()
            step += 1
            if val_score > best_val_score:
                best_val_score = val_score
                test_score = test_env.get_reward(samples)
            print("Current val:", val_score, "best val:", best_val_score, "test:", test_score)

    eval_thread = threading.Thread(
        target=evaluate, name="evaluate-%d" % i
        )
    eval_thread.start()

    timer = timeit.default_timer
    try:
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_retur`ns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_rpren = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s\n%s\n%s",
                step,
                sps,
                total_loss,
                "",
                pprint.pformat(cross_instance_stats),
                pprint.pformat(stats),
                pprint.pformat(low_level_stats),
            )
    except KeyboardInterrupt:
        plogger.close()
        return  # Try joining actors then quit.
    else:
        cross_instance_thread.join()
        instance_specific_thread.join()
        low_level_thread.join()
        eval_thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            cross_instance_free_queue.put(None)
            instance_specific_free_queue.put(None)
            low_level_free_queue.put(None)
        for actor in actor_processes:
            actor.terminate()

    plogger.close()
    return test_score

def learn(
    flags,
    actor_model,
    model,
    batch,
    optimizer,
):
    """Performs a learning (optimization) step."""
    if isinstance(model, LowLevelNet):
        learner_outputs = model(batch, flags.considered_neighbors, flags.device)
        learner_outputs["policy_logits"] = learner_outputs["policy_logits"][:, :, :flags.considered_neighbors*len(flags.ratio_map)]
    else:
        learner_outputs = model(batch, flags.device)

    # Take final value function slice for bootstrapping.
    bootstrap_value = learner_outputs["baseline"][-1]

    # Move from obs[t] -> action[t] to action[t] -> obs[t].
    batch = {key: tensor[1:] for key, tensor in batch.items()}
    learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

    rewards = batch["reward"]
    discounts = (~batch["done"]).float() * flags.discounting

    vtrace_returns = from_logits(
        behavior_policy_logits=batch["policy_logits"],
        target_policy_logits=learner_outputs["policy_logits"],
        actions=batch["action"],
        discounts=discounts,
        rewards=rewards,
        values=learner_outputs["baseline"],
        bootstrap_value=bootstrap_value,
    )

    pg_loss = compute_policy_gradient_loss(
        learner_outputs["policy_logits"],
        batch["action"],
        vtrace_returns.pg_advantages,
    )
    baseline_loss = flags.baseline_cost * compute_baseline_loss(
        vtrace_returns.vs - learner_outputs["baseline"]
    )
    entropy_loss = flags.entropy_cost * compute_entropy_loss(
        learner_outputs["policy_logits"]
    )

    total_loss = pg_loss + baseline_loss + entropy_loss

    episode_returns = batch["episode_return"][batch["done"]]
    stats = {
        "mean_episode_return": torch.mean(episode_returns).item(),
        "total_loss": total_loss.item(),
        "pg_loss": pg_loss.item(),
        "baseline_loss": baseline_loss.item(),
        "entropy_loss": entropy_loss.item(),
    }

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
    optimizer.step()

    actor_model.load_state_dict(model.state_dict())
    return stats

def act(
    flags,
    actor_index: int,
    cross_instance_free_queue: mp.SimpleQueue,
    cross_instance_full_queue: mp.SimpleQueue,
    instance_specific_free_queue: mp.SimpleQueue,
    instance_specific_full_queue: mp.SimpleQueue,
    low_level_free_queue: mp.SimpleQueue,
    low_level_full_queue: mp.SimpleQueue,
    data_queue: mp.Queue,
    cross_instance_model,
    cross_instance_buffers,
    instance_specific_model,
    instance_specific_buffers,
    low_level_model,
    low_level_buffers,
    train_X,
    train_y,
    val_X,
    val_y,
    clf
):
    try:
        logging.info("Actor %i started.", actor_index)

        torch.set_num_threads(1)

        env = Env(train_X, train_y, val_X, val_y, clf, flags.metric)
        gym_env = GymWrapper(env, flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        torch.manual_seed(seed)
        np.random.seed(seed)
        gym_env.seed(seed)
        env = Environment(gym_env)
        env_output = env.initial()

        cross_instance_agent_output = cross_instance_model(env_output)
        instance_specific_agent_output = instance_specific_model(env_output)
        low_level_agent_output = low_level_model(env_output, flags.considered_neighbors)

        # Some preperation
        cross_instance_env_output_keys = list(env_output.keys())
        cross_instance_env_output_keys.remove("neighbor_features")

        instance_specific_env_output_keys = list(env_output.keys())
        instance_specific_env_output_keys.remove("neighbor_features")
        instance_specific_action = 0

        # tmp buffer to store historical pre level episodes
        tmp_cross_instance_buf = {}
        for key in cross_instance_env_output_keys:
            tmp_cross_instance_buf[key] = []
        for key in cross_instance_agent_output:
            tmp_cross_instance_buf[key] = []

        # Write the inital rollout (pre-level).
        for key in cross_instance_env_output_keys:
            tmp_cross_instance_buf[key].append(env_output[key])
        for key in cross_instance_agent_output:
            tmp_cross_instance_buf[key].append(cross_instance_agent_output[key])


        # tmp buffer to store historical low level episodes
        tmp_low_level_buf = {}
        for key in env_output:
            tmp_low_level_buf[key] = []
        for key in low_level_agent_output:
            tmp_low_level_buf[key] = []

        # tmp splitted low level buffer to track the reward
        tmp_splitted_low_level_buf = {}
        for key in env_output:
            tmp_splitted_low_level_buf[key] = []
        for key in low_level_agent_output:
            tmp_splitted_low_level_buf[key] = []

        # Write the inital rollout (low-level).
        for key in env_output:
            tmp_low_level_buf[key].append(env_output[key])
        for key in low_level_agent_output:
            tmp_low_level_buf[key].append(low_level_agent_output[key])

        act_done = False
        while not act_done:
            instance_specific_index = instance_specific_free_queue.get()
            if instance_specific_index is None:
                act_done = True
                break

            # Write old rollout end (high-level).
            for key in instance_specific_env_output_keys:
                instance_specific_buffers[key][instance_specific_index][0, ...] = env_output[key]
            for key in instance_specific_agent_output:
                instance_specific_buffers[key][instance_specific_index][0, ...] = instance_specific_agent_output[key]

            # Do new rollout.
            for t in range(flags.instance_specific_unroll_length):

                instance_specific_scale = cross_instance_agent_output["action"].item()
                with torch.no_grad():
                    instance_specific_agent_output = instance_specific_model(env_output)
                instance_specific_action = instance_specific_agent_output["action"].item()
                instance_specific_action *= instance_specific_scale
                #instance_specific_action = 50 # hard-code

                # Create a new entry in the tmp low-level buffer
                for key in tmp_splitted_low_level_buf:
                    tmp_splitted_low_level_buf[key].append([])
                to_next = False
                while not to_next:
                    if instance_specific_action > 0:
                        with torch.no_grad():
                            low_level_agent_output = low_level_model(env_output, flags.considered_neighbors)
                        low_level_action = low_level_agent_output["action"].item()
                        dst_index = low_level_action // len(flags.ratio_map)
                        ratio = flags.ratio_map[low_level_action % len(flags.ratio_map)] 
                    else:
                        dst_index = -1
                        ratio = -1
                    to_next = True if instance_specific_action <= 1 else False

                    low_level_action = {
                        "dst_index": dst_index,
                        "ratio": ratio,
                        "next": to_next
                    }
                    env_output, info = env.step(low_level_action)
                    for key in env_output:
                        tmp_splitted_low_level_buf[key][-1].append(env_output[key])
                    for key in low_level_agent_output:
                        tmp_splitted_low_level_buf[key][-1].append(low_level_agent_output[key])
                    instance_specific_action -= 1

                if env_output["done"]:

                    for key in cross_instance_env_output_keys:
                        tmp_cross_instance_buf[key].append(env_output[key])
                    for key in cross_instance_agent_output:
                        tmp_cross_instance_buf[key].append(cross_instance_agent_output[key])

                    while len(tmp_cross_instance_buf["reward"]) > flags.cross_instance_unroll_length+1:
                        cross_instance_index = cross_instance_free_queue.get()
                        if cross_instance_index is None:
                            act_done = True
                            break

                        for key in tmp_cross_instance_buf:
                            for pre_t in range(flags.cross_instance_unroll_length+1):
                                cross_instance_buffers[key][cross_instance_index][pre_t, ...] = tmp_cross_instance_buf[key][pre_t]
                            tmp_cross_instance_buf[key] = tmp_cross_instance_buf[key][flags.cross_instance_unroll_length:]
                        cross_instance_full_queue.put(cross_instance_index)

                    cross_instance_agent_output = cross_instance_model(env_output)

                    # Assign rewards to low-level episodes, and sample
                    idx = [i for i in range(len(tmp_splitted_low_level_buf["reward"]))]
                    sampled_idx = np.random.choice(idx, size=10)
                    for key in tmp_splitted_low_level_buf:
                        tmp_splitted_low_level_buf[key] = [tmp_splitted_low_level_buf[key][i] for i in sampled_idx]

                    # Assign rewards to low-level episodes, and sample
                    for episode_id in range(len(tmp_splitted_low_level_buf["reward"])):
                        if len(tmp_splitted_low_level_buf["reward"][episode_id]) > 0:
                            tmp_splitted_low_level_buf["reward"][episode_id][-1] = env_output["reward"]
                            tmp_splitted_low_level_buf["done"][episode_id][-1] = env_output["done"]

                    # Flatten and put them to tmp_low_level_buf
                    for key in tmp_low_level_buf:
                        for episode_data in tmp_splitted_low_level_buf[key]:
                            tmp_low_level_buf[key].extend(episode_data)
                        tmp_splitted_low_level_buf[key] = []

                    # Put the data into the buffer
                    while len(tmp_low_level_buf["reward"]) > flags.low_level_unroll_length+1:
                        low_level_index = low_level_free_queue.get()
                        if low_level_index is None:
                            act_done = True
                            break

                        for key in tmp_low_level_buf:
                            for low_t in range(flags.low_level_unroll_length+1):
                                low_level_buffers[key][low_level_index][low_t, ...] = tmp_low_level_buf[key][low_t]
                            tmp_low_level_buf[key] = tmp_low_level_buf[key][flags.low_level_unroll_length:]
                        low_level_full_queue.put(low_level_index)

                    data_queue.put((env_output["episode_return"].numpy().flatten()[0], info["samples"]))

                for key in instance_specific_env_output_keys:
                    instance_specific_buffers[key][instance_specific_index][t + 1, ...] = env_output[key]
                for key in instance_specific_agent_output:
                    instance_specific_buffers[key][instance_specific_index][t + 1, ...] = instance_specific_agent_output[key]

            instance_specific_full_queue.put(instance_specific_index)

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e

def create_cross_instance_buffers(flags, observation_space, num_actions) -> Buffers:
    T = flags.cross_instance_unroll_length
    specs = dict(
        src_features=dict(size=(T + 1, *observation_space["src_features"].shape), dtype=torch.float32),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

def create_instance_specific_buffers(flags, observation_space, num_actions) -> Buffers:
    T = flags.instance_specific_unroll_length
    specs = dict(
        src_features=dict(size=(T + 1, *observation_space["src_features"].shape), dtype=torch.float32),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

def create_low_level_buffers(flags, observation_space) -> Buffers:
    T = flags.low_level_unroll_length
    specs = dict(
        src_features=dict(size=(T + 1, *observation_space["src_features"].shape), dtype=torch.float32),
        neighbor_features=dict(size=(T + 1, *observation_space["neighbor_features"].shape), dtype=torch.float32),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, observation_space["neighbor_features"].shape[0]*len(flags.ratio_map)), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers
