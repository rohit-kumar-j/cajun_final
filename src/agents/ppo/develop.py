"""Development entry point for testing framework changes.

Clone of src/agents/ppo/train.py — use this for testing migrations.
train.py is never modified.

Usage:
    python -m src.agents.ppo.develop \
        --config=src/agents/ppo/configs/g2_rotary.py \
        --logdir=logs_dev/ \
        --num_envs=64
"""
from absl import app
from absl import flags
from datetime import datetime
import os
import sys

from loguru import logger
from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from ml_collections.config_flags import config_flags
from rsl_rl.runners import OnPolicyRunner
from src.envs import env_wrappers

config_flags.DEFINE_config_file(
    "config",
    "src/agents/ppo/configs/g2_rotary.py",
    "experiment configuration.",
)
flags.DEFINE_integer("num_envs",        64,         "number of parallel environments.")
flags.DEFINE_bool(   "use_gpu",         True,       "whether to use GPU.")
flags.DEFINE_bool(   "show_gui",        False,      "whether to show GUI.")
flags.DEFINE_string( "logdir",          "logs_dev", "logdir (separate from production).")
flags.DEFINE_string( "load_checkpoint", None,       "checkpoint to load.")
FLAGS = flags.FLAGS


def _log_config_summary(config) -> None:
    env = config.environment
    logger.info("=" * 60)
    logger.info("  develop.py — CONFIG SUMMARY")
    logger.info("=" * 60)

    # Gait
    try:
        logger.info(f"[gait] name               : {env.gait.gait_name}")
        logger.info(f"[gait] stepping_frequency : {env.gait.stepping_frequency} Hz")
    except AttributeError as exc:
        logger.warning(f"[gait] could not log: {exc}")

    # Robot (Step 1 — should be present)
    try:
        r = env.robot
        logger.info(f"[robot] urdf_path         : {r.urdf_path}")
        logger.info(f"[robot] init_height        : {r.init_height} m")
        logger.info(f"[robot] torque_delay_steps : {r.motor_torque_delay_steps}")
        logger.info(f"[robot] realistic motors   : {r.enable_realistic_motor_behaviour}")
        logger.info(f"[robot] l_up/l_low/l_hip   : {r.l_up}/{r.l_low}/{r.l_hip} m")
        logger.info(f"[robot] motors             : {len(r.motors)} joints defined")
        # Report any auto-filled warnings
        if hasattr(r, "warnings"):
            missing = [k for k, v in r.warnings.items() if not v]
            if missing:
                logger.warning(f"[robot] AUTO-FILLED values: {missing}")
            else:
                logger.success("[robot] All config.robot values explicitly set")
    except AttributeError:
        logger.warning(
            "[robot] config.robot section not found — "
            "robot will use legacy go1.py. Step 1 migration not applied."
        )

    # QP (Step 3 — may not exist yet)
    try:
        logger.info(f"[qp] weight_ddq diag : {env.qp.weight_ddq.diagonal().tolist()}")
        logger.info(f"[qp] stance_kp/kd    : {env.qp.stance_kp} / {env.qp.stance_kd}")
    except AttributeError:
        logger.warning("[qp] config.qp not found — using legacy flat keys (Step 3 pending)")

    # Env
    try:
        logger.info(f"[env] env_dt          : {env.env.env_dt} s")
        logger.info(f"[env] episode_length_s: {env.env.episode_length_s} s")
        logger.info(f"[env] velocity lb/ub  : {env.env.velocity_lb} / {env.env.velocity_ub} m/s")
    except AttributeError as exc:
        logger.warning(f"[env] could not log: {exc}")

    # Training
    try:
        tr = config.training.runner
        logger.info(f"[train] experiment     : {tr.experiment_name}")
        logger.info(f"[train] max_iterations : {tr.max_iterations}")
        logger.info(f"[train] learning_rate  : {config.training.algorithm.learning_rate}")
    except AttributeError as exc:
        logger.warning(f"[train] could not log: {exc}")

    logger.info("=" * 60)


def main(argv):
    del argv
    device = "cuda" if FLAGS.use_gpu else "cpu"
    config = FLAGS.config

    logdir = os.path.join(
        FLAGS.logdir,
        config.training.runner.experiment_name,
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    os.makedirs(logdir, exist_ok=True)

    # Log to file as well
    logger.add(
        os.path.join(logdir, "develop.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        rotation="50 MB",
    )

    logger.info(f"[develop] logdir: {logdir}")
    _log_config_summary(config)

    with open(os.path.join(logdir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(config.to_yaml())
    logger.info("[develop] config saved")

    logger.info(
        f"[develop] creating {config.env_class.__name__} "
        f"x{FLAGS.num_envs} on {device}"
    )
    try:
        env = config.env_class(
            num_envs = FLAGS.num_envs,
            device   = device,
            config   = config.environment,
            show_gui = FLAGS.show_gui,
        )
    except Exception as exc:
        logger.critical(f"[develop] env creation failed: {type(exc).__name__}: {exc}")
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)

    env = env_wrappers.RangeNormalize(env)
    logger.info(f"[develop] obs shape: {env.observation_space[0].shape}")
    logger.info(f"[develop] act shape: {env.action_space[0].shape}")

    runner = OnPolicyRunner(env, config.training, logdir, device=device)
    if FLAGS.load_checkpoint:
        logger.info(f"[develop] loading checkpoint: {FLAGS.load_checkpoint}")
        runner.load(FLAGS.load_checkpoint)

    logger.info(f"[develop] training {config.training.runner.max_iterations} iterations")
    runner.learn(
        num_learning_iterations = config.training.runner.max_iterations,
        init_at_random_ep_len   = True,
    )
    logger.info("[develop] done")


if __name__ == "__main__":
    app.run(main)
