"""Train PPO policy using implementation from RSL_RL."""
from absl import app
from absl import flags
import os
import re
from datetime import datetime

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from ml_collections.config_flags import config_flags
from rsl_rl.runners import OnPolicyRunner

from src.envs import env_wrappers

config_flags.DEFINE_config_file(
    "config", "src/agents/ppo/configs/pronk.py",
    "experiment configuration.")
flags.DEFINE_integer("num_envs", 8192, "number of parallel environments.")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")
flags.DEFINE_string("logdir", "logs", "logdir.")
flags.DEFINE_string("load_checkpoint", None, "checkpoint to load.")
flags.DEFINE_string("gpu_id", "0", "GPU device ID to use (e.g., '0', '1', or '0,1').")
FLAGS = flags.FLAGS


def main(argv):
    del argv  # unused
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
    device = "cuda" if FLAGS.use_gpu else "cpu"

    config = FLAGS.config

    if FLAGS.load_checkpoint:
        #logdir = os.path.dirname(os.path.dirname(FLAGS.load_checkpoint))
        logdir = os.path.dirname(FLAGS.load_checkpoint)
        print(f"Resuming in existing log directory: {logdir}")
    else:
        logdir = os.path.join(FLAGS.logdir, config.training.runner.experiment_name,
                              datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        print(f"Starting new run in: {logdir}")

    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    # Only write config if it's a new run to avoid overwriting original settings
    config_path = os.path.join(logdir, "config.yaml")
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config.to_yaml())

    env = config.env_class(num_envs=FLAGS.num_envs,
                           device=device,
                           config=config.environment,
                           show_gui=FLAGS.show_gui)
    env = env_wrappers.RangeNormalize(env)

    runner = OnPolicyRunner(env, config.training, logdir, device=device)
    
    if FLAGS.load_checkpoint:
        runner.load(FLAGS.load_checkpoint)
        
        if runner.current_learning_iteration == 0:
            match = re.search(r'model_(\d+).pt', FLAGS.load_checkpoint)
            if match:
                checkpoint_iter = int(match.group(1))
                runner.current_learning_iteration = checkpoint_iter
                print(f"Fixed iteration counter to: {checkpoint_iter} (parsed from filename)")

    # runner.learn(num_learning_iterations=config.training.runner.max_iterations,
    #              init_at_random_ep_len=True)
    runner.learn(num_learning_iterations=max(0, config.training.runner.max_iterations - runner.current_learning_iteration), init_at_random_ep_len=True)


if __name__ == "__main__":
    app.run(main)
