import os
import shutil
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch

from src.deep_learning.dm_control.utils import define_env, define_agent, train_loop, evaluate

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training loop."""
    log.info(OmegaConf.to_yaml(cfg))

    # Define Environment
    env = define_env(cfg)

    # Define Agent
    agent = define_agent(cfg, env)

    # Define Optimizer
    optimizer = instantiate(cfg.optimizer, agent.parameters())

    # Define logging
    work_dir = os.getcwd()
    log.info(f'workspace: {work_dir}')
    if cfg.train.save_video:
        video_dir = os.path.join(work_dir, 'video')
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = None

    # Potentially load a pre-trained model
    if cfg.train.pretrained != '':
        parts = cfg.train.pretrained.split(':')
        cfg_path = parts[0]
        chkpt_path = parts[1]

        # Load the configuration file used for pretraining.
        pretrain_cfg = OmegaConf.load(os.path.join(cfg_path, '.hydra', 'config.yaml'))

        # Define the environment with the pretraining config
        env = define_env(pretrain_cfg)

        # Define a dummy agent.
        dummy_agent = define_agent(pretrain_cfg, env)

        # Load the checkpoint
        state_dict = torch.load(chkpt_path)

        # Load weights into the dummy agent.
        dummy_agent.load_state_dict(state_dict['agent'])

        # Copy weights into the real agent.
        agent.actor.load_state_dict(dummy_agent.actor.state_dict())

        # Print confirmation that weights were loaded successfully
        log.info("Loaded weights from pretrained model")

    # Train the agent
    train_loop(
        env=env,
        agent=agent,
        optimizer=optimizer,
        cfg=cfg.train,
        log=log,
        work_dir=work_dir,
        video_dir=video_dir,
    )

    # Evaluate the agent
    if cfg.eval.evaluate:
        avg_reward = evaluate(
            env=env,
            agent=agent,
            num_episodes=cfg.eval.episodes,
        )
        log.info(f'Average reward: {avg_reward}')

    # Save the agent (training loop also saves, but this is a final save)
    if cfg.train.save_model:
        model_dir = os.path.join(work_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        agent.save(os.path.join(model_dir, 'agent.pt'))

if __name__ == "__main__":
    main()
