import logging
import secrets

import gymnasium as gym

import sinergym
from cpu_generator import CPULoadGenerator
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (
    CSVLogger,
    DatetimeWrapper,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)

n_ep = 2


terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(name="MAIN", level=logging.INFO)


def main(max_steps: int | None = None) -> None:
    env = gym.make(
        "Eplus-smalldatacenter-mixed-continuous-stochastic-v1",
        building_config={"timesteps_per_hour": 4},
        context={
            "cpu_loading_fraction": (
                "Schedule:Compact",
                "Schedule Value",
                "Data Center CPU Loading Schedule",
            )
        },
        initial_context=[0.0],
    )
    env = DatetimeWrapper(env)
    env = NormalizeAction(env)
    env = NormalizeObservation(env)
    env = LoggerWrapper(env)
    env = CSVLogger(env)

    generator = CPULoadGenerator(
        load_names=("cpu_loading_fraction",),
        step_seconds=15 * 60,
        sample_type="correlation-week",
        cached=False,
    )

    try:
        for _ in range(n_ep):
            # New random seed for the environment, actions, and generator.
            episode_seed = secrets.randbits(32)
            logger.info(f"Episode seed: {episode_seed}")

            # Align the initial weekday and time with Sinergym's RunPeriod.
            generator.reset_from_sinergym(
                env,
                seed=episode_seed,
            )
            initial_load = generator.sample_at(
                generator.episode_start_time
            )["cpu_loading_fraction"]

            options = dict(env.unwrapped.default_options)
            options["initial_context"] = [initial_load]
            env.action_space.seed(episode_seed)
            obs, info = env.reset(
                seed=episode_seed,
                options=options,
            )
            # Sinergym may expose a non-zero initial time_elapsed.
            elapsed_origin = float(info["time_elapsed(hours)"])
            truncated = terminated = False
            steps = 0

            while not (terminated or truncated):
                # At time t, prepare the context consumed by the step
                # ending at t1. The generator owns the fixed 900-second
                # sampling grid; Sinergym time is checked afterwards.
                loads = generator.next_step()
                env.unwrapped.update_context(loads)

                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1

                observed_elapsed_seconds = round(
                    (
                        float(info["time_elapsed(hours)"])
                        - elapsed_origin
                    )
                    * 3600
                )
                expected_elapsed_seconds = (
                    steps * generator.step_seconds
                )
                if observed_elapsed_seconds != expected_elapsed_seconds:
                    logger.warning(
                        "Sinergym time differs from the fixed sampling "
                        "grid at step %s: expected %s s, observed %s s.",
                        steps,
                        expected_elapsed_seconds,
                        observed_elapsed_seconds,
                    )

                if max_steps is not None and steps >= max_steps:
                    break

            logger.info(
                f"Episode {env.get_wrapper_attr('episode')} stopped "
                f"after {steps} steps."
            )

        logger.info(
            "Final observation dictionary:"
        )
        logger.info(env.get_obs_dict(obs))
    finally:
        env.close()


if __name__ == "__main__":
    main()
