from typing import Any, Dict, SupportsFloat, Tuple
from gymnasium import Env
from stable_baselines3.common.monitor import Monitor
from gymnasium.core import ActType, ObsType
import time

class CustomMonitor(Monitor):
    def __init__(self,
                 env: Env,
                 filename: str | None = None,
                 allow_early_resets: bool = True,
                 reset_keywords: Tuple[str, ...] = ...,
                 info_keywords: Tuple[str, ...] = ...,
                 override_existing: bool = True):
        super().__init__(env,
                         filename,
                         allow_early_resets,
                         reset_keywords,
                         info_keywords,
                         override_existing)
    
    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info