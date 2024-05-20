from Gym.Rewards.Approaching import CircularApproachingReward
from Gym.Rewards.Hanging import Hanging

class RewardSystem():
    def __init__(self, phase: str) -> None:
        self.phase = phase
        print("Phase: ", phase)
        self.approaching_reward = CircularApproachingReward()
        self.hanging_reward = Hanging()
    
    def calculate(self, state, has_collided, dist_tether_branch, dist_drone_branch,
                   num_wraps):
        
        # Approaching reward is between -1 and 0
        approaching_reward, _, _ = self.approaching_reward.reward_fun(state, has_collided, dist_tether_branch,
                                                                dist_drone_branch, num_wraps)
        assert approaching_reward <= 0 and approaching_reward >= -1, f"was {approaching_reward}"

        # Wrapping reward is between -1 and 0
        wrapping_reward = 1.0 * min(num_wraps, 1.0) - 1.0
        assert wrapping_reward <= 0 and wrapping_reward >= -1, f"was {wrapping_reward}"
        
        # Hanging reward is between 0 and 1
        hanging_reward, hanging_done, _ = self.hanging_reward.reward_fun(state, has_collided, dist_tether_branch,
                                                        dist_drone_branch, num_wraps)
        hanging_reward = hanging_reward - 1 # Now its between -1 and 0
        assert hanging_reward <= 0 and hanging_reward >= -1, f"was {hanging_reward}"
        
        total_reward = 0
        match self.phase:
            case "all":
                if num_wraps > 0.9:
                    total_reward = 0 + wrapping_reward + hanging_reward  # Between -2 and 0
                    done = hanging_done
                else:
                    total_reward = approaching_reward + wrapping_reward - 1  # Between -3 and -1
                    done = False

            case "approaching":
                total_reward = approaching_reward + wrapping_reward - 1 # Between -3 and -1
                done = bool(num_wraps > 0.9)
            case _:
                raise ValueError("Unknown phase type")
        
        return total_reward, done

