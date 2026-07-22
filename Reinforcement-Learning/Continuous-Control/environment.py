import gymnasium as gym


class Wrapper(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self):
        state, _ = self.env.reset()
        self.step_n = 0
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step([action * 2])
        over = terminated or truncated

        self.step_n += 1
        if self.step_n >= 200 and self.render_mode is None:
            over = True
        return state, reward, over

    def render(self):
        if self.env.render_mode is not None:
            return self.env.render()

    def close(self):
        self.env.close()
