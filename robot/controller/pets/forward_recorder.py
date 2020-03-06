# forward recorder that can r the
from .model import EnBNNAgent
from robot.envs.sapien.exp.utils import RLRecorder

class Recoder(RLRecorder):
    def step(self, agent: EnBNNAgent, reward, episode_timesteps, train_output=None, **kwargs):
        env = self.get_env()
        def gen_video():
            # write the video at the neighborhood of the optimal [random] policy
            obs = env.reset()

        kwargs['video_model'] = gen_video()

        super(Recoder, self).step(agent, reward, episode_timesteps, train_output, **kwargs)

