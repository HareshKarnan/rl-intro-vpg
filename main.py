from vpg import VPG
import gym

env = gym.make('InvertedPendulum-v2')

model = VPG(env=env)

model.train(epochs=50)
