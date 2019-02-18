from vpg.py import vpg
import gym

env = gym.make('InvertedPendulum-v2')

model = vpg()