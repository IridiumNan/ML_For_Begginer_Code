from utils import *
import pickle

with open("./Q.pkl", "rb") as f:
    Qbest = pickle.load(f)

env = gym.make("CartPole-v1", render_mode="human")
policy = fetch_action_based_Q(Qbest, actions)
play_with_Q(policy, env)
