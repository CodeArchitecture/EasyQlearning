from env import Find_Goal_Environment
from agent import QAgent
import matplotlib.pyplot as plt

teminated = False
rewards = []    # total rewards in 1 episode
episodes = 20
L_rewards = []
env = Find_Goal_Environment()
agent = QAgent()

for i in range(episodes):
    env.reset()

    while not teminated:
        state = env.get_state()
        action = agent.select_action(state)
        reward, teminated = env.step(action)
        next_state = env.get_state()
        rewards.append(reward)
        print('state:{}, action:{}, reward:{}, next_state:{}'.format(state, action, reward, next_state))
        env.render()
        agent.train(state, action, next_state, reward)

    print(f'episode: {i},  total rewards: {sum(rewards)}')
    L_rewards.append(sum(rewards))
    teminated = False
    rewards = []

# show rewards curve
plt.plot(L_rewards)
plt.xlabel('episode')
plt.ylabel('total rewards')   # culmulative rewards per episode
plt.xlim(0, len(L_rewards))
plt.title('rewards vs episode')
plt.savefig('episode_rewards.jpg')
plt.show()

print(agent.Q)
