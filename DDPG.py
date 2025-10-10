import torch
import random

from environment import Wrapper
from memory_replay import MemoryReplay
from networks import Actor, Cretic
from tool_function import Tool


env = Wrapper()
memory = MemoryReplay(20000)

actor = Actor(3, 128)
actor_delay = Actor(3, 128)
actor_delay.load_state_dict(actor.state_dict())
Tool.requires_grad(actor_delay, False)

cretic = Cretic(4, 128)
cretic_target = Cretic(4, 128)
cretic_target.load_state_dict(cretic.state_dict())
Tool.requires_grad(cretic_target, False)

actorLR = 5e-4
creticLR = 5e-3
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=actorLR)
optimizer_cretic = torch.optim.Adam(cretic.parameters(), lr=creticLR)


def play():
    data = []
    reward_sum = 0

    state = env.reset()
    over = False
    while not over:
        action = actor(torch.FloatTensor(state).reshape(1, 3)).item()
        # add noise
        action += random.normalvariate(mu=0, sigma=0.2)

        next_state, reward, over = env.step(action)

        data.append((state, action, reward, next_state, over))
        reward_sum += reward

        state = next_state

    return data, reward_sum


def train_actor(state):
    Tool.requires_grad(actor, True)
    Tool.requires_grad(cretic, False)

    action = actor(state)

    input = torch.cat([state, action], dim=1)
    loss = -cretic_target(input).mean()

    loss.backward()
    optimizer_actor.step()
    optimizer_actor.zero_grad()

    return loss.item()


def train_cretic(state, action, reward, next_state, over):
    Tool.requires_grad(actor, False)
    Tool.requires_grad(cretic, True)

    input = torch.cat([state, action], dim=1)
    value = cretic(input)

    with torch.no_grad():
        next_action = actor_delay(next_state)
        input = torch.cat([next_state, next_action], dim=1)
        target = cretic_target(input)
    target = target * 0.99 * (1 - over) + reward

    loss = torch.nn.functional.mse_loss(value, target)

    loss.backward()
    optimizer_cretic.step()
    optimizer_cretic.zero_grad()

    return loss.item(), value.mean().item()


# make model training mode
actor.train()
cretic.train()


# train loop
for epoch in range(200):
    old_len = len(memory)
    while len(memory) - old_len < 200 and len(memory) != 20000:
        memory.push(play()[0])

    for i in range(200):
        state, action, reward, next_state, over = memory.sample(128)

        train_actor(state)
        _, value = train_cretic(state, action, reward, next_state, over)

        Tool.soft_update(actor, actor_delay)
        Tool.soft_update(cretic, cretic_target)

    if epoch % 20 == 0:
        test_result = sum([play()[1] for _ in range(20)]) / 20
        print(epoch, len(memory), value, test_result)

env.close()

# check the final model
over = False
env = Wrapper("human")
state = env.reset()
while not over:
    action = actor(torch.FloatTensor(state).reshape(1, 3)).item()
    state, _, over = env.step(action)

env.close()
