import simpy
from simulation import *

env = simpy.Environment()
server = MultiServer(env, 1, 0.5, 0.8)

env.process(server.setup_sim())
env.run(until=100)

print(server.jobs)