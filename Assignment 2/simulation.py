import simpy
import numpy as np
import itertools
import pandas as pd

class MultiServer:
    '''Class for a M/M/n queue system server
    '''

    def __init__(self, env, n, labmda_rate, mu):
        '''Initializes the class using the following inputs:
        env - a SimPy environment
        n - number of servers in the system
        lambda_rate - the arrival rate
        mu - capacity per server
        '''
        
        self.env = env
        self.lambda_rate = labmda_rate
        self.mu = mu

        # Beta parameter for exponential distribution
        self.beta = 1 / labmda_rate

        # Create resource with capacity n
        self.processor = simpy.Resource(env, capacity=n)

        # Create list for processed jobs dictionaries
        self.jobs = pd.DataFrame(columns=["ID", "arr_time", "proc_time", "leave_time"])
    

    def setup_sim(self):
        '''Set up simulation
        '''
        
        job_count = itertools.count()

        # for _ in range(4):
        #     job_index = next(job_count)
        #     self.env.process(self.job_request(job_index))

        # Simulate queue
        while True:
            inter_t = np.random.exponential(self.lambda_rate)

            # New job arrival at queue
            job_index = next(job_count)
            yield self.env.timeout(inter_t)
            self.env.process(self.job_request(job_index))
    

    def job_request(self, request_id):
        '''Creates job request at server
        '''

        print(f"Job {request_id} arrives at queue at time {self.env.now}")
        # self.jobs.append(dict("{:04d}".format(request_id)))
        self.jobs.iloc[request_id]["ID"] = "{:04d}".format(request_id)
        self.jobs.iloc[request_id]["arr_time"] = self.env.now

        with self.processor.request() as request:
            yield request

            print(f"Job {request_id} is starting to be processed at time {self.env.now}")
            
            self.jobs.iloc[request_id]["proc_time"] = self.env.now
            yield self.env.process(self.job_process())
            print(f"Job {request_id} is finished at {self.env.now}")
            self.jobs.iloc[request_id]["leave_time"] = self.env.now
    

    def job_process(self):
        '''Processes job for a random time
        from an exponential distribution
        with parameter mu
        '''

        process_time = np.random.exponential(self.mu)
        yield self.env.timeout(process_time)


env = simpy.Environment()
server = MultiServer(env, 1, 0.5, 0.8)

env.process(server.setup_sim())
env.run(until=100)

print(server.jobs)