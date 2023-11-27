import simpy
import numpy as np
import itertools
import pandas as pd

class MultiServer:
    '''Class for a M/M/n queue system server
    '''

    def __init__(self, env, n, lambda_rate, mu, deterministic=False, hyper_prob=None, mu2=None):
        '''Initializes the class using the following inputs:
        env - a SimPy environment
        n - number of servers in the system
        lambda_rate - the arrival rate
        mu - capacity per server
        '''
        
        self.deterministic = deterministic
        self.env = env
        self.lambda_rate = lambda_rate
        self.mu = mu
        self.rho = self.lambda_rate / (n * self.mu)
        print(f"Created system with {n} servers and load {self.rho}.")

        self.hyper_prob = hyper_prob

        # Beta parameter for exponential distributions
        self.beta_arrival = 1 / lambda_rate
        self.beta_service = 1 / mu
        if mu2:
            self.beta_service2 = 1 / mu2

        # Create resource with capacity n
        self.processor = simpy.Resource(env, capacity=n)

        # Create list for processed jobs dictionaries
        self.jobs = pd.DataFrame(columns=["jobID", "arr_time", "proc_time", "leave_time", "wait_delta"])
    

    def setup_sim(self, print_progress=False):
        '''Set up simulation. If print_progress is True,
        the simulation events are printed on the go
        '''

        self.print = print_progress
        
        job_count = itertools.count()

        # Simulate queue
        while True:

            # Determine inter-arrival time
            inter_t = np.random.exponential(self.beta_arrival)

            # New job arrival at queue
            job_index = next(job_count)
            yield self.env.timeout(inter_t)
            self.env.process(self.job_request(job_index))
    

    def job_request(self, request_id):
        '''Creates job request at server
        '''

        if self.print: print(f"Job {request_id} arrives at queue at time {self.env.now}")
        self.jobs.loc[request_id] = [None, None, None, None, None]
        self.jobs["jobID"].loc[request_id] = "{:04d}".format(request_id)
        self.jobs["arr_time"].loc[request_id] = self.env.now

        with self.processor.request() as request:
            yield request
            
            if self.print: print(f"Job {request_id} is starting to be processed at time {self.env.now}")
            
            self.jobs["proc_time"].loc[request_id] = self.env.now
            self.jobs["wait_delta"].loc[request_id] = self.env.now - self.jobs["arr_time"].loc[request_id]
            yield self.env.process(self.job_process())
            if self.print: print(f"Job {request_id} is finished at {self.env.now}")
            self.jobs["leave_time"].loc[request_id]= self.env.now
    

    def job_process(self):
        '''Processes job for a random time
        from an exponential distribution
        with parameter mu
        '''

        # Determine service time
        if self.deterministic: 
            process_time = self.mu
        else:
            if self.hyper_prob:
                choose_distribution = np.random.uniform(0, 1)
                if choose_distribution < self.hyper_prob:
                    process_time = np.random.exponential(self.beta_service)
                else:
                    process_time = np.random.exponential(self.beta_service2)
            else:
                process_time = np.random.exponential(self.beta_service)
        
        yield self.env.timeout(process_time)

