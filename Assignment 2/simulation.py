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
    

    def get_process_time(self):
        '''Determines service time depending on current strategy
        '''
        if self.deterministic:
            # Constant process time 
            process_time = self.mu
        else:
            if self.hyper_prob:
                # Draw from hyper-exponential distribution
                choose_distribution = np.random.uniform(0, 1)
                if choose_distribution < self.hyper_prob:
                    process_time = np.random.exponential(self.beta_service)
                else:
                    process_time = np.random.exponential(self.beta_service2)
            else:
                # Draw from exponential distribution
                process_time = np.random.exponential(self.beta_service)
        return process_time


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

            # New job arrival at queue after time inter_t
            job_index = next(job_count)
            yield self.env.timeout(inter_t)
            self.env.process(self.job_request(job_index))
    

    def job_request(self, request_id):
        '''Creates job request at server
        '''

        # Register arrival
        if self.print: print(f"Job {request_id} arrives at queue at time {self.env.now}")
        self.jobs.loc[request_id] = [None, None, None, None, None]
        self.jobs["jobID"].loc[request_id] = "{:04d}".format(request_id)
        self.jobs["arr_time"].loc[request_id] = self.env.now

        # Determine service time
        process_time = self.get_process_time()

        # Submit request
        with self.processor.request() as request:
            yield request
            
            if self.print: print(f"Job {request_id} is starting to be processed at time {self.env.now}")
            
            self.jobs["proc_time"].loc[request_id] = self.env.now
            self.jobs["wait_delta"].loc[request_id] = self.env.now - self.jobs["arr_time"].loc[request_id]
            yield self.env.process(self.job_process(process_time))
            if self.print: print(f"Job {request_id} is finished at {self.env.now}")
            self.jobs["leave_time"].loc[request_id]= self.env.now
    

    def job_process(self, process_time):
        '''Processes job for a duration equal to process_time
        '''

        yield self.env.timeout(process_time)


class MultiServerPriority(MultiServer):
    '''Class for a M/M/n queue system server using job priorities
    '''

    def __init__(self, env, n, lambda_rate, mu, deterministic=False, hyper_prob=None, mu2=None):
        '''Extends initialisation function of super-class with a priority-resourse server
        '''
        super(MultiServerPriority, self).__init__(env, n, lambda_rate, mu, deterministic, hyper_prob, mu2)

        self.processor = simpy.PriorityResource(env, capacity=n)


    def job_request(self, request_id):
        '''Creates job request at server
        '''

        # Register arrival
        if self.print: print(f"Job {request_id} arrives at queue at time {self.env.now}")
        self.jobs.loc[request_id] = [None, None, None, None, None]
        self.jobs["jobID"].loc[request_id] = "{:04d}".format(request_id)
        self.jobs["arr_time"].loc[request_id] = self.env.now

        # Determine service time
        process_time = self.get_process_time()

        # Submit request
        with self.processor.request(priority=process_time) as request:
            yield request
            
            if self.print: print(f"Job {request_id} is starting to be processed at time {self.env.now}")
            
            self.jobs["proc_time"].loc[request_id] = self.env.now
            self.jobs["wait_delta"].loc[request_id] = self.env.now - self.jobs["arr_time"].loc[request_id]
            yield self.env.process(self.job_process(process_time))
            if self.print: print(f"Job {request_id} is finished at {self.env.now}")
            self.jobs["leave_time"].loc[request_id]= self.env.now


def run_simulation(queue_type, n, sim_ID, runtime=100, **queue_params):
    '''Creates a queuing system model or input type
    and runs a simulation on it for predefined time
    '''

    env = simpy.Environment()
    
    match queue_type:
        case 'M/M/n': server = MultiServer(env, n=n, lambda_rate=queue_params['lambda_rate']*n, mu=queue_params['mu'])
        case 'M/M/n-Priority': server = MultiServerPriority(env, n=n, lambda_rate=queue_params['lambda_rate']*n, mu=queue_params['mu'])
        case 'M/D/n': server = MultiServer(env, n=n, lambda_rate=queue_params['lambda_rate']*n, mu=queue_params['mu'], deterministic=True)
        case 'M/H/n': server = MultiServer(env, n=n, lambda_rate=queue_params['lambda_rate']*n, mu=queue_params['mu'], hyper_prob=queue_params['hyper_prob'], mu2=queue_params['mu2'])
        case _: print("Model type error.")

    env.process(server.setup_sim(print_progress=False))
    env.run(until=runtime)
    run_results = server.jobs
    run_results.insert(0, 'simID', np.full(server.jobs.shape[0], sim_ID))
    run_results.insert(0, 'rho', np.full(server.jobs.shape[0], queue_params['lambda_rate']/queue_params['mu']))
    run_results.insert(0, 'n', np.full(server.jobs.shape[0], n))

    return run_results

# env = simpy.Environment()
# server = MultiServer(env, 2, 0.8, 1, True)
# env.process(server.setup_sim(True))
# env.run(until=50)
