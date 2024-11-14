from env import InvestESG
from pettingzoo.test import parallel_api_test

if __name__ == '__main__':
    env = InvestESG()
    parallel_api_test(env, num_cycles=100)