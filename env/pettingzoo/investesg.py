from pettingzoo import ParallelEnv
from gym.spaces import Discrete, MultiDiscrete, Box, Dict, MultiBinary
import functools

import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns
import itertools

class Company:
    def __init__(self, capital=6, climate_risk_exposure = 0.05, beta = 0.1667, greenwash_esg_coef = 2):
        self.initial_capital = capital                      # initial capital, in trillion USD
        self.capital = capital                              # current capital, in trillion USD
        self.beta = beta                                    # Beta risk factor against market performance

        self.initial_resilience \
            = climate_risk_exposure                         # initial climate risk exposure
        self.resilience \
            = climate_risk_exposure                         # capital loss ratio when a climate event occurs
        
        self.resilience_incr_rate = 3                 # increase rate of climate resilience
        self.cumu_mitigation_amount = 0    # cumulative amount invested in emissions mitigation, in trillion USD
        self.cumu_greenwash_amount = 0      # cumulative amount invested in greenwashing, in trillion USD
        self.cumu_resilience_amount = 0                   # cumulative amount invested in resilience, in trillion USD

        self.margin = 0                                     # single period profit margin
        self.capital_gain = 0                               # single period capital gain
        
        self.mitigation_pc = 0            # single period investment in emissions mitigation, in percentage of total capital
        self.greenwash_pc = 0                             # single period investment in greenwashing, in percentage of total capital
        self.resilience_pc = 0                      # single period investment in resilience, in percentage of total capital
        
        self.mitigation_amount = 0        # amount of true emissions mitigation investment, in trillion USD
        self.greenwash_amount = 0                # amount of greenwashing investment, in trillion USD
        self.resilience_amount = 0               # amount of resilience investment, in trillion USD
        self.esg_score = 0                                  # signal to be broadcasted to investors: emissions mitigation investment / total capital,
                                                            # adjusted for greenwashing
        self.bankrupt = False

        self.greenwash_esg_coef = greenwash_esg_coef       # coefficient of greenwashing_pc on ESG score

    def receive_investment(self, amount):
        """Receive investment from investors."""
        self.capital += amount

    def lose_investment(self, amount):
        """Lose investment due to climate event."""
        self.capital -= amount
    
    def make_esg_decision(self):
        """Make a decision on how to allocate capital."""
        ### update capital and cumulative investment
        # update investment amount for single period
        self.mitigation_amount = self.mitigation_pc*self.capital
        self.greenwash_amount = self.greenwash_pc*self.capital
        self.resilience_amount = self.resilience_pc*self.capital
        # update cumulative investment
        self.cumu_mitigation_amount += self.mitigation_amount
        self.cumu_greenwash_amount += self.greenwash_amount
        self.cumu_resilience_amount += self.resilience_amount
        ### update resilience
        self.resilience = self.initial_resilience \
            * np.exp(-self.resilience_incr_rate * (self.cumu_resilience_amount/self.capital))
        ### update esg score
        self.esg_score = self.mitigation_pc + self.greenwash_pc*self.greenwash_esg_coef


    def update_capital(self, environment):
        """Update the capital based on esg decision, market performance and climate event."""
        # add a random disturbance to market performance
        company_performance = np.random.normal(loc=environment.market_performance, scale=self.beta) 
        # ranges from 0.5 to 1.5 of market performance baseline most of time
        new_capital = self.capital * (1-self.mitigation_pc-self.resilience_pc-self.greenwash_pc) * company_performance
        if environment.climate_event_occurrence > 0:
            new_capital *= (1 - self.resilience*environment.climate_event_occurrence)

        # calculate margin and capital gain
        self.capital_gain = new_capital - self.capital # ending capital - starting capital
        self.margin = self.capital_gain/self.capital
        self.capital = new_capital
        if self.capital <= 0:
            self.bankrupt = True
    
    def reset(self):
        """Reset the company to the initial state."""
        self.capital = self.initial_capital
        self.resilience = self.initial_resilience
        self.mitigation_pc = 0
        self.mitigation_amount = 0
        self.greenwash_pc = 0
        self.greenwash_amount = 0
        self.resilience_pc = 0
        self.resilience_amount = 0
        self.cumu_resilience_amount = 0
        self.cumu_mitigation_amount = 0
        self.cumu_greenwash_amount = 0
        self.margin = 0
        self.capital_gain = 0
        self.esg_score = 0
        self.bankrupt = False

class Investor:
    def __init__(self, capital=6, esg_preference=0):
        self.initial_capital = capital      # initial capital
        self.cash = capital              # current cash
        self.capital = capital            # current capital including cash and investment portfolio
        self.investments = {}               # dictionary to track investments in different companies
        self.esg_preference = esg_preference # the weight of ESG in the investor's decision making
        self.utility = 0                     # single-period reward
    
    def initial_investment(self, environment):
        """Invest in all companies at the beginning of the simulation."""
        self.investments = {i: 0 for i in range(environment.num_companies)}
    
    def invest(self, amount, company_idx):
        """Invest a certain amount in a company. 
        At the end of each period, investors collect all returns and then redistribute capital in next round."""
        if self.cash < amount:
            raise ValueError("Investment amount exceeds available capital.")
        else:
            self.cash -= amount
            self.investments[company_idx] += amount
    
    def update_investment_returns(self, environment):
        """Update the capital based on market performance and climate event."""
        for company_idx, investment in self.investments.items():
            company = environment.companies[company_idx]
            self.investments[company_idx] = max(investment * (1 + company.margin), 0) # worst case is to lose all investment

    def divest(self, company_idx, environment):
        """Divest from a company."""
        investment_return = self.investments[company_idx]
        self.cash += investment_return
        environment.companies[company_idx].lose_investment(investment_return)
        self.investments[company_idx] = 0
    
    def calculate_utility(self, environment):
        """Calculate reward based on market performance and ESG preferences."""
        invest_balance = 0
        esg_reward = 0
        if self.capital == 0:
            self.utility = 0
        else:
            for company_idx, investment in self.investments.items():
                if investment == 0:
                    continue
                company = environment.companies[company_idx]
                invest_balance += investment # investment includes returns on capital and principal
                esg_reward += company.esg_score*investment

            new_capital = invest_balance + self.cash
            avg_esg_reward = esg_reward / new_capital
            profit_rate = (new_capital - self.capital) / self.capital
            self.utility = profit_rate + self.esg_preference * avg_esg_reward
            self.capital = new_capital

    def reset(self):
        """Reset the investor to the initial state."""
        self.capital = self.initial_capital
        self.cash = self.initial_capital
        self.investments = {i: 0 for i in self.investments}
        self.utility = 0


class InvestESG(ParallelEnv):
    """
    ESG investment environment.
    """

    metadata = {"name": "InvestESG"}

    def __init__(
        self,
        company_attributes=None,
        investor_attributes=None,
        num_companies=5,
        num_investors=5,
        initial_heat_prob = 0.28,
        initial_precip_prob = 0.13,
        initial_drought_prob = 0.17,
        max_steps=80,
        market_performance_baseline=1.1, 
        market_performance_variance=0.0,
        allow_resilience_investment=False,
        allow_greenwash_investment=False,
        action_capping=0.1,
        company_esg_score_observable=False,
        climate_observable=False,
        avg_esg_score_observable=False,
        esg_spending_observable=False,
        resilience_spending_observable=False,
        **kwargs
    ):
        self.max_steps = max_steps
        self.timestamp = 0

        # initialize companies and investors based on attributes if not None
        if company_attributes is not None:
            self.companies = [Company(**attributes) for attributes in company_attributes]
            self.num_companies = len(company_attributes)
        else:
            self.companies = [Company() for _ in range(num_companies)]
            self.num_companies = num_companies
        
        if investor_attributes is not None:
            self.investors = [Investor(**attributes) for attributes in investor_attributes]
            self.num_investors = len(investor_attributes)
        else:
            self.num_investors = num_investors
            self.investors = [Investor() for _ in range(num_investors)]
        
        self.agents = [f"company_{i}" for i in range(self.num_companies)] + [f"investor_{i}" for i in range(self.num_investors)]
        self.n_agents = len(self.agents)
        self.possible_agents = self.agents[:]
        self.market_performance_baseline = market_performance_baseline # initial market performance
        self.market_performance_variance = market_performance_variance # variance of market performance
        self.allow_resilience_investment = allow_resilience_investment # whether to allow resilience investment by companies
        self.allow_greenwash_investment = allow_greenwash_investment # whether to allow greenwash investment by companies

        self.initial_heat_prob = initial_heat_prob # initial probability of heat wave
        self.initial_precip_prob = initial_precip_prob # initial probability of precipitation
        self.initial_drought_prob = initial_drought_prob # initial probability of drought
        self.heat_prob = initial_heat_prob # current probability of heat wave
        self.precip_prob = initial_precip_prob # current probability of precipitation
        self.drought_prob = initial_drought_prob # current probability of drought
        self.initial_climate_risk = 1 - (1-initial_heat_prob)*(1-initial_precip_prob)*(1-initial_drought_prob) # initial probability of at least one climate event
        self.climate_risk = self.initial_climate_risk # current probability of climate event
        self.climate_event_occurrence = 0 # number of climate events occurred in the current step
        
        self.action_capping = action_capping # action capping for company action
        self.company_esg_score_observable = company_esg_score_observable
        self.climate_observable = climate_observable # whether to include climate data in the observation space
        self.avg_esg_score_observable = avg_esg_score_observable # whethter to include company avg esg socre in the observation space
        self.esg_spending_observable = esg_spending_observable # whether to include company esg spending (mitigation + greenwash spending) in the observation space
        self.resilience_spending_observable = resilience_spending_observable # whether to include company resilience spending in the observation space
        # initialize investors with initial investments dictionary
        for investor in self.investors:
            investor.initial_investment(self)

        # initialize historical data storage
        self.history = {
            "esg_investment": [],
            "greenwash_investment": [],
            "resilience_investment": [],
            "climate_risk": [],
            "climate_event_occurs": [],
            "market_performance": [],
            "market_total_wealth": [],
            "company_rewards": [[] for _ in range(self.num_companies)],
            "investor_rewards": [[] for _ in range(self.num_investors)],
            "company_capitals": [[] for _ in range(self.num_companies)],
            "company_climate_risk": [[] for _ in range(self.num_companies)],
            "investor_capitals": [[] for _ in range(self.num_investors)],
            "investor_utility": [[] for _ in range(self.num_investors)],
            "investment_matrix": np.zeros((self.num_investors, self.num_companies)),
            "company_mitigation_amount": [[] for _ in range(self.num_companies)],
            "company_greenwash_amount": [[] for _ in range(self.num_companies)],
            "company_resilience_amount": [[] for _ in range(self.num_companies)],
            "company_esg_score": [[] for _ in range(self.num_companies)],
            "company_margin": [[] for _ in range(self.num_companies)],
            "company_rewards": [[] for _ in range(self.num_companies)],
            "investor_rewards": [[] for _ in range(self.num_investors)],
        }


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        ## Each company makes 3 decisions:
        ## 1. Amount to invest in emissions mitigation (continuous)
        ## 2. amount to invest in greenwash (continuous)
        ## 3. amount to invest in resilience (continuous)
        ## Each investor has num_companies possible*2 actions: for each company, invest/not invest
        
        # if agent is a company
        if agent.startswith("company"):
            return Box(low=0, high=self.action_capping, shape=(3,))
        else:  # investor
            return MultiDiscrete(self.num_companies * [2])
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # all agents have access to the same information, namely the capital, climate resilience, ESG score, and margin of each company
        # of all companies and the investment in each company and remaining cash of each investor
        observation_size = self.num_companies * 7 + self.num_investors * (self.num_companies + 1) + 3
        observation_space = Box(low=-np.inf, high=np.inf, shape=(observation_size,))
        return observation_space

    def step(self, actions):
        """Step function for the environment."""

        rng1 = np.random.default_rng(self.timestamp) # random number generator for market performance
        rng_heat = np.random.default_rng(self.timestamp*100) # random number generator for climate event
        rng_precip = np.random.default_rng(self.timestamp*500) # random number generator for climate event
        rng_drought = np.random.default_rng(self.timestamp*1000) # random number generator for climate event

        ## unpack actions
        # first num_companies actions are for companies, the rest are for investors
        companys_actions = {k: v for k, v in actions.items() if k.startswith("company_")}
        remaining_actions = {k: v for k, v in actions.items() if k not in companys_actions}
        # Reindex investor actions to start from 0
        investors_actions = {f"investor_{i}": action for i, (k, action) in enumerate(remaining_actions.items())}

        ## action masks
        # if company is brankrupt, it cannot invest in ESG or greenwashing
        for i, company in enumerate(self.companies):
            if company.bankrupt:
                companys_actions[f"company_{i}"] = np.array([0.0, 0.0, 0.0])

        # 0. investors divest from all companies and recollect capital
        for investor in self.investors:
            for company in investor.investments:
                investor.divest(company, self)

        # 1. investors allocate capital to companies (binary decision to invest/not invest)
        for i, investor in enumerate(self.investors):
            investor_action = investors_actions[f"investor_{i}"]
            # number of companies that the investor invests in
            num_investments = np.sum(investor_action)
            if num_investments > 0:
                # equal investment in each company; round down to nearest integer to avoid exceeding capital
                # print(f"investor {i} has {investor.cash} cash, and {investor.capital} capital")
                investment_amount = np.floor(investor.cash/num_investments) 
                for j, company in enumerate(self.companies):
                    if investor_action[j]:
                        investor.invest(investment_amount, j)
                        # company receives investment
                        company.receive_investment(investment_amount)
                   
        # 2. companies invest in ESG/greenwashing/none, report margin and esg score
        for i, company in enumerate(self.companies):
            if company.bankrupt:
                continue # skip if company is bankrupt
            company.mitigation_pc, company.greenwash_pc, company.resilience_pc = companys_actions[f"company_{i}"]
            company.greenwash_pc = company.greenwash_pc if self.allow_greenwash_investment else 0.0
            company.resilience_pc = company.resilience_pc if self.allow_resilience_investment else 0.0

            company.make_esg_decision()

        # 3. update probabilities of climate event based on cumulative ESG investments across companies
        total_mitigation_investment = np.sum(np.array([company.cumu_mitigation_amount for company in self.companies]))
        self.heat_prob = self.initial_heat_prob + 0.0083*self.timestamp/(1+0.0222*total_mitigation_investment)
        self.precip_prob = self.initial_precip_prob + 0.0018*self.timestamp/(1+0.0326*total_mitigation_investment)
        self.drought_prob = self.initial_drought_prob + 0.003*self.timestamp/(1+0.038*total_mitigation_investment)
        self.climate_risk = 1 - (1-self.heat_prob)*(1-self.precip_prob)*(1-self.drought_prob)

        # 4. market performance and climate event evolution
        self.market_performance = rng1.normal(loc=self.market_performance_baseline, scale=self.market_performance_variance)   # ranges from 0.9 to 1.1 most of time
        heat_event = (rng_heat.random() < self.heat_prob).astype(int)
        precip_event = (rng_precip.random() < self.precip_prob).astype(int)
        drought_event = (rng_drought.random() < self.drought_prob).astype(int)
        self.climate_event_occurrence = heat_event + precip_event + drought_event

        # 5. companies and investors update capital based on market performance and climate event
        for company in self.companies:
            if company.bankrupt:
                continue # skip if company is bankrupt
            company.update_capital(self)
        for investor in self.investors:
            investor.update_investment_returns(self)
        # 6. investors calculate returns based on market performance
        for i, investor in enumerate(self.investors):
            investor.calculate_utility(self)

        # 7. termination and truncation
        self.timestamp += 1
        termination = {agent: self.timestamp >= self.max_steps for agent in self.agents}
        truncation = termination

        observations = self._get_observation()
        rewards = self._get_reward()
        infos = self._get_infos()

        # 8. update history
        self._update_history()
        
        if any(termination.values()):
            self.agents = []
        
        # 8. update observation for each company and investor
        return observations, rewards, termination, truncation, infos

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        for company in self.companies:
            company.reset()
        for investor in self.investors:
            investor.reset()
        self.agents = [f"company_{i}" for i in range(self.num_companies)] + [f"investor_{i}" for i in range(self.num_investors)]
        self.market_performance = 1
        self.heat_prob = self.initial_heat_prob
        self.precip_prob = self.initial_precip_prob
        self.drought_prob = self.initial_drought_prob
        self.climate_risk = self.initial_climate_risk
        self.climate_event_occurrence = 0
        self.timestamp = 0
        # reset historical data
        self.history = {
            "esg_investment": [],
            "esg_investment": [],
            "greenwash_investment": [],
            "resilience_investment": [],
            "climate_risk": [],
            "climate_event_occurs": [],
            "market_performance": [],
            "market_total_wealth": [], 
            "company_capitals": [[] for _ in range(self.num_companies)],
            "company_climate_risk": [[] for _ in range(self.num_companies)],
            "investor_capitals": [[] for _ in range(self.num_investors)],
            "investor_utility": [[] for _ in range(self.num_investors)],
            "investment_matrix": np.zeros((self.num_investors, self.num_companies)),
            "company_mitigation_amount": [[] for _ in range(self.num_companies)],
            "company_greenwash_amount": [[] for _ in range(self.num_companies)],
            "company_resilience_amount": [[] for _ in range(self.num_companies)],
            "company_esg_score": [[] for _ in range(self.num_companies)],
            "company_margin": [[] for _ in range(self.num_companies)],
            "company_rewards": [[] for _ in range(self.num_companies)],
            "investor_rewards": [[] for _ in range(self.num_investors)],
        }
        self.fig = None
        self.ax = None

        return self._get_observation(), self._get_infos()
    
    def _get_observation(self):
        """Get observation for each company and investor. Public information is shared across all agents."""
        # Collect company observations
        company_obs = []
        for i, company in enumerate(self.companies):
            avg_esg_score = 0
            esg_spending = 0
            resilience_spending = 0
            company_esg_score = 0
            if self.company_esg_score_observable:
                company_esg_score = company.esg_score
            if self.avg_esg_score_observable:
                avg_esg_score = np.mean(self.history["company_esg_score"][i]) if len(self.history["company_esg_score"][i]) else 0
            if self.esg_spending_observable:
                esg_spending = company.cumu_mitigation_amount + company.cumu_greenwash_amount
            if self.resilience_spending_observable:
                resilience_spending = company.cumu_resilience_amount
            company_obs.extend([company.capital, company.resilience, company.margin, company_esg_score, avg_esg_score, esg_spending, resilience_spending])
        # Collect investor observations
        investor_obs = []
        for investor in self.investors:
            investor_obs.extend(list(investor.investments.values()) + [investor.capital])
        climate_obs = [0, 0, 0]
        if self.climate_observable:
            climate_obs = [self.climate_risk, self.climate_event_occurrence, self.market_performance]
        full_obs = np.array(company_obs + investor_obs + climate_obs)

        # Return the same observation for all agents
        return {agent: full_obs for agent in self.agents}

    def _get_reward(self):
        """Get reward for all agents."""
        rewards = {}
        for i, company in enumerate(self.companies):
            rewards[f"company_{i}"] = company.capital_gain #TODO: ideally, we should remove investor principals from company capitals
        for i, investor in enumerate(self.investors):
            rewards[f"investor_{i}"] = investor.utility
        return rewards
    
    def _get_infos(self):
        """Get infos for all agents. Dummy infos for compatibility with pettingzoo."""
        infos = {agent: {} for agent in self.agents}
        return infos

    def _update_history(self):
        """Update historical data."""
        self.history["esg_investment"].append(sum(company.cumu_mitigation_amount for company in self.companies))
        self.history["greenwash_investment"].append(sum(company.cumu_greenwash_amount for company in self.companies))
        self.history["resilience_investment"].append(sum(company.cumu_resilience_amount for company in self.companies))
        self.history["climate_risk"].append(self.climate_risk)
        self.history["climate_event_occurs"].append(self.climate_event_occurrence)
        self.history["market_performance"].append(self.market_performance)
        # at the end of the step investors haven't collected returns yet, so company capitals include returns for investors
        self.history["market_total_wealth"].append(sum(company.capital for company in self.companies)+sum(investor.cash for investor in self.investors))
        reward = self._get_reward()
        for i, company in enumerate(self.companies):
            self.history["company_capitals"][i].append(company.capital)
            self.history["company_mitigation_amount"][i].append(company.mitigation_amount)
            self.history["company_greenwash_amount"][i].append(company.greenwash_amount)
            self.history["company_resilience_amount"][i].append(company.resilience_amount)
            self.history["company_climate_risk"][i].append(company.resilience)
            self.history["company_esg_score"][i].append(company.esg_score)
            self.history["company_margin"][i].append(company.margin)
            self.history["company_rewards"][i].append(reward[f"company_{i}"])
        for i, investor in enumerate(self.investors):
            self.history["investor_capitals"][i].append(investor.capital)
            self.history["investor_utility"][i].append(investor.utility)
            self.history["investor_rewards"][i].append(reward[f"investor_{i}"])
            for j, investment in investor.investments.items():
                self.history["investment_matrix"][i, j] += investment

    @property
    def name(self) -> str:
        """Environment name."""
        return "InvestESG"

    def render(self, mode='human', fig='fig'):
        # import pdb; pdb.set_trace()
        
        if not hasattr(self, 'fig') or self.fig is None:
            # Initialize the plot only once
            self.fig = Figure(figsize=(32, 18))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.subplots(3, 4)  # Adjusted to 2 rows and 6 columns
            plt.subplots_adjust(hspace=0.5, wspace=1)  # Increased wspace from 0.2 to 0.3
            plt.ion()  # Turn on interactive mode for plotting

            # Generate a color for each company
            self.company_colors = plt.cm.rainbow(np.linspace(0, 1, self.num_companies))
            self.investor_colors = plt.cm.rainbow(np.linspace(0, 1, self.num_investors))
        # Ensure self.ax is always a list of axes
        if not isinstance(self.ax, np.ndarray):
            self.ax = np.array([self.ax])

        # Clear previous figures to update with new data
        for row in self.ax:
            for axis in row:
                axis.cla()

        # Subplot 1: Overall ESG Investment and Climate Risk over time
        ax1 = self.ax[0][0]
        ax2 = ax1.twinx()  # Create a secondary y-axis

        ax1.plot(self.history["esg_investment"], label='Cumulative ESG Investment', color='blue')
        ax2.plot(self.history["climate_risk"], label='Climate Risk', color='orange')
        # Add vertical lines for climate events
        for i, event in enumerate(self.history["climate_event_occurs"]):
            if event==1:
                ax1.axvline(x=i, color='orange', linestyle='--', alpha=0.5)
            if event>1:
                ax1.axvline(x=i, color='red', linestyle='--', alpha=0.5)

        ax1.set_title('Overall Metrics Over Time')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Investment in ESG')
        ax1.set_ylim(0, 200)
        ax2.set_ylabel('Climate Event Probability')
        ax2.set_ylim(0, 1)  # Set limits for Climate Event Probability

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Subplot 2: Company Decisions
        ax = self.ax[0][1]
        for i in range(self.num_companies):
            mitigation = self.history["company_mitigation_amount"][i]
            ax.plot(mitigation, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Mitigation Investments Over Time')
        ax.set_ylabel('Mitigation Investment')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 3: Company Greenwash Decisions
        ax = self.ax[0][2]
        for i in range(self.num_companies):
            greenwash = self.history["company_greenwash_amount"][i]
            ax.plot(greenwash, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Greenwash Investments Over Time')
        ax.set_ylabel('Greenwash Investment')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 4: Company Resilience Decisions
        ax = self.ax[0][3]
        for i in range(self.num_companies):
            resilience = self.history["company_resilience_amount"][i]
            ax.plot(resilience, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Resilience Investments Over Time')
        ax.set_ylabel('Resilience Investment')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 5: Company Climate risk exposure over time
        ax = self.ax[1][0]  
        for i, climate_risk_history in enumerate(self.history["company_climate_risk"]):
            ax.plot(climate_risk_history, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Climate Risk Exposure Over Time')
        ax.set_ylabel('Climate Risk Exposure')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 6: Company Capitals over time
        ax = self.ax[1][1]
        for i, capital_history in enumerate(self.history["company_capitals"]):
            ax.plot(capital_history, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Capitals Over Time')
        ax.set_ylabel('Capital')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 7: Company ESG Score over time
        ax = self.ax[1][2]
        for i, esg_score_history in enumerate(self.history["company_esg_score"]):
            ax.plot(esg_score_history, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company ESG Score Over Time')
        ax.set_ylabel('ESG Score')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 8: Investment Matrix
        investment_matrix = self.history["investment_matrix"]
        ax = self.ax[1][3]
        sns.heatmap(investment_matrix, ax=ax, cmap='Reds', cbar=True, annot=True, fmt='g')

        ax.set_title('Investment Matrix')
        ax.set_ylabel('Investor ID')
        ax.set_xlabel('Company ID')

         # Subplot 9: Investor Capitals over time
        ax = self.ax[2][0]
        for i, capital_history in enumerate(self.history["investor_capitals"]):
            ax.plot(capital_history, label=f'Investor {i}', color=self.investor_colors[i])
        ax.set_title('Investor Capitals Over Time')
        ax.set_ylabel('Capital')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 10: Investor Utility over time
        ax = self.ax[2][1]
        for i, utility_history in enumerate(self.history["investor_utility"]):
            ax.plot(utility_history, label=f'Investor {i}', color=self.investor_colors[i])
        ax.set_title('Investor Utility Over Time')
        ax.set_ylabel('Utility')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 11: Cumulative Investor Utility over time
        ax = self.ax[2][2]
        for i, utility_history in enumerate(self.history["investor_utility"]):
            cumulative_utility_history = list(itertools.accumulate(utility_history))
            ax.plot(cumulative_utility_history, label=f'Investor {i}', color=self.investor_colors[i])
        ax.set_title('Cumulative Investor Utility Over Time')
        ax.set_ylabel('Cumulative Utility')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 12: Market Total Wealth over time
        ax = self.ax[2][3]
        ax.plot(self.history["market_total_wealth"], label='Total Wealth', color='green')
        ax.set_title('Market Total Wealth Over Time')
        ax.set_ylabel('Total Wealth')
        ax.set_xlabel('Timestep')
        ax.legend()

        self.fig.tight_layout()

        # Update the plots
        self.canvas.draw()
        self.canvas.flush_events()
        plt.pause(0.001)  # Pause briefly to update plots

        # TODO: Consider generate videos later
        if mode == 'human':
            plt.show(block=False)
        elif mode == 'rgb_array':
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return img
        
        

if __name__ == "__main__":
    env = InvestESG(company_attributes=[{'capital':10000,'climate_risk_exposure':0.5,'beta':0},
                                    {'capital':10000,'climate_risk_exposure':0.5,'beta':0},
                                    {'capital':10000,'climate_risk_exposure':0.5,'beta':0}], 
                                    num_investors=3, initial_climate_event_probability=0.1,
                                    market_performance_baseline=1.05, market_performance_variance=0)
    env.reset()
    company_actions = {f"company_{i}": env.action_space(f"company_{i}").sample() for i in range(env.num_companies)}
    # company 0 never does anything
    company_actions['company_0'] = 0
    company_actions['company_1'] = 0
    company_actions['company_2'] = 0
    investor_actions = {f"investor_{i}": env.action_space(f"investor_{i}").sample() for i in range(env.num_investors)}
    # mask such that investor 0 only invests in company 0
    investor_actions['investor_0'] = [1, 0, 0]
    investor_actions['investor_1'] = [0, 1, 0]
    investor_actions['investor_2'] = [0, 0, 1]
    actions = {**company_actions, **investor_actions}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    for _ in range(100):
        obs, rewards, terminations, truncations, infos = env.step(actions)
    img = env.render(mode='rgb_array')
    import pdb; pdb.set_trace()

