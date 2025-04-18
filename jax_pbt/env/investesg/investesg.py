from dataclasses import field, replace
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import random
from jax import jit
from jax import vmap
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
import numpy as np
import chex
from flax import struct

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns
import itertools
jax.config.update('jax_numpy_dtype_promotion', 'standard')

@struct.dataclass
class Company:
    bankrupt: bool = False
    initial_capital: float = 6.0
    capital: float = 6.0
    beta: float = 0.1667
    initial_resilience: float = 0.07
    resilience: float = 0.07
    resilience_incr_rate: float = 3.0
    cumu_mitigation_amount: float = 0.0
    cumu_greenwash_amount: float = 0.0
    cumu_resilience_amount: float = 0.0
    cumu_loss_amount: float = 0.0
    margin: float = 0.0
    capital_gain: float = 0.0
    mitigation_pc: float = 0.0
    greenwash_pc: float = 0.0
    resilience_pc: float = 0.0
    mitigation_amount: float = 0.0
    greenwash_amount: float = 0.0
    resilience_amount: float = 0.0
    loss_amount: float = 0.0
    esg_score: float = 0.0
    greenwash_esg_coef: float = 2.0

    def initialize_company(self, capital=6.0, climate_risk_exposure = 0.07, beta = 0.1667, greenwash_esg_coef = 2.0):
        capital = float(capital)
        greenwash_esg_coef = float(greenwash_esg_coef)
        return self.replace(
        initial_capital=capital,
        capital=capital,
        beta=beta,
        initial_resilience=climate_risk_exposure,
        resilience=climate_risk_exposure,
        resilience_incr_rate=3.0,
        cumu_mitigation_amount=0.0,
        cumu_greenwash_amount=0.0,
        cumu_resilience_amount=0.0,
        cumu_loss_amount=0.0,
        margin=0.0,
        capital_gain=0.0,
        mitigation_pc=0.0,
        greenwash_pc=0.0,
        resilience_pc=0.0,
        mitigation_amount=0.0,
        greenwash_amount=0.0,
        resilience_amount=0.0,
        esg_score=0.0,
        bankrupt=False,
        greenwash_esg_coef=greenwash_esg_coef
    )
    
    def receive_investment(self, amount):
        """Receive investment from investors."""
        return self.replace(capital=self.capital + amount)

    def lose_investment(self, amount):
        """Lose investment due to climate event."""
        return self.replace(capital=self.capital - amount)
    
    def make_esg_decision(self):
        """Make a decision on how to allocate capital."""
        
        # Calculate updated investment amounts for a single period
        mitigation_amount = self.mitigation_pc * self.capital
        greenwash_amount = self.greenwash_pc * self.capital
        resilience_amount = self.resilience_pc * self.capital

        # Calculate updated cumulative investment
        cumu_mitigation_amount = self.cumu_mitigation_amount + mitigation_amount
        cumu_greenwash_amount = self.cumu_greenwash_amount + greenwash_amount
        cumu_resilience_amount = self.cumu_resilience_amount + resilience_amount

        # Update resilience
        resilience = self.initial_resilience * jnp.exp(
            -self.resilience_incr_rate * (cumu_resilience_amount / self.capital)
        )

        # Update ESG score
        esg_score = self.mitigation_pc + self.greenwash_pc * self.greenwash_esg_coef

        # Return a new instance of the object with updated values
        return self.replace(
            mitigation_amount=mitigation_amount,
            greenwash_amount=greenwash_amount,
            resilience_amount=resilience_amount,
            cumu_mitigation_amount=cumu_mitigation_amount,
            cumu_greenwash_amount=cumu_greenwash_amount,
            cumu_resilience_amount=cumu_resilience_amount,
            resilience=resilience,
            esg_score=esg_score
        )


    def update_capital(self, state, uncertain_damage):
        """Update the capital based on market performance and climate event."""
        # add a random disturbance to market performance
        company_performance = random.normal(random.PRNGKey(0), shape=()) * self.beta + state.market_performance

        # New capital considering mitigation, resilience, and greenwashing investments
        new_capital = self.capital * (1 - self.mitigation_pc - self.resilience_pc - self.greenwash_pc) * company_performance
        # Climate event impact on capital
        new_capital_after_climate_event = jax.lax.cond(
            state.climate_event_occurrence > 0,
            lambda _: new_capital * (1.0 - (self.resilience * (1.0 - state.uncertain_economic_damage) + state.uncertain_economic_damage * uncertain_damage) * state.climate_event_occurrence.astype('float32')),  # if num_investments is 0, return 0
            lambda _: new_capital,  # else calculate the investment amount
            None
        )

        loss_amount = new_capital_after_climate_event - new_capital
        cumu_loss_amount = self.cumu_loss_amount + loss_amount

        # Calculate margin and capital gain
        capital_gain = new_capital_after_climate_event - self.capital
        margin = capital_gain / self.capital
        # Check if bankrupt
        bankrupt = new_capital_after_climate_event <= 0.0
        # Return a new object with updated capital, gain, margin, and bankruptcy status
        return self.replace(
            capital=new_capital_after_climate_event,
            capital_gain=capital_gain,
            margin=margin,
            bankrupt=bankrupt,
            loss_amount=loss_amount,
            cumu_loss_amount=cumu_loss_amount
        )
    
    def reset(self):
        """Reset the company to the initial state."""
        return self.replace(
            capital=self.initial_capital,
            resilience=self.initial_resilience,
            mitigation_pc=.0,
            mitigation_amount=.0,
            greenwash_pc=.0,
            greenwash_amount=.0,
            resilience_pc=.0,
            resilience_amount=.0,
            cumu_resilience_amount=.0,
            cumu_mitigation_amount=.0,
            cumu_greenwash_amount=.0,
            cumu_loss_amount=.0,
            loss_amount=.0,
            margin=.0,
            capital_gain=.0,
            esg_score=.0,
            bankrupt=False
        )

@struct.dataclass
class Investor:
    investments: chex.Array 
    initial_capital: float = 6.0
    cash: float = 6.0
    capital: float = 6.0
    esg_preference: float = 0.0
    utility: float = 0.0


    def initialize_investor(self, capital=6.0, esg_preference=0.0):
        capital, esg_preference, utility = float(capital), float(esg_preference), float(0.0)
        return self.replace(
            initial_capital = capital,
            cash = capital,
            capital = capital,
            esg_preference = esg_preference,
            utility = utility)
    
    def initial_investment(self, environment):
        """Invest in all companies at the beginning of the simulation."""
        # Create a new dictionary with investments set to 0 for each company
        investments = jnp.zeros(environment.num_companies)
        # Return a new object with the updated investments dictionary
        return self.replace(investments=investments)

    
    def invest(self, amount: float, company_idx: int):
        """Invest a certain amount in a company. 
        At the end of each period, investors collect all returns and then redistribute capital in next round."""
        # Update cash
        new_cash = self.cash - amount
        # Update investment for the specified company immutably
        new_investments = self.investments.at[company_idx].add(amount)
        # Return a new Investor object with updated cash and investments
        return self.replace(cash=new_cash, investments=new_investments)

    def update_investment_returns(self, state):
        """Update the capital based on market performance and climate event."""
        # Extract investments and margins into arrays
        investments = self.investments # Convert investments to JAX array
        margins = jnp.array([company.margin for company in state.companies])
        # Define the investment update function
        def update_investment(investment, margin):
            return jnp.maximum(investment * (1 + margin), 0) # Ensure no negative investment
        # Use vmap to apply update_investment over all investments and margins
        new_investments = vmap(update_investment)(investments, margins)
        return self.replace(investments=new_investments)
    


    def divest(self, company_idx: int, state) -> Tuple["Investor", "State"]:
        """Divest from a company."""
        # Get the current investment return from the company
        investment_return = self.investments[company_idx]
        # Update cash by adding the investment return (immutably)
        new_cash = self.cash + investment_return
        # Update the investments array immutably using JAX's index_update
        new_investments = self.investments.at[company_idx].set(0)
        # Update the company in the environment immutably
        # Immutably update the company in the environment
        updated_company = state.companies[company_idx].lose_investment(investment_return)
        # Create a new dictionary of companies, where only the updated company is changed
        new_companies = [updated_company if idx == company_idx else state.companies[idx] for idx in range(len(state.companies))]
        return self.replace(cash=new_cash, investments=new_investments), state.replace(companies=new_companies)

    
    def calculate_utility(self, state):
        """Calculate reward based on market performance and ESG preferences."""
        def get_utility():
            # Convert investments and company indices to JAX arrays
            investments = self.investments
            # Extract ESG scores for the companies
            esg_scores = jnp.array([company.esg_score for company in state.companies])
            # Define a helper function to calculate investment balance and ESG reward
            def investment_calc(investment, esg_score):
                return investment, esg_score * investment
            
            # Vectorize the function to apply to all investments
            invest_balances, esg_rewards = vmap(investment_calc)(investments, esg_scores)
            # Calculate total investment balance and ESG reward, skipping zero investments
            total_invest_balance = jnp.sum(invest_balances)
            total_esg_reward = jnp.sum(esg_rewards)
            # Calculate new capital and average ESG reward
            new_capital = total_invest_balance + self.cash
            avg_esg_reward = total_esg_reward / new_capital
            # Calculate profit rate
            profit_rate = (new_capital - self.capital) / self.capital
            return new_capital, profit_rate + self.esg_preference * avg_esg_reward
        # Check if capital is zero
        capital, utility = jax.lax.cond(
                    self.capital > .0,
                    lambda _: get_utility(),  # if num_investments is 0, return 0
                    lambda _: (jnp.array(0.0), jnp.array(0.0)), # else calculate the investment amount
                    None
        )

        return self.replace(capital=capital, utility=utility)
    
    def reset(self):
        """Reset the investor to the initial state."""
        return self.replace(
            capital=self.initial_capital,
            cash=self.initial_capital,
            utility=0.0
        )


@struct.dataclass
class State:

    time: int
    terminal: bool
    heat_prob: float
    precip_prob: float
    drought_prob: float
    climate_risk: float

    # History fields for tracking various investment and performance data
    history_esg_investment: chex.Array
    history_greenwash_investment: chex.Array
    history_resilience_investment: chex.Array
    history_loss_amount: chex.Array
    history_climate_risk: chex.Array
    history_climate_event_occurs: chex.Array
    history_market_performance: chex.Array
    history_market_total_wealth: chex.Array
    history_company_capitals: chex.Array
    history_company_mitigation_amount: chex.Array
    history_company_greenwash_amount: chex.Array
    history_company_resilience_amount: chex.Array
    history_company_loss_amount: chex.Array
    history_company_climate_risk: chex.Array
    history_company_esg_score: chex.Array
    history_company_margin: chex.Array
    history_company_rewards: chex.Array
    history_company_bankrupt: chex.Array
    history_investor_capitals: chex.Array
    history_investor_utility: chex.Array
    history_investor_rewards: chex.Array
    history_investment_matrix: chex.Array

    companies: List[Company]
    investors: List[Investor]
    market_performance: float = 1.0
    climate_event_occurrence: int = 0
    uncertain_economic_damage: int = 0
    new_bankrupt_standard: bool = False
    
    
    
class InvestESG(MultiAgentEnv):
    """
    JAX Compatible version of ESG investment environment.
    """

    def __init__(
        self,
        company_attributes=None,
        investor_attributes=None,
        num_companies=5,
        num_investors=5,
        initial_heat_prob = 0.28,
        initial_precip_prob = 0.13,
        initial_drought_prob = 0.17,
        max_steps=100,
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
        locked_period=1,
        real_data_seeding=False,
        uncertain_economic_damage=False,
        new_bankrupt_standard=False,
        fixed_random_seed=True,
        **kwargs
    ):
        self.max_steps = max_steps
        self.timestamp = 0

        # initialize companies and investors based on attributes if not None
        if company_attributes is not None:
            self.companies = [Company().initialize_company(**attributes) for attributes in company_attributes]
            self.num_companies = len(company_attributes)
        else:
            self.companies = [Company() for _ in range(num_companies)]
            self.num_companies = num_companies
        
        if investor_attributes is not None:
            self.investors = [Investor(investments=jnp.zeros(self.num_companies)).initialize_investor(**attributes) for attributes in investor_attributes]
            self.num_investors = len(investor_attributes)
        else:
            self.num_investors = num_investors
            self.investors = [Investor(investments=jnp.zeros(self.num_companies)) for _ in range(num_investors)]

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

        # Additional configurations
        self.locked_period = locked_period # Agent action will be locked-in for this period of time, default to 1
        self.real_data_seeding = real_data_seeding # Enforce 50% companies do hard-coded mitigation first 5 years if true, default false
        self.uncertain_economic_damage = uncertain_economic_damage # Companies will suffer from uncertain amount of economic damage if true, default false
        self.new_bankrupt_standard = new_bankrupt_standard # Companies have margin < -10% for 3 consecutive years will go bankrupt under new standard, default false
        self.fixed_random_seed = fixed_random_seed # Fix the underlying across training episode, default true

        # initialize investors with initial investments dictionary
        for idx, investor in enumerate(self.investors):
            self.investors[idx] = investor.initial_investment(self)

    def action_space(self, agent):
        ## Each company makes 3 decisions:
        ## 1. Amount to invest in emissions mitigation (continuous)
        ## 2. amount to invest in greenwash (continuous)
        ## 3. amount to invest in resilience (continuous)
        ## Each investor has num_companies possible*2 actions: for each company, invest/not invest
        # if agent is a company
        if agent.startswith("company"):
            return spaces.Box(low=0, high=self.action_capping, shape=(3,))
        else:
            return spaces.MultiDiscrete(self.num_companies*[2]) # 0: not invest, 1: invest
    
    def observation_space(self):
        # all agents have access to the same information, namely the capital, climate resilience, ESG score, and margin of each company
        # of all companies and the investment in each company and remaining cash of each investor
        observation_size = self.num_companies * 4 + self.num_investors * (self.num_companies + 1) + 3
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,))
        return observation_space
    
    def get_companies(self, state: State):
        return {f'company_{k}': v for k, v in enumerate(state.companies)}
    

    def get_investors(self, state: State):
        return {f'investor_{k}': v for k, v in enumerate(state.investors)}
    
    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_st, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_st, obs_st
        )
        return obs, states, rewards, dones, infos
    
    def step_env(
            self,
            key: chex.PRNGKey,
            state,
            actions: Dict[str, chex.Array],
    ):

        """Step function for the environment."""
        ## unpack actions
        # first num_companies actions are for companies, the rest are for investors
        companys_actions = {k: v for k, v in actions.items() if k.startswith("company_")}
        remaining_actions = {k: v for k, v in actions.items() if k not in companys_actions}
        # Reindex investor actions to start from 0
        investors_actions = {f"investor_{i}": action for i, (k, action) in enumerate(remaining_actions.items())}
        companies = self.get_companies(state)
        ## action masks
        # if company is brankrupt, it cannot invest in ESG or greenwashing
        for name, company in companies.items():
            companys_actions[name] = jax.lax.cond(
                company.bankrupt == True,
                lambda _: jnp.array([0.0, 0.0, 0.0]),
                lambda _: companys_actions[name],
                None
            )

        # 0. investors divest from all companies and recollect capital
        # Vectorize the divestment process for all investors
        investors = list(self.get_investors(state).values())

        for investor_idx, investor in enumerate(investors):
            state, investors[investor_idx] = self._divest_investor(state, investors[investor_idx])

        # Update the environment and investors
        state = state.replace(investors=investors)

        # 1. investors allocate capital to companies (binary decision to invest/not invest)
        investors = state.investors
        companies = state.companies
        for i, investor in enumerate(state.investors):
            investor_action = investors_actions[f"investor_{i}"]
            # number of companies that the investor invests in
            num_investments = jnp.sum(investor_action)
            # equal investment in each company; round down to nearest integer to avoid exceeding capital
            def calculate_investment_amount(investor_cash, num_investments):
                # Use jax.lax.cond to check if num_investments is 0
                investment_amount = jax.lax.cond(
                    num_investments > 0,
                    lambda _: jnp.floor(investor_cash / num_investments.astype('float32')),  # if num_investments is 0, return 0
                    lambda _: jnp.array(0.0),  # else calculate the investment amount
                    None
                )
                return investment_amount
            
            investment_amount = calculate_investment_amount(investor.cash, num_investments)
            
            for j, company in enumerate(state.companies):
                def i_invest_in_j(i, j, investment_amount):
                    investor, company = jax.lax.cond(
                        investor_action[j] > 0,
                        lambda _: (investors[i].invest(investment_amount, j), companies[j].receive_investment(investment_amount)), # if not invest, return the same
                        lambda _: (investors[i], companies[j]), # else calculate the investment amount
                        None
                    )
                    return investor, company
                investors[i], companies[j] = i_invest_in_j(i, j, investment_amount)
        state = state.replace(companies=companies, investors=investors)
                   
        # 2. companies invest in ESG/greenwashing/none, report margin and esg score
        # Vectorize the process_company function across all companies and their actions
        for company_idx, company in enumerate(companies):
            companies[company_idx] = self._process_company(company, jnp.array(companys_actions[f"company_{company_idx}"]))
        state = state.replace(companies=companies)

        # 3. update probabilities of climate event based on cumulative ESG investments across companies
        total_mitigation_investment = jnp.sum(jnp.array([company.cumu_mitigation_amount for company in state.companies]))
        heat_prob = self.initial_heat_prob + 0.083*state.time/(1+0.0222*total_mitigation_investment)
        precip_prob = self.initial_precip_prob + 0.018*state.time/(1+0.0326*total_mitigation_investment)
        drought_prob = self.initial_drought_prob + 0.03*state.time/(1+0.038*total_mitigation_investment)
        climate_risk = 1 - (1-heat_prob)*(1-precip_prob)*(1-drought_prob)
        state = state.replace(
            heat_prob = heat_prob,
            precip_prob = precip_prob,
            drought_prob = drought_prob,
            climate_risk = climate_risk
        )

        # 4. market performance and climate event evolution
        rng_key = key
        if self.fixed_random_seed:
            rng_heat = jax.random.key(state.time * 100) # random number generator for heat event
            rng_precip = jax.random.key(state.time * 500) # random number generator for flood event
            rng_drought = jax.random.key(state.time * 1000) # random number generator for drought event
        
        else:
            _, rng_heat = jax.random.split(key) # random number generator for heat event
            _, rng_precip = jax.random.split(rng_heat) # random number generator for flood event
            _, rng_drought = jax.random.split(rng_precip) # random number generator for drought event
        
        rng_key, rng_key1, rng_key2 = random.split(rng_key, 3)
        
        new_market_performance = random.normal(rng_key1) * self.market_performance_variance + self.market_performance_baseline
        heat_event = (random.uniform(rng_heat) < state.heat_prob).astype(int)
        precip_event = (random.uniform(rng_precip) < state.precip_prob).astype(int)
        drought_event = (random.uniform(rng_drought) < state.drought_prob).astype(int)
        
        climate_event_occurrence = heat_event + precip_event + drought_event
        
        state = state.replace(
            market_performance=new_market_performance,
            climate_event_occurrence=climate_event_occurrence
        )

        # 5. companies update capital based on market performance and climate event
        keys = random.split(key, num=3)
        for company_idx, company in enumerate(companies):
            companies[company_idx] = self._update_company_capital(company, state, keys[company_idx])
        state = state.replace(companies=companies)
        
        # 6. investors calculate returns based on market performance
        for investor_idx, investor in enumerate(investors):
            investors[investor_idx] = self._calculate_investor_utility(self._update_investor_returns(investor, state), state)
        state = state.replace(investors=investors)

        # 7. update history
        state = self._update_history(state)

        # Check bankrupt
        if self.new_bankrupt_standard:
            state = self._update_bankrupt(state)

        # 8. termination and truncation
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state)
        state = state.replace(terminal=done)
        dones = self._get_dones(done)

        infos = self._get_infos()

        return (self._get_observation(state), state, self._get_reward(state), dones, infos)


    def _divest_investor(self, state, investor):
        # Vectorized function to apply divestment across all companies
        def divest_company(company_idx, state):
            return investor.divest(company_idx, state)
        
        for i in range(self.num_companies):
            investor, state = divest_company(i, state)
        return state, investor
    
    def _process_investor_actions(self, investor: Investor, investor_action: jnp.array, companies: List[Company]):
        """Process investor actions and update company investments."""
        num_investments = jnp.sum(investor_action)
        if num_investments > 0:
            investment_amount = jnp.floor(investor.cash / num_investments)
            def invest_in_company(company, action):
                return company.receive_investment(investment_amount) if action else company
            # Use vmap to vectorize the investment in each company
            updated_companies = vmap(invest_in_company)(companies, investor_action)
            return updated_companies
        else:
            return companies
    
    def _process_company(self, company, action):
        """Update company's actions and make ESG decisions."""

        def process_company_action():
            # Unpack actions and apply conditions for greenwash and resilience investment
            mitigation_pc, greenwash_pc, resilience_pc = action
            greenwash_pc = greenwash_pc if self.allow_greenwash_investment else 0.0
            resilience_pc = resilience_pc if self.allow_resilience_investment else 0.0

            # Update the company's percentages and make ESG decisions immutably
            updated_company = company.replace(
                mitigation_pc=mitigation_pc,
                greenwash_pc=greenwash_pc,
                resilience_pc=resilience_pc
            )
            # Call the ESG decision-making method (must return an updated company)
            return updated_company.make_esg_decision()
        
        updated_company = jax.lax.cond(
            company.bankrupt,
            lambda _: company,
            lambda _: process_company_action(),
            None
        )
        return updated_company

    def _update_company_capital(self, company, state, key):
        def get_uncertain_damage():
            mean = company.resilience  # 7%
            std_dev = 0.1  # Spread around the mean
            random_value = jax.random.normal(key, shape=()) * std_dev + mean
            random_value = jax.lax.cond(
                jnp.any(jnp.isnan(random_value)),
                lambda _: mean,
                lambda _: random_value,
                None
            )
            random_value = jnp.clip(random_value, 0.0, 1.0)  # Clip to stay in range
            # Check if the result is NaN
            return random_value
        

        updated_company = jax.lax.cond(
            company.bankrupt,
            lambda _: company,
            lambda _: company.update_capital(state, get_uncertain_damage()),
            None
        )
        return updated_company
    
    def _update_investor_returns(self, investor, state):
        return investor.update_investment_returns(state)
    
    def _calculate_investor_utility(self, investor, state):
        return investor.calculate_utility(state)
    
    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> chex.Array:
        obs_batch = self._get_observation(state)
        return obs_batch

    def _get_observation(self, state: State):
        """Get observation for each company and investor. Public information is shared across all agents."""

        # Vectorized function to map over companies
        companies = self.get_companies(state)
        investors = self.get_investors(state)

        def vectorized_get_company_obs():
            capital = jnp.array([company.capital for company in companies.values()])
            resilience = jnp.array([company.resilience for company in companies.values()])
            esg_score = jnp.zeros(len(companies), dtype=float)
            margin = jnp.array([company.margin for company in companies.values()])
            if self.company_esg_score_observable:
                esg_score = jnp.array([company.esg_score for company in companies.values()])
            return jnp.stack([capital, resilience, esg_score, margin], axis=1)

        # Vectorized function to map over investors
        def vectorized_get_vestor_obs():
            investments = jnp.array([investor.investments for investor in investors.values()])
            capitals = jnp.array([investor.capital for investor in investors.values()])[:, jnp.newaxis]
            return jnp.concatenate([investments, capitals], axis=1)

        company_obs = vectorized_get_company_obs()
        investor_obs = vectorized_get_vestor_obs()
        
        climate_obs = jnp.zeros(3)
        if self.climate_observable:
            climate_obs = jnp.array([state.climate_risk, state.climate_event_occurrence.astype('float32'), state.market_performance])
        full_obs = jnp.concatenate([company_obs.flatten(), investor_obs.flatten(), climate_obs])
        return {agent: full_obs for agent in self.agents}

    def _get_infos(self):
        """Get infos for all agents. Dummy infos for compatibility with pettingzoo."""
        infos = {agent: {} for agent in self.agents}
        return infos
    
    def _get_dones(self, done):
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        return dones

    def _get_reward(self, state):
        """Get reward for all agents."""
        # Helper function to get company rewards
        def get_company_reward(i):
            return (f"company_{i}", state.companies[i].capital_gain)
        # Helper function to get investor rewards
        def get_investor_reward(i):
            return (f"investor_{i}", state.investors[i].utility)
        # Use vmap to vectorize over companies and investors
        company_rewards = [get_company_reward(company_idx) for company_idx in range(len(self.companies))]
        investor_rewards = [get_investor_reward(investor_idx) for investor_idx in range(len(self.investors))]
        # Combine both company and investor rewards into a dictionary
        rewards = {**dict(company_rewards), **dict(investor_rewards)}
        return rewards
    
    def reset(self, key=chex.PRNGKey):
        """Reset the environment."""
        
        # Helper function to reset each company
        def reset_company(company):
            return company.reset()

        # Helper function to reset each investor
        def reset_investor(investor):
            return investor.reset()

        # Reset all companies and investors using vmap
        updated_companies = [reset_company(company) for company in self.companies]

        # updated_companies = jax.vmap(reset_company)(jnp.array(self.companies))
        updated_investors = [reset_investor(investor) for investor in self.investors]
        # jax.vmap(reset_investor)(jnp.array(self.investors))

        # Reset the environment attributes immutably
        agents = dict()
        agents.update({f"company_{i}": updated_companies[i] for i in range(self.num_companies)})
        agents.update({f"investor_{i}": updated_investors[i] for i in range(self.num_investors)})
        

        state = State(
            time=0,
            terminal=False,
            market_performance=1.0,
            heat_prob=self.initial_heat_prob,
            precip_prob=self.initial_precip_prob,
            drought_prob=self.initial_drought_prob,
            climate_risk=self.initial_climate_risk,
            climate_event_occurrence=0,
            companies=updated_companies,
            investors=updated_investors,
            uncertain_economic_damage=self.uncertain_economic_damage,
            new_bankrupt_standard=self.new_bankrupt_standard,

            # Initialize history arrays with zeros, sized according to self.max_steps
            history_esg_investment=jnp.zeros(self.max_steps),
            history_greenwash_investment=jnp.zeros(self.max_steps),
            history_resilience_investment=jnp.zeros(self.max_steps),
            history_loss_amount=jnp.zeros(self.max_steps),
            history_climate_risk=jnp.zeros(self.max_steps),
            history_climate_event_occurs=jnp.zeros(self.max_steps, dtype=int),
            history_market_performance=jnp.zeros(self.max_steps),
            history_market_total_wealth=jnp.zeros(self.max_steps),

            # Company-related histories initialized with size [self.num_companies, self.max_steps]
            history_company_capitals=jnp.zeros((self.num_companies, self.max_steps)),
            history_company_mitigation_amount=jnp.zeros((self.num_companies, self.max_steps)),
            history_company_greenwash_amount=jnp.zeros((self.num_companies, self.max_steps)),
            history_company_resilience_amount=jnp.zeros((self.num_companies, self.max_steps)),
            history_company_loss_amount=jnp.zeros((self.num_companies, self.max_steps)),
            history_company_climate_risk=jnp.zeros((self.num_companies, self.max_steps)),
            history_company_esg_score=jnp.zeros((self.num_companies, self.max_steps)),
            history_company_margin=jnp.zeros((self.num_companies, self.max_steps)),
            history_company_rewards=jnp.zeros((self.num_companies, self.max_steps)),
            history_company_bankrupt=jnp.zeros((self.num_companies, self.max_steps)),

            # Investor-related histories initialized with size [self.num_investors, self.max_steps]
            history_investor_capitals=jnp.zeros((self.num_investors, self.max_steps)),
            history_investor_utility=jnp.zeros((self.num_investors, self.max_steps)),
            history_investor_rewards=jnp.zeros((self.num_investors, self.max_steps)),

            # Initialize investment matrix with size [self.num_investors, self.num_companies]
            history_investment_matrix=jnp.zeros((self.num_investors, self.num_companies)),
        )
        self.fig = None

        # Return a new environment object with updated state
        return self.get_obs(state), state


    @property
    def name(self) -> str:
        """Environment name."""
        return "InvestESG"

    def _update_bankrupt(self, state: State):
         # Loop over companies and update their respective histories
        companies = state.companies
        for company_idx, company in enumerate(state.companies):
            margin = state.history_company_margin[company_idx, state.time]
            margin_last_year = state.history_company_margin[company_idx, state.time-1]
            margin_year_before = state.history_company_margin[company_idx, state.time-2]
            margins = jnp.array([margin_year_before, margin_last_year, margin])
            companies[company_idx] = jax.lax.cond(
                jnp.all(margins < -0.1),
                lambda _: company.replace(bankrupt=True),
                lambda _: company,
                None
            )
        return state.replace(companies=companies)

    
    def _update_history(self, state: State):
        """Update historical data."""
        state = state.replace(
            history_esg_investment=state.history_esg_investment.at[state.time].set(sum(company.cumu_mitigation_amount for company in state.companies)),
            history_greenwash_investment=state.history_greenwash_investment.at[state.time].set(
            sum(company.cumu_greenwash_amount for company in state.companies)
        ),
            history_resilience_investment=state.history_resilience_investment.at[state.time].set(
            sum(company.cumu_resilience_amount for company in state.companies)
        ),
            history_loss_amount=state.history_loss_amount.at[state.time].set(
            sum(company.cumu_loss_amount for company in state.companies)
        ),
        history_climate_risk=state.history_climate_risk.at[state.time].set(state.climate_risk),
        history_climate_event_occurs=state.history_climate_event_occurs.at[state.time].set(state.climate_event_occurrence),
        history_market_performance=state.history_market_performance.at[state.time].set(state.market_performance),
        history_market_total_wealth=state.history_market_total_wealth.at[state.time].set(
            sum(company.capital for company in state.companies) + sum(investor.cash for investor in state.investors)
        ))

        # Reward computation
        reward = self._get_reward(state=state)

        # Loop over companies and update their respective histories
        for i, company in enumerate(state.companies):
            state = state.replace(
                history_company_capitals=state.history_company_capitals.at[i, state.time].set(company.capital),
                history_company_mitigation_amount=state.history_company_mitigation_amount.at[i, state.time].set(company.mitigation_amount),
                history_company_greenwash_amount=state.history_company_greenwash_amount.at[i, state.time].set(company.greenwash_amount),
                history_company_resilience_amount=state.history_company_resilience_amount.at[i, state.time].set(company.resilience_amount),
                history_company_loss_amount=state.history_company_loss_amount.at[i, state.time].set(company.loss_amount),
                history_company_climate_risk=state.history_company_climate_risk.at[i, state.time].set(company.resilience),
                history_company_esg_score=state.history_company_esg_score.at[i, state.time].set(company.esg_score),
                history_company_margin=state.history_company_margin.at[i, state.time].set(company.margin),
                history_company_rewards=state.history_company_rewards.at[i, state.time].set(reward[f"company_{i}"]),
                history_company_bankrupt=state.history_company_bankrupt.at[i, state.time].set(company.bankrupt),
            )
        # Loop over investors and update their respective histories
        for i, investor in enumerate(state.investors):
            state = state.replace(
                history_investor_capitals=state.history_investor_capitals.at[i, state.time].set(investor.capital),
                history_investor_utility=state.history_investor_utility.at[i, state.time].set(investor.utility),
                history_investor_rewards=state.history_investor_rewards.at[i, state.time].set(reward[f"investor_{i}"])
            )

            # Update the investment matrix for each investment of the investor
            for j, investment in enumerate(investor.investments):
                state = state.replace(
                    history_investment_matrix=state.history_investment_matrix.at[i, j].add(investment)
                )
        return state

    def render(self, state, mode='human', fig='fig'):
        
        if not hasattr(self, 'fig') or self.fig is None:
            # Initialize the plot only once
            self.fig = Figure(figsize=(32, 18))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.subplots(3, 4)  # Adjusted to 2 rows and 6 columns
            plt.subplots_adjust(hspace=0.5, wspace=1)  # Increased wspace from 0.2 to 0.3
            plt.ion()  # Turn on interactive mode for plotting
            plt.rcParams.update({'font.size': 14}) # make the font bigger
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
        # import pdb; pdb.set_trace()
        if len(state.history_esg_investment.shape) == 2:
            history_esg_investment = state.history_esg_investment[0]
            history_climate_risk = state.history_climate_risk[0]
            history_climate_event_occurs = state.history_climate_event_occurs[0] if len(state.history_climate_event_occurs.shape) == 2 else state.history_climate_event_occurs
            history_company_mitigation_amount = state.history_company_mitigation_amount[0]
            history_company_greenwash_amount = state.history_company_greenwash_amount[0]
            history_company_resilience_amount = state.history_company_resilience_amount[0]
            history_company_loss_amount = state.history_company_loss_amount[0]
            history_company_climate_risk = state.history_company_climate_risk[0]
            history_company_capitals = state.history_company_capitals[0]
            history_company_esg_score = state.history_company_esg_score[0]
            history_investment_matrix = state.history_investment_matrix[0]
            history_investor_capitals = state.history_investor_capitals[0]
            history_investor_utility = state.history_investor_utility[0]
            history_market_total_wealth = state.history_market_total_wealth[0]

        else:
            history_esg_investment = state.history_esg_investment
            history_climate_risk = state.history_climate_risk
            history_climate_event_occurs = state.history_climate_event_occurs
            history_company_mitigation_amount = state.history_company_mitigation_amount
            history_company_greenwash_amount = state.history_company_greenwash_amount
            history_company_resilience_amount = state.history_company_resilience_amount
            history_company_loss_amount = state.history_company_loss_amount
            history_company_climate_risk = state.history_company_climate_risk
            history_company_capitals = state.history_company_capitals
            history_company_esg_score = state.history_company_esg_score
            history_investment_matrix = state.history_investment_matrix
            history_investor_capitals = state.history_investor_capitals
            history_investor_utility = state.history_investor_utility
            history_market_total_wealth = state.history_market_total_wealth


        ax1.plot(history_esg_investment, label='Cumulative ESG Investment', color='blue')
        ax2.plot(history_climate_risk, label='Climate Risk', color='orange')
        # Add vertical lines for climate events
        for i, event in enumerate(history_climate_event_occurs):
            if event==1:
                ax1.axvline(x=i, color='orange', linestyle='--', alpha=0.5)
            if event>1:
                ax1.axvline(x=i, color='red', linestyle='--', alpha=0.5)

        ax1.set_title('Overall Metrics Over Time ')
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
            mitigation = history_company_mitigation_amount[i]
            ax.plot(mitigation, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Mitigation Investments Over Time')
        ax.set_ylabel('Mitigation Investment')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 3: Company Greenwash Decisions
        ax = self.ax[0][2]
        for i in range(self.num_companies):
            greenwash = history_company_greenwash_amount[i]
            ax.plot(greenwash, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Greenwash Investments Over Time')
        ax.set_ylabel('Greenwash Investment')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 4: Company Resilience Decisions
        ax = self.ax[0][3]
        for i in range(self.num_companies):
            resilience = history_company_resilience_amount[i]
            ax.plot(resilience, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Resilience Investments Over Time')
        ax.set_ylabel('Resilience Investment')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 5: Company Climate risk exposure over time
        ax = self.ax[1][0]  
        for i, climate_risk_history in enumerate(history_company_climate_risk):
            ax.plot(climate_risk_history, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Climate Risk Exposure Over Time')
        ax.set_ylabel('Climate Risk Exposure')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 6: Company Capitals over time
        ax = self.ax[1][1]
        for i, capital_history in enumerate(history_company_capitals):
            ax.plot(capital_history, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Capitals Over Time')
        ax.set_ylabel('Capital')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 7: Company ESG Score over time
        ax = self.ax[1][2]
        for i, esg_score_history in enumerate(history_company_esg_score):
            ax.plot(esg_score_history, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company ESG Score Over Time')
        ax.set_ylabel('ESG Score')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 8: Investment Matrix
        investment_matrix = history_investment_matrix
        ax = self.ax[1][3]
        sns.heatmap(investment_matrix, ax=ax, cmap='Reds', cbar=True, annot=True, fmt='g')

        ax.set_title('Investment Matrix')
        ax.set_ylabel('Investor ID')
        ax.set_xlabel('Company ID')

         # Subplot 9: Investor Capitals over time
        ax = self.ax[2][0]
        for i, capital_history in enumerate(history_investor_capitals):
            ax.plot(capital_history, label=f'Investor {i}', color=self.investor_colors[i])
        ax.set_title('Investor Capitals Over Time')
        ax.set_ylabel('Capital')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 10: Investor Utility over time
        ax = self.ax[2][1]
        for i, utility_history in enumerate(history_investor_utility):
            ax.plot(utility_history, label=f'Investor {i}', color=self.investor_colors[i])
        ax.set_title('Investor Utility Over Time')
        ax.set_ylabel('Utility')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 11: Cumulative Investor Utility over time
        ax = self.ax[2][2]
        for i, utility_history in enumerate(history_investor_utility):
            cumulative_utility_history = list(itertools.accumulate(utility_history))
            ax.plot(cumulative_utility_history, label=f'Investor {i}', color=self.investor_colors[i])
        ax.set_title('Cumulative Investor Utility Over Time')
        ax.set_ylabel('Cumulative Utility')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 12: Market Total Wealth over time
        ax = self.ax[2][3]
        ax.plot(history_market_total_wealth, label='Total Wealth', color='green')
        ax.set_title('Market Total Wealth Over Time')
        ax.set_ylabel('Total Wealth')
        ax.set_xlabel('Timestep')
        ax.legend()

        self.fig.tight_layout()

        # Update the plots
        self.canvas.draw()
        self.canvas.flush_events()
        # plt.pause(0.001)  # Pause briefly to update plots

        # Consider generate videos later
        if mode == 'human':
            plt.show(block=False)
        elif mode == 'rgb_array':
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return img
        
        
if __name__ == "__main__":
    env = InvestESG()
    print(env.action_space())
    print(env.observation_space())
