from env.pettingzoo.investesg import InvestESG
import wandb
import torch
import numpy as np

ACTION_MAP = {
    0: "None",
    1: "Mitigation",
    2: "Greenwashing"
}

class WrapperInvestESGEnv(InvestESG):

    def step(self, all_clipped_actions):
        combined_action = {
                f"company_{company_id}": all_clipped_actions[company_id][0] for company_id in range(self.num_companies)
            }
        combined_action.update({
                f"investor_{investor_id}": all_clipped_actions[self.num_companies + investor_id][0] for investor_id in range(self.num_investors)
        })
        observations, rewards, termination, truncation, infos = super().step(combined_action)
        return list(observations.values()), list(rewards.values()), list(termination.values()), list(truncation.values()), list(infos.values())
    
    def log(self, episode):
        d = {
                "episode": episode,
                "total climate_event_occurs": sum(self.history["climate_event_occurs"]),
                "final climate risk": self.history["climate_risk"][-1],
                "cumulative climate risk": sum(self.history["climate_risk"]),
                "final mitigation investment": self.history['esg_investment'][-1],
                "final greenwash investment": self.history['greenwash_investment'][-1],
                "final resilience investment": self.history['resilience_investment'][-1],
                "market total wealth": self.history["market_total_wealth"][-1]
            }
        for i in range(self.num_companies):
            d[f'company_{i} total investment'] = sum(self.history["investment_matrix"][:, i])
            d[f'company_{i} episodal reward'] = sum(self.history[f"company_rewards"][i])
            d[f'company_{i} final capital'] = self.history[f"company_capitals"][i][-1]
            d[f'company_{i} mitigation amount'] = sum(self.history[f"company_mitigation_amount"][i])
            if self.allow_greenwash_investment:
                d[f'company_{i} greenwash amount'] = sum(self.history[f"company_greenwash_amount"][i])
            if self.allow_resilience_investment:
                d[f'company_{i} resilience amount'] = sum(self.history[f"company_resilience_amount"][i])
            # company_decisions = self.history["company_decisions"][agent_id]
            # for action in range(3):
            #     d[f'company_{agent_id} {ACTION_MAP[action]}'] = company_decisions.count(action) / len(company_decisions)
        for i in range(self.num_investors):
            d[f'cumulative investor_{i} capital'] = sum(self.history["investor_capitals"][i])
            d[f'investor_{i} episodal reward'] = sum(self.history[f"investor_rewards"][i])
            total_investment = sum(self.history["investment_matrix"][i, :])
            d[f'investor_{i} total investment'] = total_investment
            for company, investment in enumerate(self.history["investment_matrix"][i, :]):
                d[f'investor_{i} investment to company_{company}'] = investment / total_investment

        if episode > 25000:
            columns = ["esg_investment", "climate_risk", "market_total_wealth"]
            test_table = wandb.Table(columns=[])
            for col in columns:
                test_table.add_column(name=col, data=self.history[col])
            d["history"] = test_table
        wandb.log(d)

  
