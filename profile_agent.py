"""
ProfileAgent — Deep profiling engine for congressional representatives.
Maps Committee influence, Network ties, and Alpha performance per individual.
"""
import logging
import json

log = logging.getLogger("ProfileAgent")

class ProfileAgent:
    def __init__(self, trades, ties_data, committee_data):
        self.trades = trades
        self.ties = ties_data
        self.committees = committee_data
        self.profiles = {}

    def build_all_profiles(self):
        log.info("Deep profiling 535+ members of Congress...")
        senators = set([t.senator for t in self.trades])
        
        # Calculate global means for imputation
        all_ties = [p['network_centrality'] for p in self.profiles.values() if p['network_centrality'] > 0]
        mean_ties = sum(all_ties) / len(all_ties) if all_ties else 0.5
        
        for s in senators:
            senator_trades = [t for t in self.trades if t.senator == s]
            
            # Impute missing values
            coms = self.committees.get(s, ["GENERAL"])
            centrality = self.ties.get(s, mean_ties)
            if centrality == 0.0: centrality = mean_ties
            
            self.profiles[s] = {
                "name": s,
                "trade_count": len(senator_trades),
                "avg_excess_return": sum(t.excess_return for t in senator_trades) / len(senator_trades),
                "committees": coms,
                "network_centrality": centrality,
                "risk_profile": "AGGRESSIVE" if len(senator_trades) > 20 else "MODERATE"
            }
        log.info(f"Generated {len(self.profiles)} detailed profiles.")
        return self.profiles

    def save_profiles(self, path="data/senator_profiles.json"):
        with open(path, 'w') as f:
            json.dump(self.profiles, f, indent=4)
        log.info(f"Profiles saved to {path}")
