class RuleBasedGame:
    def __init__(self, model):
        self.house_list = {h.id: h for h in model.housing_stock} 
    def run(self, households):
        for household in households.values():
            print(f"Household {household.unique_id} is attempting to buy.")
            household.rule_based_buying(self.house_list)  