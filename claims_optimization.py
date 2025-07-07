# Claims Settlement Optimization System (Enhanced & Modular)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------ #
class Config:
    CUSTOMER_VALUE_MULTIPLIER = 10000
    DEFAULT_SETTLEMENT_RATIO = 0.7
    MIN_SETTLEMENT_RATIO = 0.5

# ------------------ DATA SIMULATION ------------------ #
def generate_sample_data(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        'claim_amount': np.random.randint(10000, 200000, n),
        'legal_complexity': np.random.randint(1, 6, n),
        'customer_tenure': np.random.randint(1, 11, n),
        'prior_claims': np.random.randint(0, 6, n),
        'fraud_probability': np.random.rand(n),
        'customer_value': np.random.randint(1, 4, n),
        'settled': np.random.choice([0, 1], size=n),
        'settlement_amount': np.random.randint(5000, 150000, n),
        'legal_cost': np.random.randint(2000, 50000, n),
        'satisfaction_score': np.random.uniform(0.2, 1.0, n)
    })

# ------------------ MODEL TRAINING ------------------ #
def train_models(data: pd.DataFrame) -> Tuple[RandomForestRegressor, RandomForestRegressor, RandomForestRegressor]:
    features = ['claim_amount', 'legal_complexity', 'customer_tenure', 'prior_claims', 'fraud_probability', 'customer_value']
    X = data[features]

    model_legal_cost = RandomForestRegressor().fit(X, data['legal_cost'])
    model_settlement = RandomForestRegressor().fit(X, data['settlement_amount'])
    model_satisfaction = RandomForestRegressor().fit(X, data['satisfaction_score'])

    return model_legal_cost, model_settlement, model_satisfaction

# ------------------ OPTIMIZATION LOGIC ------------------ #
def optimize_settlement(input_data: np.ndarray,
                        model_legal_cost: RandomForestRegressor,
                        model_satisfaction: RandomForestRegressor,
                        claim_amount: float,
                        customer_value: int) -> float:

    def objective(settlement_offer: float) -> float:
        predicted_legal_cost = model_legal_cost.predict(input_data)[0]
        predicted_satisfaction = model_satisfaction.predict(input_data)[0]
        retention_value = customer_value * Config.CUSTOMER_VALUE_MULTIPLIER
        total_cost = settlement_offer + predicted_legal_cost - (predicted_satisfaction * retention_value)
        return total_cost

    init_guess = claim_amount * Config.DEFAULT_SETTLEMENT_RATIO
    bounds = [(claim_amount * Config.MIN_SETTLEMENT_RATIO, claim_amount)]
    result = minimize(lambda x: objective(x[0]), x0=[init_guess], bounds=bounds)
    return float(result.x[0])

# ------------------ RECOMMENDATION ENGINE ------------------ #
def recommend_settlement(claim_input: Dict[str, Any],
                         models: Tuple[RandomForestRegressor, RandomForestRegressor, RandomForestRegressor]) -> Dict[str, Any]:

    model_legal_cost, model_settlement, model_satisfaction = models
    input_df = pd.DataFrame([claim_input])
    input_data = input_df.values

    predicted_legal_cost = model_legal_cost.predict(input_data)[0]
    predicted_settlement = model_settlement.predict(input_data)[0]
    predicted_satisfaction = model_satisfaction.predict(input_data)[0]

    optimized_settlement = optimize_settlement(
        input_data,
        model_legal_cost,
        model_satisfaction,
        claim_input['claim_amount'],
        claim_input['customer_value']
    )

    return {
        'Claim Amount': claim_input['claim_amount'],
        'Recommended Settlement': round(optimized_settlement, 2),
        'Predicted Legal Cost': round(predicted_legal_cost, 2),
        'Predicted Satisfaction': round(predicted_satisfaction, 2),
        'Strategy': 'Settle Early' if optimized_settlement < predicted_settlement else 'Negotiate or Litigate'
    }

# ------------------ MAIN EXECUTION ------------------ #
def main():
    data = generate_sample_data()
    models = train_models(data)

    example_claim = {
        'claim_amount': 85000,
        'legal_complexity': 4,
        'customer_tenure': 5,
        'prior_claims': 1,
        'fraud_probability': 0.25,
        'customer_value': 2
    }

    recommendation = recommend_settlement(example_claim, models)

    print("\n--- Claim Settlement Recommendation ---")
    for key, value in recommendation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
