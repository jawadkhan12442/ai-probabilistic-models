# Project 2: Bayesian Belief Network
# Author: Jawad Ahmad Khan
# Purpose: Estimate the probability of wet grass using rain and sprinkler evidence.

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Create the network structure
model = DiscreteBayesianNetwork([
    ("Rain", "WetGrass"),
    ("Sprinkler", "WetGrass")
])

# Probability that it is raining
cpd_rain = TabularCPD(
    variable="Rain",
    variable_card=2,
    values=[[0.8], [0.2]]
)

# Probability that the sprinkler is on
cpd_sprinkler = TabularCPD(
    variable="Sprinkler",
    variable_card=2,
    values=[[0.6], [0.4]]
)

# Probability of wet grass for each combination of rain and sprinkler
cpd_wet_grass = TabularCPD(
    variable="WetGrass",
    variable_card=2,
    values=[
        [0.99, 0.80, 0.90, 0.01],
        [0.01, 0.20, 0.10, 0.99]
    ],
    evidence=["Rain", "Sprinkler"],
    evidence_card=[2, 2]
)

# Add probability tables to the network
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet_grass)

# Check the model
assert model.check_model(), "The Bayesian network is invalid."

# Perform probabilistic inference
inference = VariableElimination(model)

# Calculate probability of wet grass when rain is observed
result = inference.query(
    variables=["WetGrass"],
    evidence={"Rain": 1}
)

print("Bayesian Belief Network: Rain and Wet Grass")
print(result)
print("\nInterpretation:")
print("The probabilities show whether the grass is wet given that it is raining.")
