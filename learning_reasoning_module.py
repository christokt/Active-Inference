"""
LEARNING & REASONING MODULE - Causal Active Inference

Implements learning through parameter updates and causal reasoning.
The agent learns model parameters and performs counterfactual reasoning.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class LearningResult:
    """Result of learning process"""
    updated_A: np.ndarray
    updated_B: np.ndarray
    learning_rate: float
    model_evidence: float


class LearningModule:
    """
    Bayesian learning of model parameters.
    Uses Dirichlet-Categorical conjugate prior framework.
    """
    
    def __init__(self, world_model, alpha_a=1.0, alpha_b=1.0, 
                 learning_rate=0.1, verbose=False):
        """
        Initialize learning module.
        
        Args:
            world_model: The generative model to learn
            alpha_a: Prior concentration for likelihood A
            alpha_b: Prior concentration for transition B
            learning_rate: How fast to update (0-1)
            verbose: Print debug information
        """
        self.wm = world_model
        self.alpha_a = alpha_a
        self.alpha_b = alpha_b
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # Initialize count matrices (Dirichlet parameters)
        self.count_a = np.ones_like(world_model.A) * alpha_a
        self.count_b = np.ones_like(world_model.B) * alpha_b
        
        if verbose:
            print(f"Learning Module initialized:")
            print(f"  Prior α_A: {alpha_a}, α_B: {alpha_b}")
            print(f"  Learning rate: {learning_rate}")
    
    def update_likelihood_model(self, observations, beliefs):
        """
        Update likelihood matrix A: p(o|s).
        
        Given observed outcomes and inferred states, update beliefs about
        which states generate which observations.
        
        Args:
            observations: Sequence of observations
            beliefs: Posterior state beliefs
            
        Returns:
            Updated likelihood matrix A
        """
        # Count how many times each (o,s) pair occurred
        for t in range(len(observations)):
            obs = int(observations[t])
            state_belief = beliefs[t]  # Soft count
            self.count_a[obs, :] += self.learning_rate * state_belief
        
        # Normalize to probabilities: A[o,s] = count[o,s] / sum_o count[o,s]
        updated_a = self.count_a / (np.sum(self.count_a, axis=0, keepdims=True) + 1e-16)
        return updated_a
    
    def update_transition_model(self, states, actions, beliefs):
        """
        Update transition matrix B: p(s'|s,a).
        
        Given observed state transitions and actions, update beliefs
        about environment dynamics.
        
        Args:
            states: Sequence of inferred states
            actions: Sequence of taken actions
            beliefs: Posterior state beliefs
            
        Returns:
            Updated transition matrix B
        """
        # Count state transitions under each action
        for t in range(len(states) - 1):
            action = int(actions[t])
            belief_current = beliefs[t]
            belief_next = beliefs[t + 1]
            
            # Soft update using beliefs
            for s in range(self.wm.num_states):
                for s_prime in range(self.wm.num_states):
                    self.count_b[s_prime, action, s] += \
                        self.learning_rate * belief_current[s] * belief_next[s_prime]
        
        # Normalize: B[s',a,s] = count[s',a,s] / sum_s' count[s',a,s]
        updated_b = np.zeros_like(self.wm.B)
        for a in range(self.wm.B.shape[1]):
            for s in range(self.wm.num_states):
                denom = np.sum(self.count_b[:, a, s]) + 1e-16
                updated_b[:, a, s] = self.count_b[:, a, s] / denom
        
        return updated_b
    
    def learn(self, observations, actions, beliefs):
        """
        Execute learning: update both A and B based on experience.
        
        Args:
            observations: Observed outcomes
            actions: Taken actions
            beliefs: Inferred state beliefs
            
        Returns:
            LearningResult with updated models
        """
        if self.verbose:
            print(f"\nLearning from {len(observations)} observations...")
        
        # Update both models
        updated_a = self.update_likelihood_model(observations, beliefs.state_beliefs)
        inferred_states = np.argmax(beliefs.state_beliefs, axis=0)
        updated_b = self.update_transition_model(inferred_states, actions, beliefs.state_beliefs)
        
        # Model evidence proxy
        model_evidence = -beliefs.surprise
        
        if self.verbose:
            print(f"  Model evidence: {model_evidence:.4f}")
            print(f"  Learning complete!")
        
        return LearningResult(
            updated_A=updated_a,
            updated_B=updated_b,
            learning_rate=self.learning_rate,
            model_evidence=model_evidence
        )


class ReasoningModule:
    """
    Implements causal reasoning and counterfactual inference.
    
    Can answer questions like:
    - "What would happen if I took action X?"
    - "Why did action Y lead to outcome Z?"
    - "What's the causal relationship?"
    """
    
    def __init__(self, world_model, verbose=False):
        """
        Initialize reasoning module.
        
        Args:
            world_model: The generative model
            verbose: Print debug information
        """
        self.wm = world_model
        self.verbose = verbose
    
    def counterfactual_inference(self, current_state, action, num_steps=1):
        """
        Counterfactual reasoning: What would happen if I took action X from state S?
        
        Uses the transition model B to predict future states.
        
        Args:
            current_state: Starting state
            action: Action to consider
            num_steps: How many steps into the future
            
        Returns:
            Distribution over likely future states
        """
        # Start with deterministic current state
        state_dist = np.zeros(self.wm.num_states)
        state_dist[current_state] = 1.0
        
        # Rollout forward under the action
        for step in range(num_steps):
            # Apply transition: s' ~ B(s', a, s)
            state_dist = self.wm.B[:, action, :].T @ state_dist
        
        # Convert to dictionary
        result = {state: float(prob) for state, prob in enumerate(state_dist)}
        
        if self.verbose:
            print(f"Counterfactual: If I take action {action} from state {current_state}")
            print(f"  for {num_steps} steps: {result}")
        
        return result
    
    def causal_attribution(self, action, observation):
        """
        Causal attribution: How much did action X cause observation O?
        
        Measures causal strength using model structure.
        
        Args:
            action: The potential cause (action)
            observation: The effect (observation)
            
        Returns:
            Causal strength (0-1, higher = stronger causality)
        """
        # Measure state change from action
        state_effect = np.sum(np.abs(self.wm.B[:, action, :] - np.eye(self.wm.num_states)))
        
        # States that cause this observation
        obs_states = np.where(np.argmax(self.wm.A, axis=1) == observation)[0]
        obs_prob = np.sum(self.wm.A[observation, obs_states])
        
        # Causal strength: action effect × outcome probability
        causal_strength = float(np.clip(state_effect * obs_prob, 0, 1))
        
        if self.verbose:
            print(f"Causal attribution: action {action} -> observation {observation}")
            print(f"  Causal strength: {causal_strength:.3f}")
        
        return causal_strength
    
    def explain_outcome(self, observation, state):
        """
        Explanation: Why did I observe O in state S?
        
        Provides natural language explanation of an outcome.
        
        Args:
            observation: The observed outcome
            state: The inferred state
            
        Returns:
            Natural language explanation
        """
        obs_likelihood = self.wm.A[observation, state]
        likely_states = np.argsort(self.wm.A[observation, :])[::-1][:3]
        
        explanation = f"Observed {observation} (likely from states {list(likely_states)}). "
        explanation += f"P(obs|state {state}) = {obs_likelihood:.3f}. "
        explanation += "This was expected." if obs_likelihood > 0.5 else "This was surprising."
        
        return explanation
    
    def contrastive_explanation(self, actual_obs, counterfactual_obs, state):
        """
        Contrastive explanation: Why O1 instead of O2?
        
        Compares two possible outcomes and explains the difference.
        
        Args:
            actual_obs: What actually happened
            counterfactual_obs: What could have happened
            state: The state we were in
            
        Returns:
            Explanation of the difference
        """
        actual_prob = self.wm.A[actual_obs, state]
        counterfactual_prob = self.wm.A[counterfactual_obs, state]
        ratio = actual_prob / (counterfactual_prob + 1e-16)
        
        explanation = f"In state {state}: observed {actual_obs} (prob {actual_prob:.3f}) "
        explanation += f"instead of {counterfactual_obs} (prob {counterfactual_prob:.3f}). "
        explanation += f"Likelihood ratio: {ratio:.2f}x more likely."
        
        return explanation


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("LEARNING & REASONING MODULES - DEMONSTRATION")
    print("="*70 + "\n")
    
    from perception_module import SimpleGridWorldExample, PerceptionModule
    
    # Create world model
    wm = SimpleGridWorldExample.create_world_model(noise_level=0.1)
    perception = PerceptionModule(wm, verbose=False)
    learning = LearningModule(wm, verbose=True)
    reasoning = ReasoningModule(wm, verbose=True)
    
    # Learning example
    print("\n" + "-"*70)
    print("TEST 1: Learning from Observations")
    print("-"*70 + "\n")
    
    true_states = np.array([0, 1, 2, 1, 0])
    observations = SimpleGridWorldExample.create_observation_sequence(true_states, wm.A)
    actions = np.array([0, 1, 1, 0])
    
    belief_state = perception.perceive(observations)
    learning_result = learning.learn(observations, actions, belief_state)
    
    print(f"\nLearning Results:")
    print(f"  Model evidence: {learning_result.model_evidence:.4f}")
    print(f"  Updated A shape: {learning_result.updated_A.shape}")
    print(f"  Updated B shape: {learning_result.updated_B.shape}")
    
    # Counterfactual reasoning
    print("\n" + "-"*70)
    print("TEST 2: Counterfactual Reasoning")
    print("-"*70 + "\n")
    
    outcomes = reasoning.counterfactual_inference(current_state=0, action=0, num_steps=2)
    
    # Causal attribution
    print("\n" + "-"*70)
    print("TEST 3: Causal Attribution")
    print("-"*70 + "\n")
    
    for action in range(min(2, wm.B.shape[1])):
        for obs in range(min(3, wm.num_observations)):
            strength = reasoning.causal_attribution(action, obs)
    
    # Natural language explanations
    print("\n" + "-"*70)
    print("TEST 4: Natural Language Explanations")
    print("-"*70 + "\n")
    
    for obs in range(min(3, wm.num_observations)):
        explanation = reasoning.explain_outcome(obs, state=0)
        print(f"  {explanation}")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
