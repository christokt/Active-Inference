"""
PLANNING MODULE - Active Inference Planning (Section IV of Paper)

Implements Expected Free Energy (EFE) minimization for action selection.
This is the planning stage where the agent chooses actions to minimize
expected free energy in the future.

Key equations from paper:
- Definition 3: Expected Free Energy G[π, τ]
- Equation 13: EFE computation
- Equation 14: Policy evaluation
- Theorem 1: Policy scaling with test-time computation
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from scipy.special import logsumexp
from perception_module import PerceptionModule, WorldModel, BeliefState


@dataclass
class ActionValue:
    """Information about an action's value"""
    action: int
    efe: float  # Expected Free Energy
    epistemic: float  # Information gain
    pragmatic: float  # Goal-directedness
    policy_prob: float  # Probability under optimal policy


@dataclass
class PolicyEvaluation:
    """Result of planning computation"""
    policy_efe: np.ndarray  # EFE for each policy (shape: num_policies)
    optimal_policy: int  # Index of best policy
    optimal_efe: float  # EFE of best policy
    action_values: List[ActionValue]  # Detailed action analysis
    policy_probabilities: np.ndarray  # Softmax over policies
    computation_time: float  # How many iterations to find policy


class PlanningModule:
    """
    Implements planning via Expected Free Energy minimization.
    
    Agent chooses actions/policies that minimize expected future surprise
    while maximizing information gain (epistemic) and goal-directedness (pragmatic).
    
    Attributes:
        world_model: The agent's generative model
        perception: Perception module for state inference
        inverse_temp: Inverse temperature for policy softmax
        horizon: Planning horizon (how many steps ahead)
        num_policies: Number of possible action policies
        verbose: Print debug information
    """
    
    def __init__(self, 
                 world_model: WorldModel,
                 perception: PerceptionModule,
                 inverse_temp: float = 16.0,
                 horizon: int = 3,
                 num_policies: int = None,
                 verbose: bool = False):
        """
        Initialize planning module.
        
        Args:
            world_model: Generative model W(o,s,π)
            perception: Perception module for inference
            inverse_temp: Inverse temperature for policy softmax (higher = sharper)
            horizon: Planning horizon τ (steps ahead)
            num_policies: Number of policies to evaluate (default: num_actions^horizon)
            verbose: Print debug info
        """
        self.wm = world_model
        self.perception = perception
        self.inverse_temp = inverse_temp
        self.horizon = horizon
        self.verbose = verbose
        
        # Number of possible actions per timestep
        self.num_actions = world_model.B.shape[1]  # B is (num_states, num_actions, num_states)
        
        # Total number of policies (all action sequences up to horizon)
        if num_policies is None:
            self.num_policies = self.num_actions ** horizon
        else:
            self.num_policies = num_policies
        
        if verbose:
            print(f"Planning Module initialized:")
            print(f"  Actions per step: {self.num_actions}")
            print(f"  Planning horizon: {horizon}")
            print(f"  Total policies: {self.num_policies}")
    
    def get_action_sequence(self, policy_idx: int) -> np.ndarray:
        """
        Convert policy index to action sequence.
        
        Args:
            policy_idx: Index of policy (0 to num_policies-1)
            
        Returns:
            Action sequence of length horizon
        """
        action_seq = np.zeros(self.horizon, dtype=int)
        for i in range(self.horizon):
            action_seq[i] = (policy_idx // (self.num_actions ** i)) % self.num_actions
        return action_seq
    
    def predict_future_states(self, 
                             current_belief: np.ndarray,
                             action_sequence: np.ndarray) -> np.ndarray:
        """
        Predict future state distribution under action sequence.
        
        Uses generative model B: p(s_{t+1} | s_t, a_t)
        
        Args:
            current_belief: Current belief distribution q(s_t)
            action_sequence: Actions to take [a_t, a_{t+1}, ...]
            
        Returns:
            State distribution after taking actions (shape: num_states)
        """
        # Start with current belief
        state_dist = current_belief.copy()
        
        # Apply each action in sequence
        for action in action_sequence:
            # B[s', a, s] = p(s' | s, a)
            # Expected next state: E[s'] = sum_s B(s', action, s) * q(s)
            action = int(action)
            if action >= self.wm.B.shape[1]:
                action = action % self.wm.B.shape[1]
            # B @ state_dist: shape (num_states, num_states) @ (num_states,) = (num_states,)
            state_dist = self.wm.B[:, action, :] @ state_dist
        
        return state_dist
    
    def predict_future_observations(self,
                                   future_state: np.ndarray) -> np.ndarray:
        """
        Predict observation distribution in future state.
        
        Uses generative model A: p(o | s)
        
        Args:
            future_state: State distribution
            
        Returns:
            Observation distribution (shape: num_observations)
        """
        # A[o, s] = p(o | s)
        # Expected observation: E[o] = sum_s p(s) * A(o, s)
        obs_dist = self.wm.A @ future_state
        return obs_dist
    
    def compute_epistemic_value(self,
                               future_obs_dist: np.ndarray) -> float:
        """
        Compute epistemic (information-seeking) value of observation distribution.
        
        Epistemic value = Information gain = Mutual information between
        observation and state given the action.
        
        Approximated as negative entropy of predicted observation:
        H[o] = -sum_o p(o) log p(o)
        
        Lower entropy = more informative observation = higher epistemic value
        
        Args:
            future_obs_dist: Predicted observation distribution
            
        Returns:
            Epistemic value (higher = more informative)
        """
        # Avoid log(0)
        obs_dist_safe = np.clip(future_obs_dist, 1e-16, 1.0)
        
        # Entropy (uncertainty) of observation distribution
        entropy = -np.sum(future_obs_dist * np.log(obs_dist_safe))
        
        # Epistemic value is negative entropy (neg uncertainty = info gain)
        epistemic_value = -entropy
        
        return epistemic_value
    
    def compute_pragmatic_value(self,
                               future_state: np.ndarray,
                               goal_prior: np.ndarray = None) -> float:
        """
        Compute pragmatic (goal-directedness) value of state distribution.
        
        Pragmatic value = How well the predicted state distribution
        aligns with goal/preferred states.
        
        Measured as divergence from goal distribution:
        G_prag = KL[p_goal || p_predicted_state]
        
        Args:
            future_state: Predicted state distribution
            goal_prior: Preferred state distribution (default: uniform = no preference)
            
        Returns:
            Pragmatic value (higher = more goal-aligned)
        """
        if goal_prior is None:
            # If no goal specified, states are equally preferred
            goal_prior = np.ones(self.wm.num_states) / self.wm.num_states
        
        # Normalize to ensure valid probability
        goal_prior = np.clip(goal_prior, 1e-16, 1.0)
        goal_prior = goal_prior / np.sum(goal_prior)
        future_state = np.clip(future_state, 1e-16, 1.0)
        future_state = future_state / np.sum(future_state)
        
        # KL divergence: KL[q || p] = sum_x q(x) log(q(x)/p(x))
        kl_div = np.sum(goal_prior * (np.log(goal_prior) - np.log(future_state)))
        
        # Pragmatic value is negative KL (lower divergence = better alignment)
        pragmatic_value = -kl_div
        
        return pragmatic_value
    
    def compute_expected_free_energy(self,
                                    current_belief: np.ndarray,
                                    action_sequence: np.ndarray,
                                    goal_prior: np.ndarray = None,
                                    epistemic_weight: float = 0.5) -> Tuple[float, float, float]:
        """
        Compute Expected Free Energy for an action sequence (policy).
        
        From Definition 3 (paper):
        G[π, τ] = E_s[E_o[F[q(s|o)]]] - λ_e * I[s:o] - λ_p * KL[p_goal || p(s)]
        
        Simplified version used here:
        G = -(epistemic_weight * epistemic_value + pragmatic_value)
        
        Args:
            current_belief: Current state distribution q(s_t)
            action_sequence: Actions to execute [a_t, ..., a_{t+H-1}]
            goal_prior: Preferred state distribution
            epistemic_weight: Balance between epistemic and pragmatic value
            
        Returns:
            (total_efe, epistemic_value, pragmatic_value)
        """
        # Predict state under this action sequence
        future_state = self.predict_future_states(current_belief, action_sequence)
        
        # Predict observations in that state
        future_obs = self.predict_future_observations(future_state)
        
        # Compute epistemic value (information gain)
        epistemic = self.compute_epistemic_value(future_obs)
        
        # Compute pragmatic value (goal-directedness)
        pragmatic = self.compute_pragmatic_value(future_state, goal_prior)
        
        # Expected Free Energy combines both
        # Lower EFE is better
        # We minimize: EFE = -epistemic_weight * epistemic - pragmatic
        efe = -(epistemic_weight * epistemic + pragmatic)
        
        return efe, epistemic, pragmatic
    
    def plan(self,
             current_belief: np.ndarray,
             goal_prior: np.ndarray = None,
             epistemic_weight: float = 0.5) -> PolicyEvaluation:
        """
        Execute planning: find optimal policy by minimizing Expected Free Energy.
        
        This implements Equation 14 from the paper:
        π* = argmin_π G[π, τ]
        
        Then converts to probability distribution using softmax:
        p(π) = exp(-β * G[π]) / Z  (Boltzmann distribution)
        
        Args:
            current_belief: Current belief q(s_t)
            goal_prior: Preferred state distribution (default: uniform)
            epistemic_weight: Trade-off between exploration and goal-seeking
            
        Returns:
            PolicyEvaluation with optimal policy and detailed analysis
        """
        if self.verbose:
            print(f"\nPlanning over {self.num_policies} policies...")
        
        # Evaluate all policies
        policy_efes = np.zeros(self.num_policies)
        action_values = []
        
        for policy_idx in range(self.num_policies):
            # Get action sequence for this policy
            actions = self.get_action_sequence(policy_idx)
            
            # Compute Expected Free Energy
            efe, epistemic, pragmatic = self.compute_expected_free_energy(
                current_belief, actions, goal_prior, epistemic_weight
            )
            
            policy_efes[policy_idx] = efe
            
            action_values.append(ActionValue(
                action=actions[0] if len(actions) > 0 else 0,
                efe=efe,
                epistemic=epistemic,
                pragmatic=pragmatic,
                policy_prob=0.0  # Will update after softmax
            ))
        
        # Find optimal policy
        optimal_idx = np.argmin(policy_efes)
        optimal_efe = policy_efes[optimal_idx]
        
        # Convert to probability distribution (softmax)
        # p(π) = exp(-β * G[π]) / Z
        policy_probs = np.exp(-self.inverse_temp * policy_efes)
        policy_probs = policy_probs / np.sum(policy_probs)
        
        # Update probabilities in action values
        for i, av in enumerate(action_values):
            av.policy_prob = policy_probs[i]
        
        if self.verbose:
            print(f"  Optimal policy: {optimal_idx}")
            print(f"  Optimal EFE: {optimal_efe:.4f}")
            print(f"  Top 3 policies:")
            top_indices = np.argsort(policy_efes)[:3]
            for idx in top_indices:
                print(f"    Policy {idx}: EFE={policy_efes[idx]:.4f}, "
                      f"prob={policy_probs[idx]:.4f}")
        
        return PolicyEvaluation(
            policy_efe=policy_efes,
            optimal_policy=optimal_idx,
            optimal_efe=optimal_efe,
            action_values=action_values,
            policy_probabilities=policy_probs,
            computation_time=self.num_policies  # Proxy for computation cost
        )
    
    def get_best_action(self,
                       current_belief: np.ndarray,
                       goal_prior: np.ndarray = None,
                       epistemic_weight: float = 0.5) -> int:
        """
        Planning shortcut: Get the best immediate action without full policy evaluation.
        
        Args:
            current_belief: Current belief
            goal_prior: Preferred states
            epistemic_weight: Exploration vs exploitation trade-off
            
        Returns:
            Best action to take now
        """
        policy_eval = self.plan(current_belief, goal_prior, epistemic_weight)
        optimal_policy_idx = policy_eval.optimal_policy
        optimal_actions = self.get_action_sequence(optimal_policy_idx)
        return optimal_actions[0]
    
    def get_action_distribution(self,
                               current_belief: np.ndarray,
                               goal_prior: np.ndarray = None,
                               epistemic_weight: float = 0.5) -> np.ndarray:
        """
        Get probability distribution over actions (not just best action).
        
        This implements soft decision-making where the agent might take
        suboptimal actions with some probability (exploration).
        
        Args:
            current_belief: Current belief
            goal_prior: Preferred states
            epistemic_weight: Exploration parameter
            
        Returns:
            Probability distribution over immediate actions
        """
        policy_eval = self.plan(current_belief, goal_prior, epistemic_weight)
        
        # Marginalize over policies to get action distribution
        action_probs = np.zeros(self.num_actions)
        
        for policy_idx in range(self.num_policies):
            actions = self.get_action_sequence(policy_idx)
            first_action = actions[0]
            action_probs[first_action] += policy_eval.policy_probabilities[policy_idx]
        
        return action_probs


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PLANNING MODULE - Expected Free Energy Minimization")
    print("="*70 + "\n")
    
    # Create a simple world model
    num_states = 3
    num_observations = 3
    
    # Likelihood matrix A: p(o|s)
    A = np.eye(num_states) * 0.85 + np.ones((num_states, num_observations)) * 0.15 / num_states
    
    # Transition matrix B: p(s'|s,a)
    # For simplicity: 2 actions (left/right)
    B = np.zeros((num_states, 2, num_states))
    for action in range(2):
        if action == 0:  # Move left
            for s in range(num_states):
                B[(s - 1) % num_states, action, s] = 0.9
                B[s, action, s] = 0.05
                B[(s + 1) % num_states, action, s] = 0.05
        else:  # Move right
            for s in range(num_states):
                B[(s + 1) % num_states, action, s] = 0.9
                B[s, action, s] = 0.05
                B[(s - 1) % num_states, action, s] = 0.05
    
    prior = np.ones(num_states) / num_states
    wm = WorldModel(A=A, B=B, prior_states=prior, 
                    num_states=num_states, num_observations=num_observations,
                    num_timesteps=5)
    
    # Create perception and planning modules
    from perception_module import PerceptionModule
    perception = PerceptionModule(wm, verbose=False)
    planning = PlanningModule(wm, perception, verbose=True)
    
    # Example 1: Neutral goal (no preference)
    print("\n" + "-"*70)
    print("TEST 1: Planning with Neutral Goal (Exploration)")
    print("-"*70 + "\n")
    
    current_belief = np.array([0.8, 0.1, 0.1])  # Agent thinks it's in state 0
    policy_eval = planning.plan(current_belief, goal_prior=None, epistemic_weight=0.7)
    
    print(f"\nBest action: {policy_eval.action_values[policy_eval.optimal_policy].action}")
    print(f"Action distribution: {planning.get_action_distribution(current_belief, epistemic_weight=0.7)}")
    
    # Example 2: Goal-directed (prefer state 2)
    print("\n" + "-"*70)
    print("TEST 2: Planning with Goal (State 2 Preferred)")
    print("-"*70 + "\n")
    
    goal = np.array([0.0, 0.0, 1.0])  # Prefer state 2
    policy_eval = planning.plan(current_belief, goal_prior=goal, epistemic_weight=0.3)
    
    print(f"\nBest action: {policy_eval.action_values[policy_eval.optimal_policy].action}")
    print(f"Action distribution: {planning.get_action_distribution(current_belief, goal_prior=goal, epistemic_weight=0.3)}")
    
    # Example 3: Varying epistemic weight
    print("\n" + "-"*70)
    print("TEST 3: Exploration vs Exploitation Trade-off")
    print("-"*70 + "\n")
    
    for epistemic_w in [0.1, 0.5, 0.9]:
        policy_eval = planning.plan(current_belief, epistemic_weight=epistemic_w)
        action_dist = planning.get_action_distribution(current_belief, epistemic_weight=epistemic_w)
        print(f"\nEpistemic weight {epistemic_w}:")
        print(f"  Action distribution: {action_dist}")
        print(f"  Entropy: {-np.sum(action_dist * np.log(action_dist + 1e-16)):.4f}")
    
    print("\n" + "="*70)
    print("PLANNING MODULE TESTS COMPLETE")
    print("="*70)
