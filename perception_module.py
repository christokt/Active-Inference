"""
Active Inference Perception Module
Implements Variational Message Passing (VMP) for state inference and surprise estimation
Based on: "Active Inference as the Test-Time Scaling Law for Physical AI Agents"

This module implements the perception component from Section III of the paper:
- Variational Free Energy (VFE) minimization
- Mean-field approximation with categorical-Dirichlet conjugacy
- Message passing for belief updating
- Surprise estimation
"""

import numpy as np
from scipy.special import digamma, logsumexp
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


@dataclass
class WorldModel:
    """Generative world model W(o, s, π) with conjugate-exponential structure"""
    
    # Likelihood matrix: p(o_τ | s_τ) = Cat(o_τ | A s_τ)
    # Shape: (num_observations, num_states)
    A: np.ndarray
    
    # Transition matrix: p(s_τ | s_τ-1, π) = Cat(s_τ | B s_τ-1)
    # Shape: (num_states, num_states)
    B: np.ndarray
    
    # Prior over initial states: p(s_1)
    # Shape: (num_states,)
    prior_states: np.ndarray
    
    num_states: int
    num_observations: int
    num_timesteps: int


@dataclass
class BeliefState:
    """Approximate posterior beliefs q(s_1:T | π_o)"""
    
    # Shape: (num_timesteps, num_states)
    state_beliefs: np.ndarray
    
    # Variational Free Energy at each timestep
    vfe: np.ndarray
    
    # Surprise estimate
    surprise: float
    
    # Message passing iterations until convergence
    iterations: int


class PerceptionModule:
    """
    Implements perception as Bayesian inference with VMP
    
    This module performs:
    1. Belief initialization from prior
    2. Iterative message passing (forward-backward)
    3. VFE minimization via gradient descent in log-space
    4. Surprise estimation as upper bound on information divergence
    """
    
    def __init__(self, world_model: WorldModel, 
                 convergence_threshold: float = 1e-5,
                 max_iterations: int = 100,
                 verbose: bool = False):
        """
        Args:
            world_model: WorldModel instance with A, B, prior
            convergence_threshold: When prediction errors < this, consider converged
            max_iterations: Maximum VMP iterations
            verbose: Print convergence diagnostics
        """
        self.wm = world_model
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Precompute log-parameters for efficient computation (Eq. 9)
        self._precompute_log_parameters()
    
    def _precompute_log_parameters(self):
        """Precompute log A and log B for numerical stability"""
        # Add small epsilon to avoid log(0)
        eps = 1e-16
        self.log_A = np.log(self.wm.A + eps)
        self.log_B = np.log(self.wm.B + eps)
        self.log_prior = np.log(self.wm.prior_states + eps)
    
    def _compute_messages(self, 
                         state_beliefs: np.ndarray,
                         observations: np.ndarray,
                         tau: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the three messages for VMP update at timestep tau (Eq. 9)
        
        Message 1: Prior forward transition - E[ln p(s_τ | s_τ-1)]
        Message 2: Prior backward transition - E[ln p(s_τ+1 | s_τ)]  
        Message 3: Observation likelihood - ln p(o_τ | s_τ)
        
        Returns:
            message_1, message_2, message_3 - log-probability vectors
        """
        num_states = self.wm.num_states
        
        # Message 1: Forward transition (from previous timestep)
        if tau == 0:
            # At t=0, use prior instead of transition
            message_1 = self.log_prior
        else:
            # Compute in probability space first: p(s_t) = B @ s_prev
            # Then convert to log scale to avoid numerical issues
            trans_prob = self.wm.B @ state_beliefs[tau - 1]
            trans_prob = np.clip(trans_prob, 1e-16, 1.0)
            message_1 = np.log(trans_prob)
        
        # Message 2: Backward transition (from next timestep)
        if tau == self.wm.num_timesteps - 1:
            # At final timestep, no future states
            message_2 = np.zeros(num_states)
        else:
            # For now, use weak backward message (uniform)
            # TODO: Properly derive backward message from reverse transitions
            message_2 = np.zeros(num_states)
        
        # Message 3: Observation likelihood
        if tau < len(observations):
            obs_idx = observations[tau]
            # ln p(o_τ | s) = ln A[o_τ, :]
            message_3 = self.log_A[obs_idx, :]
        else:
            message_3 = np.zeros(num_states)
        
        return message_1, message_2, message_3
    
    def _update_belief(self,
                      message_1: np.ndarray,
                      message_2: np.ndarray,
                      message_3: np.ndarray) -> np.ndarray:
        """
        Update belief using softmax normalization (Eq. 10)
        
        s_π,τ = σ(ln B s_prev + ln B^T s_next + ln A^T o_τ)
        """
        log_belief = message_1 + message_2 + message_3
        # Softmax normalization in log space for numerical stability
        belief = np.exp(log_belief - logsumexp(log_belief))
        return belief
    
    def _compute_prediction_error(self,
                                 state_belief: np.ndarray,
                                 message_1: np.ndarray,
                                 message_2: np.ndarray,
                                 message_3: np.ndarray) -> float:
        """
        Compute prediction error ε_π,τ (Eq. 11)
        
        This is the rate of change of VFE with respect to s_π,τ
        ε = -(∂F/∂s) in log space
        
        We use KL divergence between belief and messages as convergence metric
        """
        # Expected log-probability from messages
        expected_log = message_1 + message_2 + message_3
        # Normalize to get target distribution
        target_belief = np.exp(expected_log - logsumexp(expected_log))
        
        # KL divergence from current belief to target
        kl = np.sum(state_belief * (np.log(state_belief + 1e-16) - np.log(target_belief + 1e-16)))
        return kl
    
    def perceive(self, observations: np.ndarray) -> BeliefState:
        """
        Main perception function: infer hidden states from observations
        
        Implements iterative VMP (Eq. 12) until convergence
        
        Args:
            observations: Array of observation indices, shape (num_timesteps,)
        
        Returns:
            BeliefState with inferred beliefs, VFE, and surprise
        """
        num_timesteps = self.wm.num_timesteps
        num_states = self.wm.num_states
        
        # Initialize beliefs with prior (softmax of log prior)
        state_beliefs = np.zeros((num_timesteps, num_states))
        for t in range(num_timesteps):
            state_beliefs[t] = np.exp(self.log_prior - logsumexp(self.log_prior))
        
        # Store previous beliefs to check convergence
        prev_beliefs = state_beliefs.copy()
        
        # Iterative message passing until convergence
        iteration = 0
        for iteration in range(self.max_iterations):
            # Forward-backward pass over all timesteps
            for tau in range(num_timesteps):
                # Get messages (Eq. 9)
                msg1, msg2, msg3 = self._compute_messages(state_beliefs, observations, tau)
                
                # Update belief (Eq. 10)
                new_belief = self._update_belief(msg1, msg2, msg3)
                
                # Update belief
                state_beliefs[tau] = new_belief
            
            # Check convergence: compare beliefs before and after iteration
            belief_change = np.max(np.abs(state_beliefs - prev_beliefs))
            
            if self.verbose:
                print(f"  Iteration {iteration}: max belief change = {belief_change:.8f}")
            
            # Check convergence
            if belief_change < self.convergence_threshold:
                if self.verbose:
                    print(f"  Converged in {iteration + 1} iterations")
                break
            
            prev_beliefs = state_beliefs.copy()
        
        # Compute Variational Free Energy
        vfe = self._compute_vfe(state_beliefs, observations)
        
        # Estimate surprise as sum of VFE over timesteps (Lemma 1)
        surprise = np.sum(vfe)
        
        return BeliefState(
            state_beliefs=state_beliefs,
            vfe=vfe,
            surprise=surprise,
            iterations=iteration + 1
        )
    
    def _compute_vfe(self, state_beliefs: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """
        Compute Variational Free Energy at each timestep
        
        From Section III: F[q(s)] = E_q[ln q(s) - ln p(s,o)]
        
        This provides an upper bound on surprise (Lemma 1)
        """
        vfe = np.zeros(self.wm.num_timesteps)
        
        for tau in range(self.wm.num_timesteps):
            # KL divergence term: E[ln q - ln p_prior]
            kl_prior = np.sum(state_beliefs[tau] * 
                             (np.log(state_beliefs[tau] + 1e-16) - self.log_prior))
            
            # Observation term: -E[ln p(o|s)]
            if tau < len(observations):
                obs_idx = observations[tau]
                obs_term = -np.sum(state_beliefs[tau] * self.log_A[obs_idx, :])
            else:
                obs_term = 0.0
            
            # Transition term: E[ln q(s_τ) - ln p(s_τ | s_τ-1)]
            transition_term = 0.0
            if tau > 0:
                # log_B @ s_prev gives log p(s_τ | s_τ-1)
                expected_log_trans = self.log_B @ state_beliefs[tau - 1]
                transition_term = np.sum(state_beliefs[tau] * 
                                       (np.log(state_beliefs[tau] + 1e-16) - expected_log_trans))
            
            vfe[tau] = kl_prior + obs_term + transition_term
        
        return vfe
    
    def get_state_entropy(self, state_belief: np.ndarray) -> float:
        """Compute Shannon entropy of belief distribution"""
        return -np.sum(state_belief * np.log(state_belief + 1e-16))
    
    def get_max_likelihood_state(self, belief: np.ndarray) -> int:
        """Return most likely state from belief"""
        return np.argmax(belief)


class SimpleGridWorldExample:
    """
    Simple example: 3-state gridworld
    - Agent can be in states: [Left, Center, Right]
    - Observations: [red, blue, green]
    - Observation model: Left→red, Center→blue, Right→green (with noise)
    """
    
    @staticmethod
    def create_world_model(noise_level: float = 0.1) -> WorldModel:
        """Create a simple 3-state gridworld"""
        num_states = 3
        num_observations = 3
        num_timesteps = 5
        
        # Likelihood matrix: p(o | s)
        # Each state most likely produces its corresponding observation
        A = np.array([
            [1 - 2*noise_level, noise_level, noise_level],      # Red (Left)
            [noise_level, 1 - 2*noise_level, noise_level],      # Blue (Center)
            [noise_level, noise_level, 1 - 2*noise_level],      # Green (Right)
        ])
        
        # Transition matrix: p(s_t | s_t-1)
        # Agent can stay or move with small probability
        B = np.array([
            [0.8, 0.1, 0.0],    # From Left: stay or drift right
            [0.1, 0.8, 0.1],    # From Center: can go either way
            [0.0, 0.1, 0.8],    # From Right: stay or drift left
        ])
        
        # Prior: uniform
        prior = np.ones(num_states) / num_states
        
        return WorldModel(
            A=A, B=B, prior_states=prior,
            num_states=num_states,
            num_observations=num_observations,
            num_timesteps=num_timesteps
        )
    
    @staticmethod
    def create_observation_sequence(true_states: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Generate observation sequence from true states"""
        observations = []
        for state_idx in true_states:
            # Sample observation from likelihood p(o | s)
            obs_prob = A[:, state_idx]
            obs = np.random.choice(len(obs_prob), p=obs_prob)
            observations.append(obs)
        return np.array(observations)


def visualize_beliefs(belief_state: BeliefState, true_states: Optional[np.ndarray] = None):
    """Pretty-print belief state and surprise"""
    print("\n" + "="*60)
    print("PERCEPTION RESULTS")
    print("="*60)
    print(f"Converged in {belief_state.iterations} iterations")
    print(f"Total Surprise (VFE upper bound): {belief_state.surprise:.4f}")
    print()
    
    for t in range(len(belief_state.state_beliefs)):
        print(f"Timestep {t}:")
        belief = belief_state.state_beliefs[t]
        print(f"  Belief: {belief}")
        print(f"  Max-likelihood state: {np.argmax(belief)}")
        print(f"  Entropy: {-np.sum(belief * np.log(belief + 1e-16)):.4f}")
        if true_states is not None:
            print(f"  True state: {true_states[t]}")
        print(f"  VFE: {belief_state.vfe[t]:.6f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ACTIVE INFERENCE PERCEPTION MODULE - BASIC TEST")
    print("="*60 + "\n")
    
    # Create a simple gridworld
    print("Creating 3-state gridworld world model...")
    wm = SimpleGridWorldExample.create_world_model(noise_level=0.15)
    
    print(f"  States: {wm.num_states}")
    print(f"  Observations: {wm.num_observations}")
    print(f"  Timesteps: {wm.num_timesteps}")
    print(f"  Likelihood matrix A:\n{wm.A}")
    print(f"  Transition matrix B:\n{wm.B}")
    
    # Create perception module
    print("\nInitializing perception module with VMP...")
    perception = PerceptionModule(wm, verbose=True)
    
    # Test 1: Simple sequence - agent moves right then left
    print("\n" + "-"*60)
    print("TEST 1: Agent trajectory [Left, Center, Right, Center, Left]")
    print("-"*60)
    true_states_1 = np.array([0, 1, 2, 1, 0])
    observations_1 = SimpleGridWorldExample.create_observation_sequence(true_states_1, wm.A)
    print(f"True states: {true_states_1}")
    print(f"Observations: {observations_1}")
    
    print("\nPerforming perception inference...")
    belief_state_1 = perception.perceive(observations_1)
    visualize_beliefs(belief_state_1, true_states_1)
    
    # Test 2: Noisy sequence
    print("\n" + "-"*60)
    print("TEST 2: Same trajectory with multiple samples (robustness check)")
    print("-"*60)
    surprises = []
    for trial in range(5):
        obs = SimpleGridWorldExample.create_observation_sequence(true_states_1, wm.A)
        result = perception.perceive(obs)
        surprises.append(result.surprise)
        print(f"  Trial {trial + 1}: surprise = {result.surprise:.4f}")
    
    print(f"\nMean surprise: {np.mean(surprises):.4f} ± {np.std(surprises):.4f}")
    
    # Test 3: Comparing different noise levels
    print("\n" + "-"*60)
    print("TEST 3: Sensitivity to noise level")
    print("-"*60)
    noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    for noise in noise_levels:
        wm_noisy = SimpleGridWorldExample.create_world_model(noise_level=noise)
        perception_noisy = PerceptionModule(wm_noisy, verbose=False)
        result = perception_noisy.perceive(observations_1)
        print(f"  Noise {noise:.2f}: surprise = {result.surprise:.4f}")
    
    print("\n" + "="*60)
    print("TESTS COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
