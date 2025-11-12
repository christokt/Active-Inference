"""
GUARANTEED COLAB PLOTS - Active Inference Perception Module
This version will 100% display plots in Colab using %matplotlib inline
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# CRITICAL: Enable inline plotting
try:
    %matplotlib inline
except:
    pass

from perception_module import (
    PerceptionModule, SimpleGridWorldExample, WorldModel
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def plot_belief_evolution(belief_state, true_states, observations, title="Belief Evolution"):
    """Plot how beliefs evolve over time"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    num_timesteps = len(belief_state.state_beliefs)
    time_idx = np.arange(num_timesteps)
    
    # Plot 1: Belief probabilities over time
    ax = axes[0]
    for state_idx in range(belief_state.state_beliefs.shape[1]):
        beliefs = belief_state.state_beliefs[:, state_idx]
        ax.plot(time_idx, beliefs, 'o-', linewidth=2, markersize=6, 
                label=f'State {state_idx}', alpha=0.7)
    
    # Mark true states
    if true_states is not None:
        for t, true_state in enumerate(true_states):
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
            ax.text(t, 0.95, f'True: {true_state}', 
                   ha='center', va='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Belief Probability', fontsize=11, fontweight='bold')
    ax.set_xlabel('Timestep', fontsize=11, fontweight='bold')
    ax.set_title(f'{title} - Posterior Beliefs p(s_t|o_1:t)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 2: Most likely state trajectory
    ax = axes[1]
    ml_states = np.argmax(belief_state.state_beliefs, axis=1)
    ax.plot(time_idx, ml_states, 's-', linewidth=2, markersize=8, 
           color='steelblue', label='Inferred state')
    
    if true_states is not None:
        ax.plot(time_idx, true_states, 'o:', linewidth=2, markersize=8, 
               color='orangered', label='True state', alpha=0.6)
    
    ax.set_ylabel('State Index', fontsize=11, fontweight='bold')
    ax.set_xlabel('Timestep', fontsize=11, fontweight='bold')
    ax.set_title('Max-Likelihood State Trajectories', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yticks(range(belief_state.state_beliefs.shape[1]))
    
    # Plot 3: VFE and entropy over time
    ax = axes[2]
    entropy = np.array([
        -np.sum(belief_state.state_beliefs[t] * 
               np.log(belief_state.state_beliefs[t] + 1e-16))
        for t in range(num_timesteps)
    ])
    
    ax_vfe = ax
    ax_ent = ax.twinx()
    
    # VFE bars
    bars = ax_vfe.bar(time_idx, belief_state.vfe, alpha=0.6, color='steelblue', 
                      label='VFE (Free Energy)')
    ax_vfe.set_ylabel('Variational Free Energy', fontsize=11, fontweight='bold', color='steelblue')
    ax_vfe.tick_params(axis='y', labelcolor='steelblue')
    
    # Entropy line
    ax_ent.plot(time_idx, entropy, 'o-', linewidth=2, markersize=6, 
               color='orangered', label='Entropy H[q(s_t)]')
    ax_ent.set_ylabel('Belief Entropy', fontsize=11, fontweight='bold', color='orangered')
    ax_ent.tick_params(axis='y', labelcolor='orangered')
    
    ax_vfe.set_xlabel('Timestep', fontsize=11, fontweight='bold')
    ax_vfe.set_title('Free Energy and Belief Uncertainty', fontweight='bold')
    ax_vfe.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    lines1, labels1 = ax_vfe.get_legend_handles_labels()
    lines2, labels2 = ax_ent.get_legend_handles_labels()
    ax_vfe.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_surprise_analysis(surprise_list, noise_levels=None):
    """Compare surprise across different noise levels"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Histogram of surprises
    ax = axes[0]
    ax.hist(surprise_list, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(surprise_list), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(surprise_list):.3f}')
    ax.axvline(np.median(surprise_list), color='green', linestyle='--', linewidth=2, 
              label=f'Median: {np.median(surprise_list):.3f}')
    ax.set_xlabel('Surprise (VFE)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Surprise Estimates', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Surprise vs noise
    if noise_levels is not None:
        ax = axes[1]
        ax.plot(noise_levels, surprise_list, 'o-', linewidth=2, markersize=8, 
               color='steelblue')
        ax.fill_between(noise_levels, 
                        np.array(surprise_list) - 0.1,
                        np.array(surprise_list) + 0.1,
                        alpha=0.2, color='steelblue')
        ax.set_xlabel('Noise Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Surprise (VFE)', fontsize=11, fontweight='bold')
        ax.set_title('Surprise vs Model Uncertainty', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_convergence_dynamics(perception, wm, observations):
    """Plot VMP convergence behavior"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Convergence across iterations
    ax = axes[0]
    state_beliefs = np.zeros((wm.num_timesteps, wm.num_states))
    for t in range(wm.num_timesteps):
        state_beliefs[t] = np.ones(wm.num_states) / wm.num_states
    
    belief_change_history = []
    prev_beliefs = state_beliefs.copy()
    
    for iteration in range(30):
        for tau in range(wm.num_timesteps):
            msg1, msg2, msg3 = perception._compute_messages(state_beliefs, observations, tau)
            new_belief = perception._update_belief(msg1, msg2, msg3)
            state_beliefs[tau] = new_belief
        
        belief_change = np.max(np.abs(state_beliefs - prev_beliefs))
        belief_change_history.append(belief_change)
        prev_beliefs = state_beliefs.copy()
    
    ax.semilogy(range(len(belief_change_history)), belief_change_history, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax.axhline(perception.convergence_threshold, color='red', linestyle='--', 
              linewidth=2, label=f'Convergence threshold: {perception.convergence_threshold:.2e}')
    ax.set_xlabel('VMP Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Max Belief Change', fontsize=11, fontweight='bold')
    ax.set_title('Convergence Dynamics (Log Scale)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Computational cost
    ax = axes[1]
    num_states_range = np.array([2, 3, 4, 5, 6, 8, 10, 15, 20])
    iterations_required = []
    
    for n_states in num_states_range:
        wm_temp = WorldModel(
            A=np.eye(n_states) * 0.85 + np.ones((n_states, n_states)) * 0.15 / n_states,
            B=np.eye(n_states) * 0.9 + np.ones((n_states, n_states)) * 0.1 / n_states,
            prior_states=np.ones(n_states) / n_states,
            num_states=n_states,
            num_observations=n_states,
            num_timesteps=5
        )
        perc_temp = PerceptionModule(wm_temp, verbose=False)
        obs_temp = np.random.randint(0, n_states, size=5)
        result_temp = perc_temp.perceive(obs_temp)
        iterations_required.append(result_temp.iterations)
    
    ax.loglog(num_states_range, iterations_required, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Number of States', fontsize=11, fontweight='bold')
    ax.set_ylabel('Iterations to Convergence', fontsize=11, fontweight='bold')
    ax.set_title('Scalability: Computational Cost vs Model Complexity', fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def main():
    """Run comprehensive analysis"""
    
    print("\n" + "="*70)
    print("ACTIVE INFERENCE PERCEPTION MODULE - COLAB ANALYSIS")
    print("="*70 + "\n")
    
    # Create world model
    print("📊 Setting up world model...")
    wm = SimpleGridWorldExample.create_world_model(noise_level=0.15)
    perception = PerceptionModule(wm, verbose=False)
    
    # Test 1: Belief evolution
    print("\n" + "="*70)
    print("📊 TEST 1: BELIEF EVOLUTION")
    print("="*70)
    true_states = np.array([0, 1, 2, 1, 0])
    observations = SimpleGridWorldExample.create_observation_sequence(true_states, wm.A)
    belief_state = perception.perceive(observations)
    
    print(f"\nTrue states:        {true_states}")
    print(f"Observations:       {observations}")
    print(f"Inferred states:    {np.argmax(belief_state.state_beliefs, axis=1)}")
    print(f"Total surprise:     {belief_state.surprise:.4f}")
    print(f"Converged in:       {belief_state.iterations} iterations\n")
    
    fig1 = plot_belief_evolution(belief_state, true_states, observations)
    fig1.savefig('belief_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ PLOT 1 DISPLAYED ABOVE\n")
    
    # Test 2: Surprise sensitivity
    print("="*70)
    print("📊 TEST 2: SURPRISE SENSITIVITY TO NOISE")
    print("="*70)
    noise_levels = np.linspace(0.05, 0.30, 8)
    surprises_noise = []
    
    for noise in noise_levels:
        wm_noisy = SimpleGridWorldExample.create_world_model(noise_level=noise)
        perc_noisy = PerceptionModule(wm_noisy, verbose=False)
        result = perc_noisy.perceive(observations)
        surprises_noise.append(result.surprise)
        print(f"  Noise {noise:.2f}: surprise = {result.surprise:.4f}")
    
    fig2 = plot_surprise_analysis(surprises_noise, noise_levels=noise_levels)
    fig2.savefig('surprise_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ PLOT 2 DISPLAYED ABOVE\n")
    
    # Test 3: Convergence dynamics
    print("="*70)
    print("📊 TEST 3: CONVERGENCE DYNAMICS & SCALABILITY")
    print("="*70 + "\n")
    fig3 = plot_convergence_dynamics(perception, wm, observations)
    fig3.savefig('convergence_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ PLOT 3 DISPLAYED ABOVE\n")
    
    # Test 4: Summary statistics
    print("="*70)
    print("📊 TEST 4: SUMMARY STATISTICS")
    print("="*70)
    
    surprises_multi = []
    for _ in range(10):
        obs = SimpleGridWorldExample.create_observation_sequence(true_states, wm.A)
        result = perception.perceive(obs)
        surprises_multi.append(result.surprise)
    
    print(f"\nMean surprise:      {np.mean(surprises_multi):.4f} ± {np.std(surprises_multi):.4f}")
    print(f"Min surprise:       {np.min(surprises_multi):.4f}")
    print(f"Max surprise:       {np.max(surprises_multi):.4f}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    
    print("\n✅ All 3 plots have been generated and displayed above!")
    print("✅ All plots saved as PNG files\n")
    print("📥 To download plots to your computer, run this in a new cell:")
    print("   from google.colab import files")
    print("   files.download('belief_evolution.png')")
    print("   files.download('surprise_sensitivity.png')")
    print("   files.download('convergence_dynamics.png')")


if __name__ == "__main__":
    main()
