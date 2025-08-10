# NECA: Neuro-Emotive Cognitive Architecture
# This script is a refactored and improved version, combining the best features
# from the provided snippets. It includes:
# - A clear, modular architecture with docstrings and type hinting.
# - An efficient forward pass that avoids redundant computations.
# - A persistent, trainable projection layer within the HRM.
# - Integration with a Prioritized Experience Replay (PER) buffer for
#   more effective continuous learning.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from typing import Dict, Tuple, List, Optional

# ==============================================================================
# 1. HELPER LAYERS
# ==============================================================================

class FilMLayer(layers.Layer):
    """
    A Feature-wise Linear Modulation (FiLM) layer.
    Modulates a feature map using a conditioning vector to produce gamma and beta
    parameters for affine transformation.
    """
    def __init__(self, **kwargs):
        super(FilMLayer, self).__init__(**kwargs)
        print("✅ FiLM Layer instantiated.")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Args:
            inputs: A tuple containing:
                - features (tf.Tensor): The input features to be modulated.
                - gamma_beta (tf.Tensor): The conditioning vector for modulation.

        Returns:
            tf.Tensor: The modulated features.
        """
        features, gamma_beta = inputs
        # Dynamically determine the feature dimension for gamma and beta
        feat_dim = tf.shape(features)[-1]
        
        # Extract gamma and beta from the conditioning vector
        gamma = gamma_beta[:, :feat_dim]
        beta = gamma_beta[:, feat_dim:]
        
        # The feature vector needs to be rank 2 (batch, features)
        # Gamma and beta are also rank 2.
        # Add dimensions if needed for broadcasting, though dense layers output rank 2.
        
        return (gamma * features) + beta

class HRMProjection(layers.Layer):
    """
    Encapsulates the query and value projection logic for the HRM module.
    This makes the HRM module cleaner and re-uses these layers.
    """
    def __init__(self, proj_dim: int = 32, **kwargs):
        super(HRMProjection, self).__init__(**kwargs)
        self.query_proj = layers.Dense(proj_dim, activation='relu', name="HRM_Query_Proj")
        # Use TimeDistributed to apply the dense layer to each time step of the value sequence.
        self.value_proj = layers.TimeDistributed(
            layers.Dense(proj_dim, activation='relu'), name="HRM_Value_Proj"
        )
        print(f"✅ HRMProjection instantiated with projection dim {proj_dim}.")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            inputs: A tuple containing:
                - query_vector (tf.Tensor): The query vector for attention.
                - value_sequence (tf.Tensor): The sequence to attend over.

        Returns:
            A tuple of (projected_query, projected_value).
        """
        query_vector, value_sequence = inputs
        # Project query and add a time dimension for attention compatibility.
        projected_query = self.query_proj(query_vector)
        projected_query_expanded = tf.expand_dims(projected_query, axis=1) # (batch, 1, proj_dim)
        
        # Project the value sequence across its time dimension.
        projected_value = self.value_proj(value_sequence) # (batch, time, proj_dim)
        
        return projected_query_expanded, projected_value

# ==============================================================================
# 2. CORE NECA MODULES
# ==============================================================================

class InteroceptiveStateModule(Model):
    """
    Processes internal physiological states to generate a core affect representation.
    Neural Analogue: Insula, Homeostatic Systems.
    """
    def __init__(self, core_affect_dim: int = 2, **kwargs):
        super(InteroceptiveStateModule, self).__init__(**kwargs)
        self.lstm = layers.LSTM(16, name="ISM_LSTM")
        self.dense_output = layers.Dense(core_affect_dim, name="ISM_Core_Affect")
        print("✅ ISM (Analogue: Insula) instantiated.")

    def call(self, internal_state: tf.Tensor) -> tf.Tensor:
        state_reshaped = tf.expand_dims(internal_state, axis=1)
        lstm_out = self.lstm(state_reshaped)
        core_affect = self.dense_output(lstm_out)
        return core_affect

class ReactivePathway(Model):
    """
    A fast, low-resolution pathway for immediate threat/salience detection.
    Neural Analogue: Thalamo-Amygdala "Low Road".
    """
    def __init__(self, **kwargs):
        super(ReactivePathway, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(16, (3, 3), activation='relu', name="Reactive_Conv1")
        self.pool1 = layers.MaxPooling2D((2, 2), name="Reactive_Pool1")
        self.flatten = layers.Flatten(name="Reactive_Flatten")
        self.dense_output = layers.Dense(1, activation='sigmoid', name="Reactive_Salience_Output")
        print("✅ ReactivePathway (Analogue: Amygdala Low-Road) instantiated.")

    def call(self, sensory_input: tf.Tensor) -> tf.Tensor:
        x = self.conv1(sensory_input)
        x = self.pool1(x)
        x = self.flatten(x)
        salience = self.dense_output(x)
        return salience

class HierarchicalRecurrentMemory(Model):
    """
    Processes a memory engram through a hierarchy of recurrent layers to
    retrieve a context vector via attention, guided by a query.
    Neural Analogue: Hippocampus / Cortex interaction.
    """
    def __init__(self, context_dim: int = 10, proj_dim: int = 64, **kwargs):
        super(HierarchicalRecurrentMemory, self).__init__(**kwargs)
        # Four hierarchical recurrent layers
        self.layer1_encoder = layers.LSTM(128, return_sequences=True, name="HRM_L1_Episodic")
        self.layer2_abstractor = layers.LSTM(64, return_sequences=True, name="HRM_L2_Abstraction")
        self.layer3_generalizer = layers.LSTM(32, return_sequences=True, name="HRM_L3_Thematic")
        self.layer4_synthesizer = layers.LSTM(16, return_sequences=True, name="HRM_L4_Synthesis")
        
        # Persistent projection and attention layers
        self.projection = HRMProjection(proj_dim=proj_dim)
        self.attention = layers.Attention(name="HRM_Attention")
        self.flatten = layers.Flatten(name="HRM_Flatten")
        self.dense_output = layers.Dense(context_dim, name="HRM_Context_Output")
        print("✅ HierarchicalRecurrentMemory (Analogue: Hippocampus/Cortex) instantiated.")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        query_vector, memory_engram = inputs
        
        # Expand memory vector to have a time dimension for LSTMs
        memory_reshaped = tf.expand_dims(memory_engram, axis=1)

        # Process memory through the hierarchy
        l1_out = self.layer1_encoder(memory_reshaped)
        l2_out = self.layer2_abstractor(l1_out)
        l3_out = self.layer3_generalizer(l2_out)
        l4_out = self.layer4_synthesizer(l3_out) # This is the value sequence

        # Project query and value sequence
        query_proj, value_proj = self.projection((query_vector, l4_out))
        
        # Attend to the processed memory sequence
        attention_out = self.attention([query_proj, value_proj])
        attention_flat = self.flatten(attention_out)
        
        context = self.dense_output(attention_flat)
        return context

class NeuromodulatoryGatingSystem(Model):
    """
    Generates modulation parameters (gamma, beta) for the FiLM layer based
    on core affect and cognitive appraisal.
    Neural Analogue: Neurochemical systems (e.g., dopamine, serotonin).
    """
    def __init__(self, modulation_dim: int, **kwargs):
        super(NeuromodulatoryGatingSystem, self).__init__(**kwargs)
        self.dense1 = layers.Dense(32, activation='relu', name="NMS_Dense1")
        # Output layer must produce 2 * modulation_dim parameters for gamma and beta
        self.output_params = layers.Dense(2 * modulation_dim, name="NMS_Gamma_Beta_Output")
        print("✅ NMS (Analogue: Neurochemical Systems) instantiated.")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        core_affect, cognitive_appraisal = inputs
        combined_input = tf.concat([core_affect, cognitive_appraisal], axis=1)
        x = self.dense1(combined_input)
        gamma_beta_params = self.output_params(x)
        return gamma_beta_params

class DeliberativePathway(Model):
    """
    The main cognitive processing pathway. Integrates sensory data, salience,
    affect, and memory to produce actions and a cognitive appraisal. Its output
    is modulated by the NMS.
    Neural Analogue: Prefrontal Cortex (PFC) "High Road".
    """
    def __init__(self, num_actions: int, cognitive_dim: int, lstm_dim: int, **kwargs):
        super(DeliberativePathway, self).__init__(**kwargs)
        self.sensory_encoder = layers.Dense(128, activation='relu', name="Delib_Sensory_Enc")
        self.lstm = layers.LSTM(lstm_dim, name="Deliberative_LSTM")
        self.appraisal_output = layers.Dense(cognitive_dim, name="Deliberative_Appraisal")
        self.film_layer = FilMLayer(name="Deliberative_FiLM")
        self.action_output = layers.Dense(num_actions, activation='softmax', name="Deliberative_Action")
        print("✅ DeliberativePathway (Analogue: PFC High-Road) instantiated.")

    def call(self, inputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        # 1. Encode and combine all inputs for the LSTM
        sensory_flat = layers.Flatten()(inputs['sensory_input'])
        sensory_bottleneck = self.sensory_encoder(sensory_flat)
        
        combined_inputs = tf.concat([
            sensory_bottleneck,
            inputs['salience'],
            inputs['core_affect'],
            inputs['context']
        ], axis=1)
        
        # 2. Process through LSTM
        combined_inputs_reshaped = tf.expand_dims(combined_inputs, axis=1)
        lstm_output = self.lstm(combined_inputs_reshaped)
        
        # 3. Generate cognitive appraisal from the LSTM's output
        cognitive_appraisal = self.appraisal_output(lstm_output)
        
        # 4. Modulate the LSTM output using the FiLM layer
        modulated_output = self.film_layer([lstm_output, inputs['gamma_beta_params']])
        
        # 5. Produce the final action from the modulated output
        action = self.action_output(modulated_output)
        
        return action, cognitive_appraisal

# ==============================================================================
# 3. TOP-LEVEL NECA MODEL
# ==============================================================================

class NECA(Model):
    """
    The integrated Neuro-Emotive Cognitive Architecture.
    This model orchestrates the flow of information between all sub-modules.
    """
    def __init__(self, internal_state_dim: int, num_actions: int, context_dim: int, **kwargs):
        super(NECA, self).__init__(**kwargs)
        print("\n--- Initializing Full NECA Model ---")
        
        lstm_dim = 64 # Dimension for the deliberative LSTM and NMS modulation
        
        self.ism = InteroceptiveStateModule(core_affect_dim=2)
        self.reactive_pathway = ReactivePathway()
        self.hrm = HierarchicalRecurrentMemory(context_dim=context_dim)
        self.nms = NeuromodulatoryGatingSystem(modulation_dim=lstm_dim)
        self.deliberative_pathway = DeliberativePathway(
            num_actions=num_actions, cognitive_dim=context_dim, lstm_dim=lstm_dim
        )
        
        print("--- NECA Model Initialized Successfully ---\n")

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False, verbose: bool = False) -> Dict[str, tf.Tensor]:
        """
        Executes a single, efficient forward pass through the architecture.
        """
        if verbose: print("\n--- NECA Forward Pass ---")

        # === Stage 1: Parallel Initial Processing ===
        core_affect = self.ism(inputs['internal_state'])
        salience = self.reactive_pathway(inputs['sensory_input'])
        query_vector = layers.Flatten()(inputs['sensory_input'])
        context = self.hrm((query_vector, inputs['memory_to_process']))

        if verbose:
            print(f"-> [ISM] Core Affect: {core_affect.numpy()}")
            print(f"-> [Reactive] Salience: {salience.numpy().flatten()}")
            print(f"-> [HRM] Context Vector Retrieved.")

        # === Stage 2: Efficient Deliberation and Modulation ===
        # This refactored flow avoids running the deliberative pathway twice.
        
        # First, get the cognitive appraisal to generate FiLM params.
        # We can do this by running the deliberative path up to the appraisal output.
        # A more efficient way is to recognize that NMS only needs core_affect and appraisal.
        # We can get appraisal from a preliminary pass and then modulate.
        
        # Let's create a temporary combined input for the deliberative LSTM
        sensory_flat = layers.Flatten()(inputs['sensory_input'])
        sensory_bottleneck = self.deliberative_pathway.sensory_encoder(sensory_flat)
        combined_for_lstm = tf.concat([sensory_bottleneck, salience, core_affect, context], axis=1)
        lstm_in = tf.expand_dims(combined_for_lstm, axis=1)
        
        # Get the unmodulated LSTM output
        lstm_out = self.deliberative_pathway.lstm(lstm_in)
        
        # Generate the cognitive appraisal from this output
        cognitive_appraisal = self.deliberative_pathway.appraisal_output(lstm_out)
        if verbose: print(f"-> [Deliberative] Cognitive Appraisal Generated.")

        # Now, use the appraisal to get neuromodulation parameters
        gamma_beta_params = self.nms((core_affect, cognitive_appraisal))
        if verbose: print("-> [NMS] FiLM Modulation Parameters Generated.")

        # Finally, modulate the LSTM output and get the action
        modulated_output = self.deliberative_pathway.film_layer([lstm_out, gamma_beta_params])
        final_action = self.deliberative_pathway.action_output(modulated_output)
        if verbose: print("-> [Deliberative] Final, Modulated Action Produced.")

        return {
            "action": final_action,
            "core_affect": core_affect,
            "cognitive_appraisal": cognitive_appraisal
        }

# ==============================================================================
# 4. PRIORITIZED EXPERIENCE REPLAY (PER)
# ==============================================================================

class PrioritizedReplayBuffer:
    """
    A replay buffer that samples experiences based on their priority (e.g., TD-error),
    leading to more efficient learning.
    """
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha # Controls how much prioritization is used
        self.epsilon = 1e-6 # Small constant to ensure all priorities are non-zero
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def add(self, experience: Dict, priority: Optional[float] = None):
        """Adds an experience to the buffer."""
        if priority is None:
            # New experiences get max priority to ensure they are sampled at least once.
            max_p = self.priorities.max() if self.buffer else 1.0
            priority = max_p

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Samples a batch of experiences with importance-sampling weights."""
        if not self.buffer:
            raise ValueError("Cannot sample from an empty buffer.")

        current_size = len(self.buffer)
        prios = self.priorities[:current_size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(current_size, batch_size, p=probs)
        
        # Importance-sampling weights correct for the bias introduced by prioritized sampling
        weights = (current_size * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize for stability

        # Unzip experiences
        samples = [self.buffer[i] for i in indices]
        
        return indices, samples, tf.constant(weights, dtype=tf.float32)

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray):
        """Updates the priorities of sampled experiences."""
        for i, p in zip(indices, new_priorities):
            self.priorities[i] = p + self.epsilon

    @property
    def size(self):
        return len(self.buffer)


# ==============================================================================
# 5. DEMO EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    SENSORY_SHAPE = (48, 48, 1)
    INTERNAL_STATE_DIM = 3
    CONTEXT_DIM = 10
    NUM_ACTIONS = 5
    
    BUFFER_CAPACITY = 10000
    REPLAY_BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    REPLAY_EVERY_N_STEPS = 4
    TOTAL_STEPS = 5000
    
    # --- Instantiation ---
    neca_model = NECA(
        internal_state_dim=INTERNAL_STATE_DIM,
        num_actions=NUM_ACTIONS,
        context_dim=CONTEXT_DIM
    )
    
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    replay_buffer = PrioritizedReplayBuffer(capacity=BUFFER_CAPACITY)

    # --- Build the model by doing a single forward pass ---
    dummy_sensory = tf.random.normal([1, *SENSORY_SHAPE])
    dummy_internal = tf.random.normal([1, INTERNAL_STATE_DIM])
    dummy_memory = tf.random.normal([1, np.prod(SENSORY_SHAPE) + INTERNAL_STATE_DIM])
    
    _ = neca_model({
        'sensory_input': dummy_sensory,
        'internal_state': dummy_internal,
        'memory_to_process': dummy_memory
    }, verbose=True)
    
    neca_model.summary()

    # --- Continuous Learning Loop with PER ---
    print("\n--- Starting Continuous Learning Simulation with PER ---\n")
    for step in range(1, TOTAL_STEPS + 1):
        # 1. Simulate a new experience
        sensory_input = tf.random.normal([1, *SENSORY_SHAPE])
        internal_state = tf.random.normal([1, INTERNAL_STATE_DIM])
        # A memory engram could be a flattened past state
        memory_to_process = tf.random.normal([1, np.prod(SENSORY_SHAPE) + INTERNAL_STATE_DIM])
        label = tf.random.uniform([1], 0, NUM_ACTIONS, dtype=tf.int32)

        # 2. Add the new experience to the buffer (with max priority initially)
        experience = {
            'sensory_input': sensory_input,
            'internal_state': internal_state,
            'memory_to_process': memory_to_process,
            'label': label
        }
        replay_buffer.add(experience)

        # 3. Periodically sample from the buffer and train
        if step > REPLAY_BATCH_SIZE and (step % REPLAY_EVERY_N_STEPS == 0):
            indices, samples, is_weights = replay_buffer.sample(REPLAY_BATCH_SIZE)
            
            # Unzip the batch of experiences
            batch_sensory = tf.concat([s['sensory_input'] for s in samples], axis=0)
            batch_internal = tf.concat([s['internal_state'] for s in samples], axis=0)
            batch_memory = tf.concat([s['memory_to_process'] for s in samples], axis=0)
            batch_labels = tf.concat([s['label'] for s in samples], axis=0)
            
            with tf.GradientTape() as tape:
                # Get model predictions
                predictions = neca_model({
                    'sensory_input': batch_sensory,
                    'internal_state': batch_internal,
                    'memory_to_process': batch_memory
                }, training=True)
                
                # Calculate loss for each sample in the batch
                per_example_loss = loss_fn(batch_labels, predictions['action'])
                
                # Apply importance-sampling weights
                weighted_loss = tf.reduce_mean(per_example_loss * is_weights)

            # Apply gradients
            grads = tape.gradient(weighted_loss, neca_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, neca_model.trainable_variables))
            
            # Update priorities in the buffer with the new loss values
            new_priorities = per_example_loss.numpy()
            replay_buffer.update_priorities(indices, new_priorities)

            if step % 200 == 0:
                print(f"Step: {step}/{TOTAL_STEPS} | Replay Loss: {weighted_loss.numpy():.4f} | Buffer Size: {replay_buffer.size}")

    print("\n--- Continuous Learning Simulation Complete ---")

