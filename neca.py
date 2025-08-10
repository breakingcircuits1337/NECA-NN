# NECA: Neuro-Emotive Cognitive Architecture
# This script is a refactored and improved version. Key features include:
# - A clear, modular architecture with docstrings and type hinting.
# - Stateful LSTMs in the HierarchicalRecurrentMemory (HRM) for persistent context.
# - A high-performance training loop using a @tf.function decorated train_step.
# - An efficient forward pass that avoids redundant computations.

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
        feat_dim = tf.shape(features)[-1]
        gamma = gamma_beta[:, :feat_dim]
        beta = gamma_beta[:, feat_dim:]
        return (gamma * features) + beta

class HRMProjection(layers.Layer):
    """
    Encapsulates the query and value projection logic for the HRM module.
    This makes the HRM module cleaner and re-uses these layers.
    """
    def __init__(self, proj_dim: int = 32, **kwargs):
        super(HRMProjection, self).__init__(**kwargs)
        self.query_proj = layers.Dense(proj_dim, activation='relu', name="HRM_Query_Proj")
        self.value_proj = layers.TimeDistributed(
            layers.Dense(proj_dim, activation='relu'), name="HRM_Value_Proj"
        )
        print(f"✅ HRMProjection instantiated with projection dim {proj_dim}.")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        query_vector, value_sequence = inputs
        projected_query = self.query_proj(query_vector)
        projected_query_expanded = tf.expand_dims(projected_query, axis=1)
        projected_value = self.value_proj(value_sequence)
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
    Processes a memory engram through a hierarchy of STATEFUL recurrent layers to
    retrieve a context vector via attention, guided by a query.
    Neural Analogue: Hippocampus / Cortex interaction.
    """
    def __init__(self, context_dim: int = 10, proj_dim: int = 64, 
                 stateful: bool = False, batch_size: Optional[int] = None, **kwargs):
        super(HierarchicalRecurrentMemory, self).__init__(**kwargs)
        
        self.layer1_encoder = layers.LSTM(128, return_sequences=True, name="HRM_L1_Episodic", stateful=stateful, batch_input_shape=(batch_size, None, None) if stateful else None)
        self.layer2_abstractor = layers.LSTM(64, return_sequences=True, name="HRM_L2_Abstraction", stateful=stateful)
        self.layer3_generalizer = layers.LSTM(32, return_sequences=True, name="HRM_L3_Thematic", stateful=stateful)
        self.layer4_synthesizer = layers.LSTM(16, return_sequences=True, name="HRM_L4_Synthesis", stateful=stateful)
        
        self.projection = HRMProjection(proj_dim=proj_dim)
        self.attention = layers.Attention(name="HRM_Attention")
        self.flatten = layers.Flatten(name="HRM_Flatten")
        self.dense_output = layers.Dense(context_dim, name="HRM_Context_Output")
        print(f"✅ HierarchicalRecurrentMemory (Stateful: {stateful}) instantiated.")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        query_vector, memory_engram = inputs
        memory_reshaped = tf.expand_dims(memory_engram, axis=1)

        l1_out = self.layer1_encoder(memory_reshaped)
        l2_out = self.layer2_abstractor(l1_out)
        l3_out = self.layer3_generalizer(l2_out)
        l4_out = self.layer4_synthesizer(l3_out)

        query_proj, value_proj = self.projection((query_vector, l4_out))
        
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
    affect, and memory to produce actions and a cognitive appraisal.
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
        sensory_flat = layers.Flatten()(inputs['sensory_input'])
        sensory_bottleneck = self.sensory_encoder(sensory_flat)
        
        combined_inputs = tf.concat([
            sensory_bottleneck,
            inputs['salience'],
            inputs['core_affect'],
            inputs['context']
        ], axis=1)
        
        combined_inputs_reshaped = tf.expand_dims(combined_inputs, axis=1)
        lstm_output = self.lstm(combined_inputs_reshaped)
        
        cognitive_appraisal = self.appraisal_output(lstm_output)
        modulated_output = self.film_layer([lstm_output, inputs['gamma_beta_params']])
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
    def __init__(self, internal_state_dim: int, num_actions: int, context_dim: int, 
                 stateful_memory: bool = False, batch_size: Optional[int] = None, **kwargs):
        super(NECA, self).__init__(**kwargs)
        print("\n--- Initializing Full NECA Model ---")
        
        lstm_dim = 64
        
        self.ism = InteroceptiveStateModule(core_affect_dim=2)
        self.reactive_pathway = ReactivePathway()
        self.hrm = HierarchicalRecurrentMemory(
            context_dim=context_dim, stateful=stateful_memory, batch_size=batch_size
        )
        self.nms = NeuromodulatoryGatingSystem(modulation_dim=lstm_dim)
        self.deliberative_pathway = DeliberativePathway(
            num_actions=num_actions, cognitive_dim=context_dim, lstm_dim=lstm_dim
        )
        
        self.stateful_memory = stateful_memory
        print("--- NECA Model Initialized Successfully ---\n")

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False, verbose: bool = False) -> Dict[str, tf.Tensor]:
        if verbose: print("\n--- NECA Forward Pass ---")

        core_affect = self.ism(inputs['internal_state'])
        salience = self.reactive_pathway(inputs['sensory_input'])
        query_vector = layers.Flatten()(inputs['sensory_input'])
        context = self.hrm((query_vector, inputs['memory_to_process']))

        if verbose:
            print(f"-> [ISM] Core Affect: {core_affect.numpy()}")
            print(f"-> [Reactive] Salience: {salience.numpy().flatten()}")
            print(f"-> [HRM] Context Vector Retrieved.")

        sensory_flat = layers.Flatten()(inputs['sensory_input'])
        sensory_bottleneck = self.deliberative_pathway.sensory_encoder(sensory_flat)
        combined_for_lstm = tf.concat([sensory_bottleneck, salience, core_affect, context], axis=1)
        lstm_in = tf.expand_dims(combined_for_lstm, axis=1)
        
        lstm_out = self.deliberative_pathway.lstm(lstm_in)
        cognitive_appraisal = self.deliberative_pathway.appraisal_output(lstm_out)
        if verbose: print(f"-> [Deliberative] Cognitive Appraisal Generated.")

        gamma_beta_params = self.nms((core_affect, cognitive_appraisal))
        if verbose: print("-> [NMS] FiLM Modulation Parameters Generated.")

        modulated_output = self.deliberative_pathway.film_layer([lstm_out, gamma_beta_params])
        final_action = self.deliberative_pathway.action_output(modulated_output)
        if verbose: print("-> [Deliberative] Final, Modulated Action Produced.")

        return {
            "action": final_action,
            "core_affect": core_affect,
            "cognitive_appraisal": cognitive_appraisal
        }
        
    def reset_states(self):
        """Resets the states of the stateful layers in the model."""
        if self.stateful_memory:
            self.hrm.reset_states()
            print("HRM states have been reset.")
        else:
            print("Model is not stateful. No states to reset.")

# ==============================================================================
# 4. OPTIMIZED TRAINING STEP
# ==============================================================================

@tf.function
def train_step(model: Model, optimizer: optimizers.Optimizer, loss_fn, 
               inputs: Dict[str, tf.Tensor], label: tf.Tensor) -> tf.Tensor:
    """
    Performs a single, compiled training step.
    The @tf.function decorator compiles this into a high-performance static graph.
    """
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(label, predictions['action'])
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss

# ==============================================================================
# 5. DEMO EXECUTION BLOCK: STATEFUL ONLINE LEARNING
# ==============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    SENSORY_SHAPE = (48, 48, 1)
    INTERNAL_STATE_DIM = 3
    CONTEXT_DIM = 10
    NUM_ACTIONS = 5
    
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    TOTAL_STEPS = 5000
    EPISODE_LENGTH = 100
    
    # --- Instantiation ---
    neca_model = NECA(
        internal_state_dim=INTERNAL_STATE_DIM,
        num_actions=NUM_ACTIONS,
        context_dim=CONTEXT_DIM,
        stateful_memory=True,
        batch_size=BATCH_SIZE
    )
    
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # --- Build the model by doing a single forward pass ---
    dummy_sensory = tf.random.normal([BATCH_SIZE, *SENSORY_SHAPE])
    dummy_internal = tf.random.normal([BATCH_SIZE, INTERNAL_STATE_DIM])
    dummy_memory = tf.random.normal([BATCH_SIZE, np.prod(SENSORY_SHAPE) + INTERNAL_STATE_DIM])
    
    _ = neca_model({
        'sensory_input': dummy_sensory,
        'internal_state': dummy_internal,
        'memory_to_process': dummy_memory
    }, verbose=True)
    
    neca_model.summary()

    # --- Stateful Online Learning Loop with Optimized train_step ---
    print("\n--- Starting Stateful Online Learning Simulation with @tf.function ---\n")
    for step in range(1, TOTAL_STEPS + 1):
        # 1. Simulate a new experience
        sensory_input = tf.random.normal([BATCH_SIZE, *SENSORY_SHAPE])
        internal_state = tf.random.normal([BATCH_SIZE, INTERNAL_STATE_DIM])
        memory_to_process = tf.random.normal([BATCH_SIZE, np.prod(SENSORY_SHAPE) + INTERNAL_STATE_DIM])
        label = tf.random.uniform([BATCH_SIZE], 0, NUM_ACTIONS, dtype=tf.int32)

        # 2. Perform a single training step by calling the compiled function
        loss = train_step(
            model=neca_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            inputs={
                'sensory_input': sensory_input,
                'internal_state': internal_state,
                'memory_to_process': memory_to_process
            },
            label=label
        )
        
        # 3. Periodically reset the model's state
        if step % EPISODE_LENGTH == 0:
            print(f"\n--- End of Episode at Step {step}. Resetting HRM states. ---\n")
            neca_model.reset_states()

        if step % 100 == 0:
            print(f"Step: {step}/{TOTAL_STEPS} | Online Loss: {loss.numpy():.4f}")

    print("\n--- Stateful Online Learning Simulation Complete ---")
