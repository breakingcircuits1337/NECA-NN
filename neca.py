"""
Enhanced Neuro-Emotive Cognitive Architecture (NECA) with Neurochemical Modulation

A biologically-inspired neural network for emotion simulation incorporating:
- Plutchik's 8 primary emotions (joy, sadness, anger, fear, trust, disgust, surprise, anticipation)
- Detailed neurochemical modulation (dopamine, serotonin, norepinephrine, cortisol, etc.)
- Chemical milieu affecting learning rates, attention, and emotional dynamics

Author: AI Research Engineer
Framework: TensorFlow/Keras
Python Version: 3.8+
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class NeurochemicalSystem(tf.keras.layers.Layer):
    """
    Simulates the chemical milieu of neurotransmitters, hormones, and modulators.
    
    Neurobiological Analogue: Neurochemical Systems
    - Represents the complex interplay of dopamine, serotonin, norepinephrine, cortisol, etc.
    - Provides dynamic modulation of neural circuits across multiple timescales
    - Acts as sophisticated gain controls rather than simple on/off switches
    """
    
    def __init__(self, **kwargs):
        super(NeurochemicalSystem, self).__init__(**kwargs)
        
        # Initialize neurochemical concentrations as trainable parameters
        # These represent baseline levels that can be learned
        self.dopamine_baseline = self.add_weight(
            name='dopamine_baseline', shape=(1,), initializer='ones', trainable=True
        )
        self.serotonin_baseline = self.add_weight(
            name='serotonin_baseline', shape=(1,), initializer='ones', trainable=True
        )
        self.norepinephrine_baseline = self.add_weight(
            name='norepinephrine_baseline', shape=(1,), initializer='ones', trainable=True
        )
        self.cortisol_baseline = self.add_weight(
            name='cortisol_baseline', shape=(1,), initializer='zeros', trainable=True
        )
        self.oxytocin_baseline = self.add_weight(
            name='oxytocin_baseline', shape=(1,), initializer='ones', trainable=True
        )
        
        # Networks to compute neurochemical dynamics based on emotional state
        self.dopamine_net = layers.Dense(16, activation='relu', name='dopamine_dynamics')
        self.serotonin_net = layers.Dense(16, activation='relu', name='serotonin_dynamics')
        self.norepinephrine_net = layers.Dense(16, activation='relu', name='norepinephrine_dynamics')
        self.cortisol_net = layers.Dense(16, activation='relu', name='cortisol_dynamics')
        self.oxytocin_net = layers.Dense(16, activation='relu', name='oxytocin_dynamics')
        
        # Output layers for each neurochemical
        self.dopamine_out = layers.Dense(1, activation='sigmoid', name='dopamine_level')
        self.serotonin_out = layers.Dense(1, activation='sigmoid', name='serotonin_level')
        self.norepinephrine_out = layers.Dense(1, activation='sigmoid', name='norepinephrine_level')
        self.cortisol_out = layers.Dense(1, activation='sigmoid', name='cortisol_level')
        self.oxytocin_out = layers.Dense(1, activation='sigmoid', name='oxytocin_level')
        
        # Endocannabinoid system for homeostatic regulation
        self.endocannabinoid_net = layers.Dense(8, activation='relu', name='endocannabinoid_net')
        self.endocannabinoid_out = layers.Dense(1, activation='sigmoid', name='endocannabinoid_level')
    
    def call(self, emotional_state, reward_prediction_error, social_context=None, training=None):
        """
        Compute neurochemical levels based on current emotional and contextual state.
        
        Args:
            emotional_state: Current emotional state vector (8 emotions)
            reward_prediction_error: Error signal for dopamine modulation
            social_context: Optional social context information
        
        Returns:
            neurochemical_state: Dictionary of current neurochemical levels
        """
        batch_size = tf.shape(emotional_state)[0]
        
        # Compute dynamic neurochemical responses
        
        # Dopamine: Reward, motivation, pleasure, focus
        # Responds to reward prediction error and positive emotions (joy, anticipation)
        dopamine_input = tf.concat([
            emotional_state,
            tf.expand_dims(reward_prediction_error, axis=-1)
        ], axis=-1)
        dopamine_features = self.dopamine_net(dopamine_input)
        dopamine_level = self.dopamine_out(dopamine_features) + self.dopamine_baseline
        
        # Serotonin: Mood stabilization, well-being, anxiety regulation
        # Stabilized by positive emotions (joy, trust), reduced by negative emotions
        serotonin_features = self.serotonin_net(emotional_state)
        serotonin_level = self.serotonin_out(serotonin_features) + self.serotonin_baseline
        
        # Norepinephrine: Arousal, alertness, vigilance, stress response
        # Increased by fear, anger, surprise; decreased by trust, joy
        norepinephrine_features = self.norepinephrine_net(emotional_state)
        norepinephrine_level = self.norepinephrine_out(norepinephrine_features) + self.norepinephrine_baseline
        
        # Cortisol: Primary stress hormone
        # Elevated by fear, anger, disgust, sadness
        cortisol_features = self.cortisol_net(emotional_state)
        cortisol_level = self.cortisol_out(cortisol_features) + self.cortisol_baseline
        
        # Oxytocin: Social bonding, trust, empathy
        # Enhanced by trust, joy; contextually modulated by social information
        if social_context is not None:
            oxytocin_input = tf.concat([emotional_state, social_context], axis=-1)
        else:
            oxytocin_input = emotional_state
        oxytocin_features = self.oxytocin_net(oxytocin_input)
        oxytocin_level = self.oxytocin_out(oxytocin_features) + self.oxytocin_baseline
        
        # Endocannabinoids: Homeostatic regulation
        # Provides retrograde signaling to prevent runaway neurochemical responses
        total_neurochemical_activity = tf.concat([
            dopamine_level, serotonin_level, norepinephrine_level, 
            cortisol_level, oxytocin_level
        ], axis=-1)
        endocannabinoid_features = self.endocannabinoid_net(total_neurochemical_activity)
        endocannabinoid_level = self.endocannabinoid_out(endocannabinoid_features)
        
        # Apply homeostatic regulation via endocannabinoids
        regulation_factor = 1.0 - 0.3 * endocannabinoid_level
        
        return {
            'dopamine': dopamine_level * regulation_factor,
            'serotonin': serotonin_level * regulation_factor,
            'norepinephrine': norepinephrine_level * regulation_factor,
            'cortisol': cortisol_level * regulation_factor,
            'oxytocin': oxytocin_level * regulation_factor,
            'endocannabinoids': endocannabinoid_level
        }


class EmotionalStateProcessor(tf.keras.layers.Layer):
    """
    Processes core affect into discrete emotional states based on Plutchik's model.
    
    Neurobiological Analogue: Limbic System Integration
    - Maps continuous affective dimensions to discrete emotional categories
    - Implements Plutchik's 8 primary emotions in emotion wheel
    - Provides basis for neurochemical modulation
    """
    
    def __init__(self, **kwargs):
        super(EmotionalStateProcessor, self).__init__(**kwargs)
        
        # Plutchik's 8 primary emotions:
        # Joy, Sadness, Anger, Fear, Trust, Disgust, Surprise, Anticipation
        self.emotion_names = [
            'joy', 'sadness', 'anger', 'fear', 
            'trust', 'disgust', 'surprise', 'anticipation'
        ]
        
        # Network to map core affect to emotional states
        self.emotion_mapper = keras.Sequential([
            layers.Dense(32, activation='relu', name='emotion_hidden1'),
            layers.Dense(16, activation='relu', name='emotion_hidden2'),
            layers.Dense(8, activation='softmax', name='emotion_probabilities')
        ], name='emotion_state_network')
        
        # Emotional valence for reward calculation
        # Positive: joy, trust, surprise (contextual)
        # Negative: sadness, anger, fear, disgust, anticipation (threat-related)
        self.emotion_valences = tf.constant([
            1.0,   # joy (positive)
            -0.8,  # sadness (negative)
            -0.6,  # anger (negative, but mobilizing)
            -1.0,  # fear (strongly negative)
            0.8,   # trust (positive)
            -0.9,  # disgust (strongly negative)
            0.0,   # surprise (neutral, context-dependent)
            -0.3   # anticipation (slightly negative, uncertainty)
        ], dtype=tf.float32)
    
    def call(self, core_affect, context_info=None, training=None):
        """
        Convert core affect to discrete emotional states.
        
        Args:
            core_affect: Core affect vector (valence, arousal)
            context_info: Optional contextual information
        
        Returns:
            emotion_state: Probability distribution over 8 emotions
            emotion_valence: Weighted emotional valence
        """
        if context_info is not None:
            emotion_input = tf.concat([core_affect, context_info], axis=-1)
        else:
            emotion_input = core_affect
        
        # Map to emotional probabilities
        emotion_state = self.emotion_mapper(emotion_input)
        
        # Calculate overall emotional valence
        emotion_valence = tf.reduce_sum(
            emotion_state * self.emotion_valences, axis=-1, keepdims=True
        )
        
        return emotion_state, emotion_valence


class EnhancedFilMLayer(tf.keras.layers.Layer):
    """
    Enhanced Feature-wise Linear Modulation with neurochemical influence.
    
    Neurobiological Analogue: Neurochemical Modulation of Neural Circuits
    - Different neurochemicals provide specialized modulation patterns
    - Enables timescale-specific and circuit-specific modulation
    - Implements sophisticated gain control mechanisms
    """
    
    def __init__(self, **kwargs):
        super(EnhancedFilMLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        Apply neurochemically-influenced feature modulation.
        
        Args:
            inputs: List containing [features, neurochemical_state]
                   features: Tensor to be modulated
                   neurochemical_state: Dictionary of neurochemical levels
        
        Returns:
            modulated_features: Neurochemically modulated tensor
        """
        features, neurochemical_state = inputs
        
        # Extract neurochemical levels
        dopamine = neurochemical_state['dopamine']
        serotonin = neurochemical_state['serotonin']
        norepinephrine = neurochemical_state['norepinephrine']
        cortisol = neurochemical_state['cortisol']
        oxytocin = neurochemical_state['oxytocin']
        
        # Dopamine: Increases gain on reward-related features
        # Acts as attention modulator for motivationally relevant information
        dopamine_gain = 1.0 + 0.5 * dopamine
        
        # Serotonin: Stabilizes neural activity, reduces volatility
        # Acts as a global stabilizer preventing extreme responses
        serotonin_stability = 0.1 + 0.9 * serotonin
        
        # Norepinephrine: Increases gain on salient/threat-related features
        # Enhances attention to potentially important stimuli
        norepinephrine_gain = 1.0 + 0.8 * norepinephrine
        
        # Cortisol: Suppresses deliberative processing under stress
        # Biases system toward fast, reactive responses
        cortisol_suppression = 1.0 - 0.3 * cortisol
        
        # Oxytocin: Increases social feature weights, reduces threat sensitivity
        oxytocin_social_bias = 1.0 + 0.3 * oxytocin
        
        # Combine neurochemical effects
        combined_gain = dopamine_gain * norepinephrine_gain * cortisol_suppression * oxytocin_social_bias
        combined_stability = serotonin_stability
        
        # Apply modulation: features are scaled by combined effects
        modulated_features = features * combined_gain * combined_stability
        
        return modulated_features


class InteroceptiveStateModule(tf.keras.layers.Layer):
    """
    Enhanced Interoceptive State Module with emotional processing.
    
    Neurobiological Analogue: Insula, Gut-Brain Axis, Homeostasis
    - Integrates bodily signals with emotional state processing
    - Generates core affect that influences entire system
    - Provides foundation for neurochemical modulation
    """
    
    def __init__(self, core_affect_dim=2, hidden_units=32, **kwargs):
        super(InteroceptiveStateModule, self).__init__(**kwargs)
        self.core_affect_dim = core_affect_dim
        self.hidden_units = hidden_units
        
        # Enhanced LSTM with emotional context integration
        self.lstm = layers.LSTM(self.hidden_units, return_state=True, name='ism_lstm')
        
        # Core affect generation (valence, arousal)
        self.core_affect_output = layers.Dense(
            self.core_affect_dim, 
            activation='tanh',
            name='core_affect'
        )
        
        # Emotional state processor
        self.emotion_processor = EmotionalStateProcessor()
        
        # Neurochemical system
        self.neurochemical_system = NeurochemicalSystem()
        
    def call(self, internal_state, reward_prediction_error=None, social_context=None, training=None):
        """
        Enhanced internal state processing with emotional and neurochemical dynamics.
        
        Args:
            internal_state: Physiological state tensor
            reward_prediction_error: For dopamine modulation
            social_context: For oxytocin modulation
        
        Returns:
            Dictionary containing core affect, emotions, and neurochemicals
        """
        # Process internal state through LSTM
        lstm_output, h_state, c_state = self.lstm(internal_state)
        
        # Generate core affect
        core_affect = self.core_affect_output(lstm_output)
        
        # Process emotions from core affect
        emotion_state, emotion_valence = self.emotion_processor(
            core_affect, social_context, training=training
        )
        
        # Compute neurochemical state
        if reward_prediction_error is None:
            reward_prediction_error = tf.zeros((tf.shape(core_affect)[0],))
        
        neurochemical_state = self.neurochemical_system(
            emotion_state, 
            reward_prediction_error,
            social_context,
            training=training
        )
        
        return {
            'core_affect': core_affect,
            'emotion_state': emotion_state,
            'emotion_valence': emotion_valence,
            'neurochemical_state': neurochemical_state,
            'lstm_state': (h_state, c_state)
        }


class ReactivePathway(tf.keras.layers.Layer):
    """
    Enhanced Reactive Pathway with neurochemical modulation.
    
    Neurobiological Analogue: Thalamo-Amygdala "Low Road"
    - Fast threat/salience detection enhanced by norepinephrine
    - Modulated by cortisol (stress response) and serotonin (anxiety regulation)
    """
    
    def __init__(self, **kwargs):
        super(ReactivePathway, self).__init__(**kwargs)
        
        # CNN layers for rapid processing
        self.conv1 = layers.Conv2D(16, 3, activation='relu', name='reactive_conv1')
        self.pool1 = layers.MaxPooling2D(2, name='reactive_pool1')
        self.conv2 = layers.Conv2D(32, 3, activation='relu', name='reactive_conv2')
        self.pool2 = layers.MaxPooling2D(2, name='reactive_pool2')
        self.flatten = layers.Flatten(name='reactive_flatten')
        
        # Enhanced salience detection with emotional context
        self.salience_network = keras.Sequential([
            layers.Dense(64, activation='relu', name='salience_hidden'),
            layers.Dense(1, activation='sigmoid', name='salience_output')
        ], name='salience_network')
        
        # Neurochemical modulation layer
        self.film_layer = EnhancedFilMLayer(name='reactive_film')
        
    def call(self, sensory_input, neurochemical_state=None, training=None):
        """
        Enhanced reactive processing with neurochemical modulation.
        
        Args:
            sensory_input: Raw sensory data
            neurochemical_state: Current neurochemical levels
        
        Returns:
            salience: Modulated salience signal
        """
        # CNN processing
        x = self.conv1(sensory_input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        
        # Apply neurochemical modulation if available
        if neurochemical_state is not None:
            x = self.film_layer([x, neurochemical_state])
        
        # Generate salience signal
        salience = self.salience_network(x)
        
        return salience


class HierarchicalRecurrentMemory(tf.keras.layers.Layer):
    """
    Enhanced Hierarchical Memory with emotional context integration.
    
    Neurobiological Analogue: Hippocampus with Emotional Modulation
    - Memory formation and retrieval influenced by emotional state
    - Cortisol affects memory consolidation
    - Norepinephrine enhances memory for salient events
    """
    
    def __init__(self, word_dim=64, sentence_dim=64, attention_dim=32, **kwargs):
        super(HierarchicalRecurrentMemory, self).__init__(**kwargs)
        
        self.word_dim = word_dim
        self.sentence_dim = sentence_dim
        self.attention_dim = attention_dim
        
        # Enhanced encoders with neurochemical modulation
        self.word_encoder = layers.GRU(self.word_dim, return_sequences=True, name='word_encoder')
        self.word_attention = layers.Dense(self.attention_dim, activation='tanh', name='word_attention')
        self.word_context = layers.Dense(1, name='word_context')
        
        self.sentence_encoder = layers.GRU(self.sentence_dim, return_sequences=True, name='sentence_encoder')
        self.sentence_attention = layers.Dense(self.attention_dim, activation='tanh', name='sentence_attention')
        self.sentence_context = layers.Dense(1, name='sentence_context')
        
        # Neurochemical modulation layers
        self.word_film = EnhancedFilMLayer(name='memory_word_film')
        self.sentence_film = EnhancedFilMLayer(name='memory_sentence_film')
        
    def call(self, event_sequence, neurochemical_state=None, training=None):
        """
        Enhanced memory processing with emotional/neurochemical context.
        
        Args:
            event_sequence: Sequence of events to process
            neurochemical_state: Current neurochemical levels
        
        Returns:
            context_vector: Emotionally-modulated memory representation
        """
        batch_size = tf.shape(event_sequence)[0]
        num_sentences = tf.shape(event_sequence)[1]
        num_words = tf.shape(event_sequence)[2]
        
        # Reshape for word-level processing
        reshaped_input = tf.reshape(event_sequence, (-1, num_words, event_sequence.shape[-1]))
        
        # Word-level encoding with neurochemical modulation
        word_encodings = self.word_encoder(reshaped_input)
        
        if neurochemical_state is not None:
            # Expand neurochemical state for sequence processing
            expanded_neurochemical = {}
            for key, value in neurochemical_state.items():
                # Expand to match word sequence dimensions
                expanded_value = tf.expand_dims(value, axis=1)
                expanded_value = tf.tile(expanded_value, [1, num_words])
                expanded_neurochemical[key] = expanded_value
            
            word_encodings = self.word_film([word_encodings, expanded_neurochemical])
        
        # Word-level attention
        word_att = self.word_attention(word_encodings)
        word_att_weights = tf.nn.softmax(self.word_context(word_att), axis=1)
        
        # Sentence representations
        sentence_vectors = tf.reduce_sum(word_encodings * word_att_weights, axis=1)
        sentence_vectors = tf.reshape(sentence_vectors, (batch_size, num_sentences, self.word_dim))
        
        # Sentence-level encoding with neurochemical modulation
        sentence_encodings = self.sentence_encoder(sentence_vectors)
        
        if neurochemical_state is not None:
            # Expand neurochemical state for sentence sequence processing
            expanded_neurochemical = {}
            for key, value in neurochemical_state.items():
                expanded_value = tf.expand_dims(value, axis=1)
                expanded_value = tf.tile(expanded_value, [1, num_sentences])
                expanded_neurochemical[key] = expanded_value
            
            sentence_encodings = self.sentence_film([sentence_encodings, expanded_neurochemical])
        
        # Sentence-level attention
        sent_att = self.sentence_attention(sentence_encodings)
        sent_att_weights = tf.nn.softmax(self.sentence_context(sent_att), axis=1)
        
        # Final context vector
        context_vector = tf.reduce_sum(sentence_encodings * sent_att_weights, axis=1)
        
        return context_vector


class DeliberativePathway(tf.keras.layers.Layer):
    """
    Enhanced Deliberative Pathway with comprehensive neurochemical modulation.
    
    Neurobiological Analogue: Prefrontal Cortex with Neurochemical Control
    - Complex cognitive processing modulated by full neurochemical milieu
    - Learning rates affected by dopamine and cortisol
    - Social processing enhanced by oxytocin
    - Attention controlled by norepinephrine
    """
    
    def __init__(self, lstm_units=128, action_dim=8, appraisal_dim=16, **kwargs):
        super(DeliberativePathway, self).__init__(**kwargs)
        
        self.lstm_units = lstm_units
        self.action_dim = action_dim
        self.appraisal_dim = appraisal_dim
        
        # Enhanced input processing with emotional context
        self.sensory_processor = layers.Dense(64, activation='relu', name='sensory_proc')
        self.salience_processor = layers.Dense(32, activation='relu', name='salience_proc')
        self.affect_processor = layers.Dense(32, activation='relu', name='affect_proc')
        self.context_processor = layers.Dense(64, activation='relu', name='context_proc')
        self.emotion_processor = layers.Dense(32, activation='relu', name='emotion_proc')
        
        # Main cognitive processing
        self.cognitive_lstm = layers.LSTM(self.lstm_units, return_sequences=False, name='cognitive_lstm')
        
        # Neurochemical modulation layer
        self.cognitive_film = EnhancedFilMLayer(name='cognitive_film')
        
        # Enhanced output layers
        self.action_output = layers.Dense(self.action_dim, activation='softmax', name='action_selection')
        self.appraisal_output = layers.Dense(self.appraisal_dim, activation='linear', name='cognitive_appraisal')
        
        # Learning rate modulation based on neurochemicals
        self.learning_rate_modulator = layers.Dense(1, activation='sigmoid', name='learning_rate_mod')
        
    def call(self, inputs, training=None):
        """
        Enhanced deliberative processing with full neurochemical integration.
        
        Args:
            inputs: Dictionary containing all input modalities and neurochemical state
        
        Returns:
            Enhanced outputs including learning rate modulation
        """
        sensory_input = inputs['sensory_input']
        salience = inputs['salience']
        core_affect = inputs['core_affect']
        context = inputs['context']
        emotion_state = inputs['emotion_state']
        neurochemical_state = inputs['neurochemical_state']
        
        # Enhanced input processing
        processed_sensory = self.sensory_processor(
            tf.reshape(sensory_input, (tf.shape(sensory_input)[0], -1))
        )
        processed_salience = self.salience_processor(salience)
        processed_affect = self.affect_processor(core_affect)
        processed_context = self.context_processor(context)
        processed_emotions = self.emotion_processor(emotion_state)
        
        # Integrate all information sources
        integrated_input = tf.concat([
            processed_sensory, processed_salience, processed_affect,
            processed_context, processed_emotions
        ], axis=-1)
        
        # Expand for LSTM processing
        integrated_input = tf.expand_dims(integrated_input, axis=1)
        
        # Main cognitive processing
        cognitive_output = self.cognitive_lstm(integrated_input)
        
        # Apply neurochemical modulation
        modulated_output = self.cognitive_film([cognitive_output, neurochemical_state])
        
        # Generate outputs
        action_probs = self.action_output(modulated_output)
        cognitive_appraisal = self.appraisal_output(modulated_output)
        
        # Compute learning rate modulation
        # Dopamine enhances learning for positive outcomes
        # Cortisol impairs learning under stress
        neurochemical_input = tf.concat([
            neurochemical_state['dopamine'],
            neurochemical_state['cortisol'],
            neurochemical_state['serotonin']
        ], axis=-1)
        learning_rate_modifier = self.learning_rate_modulator(neurochemical_input)
        
        return {
            'action': action_probs,
            'cognitive_appraisal': cognitive_appraisal,
            'learning_rate_modifier': learning_rate_modifier
        }


class EnhancedNECA(tf.keras.Model):
    """
    Enhanced Neuro-Emotive Cognitive Architecture with full neurochemical integration.
    
    Integrates discrete emotions (Plutchik's 8), neurochemical modulation,
    and dynamic learning rate adjustment for biologically-inspired AI.
    """
    
    def __init__(self, 
                 internal_state_dim=3,
                 core_affect_dim=2,
                 action_dim=8,
                 appraisal_dim=16,
                 **kwargs):
        super(EnhancedNECA, self).__init__(**kwargs)
        
        self.internal_state_dim = internal_state_dim
        self.core_affect_dim = core_affect_dim
        self.action_dim = action_dim
        self.appraisal_dim = appraisal_dim
        
        # Enhanced component modules
        self.ism = InteroceptiveStateModule(
            core_affect_dim=self.core_affect_dim,
            name='enhanced_ism'
        )
        
        self.reactive_pathway = ReactivePathway(name='enhanced_reactive')
        
        self.hrm = HierarchicalRecurrentMemory(name='enhanced_hrm')
        
        self.deliberative_pathway = DeliberativePathway(
            action_dim=self.action_dim,
            appraisal_dim=self.appraisal_dim,
            name='enhanced_deliberative'
        )
        
        # Reward prediction error computation for dopamine modulation
        self.reward_predictor = layers.Dense(1, activation='linear', name='reward_predictor')
        
    def call(self, inputs, previous_reward=None, training=None):
        """
        Enhanced forward pass with full neurochemical integration.
        
        Args:
            inputs: Input dictionary with event_sequence and internal_state
            previous_reward: Previous reward for dopamine modulation
            training: Training mode flag
        
        Returns:
            Comprehensive output including emotions and neurochemicals
        """
        event_sequence = inputs['event_sequence']
        internal_state = inputs['internal_state']
        social_context = inputs.get('social_context', None)
        
        # Extract current sensory input
        current_sensory_input = event_sequence[:, -1, -1, :]
        current_sensory_input = tf.expand_dims(tf.expand_dims(current_sensory_input, axis=1), axis=1)
        if len(current_sensory_input.shape) == 3:
            current_sensory_input = tf.expand_dims(current_sensory_input, axis=-1)
        
        # Compute reward prediction error if previous reward available
        if previous_reward is not None:
            predicted_reward = self.reward_predictor(tf.reshape(current_sensory_input, (tf.shape(current_sensory_input)[0], -1)))
            reward_prediction_error = previous_reward - tf.squeeze(predicted_reward)
        else:
            reward_prediction_error = tf.zeros((tf.shape(current_sensory_input)[0],))
        
        # Process internal state through enhanced ISM
        ism_output = self.ism(
            internal_state, 
            reward_prediction_error,
            social_context,
            training=training
        )
        
        core_affect = ism_output['core_affect']
        emotion_state = ism_output['emotion_state']
        emotion_valence = ism_output['emotion_valence']
        neurochemical_state = ism_output['neurochemical_state']
        
        # Process through reactive pathway with neurochemical modulation
        salience = self.reactive_pathway(
            current_sensory_input, 
            neurochemical_state, 
            training=training
        )
        
        # Process through hierarchical memory with neurochemical modulation
        context = self.hrm(
            event_sequence, 
            neurochemical_state, 
            training=training
        )
        
        # Process through deliberative pathway
        deliberative_inputs = {
            'sensory_input': current_sensory_input,
            'salience': salience,
            'core_affect': core_affect,
            'context': context,
            'emotion_state': emotion_state,
            'neurochemical_state': neurochemical_state
        }
        
        deliberative_output = self.deliberative_pathway(
            deliberative_inputs, 
            training=training
        )
        
        return {
            'action': deliberative_output['action'],
            'core_affect': core_affect,
            'emotion_state': emotion_state,
            'emotion_valence': emotion_valence,
            'neurochemical_state': neurochemical_state,
            'cognitive_appraisal': deliberative_output['cognitive_appraisal'],
            'learning_rate_modifier': deliberative_output['learning_rate_modifier'],
            'salience': salience,
            'context': context,
            'reward_prediction_error': reward_prediction_error
        }
    
    def get_enhanced_reward_signal(self, outputs):
        """
        Enhanced reward signal incorporating emotions and neurochemical balance.
        
        This represents the key biological insight that emotional well-being,
        neurochemical balance, and homeostatic stability serve as internal
        reward signals for learning.
        
        Args:
            outputs: Full model outputs including emotions and neurochemicals
        
        Returns:
            reward: Multi-component reward signal
        """
        # Basic emotional valence reward
        emotion_reward = outputs['emotion_valence']
        
        # Neurochemical balance reward
        neurochemical_state = outputs['neurochemical_state']
        
        # Reward balanced neurochemical levels (avoid extremes)
        dopamine_balance = 1.0 - tf.abs(neurochemical_state['dopamine'] - 0.6)  # Optimal dopamine ~0.6
        serotonin_balance = neurochemical_state['serotonin']  # Higher serotonin is better
        norepinephrine_penalty = -0.3 * tf.maximum(0.0, neurochemical_state['norepinephrine'] - 0.7)  # Penalize high stress
        cortisol_penalty = -0.5 * neurochemical_state['cortisol']  # Penalize stress hormone
        oxytocin_bonus = 0.2 * neurochemical_state['oxytocin']  # Reward social bonding
        
        neurochemical_reward = (
            dopamine_balance + serotonin_balance + 
            norepinephrine_penalty + cortisol_penalty + oxytocin_bonus
        ) / 5.0  # Normalize
        
        # Homeostatic stability reward (penalize extreme changes)
        core_affect = outputs['core_affect']
        stability_reward = -0.1 * tf.reduce_mean(tf.square(core_affect), axis=-1, keepdims=True)
        
        # Combined reward signal
        total_reward = (
            0.4 * emotion_reward +           # 40% emotional valence
            0.4 * neurochemical_reward +     # 40% neurochemical balance
            0.2 * stability_reward           # 20% homeostatic stability
        )
        
        return tf.squeeze(total_reward)
    
    def get_emotion_specific_learning_rates(self, outputs):
        """
        Compute emotion-specific learning rate modulations.
        
        Different emotions should lead to different learning strategies:
        - Joy/Trust: Consolidate positive experiences (higher learning rate)
        - Fear/Disgust: Rapid avoidance learning (very high learning rate)
        - Sadness: Reduced learning (lower learning rate)
        - Surprise: Enhanced learning for unexpected events
        
        Returns:
            Dictionary of learning rate modifiers for different systems
        """
        emotion_state = outputs['emotion_state']
        neurochemical_state = outputs['neurochemical_state']
        
        # Base learning rate modifier from deliberative pathway
        base_modifier = outputs['learning_rate_modifier']
        
        # Emotion-specific modulations
        joy_boost = emotion_state[:, 0] * 0.3        # Joy enhances learning
        fear_boost = emotion_state[:, 3] * 0.8       # Fear drives rapid learning
        sadness_reduction = emotion_state[:, 1] * -0.4  # Sadness reduces learning
        surprise_boost = emotion_state[:, 6] * 0.5   # Surprise enhances learning
        
        emotion_modifier = tf.expand_dims(
            joy_boost + fear_boost + sadness_reduction + surprise_boost, axis=-1
        )
        
        # Neurochemical modulations
        dopamine_boost = neurochemical_state['dopamine'] * 0.2  # Dopamine enhances learning
        cortisol_reduction = neurochemical_state['cortisol'] * -0.3  # Cortisol impairs learning
        
        neurochemical_modifier = dopamine_boost + cortisol_reduction
        
        # Combined learning rate
        final_learning_rate = base_modifier + emotion_modifier + neurochemical_modifier
        final_learning_rate = tf.nn.sigmoid(final_learning_rate)  # Keep in [0,1] range
        
        return {
            'overall': final_learning_rate,
            'emotional': base_modifier + emotion_modifier,
            'neurochemical': base_modifier + neurochemical_modifier,
            'base': base_modifier
        }


def create_enhanced_dummy_data(batch_size=2):
    """
    Create enhanced dummy input data including social context.
    
    Returns:
        Dictionary with enhanced dummy input tensors
    """
    # Event sequence: (batch_size, num_sentences, num_words, word_dim)
    event_sequence = tf.random.normal((batch_size, 5, 10, 64))
    
    # Internal state: (batch_size, sequence_length, internal_state_dim)
    # Represents: [energy_level, physiological_integrity, stress_load]
    internal_state = tf.random.normal((batch_size, 10, 3))
    
    # Social context (optional): presence of others, social support, etc.
    social_context = tf.random.normal((batch_size, 8))
    
    return {
        'event_sequence': event_sequence,
        'internal_state': internal_state,
        'social_context': social_context
    }


def analyze_emotional_state(emotion_state, neurochemical_state):
    """
    Analyze and interpret the current emotional and neurochemical state.
    
    Args:
        emotion_state: Probability distribution over 8 emotions
        neurochemical_state: Dictionary of neurochemical levels
    
    Returns:
        Human-readable analysis of the emotional state
    """
    emotion_names = [
        'Joy', 'Sadness', 'Anger', 'Fear', 
        'Trust', 'Disgust', 'Surprise', 'Anticipation'
    ]
    
    # Find dominant emotions
    dominant_emotion_idx = tf.argmax(emotion_state, axis=-1).numpy()
    emotion_probs = emotion_state.numpy()
    
    analysis = []
    
    for i in range(emotion_state.shape[0]):
        dominant_emotion = emotion_names[dominant_emotion_idx[i]]
        dominant_prob = emotion_probs[i, dominant_emotion_idx[i]]
        
        # Neurochemical analysis
        dopamine = neurochemical_state['dopamine'][i].numpy()[0]
        serotonin = neurochemical_state['serotonin'][i].numpy()[0]
        norepinephrine = neurochemical_state['norepinephrine'][i].numpy()[0]
        cortisol = neurochemical_state['cortisol'][i].numpy()[0]
        oxytocin = neurochemical_state['oxytocin'][i].numpy()[0]
        
        sample_analysis = f"""
Sample {i+1} Emotional Analysis:
- Dominant Emotion: {dominant_emotion} ({dominant_prob:.3f})
- Neurochemical Profile:
  * Dopamine (motivation): {dopamine:.3f}
  * Serotonin (well-being): {serotonin:.3f}
  * Norepinephrine (arousal): {norepinephrine:.3f}
  * Cortisol (stress): {cortisol:.3f}
  * Oxytocin (social bonding): {oxytocin:.3f}
- Emotional State: """
        
        if dominant_emotion in ['Joy', 'Trust']:
            sample_analysis += "Positive emotional state, likely to enhance learning"
        elif dominant_emotion in ['Fear', 'Anger']:
            sample_analysis += "Negative but activating, may enhance threat-related learning"
        elif dominant_emotion in ['Sadness', 'Disgust']:
            sample_analysis += "Negative emotional state, may reduce overall learning"
        elif dominant_emotion == 'Surprise':
            sample_analysis += "Neutral emotion, context-dependent valence"
        else:  # Anticipation
            sample_analysis += "Anticipatory state, heightened attention"
        
        analysis.append(sample_analysis)
    
    return analysis


# =============================================================================
# MAIN EXECUTION BLOCK WITH ENHANCED EMOTIONAL ANALYSIS
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED NEURO-EMOTIVE COGNITIVE ARCHITECTURE (NECA)")
    print("WITH NEUROCHEMICAL MODULATION AND PLUTCHIK'S EMOTIONS")
    print("=" * 80)
    print()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create Enhanced NECA model
    print("Initializing Enhanced NECA model...")
    neca_model = EnhancedNECA(
        internal_state_dim=3,   # energy_level, integrity, stress_load
        core_affect_dim=2,      # valence, arousal
        action_dim=8,           # number of possible actions
        appraisal_dim=16        # dimension of cognitive appraisal vector
    )
    
    # Create enhanced dummy input data
    print("Creating enhanced dummy input data...")
    dummy_inputs = create_enhanced_dummy_data(batch_size=2)
    
    print(f"Event sequence shape: {dummy_inputs['event_sequence'].shape}")
    print(f"Internal state shape: {dummy_inputs['internal_state'].shape}")
    print(f"Social context shape: {dummy_inputs['social_context'].shape}")
    print()
    
    # Perform forward pass
    print("Performing forward pass through Enhanced NECA...")
    outputs = neca_model(dummy_inputs, training=False)
    
    # Display basic outputs
    print("\nBasic NECA Outputs:")
    print("-" * 40)
    print(f"Action probabilities shape: {outputs['action'].shape}")
    print(f"Action probabilities:\n{outputs['action'].numpy()}")
    print()
    
    print(f"Core affect (valence, arousal) shape: {outputs['core_affect'].shape}")
    print(f"Core affect values:\n{outputs['core_affect'].numpy()}")
    print()
    
    # Display emotional analysis
    print("\nEMOTIONAL STATE ANALYSIS:")
    print("-" * 40)
    emotion_analyses = analyze_emotional_state(
        outputs['emotion_state'], 
        outputs['neurochemical_state']
    )
    for analysis in emotion_analyses:
        print(analysis)
    
    # Display enhanced reward signals
    print("\nENHANCED REWARD SYSTEM:")
    print("-" * 40)
    enhanced_reward = neca_model.get_enhanced_reward_signal(outputs)
    print(f"Enhanced reward signal: {enhanced_reward.numpy()}")
    
    learning_rates = neca_model.get_emotion_specific_learning_rates(outputs)
    print(f"Overall learning rate modifier: {learning_rates['overall'].numpy().flatten()}")
    print(f"Emotional component: {learning_rates['emotional'].numpy().flatten()}")
    print(f"Neurochemical component: {learning_rates['neurochemical'].numpy().flatten()}")
    print()
    
    # Model architecture summary
    print("Enhanced Model Architecture Summary:")
    print("-" * 40)
    neca_model.build(input_shape={
        'event_sequence': (None, 5, 10, 64),
        'internal_state': (None, 10, 3),
        'social_context': (None, 8)
    })
    
    total_params = sum([tf.size(w).numpy() for w in neca_model.trainable_weights])
    print(f"Total trainable parameters: {total_params:,}")
    print()
    
    # Demonstrate neurochemical dynamics
    print("NEUROCHEMICAL SYSTEM DEMONSTRATION:")
    print("-" * 40)
    print("Simulating different emotional scenarios...")
    
    # Create scenarios with different emotional profiles
    scenarios = [
        # Scenario 1: High stress (fear, cortisol)
        {'name': 'High Stress Scenario', 
         'internal_state': tf.constant([[[-1.0, -0.8, 2.0]]], dtype=tf.float32)},
        # Scenario 2: Positive social interaction (joy, oxytocin)
        {'name': 'Positive Social Scenario',
         'internal_state': tf.constant([[[1.0, 0.8, -0.5]]], dtype=tf.float32)},
        # Scenario 3: Neutral/balanced state
        {'name': 'Balanced Scenario',
         'internal_state': tf.constant([[[0.0, 0.0, 0.0]]], dtype=tf.float32)}
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        scenario_input = dummy_inputs.copy()
        scenario_input['internal_state'] = tf.tile(
            scenario['internal_state'], [1, 10, 1]
        )
        scenario_input = {k: v[:1] for k, v in scenario_input.items()}  # Use batch size 1
        
        scenario_output = neca_model(scenario_input, training=False)
        scenario_reward = neca_model.get_enhanced_reward_signal(scenario_output)
        
        dominant_emotion_idx = tf.argmax(scenario_output['emotion_state']).numpy()
        emotion_names = ['Joy', 'Sadness', 'Anger', 'Fear', 'Trust', 'Disgust', 'Surprise', 'Anticipation']
        dominant_emotion = emotion_names[dominant_emotion_idx]
        
        print(f"  Dominant Emotion: {dominant_emotion}")
        print(f"  Reward Signal: {scenario_reward.numpy()[0]:.3f}")
        print(f"  Dopamine: {scenario_output['neurochemical_state']['dopamine'].numpy()[0,0]:.3f}")
        print(f"  Cortisol: {scenario_output['neurochemical_state']['cortisol'].numpy()[0,0]:.3f}")
        print(f"  Serotonin: {scenario_output['neurochemical_state']['serotonin'].numpy()[0,0]:.3f}")
    
    # =============================================================================
    # ENHANCED REINFORCEMENT LEARNING FRAMEWORK
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("ENHANCED REINFORCEMENT LEARNING FRAMEWORK")
    print("=" * 80)
    print("""
The Enhanced NECA model incorporates sophisticated emotional and neurochemical
dynamics that create a multi-layered learning system inspired by biological
emotion regulation and neurochemical modulation.

KEY ENHANCEMENTS:

1. DISCRETE EMOTIONAL STATES (Plutchik's 8 Primary Emotions):
   - Joy, Sadness, Anger, Fear, Trust, Disgust, Surprise, Anticipation
   - Each emotion has different learning implications:
     * Positive emotions (Joy, Trust) → Enhanced consolidation of successful behaviors
     * Negative emotions (Fear, Disgust) → Rapid avoidance learning
     * Activating emotions (Anger, Surprise) → Heightened attention and learning
     * Deactivating emotions (Sadness) → Reduced learning rate, conservation mode

2. NEUROCHEMICAL MODULATION SYSTEM:
   - Dopamine: Reward prediction, motivation, attention to rewarding stimuli
   - Serotonin: Mood stabilization, reduces learning volatility
   - Norepinephrine: Arousal, attention to salient/threatening stimuli
   - Cortisol: Stress response, impairs deliberative processing, biases toward reactive responses
   - Oxytocin: Social bonding, increases trust, reduces threat sensitivity
   - Endocannabinoids: Homeostatic regulation, prevents runaway neurochemical responses

3. DYNAMIC LEARNING RATE MODULATION:
   - Base learning rate modified by emotional state and neurochemical balance
   - Fear/disgust → Very high learning rate for threat avoidance
   - Joy/trust → High learning rate for positive behavior reinforcement
   - Sadness → Reduced learning rate, energy conservation
   - High cortisol → Impaired learning, stress response mode
   - High dopamine → Enhanced learning for rewarding outcomes

4. MULTI-COMPONENT REWARD SYSTEM:
   - Emotional valence (40%): Positive emotions rewarded, negative penalized
   - Neurochemical balance (40%): Rewards balanced neurochemistry, avoids extremes
   - Homeostatic stability (20%): Penalizes extreme emotional swings

5. CONTEXT-DEPENDENT PROCESSING:
   - Social context influences oxytocin levels and trust-related behaviors
   - Threat context enhances norepinephrine and fear-based learning
   - Reward context boosts dopamine and approach behaviors

BIOLOGICAL TRAINING PRINCIPLES:

1. EMOTION-DRIVEN LEARNING:
   - Different emotions create different learning contexts
   - The system learns not just what to do, but when different strategies are appropriate
   - Emotional states serve as internal context signals for behavioral flexibility

2. NEUROCHEMICAL HOMEOSTASIS:
   - The system learns to maintain balanced neurochemical states
   - Actions that restore neurochemical balance are reinforced
   - This creates intrinsic motivation for emotional well-being

3. TEMPORAL DYNAMICS:
   - Fast neurochemical changes (milliseconds-seconds): Dopamine, norepinephrine
   - Medium changes (minutes): Serotonin modulation, emotional state transitions
   - Slow changes (hours): Cortisol stress response, long-term mood regulation

4. ADAPTIVE LEARNING RATES:
   - Learning rate automatically adjusts based on emotional and neurochemical context
   - High-stakes situations (fear, threat) → Rapid learning
   - Positive situations (joy, reward) → Consolidation learning
   - Stress situations (high cortisol) → Reduced complex learning, bias toward habits

TRAINING ENVIRONMENT REQUIREMENTS:

1. MULTI-MODAL SENSORY INPUT:
   - Visual, auditory, and interoceptive (internal body state) information
   - Social cues for oxytocin and trust-related learning

2. PHYSIOLOGICAL SIMULATION:
   - Internal energy, stress, and health states that change based on actions
   - Physiological needs that create drive states

3. SOCIAL INTERACTION:
   - Other agents or humans for social learning and oxytocin modulation
   - Cooperation and competition scenarios

4. DIVERSE EMOTIONAL CONTEXTS:
   - Threat scenarios for fear-based learning
   - Reward scenarios for dopamine-driven learning
   - Social bonding scenarios for oxytocin effects
   - Stress scenarios for cortisol dynamics

PRACTICAL IMPLEMENTATION:

The enhanced NECA can be trained using standard RL algorithms (PPO, A3C, SAC) with
the following modifications:

1. Use emotion-specific learning rates for different parts of the network
2. Include neurochemical balance in the reward function
3. Implement experience replay that weights memories based on emotional salience
4. Use curriculum learning that gradually introduces more complex emotional scenarios

This creates an AI system that doesn't just maximize reward, but learns to maintain
emotional and neurochemical well-being while adapting to complex, dynamic environments.
The result is more robust, flexible, and biologically plausible artificial intelligence.
""")
    
    print("Enhanced NECA demonstration completed successfully!")
    print("=" * 80)
