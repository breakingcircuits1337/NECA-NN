The model integrates several modules that work together to process information, generate emotional responses, and make decisions. Unlike traditional AI models that often focus on a single objective, NECA is designed to maintain a state of emotional and neurochemical well-being, making its behavior more adaptive and realistic.

The architecture is built around a few key principles:

    Dual-Process Cognition: It has separate "fast" reactive and "slow" deliberative pathways, mimicking instinctive and thoughtful brain functions.

    Emotional Foundation: It uses Plutchik's 8 primary emotions (joy, trust, fear, surprise, sadness, disgust, anger, and anticipation) as a core component for evaluating situations and guiding behavior.

    Neurochemical Modulation: The entire system is influenced by a simulated "chemical soup" of key neurochemicals, which dynamically alters how the network processes information.
  Key Components Explained

Here’s a look at what each major class in the code does:

🧠 Interoceptive State Module (InteroceptiveStateModule)

This module acts like the brain's insula or gut-brain axis. It takes in the agent's basic physiological state (e.g., energy levels, stress) and generates a "core affect"—a fundamental feeling described by valence (positive/negative) and arousal (high/low energy). This core affect is then interpreted by the EmotionalStateProcessor to produce a specific emotional state (like fear or joy).

🧪 Neurochemical System (NeurochemicalSystem)

This is the heart of the model's biological realism. It simulates the levels of crucial brain chemicals:

    Dopamine: The "motivation" chemical, tied to reward prediction.

    Serotonin: The "well-being" chemical, for mood stability.

    Norepinephrine: The "alertness" chemical, for arousal and attention.

    Cortisol: The "stress" hormone, which can suppress complex thought.

    Oxytocin: The "social bonding" chemical, influenced by social context.

    Endocannabinoids: A regulatory system that keeps the other chemicals in balance (homeostasis).

The levels of these chemicals are determined by the agent's emotional state and context, and they, in turn, influence the rest of the network.

⚡ Reactive and Deliberative Pathways

The model processes sensory information via two distinct routes, much like the human brain:

    Reactive Pathway (ReactivePathway): This is the "low road" (thalamo-amygdala pathway). It uses simple convolutional layers for a fast, rough analysis of sensory input to quickly detect potential threats or important stimuli (salience). It's heavily influenced by norepinephrine (arousal) and cortisol (stress).

    Deliberative Pathway (DeliberativePathway): This is the "high road," analogous to the prefrontal cortex. It integrates all available information—sensory input, emotional state, memory, and salience—to perform complex reasoning and select a final action. Its processing is modulated by the full suite of neurochemicals.

🎛️ Enhanced FilM Layer (EnhancedFilMLayer)

FilM (Feature-wise Linear Modulation) is the mechanism that allows the neurochemicals to exert their influence. This special layer acts as a set of dynamic "knobs" or "dials" on the neural pathways. For example:

    High cortisol might "turn down the volume" on deliberative thought.

    High dopamine might "turn up the volume" on features related to a potential reward.

This allows the model's processing style to change fluidly based on its neuro-emotive state.

Learning and Decision-Making

The most innovative aspect of NECA is its approach to learning, which is guided by two advanced concepts:

1. Multi-Component Reward System

The model isn't just trying to maximize an external score. Its internal reward signal is a weighted combination of:

    Emotional Valence (40%): The agent feels "good" for being in positive emotional states.

    Neurochemical Balance (40%): It's rewarded for keeping its brain chemistry in a healthy, stable range.

    Homeostatic Stability (20%): It's penalized for wild swings in emotion, promoting stable behavior.

2. Dynamic, Emotion-Specific Learning

The model's learning rate isn't fixed. It changes based on the situation:

    High Fear/Surprise: The learning rate skyrockets to quickly adapt to new or threatening information.

    High Joy/Trust: The learning rate is high to consolidate successful strategies.

    High Sadness: The learning rate is reduced, promoting a state of energy conservation.

    High Cortisol (Stress): The learning rate for complex tasks is impaired, biasing the agent toward simple, reactive habits.

This allows the agent to learn not just what to do, but to apply different learning strategies in different emotional contexts, making it highly adaptive.
