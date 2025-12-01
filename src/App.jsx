import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Search, 
  Clock, 
  ArrowLeft, 
  MoreHorizontal, 
  FileText, 
  MessageSquare, 
  Plus, 
  X, 
  BookOpen, 
  Bookmark, 
  Share2, 
  Download, 
  User, 
  TrendingUp, 
  ChevronDown, 
  ChevronUp, 
  Quote, 
  ExternalLink,
  Bot,
  Activity,
  Sparkles,
  GitBranch, 
  GitCommit, 
  ArrowRight, 
  ArrowUp,
  Dna, 
  Microscope, 
  Layers, 
  ZoomIn,
  XCircle,
  CheckCircle2,
  Lightbulb,
  History,
  Filter
} from 'lucide-react';

// --- MOCK DATA FOR HAYSTACK ---

const MOCK_AUTHORS = {
  "1": { id: "1", name: "Dr. Sarah Chen", cScore: 45, affiliation: "MIT CSAIL", img: "https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah" },
  "2": { id: "2", name: "J. Smith", cScore: 32, affiliation: "Stanford", img: "https://api.dicebear.com/7.x/avataaars/svg?seed=Smith" },
  "3": { id: "3", name: "A. Patel", cScore: 28, affiliation: "Oxford", img: "https://api.dicebear.com/7.x/avataaars/svg?seed=Patel" },
  "4": { id: "4", name: "M. Johnson", cScore: 55, affiliation: "Harvard", img: "https://api.dicebear.com/7.x/avataaars/svg?seed=Johnson" },
  "5": { id: "5", name: "L. Wei", cScore: 41, affiliation: "Tsinghua", img: "https://api.dicebear.com/7.x/avataaars/svg?seed=Wei" },
  "6": { id: "6", name: "Dr. Elena Ricci", cScore: 48, affiliation: "ETH Zurich", img: "https://api.dicebear.com/7.x/avataaars/svg?seed=Elena" },
  "7": { id: "7", name: "Dr. Kenji Tanaka", cScore: 52, affiliation: "Broad Institute", img: "https://api.dicebear.com/7.x/avataaars/svg?seed=Kenji" },
  "8": { id: "8", name: "Dr. Fatima Rossi", cScore: 49, affiliation: "Caltech", img: "https://api.dicebear.com/7.x/avataaars/svg?seed=Rossi" }
};

const MOCK_PAPERS = [
  {
    id: 1,
    title: "Generative Agents: Interactive Simulacra of Human Behavior",
    date: "April 2023",
    authors: ["1", "2", "3", "4", "5"],
    journal: "arXiv",
    findings: [
      "Agents produce believable individual and emergent social behaviors.",
      "Memory stream architecture enables reflection and planning.",
      "Evaluation shows agents outperform standard LLMs in consistency."
    ],
    tags: ["Highly influential", "Trending", "Agentic AI"],
    concepts: ["Memory Streams", "Agent Simulation", "Emergent Behavior"],
    abstract: "In this work, we demonstrate generative agents—computational software agents that simulate believable human behavior. Generative agents wake up, cook breakfast, and head to work; artists paint, while authors write; they form opinions, notice each other, and initiate conversations; they remember and reflect on days past as they plan the next day.",
    citations: 1243,
    figures: [
        { title: "Architecture Diagram", url: "https://placehold.co/600x400/e0e7ff/4f46e5?text=Architecture+Diagram" },
        { title: "Interaction Graph", url: "https://placehold.co/600x400/fae8ff/86198f?text=Interaction+Graph" }
    ],
    score: {
      total: 94,
      breakdown: { impact: 28, author: 32, citations: 34 }
    }
  },
  {
    id: 2,
    title: "Attention Is All You Need",
    date: "June 2017",
    authors: ["3", "4", "2", "1"],
    journal: "NeurIPS",
    findings: [
      "Proposed the Transformer model based solely on attention mechanisms.",
      "Achieved state-of-the-art results on English-to-German translation.",
      "Significantly reduced training time compared to recurrent layers."
    ],
    tags: ["Seminal Work", "Transformer", "NLP"],
    concepts: ["Self-Attention", "Transformers", "Positional Encoding"],
    abstract: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
    citations: 85000,
    figures: [
        { title: "Transformer Model", url: "https://placehold.co/600x400/e0e7ff/4f46e5?text=Transformer+Model" },
        { title: "Attention Heads", url: "https://placehold.co/600x400/dcfce7/166534?text=Attention+Heads" }
    ],
    score: {
      total: 98,
      breakdown: { impact: 30, author: 33, citations: 35 }
    }
  },
  {
    id: 3,
    title: "A Survey on Large Language Models",
    date: "March 2023",
    authors: ["5", "1"],
    journal: "Nature Machine Intelligence",
    findings: [
      "Comprehensive review of LLM development history.",
      "Analyzes evolutionary tree of language models.",
      "Discusses emergent abilities not present in smaller models."
    ],
    tags: ["Literature review", "LLM", "Survey"],
    concepts: ["Large Language Models", "Emergent Abilities", "Prompting"],
    abstract: "Large Language Models (LLMs) have drawn a lot of attention due to their strong performance on a wide range of natural language processing tasks. This survey provides a comprehensive review of the recent advances in LLMs.",
    citations: 890,
    figures: [
        { title: "LLM Evolution Tree", url: "https://placehold.co/600x400/ffedd5/9a3412?text=LLM+Evolution+Tree" }
    ],
    score: {
      total: 82,
      breakdown: { impact: 32, author: 28, citations: 22 }
    }
  },
  {
    id: 4,
    title: "Constitutional AI: Harmlessness from AI Feedback",
    date: "December 2022",
    authors: ["2", "3", "4", "1"],
    journal: "Anthropic Research",
    findings: [
      "Uses AI feedback to train a harmless AI assistant.",
      "Reduces reliance on human labels for safety training.",
      "Introduces the concept of a constitution for AI behavior."
    ],
    tags: ["Safety", "Alignment", "RLAIF"],
    concepts: ["RLAIF", "AI Safety", "Constitutional AI"],
    abstract: "We explore the use of AI feedback for training harmless AI assistants. We train a preference model to predict which of two responses is better according to a set of principles, and then fine-tune a language model to maximize this preference model.",
    citations: 450,
    figures: [
        { title: "RLAIF Process", url: "https://placehold.co/600x400/f1f5f9/475569?text=RLAIF+Process" }
    ],
    score: {
      total: 76,
      breakdown: { impact: 20, author: 30, citations: 26 }
    }
  },
  {
    id: 5,
    title: "Chain-of-Thought Prompting Elicits Reasoning",
    date: "January 2022",
    authors: ["1", "5", "3"],
    journal: "NeurIPS",
    findings: [
      "Improves reasoning abilities of large language models.",
      "Demonstrates that scale enables complex reasoning.",
      "Simple prompts can trigger complex multi-step logic."
    ],
    tags: ["Prompt Engineering", "Reasoning", "Trending"],
    concepts: ["Chain-of-Thought", "Reasoning", "Few-Shot Learning"],
    abstract: "We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models.",
    citations: 3200,
    figures: [
        { title: "CoT Examples", url: "https://placehold.co/600x400/f3e8ff/6b21a8?text=CoT+Examples" }
    ],
    score: {
      total: 89,
      breakdown: { impact: 30, author: 32, citations: 27 }
    }
  },
  {
    id: 6,
    title: "Deep Reinforcement Learning from Human Preferences",
    date: "June 2017",
    authors: ["4", "1"],
    journal: "NIPS",
    findings: [
      "Enables RL without a defined reward function.",
      "Uses human feedback on short clips of agent behavior.",
      "Scales to complex tasks like Atari games and robotics."
    ],
    tags: ["RLHF", "Reinforcement Learning"],
    concepts: ["RLHF", "Reward Modeling", "Human Preferences"],
    abstract: "For many tasks, it is difficult to communicate a precise objective to an agent. We explore deep reinforcement learning from human preferences, allowing agents to solve tasks with no access to a reward function.",
    citations: 2100,
    figures: [
        { title: "Preference Interface", url: "https://placehold.co/600x400/ecfccb/3f6212?text=Preference+Interface" }
    ],
    score: {
      total: 85,
      breakdown: { impact: 29, author: 31, citations: 25 }
    }
  },
    {
    id: 7,
    title: "Integrated Assessment of Global Climate Scenarios: A Probabilistic Approach",
    date: "August 2022",
    authors: ["6", "2"],
    journal: "Nature Climate Change",
    findings: [
      "Probabilistic models provide robust projections of climate futures.",
      "Identifies key policy levers for mitigating worst-case scenarios.",
      "Highlights the economic impacts of delayed climate action."
    ],
    tags: ["Climate Modeling", "Policy Impact", "Economics"],
    concepts: ["Integrated Assessment Models", "Probabilistic Forecasting", "Climate Economics"],
    abstract: "This paper presents a novel probabilistic framework for integrated assessment models (IAMs) to evaluate climate change scenarios. By incorporating uncertainty across socio-economic and climate systems, we provide more robust projections and analyze the efficacy of different policy interventions under a range of possible futures.",
    citations: 530,
    figures: [
        { title: "Scenario Funnel Plot", url: "https://placehold.co/600x400/dbeafe/1e3a8a?text=Scenario+Funnel+Plot" }
    ],
    score: {
      total: 88,
      breakdown: { impact: 31, author: 29, citations: 28 }
    }
  },
  {
    id: 8,
    title: "High-throughput screening of CRISPR-Cas9 guide RNA efficacy",
    date: "May 2021",
    authors: ["7", "5", "1"],
    journal: "Cell",
    findings: [
      "Develops a deep learning model to predict gRNA on-target efficacy.",
      "Demonstrates >95% accuracy in predicting highly active gRNAs.",
      "Enables genome-wide library design with high confidence."
    ],
    tags: ["CRISPR", "Deep Learning", "Genomics"],
    concepts: ["Guide RNA", "Gene Editing", "High-throughput Screening"],
    abstract: "The efficacy of CRISPR-Cas9 gene editing is critically dependent on the choice of guide RNA (gRNA). We developed a deep learning model, trained on a large-scale dataset of gRNA activity, that accurately predicts on-target efficacy. This enables the design of highly efficient gRNAs for various applications.",
    citations: 1500,
    figures: [
        { title: "Model Architecture", url: "https://placehold.co/600x400/fee2e2/991b1b?text=Model+Architecture" }
    ],
    score: {
      total: 91,
      breakdown: { impact: 33, author: 30, citations: 28 }
    }
  },
  {
    id: 9,
    title: "Fault-Tolerant Quantum Computation with Stabilizer Codes",
    date: "July 2020",
    authors: ["8", "4"],
    journal: "Physical Review Letters",
    findings: [
      "Presents a new class of stabilizer codes for error correction.",
      "Demonstrates a significant reduction in qubit overhead.",
      "Achieves a lower logical error rate for a given physical error rate."
    ],
    tags: ["Quantum Computing", "Error Correction", "Qubits"],
    concepts: ["Stabilizer Codes", "Quantum Error Correction", "Fault Tolerance"],
    abstract: "Fault-tolerant quantum computation is essential for building large-scale quantum computers. We introduce a new family of quantum stabilizer codes that offer improved performance over existing codes, reducing the resource overhead required for fault tolerance and bringing practical quantum computation a step closer.",
    citations: 980,
    figures: [
        { title: "Code Performance", url: "https://placehold.co/600x400/eef2ff/4338ca?text=Code+Performance" }
    ],
    score: {
      total: 86,
      breakdown: { impact: 30, author: 29, citations: 27 }
    }
  },
  {
    id: 10,
    title: "The Role of Ocean Currents in Arctic Ice Melt Acceleration",
    date: "January 2023",
    authors: ["6", "3"],
    journal: "Science",
    findings: [
      "Warmer Atlantic currents are penetrating deeper into the Arctic Ocean.",
      "This influx of heat is a primary driver of basal melt of sea ice.",
      "Model projections show a potential for an ice-free Arctic summer by 2040."
    ],
    tags: ["Climate Change", "Oceanography", "Arctic"],
    concepts: ["Ocean Currents", "Sea Ice Melt", "Climate Projections"],
    abstract: "We combine satellite data and ocean models to demonstrate the increasing influence of Atlantic water influx on Arctic sea ice. Our findings reveal that warmer, saltier currents are accelerating the melting process from below, a factor previously underestimated in climate models, with significant implications for global climate patterns.",
    citations: 750,
    figures: [
        { title: "Current Influx Map", url: "https://placehold.co/600x400/dcfce7/14532d?text=Current+Influx+Map" }
    ],
    score: {
      total: 90,
      breakdown: { impact: 34, author: 28, citations: 28 }
    }
  },
  {
    id: 11,
    title: "Improving CRISPR-Cas9 specificity using novel guide RNA designs",
    date: "November 2022",
    authors: ["7", "1"],
    journal: "Nature Biotechnology",
    findings: [
      "Engineered 'se-gRNAs' significantly reduce off-target effects.",
      "Maintains high on-target activity across multiple gene loci.",
      "Provides a safer approach for therapeutic gene editing applications."
    ],
    tags: ["CRISPR", "Gene Therapy", "Safety"],
    concepts: ["Off-target Effects", "Guide RNA Engineering", "Specificity"],
    abstract: "Off-target mutations are a major concern for the therapeutic use of CRISPR-Cas9. We have engineered specificity-enhanced guide RNAs (se-gRNAs) that dramatically reduce off-target editing while preserving high on-target efficiency. This work represents a significant step towards safer clinical applications of gene editing.",
    citations: 1100,
    figures: [
        { title: "Specificity Comparison", url: "https://placehold.co/600x400/fce7f3/831843?text=Specificity+Comparison" }
    ],
    score: {
      total: 93,
      breakdown: { impact: 33, author: 32, citations: 28 }
    }
  },
  {
    id: 12,
    title: "Decoherence in Superconducting Qubits: A Review",
    date: "September 2021",
    authors: ["8", "5", "2"],
    journal: "Reviews of Modern Physics",
    findings: [
      "Comprehensive survey of decoherence mechanisms in superconducting qubits.",
      "Analyzes the impact of material defects, radiation, and magnetic fields.",
      "Discusses state-of-the-art mitigation strategies and future directions."
    ],
    tags: ["Quantum Computing", "Superconductivity", "Review"],
    concepts: ["Decoherence", "Qubits", "Superconducting Circuits"],
    abstract: "Decoherence—the loss of quantum information—is the primary obstacle to building scalable quantum computers. This review provides a detailed overview of the various sources of decoherence affecting superconducting qubits, the leading platform for quantum computation. We summarize the current understanding and outline strategies for improving qubit coherence times.",
    citations: 1800,
    figures: [
        { title: "Decoherence Sources", url: "https://placehold.co/600x400/f0f9ff/075985?text=Decoherence+Sources" }
    ],
    score: {
      total: 89,
      breakdown: { impact: 32, author: 30, citations: 27 }
    }
  },
  {
    id: 13,
    title: "Adam: A Method for Stochastic Optimization",
    date: "January 2015",
    authors: ["2", "3"],
    journal: "ICLR",
    findings: [
      "Proposes Adam, an adaptive learning rate optimization algorithm.",
      "Combines advantages of AdaGrad and RMSProp.",
      "Well-suited to problems with large data and/or parameters."
    ],
    tags: ["Optimization", "Seminal Work", "Machine Learning"],
    concepts: ["Adam", "SGD", "Momentum", "Gradient Descent"],
    abstract: "We present Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well-suited for problems that are large in terms of data and/or parameters.",
    citations: 75000,
    figures: [
        { title: "Adam Algorithm", url: "https://placehold.co/600x400/d1fae5/059669?text=Adam+Algorithm" }
    ],
    score: {
      total: 97,
      breakdown: { impact: 35, author: 28, citations: 34 }
    }
  }
];

const SUBJECTS = {
  nlp: {
    id: 'nlp',
    name: 'Natural Language Processing',
    nodes: [
      { id: 'root', label: 'Statistical NLP', year: 1990, x: 50, y: 10, type: 'paradigm', concepts: ['Frequency', 'N-Grams', 'Probability'], mutation: 'Applying statistics to language rules' },
      { id: 'rnn', label: 'Recurrent Neural Networks', year: 2010, x: 30, y: 30, type: 'paradigm', concepts: ['Sequential Processing', 'Hidden State', 'Backprop'], mutation: 'Neural memory via loops', deprecated: ['Hand-crafted features'] },
      { id: 'lstm', label: 'LSTMs', year: 2013, x: 30, y: 50, type: 'refinement', concepts: ['Gating Mechanisms', 'Long-term Memory', 'Cell State'], mutation: 'Solved Vanishing Gradient problem' },
      { id: 'seq2seq', label: 'Seq2Seq (Encoder-Decoder)', year: 2014, x: 50, y: 50, type: 'bridge', concepts: ['Encoding', 'Decoding', 'Fixed Vector'], mutation: 'Mapping sequence to sequence' },
      { id: 'attention', label: 'Attention Mechanisms', year: 2015, x: 70, y: 60, type: 'breakthrough', concepts: ['Context Vector', 'Alignment', 'Soft Weights'], mutation: 'Focusing on relevant parts of input' },
      { id: 'transformer', label: 'Transformers', year: 2017, x: 70, y: 80, type: 'paradigm', concepts: ['Self-Attention', 'Positional Encoding', 'Parallelism'], mutation: 'Discarding Recurrence entirely (Attention Is All You Need)', deprecated: ['Sequential Processing', 'Recurrence'] },
      { id: 'bert', label: 'BERT (Encoders)', year: 2018, x: 50, y: 95, type: 'refinement', concepts: ['Bidirectionality', 'Masked LM'], mutation: 'Deep conceptual understanding' },
      { id: 'gpt', label: 'GPT (Decoders)', year: 2018, x: 90, y: 95, type: 'refinement', concepts: ['Unidirectionality', 'Generative Pre-training'], mutation: 'Scale & Generation' }
    ],
    edges: [
      { from: 'root', to: 'rnn' }, { from: 'root', to: 'seq2seq' }, { from: 'rnn', to: 'lstm' },
      { from: 'lstm', to: 'seq2seq' }, { from: 'seq2seq', to: 'attention' },
      { from: 'attention', to: 'transformer' }, { from: 'transformer', to: 'bert' },
      { from: 'transformer', to: 'gpt' }
    ]
  },
  ml_optimization: {
    id: 'ml_optimization',
    name: 'Machine Learning Optimization',
    nodes: [
      { id: 'gd', label: 'Gradient Descent', year: 1952, x: 50, y: 10, type: 'paradigm', concepts: ['Gradients', 'Learning Rate'], mutation: 'Iteratively moving towards a local minimum.' },
      { id: 'sgd', label: 'Stochastic Gradient Descent', year: 1952, x: 50, y: 30, type: 'refinement', concepts: ['Mini-batches', 'Stochasticity'], mutation: 'Faster, noisier updates using single samples.' },
      { id: 'momentum', label: 'Momentum', year: 1964, x: 30, y: 50, type: 'refinement', concepts: ['Velocity', 'Dampening oscillations'], mutation: 'Accelerating gradients in the right direction.' },
      { id: 'adagrad', label: 'AdaGrad', year: 2011, x: 70, y: 50, type: 'breakthrough', concepts: ['Per-parameter learning rates'], mutation: 'Adaptive learning rates for different parameters.' },
      { id: 'rmsprop', label: 'RMSProp', year: 2012, x: 70, y: 70, type: 'refinement', concepts: ['Exponentially weighted average'], mutation: 'Resolving AdaGrad\'s diminishing learning rates.' },
      { id: 'adam', label: 'Adam', year: 2015, x: 50, y: 90, type: 'paradigm', concepts: ['Adaptive Moment Estimation', 'Bias Correction'], mutation: 'Combining Momentum and RMSProp.' }
    ],
    edges: [
      { from: 'gd', to: 'sgd' }, { from: 'sgd', to: 'momentum' }, { from: 'sgd', to: 'adagrad' },
      { from: 'adagrad', to: 'rmsprop' }, { from: 'momentum', to: 'adam' }, { from: 'rmsprop', to: 'adam' }
    ]
  },
  climate: {
    id: 'climate',
    name: 'Climate Change Models',
    nodes: [
      { id: 'egcm', label: 'Early GCMs', year: 1975, x: 50, y: 15, type: 'paradigm', concepts: ['Atmospheric Dynamics', 'Grid Cells'], mutation: 'First 3D simulations of atmospheric circulation.' },
      { id: 'aogcm', label: 'Coupled AOGCMs', year: 1990, x: 50, y: 40, type: 'refinement', concepts: ['Ocean-Atmosphere Interaction'], mutation: 'Coupling atmospheric and ocean models.' },
      { id: 'esm', label: 'Earth System Models', year: 2001, x: 50, y: 65, type: 'breakthrough', concepts: ['Biogeochemical Cycles', 'Carbon Cycle'], mutation: 'Incorporating biology and chemistry.' },
      { id: 'iam', label: 'Integrated Assessment Models', year: 2010, x: 30, y: 85, type: 'bridge', concepts: ['Socioeconomics', 'Policy Scenarios'], mutation: 'Linking ESMs with economic models.' },
      { id: 'hrm', label: 'High-Resolution Models', year: 2020, x: 70, y: 85, type: 'refinement', concepts: ['Kilometer-scale', 'Extreme Weather'], mutation: 'Resolving smaller-scale climate phenomena.' }
    ],
    edges: [
      { from: 'egcm', to: 'aogcm' }, { from: 'aogcm', to: 'esm' }, { from: 'esm', to: 'iam' }, { from: 'esm', to: 'hrm' }
    ]
  },
  crispr: {
    id: 'crispr',
    name: 'CRISPR Technology',
    nodes: [
      { id: 'discovery', label: 'Discovery of CRISPR repeats', year: 1987, x: 50, y: 10, type: 'paradigm', concepts: ['Repeated sequences'], mutation: 'Identification of clustered repeats in E. coli.' },
      { id: 'cas_genes', label: 'Identification of Cas genes', year: 2002, x: 50, y: 30, type: 'refinement', concepts: ['CRISPR-associated genes'], mutation: 'Linking genes to the CRISPR loci.' },
      { id: 'function', label: 'Function as adaptive immunity', year: 2007, x: 50, y: 50, type: 'breakthrough', concepts: ['Phage defense', 'Spacers'], mutation: 'Demonstrating CRISPR provides immunity against viruses.' },
      { id: 'cas9_cleavage', label: 'Cas9 DNA Cleavage', year: 2012, x: 35, y: 70, type: 'breakthrough', concepts: ['tracrRNA', 'gRNA', 'DSB'], mutation: 'Repurposing Cas9 for programmable DNA cutting.' },
      { id: 'gene_editing', label: 'Genome Editing in Eukaryotes', year: 2013, x: 35, y: 90, type: 'paradigm', concepts: ['Human cells', 'Multiplexing'], mutation: 'Applying CRISPR-Cas9 for gene editing in human cells.' },
      { id: 'prime', label: 'Prime Editing', year: 2019, x: 65, y: 90, type: 'refinement', concepts: ['Reverse transcriptase', 'pegRNA'], mutation: 'Search-and-replace gene editing without double-strand breaks.' }
    ],
    edges: [
      { from: 'discovery', to: 'cas_genes' }, { from: 'cas_genes', to: 'function' },
      { from: 'function', to: 'cas9_cleavage' }, { from: 'cas9_cleavage', to: 'gene_editing' },
      { from: 'gene_editing', to: 'prime' }
    ]
  },
  quantum: {
    id: 'quantum',
    name: 'Quantum Computing Gates',
    nodes: [
      { id: 'single_qubit', label: 'Single-Qubit Gates', year: 1995, x: 50, y: 20, type: 'paradigm', concepts: ['Hadamard', 'Pauli-X, Y, Z'], mutation: 'Basic quantum operations on a single qubit.' },
      { id: 'cnot', label: 'CNOT Gate', year: 1995, x: 50, y: 45, type: 'breakthrough', concepts: ['Entanglement', 'Controlled operation'], mutation: 'A two-qubit gate essential for universal computation.' },
      { id: 'universal_sets', label: 'Universal Gate Sets', year: 1995, x: 50, y: 65, type: 'refinement', concepts: ['Universality', 'Approximation'], mutation: 'Proving a small set of gates can approximate any quantum operation.' },
      { id: 'qec', label: 'Quantum Error Correction', year: 1995, x: 25, y: 85, type: 'bridge', concepts: ['Shor code', 'Stabilizer codes'], mutation: 'Protecting quantum information from noise.' },
      { id: 'surface_code', label: 'Surface Codes', year: 1997, x: 75, y: 85, type: 'refinement', concepts: ['2D lattice', 'High threshold'], mutation: 'A practical approach to fault-tolerant quantum computation.' }
    ],
    edges: [
      { from: 'single_qubit', to: 'cnot' }, { from: 'cnot', to: 'universal_sets' },
      { from: 'universal_sets', to: 'qec' }, { from: 'universal_sets', to: 'surface_code' }
    ]
  }
};

const TRENDING_SEARCHES = ["Machine Learning Optimization", "Climate Change Models", "CRISPR efficacy", "Quantum Computing Gates"];
const RECENT_SEARCHES = ["Reinforcement Learning", "Neural Radiance Fields", "Graph Neural Networks"];

const suggestedSearchResults = {
  "Machine Learning Optimization": [13, 2, 5, 6],
  "Climate Change Models": [7, 10],
  "CRISPR efficacy": [8, 11],
  "Quantum Computing Gates": [9, 12]
};

// --- SHARED COMPONENTS ---

const Tooltip = ({ content, children }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const handleMouseEnter = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setPosition({ x: rect.left, y: rect.bottom + 5 });
    setIsVisible(true);
  };

  return (
    <div className="relative inline-block" onMouseEnter={handleMouseEnter} onMouseLeave={() => setIsVisible(false)}>
      {children}
      {isVisible && (
        <div 
          className="fixed z-50 px-2 py-1 text-xs font-medium text-white bg-gray-800 rounded shadow-lg whitespace-nowrap"
          style={{ left: position.x, top: position.y }}
        >
          {content}
        </div>
      )}
    </div>
  );
};

const AuthorTooltip = ({ authorId, children, onAuthorClick }) => {
  const [isVisible, setIsVisible] = useState(false);
  const author = MOCK_AUTHORS[authorId];
  
  if (!author) return <span>{children}</span>;

  return (
    <div 
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      <span 
        className="cursor-pointer hover:text-blue-600 hover:underline decoration-blue-600 underline-offset-2 transition-colors duration-150"
        onClick={(e) => { e.stopPropagation(); onAuthorClick(author); }}
      >
        {children}
      </span>
      
      {isVisible && (
        <div className="absolute z-50 left-0 top-full mt-2 w-64 bg-white border border-gray-200 rounded-lg shadow-xl p-4 flex flex-col gap-3 animate-in fade-in zoom-in duration-200">
          <div className="flex items-center gap-3">
            <img src={author.img} alt={author.name} className="w-12 h-12 rounded-full bg-gray-100" />
            <div>
              <div 
                className="font-bold text-gray-900 cursor-pointer hover:text-blue-600"
                onClick={(e) => { e.stopPropagation(); onAuthorClick(author); }}
              >
                {author.name}
              </div>
              <div className="text-xs text-gray-500">{author.affiliation}</div>
            </div>
          </div>
          <div className="flex justify-between items-center pt-2 border-t border-gray-100">
            <div className="text-xs text-gray-600">
              <span className="font-semibold text-gray-900">C-Score:</span> {author.cScore}
            </div>
            <button 
              className="text-xs text-blue-600 font-medium hover:underline"
              onClick={(e) => { e.stopPropagation(); onAuthorClick(author); }}
            >
              View Profile
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

const Tag = ({ text, tooltip, type = "default" }) => {
  const getStyle = () => {
    switch (type) {
      case "trending": return "border-orange-200 bg-orange-50 text-orange-700";
      case "influential": return "border-blue-200 bg-blue-50 text-blue-700";
      default: return "border-gray-200 bg-gray-50 text-gray-600";
    }
  };

  const content = (
    <span className={`px-2 py-0.5 text-[11px] font-medium border rounded-md whitespace-nowrap ${getStyle()}`}>
      {text}
    </span>
  );

  return tooltip ? <Tooltip content={tooltip}>{content}</Tooltip> : content;
};

const ScoreBar = ({ label, value, max, colorClass }) => (
  <div className="mb-2">
    <div className="flex justify-between text-xs mb-1">
      <span className="text-gray-600 font-medium">{label}</span>
      <span className="font-bold text-gray-900">{value}</span>
    </div>
    <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
      <div 
        className={`h-full rounded-full ${colorClass}`} 
        style={{ width: `${(value / max) * 100}%` }}
      ></div>
    </div>
  </div>
);

const ConceptBadge = ({ label, type = 'neutral' }) => {
  const styles = {
    neutral: "bg-slate-100 text-slate-700 border-slate-200",
    new: "bg-emerald-50 text-emerald-700 border-emerald-100",
    deprecated: "bg-rose-50 text-rose-700 border-rose-100 line-through decoration-rose-400/50 opacity-70"
  };

  return (
    <span className={`text-xs px-2 py-1 rounded-md border font-medium ${styles[type]}`}>
      {label}
    </span>
  );
};

const Modal = ({ isOpen, onClose, title, children }) => {
   if (!isOpen) return null;
   return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm p-4">
         <div className="bg-white rounded-xl shadow-xl max-w-md w-full overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center">
               <h3 className="font-bold text-gray-900">{title}</h3>
               <button onClick={onClose} className="text-gray-400 hover:text-gray-600"><X size={20}/></button>
            </div>
            <div className="p-6">
               {children}
            </div>
         </div>
      </div>
   );
};

const EvolutionaryGraph = ({ nodes, edges, activeNodeId, onNodeSelect }) => {
  return (
    <div className="relative w-full h-full overflow-visible select-none">
      <svg className="w-full h-full" viewBox="0 0 120 120" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="8" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="#cbd5e1" />
          </marker>
        </defs>
        <g opacity="0.05">
          <rect x="0" y="0" width="120" height="30" fill="#64748b" />
          <rect x="0" y="30" width="120" height="40" fill="#3b82f6" />
          <rect x="0" y="70" width="120" height="50" fill="#8b5cf6" />
        </g>
        {edges.map((edge, i) => {
          const start = nodes.find(n => n.id === edge.from);
          const end = nodes.find(n => n.id === edge.to);
          if (!start || !end) return null;
          return (
            <path
              key={i}
              d={`M ${start.x} ${start.y} C ${start.x} ${(start.y + end.y)/2}, ${end.x} ${(start.y + end.y)/2}, ${end.x} ${end.y}`}
              fill="none"
              stroke="#cbd5e1"
              strokeWidth="0.5"
              markerEnd="url(#arrowhead)"
              className="transition-all duration-500"
            />
          );
        })}
        {nodes.map((node) => {
          const isActive = node.id === activeNodeId;
          const nodeColor = node.type === 'paradigm' ? '#6366f1' : node.type === 'breakthrough' ? '#ec4899' : '#64748b';
          return (
            <g 
              key={node.id} 
              onClick={() => onNodeSelect(node)}
              className="cursor-pointer transition-all duration-300 hover:opacity-80"
            >
              {isActive && (
                <circle cx={node.x} cy={node.y} r="6" fill={nodeColor} opacity="0.2">
                  <animate attributeName="r" values="6;10;6" dur="2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.2;0;0.2" dur="2s" repeatCount="indefinite" />
                </circle>
              )}
              <circle 
                cx={node.x} 
                cy={node.y} 
                r={isActive ? 4 : 2.5} 
                fill={isActive ? nodeColor : "white"}
                stroke={nodeColor}
                strokeWidth={isActive ? 1 : 0.8}
                className="drop-shadow-sm"
              />
              <text x={node.x + 5} y={node.y} alignmentBaseline="middle" fontSize="2.5" fill="#94a3b8" fontFamily="monospace">
                {node.year}
              </text>
              <text x={node.x + 5} y={node.y + 3.5} fontSize="3" fontWeight={isActive ? "bold" : "normal"} fill={isActive ? "#1e293b" : "#64748b"}>
                {node.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

const DnaAnalysisPanel = ({ node }) => {
  if (!node) return <div className="text-slate-400 p-8 text-center flex flex-col items-center h-full justify-center gap-2">
    <Microscope size={32} className="text-slate-200" />
    <span>Select a node to analyze genetic makeup</span>
  </div>;

  return (
    <div key={node.id} className="h-full flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-6">
        <div className="flex items-center gap-2 text-xs font-bold text-indigo-500 uppercase tracking-wider mb-2">
          <Microscope size={14} /> Concept Analysis
        </div>
        <h2 className="text-2xl font-bold text-slate-800">{node.label}</h2>
        <div className="flex items-center gap-4 mt-2 text-sm text-slate-500">
          <span className="flex items-center gap-1 bg-slate-100 px-2 py-0.5 rounded"><Clock size={12}/> {node.year}</span>
          <span className="capitalize px-2 py-0.5 bg-indigo-50 text-indigo-700 rounded text-xs font-bold">{node.type}</span>
        </div>
      </div>
      <div className="bg-gradient-to-br from-indigo-50 to-purple-50 p-5 rounded-xl border border-indigo-100 mb-6 shadow-sm">
        <h3 className="text-sm font-bold text-indigo-900 mb-2 flex items-center gap-2">
          <Dna size={16} />
          The Evolutionary Leap
        </h3>
        <p className="text-indigo-800 leading-relaxed text-sm">
          {node.mutation}
        </p>
      </div>
      <div className="space-y-6 flex-1 overflow-y-auto pr-2 custom-scrollbar">
        <div>
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <CheckCircle2 size={14} className="text-emerald-500" /> Introduced Concepts
          </h4>
          <div className="flex flex-wrap gap-2">
            {node.concepts.map(c => <ConceptBadge key={c} label={c} type="new" />)}
          </div>
        </div>
        {node.deprecated && (
          <div>
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
              <XCircle size={14} className="text-rose-500" /> Deprecated / Replaced
            </h4>
            <div className="flex flex-wrap gap-2">
              {node.deprecated.map(c => <ConceptBadge key={c} label={c} type="deprecated" />)}
            </div>
            <p className="text-xs text-slate-400 mt-2 italic">
              These concepts were dominant in previous generations but abandoned by this paradigm.
            </p>
          </div>
        )}
        <div className="pt-6 border-t border-slate-100">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <FileText size={14} /> Seminal Papers (The Fossil Record)
          </h4>
          <div className="space-y-3">
            {[1, 2].map(i => (
              <div key={i} className="group flex items-start gap-3 p-3 rounded-lg hover:bg-slate-50 cursor-pointer border border-transparent hover:border-slate-200 transition-all">
                <div className="mt-1 h-8 w-8 bg-slate-200 rounded text-xs font-bold flex items-center justify-center text-slate-500 group-hover:bg-indigo-100 group-hover:text-indigo-600 transition-colors">
                  PDF
                </div>
                <div>
                  <div className="text-sm font-bold text-slate-700 group-hover:text-indigo-700 transition-colors">
                    {node.label === 'CRISPR-Cas9' ? 'A Programmable Dual-RNA-Guided...' : `Seminal Paper for ${node.label}`}
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">
                    {node.label === 'CRISPR-Cas9' ? 'Jinek, Doudna, Charpentier et al.' : 'Key Authors et al.'} • Cited by 15,000+
                  </div>
                </div>
                <ArrowRight size={14} className="ml-auto text-slate-300 group-hover:text-indigo-400" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// --- SUB-COMPONENTS (MOVED OUTSIDE) ---

const PaperCard = ({ paper, expandedPaperId, setExpandedPaperId, searchContextPapers, toggleSearchContext, toggleAiContext, isAiSelected, setShowCitationModal, handleAuthorClick, handleConceptClick, handleCitationClick }) => {
  const isExpanded = expandedPaperId === paper.id;
  const isSearchSelected = searchContextPapers.some(p => p.id === paper.id);
  const [showAllAuthors, setShowAllAuthors] = useState(false);

  const getScoreColor = (score) => {
    if (score >= 90) return 'text-emerald-600';
    if (score >= 75) return 'text-indigo-600';
    if (score >= 60) return 'text-amber-500';
    return 'text-gray-500';
  };

  const displayedAuthors = showAllAuthors ? paper.authors : paper.authors.slice(0, 3);

  return (
    <div 
      onClick={() => setExpandedPaperId(isExpanded ? null : paper.id)}
      className={`bg-white rounded-lg border transition-all duration-200 cursor-pointer ${isExpanded ? 'ring-2 ring-indigo-500 border-transparent shadow-lg' : 'border-gray-200 shadow-sm hover:border-indigo-300'}`}
    >
      {/* Main Card Content */}
      <div className="p-4">
        <div className="flex justify-between items-start gap-4">
          <div className="flex-1">
            <h3 className="text-lg font-bold text-gray-900 leading-snug mb-1 hover:text-indigo-600">
              {paper.title}
            </h3>
            
            {/* Meta Row: Date | Authors */}
            <div className="flex items-center flex-wrap gap-x-3 gap-y-1 mb-3 text-sm">
              <span className="text-gray-500 font-medium">{paper.date}</span>
              <span className="text-gray-300">|</span>
              <div className="flex items-center flex-wrap gap-1 text-gray-700">
                {displayedAuthors.map((authorId, idx) => (
                  <React.Fragment key={authorId}>
                    <AuthorTooltip authorId={authorId} onAuthorClick={handleAuthorClick}>
                      {MOCK_AUTHORS[authorId]?.name}
                    </AuthorTooltip>
                    {idx < displayedAuthors.length - 1 && <span>,</span>}
                  </React.Fragment>
                ))}
                {!showAllAuthors && paper.authors.length > 3 && (
                  <span 
                     className="text-gray-400 hover:text-gray-600 cursor-pointer ml-1 text-xs bg-gray-100 px-1.5 py-0.5 rounded-full"
                     title="Click to see full author list"
                     onClick={(e) => { e.stopPropagation(); setShowAllAuthors(true); }}
                  >
                    +{paper.authors.length - 3} more
                  </span>
                )}
              </div>
            </div>

            {/* Tags Row */}
            <div className="flex items-center flex-wrap gap-2 mb-3">
              <span className="flex items-center gap-1.5 font-medium text-gray-900 text-[11px]">
                <BookOpen size={12} className="text-gray-400" />
                {paper.journal}
              </span>
              <span className="text-gray-300 mx-1">|</span>
              {paper.tags.map(tag => (
                 <Tag key={tag} text={tag} type={tag === 'Trending' ? 'trending' : tag === 'Highly influential' ? 'influential' : 'default'} />
              ))}
            </div>

            {/* Key Findings */}
            <div className="space-y-1 mb-4">
              {paper.findings.map((finding, idx) => (
                <div key={idx} className="flex items-start gap-2 text-sm text-gray-600">
                  <div className="mt-1.5 w-1 h-1 rounded-full bg-indigo-400 shrink-0" />
                  <span className="leading-tight">{finding}</span>
                </div>
              ))}
            </div>

            {/* Concepts Row */}
            <div className="flex items-center gap-2 mb-3">
               <Layers size={14} className="text-indigo-500 shrink-0" />
               <div className="flex flex-wrap gap-2">
                  {paper.concepts.map(concept => (
                     <span 
                       key={concept} 
                       onClick={(e) => handleConceptClick(concept, e)}
                       className="px-2 py-0.5 bg-slate-50 text-slate-700 text-[10px] font-medium border border-slate-200 rounded hover:bg-indigo-50 hover:text-indigo-700 hover:border-indigo-200 transition-colors"
                     >
                        {concept}
                     </span>
                  ))}
               </div>
            </div>
          </div>
          
          {/* Actions & Score Column (Right) */}
          <div className="flex flex-col gap-3 shrink-0 items-end">
             
             {/* Composite Score */}
             <div className="flex flex-col items-center justify-center p-2 bg-gray-50 rounded-lg border border-gray-100 min-w-[70px]">
               <div className={`text-2xl font-black leading-none ${getScoreColor(paper.score.total)}`}>
                 {paper.score.total}
               </div>
               <div className="text-[9px] uppercase font-bold text-gray-400 tracking-wider mt-1">Score</div>
             </div>

             {/* Secondary Actions */}
             <div className="flex gap-1 justify-end mt-auto">
               <button 
                onClick={(e) => toggleSearchContext(paper, e)}
                className={`p-1.5 rounded-md border transition-colors ${isSearchSelected ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-gray-500 border-gray-200 hover:bg-gray-50'}`}
                title={isSearchSelected ? "Remove from Search Context" : "Add to Search Context"}
               >
                 {isSearchSelected ? <Plus size={14} className="rotate-45" /> : <Plus size={14} />}
               </button>
               <button 
                 onClick={(e) => { e.stopPropagation(); setShowCitationModal(true); }}
                 className="p-1.5 bg-white text-gray-500 border border-gray-200 rounded-md hover:bg-gray-50" 
                 title="Cite"
                >
                 <Quote size={14} />
               </button>
               <button 
                onClick={(e) => toggleAiContext(paper, e)}
                className={`p-1.5 rounded-md border transition-colors ${isAiSelected ? 'bg-purple-600 text-white border-purple-600' : 'bg-white text-gray-500 border-gray-200 hover:bg-gray-50'}`}
                title="Ask AI"
               >
                 <Bot size={14} />
               </button>
             </div>
          </div>
        </div>
      </div>

      {/* Expanded View (Inline) */}
      {isExpanded && (
        <div className="border-t border-gray-100 bg-gray-50/50 p-6 animate-in slide-in-from-top-2 duration-200 cursor-default" onClick={(e) => e.stopPropagation()}>
          <div className="grid md:grid-cols-3 gap-8">
            {/* Left Col: Abstract & Figures */}
            <div className="md:col-span-2 space-y-6">
              <div>
                <h4 className="text-sm font-bold text-gray-900 mb-2 uppercase tracking-wide">Abstract</h4>
                <p className="text-gray-700 leading-relaxed text-sm">{paper.abstract}</p>
              </div>
              
              <div>
                <h4 className="text-sm font-bold text-gray-900 mb-3 uppercase tracking-wide">Extracted Figures</h4>
                <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-thin scrollbar-thumb-gray-300">
                  {paper.figures.map((fig, idx) => (
                    <div key={idx} className="shrink-0 group relative rounded-lg overflow-hidden border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
                      <img src={fig.url} alt={fig.title} className="w-64 h-auto object-cover bg-white" />
                      <div className="absolute bottom-0 inset-x-0 bg-white/90 backdrop-blur-sm p-2 text-xs font-bold text-gray-800 border-t border-gray-100">
                        {fig.title}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                 <h4 className="text-sm font-bold text-gray-900 mb-2 uppercase tracking-wide">Key Citations</h4>
                 <ul className="text-sm space-y-1">
                    <li className="text-indigo-600 hover:underline cursor-pointer" onClick={() => handleCitationClick("Attention is All You Need")}>Vaswani et al. (2017) - Attention is All You Need</li>
                    <li className="text-indigo-600 hover:underline cursor-pointer" onClick={() => handleCitationClick("On the Opportunities and Risks")}>Bommasani et al. (2021) - On the Opportunities and Risks of Foundation Models</li>
                 </ul>
              </div>
            </div>

            {/* Right Col: Score Breakdown & Related */}
            <div className="border-l border-gray-200 pl-6">
              
              <div className="mb-8">
                 <h4 className="text-sm font-bold text-gray-900 mb-4 uppercase tracking-wide flex items-center gap-2">
                   <Activity size={16} /> Score Breakdown
                 </h4>
                 <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                    <ScoreBar label="Relevance" value={29} max={35} colorClass="bg-amber-500" />
                    <ScoreBar label="Journal Impact Factor" value={paper.score.breakdown.impact} max={35} colorClass="bg-purple-500" />
                    <ScoreBar label="1st Author C-Score" value={paper.score.breakdown.author} max={35} colorClass="bg-blue-500" />
                    <ScoreBar label="Recent Citations" value={paper.score.breakdown.citations} max={35} colorClass="bg-emerald-500" />
                 </div>
              </div>

              <h4 className="text-sm font-bold text-gray-900 mb-3 uppercase tracking-wide">Related Papers</h4>
              <div className="space-y-3">
                {MOCK_PAPERS.filter(p => p.id !== paper.id).slice(0, 3).map(related => (
                  <div key={related.id} className="group cursor-pointer">
                    <div className="font-semibold text-gray-800 text-sm group-hover:text-indigo-600 leading-tight mb-1">
                      {related.title}
                    </div>
                    <div className="text-xs text-gray-500 mb-1">{related.date}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div className="mt-6 pt-4 border-t border-gray-200 flex justify-end gap-3">
             <button className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gray-900 rounded-lg hover:bg-gray-800">
                <ExternalLink size={16} /> View at Source
             </button>
          </div>
        </div>
      )}
    </div>
  );
};

const SearchBar = ({ inputValue, setInputValue, handleSearchSubmit, setShowAddContextModal, searchContextPapers, size = 'large' }) => {
  const [isFocused, setIsFocused] = useState(false);
  // Mock handler to simulate selecting from history
  const handleHistorySelect = (term) => {
    setInputValue(term);
    // In a real app, you might trigger a search immediately
  };

  const isSmall = size === 'small';

  return (
    <div className="w-full relative">
      <form onSubmit={handleSearchSubmit} className={`relative group transition-all ${isSmall ? '' : 'shadow-lg rounded-2xl focus-within:ring-2 ring-indigo-100'}`}>
        <input
          type="text"
          className={`w-full border outline-none focus:border-indigo-500 placeholder:text-gray-400
            ${isSmall 
              ? 'h-10 pl-4 pr-24 text-base rounded-lg border-gray-200 bg-gray-50 focus:bg-white' 
              : 'h-14 pl-5 pr-32 text-lg rounded-2xl border-2 border-gray-100'
            }`}
          placeholder={isSmall ? "Search..." : "Search for papers, topics, or authors..."}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setTimeout(() => setIsFocused(false), 200)}
        />
        
        <button 
          type="button"
          onClick={() => setShowAddContextModal(true)}
          className={`absolute aspect-square text-gray-400 hover:bg-indigo-50 hover:text-indigo-600 flex items-center justify-center transition-colors
            ${isSmall 
              ? 'right-10 top-1.5 bottom-1.5 rounded-lg' 
              : 'right-16 top-2 bottom-2 rounded-xl'
            }`}
          title="Add paper to search context"
        >
          <div className="relative">
            <FileText size={isSmall ? 18 : 20} />
            {searchContextPapers.length > 0 ? (
               <div className="absolute -top-2 -right-2 bg-indigo-600 text-white text-[10px] font-bold rounded-full w-4 h-4 flex items-center justify-center border-2 border-white">
                 {searchContextPapers.length}
               </div>
            ) : (
               <div className="absolute -bottom-1 -right-1 bg-white rounded-full"><Plus size={10} className="text-indigo-600 stroke-[3]"/></div>
            )}
          </div>
        </button>

        <button 
          type="submit"
          className={`absolute aspect-square text-white flex items-center justify-center transition-colors
            ${isSmall 
              ? 'right-1.5 top-1.5 bottom-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700' 
              : 'right-2 top-2 bottom-2 rounded-xl bg-indigo-600 hover:bg-indigo-700'
            }`}
        >
          <Search size={isSmall ? 18 : 20} />
        </button>
      </form>

      {isFocused && !inputValue && !isSmall && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden z-50">
          <div className="px-4 py-2 text-xs font-bold text-gray-400 uppercase tracking-wider bg-gray-50">Recent Searches</div>
          {RECENT_SEARCHES.map(term => (
             <button 
               key={term} 
               onMouseDown={() => handleHistorySelect(term)}
               className="w-full text-left px-4 py-3 hover:bg-indigo-50 flex items-center gap-3 text-sm text-gray-700 transition-colors"
             >
               <History size={14} className="text-gray-400" /> {term}
             </button>
          ))}
        </div>
      )}
    </div>
  );
};

const Header = ({ view, setView, inputValue, setInputValue, handleSearchSubmit, aiContextPapers, searchContextPapers, setShowAddContextModal }) => (
  <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-sm border-b border-gray-200 h-14 flex items-center px-4 justify-between shadow-sm">
    <div className="flex items-center gap-6 flex-1">
      <div 
        onClick={() => { setView('home'); setInputValue(''); }}
        className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity shrink-0"
      >
        <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white">
          <Search size={18} strokeWidth={3} />
        </div>
        <span className="font-bold text-xl tracking-tight text-gray-900">Haystack</span>
      </div>

      {view !== 'home' && (
        <div className="w-full max-w-lg hidden md:block">
           <SearchBar 
             inputValue={inputValue} 
             setInputValue={setInputValue} 
             handleSearchSubmit={handleSearchSubmit}
             setShowAddContextModal={setShowAddContextModal}
             searchContextPapers={searchContextPapers}
             size="small"
           />
        </div>
      )}
    </div>

    <div className="flex items-center gap-3 shrink-0">
      <button 
        onClick={() => setView('ai')}
        className={`flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${view === 'ai' ? 'bg-indigo-600 text-white shadow-sm' : 'text-gray-600 hover:bg-gray-100'}`}
      >
        <Bot size={18} />
        <span className="hidden sm:inline">AI Assistant</span>
        {aiContextPapers.length > 0 && (
           <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${view === 'ai' ? 'bg-white text-indigo-600' : 'bg-indigo-600 text-white'}`}>{aiContextPapers.length}</span>
        )}
      </button>
      <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center text-gray-500 cursor-pointer hover:bg-gray-300">
        <User size={18} />
      </div>
    </div>
  </header>
);

const HomeView = ({ inputValue, setInputValue, handleSearchSubmit, setView, setShowAddContextModal, handleTrendingClick, searchContextPapers, handleSuggestedSearch }) => (
  <div className="max-w-4xl mx-auto px-4 py-12 flex flex-col items-center">
    <h1 className="text-4xl font-extrabold text-gray-900 mb-8 tracking-tight">Find clarity in chaos.</h1>
    
    {/* Main Search */}
    <div className="w-full max-w-2xl relative mb-12">
      <SearchBar 
        inputValue={inputValue} 
        setInputValue={setInputValue} 
        handleSearchSubmit={handleSearchSubmit}
        setShowAddContextModal={setShowAddContextModal}
        searchContextPapers={searchContextPapers}
      />
      
      {/* Suggested / History */}
      <div className="flex flex-wrap gap-2 mt-4 justify-center">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider py-1">Suggested:</span>
        {TRENDING_SEARCHES.map((term, i) => (
          <button 
            key={i} 
            onClick={() => handleSuggestedSearch(term)}
            className="px-3 py-1 bg-white border border-gray-200 hover:border-indigo-300 text-sm text-gray-600 rounded-full transition-colors flex items-center gap-1"
          >
            <TrendingUp size={12} className="text-indigo-500" />
            {term}
          </button>
        ))}
      </div>
    </div>

    {/* Trending Feed (Expanded to Full Width) */}
    <div className="w-full mt-12">
      <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2 mb-6">
        <TrendingUp size={20} className="text-orange-500" /> Trending in your network
      </h2>
      <div className="grid md:grid-cols-2 gap-6">
         {MOCK_PAPERS.map(paper => (
           <div key={paper.id} className="bg-white p-5 rounded-xl border border-gray-200 shadow-sm hover:shadow-md hover:border-indigo-300 transition-all cursor-pointer group" onClick={() => handleTrendingClick(paper)}>
              <div className="flex justify-between items-start mb-3">
                 <span className="font-bold text-gray-900 text-lg line-clamp-2 group-hover:text-indigo-600 transition-colors pr-4">{paper.title}</span>
                 <Tag text={paper.journal} type="journal" />
              </div>
              <div className="text-sm text-gray-500 line-clamp-3 leading-relaxed mb-3">{paper.abstract}</div>
              
              <div className="flex items-center gap-2 text-xs text-gray-400 font-medium">
                 <div className="flex -space-x-2 overflow-hidden">
                    {paper.authors.slice(0, 3).map(aId => (
                       <img key={aId} src={MOCK_AUTHORS[aId].img} alt="" className="inline-block h-6 w-6 rounded-full ring-2 ring-white bg-gray-100" />
                    ))}
                 </div>
                 <span>{paper.date}</span>
              </div>
           </div>
         ))}
      </div>
    </div>
  </div>
);

const YearSlider = () => {
  const minYear = 2015;
  const currentYear = new Date().getFullYear();
  const [yearRange, setYearRange] = useState([minYear, currentYear]);

  const handleSliderChange = (index, value) => {
    const newRange = [...yearRange];
    newRange[index] = value;
    if (index === 0 && newRange[0] > newRange[1]) {
      newRange[1] = newRange[0];
    }
    if (index === 1 && newRange[1] < newRange[0]) {
      newRange[0] = newRange[1];
    }
    setYearRange(newRange);
  };

  const handleInputChange = (index, value) => {
    const parsedValue = value === '' ? (index === 0 ? minYear : currentYear) : parseInt(value, 10);
    if (!isNaN(parsedValue)) {
      handleSliderChange(index, parsedValue);
    }
  };

  const range = currentYear - minYear;
  const leftPercent = ((yearRange[0] - minYear) / range) * 100;
  const rightPercent = ((yearRange[1] - minYear) / range) * 100;

  return (
    <div>
      <h3 className="font-bold text-gray-900 text-sm mb-2">Publication Year</h3>
      
      <div className="relative h-5 flex items-center mb-2">
        <div className="relative w-full h-1 bg-gray-200 rounded-lg">
          <div className="absolute h-1 bg-indigo-600 rounded-lg" style={{ left: `${leftPercent}%`, width: `${rightPercent - leftPercent}%` }}></div>
          <input
            type="range"
            min={minYear}
            max={currentYear}
            value={yearRange[0]}
            onChange={(e) => handleSliderChange(0, parseInt(e.target.value, 10))}
            className="absolute w-full h-1 appearance-none bg-transparent pointer-events-auto [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-indigo-600 [&::-webkit-slider-thumb]:cursor-pointer"
          />
          <input
            type="range"
            min={minYear}
            max={currentYear}
            value={yearRange[1]}
            onChange={(e) => handleSliderChange(1, parseInt(e.target.value, 10))}
            className="absolute w-full h-1 appearance-none bg-transparent pointer-events-auto [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-indigo-600 [&::-webkit-slider-thumb]:cursor-pointer"
          />
        </div>
      </div>
      <div className="flex items-center justify-between gap-1">
        <input type="number" value={yearRange[0]} onChange={(e) => handleInputChange(0, e.target.value)} min={minYear} max={currentYear} className="w-full text-sm border-gray-200 rounded-md text-center focus:ring-indigo-500 focus:border-indigo-500 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none" />
        <span className="text-gray-400">-</span>
        <input type="number" value={yearRange[1]} onChange={(e) => handleInputChange(1, e.target.value)} min={minYear} max={currentYear} className="w-full text-sm border-gray-200 rounded-md text-center focus:ring-indigo-500 focus:border-indigo-500 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none" />
      </div>
    </div>
  );
};

const SearchResultsView = ({ searchQuery, setView, handleConceptClick, searchContextPapers, aiContextPapers, toggleSearchContext, toggleAiContext, expandedPaperId, setExpandedPaperId, setShowCitationModal, handleAuthorClick, handleCitationClick }) => {
  const [activeQuickFilters, setActiveQuickFilters] = useState([]);
  const [activeFilters, setActiveFilters] = useState({});

  const toggleQuickFilter = (filter) => {
    setActiveQuickFilters(prev => 
      prev.includes(filter) 
        ? prev.filter(f => f !== filter) 
        : [...prev, filter]
    );
  };

  const toggleFilter = (title, option) => {
    setActiveFilters(prev => {
      const currentOptions = prev[title] || [];
      const newOptions = currentOptions.includes(option)
        ? currentOptions.filter(o => o !== option)
        : [...currentOptions, option];
      return {
        ...prev,
        [title]: newOptions,
      };
    });
  };

  const results = useMemo(() => {
    if (!searchQuery) {
        return [];
    }

    if (suggestedSearchResults[searchQuery]) {
      const paperIds = suggestedSearchResults[searchQuery];
      return MOCK_PAPERS.filter(p => paperIds.includes(p.id));
    }
    
    const queryWords = searchQuery.toLowerCase().split(' ').filter(w => w);
    
    return MOCK_PAPERS.filter(paper => {
      const paperContent = [
        paper.title,
        paper.abstract,
        ...paper.concepts,
        ...paper.tags,
        paper.journal,
        ...paper.authors.map(id => MOCK_AUTHORS[id].name)
      ].join(' ').toLowerCase();

      return queryWords.every(word => paperContent.includes(word));
    });
  }, [searchQuery]);

  // Derived top concepts based on results
  const topConcepts = useMemo(() => {
    if (results.length === 0) return [];
    const allConcepts = results.flatMap(paper => paper.concepts);
    const conceptCounts = allConcepts.reduce((acc, concept) => {
      acc[concept] = (acc[concept] || 0) + 1;
      return acc;
    }, {});
    return Object.entries(conceptCounts)
      .sort(([,a],[,b]) => b-a)
      .slice(0, 4)
      .map(([concept]) => concept);
  }, [results]);

  const showingText = searchContextPapers.length > 0 
    ? `Showing results related to search context (${searchContextPapers.length} papers)`
    : `Showing ${results.length} results for "${searchQuery}"`;

  const FilterSection = ({ title, options, activeOptions, onToggle }) => (
    <div>
      <h3 className="font-bold text-gray-900 mb-3 text-sm flex items-center gap-2">{title}</h3>
      <div className="flex flex-wrap gap-2">
        {options.map(opt => {
          const isActive = activeOptions?.includes(opt);
          return (
            <button 
              key={opt} 
              onClick={() => onToggle(title, opt)}
              className={`px-2 py-1 text-[10px] font-medium border rounded-full transition-colors ${
                isActive ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white border-gray-200 text-gray-600 hover:bg-indigo-50 hover:border-indigo-200'
              }`}
            >{opt}</button>
          );
        })}
      </div>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 flex gap-6">
      {/* Filters Sidebar (Left - Condensed) */}
      <div className="w-56 hidden lg:block shrink-0">
        <div className="sticky top-20 space-y-6">
          <div>
            <h3 className="font-bold text-gray-900 mb-3 text-sm flex items-center gap-2"><Filter size={14} /> Filters</h3>
            <div className="space-y-2">
              {["Review Articles", "Open Access", "Recent (Last Year)"].map(filter => {
                const isActive = activeQuickFilters.includes(filter);
                return (
                  <button 
                    key={filter}
                    onClick={() => toggleQuickFilter(filter)}
                    className={`w-full text-left px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                      isActive 
                        ? 'bg-indigo-600 text-white border border-indigo-600' 
                        : 'bg-white border border-gray-200 text-gray-600 hover:bg-indigo-50 hover:border-indigo-200'
                    }`}
                  >{filter}</button>
                );
              })}
            </div>
          </div>
          
          <YearSlider />
          <FilterSection title="Study Type" options={["Experimental", "Theoretical", "Survey"]} activeOptions={activeFilters["Study Type"]} onToggle={toggleFilter} />
          <FilterSection title="Source" options={["arXiv", "NeurIPS", "Nature", "Science", "ICLR"]} activeOptions={activeFilters["Source"]} onToggle={toggleFilter} />
        </div>
      </div>

      {/* Results Feed */}
      <div className="flex-1 space-y-3 pb-20">
        
        {/* Top Concepts Bar - Subtler */}
        <div className="mb-4 flex items-center gap-3">
           <div className="flex items-center gap-2 text-slate-500 text-xs font-bold uppercase tracking-wide">
              Top Concepts:
           </div>
           <div className="flex gap-2">
              {topConcepts.map(c => (
                 <button 
                   key={c} 
                   onClick={() => handleConceptClick(c)}
                   className="px-3 py-1 bg-indigo-50/50 text-indigo-700 text-xs font-medium rounded-full hover:bg-indigo-100 transition-colors"
                 >
                    {c}
                 </button>
              ))}
           </div>
        </div>

        <div className="flex justify-between items-end mb-4">
           <div className="text-sm text-gray-500">{showingText}</div>
           <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Sort by:</span>
              <select className="text-xs border-none bg-transparent font-medium focus:ring-0 cursor-pointer">
                 <option>Relevance</option>
                 <option>Date</option>
                 <option>Citations</option>
                 <option>Composite Score</option>
              </select>
           </div>
        </div>

        {results.length > 0 ? results.map(paper => (
          <PaperCard 
            key={paper.id} 
            paper={paper} 
            expandedPaperId={expandedPaperId} 
            setExpandedPaperId={setExpandedPaperId}
            searchContextPapers={searchContextPapers}
            aiContextPapers={aiContextPapers}
            toggleSearchContext={toggleSearchContext}
            toggleAiContext={toggleAiContext}
            isAiSelected={aiContextPapers.some(p => p.id === paper.id)}
            setShowCitationModal={setShowCitationModal}
            handleAuthorClick={handleAuthorClick}
            handleConceptClick={handleConceptClick}
            handleCitationClick={handleCitationClick}
          />
        )) : (
          <div className="text-center py-20">
            <h3 className="text-lg font-bold text-gray-800">No results found</h3>
            <p className="text-gray-500 mt-2">Try adjusting your search query or filters.</p>
          </div>
        )}
      </div>
    </div>
  );
};

const ConceptEvolutionView = ({ setView, currentSubject, activeNode, setActiveNode }) => {
  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-3.5rem)] overflow-hidden">
      {/* Navigation / Header for Concepts */}
      <div className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-start z-20">
         <div className="flex items-center gap-4">
            <button onClick={() => setView('search')} className="flex items-center gap-2 text-slate-500 hover:text-indigo-700 transition-colors text-sm font-medium">
              <ArrowLeft size={16} /> Back to Search
            </button>
         </div>
      </div>

      {/* Main Split View */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Left Panel: The Evolutionary Stream */}
        <div className="w-7/12 bg-slate-50 relative border-r border-slate-200 flex flex-col">
          <div className="bg-white p-4 border-b border-slate-200">
            <h2 className="text-sm font-bold text-black uppercase tracking-wider">{SUBJECTS[currentSubject].name}</h2>
            <p className="text-xs text-black">
              1990 - Present
            </p>
          </div>
          
          {/* Legend */}
          <div className="absolute bottom-6 left-6 z-10 flex gap-4 bg-white/80 backdrop-blur p-2 rounded-lg border border-slate-200 shadow-sm text-[10px] font-medium text-slate-600">
             <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-indigo-500"></div>Paradigm Shift</div>
             <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-pink-500"></div>Breakthrough</div>
             <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-slate-500"></div>Refinement</div>
          </div>

          <div className="flex-1 p-4 overflow-hidden">
            <EvolutionaryGraph 
              nodes={SUBJECTS[currentSubject].nodes} 
              edges={SUBJECTS[currentSubject].edges} 
              activeNodeId={activeNode?.id} 
              onNodeSelect={setActiveNode} 
            />
          </div>
        </div>

        {/* Right Panel: Concept Inspector */}
        <div className="w-5/12 bg-white p-8 shadow-[-4px_0_24px_rgba(0,0,0,0.02)] z-20 overflow-hidden">
          <DnaAnalysisPanel node={activeNode} />
        </div>

      </div>
    </div>
  );
};

const AuthorProfileView = ({ selectedAuthor, setView, setExpandedPaperId }) => {
   if (!selectedAuthor) return null;
   
   return (
      <div className="max-w-4xl mx-auto px-4 py-12">
         <button onClick={() => setView('search')} className="flex items-center gap-2 text-gray-500 hover:text-indigo-600 mb-6 transition-colors">
            <ArrowLeft size={18} /> Back to Search
         </button>
         
         <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden shadow-sm">
            <div className="h-32 bg-gradient-to-r from-indigo-500 to-purple-600"></div>
            <div className="px-8 pb-8">
               <div className="relative flex justify-between items-end -mt-12 mb-6">
                  <img src={selectedAuthor.img} alt={selectedAuthor.name} className="w-24 h-24 rounded-full border-4 border-white bg-white shadow-md" />
                  <div className="flex gap-2">
                     <button className="px-4 py-2 bg-indigo-600 text-white font-bold rounded-lg text-sm hover:bg-indigo-700 transition-colors">Follow</button>
                     <button className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50"><Share2 size={18} className="text-gray-600"/></button>
                  </div>
               </div>
               
               <div>
                  <h1 className="text-3xl font-bold text-gray-900 mb-1">{selectedAuthor.name}</h1>
                  <p className="text-lg text-gray-600 mb-4">{selectedAuthor.affiliation}</p>
                  
                  <div className="flex gap-6 mb-8 border-b border-gray-100 pb-6">
                     <div>
                        <div className="text-2xl font-bold text-gray-900">{selectedAuthor.cScore}</div>
                        <div className="text-xs text-gray-500 font-medium uppercase tracking-wide">C-Score</div>
                     </div>
                     <div>
                        <div className="text-2xl font-bold text-gray-900">12.4k</div>
                        <div className="text-xs text-gray-500 font-medium uppercase tracking-wide">Citations</div>
                     </div>
                     <div>
                        <div className="text-2xl font-bold text-gray-900">45</div>
                        <div className="text-xs text-gray-500 font-medium uppercase tracking-wide">Publications</div>
                     </div>
                  </div>

                  <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Publications</h2>
                  <div className="space-y-4">
                     {MOCK_PAPERS.filter(p => p.authors.includes(selectedAuthor.id)).map(paper => (
                        <div key={paper.id} className="p-4 border border-gray-200 rounded-xl hover:border-indigo-300 transition-colors cursor-pointer" onClick={() => { setView('search'); setExpandedPaperId(paper.id); }}>
                           <div className="font-bold text-gray-900 mb-1">{paper.title}</div>
                           <div className="text-sm text-gray-500">{paper.date} • {paper.journal}</div>
                        </div>
                     ))}
                  </div>
               </div>
            </div>
         </div>
      </div>
   );
};

const AiAssistantPage = ({ setView, aiContextPapers, toggleAiContext, messages, msgInput, setMsgInput, sendMessage }) => {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 h-[calc(100vh-3.5rem)]">
      <div className="mb-4">
         <button onClick={() => setView('search')} className="flex items-center gap-2 text-gray-500 hover:text-indigo-600 transition-colors text-sm font-medium">
            <ArrowLeft size={16} /> Back to Search Results
         </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[90%]">
        {/* Left Column: Context Papers */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm flex flex-col overflow-hidden h-full">
          <div className="p-4 border-b border-gray-100 bg-gray-50/50">
             <h3 className="font-bold text-gray-900 flex items-center gap-2">
               <Bookmark size={16} className="text-indigo-600" /> Context Library
             </h3>
             <p className="text-xs text-gray-500 mt-1">Papers added to your active session.</p>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
             {aiContextPapers.length > 0 ? aiContextPapers.map(p => (
               <div key={p.id} className="bg-white border border-gray-100 p-3 rounded-lg shadow-sm group hover:border-indigo-200 transition-colors">
                  <div className="font-semibold text-xs text-gray-800 line-clamp-2 leading-snug mb-2">{p.title}</div>
                  <div className="flex justify-between items-center">
                     <span className="text-[10px] bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">{p.date}</span>
                     <button onClick={(e) => toggleAiContext(p, e)} className="text-gray-300 hover:text-red-500"><X size={12}/></button>
                  </div>
               </div>
             )) : (
               <div className="text-center py-10 text-gray-400 text-xs px-6">
                 <div className="mb-2 mx-auto w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center"><Plus size={16}/></div>
                 Add papers from search results to discuss them here.
               </div>
             )}
          </div>
        </div>

        {/* Center Column: Chat Interface */}
        <div className="lg:col-span-2 bg-white rounded-2xl border border-gray-200 shadow-sm flex flex-col overflow-hidden h-full">
          <div className="p-4 border-b border-gray-100 bg-gray-50/50 flex justify-between items-center">
             <div className="flex items-center gap-2 font-bold text-gray-900">
                <Bot size={20} className="text-indigo-600" /> 
                Research Assistant
             </div>
             <div className="text-xs text-gray-500 bg-white border border-gray-200 px-2 py-1 rounded-full flex items-center gap-1">
                <Sparkles size={10} className="text-indigo-500" /> GPT-4o
             </div>
          </div>
          
          <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-gray-50/30">
             {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                   {msg.role === 'system' && <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center mr-3 mt-1 shrink-0 text-indigo-600"><Bot size={16}/></div>}
                   <div className={`max-w-[85%] rounded-2xl px-5 py-3 text-sm leading-relaxed shadow-sm ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-br-none' : 'bg-white border border-gray-200 text-gray-700 rounded-bl-none'}`}>
                      {msg.text}
                   </div>
                </div>
             ))}
          </div>

          <div className="p-4 border-t border-gray-200 bg-white">
             <form onSubmit={sendMessage} className="relative">
                <input 
                   type="text" 
                   value={msgInput}
                   onChange={(e) => setMsgInput(e.target.value)}
                   placeholder="Ask a question about the papers..." 
                   className="w-full pl-5 pr-12 py-3.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white text-sm shadow-inner transition-all"
                />
                <button type="submit" className="absolute right-2 top-2 p-1.5 text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors">
                   <ArrowLeft size={20} className="rotate-90 stroke-[3]" />
                </button>
             </form>
          </div>
        </div>

        {/* Right Column: Suggestions */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm flex flex-col overflow-hidden h-full">
          <div className="p-4 border-b border-gray-100 bg-gray-50/50">
             <h3 className="font-bold text-gray-900 flex items-center gap-2">
               <TrendingUp size={16} className="text-emerald-600" /> Suggested Readings
             </h3>
             <p className="text-xs text-gray-500 mt-1">Based on your current context.</p>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
             {MOCK_PAPERS.filter(p => !aiContextPapers.find(cp => cp.id === p.id)).slice(0, 4).map(p => (
               <div key={p.id} className="p-3 border border-dashed border-gray-300 rounded-lg hover:border-indigo-400 hover:bg-indigo-50/30 transition-all cursor-pointer group" onClick={() => toggleAiContext(p)}>
                  <div className="font-semibold text-xs text-gray-800 line-clamp-2 mb-2 group-hover:text-indigo-700">{p.title}</div>
                  <div className="flex justify-between items-center">
                     <span className="text-[10px] text-gray-400">{p.journal}</span>
                     <button className="text-indigo-600 opacity-0 group-hover:opacity-100 text-[10px] font-bold bg-white px-2 py-0.5 rounded shadow-sm">Add +</button>
                  </div>
               </div>
             ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// --- MAIN APPLICATION ---

const conceptToSubjectMap = {
  // NLP
  'Memory Streams': 'nlp', 'Agent Simulation': 'nlp', 'Emergent Behavior': 'nlp', 'Self-Attention': 'nlp', 'Transformers': 'nlp', 'Positional Encoding': 'nlp', 'Large Language Models': 'nlp', 'Emergent Abilities': 'nlp', 'Prompting': 'nlp', 'RLAIF': 'nlp', 'AI Safety': 'nlp', 'Constitutional AI': 'nlp', 'Chain-of-Thought': 'nlp', 'Reasoning': 'nlp', 'Few-Shot Learning': 'nlp', 'RLHF': 'nlp', 'Reward Modeling': 'nlp', 'Human Preferences': 'nlp',

  // Climate
  'Integrated Assessment Models': 'climate', 'Probabilistic Forecasting': 'climate', 'Climate Economics': 'climate', 'Ocean Currents': 'climate', 'Sea Ice Melt': 'climate', 'Climate Projections': 'climate',

  // CRISPR
  'Guide RNA': 'crispr', 'Gene Editing': 'crispr', 'High-throughput Screening': 'crispr', 'Off-target Effects': 'crispr', 'Guide RNA Engineering': 'crispr', 'Specificity': 'crispr',

  // Quantum
  'Stabilizer Codes': 'quantum', 'Quantum Error Correction': 'quantum', 'Fault Tolerance': 'quantum', 'Decoherence': 'quantum', 'Qubits': 'quantum', 'Superconducting Circuits': 'quantum',
  
  // ML Optimization
  'Gradient Descent': 'ml_optimization', 'SGD': 'ml_optimization', 'Momentum': 'ml_optimization', 'Adam': 'ml_optimization'
};

export default function HaystackApp() {
  const [view, setView] = useState('home'); // home, search, author, ai, concepts
  const [searchQuery, setSearchQuery] = useState('');
  const [inputValue, setInputValue] = useState('');
  
  // Separate Contexts
  const [searchContextPapers, setSearchContextPapers] = useState([]);
  const [aiContextPapers, setAiContextPapers] = useState([]);
  
  const [expandedPaperId, setExpandedPaperId] = useState(null);
  const [selectedAuthor, setSelectedAuthor] = useState(null);
  const [showCitationModal, setShowCitationModal] = useState(false);
  const [showAddContextModal, setShowAddContextModal] = useState(false);
  const [contextSearchQuery, setContextSearchQuery] = useState('');
  const [messages, setMessages] = useState([
    { role: 'system', text: 'Hello! I can help you synthesize findings from your selected papers. What would you like to know?' }
  ]);
  const [msgInput, setMsgInput] = useState('');

  // Evolution Tree State
  const [currentSubject, setCurrentSubject] = useState('nlp');
  const [activeNode, setActiveNode] = useState(null);

  // Initialize active node when view changes to concepts
  useEffect(() => {
    if (view === 'concepts') {
      const subjectNodes = SUBJECTS[currentSubject].nodes;
      // Check if activeNode is valid for the current subject, if not, reset it
      if (!activeNode || !subjectNodes.find(n => n.id === activeNode.id)) {
        const defaultNode = subjectNodes.find(n => n.type === 'breakthrough') || subjectNodes[subjectNodes.length - 1];
        setActiveNode(defaultNode);
      }
    }
  }, [view, currentSubject, activeNode]);


  // Handle Search Submit
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      setSearchQuery(inputValue);
      setView('search');
    }
  };

  const handleSuggestedSearch = (term) => {
    setInputValue(term);
    setSearchQuery(term);
    setView('search');
  };

  const handleAuthorClick = (author) => {
    setSelectedAuthor(author);
    setView('author');
  };

  const handleConceptClick = (concept, e) => {
    if(e) e.stopPropagation();
    const subject = conceptToSubjectMap[concept] || Object.keys(SUBJECTS).find(key => SUBJECTS[key].nodes.some(node => node.concepts.includes(concept))) || 'nlp';
    setCurrentSubject(subject); 
    
    // Find the node that introduces this concept
    const node = SUBJECTS[subject].nodes.find(n => n.concepts.includes(concept));
    setActiveNode(node || null);

    setView('concepts');
  };

  // Toggle Search Context
  const toggleSearchContext = (paper, e) => {
    e && e.stopPropagation();
    if (searchContextPapers.find(p => p.id === paper.id)) {
      setSearchContextPapers(searchContextPapers.filter(p => p.id !== paper.id));
    } else {
      setSearchContextPapers([...searchContextPapers, paper]);
    }
  };

  // Toggle AI Context
  const toggleAiContext = (paper, e) => {
    e && e.stopPropagation();
    if (aiContextPapers.find(p => p.id === paper.id)) {
      setAiContextPapers(aiContextPapers.filter(p => p.id !== paper.id));
    } else {
      setAiContextPapers([...aiContextPapers, paper]);
      setView('ai'); 
    }
  };

  const handleTrendingClick = (paper) => {
    setSearchQuery(paper.title); 
    setInputValue(paper.title);
    setView('search');
    setExpandedPaperId(paper.id);
  };

  const handleCitationClick = (citationName) => {
    setSearchQuery(citationName);
    setInputValue(citationName);
    setView('search');
    const found = MOCK_PAPERS.find(p => p.title.includes(citationName) || citationName.includes(p.title));
    if (found) setExpandedPaperId(found.id);
  };

  const sendMessage = (e) => {
    e.preventDefault();
    if (!msgInput.trim()) return;
    setMessages([...messages, { role: 'user', text: msgInput }]);
    setTimeout(() => {
      setMessages(prev => [...prev, { role: 'system', text: `I've analyzed the papers in your context regarding "${msgInput}". Here is a summary based on the abstracts provided...` }]);
    }, 1000);
    setMsgInput('');
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-900 selection:bg-indigo-100 selection:text-indigo-900">
      <Header 
        view={view} 
        setView={setView} 
        inputValue={inputValue} 
        setInputValue={setInputValue} 
        handleSearchSubmit={handleSearchSubmit} 
        aiContextPapers={aiContextPapers}
        searchContextPapers={searchContextPapers}
        setShowAddContextModal={setShowAddContextModal}
      />
      
      <main className="transition-all duration-300">
        {view === 'home' && (
          <HomeView 
            inputValue={inputValue} 
            setInputValue={setInputValue} 
            handleSearchSubmit={handleSearchSubmit} 
            setView={setView} 
            setShowAddContextModal={setShowAddContextModal}
            handleTrendingClick={handleTrendingClick}
            searchContextPapers={searchContextPapers}
            handleSuggestedSearch={handleSuggestedSearch}
          />
        )}
        {view === 'search' && (
          <SearchResultsView 
            searchQuery={searchQuery}
            setView={setView}
            handleConceptClick={handleConceptClick}
            searchContextPapers={searchContextPapers}
            aiContextPapers={aiContextPapers}
            toggleSearchContext={toggleSearchContext}
            toggleAiContext={toggleAiContext}
            expandedPaperId={expandedPaperId}
            setExpandedPaperId={setExpandedPaperId}
            setShowCitationModal={setShowCitationModal}
            handleAuthorClick={handleAuthorClick}
            handleCitationClick={handleCitationClick}
          />
        )}
        {view === 'author' && (
          <AuthorProfileView 
            selectedAuthor={selectedAuthor}
            setView={setView}
            setExpandedPaperId={setExpandedPaperId}
          />
        )}
        {view === 'ai' && (
          <AiAssistantPage 
            setView={setView}
            aiContextPapers={aiContextPapers}
            toggleAiContext={toggleAiContext}
            messages={messages}
            msgInput={msgInput}
            setMsgInput={setMsgInput}
            sendMessage={sendMessage}
          />
        )}
        {view === 'concepts' && (
          <ConceptEvolutionView 
            setView={setView}
            currentSubject={currentSubject}
            activeNode={activeNode}
            setActiveNode={setActiveNode}
          />
        )}
      </main>

      <Modal isOpen={showCitationModal} onClose={() => setShowCitationModal(false)} title="Cite this Paper">
         <div className="space-y-4">
            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-sm font-mono text-gray-600 break-all">
               @article{`{paper_id}`},<br/>
               &nbsp;&nbsp;title={`{Title}`},<br/>
               &nbsp;&nbsp;author={`{Authors}`},<br/>
               &nbsp;&nbsp;journal={`{Journal}`},<br/>
               &nbsp;&nbsp;year={`{Year}`}
            </div>
            <div className="flex gap-2">
               <button onClick={() => setShowCitationModal(false)} className="flex-1 py-2 bg-indigo-600 text-white rounded-lg font-bold text-sm hover:bg-indigo-700">Copy BibTeX</button>
               <button onClick={() => setShowCitationModal(false)} className="flex-1 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg font-bold text-sm hover:bg-gray-50">Download .ris</button>
            </div>
         </div>
      </Modal>

      <Modal isOpen={showAddContextModal} onClose={() => setShowAddContextModal(false)} title="Add to Search Context">
         <div className="space-y-4">
            <input 
              type="text" 
              placeholder="Search by title or DOI..." 
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              value={contextSearchQuery}
              onChange={(e) => setContextSearchQuery(e.target.value)}
            />
            <div className="max-h-60 overflow-y-auto space-y-2">
               {MOCK_PAPERS.filter(p => p.title.toLowerCase().includes(contextSearchQuery.toLowerCase())).map(p => (
                 <div key={p.id} className="flex justify-between items-center p-2 hover:bg-gray-50 rounded border border-transparent hover:border-gray-200">
                    <div className="text-sm truncate pr-2">
                      <div className="font-medium text-gray-900 truncate">{p.title}</div>
                      <div className="text-xs text-gray-500">{p.authors.length} authors • {p.date}</div>
                    </div>
                    {searchContextPapers.find(cp => cp.id === p.id) ? (
                      <span className="text-xs text-green-600 font-medium px-2">Added</span>
                    ) : (
                      <button onClick={() => toggleSearchContext(p)} className="p-1 text-indigo-600 hover:bg-indigo-50 rounded"><Plus size={16} /></button>
                    )}
                 </div>
               ))}
            </div>
         </div>
      </Modal>
    </div>
  );
}
