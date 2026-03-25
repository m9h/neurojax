# EEG/MEG Tooling Landscape Analysis

## Executive Summary
For a new tool competing with **osl-dynamics** while offering **BIDS support** and a **lightweight** footprint, the strongest strategic direction is to **fork or extend MNELAB**.

**Why?**
*   **GAP**: `osl-dynamics` lacks native BIDS support, requiring data conversion.
*   **SOLUTION**: MNELAB has robust, native BIDS integration (via `mne-bids`).
*   **OPPORTUNITY**: Integrating "Dynamic Connectivity" (HMMs) into the MNELAB GUI creates a tool that solves the "BIDS-to-Analysis" friction point that currently exists in the OSL ecosystem.

## Tool Analysis Matrix

| Feature | osl-dynamics | MNELAB | generic MNE-Python | EEGLAB (Octave) | SigViewer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Focus** | HMMs / Dyn. Conn. | GUI Frontend | General Processing | Processing | Viewing |
| **BIDS Support** | ❌ (Custom formats) | ✅ (Native Read/Write) | ✅ (via mne-bids) | ⚠️ (Plugin required) | ❌ |
| **Lightweight?** | Medium (Dependencies) | ✅ (GUI wrapper) | ❌ (Large dep tree) | ❌ (Heavy legacy) | ✅ (C++) |
| **Octave Support** | ❌ | ❌ | ❌ | ✅ (Slow/Glitchy) | ❌ |
| **Language** | Python | Python | Python | MATLAB/Octave | C++ |

## Detailed Findings

### 1. The Competitor: `osl-dynamics`
*   **Strengths**: Specialized in Hidden Markov Models (HMMs) and Dynamic Network Modes (DyNeMo). Best-in-class for fast scale functional connectivity.
*   **Weaknesses**:
    *   **Data Ingestion**: Does not digest BIDS datasets directly. Users must convert `.bids` -> `.npy` or `.mat` manually.
    *   **User Interface**: Primarily code/script-based.
*   **Your Angle**: A tool that applies these advanced dynamic models *directly* to the BIDS directory structure without an intermediate conversion step.

### 2. The Frontend: MNELAB
*   **Status**: Active, Python-based GUI.
*   **Architecture**: It is effectively a "thin" GUI wrapper around `mne-python`. This makes it the ideal "lightweight" starting point. It inherits `mne-bids` capabilities for free.
*   **Recommendation**: Instead of building a GUI from scratch, fork MNELAB and add a "Dynamic Connectivity" tab. This tab could behave like `osl-dynamics` but read directly from the loaded BIDS structure.

### 3. The Constraint: Octave Support
*   **EEGLAB**: Runs on Octave but is ~2x slower and has significant graphical glitches (scrolling issues). Plugins often fail.
*   **FieldTrip**: Similar issues; intended for MATLAB.
*   **Conclusion**: Supporting Octave will likely cripple a modern "lightweight" tool. The Python ecosystem (MNE/MNELAB) offers superior performance and BIDS compliance. If "lightweight" is the goal, avoiding the MATLAB Runtime (required for SPM-Python) and Octave overhead is critical.

### 4. Lightweight Alternatives
*   **MNE-CPP**: Very lightweight (C++) and performant for real-time. relying on Python for BIDS. Good for *acquisition* but likely too low-level for *dynamic connectivity modeling*.
*   **Dynamax**: A generic JAX library for probabilistic state space models (HMMs).
    *   **Strategy**: Use `Dynamax` as the backend engine. It is JAX-based (unlike `glhmm` or `osl-dynamics`), enabling high-performance, differentiable modeling. You can replicate `osl-dynamics` features by feeding Time-Delay Embedded (TDE) data into `Dynamax`'s specialized HMMs.

## Proposed "Killer App" Architecture
To satisfy your requirements (BIDS + Compete with OSL + Lightweight + Modern/JAX):

1.  **Base**: Fork MNELAB (provides GUI + MNE-BIDS).
2.  **Engine**: Integrate `Dynamax` (JAX) for the modeling core.
3.  **Workflow**:
    *   **Load**: Open BIDS folder (MNELAB native).
    *   **Preprocess**: Standard MNE-Python cleaning + **Time-Delay Embedding (TDE)** step.
    *   **Analyze**: New "Dynamics" Plugin -> Runs `Dynamax` HMM on TDE data.
    *   **Visualize**: Plot State Probabilities and Transition Matrices (New Widgets).
