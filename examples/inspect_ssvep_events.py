
import os
import sys
import mne
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from neurojax.io.cmi import CMILoader

SUBJECT_ID = "sub-NDARGU729WUR"

def inspect_events():
    print(f"=== Inspecting SSVEP Events for {SUBJECT_ID} ===")
    
    loader = CMILoader(SUBJECT_ID)
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print("\n--- Raw Annotations ---")
    print(raw.annotations)
    
    # Extract Events
    # MNE reads .set events automatically into annotations or stim channel
    events, event_id = mne.events_from_annotations(raw)
    
    print(f"\nTotal Events: {len(events)}")
    print(f"Event IDs: {event_id}")
    
    # Analyze Inter-Stimulus Intervals or patterns
    # Standard SSVEP (CCD):
    # - Block structre?
    # - Stimulus (30Hz) ON
    # - Targets (Contrast Change)
    
    # Let's plot the event track
    fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_id)
    plt.savefig("ssvep_event_plot.png")
    print("Saved 'ssvep_event_plot.png'")
    
    # print first 20 events
    print("\nFirst 20 Events:")
    for i in range(min(20, len(events))):
        s, _, id = events[i]
        # Reverse map ID to name
        name = [k for k, v in event_id.items() if v == id][0]
        print(f"Sample {s}: {name} ({id}) -> Time: {s/raw.info['sfreq']:.2f}s")

if __name__ == "__main__":
    inspect_events()
