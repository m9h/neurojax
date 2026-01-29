"""Inspect HCP MAT file deep structure."""
import scipy.io
import numpy as np

fname = 'examples/hcp_data/104012_MEG_10-Motort_tmegpreproc_TFLA.mat'

def inspect_struct(name, obj, indent=0):
    pref = "  " * indent
    if isinstance(obj, np.ndarray):
        print(f"{pref}{name}: {type(obj)} {obj.shape} {obj.dtype}")
        if obj.dtype.names:
            for field in obj.dtype.names:
                val = obj[field][0, 0] # FieldTrip structs are usually 1x1 arrays
                inspect_struct(field, val, indent + 1)
        elif obj.size < 10:
             print(f"{pref}  Value: {obj}")
    else:
        print(f"{pref}{name}: {type(obj)} {obj}")

try:
    mat = scipy.io.loadmat(fname, squeeze_me=True, struct_as_record=False)
    # struct_as_record=False makes MATLAB structs into objects with attributes
    
    if 'data' in mat:
        data = mat['data']
        print(f"Data type: {type(data)}")
        
        # Access attributes
        print("fsample:", data.fsample)
        print("label shape:", data.label.shape)
        print("Label[0]:", data.label[0])
        print("trial shape:", data.trial.shape)
        
        # Trial is likely an array of objects (if multiple trials) or a single array
        if data.trial.dtype == np.object_:
            print("Trial[0] shape:", data.trial[0].shape)
        else:
             print("Trial content shape:", data.trial.shape)

        if hasattr(data, 'grad'):
            grad = data.grad
            print("Grad attributes:", dir(grad))
            if hasattr(grad, 'chanpos'):
                print("grad.chanpos:", grad.chanpos.shape)
            if hasattr(grad, 'coilpos'):
                print("grad.coilpos:", grad.coilpos.shape)
            if hasattr(grad, 'label'):
                print("grad.label shape:", grad.label.shape)
                
except Exception as e:
    print(f"Failed: {e}")
