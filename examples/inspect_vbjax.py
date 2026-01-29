
import vbjax
import inspect

print("VBJAX Version:", vbjax.__version__)
print("\nDir(vbjax):")
print(dir(vbjax))

print("\nChecking for Jansen-Rit...")
if hasattr(vbjax, 'make_jansen_rit_step'):
    print("Found make_jansen_rit_step")
elif hasattr(vbjax, 'jansen_rit'):
    print("Found jansen_rit")
else:
    print("Jansen-Rit not found in top level.")

print("\nChecking for Montbrio...")
if hasattr(vbjax, 'make_montbrio_pazo_roxin_step'):
    print("Found make_montbrio_pazo_roxin_step")
