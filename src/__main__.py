import importlib
import runpy
import warnings
# silence already imported warning from runpy
warnings.simplefilter('ignore')


def run_submodule(name: str) -> None:
    """Executes a submodule as a .py file."""
    full_name = importlib.util.find_spec('.'+name, package=__package__).name
    print(f'Running submodule {name} ({full_name})...')
    runpy.run_module(full_name, run_name='__main__')
    print()


run_submodule('linalg')
run_submodule('gens')
run_submodule('sun')
run_submodule('action')
run_submodule('utils')
run_submodule('heat')
