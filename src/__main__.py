import runpy


def run_submodule(name: str) -> None:
    """Executes a submodule as a .py file."""
    print(f'Running submodule {name}...')
    runpy.run_module(name, run_name='__main__')
    print()


run_submodule('linalg')
run_submodule('gens')
run_submodule('sun')
run_submodule('action')
run_submodule('utils')
run_submodule('heat')
