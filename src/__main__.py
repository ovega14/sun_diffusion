import runpy


def run_submodule(name: str) -> None:
    """Executes a submodule as a .py file."""
    runpy.run_module(name, run_name='__main__')


run_submodule('linalg')
run_submodule('gens')
run_submodule('sun')
run_submodule('action')
run_submodule('utils')
