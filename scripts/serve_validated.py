"""Run the unchanged, hash-checked final ordinary-service launcher."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    launcher = Path('/home/hrxu/nano-full-opt-20261024/serve_validated.py')
    if not launcher.is_file():
        raise SystemExit('Frozen delivery missing; read docs/RUNNING.md before launching.')
    runpy.run_path(str(launcher), run_name='__main__')
