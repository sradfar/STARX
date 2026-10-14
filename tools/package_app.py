"""Create a clean deployment archive from an explicit allowlist."""
import hashlib
import json
from pathlib import Path
import zipfile

APP = Path(__file__).resolve().parents[1]
DEST = APP.parent/'STARX_Streamlit_deployment.zip'
paths = [APP/name for name in ['app.py', 'requirements.txt', 'README.md', 'DEPLOY.md', '.gitignore']]
for directory in ['starx', 'assets', 'tests', 'tools', '.streamlit', '.github']:
    paths += [p for p in (APP/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc')]
paths = [p for p in paths if p.name != 'secrets.toml']
allowed_extensions = {'.py', '.json', '.csv', '.gz', '.ubj', '.png', '.jpg', '.txt', '.md', '.toml', '.yml', ''}
for path in paths:
    assert path.suffix in allowed_extensions, path
    assert not any(x in path.parts for x in ['.venv', '.runtime', 'runtime', 'test-output']), path
    assert path.stat().st_size < 100*1024**2, f'Exceeds GitHub per-file limit: {path}'
with zipfile.ZipFile(DEST, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(paths):
        z.write(path, path.relative_to(APP).as_posix())
report = dict(archive=str(DEST), files=len(paths), archive_mb=round(DEST.stat().st_size/1e6, 1),
              maximum_file_mb=round(max(p.stat().st_size for p in paths)/1e6, 1),
              sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
              source_models=30, excludes='manuscripts, credentials, environments, session runtime and test outputs')
(APP/'test-output/package_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
