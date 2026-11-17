"""Standard-library integrity verification, usable before installing PyTorch."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--json',action='store_true');args=p.parse_args()
    manifest=json.loads((ROOT/'PACKAGE_SHA256.json').read_text('utf8'));failed=[]
    for relative,expected in manifest.items():
        path=(ROOT/relative).resolve()
        if ROOT not in path.parents or not path.is_file():failed.append(relative);continue
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:failed.append(relative)
    result=dict(passed=not failed,checked_files=len(manifest),failed=failed)
    print(json.dumps(result,ensure_ascii=False) if args.json else ('PASS' if not failed else 'FAIL')+': '+str(len(manifest))+' files')
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
