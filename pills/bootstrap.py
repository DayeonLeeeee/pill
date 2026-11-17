"""Install pinned tested dependencies into a package-local virtual environment."""
from pathlib import Path
import argparse,os,subprocess,sys,venv

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    if sys.version_info[:2]!=(3,12):raise SystemExit('Python 3.12 64-bit로 실행하세요. 예: py -3.12 bootstrap.py --device cpu')
    if sys.maxsize<=2**32:raise SystemExit('64-bit Python이 필요합니다.')
    root=Path(__file__).resolve().parent;env=root/'.venv';python=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():venv.EnvBuilder(with_pip=True).create(env)
    index='https://download.pytorch.org/whl/cu128' if a.device=='cuda' else 'https://download.pytorch.org/whl/cpu'
    subprocess.run([str(python),'-m','pip','install','--no-cache-dir','torch==2.10.0','torchvision==0.25.0','--index-url',index],check=True)
    subprocess.run([str(python),'-m','pip','install','--no-cache-dir','-r',str(root/'requirements.txt')],check=True)
    subprocess.run([str(python),str(root/'verify_package.py')],check=True)
    print('설치 완료:',python)
    print('예제 실행:',python,root/'predict.py','--inputs',root/'examples/inputs.json','--device',a.device,'--search')

if __name__=='__main__':main()
