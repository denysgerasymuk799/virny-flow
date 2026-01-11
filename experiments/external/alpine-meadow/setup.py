import os
from setuptools import setup, find_packages
import bz2


packages = find_packages(where='.')
with open('requirements.txt') as f:
    install_requires = [r.rstrip() for r in f.readlines()
                        if not r.startswith('#')]


with open(os.path.join(
        os.path.dirname(__file__), 'alpine_meadow', '__version__.py')) as f:
    for line in f:
        if line.startswith('__version__ ='):
            _, _, version = line.partition('=')
            VERSION = version.strip(" \n'\"")
            break
    else:
        raise RuntimeError('unable to read the version from alpine_meadow/__version__.py')


pkl_files_directory = os.path.join(os.path.dirname(__file__), 'alpine_meadow', 'core', 'meta_learning', 'files')
for file in os.listdir(pkl_files_directory):
    pkl_file = os.path.join(pkl_files_directory, file[:-5])
    if file.endswith('.pbz2') and not os.path.exists(pkl_file):
        with bz2.BZ2File(os.path.join(pkl_files_directory, file), 'rb') as f:
            data = f.read()
        with open(pkl_file, 'wb') as f:
            f.write(data)

setup(
    name='alpine-meadow',
    author='Zeyuan Shang',
    author_email='zs@einblick.ai',
    description='Interactive AutoML',
    version=VERSION,
    packages=packages,
    install_requires=install_requires,
    include_package_data=True,
    python_requires='>=3.6.*',
    url='https://www.einblick.ai'
)
