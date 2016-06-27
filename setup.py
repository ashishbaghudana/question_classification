import setuptools
from packagename.version import Version
from pip.req import parse_requirements

requirements = parse_requirements('requirements.txt', session=False)

setuptools.setup(name='questions',
                 version=Version('1.0.0').number,
                 description='Question Classification - What, Why, When, Affirmative',
                 long_description=open('README.md').read().strip(),
                 author='Ashish Baghudana',
                 author_email='ashish@baghudana.com',
                 url='https://github.com/ashishbaghudana/question-classification',
                 py_modules=['questions'],
                 install_requires=[str(ir.req) for ir in requirements],
                 license='MIT License',
                 zip_safe=False,
                 keywords='Question Classification',
                 classifiers=['Packages'])
