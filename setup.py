import setuptools
from packagename.version import Version


setuptools.setup(name='questions',
                 version=Version('1.0.0').number,
                 description='Question Classification - What, Why, When, Affirmative',
                 long_description=open('README.md').read().strip(),
                 author='Ashish Baghudana',
                 author_email='ashish@baghudana.com',
                 url='https://github.com/ashishbaghudana/question-classification',
                 py_modules=['questions'],
                 install_requires=[],
                 license='MIT License',
                 zip_safe=False,
                 keywords='Question Classification',
                 classifiers=['Packages'])
