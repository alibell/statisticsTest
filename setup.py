from setuptools import setup, find_packages
    
    
setup(name='statistics_test', 
        version='0.0.1',
        license='',
        author='Ali BELLAMINE',
        author_email='contact@alibellamine.me',
        description='Additional statistics test absent from Scipy.',
        long_description=open('README.md').read(),
        packages = find_packages()
    )