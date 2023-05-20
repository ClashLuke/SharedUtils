import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Lucas Nestler",
    author_email="github.sharedutils@nestler.sh",
    name='sharedutils',
    license='BSD',
    description="Easy usage of Python's new SharedMemory for reduced memory and CPU cost",
    version='0.0.7',
    long_description=README,
    url='https://github.com/clashluke/sharedutils',
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    long_description_content_type="text/markdown",
    install_requires=["numpy"],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
