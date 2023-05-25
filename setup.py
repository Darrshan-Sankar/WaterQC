from setuptools import setup, find_packages

setup(
    name='waterqc',
    version='0.1.1',
    description='Predict the fresh water quality',
    packages=['waterqc'],
    package_data={'':["Color.npy","Month.npy","Source.npy","dataset.csv","scaler.save","model_0.8708945702948415.pkl"]},
    install_requires=[
        'numpy',
        'pandas',
        'polars',
        'joblib',
        'matplotlib',
        'scikit-learn-intelex',
        'scikit-learn',
        'xgboost==1.5',
        'pyarrow',
        'seaborn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    include_package_data=True,
    zip_safe=False
)
