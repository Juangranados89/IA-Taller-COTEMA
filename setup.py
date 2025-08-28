from setuptools import setup, find_packages

setup(
    name="cotema-analytics",
    version="1.0.0",
    description="COTEMA Analytics - Predictive Maintenance System",
    packages=find_packages(),
    install_requires=[
        "Flask==2.3.3",
        "gunicorn==21.2.0",
        "Werkzeug==2.3.7"
    ],
    python_requires=">=3.11"
)
