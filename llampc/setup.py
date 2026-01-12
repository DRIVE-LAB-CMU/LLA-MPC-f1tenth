from setuptools import setup

package_name = 'llampc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Henry Liao',
    author_email='hzl@andrew.cmu.edu',
    maintainer='DRIVE Lab',
    description='f1tenth llampc',
    license='MIT',
    packages=['llampc'],
    entry_points={
        'console_scripts': [
            'llampc = llampc.llampc_node:main',
        ],
    },
)
