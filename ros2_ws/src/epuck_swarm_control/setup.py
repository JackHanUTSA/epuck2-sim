from setuptools import setup

package_name = 'epuck_swarm_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/epuck_braitenberg_launch.py',
            'launch/epuck_team_work_launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JackHanUTSA',
    maintainer_email='zhifeng.han@utsa.edu',
    description='Reactive Braitenberg-style control for Webots e-puck2 via ROS 2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'epuck_braitenberg = epuck_swarm_control.epuck_braitenberg:main',
            'epuck_team_dashboard = epuck_swarm_control.epuck_team_dashboard:main',
        ],
    },
)
