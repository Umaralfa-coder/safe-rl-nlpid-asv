from setuptools import setup

package_name = 'safe_rl_nlpid'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Umar Alfa',
    maintainer_email='umar.alfa@imt-nord-europe.fr',
    description='Safety-shielded RL tuning of an NLPID controller for ASVs (VRX / ROS 2).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'buran_controller = safe_rl_nlpid.buran_controller:main',
            'wq_sim = safe_rl_nlpid.wq_sim_node:main',
        ],
    },
)
