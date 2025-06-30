from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mppi_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models', 'curiosity'),
            glob('mppi_controller/models/curiosity/*.urdf')),
        (os.path.join('share', package_name, 'models', 'curiosity', 'meshes'),
            glob('mppi_controller/models/curiosity/meshes/*')),
        (os.path.join('share', package_name, 'models', 'canadarm'),
            glob('mppi_controller/models/canadarm/urdf/*.urdf')),
        (os.path.join('share', package_name, 'models', 'canadarm', 'meshes'),
            glob('mppi_controller/models/canadarm/meshes/*')),
        (os.path.join('share', package_name, 'models', 'franka'),
            glob('mppi_controller/models/franka/*.urdf')),
        (os.path.join('share', package_name, 'models', 'franka', 'meshes', 'collision'),
            glob('mppi_controller/models/franka/meshes/collision/*')),
        (os.path.join('share', package_name, 'models', 'franka', 'meshes', 'visual'),
            glob('mppi_controller/models/franka/meshes/visual/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='ch3044637@khu.ac.kr',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'curiosity_controller_node = mppi_controller.curiosity_controller_node:main',
            'canadarm_controller_node = mppi_controller.canadarm_controller_node:main',
            'franka_controller_node = mppi_controller.franka_controller_node:main',
        ],
    },
)
