from setuptools import setup

package_name = 'yolov4_kalman_fixed'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    py_modules=[
        'scripts.yolov4_kalman_node',
    ],
    install_requires=['setuptools'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    zip_safe=True,
    author='Your Name',
    author_email='your@email.com',
    maintainer='Your Name',
    maintainer_email='your@email.com',
    keywords=['ROS2'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=(
        'A ROS2 node that applies a Kalman Filter to the output of the YOLOv4 object detection algorithm.'
    ),
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov4_kalman_node = scripts.yolov4_kalman_node:main'
        ],
    },
)

