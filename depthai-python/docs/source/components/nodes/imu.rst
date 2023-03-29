IMU
===

IMU (`intertial measurement unit <https://en.wikipedia.org/wiki/Inertial_measurement_unit>`__) node can be used to receive data
from the IMU chip on the device. Our OAK devices use either:

- `BNO085 <https://www.ceva-dsp.com/product/bno080-085/>`__ (`datasheet here <https://www.ceva-dsp.com/wp-content/uploads/2019/10/BNO080_085-Datasheet.pdf>`__) 9-axis sensor, combining accelerometer, gyroscope, and magnetometer. It also does sensor fusion on the (IMU) chip itself. We have efficiently integrated `this driver <https://github.com/hcrest/bno080-driver>`__ into the DepthAI.
- `BMI270 <https://www.bosch-sensortec.com/products/motion-sensors/imus/bmi270/>`__ 6-axis sensor, combining accelerometer and gyroscope

. The IMU chip is connected to the `RVC <https://docs.luxonis.com/projects/hardware/en/latest/pages/rvc/rvc2.html#rvc2>`__
over SPI.


How to place it
###############

.. tabs::

  .. code-tab:: py

    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)

  .. code-tab:: c++

    dai::Pipeline pipeline;
    auto imu = pipeline.create<dai::node::IMU>();


Inputs and Outputs
##################

.. code-block::

  ┌──────────────┐
  │              │
  │              │      out
  │     IMU      ├─────────►
  │              │
  │              │
  └──────────────┘

**Message types**

- :code:`out` - :ref:`IMUData`

Limitations
###########

- For BNO086, gyroscope frequency above 400Hz can produce some jitter from time to time due to sensor HW limitation.
- **Maximum frequencies**: 500 Hz raw accelerometer, 1000 Hz raw gyroscope values individually, and 500Hz combined (synced) output. You can obtain the combined synced 500Hz output with :code:`imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 500)`.

Usage
#####

.. tabs::

  .. code-tab:: py

    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)

    # enable ACCELEROMETER_RAW and GYROSCOPE_RAW at 100 hz rate
    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 100)
    # above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu.setBatchReportThreshold(1)
    # maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    # if lower or equal to batchReportThreshold then the sending is always blocking on device
    # useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu.setMaxBatchReports(10)

  .. code-tab:: c++

    dai::Pipeline pipeline;
    auto imu = pipeline.create<dai::node::IMU>();

    // enable ACCELEROMETER_RAW and GYROSCOPE_RAW at 100 hz rate
    imu->enableIMUSensor({dai::IMUSensor::ACCELEROMETER_RAW, dai::IMUSensor::GYROSCOPE_RAW}, 100);
    // above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu->setBatchReportThreshold(1);
    // maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    // if lower or equal to batchReportThreshold then the sending is always blocking on device
    // useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu->setMaxBatchReports(10);

IMU devices
###########

List of devices that have an IMU sensor on-board:

* `OAK-D <https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1098OAK.html>`__
* `OAK-D-PoE <https://docs.luxonis.com/projects/hardware/en/latest/pages/SJ2088POE.html>`__
* `OAK-D CM4 PoE <https://docs.luxonis.com/projects/hardware/en/latest/pages/SJ2088POE.html>`__
* `OAK-FFC-3P <https://docs.luxonis.com/projects/hardware/en/latest/pages/DM1090.html>`__
* `OAK-FFC-4P <https://docs.luxonis.com/projects/hardware/en/latest/pages/DD2090.html>`__
* `OAK-D Pro <https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9098pro.html>`__ (All varients)
* `OAK-D S2 <https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9098s2.html>`__ (All varients)
* `OAK-D S2 PoE <https://docs.luxonis.com/projects/hardware/en/latest/pages/NG9097s2.html>`__ (All varients)
* `OAK-D Pro PoE <https://docs.luxonis.com/projects/hardware/en/latest/pages/NG9097pro.html>`__ (All varients)


IMU sensors
###########

When enabling the IMU sensors (:code:`imu.enableIMUSensor()`), you can select between the following sensors:

- :code:`ACCELEROMETER_RAW`
- :code:`ACCELEROMETER`
- :code:`LINEAR_ACCELERATION`
- :code:`GRAVITY`
- :code:`GYROSCOPE_RAW`
- :code:`GYROSCOPE_CALIBRATED`
- :code:`GYROSCOPE_UNCALIBRATED`
- :code:`MAGNETOMETER_RAW`
- :code:`MAGNETOMETER_CALIBRATED`
- :code:`MAGNETOMETER_UNCALIBRATED`
- :code:`ROTATION_VECTOR`
- :code:`GAME_ROTATION_VECTOR`
- :code:`GEOMAGNETIC_ROTATION_VECTOR`
- :code:`ARVR_STABILIZED_ROTATION_VECTOR`
- :code:`ARVR_STABILIZED_GAME_ROTATION_VECTOR`

Here are **descriptions of all sensors**:

.. autoclass:: depthai.IMUSensor
  :noindex:

Examples of functionality
#########################

- :ref:`IMU Accelerometer & Gyroscope`
- :ref:`IMU Rotation Vector`

Reference
#########

.. tabs::

  .. tab:: Python

    .. autoclass:: depthai.node.IMU
      :members:
      :inherited-members:
      :noindex:

  .. tab:: C++

    .. doxygenclass:: dai::node::IMU
      :project: depthai-core
      :members:
      :private-members:
      :undoc-members:

.. include::  ../../includes/footer-short.rst
