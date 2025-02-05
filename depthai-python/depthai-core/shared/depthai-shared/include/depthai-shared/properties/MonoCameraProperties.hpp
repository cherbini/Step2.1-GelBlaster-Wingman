#pragma once

#include <vector>

#include "depthai-shared/common/CameraBoardSocket.hpp"
#include "depthai-shared/common/CameraImageOrientation.hpp"
#include "depthai-shared/common/FrameEvent.hpp"
#include "depthai-shared/datatype/RawCameraControl.hpp"
#include "depthai-shared/properties/Properties.hpp"

namespace dai {

/**
 * Specify properties for MonoCamera such as camera ID, ...
 */
struct MonoCameraProperties : PropertiesSerializable<Properties, MonoCameraProperties> {
    static constexpr int AUTO = -1;

    /**
     * Select the camera sensor resolution: 1280×720, 1280×800, 640×400, 640×480, 1920×1200
     */
    enum class SensorResolution : int32_t { THE_720_P, THE_800_P, THE_400_P, THE_480_P, THE_1200_P };

    /*
     * Initial controls applied to MonoCamera node
     */
    RawCameraControl initialControl;

    /**
     * Which socket will mono camera use
     */
    CameraBoardSocket boardSocket = CameraBoardSocket::AUTO;

    /**
     * Which camera name will mono camera use
     */
    std::string cameraName = "";

    /**
     * Camera sensor image orientation / pixel readout
     */
    CameraImageOrientation imageOrientation = CameraImageOrientation::AUTO;

    /**
     * Select the camera sensor resolution
     */
    SensorResolution resolution = SensorResolution::THE_720_P;
    /**
     * Camera sensor FPS
     */
    float fps = 30.0;
    /**
     * Isp 3A rate (auto focus, auto exposure, auto white balance, camera controls etc.).
     * Value (-1) is auto-mode. For USB devices will set 3A fps to maximum 30 fps, for POE devices to maximum 20 fps.
     * Can be overriden by setting explicitly.
     * Default (0) matches the camera FPS, meaning that 3A is running on each frame.
     * Reducing the rate of 3A reduces the CPU usage on CSS, but also increases the convergence rate of 3A.
     * Note that camera controls will be processed at this rate. E.g. if camera is running at 30 fps, and camera control is sent at every frame,
     * but 3A fps is set to 15, the camera control messages will be processed at 15 fps rate, which will lead to queueing.

     */
    int isp3aFps = 0;
    /**
     * Frame pool size for the main output, ISP processed
     */
    int numFramesPool = 3;
    /**
     * Frame pool size for the `raw` output
     */
    int numFramesPoolRaw = 3;
    /**
     * List of events to receive, the rest will be ignored
     */
    std::vector<dai::FrameEvent> eventFilter = {dai::FrameEvent::READOUT_START};
};

DEPTHAI_SERIALIZE_EXT(
    MonoCameraProperties, initialControl, boardSocket, cameraName, imageOrientation, resolution, fps, isp3aFps, numFramesPool, numFramesPoolRaw);

}  // namespace dai
