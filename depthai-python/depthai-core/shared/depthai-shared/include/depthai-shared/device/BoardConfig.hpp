#pragma once

// std
#include <cstdint>
#include <unordered_map>

// project
#include "depthai-shared/common/CameraBoardSocket.hpp"
#include "depthai-shared/common/CameraImageOrientation.hpp"
#include "depthai-shared/common/CameraSensorType.hpp"
#include "depthai-shared/common/UsbSpeed.hpp"
#include "depthai-shared/common/optional.hpp"
#include "depthai-shared/log/LogLevel.hpp"
#include "depthai-shared/utility/Serialization.hpp"
#include "depthai-shared/xlink/XLinkConstants.hpp"

namespace dai {

constexpr static uint32_t BOARD_CONFIG_MAGIC1 = 0x78010000U;
constexpr static uint32_t BOARD_CONFIG_MAGIC2 = 0x21ea17e6U;

struct BoardConfig {
    constexpr static uint32_t SIPP_BUFFER_DEFAULT_SIZE = 18 * 1024;
    constexpr static uint32_t SIPP_DMA_BUFFER_DEFAULT_SIZE = 16 * 1024;
    constexpr static uint32_t ISP_3A_DEFAULT_MAX_FPS_USB = 30;
    constexpr static uint32_t ISP_3A_DEFAULT_MAX_FPS_ETHERNET = 20;

    /// USB related config
    struct USB {
        uint16_t vid = 0x03e7, pid = 0xf63b;
        uint16_t flashBootedVid = 0x03e7, flashBootedPid = 0xf63d;
        UsbSpeed maxSpeed = UsbSpeed::SUPER;
        /**
         * Isp 3A max rate (auto focus, auto exposure, auto white balance) for USB.
         */
        uint32_t isp3aMaxFps = ISP_3A_DEFAULT_MAX_FPS_USB;
    };

    USB usb;

    /// Network configuration
    struct Network {
        /// Network MTU, 0 is auto (usually 1500 for Ethernet) or forwarded
        /// from bootloader (not yet implemented there).
        /// Note: not advised to increase past 1500 for now
        uint16_t mtu = 0;
        /// Sets the `TCP_NODELAY` option for XLink TCP sockets (disable Nagle's algorithm),
        /// reducing latency at the expense of a small hit for max throughput. Default is `true`
        bool xlinkTcpNoDelay = true;
        /**
         * Isp 3A max rate (auto focus, auto exposure, auto white balance) for Ethernet.
         */
        uint32_t isp3aMaxFps = ISP_3A_DEFAULT_MAX_FPS_ETHERNET;
    };

    Network network;

    /// Optional list of FreeBSD sysctl parameters to be set (system, network, etc.).
    /// For example: "net.inet.tcp.delayed_ack=0" (this one is also set by default)
    std::vector<std::string> sysctl;

    /// Watchdog config
    tl::optional<uint32_t> watchdogTimeoutMs;
    tl::optional<uint32_t> watchdogInitialDelayMs;

    /**
     * SIPP (Signal Image Processing Pipeline) internal memory pool.
     * SIPP is a framework used to schedule HW filters, e.g. ISP, Warp, Median filter etc.
     * Changing the size of this pool is meant for advanced use cases, pushing the limits of the HW.
     * By default memory is allocated in high speed CMX memory. Setting to 0 will allocate in DDR 256 kilobytes.
     * Units are bytes.
     */
    uint32_t sippBufferSize = SIPP_BUFFER_DEFAULT_SIZE;
    /**
     * SIPP (Signal Image Processing Pipeline) internal DMA memory pool.
     * SIPP is a framework used to schedule HW filters, e.g. ISP, Warp, Median filter etc.
     * Changing the size of this pool is meant for advanced use cases, pushing the limits of the HW.
     * Memory is allocated in high speed CMX memory
     * Units are bytes.
     */
    uint32_t sippDmaBufferSize = SIPP_DMA_BUFFER_DEFAULT_SIZE;

    /// GPIO config
    struct GPIO {
        enum Mode : std::int8_t { ALT_MODE_0 = 0, ALT_MODE_1, ALT_MODE_2, ALT_MODE_3, ALT_MODE_4, ALT_MODE_5, ALT_MODE_6, DIRECT };
        Mode mode = Mode::DIRECT;
        enum Direction : std::int8_t { INPUT = 0, OUTPUT = 1 };
        Direction direction = Direction::INPUT;
        enum Level : std::int8_t { LOW = 0, HIGH = 1 };
        Level level = Level::LOW;
        enum Pull : std::int8_t { NO_PULL = 0, PULL_UP = 1, PULL_DOWN = 2, BUS_KEEPER = 3 };
        Pull pull = Pull::NO_PULL;
        /// Drive strength in mA (2, 4, 8 and 12mA)
        enum Drive : std::int8_t { MA_2 = 2, MA_4 = 4, MA_8 = 8, MA_12 = 12 };
        Drive drive = MA_2;
        bool schmitt = false, slewFast = false;
        GPIO() = default;
        GPIO(Direction direction) : direction(direction) {}
        GPIO(Direction direction, Level level) : direction(direction), level(level) {}
        GPIO(Direction direction, Level level, Pull pull) : direction(direction), level(level), pull(pull) {}
        GPIO(Direction direction, Mode mode) : mode(mode), direction(direction) {}
        GPIO(Direction direction, Mode mode, Pull pull) : mode(mode), direction(direction), pull(pull) {}
    };
    std::unordered_map<std::int8_t, GPIO> gpio;

    // Uart config

    /// UART instance config
    struct UART {
        // TBD
        // std::int8_t tx, rx;
        std::int8_t tmp;
    };
    /// UART instance map
    std::unordered_map<std::int8_t, UART> uart;

    /// PCIe config
    tl::optional<bool> pcieInternalClock;

    /// USB3 phy config
    tl::optional<bool> usb3PhyInternalClock;

    /// MIPI 4Lane RGB config
    tl::optional<bool> mipi4LaneRgb;

    /// eMMC config
    tl::optional<bool> emmc;

    /// log path
    tl::optional<std::string> logPath;

    /// Max log size
    tl::optional<size_t> logSizeMax;

    /// log verbosity
    tl::optional<LogLevel> logVerbosity;

    /// log device prints
    tl::optional<bool> logDevicePrints;

    bool nonExclusiveMode = false;

    // TODO(themarpe) - add later when applicable
    // // Socket description
    // struct Socket {
    //     int i2cBus;
    //     int mipiStart, mipiEnd;  // inclusive
    //     int gpioPwr, gpioRst;
    //     float voltageCore, voltageAnalog, voltageInterface;
    //     // TODO(themarpe) - tbd if better placed here
    //     // tl::optional<CameraBoardSocket> syncTo;
    // };
    // std::unordered_map<CameraBoardSocket, Socket> socket;

    /// Camera description
    struct Camera {
        std::string name;
        // TODO(themarpe) - add later when applicable
        // std::vector<std::string> sensorName;
        tl::optional<CameraSensorType> sensorType;
        // std::vector<vector> vcm;
        // tl::optional<CameraBoardSocket> syncTo;
        tl::optional<CameraImageOrientation> orientation;
    };
    std::unordered_map<CameraBoardSocket, Camera> camera;

    struct IMU {
        IMU() {
            bus = 0;
            interrupt = 53;
            wake = 34;
            csGpio = 8;
        }
        int8_t bus, interrupt, wake, csGpio;
    };
    tl::optional<IMU> imu;
};

DEPTHAI_SERIALIZE_EXT(BoardConfig::USB, vid, pid, flashBootedVid, flashBootedPid, maxSpeed, isp3aMaxFps);
DEPTHAI_SERIALIZE_EXT(BoardConfig::Network, mtu, xlinkTcpNoDelay, isp3aMaxFps);
DEPTHAI_SERIALIZE_EXT(BoardConfig::GPIO, mode, direction, level, pull, drive, schmitt, slewFast);
DEPTHAI_SERIALIZE_EXT(BoardConfig::UART, tmp);
DEPTHAI_SERIALIZE_EXT(BoardConfig::Camera, name, sensorType, orientation);
DEPTHAI_SERIALIZE_EXT(BoardConfig::IMU, bus, interrupt, wake, csGpio);
DEPTHAI_SERIALIZE_EXT(BoardConfig,
                      usb,
                      network,
                      sysctl,
                      watchdogTimeoutMs,
                      watchdogInitialDelayMs,
                      gpio,
                      uart,
                      pcieInternalClock,
                      usb3PhyInternalClock,
                      emmc,
                      logPath,
                      logSizeMax,
                      logVerbosity,
                      logDevicePrints,
                      nonExclusiveMode,
                      camera,
                      sippBufferSize,
                      sippDmaBufferSize,
                      imu);

}  // namespace dai
