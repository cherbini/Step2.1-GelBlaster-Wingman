import warnings
from enum import Enum
from typing import Optional, Union, Any, Dict, Tuple

import cv2
import depthai as dai
import numpy as np
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component
from depthai_sdk.components.parser import parse_cam_socket, parse_median_filter, parse_encode
from depthai_sdk.components.undistort import _get_mesh
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout
from depthai_sdk.oak_outputs.xout.xout_depth import XoutDepth
from depthai_sdk.oak_outputs.xout.xout_disparity import XoutDisparity
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_h26x import XoutH26x
from depthai_sdk.oak_outputs.xout.xout_mjpeg import XoutMjpeg
from depthai_sdk.replay import Replay
from depthai_sdk.visualize.configs import StereoColor


class WLSLevel(Enum):
    """WLS filter level"""
    LOW = (1000, 0.8)
    MEDIUM = (6000, 1.5)
    HIGH = (12000, 2.0)


class StereoComponent(Component):

    @property
    def depth(self) -> dai.Node.Output:
        # Depth output from the StereoDepth node.
        return self.node.depth

    @property
    def disparity(self) -> dai.Node.Output:
        # Disparity output from the StereoDepth node.
        return self.node.disparity

    def __init__(self,
                 pipeline: dai.Pipeline,
                 resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
                 fps: Optional[float] = None,
                 left: Union[None, CameraComponent, dai.node.MonoCamera] = None,  # Left mono camera
                 right: Union[None, CameraComponent, dai.node.MonoCamera] = None,  # Right mono camera
                 replay: Optional[Replay] = None,
                 args: Any = None,
                 name: Optional[str] = None,
                 encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None):
        """
        Args:
            pipeline (dai.Pipeline): DepthAI pipeline
            resolution (str/SensorResolution): If monochrome cameras aren't already passed, create them and set specified resolution
            fps (float): If monochrome cameras aren't already passed, create them and set specified FPS
            left (None / dai.None.Output / CameraComponent): Left mono camera source. Will get handled by Camera object.
            right (None / dai.None.Output / CameraComponent): Right mono camera source. Will get handled by Camera object.
            replay (Replay object, optional): Replay
            args (Any, optional): Use user defined arguments when constructing the pipeline
            name (str, optional): Name of the output stream
            encode (str/bool/Profile, optional): Encode the output stream
        """
        super().__init__()
        self.out = self.Out(self)

        self.left: Union[None, CameraComponent, dai.node.MonoCamera, dai.node.ColorCamera, dai.Node.Output]
        self.right: Union[None, CameraComponent, dai.node.MonoCamera, dai.node.ColorCamera, dai.Node.Output]

        self._left_stream: dai.Node.Output
        self._right_stream: dai.Node.Output

        self.colormap = None

        self._replay: Optional[Replay] = replay
        self._resolution: Optional[Union[str, dai.MonoCameraProperties.SensorResolution]] = resolution
        self._fps: Optional[float] = fps
        self._args: Dict = args
        self.name = name

        self.left = left
        self.right = right

        self.node: dai.node.StereoDepth = pipeline.createStereoDepth()
        self.node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

        self.encoder = None
        if encode:
            self.encoder = pipeline.createVideoEncoder()
            # MJPEG by default
            self._encoderProfile = parse_encode(encode)

        # Configuration variables
        self._colorize = None
        self._postprocess_colormap = None
        self.wls_enabled = None
        self._wls_level = None
        self._wls_lambda = None
        self._wls_sigma = None

        self._undistortion_offset: Optional[int] = None

    def _get_output_stream(self, input: Union[
        CameraComponent, dai.node.MonoCamera, dai.node.ColorCamera, dai.Node.Output
    ]) -> dai.Node.Output:
        if isinstance(input, CameraComponent):
            return input.stream
        elif isinstance(input, dai.node.MonoCamera):
            return input.out
        elif isinstance(input, dai.node.ColorCamera):
            return input.video
        elif isinstance(input, dai.Node.Output):
            return input
        else:
            raise ValueError('get_output_stream() accepts either CameraComponent,'
                             'dai.node.MonoCamera, dai.node.ColorCamera, dai.Node.Output!')

    def _get_stream_size(self,
                         input: Union[CameraComponent, dai.node.MonoCamera, dai.node.ColorCamera, dai.Node.Output]) -> \
            Optional[Tuple[int, int]]:
        if isinstance(input, CameraComponent):
            return input.stream_size
        elif isinstance(input, dai.node.MonoCamera):
            return input.getResolutionSize()
        elif isinstance(input, dai.node.ColorCamera):
            return input.getVideoSize()
        else:
            return None

    def on_init(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        if not self._replay:
            # Live stream, check whether we have correct cameras
            if len(device.getCameraSensorNames()) == 1:
                raise Exception('OAK-1 camera does not have Stereo camera pair!')

            # If not specified, default to 400P resolution for faster processing
            self._resolution = self._resolution or dai.MonoCameraProperties.SensorResolution.THE_400_P

            if not self.left:
                self.left = CameraComponent(pipeline, 'left', self._resolution, self._fps, replay=self._replay)
                self.left.on_init(pipeline, device, version)
            if not self.right:
                self.right = CameraComponent(pipeline, 'right', self._resolution, self._fps, replay=self._replay)
                self.right.on_init(pipeline, device, version)

            if 0 < len(device.getIrDrivers()):
                laser = self._args.get('irDotBrightness', None)
                laser = laser if laser is not None else 800
                if 0 < laser:
                    device.setIrLaserDotProjectorBrightness(laser)
                    print(f'Setting IR laser dot projector brightness to {laser}mA')

                led = self._args.get('irFloodBrightness', None)
                if led is not None:
                    device.setIrFloodLightBrightness(int(led))
                    print(f'Setting IR flood LED brightness to {int(led)}mA')

            input_size = self._get_stream_size(self.left)
            if input_size:
                self.node.setInputResolution(*input_size)

        self._left_stream = self._get_output_stream(self.left)
        self._right_stream = self._get_output_stream(self.right)

        if self._replay:  # Replay
            self._replay.initStereoDepth(self.node, left_name=self.left._source, right_name=self.right._source)
        else:
            self._left_stream.link(self.node.left)
            self._right_stream.link(self.node.right)

        colormap_manip = None
        if self.colormap:
            colormap_manip = pipeline.create(dai.node.ImageManip)
            colormap_manip.initialConfig.setColormap(self.colormap, self.node.initialConfig.getMaxDisparity())
            colormap_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
            colormap_manip.setMaxOutputFrameSize(1200 * 800 * 3)
            self.node.disparity.link(colormap_manip.inputImage)

        if self.encoder:
            try:
                fps = self.left.get_fps()  # CameraComponent
            except AttributeError:
                fps = self.left.getFps()  # MonoCamera

            self.encoder.setDefaultProfilePreset(fps, self._encoderProfile)
            if colormap_manip:
                colormap_manip.out.link(self.encoder.input)
            else:
                self.node.disparity.link(self.encoder.input)

        self.node.setRectifyEdgeFillColor(0)

        if self._undistortion_offset is not None:
            calibData = self._replay._calibData if self._replay else device.readCalibration()
            w_frame, h_frame = self._get_stream_size(self.left)
            mapX_left, mapY_left, mapX_right, mapY_right = self._get_maps(w_frame, h_frame, calibData)
            mesh_l = _get_mesh(mapX_left, mapY_left)
            mesh_r = _get_mesh(mapX_right, mapY_right)
            meshLeft = list(mesh_l.tobytes())
            meshRight = list(mesh_r.tobytes())
            self.node.loadMeshData(meshLeft, meshRight)

        if self._args:
            self._config_stereo_args(self._args)

    def config_undistortion(self, M2_offset: int = 0):
        self._undistortion_offset = M2_offset

    def _config_stereo_args(self, args: Dict):
        if not isinstance(args, Dict):
            args = vars(args)  # Namespace -> Dict

        self.config_stereo(
            confidence=args.get('disparityConfidenceThreshold', None),
            median=args.get('stereoMedianSize', None),
            extended=args.get('extendedDisparity', None),
            subpixel=args.get('subpixel', None),
            lr_check=args.get('lrCheck', None),
            sigma=args.get('sigma', None),
            lr_check_threshold=args.get('lrcThreshold', None),
        )

    def config_stereo(self,
                      confidence: Optional[int] = None,
                      align: Union[None, str, dai.CameraBoardSocket] = None,
                      median: Union[None, int, dai.MedianFilter] = None,
                      extended: Optional[bool] = None,
                      subpixel: Optional[bool] = None,
                      lr_check: Optional[bool] = None,
                      sigma: Optional[int] = None,
                      lr_check_threshold: Optional[int] = None,
                      ) -> None:
        """
        Configures StereoDepth modes and options.
        """
        if confidence is not None: self.node.initialConfig.setConfidenceThreshold(confidence)
        if align is not None: self.node.setDepthAlign(parse_cam_socket(align))
        if median is not None: self.node.setMedianFilter(parse_median_filter(median))
        if extended is not None: self.node.initialConfig.setExtendedDisparity(extended)
        if subpixel is not None: self.node.initialConfig.setSubpixel(subpixel)
        if lr_check is not None: self.node.initialConfig.setLeftRightCheck(lr_check)
        if sigma is not None: self.node.initialConfig.setBilateralFilterSigma(sigma)
        if lr_check_threshold is not None: self.node.initialConfig.setLeftRightCheckThreshold(lr_check_threshold)

    def config_postprocessing(self,
                              colorize: Union[StereoColor, bool] = None,
                              colormap: int = None
                              ) -> None:
        """
        Configures postprocessing options.

        Args:
            colorize: Colorize the disparity map. Can be either a StereoColor enum, string or bool.
            colormap: Colormap to use for colorizing the disparity map.
        """
        if colorize is None:
            self._colorize = StereoColor.GRAY
        elif isinstance(colorize, bool):
            self._colorize = StereoColor.RGB if colorize else StereoColor.GRAY
        elif isinstance(colorize, StereoColor):
            self._colorize = colorize
        elif isinstance(colorize, str):
            self._colorize = StereoColor[colorize.upper()]

        self._postprocess_colormap = colormap

    def config_wls(self,
                   wls_level: Union[WLSLevel, str] = None,
                   wls_lambda: float = None,
                   wls_sigma: float = None
                   ) -> None:
        """
        Configures WLS filter options.

        Args:
            wls_level: WLS filter level. Can be either a WLSLevel enum or string.
            wls_lambda: WLS filter lambda.
            wls_sigma: WLS filter sigma.
        """
        self.wls_enabled = True if wls_level else False

        if isinstance(wls_level, WLSLevel):
            self._wls_level = wls_level
        elif isinstance(wls_level, str):
            self._wls_level = WLSLevel[wls_level.upper()]

        self._wls_lambda = wls_lambda
        self._wls_sigma = wls_sigma

    def set_colormap(self, colormap: dai.Colormap):
        """
        Sets the colormap to use for colorizing the disparity map. Used for on-device postprocessing.

        Args:
            colormap: Colormap to use for colorizing the disparity map.
        """
        self.colormap = colormap

    def _get_disparity_factor(self, device: dai.Device) -> float:
        """
        Calculates the disparity factor used to calculate depth from disparity.
        `depth = disparity_factor / disparity`
        @param device: OAK device
        """
        calib = device.readCalibration()
        baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10  # mm
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, self.right.getResolutionSize())
        focalLength = intrinsics[0][0]
        disp_levels = self.node.getMaxDisparity() / 95
        return baseline * focalLength * disp_levels

    def _get_maps(self, width: int, height: int, calib: dai.CalibrationHandler):
        imageSize = (width, height)
        M1 = np.array(calib.getCameraIntrinsics(calib.getStereoLeftCameraId(), width, height))
        M2 = np.array(calib.getCameraIntrinsics(calib.getStereoRightCameraId(), width, height))
        d1 = np.array(calib.getDistortionCoefficients(calib.getStereoLeftCameraId()))
        d2 = np.array(calib.getDistortionCoefficients(calib.getStereoRightCameraId()))
        R1 = np.array(calib.getStereoLeftRectificationRotation())
        R2 = np.array(calib.getStereoRightRectificationRotation())

        # increaseOffset = -100 if width == 1152 else -166.67
        """ increaseOffset = 0
        M2_focal = M2.copy()
        M2_focal[0][0] += increaseOffset
        M2_focal[1][1] += increaseOffset
        kScaledL = M2_focal
        kScaledR = kScaledL """

        M2[0][0] += self._undistortion_offset
        M2[1][1] += self._undistortion_offset

        mapX_l, mapY_l = cv2.initUndistortRectifyMap(M1, d1, R1, M2, imageSize, cv2.CV_32FC1)
        mapX_r, mapY_r = cv2.initUndistortRectifyMap(M2, d2, R2, M2, imageSize, cv2.CV_32FC1)
        return mapX_l, mapY_l, mapX_r, mapY_r

    """
    Available outputs (to the host) of this component
    """

    class Out:
        def __init__(self, stereo_component: 'StereoComponent'):
            self._comp = stereo_component

        def main(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            # By default, we want to show disparity
            return self.depth(pipeline, device)

        def disparity(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            if self._comp.colormap:
                warnings.warn('Colormap set with `set_colormap` is ignored when using disparity output. '
                              'Please use `configure_postprocessing` instead.')

            fps = self._comp.left.get_fps() if self._comp._replay is None else self._comp._replay.getFps()

            out = XoutDisparity(
                disparity_frames=StreamXout(self._comp.node.id, self._comp.disparity, name=self._comp.name),
                mono_frames=StreamXout(self._comp.node.id, self._comp._right_stream, name=self._comp.name),
                max_disp=self._comp.node.getMaxDisparity(),
                fps=fps,
                colorize=self._comp._colorize,
                colormap=self._comp._postprocess_colormap,
                use_wls_filter=self._comp.wls_enabled,
                wls_level=self._comp._wls_level,
                wls_lambda=self._comp._wls_lambda,
                wls_sigma=self._comp._wls_sigma
            )

            return self._comp._create_xout(pipeline, out)

        def rectified_left(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            fps = self._comp.left.getFps() if self._comp._replay is None else self._comp._replay.get_fps()
            out = XoutFrames(
                frames=StreamXout(self._comp.node.id, self._comp.node.rectifiedLeft),
                fps=fps)
            out.name = 'Rectified left'
            return self._comp._create_xout(pipeline, out)

        def rectified_right(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            fps = self._comp.left.getFps() if self._comp._replay is None else self._comp._replay.get_fps()
            out = XoutFrames(
                frames=StreamXout(self._comp.node.id, self._comp.node.rectifiedRight),
                fps=fps)
            out.name = 'Rectified right'
            return self._comp._create_xout(pipeline, out)

        def depth(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            if self._comp.colormap:
                warnings.warn('Colormap set with `set_colormap` is ignored when using depth output. '
                              'Please use `configure_postprocessing` instead.')

            fps = self._comp.left.get_fps() if self._comp._replay is None else self._comp._replay.getFps()
            out = XoutDepth(
                device=device,
                frames=StreamXout(self._comp.node.id, self._comp.depth, name=self._comp.name),
                mono_frames=StreamXout(self._comp.node.id, self._comp._right_stream, name=self._comp.name),
                fps=fps,
                colorize=self._comp._colorize,
                colormap=self._comp._postprocess_colormap,
                use_wls_filter=self._comp.wls_enabled,
                wls_level=self._comp._wls_level,
                wls_lambda=self._comp._wls_lambda,
                wls_sigma=self._comp._wls_sigma
            )
            return self._comp._create_xout(pipeline, out)

        def encoded(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            if not self._comp.encoder:
                raise RuntimeError('Encoder not enabled, cannot output encoded frames')

            if self._comp.wls_enabled:
                warnings.warn('WLS filter is enabled, but cannot be applied to encoded frames.')

            if self._comp._encoderProfile == dai.VideoEncoderProperties.Profile.MJPEG:
                out = XoutMjpeg(frames=StreamXout(self._comp.encoder.id, self._comp.encoder.bitstream),
                                color=self._comp.colormap is not None,
                                lossless=self._comp.encoder.getLossless(),
                                fps=self._comp.encoder.getFrameRate(),
                                frame_shape=(1200, 800))
            else:
                out = XoutH26x(frames=StreamXout(self._comp.encoder.id, self._comp.encoder.bitstream),
                               color=self._comp.colormap is not None,
                               profile=self._comp._encoderProfile,
                               fps=self._comp.encoder.getFrameRate(),
                               frame_shape=(1200, 800))

            return self._comp._create_xout(pipeline, out)
