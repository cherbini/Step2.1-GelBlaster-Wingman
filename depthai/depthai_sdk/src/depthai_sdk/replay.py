import os
import time
from enum import IntEnum
from threading import Thread
from time import monotonic

import depthai as dai
import numpy as np

from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.readers.abstract_reader import AbstractReader
from depthai_sdk.utils import *

_fileTypes = ['color', 'left', 'right', 'disparity', 'depth']
_videoExt = ['.mjpeg', '.avi', '.mp4', '.h265', '.h264', '.webm']
_imageExt = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm',
             '.pnm', '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']

class ReplayStream:
    stream_name: str # XLink stream name
    queue: dai.DataInputQueue # Input queue
    frame: np.ndarray # Last read frame from Reader (ndarray)
    imgFrame: dai.ImgFrame # Last read ImgFrame from Reader (dai.ImgFrame)
    _shape: Tuple[int, int] # width, height
    disabled: bool
    size_bytes: int # bytes

    @property
    def shape(self) -> Tuple[int,int]:
        return self.resize if self.resize else self._shape

    def __init__(self):
        self.node: dai.node.XLinkIn = None
        self.disabled = False
        self.stream_name = ''
        self.camera_socket: dai.CameraBoardSocket = None

        self.resize: Tuple[int,int] = None
        self.resize_mode: ResizeMode = None

    def get_socket(self) -> dai.CameraBoardSocket:
        if self.camera_socket:
            return self.camera_socket
        if 'left' in self.stream_name:
            return dai.CameraBoardSocket.LEFT
        elif 'right' in self.stream_name:
            return dai.CameraBoardSocket.RIGHT
        else:
            return dai.CameraBoardSocket.RGB
        # raise Exception("Please specify replay stream CameraBoardSocket via replay.specify_socket()")

class Replay:
    def __init__(self, path: str):
        """
        Helper file to replay recorded depthai stream. It reads from recorded files (mjpeg/avi/mp4/h265/h264/bag)
        and sends frames back to OAK camera to replay the scene, including depth reconstruction from 2 synced mono
        streams.
    
        Args:
            path (str): Path to the recording folder
        """
        self.path = self._get_path(path)

        self.disabledStreams: List[str] = []
        self.streams: Dict[str, ReplayStream] = {}

        self._inputQueues = dict()  # dai.InputQueue dictionary for each stream
        self._seqNum = 0  # Frame sequence number, added to each imgFrame
        self._now: monotonic = None
        self._colorSize = None
        self._keepAR = True  # By default, crop image as needed to keep the aspect ratio
        self._pause = False
        self._calibData = None

        self.fps: float = 30.0
        self.thread: Optional[Thread] = None
        self._stop: bool = False  # Stop the thread that's sending frames to the OAK camera

        self.xins: List[str] = []  # Name of XLinkIn streams

        self.reader: Optional[AbstractReader] = None
        self.frames: Dict[str, np.ndarray] = dict()  # Frames read from Readers
        self.imgFrames: Dict[str, dai.ImgFrame] = dict()  # Last frame sent to the device

        def cntFilesExt(path: Path, ext: Union[str, List[str]]) -> int:
            def fileWithExt(file: str) -> bool:
                if isinstance(ext, List):
                    return os.path.splitext(file)[1] in ext
                elif isinstance(ext, str):
                    return file.endswith(ext)
                else:
                    raise ValueError('ext should be either str or List[str]!')

            return [fileWithExt(f) for f in os.listdir(str(path))].count(True)

        if self.path.is_dir():  # Provided path is a folder
            if 0 < cntFilesExt(self.path, _imageExt):
                from .readers.image_reader import ImageReader
                self.reader = ImageReader(self.path)
            elif 0 < cntFilesExt(self.path, _videoExt):
                from .readers.videocap_reader import VideoCapReader
                self.reader = VideoCapReader(self.path)
            elif cntFilesExt(self.path, '.bag') == 1:
                from .readers.rosbag_reader import RosbagReader
                self.reader = RosbagReader(self.path)
            elif cntFilesExt(self.path, '.mcap') == 1:
                from .readers.mcap_reader import McapReader
                self.reader = McapReader(self.path)
            elif cntFilesExt(self.path, '.db3') == 1:
                from .readers.db3_reader import Db3Reader
                self.reader = Db3Reader(self.path)
            else:
                raise RuntimeError("Path invalid - no recordings found.")

            # Read calibration file
            calib_file = self.path / 'calib.json'
            if calib_file.exists():
                self._calibData = dai.CalibrationHandler(str(calib_file))

        else:  # Provided path is a file
            if self.path.suffix in _videoExt:
                from .readers.videocap_reader import VideoCapReader
                self.reader = VideoCapReader(self.path)
            elif self.path.suffix in _imageExt:
                from .readers.image_reader import ImageReader
                self.reader = ImageReader(self.path)
            else:
                raise NotImplementedError('Please select folder')

        # Read all available streams
        for stream_name in self.reader.getStreams():
            stream = ReplayStream()
            stream._shape = self.reader.getShape(stream_name)
            stream.size_bytes = self.reader.get_message_size(stream_name)
            self.streams[stream_name] = stream

    def _get_path(self, path: str) -> Path:
        """
        Either use local depthai-recording, YT link, mp4 url
        @param path: depthai-recording path.
        @return: Replay module
        """
        if isUrl(path):
            if isYoutubeLink(path):
                # Overwrite source - so Replay class can use it
                return downloadYTVideo(path)
            else:
                return downloadContent(path)

        if Path(path).resolve().exists():
            return Path(path).resolve()

        recording_name: str = path
        # Check if we have it stored locally
        path: Path = getLocalRecording(recording_name)
        if path is not None:
            return path

        # Try to download from the server
        dic = getAvailableRecordings()
        if recording_name in dic:
            arr = dic[recording_name]
            print("Downloading depthai recording '{}' from Luxonis' servers, in total {:.2f} MB".format(recording_name,
                                                                                                        arr[1] / 1e6))
            path = downloadRecording(recording_name, arr[0])
            return path
        else:
            raise ValueError(f"DepthAI recording '{recording_name}' was not found on the server!")

    def toggle_pause(self):
        """
        Toggle pausing of sending frames to the OAK camera.
        """
        self._pause = not self._pause

    def set_fps(self, fps: float):
        """
        Sets frequency at which Replay module will send frames to the camera. Default 30FPS.
        """
        if type(self.reader).__name__ == 'ImageReader':
            self.reader.set_cycle_fps(fps)
        else:
            self.fps = fps

    def get_fps(self) -> float:
        return self.fps

    def resize(self, stream_name: str, size: Tuple[int,int], mode: ResizeMode = ResizeMode.STRETCH):
        """
        Resize color frames prior to sending them to the device.

        Args:
            stream_name (str): Name of the stream we want to resize
            size (Tuple(width, heigth)): Size of color frames that are sent to the camera
            mode (ResizeMode): How to actually resize the stream
        """
        self.streams[stream_name].resize = size
        self.streams[stream_name].resize_mode = mode

    def keepAspectRatio(self, keepAspectRatio: bool):
        raise Exception('keepAspectRatio() has been deprecated, use resize(mode=ResizeMode) to set whether to keep AR!')

    def disableStream(self, stream_name: str, disableReading: bool = False):
        """
        Disable sending a recorded stream to the device.

        Args:
            streamName(str): Name of the stream to disable (eg. 'left', 'color', 'depth', etc.)
            disableReading (bool, Optional): Also disable reading frames from the file
        """
        if disableReading:
            self.reader.disableStream(stream_name)

        if stream_name not in self.streams:
            print(f"There's no stream '{stream_name}' available!")
            return

        self.streams[stream_name].disabled = True

    def specify_socket(self, stream_name: str, socket: dai.CameraBoardSocket):
        if stream_name not in self.streams:
            print(f"There's no stream '{stream_name}' available!")
            return
        self.streams[stream_name].camera_socket = socket

    def initPipeline(self, pipeline: dai.Pipeline = None):
        """
        Prepares the pipeline for replaying. It creates XLinkIn nodes and sets up StereoDepth node.
        Returns: dai.Pipeline
        """
        if pipeline is None:  # Create pipeline if not passed
            pipeline = dai.Pipeline()

        if self._calibData is not None:
            pipeline.setCalibrationData(self._calibData)

        def createXIn(p: dai.Pipeline, xlink_stream_name: str, size: int):
            xin = p.create(dai.node.XLinkIn)
            xin.setMaxDataSize(size)
            xin.setStreamName(xlink_stream_name)
            return xin

        for name, stream in self.streams.items():
            if stream.disabled: continue

            stream.stream_name = name + '_in'
            stream.node = createXIn(pipeline, stream.stream_name, stream.size_bytes)

        return pipeline

    def initStereoDepth(self, stereo: dai.node.StereoDepth, left_name: str='left', right_name: str='right', align_to: str = ''):
        if left_name not in self.streams or right_name not in self.streams:
            raise Exception("Tried to init StereoDepth, but left/right streams aren't available!")

        left = self.streams[left_name]
        right = self.streams[right_name]

        stereo.setInputResolution(left.shape)

        if not left.camera_socket:
            left.camera_socket = dai.CameraBoardSocket.LEFT
        if not right.camera_socket:
            right.camera_socket = dai.CameraBoardSocket.RIGHT

        if align_to:  # Enable RGB-depth alignment
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setOutputSize(*self.streams[align_to].shape)

        left.node.out.link(stereo.left)
        right.node.out.link(stereo.right)

    def start(self, cb):
        """
        Start sending frames to the OAK device on a new thread
        """
        self.thread = Thread(target=self.run, args=(cb,))
        self.thread.start()

    def run(self, cb):
        delay = 1.0 / self.fps
        while True:
            if not self.sendFrames(cb): break
            time.sleep(delay)
            if self._stop: break
        print('Replay `run` thread stopped')
        self._stop = True

    def sendFrames(self, cb=None) -> bool:
        """
        Reads and sends recorded frames from all enabled streams to the OAK camera.

        Returns:
            bool: True if successful, otherwise False.
        """
        if not self._pause:  # If replaying is paused, don't read new frames
            if not self._readFrames():
                return False  # End of the recording

        self._now = monotonic()
        for stream_name, stream in self.streams.items():
            stream.imgFrame =self._createImgFrame(stream)
            # Save the imgFrame
            if cb:  # callback
                cb(stream_name, stream.imgFrame)

            # Don't send these frames to the OAK camera
            if stream.disabled: continue

            # Send an imgFrame to the OAK camera
            stream.queue.send(stream.imgFrame)

        self._seqNum += 1
        return True

    def createQueues(self, device: dai.Device):
        """
        Creates input queue for each enabled stream
        
        Args:
            device (dai.Device): Device to which we will stream frames
        """

        for name, stream in self.streams.items():
            if stream.stream_name:
                stream.queue = device.getInputQueue(stream.stream_name)

    def getStreams(self) -> List[str]:
        return [name for name, stream in self.streams.items()]

    def _resize_frame(self, frame: np.ndarray, size: Tuple[int,int], mode: ResizeMode) -> np.ndarray:
        if mode == ResizeMode.STRETCH:
            # No need to keep aspect ratio, image will be squished
            return cv2.resize(frame, size)
        elif mode == ResizeMode.CROP:
            cropped = cropToAspectRatio(frame, size)
            return cv2.resize(cropped, size)
        elif mode == ResizeMode.FULL_CROP:
            w = frame.shape[1]
            start_w = int((w - size[0]) / 2)
            h = frame.shape[0]
            start_h = int((h - size[1]) / 2)
            return frame[start_h:h-start_h, start_w:w-start_w]

    def _createNewFrame(self, cvFrame) -> dai.ImgFrame:
        imgFrame = dai.ImgFrame()
        imgFrame.setData(cvFrame)
        imgFrame.setTimestamp(self._now)
        imgFrame.setSequenceNum(self._seqNum)
        shape = cvFrame.shape[::-1]
        imgFrame.setWidth(shape[0])
        imgFrame.setHeight(shape[1])
        return imgFrame

    def _createImgFrame(self, stream: ReplayStream) -> dai.ImgFrame:
        cvFrame: np.ndarray = stream.frame
        if stream.resize:
            cvFrame = self._resize_frame(cvFrame, stream.resize, stream.resize_mode)


        if cvFrame.shape[-1] == 3: # 3 channels = RGB
            # Resize/crop color frame as specified by the user
            # cv2 reads frames in interleaved format, and most networks expect planar by default
            cvFrame = toPlanar(cvFrame)
            imgFrame = self._createNewFrame(cvFrame)
            imgFrame.setType(dai.RawImgFrame.Type.BGR888p)
            imgFrame.setInstanceNum(int(stream.get_socket()))
        elif cvFrame.dtype == np.uint8:
            imgFrame = self._createNewFrame(cvFrame)
            imgFrame.setType(dai.RawImgFrame.Type.RAW8)
            imgFrame.setInstanceNum(int(stream.get_socket()))
        elif cvFrame.dtype == np.uint16:
            imgFrame = self._createNewFrame(cvFrame)
            imgFrame.setType(dai.RawImgFrame.Type.RAW16)
        else:
            raise Exception('Unknown frame types')

        return imgFrame

    def _readFrames(self) -> bool:
        """
        Reads frames from all Readers.
        
        Returns:
            bool: True if successful, otherwise False.
        """
        frames = self.reader.read()
        if not frames:
            return False  # No more frames!

        for name, frame in frames.items():
            self.streams[name].frame = frame

        # Compress 3-plane frame to a single plane
        # for name, frame in self.frames.items():
        #     if name in ["left", "right", "disparity"] and len(frame.shape) == 3:
        #         self.frames[name] = frame[:, :, 0]  # All 3 planes are the same
        return True


    def getShape(self, name: str) -> Tuple[int, int]:
        """
        Get shape of a stream
        """
        return self.streams[name].shape

    def close(self):
        """
        Closes all video readers.
        """
        self._stop = True
        if self.thread:
            self.thread.join()
        self.reader.close()
