import threading
from enum import IntEnum
from time import sleep

from pythonosc import udp_client
from pythonosc.osc_server import BlockingOSCUDPServer

from ue5osc.osc_dispatcher import OSCMessageReceiver


class TexturedSurface(IntEnum):
    FLOOR = 0
    WALL = 1
    CEILING = 2


NUM_TEXTURES = 42


class Light(IntEnum):
    SKY = 0


LIGHT_INTENSITIES = 1  # placeholder


class Communicator:
    """This handles interaction between the UE5 environment and a Python script."""

    def __init__(self, ip: str, ue_port: int, py_port: int):
        """Initialize OSC client and server."""
        self.ip = ip
        self.ue_port = ue_port
        self.py_port = py_port

        self.message_handler = OSCMessageReceiver()
        self.server = BlockingOSCUDPServer(
            (self.ip, self.py_port), self.message_handler.dispatcher
        )
        self.client = udp_client.SimpleUDPClient(self.ip, self.ue_port)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close_osc()

    def close_osc(self) -> None:
        """Closes the OSC server and joins the server."""
        self.server.shutdown()
        self.server_thread.join()

    def send_and_await(self, osc_address: str):
        """Sends command and waits for a return value before continuing."""
        dummy_data = 0.0
        self.client.send_message(osc_address, dummy_data)
        return self.message_handler.wait_for_response()

    def get_project_name(self) -> str:
        """Returns the name of the current connected project."""
        return self.send_and_await("/get/project")

    def get_raycast_distance(self) -> float:
        """Returns the length of the raycast. We use this in order to decide whether a
        movement forward is valid. If the raycast returns a value other than 0 it means
        the robot would hit the wall."""
        return self.send_and_await("/get/raycast")

    def set_raycast_length(self, length: float, delay: float = 0.0) -> None:
        """Sets the length of the raycast."""
        self.client.send_message("/set/raycast", float(length))
        sleep(delay)

    def get_location(self) -> tuple[float, float, float]:
        """Returns x, y, z location of the player in the Unreal Environment."""
        return self.send_and_await("/get/location")

    def set_location(self, x: float, y: float, z: float, delay: float = 0.0) -> None:
        """Sets X, Y, and Z values of an Unreal Camera."""
        self.client.send_message("/set/location", [x, y, z])
        sleep(delay)

    def set_location_xy(self, x: float, y: float, delay: float = 0.0) -> None:
        """Sets X and Y values of an Unreal Camera. Z is taken from Unreal."""
        # TODO: add this functionality directly to UE
        _, _, unreal_z = self.get_location()
        self.set_location(x, y, unreal_z, delay)

    def get_rotation(self) -> tuple[float, float, float]:
        """Returns roll, pitch, and yaw in degrees."""
        return self.send_and_await("/get/rotation")

    def set_rotation(self, roll: float, pitch: float, yaw: float) -> None:
        """Set rotation in degrees."""
        self.client.send_message("/set/rotation", [roll, pitch, yaw])

    def set_yaw(self, yaw: float) -> None:
        """Set the robot's yaw in degrees."""
        self.client.send_message("/set/yaw", float(yaw))

    def move_forward(self, amount: float) -> None:
        """Move robot forward."""
        self.client.send_message("/move/forward", float(amount))

    def move_backward(self, amount: float) -> None:
        """Move robot backwards."""
        self.client.send_message("/move/forward", float(-amount))

    def rotate_left(self, degree: float) -> None:
        """Rotate robot left."""
        self.client.send_message("/rotate/left", float(degree))

    def rotate_right(self, degree: float) -> None:
        """Rotate robot right."""
        self.client.send_message("/rotate/right", float(degree))

    def set_resolution(self, resolution: str, delay: float = 0.0) -> None:
        """Allows you to set resolution of images in the form of ResXxResY."""
        self.client.send_message("/set/resolution", resolution)
        sleep(delay)

    def save_image(self, filename: str, delay: float = 0.0) -> None:
        """Takes screenshot with the default name."""
        # Unreal Engine Needs a forward / to separate folder from the filenames
        filename = "/".join(filename.rsplit("\\", 1))
        self.client.send_message("/save/image", str(filename))
        sleep(delay)

    def console(self, message: str) -> None:
        """Sends Unreal Engine console commands (only works in development mode)."""
        self.client.send_message("/console", message)

    def toggle_camera_view(self) -> None:
        """Toggles the camera between 1st and 3rd person views."""
        dummy = 0.0
        self.client.send_message("/toggle/view", dummy)

    def set_quality(self, graphics_level: int, delay: float = 0.0) -> None:
        """Set the graphics quality level from 0 (low) to 4 (high)."""
        self.client.send_message("/set/quality", graphics_level)
        sleep(delay)

    def set_texture(self, object: TexturedSurface, material: int) -> None:
        """Set the texture of walls/floors/ceilings to a different material."""
        self.client.send_message("/set/texture", [object, material])

    def set_lighting(self, light_name: Light, intensity: int) -> None:
        """Set the lighting of the environment."""
        self.client.send_message("/set/light", [light_name, intensity])

    def reset(self, delay: float = 0.0) -> None:
        """Reset agent to the start location using a UE Blueprint command."""
        # The python OSC library send_message method always requires a value
        self.client.send_message("/reset", 0.0)
        sleep(delay)
        # TODO: add reasonable defaults
        # self.set_quality(2)
        # textures
        # self.set_texture(textureObject, 0.0)
        # aspect ratio
        # self.set_resolution("244x244")
