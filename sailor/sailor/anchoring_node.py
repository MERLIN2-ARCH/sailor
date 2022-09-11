
import rclpy
from rclpy.node import Node

from sailor_interfaces.msg import Percept
from sailor_interfaces.msg import PerceptArray


class AnchoringNode(Node):

    def __init__(self) -> None:
        super().__init__("anchoring_node")


def main():
    rclpy.init()
    rclpy.spin(AnchoringNode())
    rclpy.shutdown()
