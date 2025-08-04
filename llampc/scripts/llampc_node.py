import numba

import rclpy
from rclpy.node import Node

from llampc.nmpc_gen import setup_mpc_from_json
from llampc.params import F110


class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.solver = setup_mpc_from_json()

def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()