import time
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters

class GetParamClientAsync(Node):
    def __init__(self, node_name: str):
        super().__init__('get_param_client_async')
        self.topic_name = f'{node_name}/get_parameters'
        self.cli = self.create_client(GetParameters, self.topic_name)

        start_time = time.time()
        self.service_available = False
        while time.time() - start_time < 5.0:
            if self.cli.wait_for_service(timeout_sec=1.0):
                self.service_available = True
                break
        
        if not self.service_available:
            self.get_logger().warn(f"Service '{self.topic_name}' not available after 5 seconds")

        self.req = GetParameters.Request()

    def send_request(self, params_name_list):
        if not self.service_available:
            return None
        
        self.req.names = params_name_list
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    