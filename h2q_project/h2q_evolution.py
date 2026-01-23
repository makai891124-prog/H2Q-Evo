import time
import logging

class H2QEvolutionSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # 创建一个文件处理器，将日志写入文件
        fh = logging.FileHandler('h2q_evolution.log')
        fh.setLevel(logging.INFO)
        # 创建一个格式器，定义日志消息的格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # 将文件处理器添加到logger对象
        self.logger.addHandler(fh)

    def step(self, input_data):
        self.logger.info(f"Starting step with input: {input_data}")
        start_time = time.time()

        try:
            # 模拟一些处理
            intermediate_result = self._process_input(input_data)
            output_data = self._generate_output(intermediate_result)

            end_time = time.time()
            execution_time = end_time - start_time
            self.logger.info(f"Step completed in {execution_time:.4f} seconds with output: {output_data}")

            return output_data
        except Exception as e:
            self.logger.error(f"Error during step execution: {e}", exc_info=True)
            raise

    def _process_input(self, input_data):
        self.logger.debug(f"Processing input data: {input_data}")
        # 这里添加实际的处理逻辑
        processed_data = input_data + "_processed"
        self.logger.debug(f"Processed data: {processed_data}")
        return processed_data

    def _generate_output(self, intermediate_result):
        self.logger.debug(f"Generating output from intermediate result: {intermediate_result}")
        # 这里添加实际的输出生成逻辑
        output_data = intermediate_result + "_output"
        self.logger.debug(f"Generated output: {output_data}")
        return output_data

if __name__ == '__main__':
    # 配置logging
    logging.basicConfig(level=logging.INFO)

    evolution_system = H2QEvolutionSystem()
    input_data = "initial_input"
    try:
        output = evolution_system.step(input_data)
        print(f"Final Output: {output}")
    except Exception as e:
        print(f"An error occurred: {e}")
