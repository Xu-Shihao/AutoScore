# logger_config.py

import logging

def setup_logger(name='autoscore', log_file='./log/app.log', level=logging.DEBUG, add_console_handler=False):
    """设置logger

    Args:
        name (str): logger的名称
        log_file (str): 日志文件的名称
        level (int): 日志级别

    Returns:
        logging.Logger: 配置好的logger对象
    """
    # 创建一个logger对象
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if add_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# 如果这个文件被直接运行,执行一个简单的测试
if __name__ == "__main__":
    logger = setup_logger()
    logger.debug('这是一条调试日志')
    logger.info('这是一条信息日志')
    logger.warning('这是一条警告日志')
    logger.error('这是一条错误日志')
    logger.critical('这是一条严重错误日志')