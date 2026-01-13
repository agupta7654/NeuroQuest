import time
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from antropy import higuchi_fd
import numpy as np

def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board_id = BoardIds.CYTON_DAISY_BOARD.value
    params.serial_port = "COM3"
    try:
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
        while True:
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'sleeping main thread')
            time.sleep(2)
            data = board.get_current_board_data(num_samples=1250) # Fetch latest 1250 samples
            print(BoardShim.get_eeg_channels(board_id))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'board' in locals() and board.is_prepared():
            board.stop_stream()
            board.release_session()

if __name__ == "__main__":
    main()
