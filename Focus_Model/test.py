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
    loops = 0

    try:
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()

        # plt.ion()
        # fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        # axes = axes.flatten()

        while True:
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'sleeping main thread')
            time.sleep(2)

            # Fetch latest 1250 samples
            data = board.get_current_board_data(num_samples=1250)
            if data.shape[1] > 50:
                data = data[:, 50:]  # remove first 50 samples if needed

            # Clear previous plots
            # axes[0].cla()
            # axes[1].cla()

            # Plot channel 7 and 8 (indexes 6 and 7)
            # axes[0].plot(data[6])
            # axes[1].plot(data[7])
            # axes[0].set_title(f"Loop {loops} - Channel 7")
            # axes[1].set_title(f"Loop {loops} - Channel 8")

            # Set Y-axis limits
            # axes[0].set_ylim(0, 20000)
            # axes[1].set_ylim(0, 20000)

            # Compute and print Higuchi FD for each channel
            fd7 = higuchi_fd(data[6])  # channel 7
            fd8 = higuchi_fd(data[7])  # channel 8
            print(f"Loop {loops} - Higuchi FD: Channel 7 = {fd7:.4f}, Channel 8 = {fd8:.4f}")

            # fig.canvas.draw()
            # plt.pause(0.001)

            loops += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'board' in locals() and board.is_prepared():
            board.stop_stream()
            board.release_session()

if __name__ == "__main__":
    main()
